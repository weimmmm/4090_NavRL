"""Direct t+3 (0.48 s) action-conditioned LiDAR diffusion world model.

This experiment keeps the trained two-channel circular VAE and the original
LaGen-style 2D UNet topology.  It changes only the temporal target and the
conditioning sequence: three consecutive ten-action chunks condition a direct
prediction of the third future LiDAR frame.  It never uses future ego state.

Examples:

  python -m lidar_wam.runner.world_direct_horizon inspect --data-root DATA
  python -m lidar_wam.runner.world_direct_horizon smoke --data-root DATA \
      --init-checkpoint outputs/world_circular_causal_8h/best.pt
  python -m lidar_wam.runner.world_direct_horizon train --data-root DATA \
      --init-checkpoint outputs/world_circular_causal_8h/best.pt --overfit
  python -m lidar_wam.runner.world_direct_horizon evaluate --data-root DATA \
      --split val
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from scipy.spatial import cKDTree
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lidar_wam.data_v2 import V2WindowDataset, file_sha256
from lidar_wam.models.world import DirectHorizonWorldModel
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_geometry import (
    load_rays,
    predict_transform,
    warp_frame,
)
from lidar_wam.runner.world_decoded_aux import (
    decoded_losses,
    predicted_x0,
    select_aux_indices,
)


HORIZON = 3
SIM_DT_SECONDS = 0.016
FRAME_DT_SECONDS = 0.16
TARGET_DT_SECONDS = HORIZON * FRAME_DT_SECONDS
DEFAULT_RUN_NAME = "world_direct_t3"
VOXEL_CHAMFER_SIZE_M = 0.20
LAGEN_CHAMFER_POINT_COUNT = 1500


def _decode_token(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def _read_rows(h5, key, indices, dtype=np.float32):
    """Read arbitrary HDF5 rows while satisfying h5py's sorted-index rule."""
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    value = np.asarray(h5[key][indices[order]], dtype=dtype)
    return value[inverse]


def _candidate_chains(split, data_root, latent_root):
    cache = np.load(latent_root / f"{split}.npz")
    source = np.asarray(cache["source_index"], dtype=np.int64)
    source_position = {int(index): position for position, index in enumerate(source)}
    if len(source_position) != len(source):
        raise ValueError(f"Duplicate source_index entries in {split} latent cache")

    with h5py.File(data_root / f"navrl_static_{split}.h5", "r") as h5:
        tokens = h5["token"][:]
        previous_tokens = h5["prev_token"][:]
        scenes = h5["scene_token"][:]
        frames = h5["frame_idx"][:]
        seeds = h5["terrain_seed"][:]
        action_mask = h5["action_mask"][:]
        step_delta = h5["step_delta"][:]

    successor = {}
    for row, previous_token in enumerate(previous_tokens):
        # Scene-start rows can share an empty prev_token, but no valid token may
        # have two successors.
        if not _decode_token(previous_token):
            continue
        if previous_token in successor:
            raise ValueError(
                "prev_token is not unique; transition chains are ambiguous at "
                f"rows {successor[previous_token]} and {row}")
        successor[previous_token] = row

    valid_source = set(source.tolist())
    finite_cache = (
        np.isfinite(np.asarray(cache["actions"])).all(axis=(1, 2))
        & np.isfinite(np.asarray(cache["state"])).all(axis=1)
        & np.isfinite(np.asarray(cache["previous"])).reshape(len(source), -1).all(axis=1)
        & np.isfinite(np.asarray(cache["target"])).reshape(len(source), -1).all(axis=1)
    )
    chains = []
    chain_seeds = []
    rejection = defaultdict(int)
    for start in source:
        start = int(start)
        chain = [start]
        for _ in range(HORIZON - 1):
            following = successor.get(tokens[chain[-1]])
            if following is None or int(following) not in valid_source:
                break
            chain.append(int(following))
        if len(chain) != HORIZON:
            rejection["short_chain"] += 1
            continue
        if any(scenes[row] != scenes[start] for row in chain):
            rejection["scene_boundary"] += 1
            continue
        if any(int(frames[row]) != int(frames[start]) + offset
               for offset, row in enumerate(chain)):
            rejection["nonconsecutive_frame"] += 1
            continue
        # All three rows contribute an action chunk, including the final row.
        if any(int(step_delta[row]) != 10 for row in chain):
            rejection["wrong_step_delta"] += 1
            continue
        if any(not bool(np.all(action_mask[row])) for row in chain):
            rejection["invalid_action_mask"] += 1
            continue
        positions = np.asarray([source_position[row] for row in chain], dtype=np.int64)
        if not bool(finite_cache[positions].all()):
            rejection["nonfinite_cache_value"] += 1
            continue
        chains.append(chain)
        chain_seeds.append(int(seeds[start]))

    if not chains:
        raise RuntimeError(f"No valid direct-t+{HORIZON} chains in {split}")
    return (np.asarray(chains, dtype=np.int64),
            np.asarray(chain_seeds, dtype=np.int16),
            source_position, dict(rejection))


class DirectHorizonDataset(Dataset):
    """Current latent, three action chunks, and the direct t+3 target."""

    def __init__(self, split, data_root, latent_root, source_chains=None,
                 limit_per_seed=None, random_seed=42, overfit=False,
                 include_previous_image=False, include_drone_state=False):
        self.split = split
        self.data_root = Path(data_root)
        self.latent_root = Path(latent_root)
        cache = np.load(self.latent_root / f"{split}.npz")
        metadata = json.loads((self.latent_root / "metadata.json").read_text())
        self.scale = float(metadata["scaling_factor"])
        all_chains, all_seeds, source_position, self.rejection = _candidate_chains(
            split, self.data_root, self.latent_root)

        if source_chains is not None:
            position = {tuple(int(v) for v in chain): i
                        for i, chain in enumerate(all_chains)}
            requested = [tuple(int(v) for v in chain) for chain in source_chains]
            missing = [chain for chain in requested if chain not in position]
            if missing:
                raise ValueError(f"Manifest contains invalid chains: {missing[:3]}")
            keep = np.asarray([position[chain] for chain in requested], dtype=np.int64)
        elif overfit:
            keep = np.arange(min(128, len(all_chains)), dtype=np.int64)
        elif limit_per_seed is not None:
            rng = np.random.default_rng(random_seed)
            pieces = []
            for seed in sorted(set(all_seeds.tolist())):
                candidates = np.flatnonzero(all_seeds == seed)
                if len(candidates) < limit_per_seed:
                    raise RuntimeError(
                        f"{split} seed {seed} only has {len(candidates)} chains; "
                        f"requested {limit_per_seed}")
                chosen = rng.choice(candidates, limit_per_seed, replace=False)
                pieces.append(np.sort(chosen))
            keep = np.concatenate(pieces)
        else:
            keep = np.arange(len(all_chains), dtype=np.int64)

        self.source_indices = all_chains[keep]
        self.seeds = all_seeds[keep]
        chain_positions = np.asarray([
            [source_position[int(row)] for row in chain]
            for chain in self.source_indices
        ], dtype=np.int64)
        self.previous = torch.from_numpy(
            np.asarray(cache["previous"][chain_positions[:, 0]]).copy()
            * self.scale).float()
        self.target = torch.from_numpy(
            np.asarray(cache["target"][chain_positions[:, -1]]).copy()
            * self.scale).float()
        self.actions = torch.from_numpy(
            np.asarray(cache["actions"][chain_positions]).copy()).float()
        self.state = torch.from_numpy(stage1.causal_state(
            np.asarray(cache["state"][chain_positions[:, 0]]).copy())).float()

        h5_path = self.data_root / f"navrl_static_{split}.h5"
        with h5py.File(h5_path, "r") as h5:
            target_rows = self.source_indices[:, -1]
            self.target_image = torch.from_numpy(
                _read_rows(h5, "range_values", target_rows)).float()
            self.previous_image = None
            self.previous_drone_state = None
            if include_previous_image:
                self.previous_image = torch.from_numpy(
                    _read_rows(h5, "prev_range_values", self.source_indices[:, 0])).float()
            if include_drone_state:
                self.previous_drone_state = _read_rows(
                    h5, "prev_drone_state", self.source_indices[:, 0])

        tensors = (self.previous, self.target, self.actions,
                   self.state, self.target_image)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError(f"Non-finite tensor in selected {split} direct-t+3 data")
        if self.actions.shape[1:] != (HORIZON, 10, 3):
            raise ValueError(f"Unexpected action shape {tuple(self.actions.shape)}")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return (self.previous[index], self.target[index], self.actions[index],
                self.state[index], self.target_image[index],
                torch.from_numpy(self.source_indices[index]))


def load_one_step_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if set(incompatible.missing_keys) != {"condition.segment"}:
        raise ValueError(f"Unexpected missing checkpoint keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    return checkpoint


def manifest_dataset(split, args, run_dir, include_baseline_inputs=False):
    manifest_path = run_dir / f"fixed_{split}_manifest.json"
    source_chains = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("version") != 1 or manifest.get("horizon") != HORIZON
                or manifest.get("random_seed") != args.seed):
            raise ValueError(f"Incompatible manifest: {manifest_path}")
        source_chains = [row["source_indices"] for row in manifest["rows"]]
    dataset = DirectHorizonDataset(
        split, args.data_root, args.latent_root,
        source_chains=source_chains,
        limit_per_seed=None if source_chains is not None else args.samples_per_seed,
        random_seed=args.seed,
        include_previous_image=include_baseline_inputs,
        include_drone_state=include_baseline_inputs,
    )
    if source_chains is None:
        rows = [{"source_indices": [int(v) for v in chain], "seed": int(seed)}
                for chain, seed in zip(dataset.source_indices, dataset.seeds)]
        stage1.save_json(manifest_path, {
            "version": 1, "split": split, "horizon": HORIZON,
            "target_seconds": TARGET_DT_SECONDS, "random_seed": args.seed,
            "samples_per_seed": args.samples_per_seed, "rows": rows,
        })
    return dataset


def generate(model, scheduler, previous, actions, state, seed, num_steps):
    return stage1.generate(model, scheduler, previous, actions, state, seed,
                           num_steps=num_steps)


def squared_chamfer_points(prediction, target):
    if not len(prediction) and not len(target):
        return 0.0
    if not len(prediction) or not len(target):
        # LiDAR points are clipped to a 10 m range.  Assign the maximum
        # one-sided squared distance in both empty/non-empty directions; the
        # outer 1/2 in the Chamfer definition therefore leaves 100 m^2.
        return 100.0
    target_to_prediction = np.square(
        cKDTree(prediction).query(target)[0]).mean()
    prediction_to_target = np.square(
        cKDTree(target).query(prediction)[0]).mean()
    return float(0.5 * (target_to_prediction + prediction_to_target))


def voxel_anchor_points(points, voxel_size=VOXEL_CHAMFER_SIZE_M):
    """Return one center per occupied voxel plus a shared sensor-origin anchor.

    The anchor makes the set non-empty without padding it with thousands of
    duplicate zeros as the original LaGen evaluator does.  Both prediction and
    target receive exactly one anchor, so two genuinely empty scans have zero
    distance while one-sided empty scans retain a finite geometric penalty.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points):
        voxel_indices = np.unique(
            np.floor(points / float(voxel_size)).astype(np.int64), axis=0)
        centers = (voxel_indices.astype(np.float64) + 0.5) * float(voxel_size)
    else:
        centers = np.empty((0, 3), dtype=np.float64)
    return np.concatenate((centers, np.zeros((1, 3), dtype=np.float64)), axis=0)


def voxel_anchor_squared_chamfer_points(prediction, target,
                                        voxel_size=VOXEL_CHAMFER_SIZE_M):
    """Bidirectional mean squared Chamfer after sparse voxelization (m^2)."""
    prediction = voxel_anchor_points(prediction, voxel_size)
    target = voxel_anchor_points(target, voxel_size)
    target_to_prediction = np.square(
        cKDTree(prediction).query(target)[0]).mean()
    prediction_to_target = np.square(
        cKDTree(target).query(prediction)[0]).mean()
    return float(0.5 * (target_to_prediction + prediction_to_target))


def lagen_zero_pad_points(points, count=LAGEN_CHAMFER_POINT_COUNT):
    """Match LaGen's fixed-cardinality prefix-copy and zero-padding logic."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    count = int(count)
    if count <= 0:
        raise ValueError("fixed point count must be positive")
    padded = np.zeros((count, 3), dtype=np.float64)
    copied = min(len(points), count)
    if copied:
        padded[:copied] = points[:copied]
    return padded


def lagen_zero_pad_squared_chamfer_points(
        prediction, target, count=LAGEN_CHAMFER_POINT_COUNT):
    """LaGen-style bidirectional squared Chamfer on zero-padded clouds."""
    prediction = lagen_zero_pad_points(prediction, count)
    target = lagen_zero_pad_points(target, count)
    target_to_prediction = np.square(
        cKDTree(prediction).query(target)[0]).mean()
    prediction_to_target = np.square(
        cKDTree(target).query(prediction)[0]).mean()
    return float(0.5 * (target_to_prediction + prediction_to_target))


def linear_symmetric_nn_points(prediction, target):
    if not len(prediction) and not len(target):
        return 0.0
    if not len(prediction) or not len(target):
        return 20.0
    return float((cKDTree(prediction).query(target)[0].mean()
                  + cKDTree(target).query(prediction)[0].mean()) / 2.0)


def frame_metrics(prediction, target, mask_threshold=1.5):
    pred_hit = prediction[1, :, :18] > mask_threshold
    gt_hit = target[1, :, :18] > 0.0
    pred_range = np.clip((prediction[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
    gt_range = np.clip((target[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
    pred_points = stage1.to_points(prediction, mask_threshold)
    target_points = stage1.to_points(target, 0.0)
    tp = int((pred_hit & gt_hit).sum())
    fp = int((pred_hit & ~gt_hit).sum())
    fn = int((~pred_hit & gt_hit).sum())
    gt_empty = not bool(gt_hit.any())
    pred_empty = not bool(pred_hit.any())
    return {
        "cd_paper_m2": squared_chamfer_points(pred_points, target_points),
        "voxel_anchor_cd_m2": voxel_anchor_squared_chamfer_points(
            pred_points, target_points),
        "lagen_zero_pad_1500_cd_m2": lagen_zero_pad_squared_chamfer_points(
            pred_points, target_points),
        "symmetric_nn_distance_m": linear_symmetric_nn_points(
            pred_points, target_points),
        "valid_range_abs_error_sum_m": float(
            np.abs(pred_range - gt_range)[gt_hit].sum()),
        "valid_range_count": int(gt_hit.sum()),
        "tp": tp, "fp": fp, "fn": fn,
        "gt_empty": gt_empty, "pred_empty": pred_empty,
        "false_empty_frame": bool((not gt_empty) and pred_empty),
        "false_hit_frame": bool(gt_empty and (not pred_empty)),
        "gt_hit_count": int(gt_hit.sum()),
        "pred_hit_count": int(pred_hit.sum()),
    }


def summarize_rows(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    valid_error = sum(row["valid_range_abs_error_sum_m"] for row in rows)
    valid_count = sum(row["valid_range_count"] for row in rows)
    return {
        "samples": len(rows),
        "cd_paper_m2": float(np.mean([row["cd_paper_m2"] for row in rows])),
        "median_cd_paper_m2": float(np.median([row["cd_paper_m2"] for row in rows])),
        "voxel_anchor_cd_m2": float(np.mean([
            row["voxel_anchor_cd_m2"] for row in rows])),
        "lagen_zero_pad_1500_cd_m2": float(np.mean([
            row["lagen_zero_pad_1500_cd_m2"] for row in rows])),
        "symmetric_nn_distance_m": float(np.mean([
            row["symmetric_nn_distance_m"] for row in rows])),
        "valid_range_mae_m": valid_error / max(valid_count, 1),
        "mask_precision": tp / max(tp + fp, 1),
        "mask_recall": tp / max(tp + fn, 1),
        "mask_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_empty_frames": sum(row["false_empty_frame"] for row in rows),
        "false_hit_frames": sum(row["false_hit_frame"] for row in rows),
        "gt_hit_count": sum(row["gt_hit_count"] for row in rows),
        "pred_hit_count": sum(row["pred_hit_count"] for row in rows),
        "latent_mse": (float(np.mean([row["latent_mse"] for row in rows]))
                       if "latent_mse" in rows[0] else None),
        "copy_latent_mse": (
            float(np.mean([row["copy_latent_mse"] for row in rows]))
            if "copy_latent_mse" in rows[0] else None),
    }


def evaluation_key(summary):
    """Zero false-empty checkpoints win; geometry selects among them."""
    false_empty = int(summary["false_empty_frames"])
    return (int(false_empty > 0), false_empty,
            float(summary["cd_paper_m2"]),
            int(summary["false_hit_frames"]),
            -float(summary["mask_f1"]))


def copy_baseline_summary(dataset):
    if dataset.previous_image is None:
        raise ValueError("Copy baseline requires include_previous_image=True")
    latent_mse = ((dataset.previous - dataset.target).square()
                  .flatten(1).mean(1).numpy())
    rows = []
    for index in range(len(dataset)):
        row = frame_metrics(dataset.previous_image[index].numpy(),
                            dataset.target_image[index].numpy(), 0.0)
        row["copy_latent_mse"] = float(latent_mse[index])
        rows.append(row)
    return summarize_rows(rows)


@torch.no_grad()
def evaluate_direct(model, vae, dataset, scale, ddim_steps, batch_size,
                    seed, mask_threshold, shuffle_actions=False):
    model.eval()
    scheduler = stage1.DDIMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon", clip_sample=False)
    rows = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    offset = 0
    for batch in loader:
        previous, target, actions, state, target_image, source = batch
        previous = previous.to(stage1.DEVICE)
        target = target.to(stage1.DEVICE)
        actions = actions.to(stage1.DEVICE)
        state = state.to(stage1.DEVICE)
        if shuffle_actions:
            if len(actions) < 2:
                raise ValueError("Action shuffle evaluation requires batch size >= 2")
            actions = torch.roll(actions, shifts=1, dims=0)
        prediction_latent = generate(
            model, scheduler, previous, actions, state,
            seed + offset, ddim_steps)
        prediction = vae.decode(prediction_latent / scale).sample.cpu().numpy()
        target_np = target_image.numpy()
        latent_mse = ((prediction_latent - target).square().flatten(1).mean(1)
                      .cpu().numpy())
        copy_latent_mse = ((previous - target).square().flatten(1).mean(1)
                           .cpu().numpy())
        for index in range(len(prediction)):
            metrics = frame_metrics(prediction[index], target_np[index],
                                    mask_threshold)
            metrics.update({
                "source_indices": [int(v) for v in source[index].numpy()],
                "seed": int(dataset.seeds[offset + index]),
                "latent_mse": float(latent_mse[index]),
                "copy_latent_mse": float(copy_latent_mse[index]),
            })
            rows.append(metrics)
        offset += len(prediction)
    summary = summarize_rows(rows)
    by_seed = {}
    for seed_value in sorted({row["seed"] for row in rows}):
        by_seed[str(seed_value)] = summarize_rows(
            [row for row in rows if row["seed"] == seed_value])
    return {"summary": summary, "by_seed": by_seed, "rows": rows}


def _load_initial_or_resume(args, model, optimizer, run_dir):
    if args.resume:
        checkpoint = stage1.load_model(run_dir / "latest.pt", model)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = (args.segment_lr if group.get("name") == "segment"
                           else args.lr)
        return checkpoint, int(checkpoint["step"]) + 1
    checkpoint = load_one_step_checkpoint(args.init_checkpoint, model)
    return checkpoint, 1


def _make_optimizer(model, lr, segment_lr):
    segment = model.condition.segment
    base = [parameter for parameter in model.parameters() if parameter is not segment]
    return torch.optim.AdamW([
        {"params": base, "lr": lr, "name": "base"},
        {"params": [segment], "lr": segment_lr, "name": "segment"},
    ])


def train(args):
    run_dir = args.out / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_data = DirectHorizonDataset(
        "train", args.data_root, args.latent_root, overfit=args.overfit)
    if args.overfit:
        val_data = train_data
    else:
        val_data = manifest_dataset("val", args, run_dir)

    latent_meta = json.loads((args.latent_root / "metadata.json").read_text())
    vae_identity = stage1.circular_vae_identity()
    if any(latent_meta.get(key) != value for key, value in vae_identity.items()):
        raise ValueError("Circular VAE weights changed; rebuild the latent cache")

    config = {
        "stage": "direct_t3_world", "horizon": HORIZON,
        "target_seconds": TARGET_DT_SECONDS, "steps": args.steps,
        "batch_size": args.batch_size, "lr": args.lr,
        "segment_lr": args.segment_lr, "seed": args.seed,
        "train_samples": len(train_data), "validation_samples": len(val_data),
        "initial_checkpoint": str(args.init_checkpoint),
        "vae_sha256": latent_meta["vae_sha256"],
        "scaling_factor": float(latent_meta["scaling_factor"]),
        "state": stage1.CAUSAL_STATE_DEFINITION,
        "actions": "three ordered normalized_action_sequence chunks [3,10,3]",
        "architecture": "stage1.WorldModel UNet with 30 action tokens and one state token",
        "loss": {
            "epsilon_mse": 1.0, "x0_l1": args.latent_weight,
            "mask_bce": args.mask_weight,
            "empty_mask_bce": args.empty_mask_weight,
            "range_l1": args.range_weight,
            "presence": args.presence_weight,
            "empty_suppression": args.empty_suppress_weight,
        },
        "ddim_steps": args.ddim_steps,
        "selection": "zero false-empty first, then all-sample cd_paper_m2",
    }
    stage1.save_json(run_dir / "config.json", config)

    model = DirectHorizonWorldModel().to(stage1.DEVICE).float()
    optimizer = _make_optimizer(model, args.lr, args.segment_lr)
    initial, first_step = _load_initial_or_resume(args, model, optimizer, run_dir)
    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    scale = float(latent_meta["scaling_factor"])
    noise_scheduler = stage1.DDPMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon")
    loader = stage1.infinite(DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True))

    history_path = run_dir / "history.json"
    history = (json.loads(history_path.read_text())
               if args.resume and history_path.exists() else [])
    best_path = run_dir / "best_metrics.json"
    best_key = (math.inf,) * 5
    if args.resume and best_path.exists():
        saved = json.loads(best_path.read_text()).get("selection_key")
        if isinstance(saved, list) and len(saved) == 5:
            best_key = tuple(saved)

    if first_step == 1 and not args.skip_initial_eval:
        initial_eval = evaluate_direct(
            model, vae, val_data, scale, args.ddim_steps,
            args.eval_batch_size, args.seed, args.mask_threshold)
        best_key = evaluation_key(initial_eval["summary"])
        stage1.save_json(run_dir / "initial_validation.json", initial_eval)
        initial_summary = {
            "step": 0, "external_checkpoint": str(args.init_checkpoint),
            "selection_key": list(best_key), **initial_eval["summary"],
        }
        stage1.save_json(best_path, initial_summary)
        # Ensure evaluation always has a self-contained t+3 checkpoint even if
        # fine-tuning never improves on the warm start.
        stage1.save_model(run_dir / "best.pt", model, optimizer, 0,
                          initial_summary)
        print(json.dumps({"initial_validation": initial_eval["summary"]}), flush=True)

    deadline = time.monotonic() + args.max_hours * 3600 if args.max_hours else None
    for step in range(first_step, args.steps + 1):
        model.train()
        previous, target, actions, state, target_image, _ = [
            value.to(stage1.DEVICE, non_blocking=True) for value in next(loader)]
        noise = torch.randn_like(target)
        timesteps = torch.randint(0, 1000, (len(target),),
                                  device=stage1.DEVICE, dtype=torch.long)
        noisy = noise_scheduler.add_noise(target, noise, timesteps)
        optimizer.zero_grad(set_to_none=True)
        epsilon = model(noisy, previous, actions, state, timesteps)
        epsilon_loss = F.mse_loss(epsilon, noise)

        eligible = select_aux_indices(
            timesteps, target_image, args.aux_t_max,
            args.aux_batch_max, args.aux_empty_max)
        zero = epsilon_loss * 0.0
        latent_loss = nonempty_mask_bce = empty_mask_bce = zero
        range_loss = presence_loss = empty_loss = zero
        aux_stats = {"aux_nonempty_samples": 0, "aux_empty_samples": 0}
        if len(eligible):
            x0 = predicted_x0(noisy[eligible], epsilon[eligible],
                              timesteps[eligible], noise_scheduler)
            latent_loss = F.l1_loss(x0, target[eligible])
            (nonempty_mask_bce, empty_mask_bce, range_loss,
             presence_loss, empty_loss, _, aux_stats) = decoded_losses(
                vae, x0, target_image[eligible], scale,
                args.mask_threshold, args.empty_topk, args.empty_margin)

        loss = (epsilon_loss
                + args.latent_weight * latent_loss
                + args.mask_weight * nonempty_mask_bce
                + args.empty_mask_weight * empty_mask_bce
                + args.range_weight * range_loss
                + args.presence_weight * presence_loss
                + args.empty_suppress_weight * empty_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            record = {
                "step": step, "loss": float(loss),
                "epsilon_mse": float(epsilon_loss), "x0_l1": float(latent_loss),
                "mask_bce": float(nonempty_mask_bce),
                "empty_mask_bce": float(empty_mask_bce),
                "range_l1": float(range_loss), "presence": float(presence_loss),
                "empty_suppression": float(empty_loss),
                "aux_samples": int(len(eligible)), **aux_stats,
                "grad_norm": float(grad),
            }
            if stage1.DEVICE.type == "cuda":
                record["peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated() / 2**30)
            history.append(record)
            print(json.dumps(record), flush=True)

        reached_limit = deadline is not None and time.monotonic() >= deadline
        if step == args.steps or step % args.eval_every == 0 or reached_limit:
            result = evaluate_direct(
                model, vae, val_data, scale, args.ddim_steps,
                args.eval_batch_size, args.seed, args.mask_threshold)
            current_key = evaluation_key(result["summary"])
            summary = {
                "step": step, "initial_checkpoint_step": int(initial["step"]),
                "selection_key": list(current_key), **result["summary"],
            }
            print(json.dumps({"validation": summary}), flush=True)
            stage1.save_json(run_dir / f"validation_step_{step}.json", result)
            stage1.save_model(run_dir / "latest.pt", model, optimizer, step, summary)
            if current_key < best_key:
                best_key = current_key
                stage1.save_model(run_dir / "best.pt", model, optimizer, step, summary)
                stage1.save_json(best_path, summary)
            stage1.save_json(history_path, history)
        if reached_limit:
            print(json.dumps({"stopped_after_hours": args.max_hours,
                              "step": step}), flush=True)
            break

    if args.overfit and best_path.exists():
        best = json.loads(best_path.read_text())
        copy_data = DirectHorizonDataset(
            "train", args.data_root, args.latent_root, overfit=True,
            include_previous_image=True)
        copy = copy_baseline_summary(copy_data)
        gate = {
            "best": best, "copy": copy,
            "passed": (best["false_empty_frames"] == 0
                       and best["latent_mse"] < copy["copy_latent_mse"]),
            "criteria": "zero false-empty and generated latent MSE below copy",
        }
        stage1.save_json(run_dir / "overfit_gate.json", gate)
        print(json.dumps({"overfit_gate": gate}), flush=True)


@torch.no_grad()
def _decode_latents(vae, latents, scale):
    return vae.decode(latents / scale).sample.cpu().numpy()


def _method_rows(frames, target_frames, dataset, method, mask_threshold,
                 latent_errors=None):
    rows = []
    for index, (frame, target) in enumerate(zip(frames, target_frames)):
        row = frame_metrics(frame, target, mask_threshold)
        row.update({
            "method": method,
            "source_indices": [int(v) for v in dataset.source_indices[index]],
            "seed": int(dataset.seeds[index]),
        })
        if latent_errors is not None:
            row["latent_mse"] = float(latent_errors[index])
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_all(args):
    run_dir = args.out / args.run_name
    dataset = manifest_dataset(
        args.split, args, run_dir, include_baseline_inputs=True)
    metadata = json.loads((args.latent_root / "metadata.json").read_text())
    vae_identity = stage1.circular_vae_identity()
    if any(metadata.get(key) != value for key, value in vae_identity.items()):
        raise ValueError("Circular VAE weights changed; evaluation cache is stale")
    scale = float(metadata["scaling_factor"])
    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    direct = DirectHorizonWorldModel().to(stage1.DEVICE).float().eval()
    checkpoint = stage1.load_model(args.checkpoint, direct)
    scheduler = stage1.DDIMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon", clip_sample=False)

    target_frames = dataset.target_image.numpy()
    method_frames = {
        "copy": dataset.previous_image.numpy(),
        "vae_target_reconstruction": _decode_latents(
            vae, dataset.target.to(stage1.DEVICE), scale),
    }
    method_latents = {}
    direct_latents = []
    shuffled_latents = []
    autoregressive_latents = []

    autoregressive = None
    if not args.skip_autoregressive:
        autoregressive = stage1.WorldModel().to(stage1.DEVICE).float().eval()
        stage1.load_model(args.autoregressive_checkpoint, autoregressive)

    for start in range(0, len(dataset), args.eval_batch_size):
        stop = min(start + args.eval_batch_size, len(dataset))
        previous = dataset.previous[start:stop].to(stage1.DEVICE)
        actions = dataset.actions[start:stop].to(stage1.DEVICE)
        state = dataset.state[start:stop].to(stage1.DEVICE)
        direct_latents.append(generate(
            direct, scheduler, previous, actions, state,
            args.seed + start, args.ddim_steps).cpu())
        if len(actions) < 2:
            raise ValueError("Evaluation batch size must not leave a final batch of one")
        shuffled_latents.append(generate(
            direct, scheduler, previous, torch.roll(actions, 1, 0), state,
            args.seed + start, args.ddim_steps).cpu())
        if autoregressive is not None:
            rolled = previous
            for horizon in range(HORIZON):
                rolled = stage1.generate(
                    autoregressive, scheduler, rolled, actions[:, horizon], state,
                    args.seed + start + horizon * 100000,
                    num_steps=args.ddim_steps)
            autoregressive_latents.append(rolled.cpu())

    method_latents["direct_t3"] = torch.cat(direct_latents)
    method_latents["direct_t3_shuffled_actions"] = torch.cat(shuffled_latents)
    if autoregressive_latents:
        method_latents["autoregressive_one_step_x3"] = torch.cat(
            autoregressive_latents)
    for method, latents in method_latents.items():
        decoded = []
        for start in range(0, len(latents), args.eval_batch_size):
            decoded.append(_decode_latents(
                vae, latents[start:start + args.eval_batch_size].to(stage1.DEVICE),
                scale))
        method_frames[method] = np.concatenate(decoded)

    raw_root = args.raw_data_root
    warped = []
    ray_cache = {}
    for index in range(len(dataset)):
        seed_value = int(dataset.seeds[index])
        if seed_value not in ray_cache:
            ray_cache[seed_value] = load_rays(raw_root, args.split, seed_value)
        rays, azimuth, elevation = ray_cache[seed_value]
        transform = predict_transform(
            dataset.previous_drone_state[index], dt=TARGET_DT_SECONDS)
        warped.append(warp_frame(dataset.previous_image[index].numpy(), transform,
                                 rays, azimuth, elevation))
    method_frames["velocity_warp_0p48s"] = np.asarray(warped)

    all_rows = {}
    summaries = {}
    per_seed = {}
    target_latents = dataset.target.numpy()
    for method, frames in method_frames.items():
        latent_errors = None
        if method in method_latents:
            latent_errors = np.square(
                method_latents[method].numpy() - target_latents).reshape(
                    len(dataset), -1).mean(1)
        threshold = (0.0 if method in ("copy", "velocity_warp_0p48s")
                     else args.mask_threshold)
        rows = _method_rows(frames, target_frames, dataset, method,
                            threshold, latent_errors)
        all_rows[method] = rows
        summaries[method] = summarize_rows(rows)
        per_seed[method] = {
            str(seed): summarize_rows([row for row in rows if row["seed"] == seed])
            for seed in sorted({row["seed"] for row in rows})
        }

    comparison = {
        "split": args.split, "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "horizon": HORIZON, "target_seconds": TARGET_DT_SECONDS,
        "manifest": str(run_dir / f"fixed_{args.split}_manifest.json"),
        "summaries": summaries, "per_seed": per_seed,
        "rows": all_rows,
    }
    output_dir = run_dir / f"evaluation_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1.save_json(output_dir / "metrics.json", comparison)
    save_comparison_figures(output_dir, dataset, target_frames, method_frames,
                            args.mask_threshold, args.figure_samples)
    print(json.dumps({"summaries": summaries}, indent=2), flush=True)


def save_comparison_figures(output_dir, dataset, target, methods,
                            mask_threshold, count):
    names = ["copy", "direct_t3"]
    if "autoregressive_one_step_x3" in methods:
        names.append("autoregressive_one_step_x3")
    names.extend(["velocity_warp_0p48s", "vae_target_reconstruction"])
    for index in range(min(count, len(dataset))):
        columns = [("current", methods["copy"][index]),
                   ("target_t3", target[index])]
        columns.extend((name, methods[name][index]) for name in names[1:])
        fig, axes = plt.subplots(3, len(columns), figsize=(4 * len(columns), 10),
                                 constrained_layout=True)
        for column, (name, frame) in enumerate(columns):
            threshold = (0.0 if name in (
                "current", "target_t3", "velocity_warp_0p48s")
                         else mask_threshold)
            valid = frame[1, :, :18] > threshold
            ranges = np.clip((frame[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
            axes[0, column].imshow(np.where(valid, ranges, np.nan).T,
                                   origin="lower", vmin=0, vmax=10,
                                   aspect="auto")
            axes[0, column].set_title(name)
            axes[1, column].imshow(valid.T, origin="lower", vmin=0, vmax=1,
                                   cmap="gray", aspect="auto")
            points = stage1.to_points(frame, threshold)
            if len(points):
                axes[2, column].scatter(points[:, 0], points[:, 1], s=2)
            axes[2, column].set_xlim(-10, 10)
            axes[2, column].set_ylim(-10, 10)
            axes[2, column].set_aspect("equal")
        axes[0, 0].set_ylabel("range image")
        axes[1, 0].set_ylabel("hit mask")
        axes[2, 0].set_ylabel("point cloud XY")
        chain = "_".join(str(int(v)) for v in dataset.source_indices[index])
        fig.savefig(output_dir / f"comparison_{index:03d}_{chain}.png", dpi=140)
        plt.close(fig)


def inspect(args):
    report = {}
    for split in ("train", "val", "test"):
        chains, seeds, _, rejection = _candidate_chains(
            split, args.data_root, args.latent_root)
        unique, counts = np.unique(seeds, return_counts=True)
        report[split] = {
            "chains": len(chains),
            "per_seed": {str(int(seed)): int(count)
                         for seed, count in zip(unique, counts)},
            "rejection": rejection,
            "first_chain": [int(v) for v in chains[0]],
        }
    print(json.dumps(report, indent=2), flush=True)


def smoke(args):
    dataset = DirectHorizonDataset(
        "train", args.data_root, args.latent_root, overfit=True)
    model = DirectHorizonWorldModel().to(stage1.DEVICE).float()
    checkpoint = load_one_step_checkpoint(args.init_checkpoint, model)
    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    previous, target, actions, state, target_image, source = [
        value[:2].to(stage1.DEVICE) for value in next(iter(DataLoader(
            dataset, batch_size=2, shuffle=False)))]
    scheduler = stage1.DDPMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon")
    noise = torch.randn_like(target)
    timesteps = torch.tensor([10, 400], device=stage1.DEVICE)
    noisy = scheduler.add_noise(target, noise, timesteps)
    epsilon = model(noisy, previous, actions, state, timesteps)
    x0 = predicted_x0(noisy, epsilon, timesteps, scheduler)
    loss = F.mse_loss(epsilon, noise) + 0.1 * F.l1_loss(x0, target)
    loss.backward()
    ddim = stage1.DDIMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon", clip_sample=False)
    with torch.no_grad():
        generated = generate(model, ddim, previous, actions, state,
                             args.seed, args.ddim_steps)
        decoded = vae.decode(generated / dataset.scale).sample
    print(json.dumps({
        "checkpoint_step": int(checkpoint["step"]),
        "source_indices": source.cpu().tolist(),
        "previous_shape": list(previous.shape),
        "actions_shape": list(actions.shape),
        "condition_tokens": HORIZON * 10 + 1,
        "target_shape": list(target.shape),
        "epsilon_shape": list(epsilon.shape),
        "decoded_shape": list(decoded.shape),
        "loss": float(loss),
        "finite_gradients": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()),
    }, indent=2), flush=True)


class WorldV2View(Dataset):
    """World-model projection of the shared v2 policy/window dataset."""

    def __init__(self, base: V2WindowDataset):
        self.base = base
        self.scale = base.scale
        self.seeds = base.seeds
        self.source_indices = base.source_indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        (previous, target, actions, state, target_image, source,
         _goal, _proprio, _past, _past_mask) = self.base[index]
        shard = int(self.base.shard[index])
        current = int(self.base.rows[index, 0])
        h5, _ = self.base._handles(shard)
        current_image = torch.from_numpy(
            np.asarray(h5["range_values"][current], np.float32))
        return previous, target, actions, state, target_image, source, current_image


def _world_v2_runtime(precision):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable on the selected GPU")
    stage1.DEVICE = device
    return distributed, rank, world_size, local_rank, device


def _world_autocast(device, precision):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and precision == "bf16"
            else contextlib.nullcontext())


def _world_unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _load_world_initial_v2(path, model):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if any(key.startswith("world.") for key in state):
        state = {key[len("world."):]: value for key, value in state.items()
                 if key.startswith("world.")}
    model.load_state_dict(state, strict=True)
    return int(payload.get("step", -1))


@torch.no_grad()
def evaluate_world_v2(model, vae, dataset, args, device):
    model = _world_unwrap(model).eval()
    scheduler = stage1.DDIMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon", clip_sample=False)
    predicted_rows, shuffled_rows, copy_rows = [], [], []
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    offset = 0
    for batch in loader:
        previous, target, actions, state, target_image, source, current_image = batch
        previous, target, actions, state = [
            value.to(device, non_blocking=True)
            for value in (previous, target, actions, state)]
        with _world_autocast(device, args.precision):
            if hasattr(model, "prepare"):
                model.prepare(previous)
            prediction_latent = generate(
                model, scheduler, previous, actions, state,
                args.seed + offset, args.ddim_steps)
            shuffled_action = (torch.roll(actions, 1, 0) if len(actions) > 1
                               else actions.flip(1))
            shuffled_latent = generate(
                model, scheduler, previous, shuffled_action, state,
                args.seed + offset, args.ddim_steps)
            prediction = vae.decode(prediction_latent / dataset.scale).sample.float().cpu().numpy()
            shuffled = vae.decode(shuffled_latent / dataset.scale).sample.float().cpu().numpy()
        target_np, current_np = target_image.numpy(), current_image.numpy()
        for index in range(len(prediction)):
            base = {"source_indices": [int(v) for v in source[index]],
                    "seed": int(dataset.seeds[offset+index])}
            row = frame_metrics(prediction[index], target_np[index], args.mask_threshold)
            row.update(base); predicted_rows.append(row)
            row = frame_metrics(shuffled[index], target_np[index], args.mask_threshold)
            row.update(base); shuffled_rows.append(row)
            row = frame_metrics(current_np[index], target_np[index], 0.0)
            row.update(base); copy_rows.append(row)
        offset += len(prediction)
    predicted = summarize_rows(predicted_rows)
    shuffled = summarize_rows(shuffled_rows)
    copy = summarize_rows(copy_rows)
    prediction_metric = predicted["symmetric_nn_distance_m"]
    copy_metric = copy["symmetric_nn_distance_m"]
    shuffled_metric = shuffled["symmetric_nn_distance_m"]
    gate = {
        "prediction_improvement": 1-prediction_metric/max(copy_metric, 1e-8),
        "shuffle_degradation": shuffled_metric/max(prediction_metric, 1e-8)-1,
    }
    gate["passed"] = (gate["prediction_improvement"] >= 0.10
                      and gate["shuffle_degradation"] >= 0.05)
    return {"prediction": predicted, "copy": copy, "shuffled": shuffled,
            "gate": gate}


def _save_world_v2(path, model, optimizer, step, args, validation,
                   initial_step, best_key):
    payload = {
        "format": "navrl-world-direct-t3-v2",
        "model": _world_unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(), "step": int(step),
        "initial_checkpoint_step": int(initial_step),
        "validation": validation, "best_key": list(best_key),
        "provenance": {
            "dataset_manifest_sha256": file_sha256(args.dataset_root/"manifest.json"),
            "latent_metadata_sha256": file_sha256(args.latent_root/"metadata.json"),
            "initial_checkpoint_sha256": file_sha256(args.init_checkpoint),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_world_v2(args):
    distributed, rank, world_size, local_rank, device = _world_v2_runtime(args.precision)
    main = rank == 0
    stage1.seed_everything(args.seed+rank)
    run_dir = args.out/("world_direct_t3_v2_overfit" if args.overfit
                        else "world_direct_t3_v2")
    if main:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    train_base = V2WindowDataset(
        "train", args.dataset_root, args.latent_root, args.index_root,
        overfit=args.overfit, random_seed=args.seed)
    val_base = (V2WindowDataset(
        "train", args.dataset_root, args.latent_root, args.index_root,
        overfit=True, random_seed=args.seed, limit=args.samples_per_seed)
        if args.overfit else V2WindowDataset(
            "val", args.dataset_root, args.latent_root, args.index_root,
            samples_per_seed=args.samples_per_seed, random_seed=args.seed))
    train_data, val_data = WorldV2View(train_base), WorldV2View(val_base)
    model = DirectHorizonWorldModel().to(device).float()
    optimizer = _make_optimizer(model, args.lr, args.segment_lr)
    initial_step = _load_world_initial_v2(args.init_checkpoint, model)
    # A checkpoint that proves it uses actions and beats frame-copy must always
    # outrank one with slightly better reconstruction metrics that fails the
    # causal world-model gate.
    first_step, best_key = 1, (math.inf,)*6
    if args.resume:
        payload = torch.load(run_dir/"latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        first_step = int(payload["step"])+1
        best_key = tuple(payload.get("best_key", best_key))
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                         output_device=local_rank,
                                         broadcast_buffers=False)
    vae = stage1.load_circular_vae(); vae.requires_grad_(False)
    scheduler = stage1.DDPMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon")
    sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed) if distributed else None
    loader = stage1.infinite(DataLoader(
        train_data, batch_size=args.micro_batch_size, sampler=sampler,
        shuffle=sampler is None, drop_last=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0))
    if main:
        stage1.save_json(run_dir/"config.json", {
            "format": "navrl-world-direct-t3-v2", "steps": args.steps,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation": args.grad_accumulation,
            "world_size": world_size,
            "global_batch_size": args.micro_batch_size*args.grad_accumulation*world_size,
            "precision": args.precision, "lr": args.lr,
            "segment_lr": args.segment_lr, "initial_checkpoint": str(args.init_checkpoint),
            "target": "t+3 / 0.48 seconds", "future_state_leakage": False,
        })
    history = []
    try:
        for step in range(first_step, args.steps+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            totals = defaultdict(float)
            for micro in range(args.grad_accumulation):
                batch = next(loader)
                previous, target, actions, state, target_image = [
                    value.to(device, non_blocking=True) for value in batch[:5]]
                noise = torch.randn_like(target)
                timestep = torch.randint(0, 1000, (len(target),), device=device)
                noisy = scheduler.add_noise(target, noise, timestep)
                sync = micro == args.grad_accumulation-1
                no_sync = contextlib.nullcontext() if sync or not distributed else model.no_sync()
                with no_sync, _world_autocast(device, args.precision):
                    epsilon = model(noisy, previous, actions, state, timestep)
                    epsilon_loss = F.mse_loss(epsilon, noise)
                    eligible = select_aux_indices(
                        timestep, target_image, args.aux_t_max,
                        args.aux_batch_max, args.aux_empty_max)
                    zero = epsilon_loss*0
                    latent = mask = empty_mask = range_loss = presence = empty = zero
                    if len(eligible):
                        x0 = predicted_x0(noisy[eligible], epsilon[eligible],
                                          timestep[eligible], scheduler)
                        latent = F.l1_loss(x0, target[eligible])
                        (mask, empty_mask, range_loss, presence, empty, _, _) = decoded_losses(
                            vae, x0, target_image[eligible], train_data.scale,
                            args.mask_threshold, args.empty_topk, args.empty_margin)
                    loss = (epsilon_loss + args.latent_weight*latent
                            + args.mask_weight*mask + args.empty_mask_weight*empty_mask
                            + args.range_weight*range_loss + args.presence_weight*presence
                            + args.empty_suppress_weight*empty) / args.grad_accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite world loss at step {step}")
                loss.backward()
                totals["loss"] += float(loss.detach())
                totals["epsilon_mse"] += float(epsilon_loss.detach())/args.grad_accumulation
                totals["x0_l1"] += float(latent.detach())/args.grad_accumulation
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad):
                raise FloatingPointError(f"non-finite world gradient at step {step}")
            optimizer.step()
            if main and (step == 1 or step % args.log_every == 0):
                record = {"step": step, **totals, "grad_norm": float(grad)}
                if device.type == "cuda":
                    record["peak_allocated_gib"] = (
                        torch.cuda.max_memory_allocated(device) / 2**30)
                history.append(record); print(json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                if distributed: dist.barrier()
                if main:
                    result = evaluate_world_v2(model, vae, val_data, args, device)
                    current_key = (
                        int(not result["gate"]["passed"]),
                        *evaluation_key(result["prediction"]),
                    )
                    print(json.dumps({"step": step, "validation": result}), flush=True)
                    _save_world_v2(run_dir/"latest.pt", model, optimizer, step,
                                   args, result, initial_step, min(best_key, current_key))
                    if current_key < best_key:
                        best_key = current_key
                        _save_world_v2(run_dir/"best.pt", model, optimizer, step,
                                       args, result, initial_step, best_key)
                        stage1.save_json(run_dir/"best_metrics.json",
                                         {"step": step, **result})
                    stage1.save_json(run_dir/"world_gate.json", result["gate"])
                    stage1.save_json(run_dir/"history.json", history)
                if distributed: dist.barrier()
    finally:
        train_base.close()
        if val_base is not train_base: val_base.close()
        if distributed: dist.destroy_process_group()


def _add_v2_common(parser):
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=stage1.OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-seed", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--mask-threshold", type=float, default=1.5)


def _add_common(parser):
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=stage1.OUT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-seed", type=int, default=256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect")
    _add_common(inspect_parser)

    smoke_parser = commands.add_parser("smoke")
    _add_common(smoke_parser)
    smoke_parser.add_argument("--init-checkpoint", type=Path, required=True)
    smoke_parser.add_argument("--ddim-steps", type=int, default=2)

    train_parser = commands.add_parser("train")
    _add_common(train_parser)
    train_parser.add_argument("--init-checkpoint", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--eval-batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--segment-lr", type=float, default=None)
    train_parser.add_argument("--eval-every", type=int, default=None)
    train_parser.add_argument("--log-every", type=int, default=20)
    train_parser.add_argument("--ddim-steps", type=int, default=20)
    train_parser.add_argument("--latent-weight", type=float, default=0.1)
    train_parser.add_argument("--mask-weight", type=float, default=0.1)
    train_parser.add_argument("--empty-mask-weight", type=float, default=0.1)
    train_parser.add_argument("--range-weight", type=float, default=0.05)
    train_parser.add_argument("--presence-weight", type=float, default=0.02)
    train_parser.add_argument("--empty-suppress-weight", type=float, default=0.2)
    train_parser.add_argument("--aux-t-max", type=int, default=500)
    train_parser.add_argument("--aux-batch-max", type=int, default=64)
    train_parser.add_argument("--aux-empty-max", type=int, default=16)
    train_parser.add_argument("--empty-topk", type=int, default=8)
    train_parser.add_argument("--empty-margin", type=float, default=0.5)
    train_parser.add_argument("--mask-threshold", type=float, default=1.5)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--max-hours", type=float, default=None)
    train_parser.add_argument("--overfit", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--skip-initial-eval", action="store_true")

    evaluate_parser = commands.add_parser("evaluate")
    _add_common(evaluate_parser)
    evaluate_parser.add_argument("--split", choices=("val", "test"), default="val")
    evaluate_parser.add_argument("--checkpoint", type=Path, default=None)
    evaluate_parser.add_argument("--autoregressive-checkpoint", type=Path,
                                 default=None)
    evaluate_parser.add_argument("--skip-autoregressive", action="store_true")
    evaluate_parser.add_argument("--raw-data-root", type=Path, default=None)
    evaluate_parser.add_argument("--eval-batch-size", type=int, default=8)
    evaluate_parser.add_argument("--ddim-steps", type=int, default=20)
    evaluate_parser.add_argument("--mask-threshold", type=float, default=1.5)
    evaluate_parser.add_argument("--figure-samples", type=int, default=16)

    train_v2_parser = commands.add_parser("train-v2")
    _add_v2_common(train_v2_parser)
    train_v2_parser.add_argument("--init-checkpoint", type=Path, required=True)
    train_v2_parser.add_argument("--steps", type=int, default=20000)
    train_v2_parser.add_argument("--micro-batch-size", type=int, default=1)
    train_v2_parser.add_argument("--grad-accumulation", type=int, default=4)
    train_v2_parser.add_argument("--lr", type=float, default=1e-5)
    train_v2_parser.add_argument("--segment-lr", type=float, default=5e-5)
    train_v2_parser.add_argument("--eval-every", type=int, default=500)
    train_v2_parser.add_argument("--log-every", type=int, default=20)
    train_v2_parser.add_argument("--latent-weight", type=float, default=0.1)
    train_v2_parser.add_argument("--mask-weight", type=float, default=0.1)
    train_v2_parser.add_argument("--empty-mask-weight", type=float, default=0.1)
    train_v2_parser.add_argument("--range-weight", type=float, default=0.05)
    train_v2_parser.add_argument("--presence-weight", type=float, default=0.02)
    train_v2_parser.add_argument("--empty-suppress-weight", type=float, default=0.2)
    train_v2_parser.add_argument("--aux-t-max", type=int, default=500)
    train_v2_parser.add_argument("--aux-batch-max", type=int, default=64)
    train_v2_parser.add_argument("--aux-empty-max", type=int, default=16)
    train_v2_parser.add_argument("--empty-topk", type=int, default=8)
    train_v2_parser.add_argument("--empty-margin", type=float, default=0.5)
    train_v2_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_v2_parser.add_argument("--overfit", action="store_true")
    train_v2_parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    if args.command == "train-v2":
        args.dataset_root = args.dataset_root.expanduser().resolve()
        args.latent_root = args.latent_root.expanduser().resolve()
        args.index_root = ((args.latent_root.parent/"index") if args.index_root is None
                           else args.index_root.expanduser().resolve())
        args.out = args.out.expanduser().resolve()
        args.init_checkpoint = args.init_checkpoint.expanduser().resolve()
        if args.overfit and args.steps == 20000:
            args.steps = 1000
            args.eval_every = min(args.eval_every, 200)
        train_world_v2(args)
        return
    args.data_root = args.data_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.latent_root = ((args.out / stage1.LATENT_DIR) if args.latent_root is None
                        else args.latent_root.expanduser().resolve())
    stage1.DATA = args.data_root
    stage1.seed_everything(args.seed)

    if args.command == "inspect":
        inspect(args)
        return
    if args.command == "smoke":
        args.init_checkpoint = args.init_checkpoint.expanduser().resolve()
        smoke(args)
        return
    if args.command == "train":
        args.init_checkpoint = args.init_checkpoint.expanduser().resolve()
        if args.overfit:
            args.run_name = ("world_direct_t3_overfit" if args.run_name == DEFAULT_RUN_NAME
                             else args.run_name)
            args.steps = 1000 if args.steps is None else args.steps
            args.batch_size = 128 if args.batch_size is None else args.batch_size
            args.lr = 1e-4 if args.lr is None else args.lr
            args.segment_lr = args.lr if args.segment_lr is None else args.segment_lr
            args.eval_every = 200 if args.eval_every is None else args.eval_every
        else:
            args.steps = 20000 if args.steps is None else args.steps
            args.batch_size = 1024 if args.batch_size is None else args.batch_size
            args.lr = 1e-5 if args.lr is None else args.lr
            args.segment_lr = 5e-5 if args.segment_lr is None else args.segment_lr
            args.eval_every = 500 if args.eval_every is None else args.eval_every
        print(f"device={stage1.DEVICE}", flush=True)
        train(args)
        return

    args.checkpoint = ((args.out / args.run_name / "best.pt")
                       if args.checkpoint is None
                       else args.checkpoint.expanduser().resolve())
    args.autoregressive_checkpoint = (
        (args.out / "world_circular_causal_8h" / "best.pt")
        if args.autoregressive_checkpoint is None
        else args.autoregressive_checkpoint.expanduser().resolve())
    args.raw_data_root = (args.data_root.parent if args.raw_data_root is None
                          else args.raw_data_root.expanduser().resolve())
    evaluate_all(args)


if __name__ == "__main__":
    main()
