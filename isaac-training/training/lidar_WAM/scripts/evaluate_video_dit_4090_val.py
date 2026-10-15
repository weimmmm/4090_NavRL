"""Evaluate direct or residual t+1 models on shared 4090 val windows.

Read raw images and re-encode with the checkpoint's matched 5 m VAE. The
4090 latent cache belongs to a different VAE and must not be reused here.
"""

from __future__ import annotations

import argparse
import bisect
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner import stage1
from lidar_wam.runner.video_dit_cached_data import sha256


def shard_key(path):
    return "/".join(Path(path).parts[-3:])


class RawValidationWindows(Dataset):
    def __init__(self, dataset_root, index_root, config, count, seed,
                 indices_file=None):
        self.dataset_root = Path(dataset_root)
        self.index_root = Path(index_root) / "val"
        metadata = json.loads((self.index_root / "metadata.json").read_text())
        self.entries = metadata["entries"]
        counts = np.asarray(metadata["counts"], dtype=np.int64)
        self.ends = np.cumsum(counts)
        self.total_windows = int(self.ends[-1])
        held_out = {shard_key(entry["source"])
                    for entry in config["validation_shards"]}
        training = {shard_key(entry["source"])
                    for entry in config["train_shards"]}
        assert not (held_out & training), "L20 train/val shards overlap"
        eligible = []
        self.held_out_shards = []
        for shard, (entry, size) in enumerate(zip(self.entries, counts)):
            if shard_key(entry["dataset"]) in held_out:
                start = 0 if shard == 0 else int(self.ends[shard - 1])
                eligible.append(np.arange(start, start + size, dtype=np.int64))
                self.held_out_shards.append(entry["dataset"])
        candidates = np.concatenate(eligible) if eligible else np.empty(0, np.int64)
        self.eligible_windows = len(candidates)
        if len(candidates) < count:
            raise ValueError(f"Need {count} independent windows, got {len(candidates)}")
        self.indices = np.sort(np.random.default_rng(seed).choice(
            candidates, count, replace=False))
        if indices_file is not None:
            indices = np.load(indices_file)
            if (indices.ndim != 1 or len(indices) != count
                    or len(np.unique(indices)) != count
                    or not np.isin(indices, candidates).all()):
                raise ValueError("Indices must identify distinct shared validation windows")
            self.indices = indices.astype(np.int64, copy=False)
        self.rows = {}
        self.handles = {}

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["rows"] = {}
        state["handles"] = {}
        return state

    def __getitem__(self, index):
        global_index = int(self.indices[index])
        shard = bisect.bisect_right(self.ends, global_index)
        start = 0 if shard == 0 else int(self.ends[shard - 1])
        if shard not in self.rows:
            self.rows[shard] = np.load(
                self.index_root / f"rows_{shard:04d}.npy", mmap_mode="r")
            self.handles[shard] = h5py.File(
                self.dataset_root / self.entries[shard]["dataset"], "r")
        chain = self.rows[shard][global_index - start]
        frames = self.handles[shard]["frames"]
        images = np.stack([np.asarray(frames["range_values"][int(row)],
                                      dtype=np.float32) for row in chain])
        return torch.from_numpy(images)


def frame_metrics(prediction, truth, directions, range_m, threshold):
    gt_hit = truth[1, :, :18] > 0
    pred_hit = prediction[1, :, :18] > threshold
    gt_depth = np.clip((truth[0, :, :18] + 1) * range_m / 2, 0, range_m)
    pred_depth = np.clip((prediction[0, :, :18] + 1) * range_m / 2, 0, range_m)
    error = np.abs(pred_depth - gt_depth)
    p = directions[pred_hit] * pred_depth[pred_hit, None]
    g = directions[gt_hit] * gt_depth[gt_hit, None]
    if not len(p) and not len(g):
        cd = 0.0
    elif not len(p) or not len(g):
        cd = float(range_m ** 2)
    else:
        cd = 0.5 * (np.square(cKDTree(g).query(p)[0]).mean()
                    + np.square(cKDTree(p).query(g)[0]).mean())
    hit_mae = float(error[gt_hit].mean()) if gt_hit.any() else 0.0
    return {
        "cd_m2": float(cd), "gt_hits": int(gt_hit.sum()),
        "pred_hits": int(pred_hit.sum()),
        "depth_error_sum_m": float(error[gt_hit].sum()),
        "all_depth_error_sum_m": float(error.sum()),
        "generation_loss": 0.5 * (
            float((pred_hit != gt_hit).mean()) + hit_mae / range_m),
        "tp": int((pred_hit & gt_hit).sum()),
        "fp": int((pred_hit & ~gt_hit).sum()),
        "fn": int((~pred_hit & gt_hit).sum()),
    }


def summarize_images(rows):
    nonempty = [row for row in rows if row["gt_hits"]]
    both_nonempty = [row for row in nonempty if row["pred_hits"]]
    tp, fp, fn = [sum(row[key] for row in rows) for key in ("tp", "fp", "fn")]
    valid_count = sum(row["gt_hits"] for row in rows)
    return {
        "generation_loss": float(np.mean([row["generation_loss"] for row in rows])),
        "depth_mae_gt_hits_m": sum(row["depth_error_sum_m"] for row in rows) / max(valid_count, 1),
        "depth_mae_all_rays_m": sum(row["all_depth_error_sum_m"] for row in rows) / (len(rows)*108*18),
        "nonempty_gt_cd_m2": float(np.mean([row["cd_m2"] for row in nonempty])) if nonempty else None,
        "both_nonempty_cd_m2": float(np.mean([row["cd_m2"] for row in both_nonempty])) if both_nonempty else None,
        "nonempty_gt_samples": len(nonempty),
        "false_empty_frames": sum(row["gt_hits"] > 0 and row["pred_hits"] == 0 for row in rows),
        "false_hit_frames": sum(row["gt_hits"] == 0 and row["pred_hits"] > 0 for row in rows),
        "mask_f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def to_next_latent(generated, current, residual_scale):
    if residual_scale is None:
        return generated
    return current + residual_scale * generated


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--video-index-root", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--vae-chunk-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--selection-run", type=Path,
                        help="Run whose held-out shard configuration defines the shared subset")
    parser.add_argument("--indices-file", type=Path,
                        help="Reuse exactly these saved global validation window indices")
    parser.add_argument("--extra-mask-threshold", type=float, action="append", default=[])
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.vae_chunk_size, args.flow_steps) <= 0:
        raise ValueError("Counts must be positive")
    if (args.out / "metrics.json").exists():
        raise FileExistsError("Evaluation already completed; select a new output directory")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    config = json.loads((args.run / "config.json").read_text())
    selection_config = (json.loads((args.selection_run / "config.json").read_text())
                        if args.selection_run is not None else config)
    manifest = json.loads((args.dataset_root / "manifest.json").read_text())
    range_m = float(manifest["lidar"]["recording_range_m"])
    if range_m != 5.0:
        raise ValueError("The migrated model expects 5 m normalized range images")
    payload = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    expected_vae_sha256 = payload.get("vae", {}).get(
        "vae_sha256", config.get("vae_sha256"))
    if expected_vae_sha256 is None or sha256(args.vae_checkpoint) != expected_vae_sha256:
        raise ValueError("The selected VAE does not match the trained model")
    dataset = RawValidationWindows(args.dataset_root, args.video_index_root,
                                   selection_config, args.samples, args.seed,
                                   args.indices_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    architecture = payload["architecture"]
    if (architecture.get("target_offset_lidar_frames", 1) != 1
            or architecture.get("prediction_frames", 1) != 1):
        raise ValueError("Expected a direct t+1 checkpoint")
    prediction_target = architecture.get("prediction_target", "absolute_latent")
    if prediction_target not in ("absolute_latent", "normalized_residual"):
        raise ValueError(f"Unsupported prediction target: {prediction_target}")
    residual_scale = None
    if prediction_target == "normalized_residual":
        residual_scale = torch.tensor(architecture["residual_scale"],
                                      device=device).reshape(1, 4, 1, 1)
        if not bool(torch.isfinite(residual_scale).all()) or bool((residual_scale <= 0).any()):
            raise ValueError("Invalid residual normalization scale")
    scale = float(payload["vae"]["scaling_factor"])
    vae_config = json.loads((args.vae_checkpoint.parent / "config.json").read_text())
    if not np.isclose(scale, vae_config["scaling_factor"], rtol=0, atol=1e-12):
        raise ValueError("Checkpoint/VAE scaling factors differ")
    model = LiDARVideoDiT(architecture["width"], architecture["depth"],
                         architecture["heads"], architecture["mlp_ratio"])
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    checkpoint_step = int(payload["step"])
    del payload
    vae = stage1.make_vae(circular=True)
    vae.load_state_dict(load_file(str(args.vae_checkpoint), device="cpu"), strict=True)
    vae = vae.to(device).float().eval().requires_grad_(False)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "validation_indices.npy", dataset.indices)
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    rows = []
    extra_rows = {threshold: [] for threshold in args.extra_mask_threshold}
    loss_sum = flow_sum = latent_l1_sum = 0.0
    started = time.monotonic()
    print(json.dumps({"checkpoint_step": checkpoint_step,
                      "validation_windows": dataset.total_windows,
                      "independent_windows": dataset.eligible_windows,
                      "samples": len(dataset), "range_m": range_m}), flush=True)
    for images in loader:
        b = len(images)
        flat = images.to(device, non_blocking=True).flatten(0, 1)
        latent = torch.cat([vae.encode(chunk).latent_dist.mode() * scale
                            for chunk in flat.split(args.vae_chunk_size)])
        latent = latent.reshape(b, 4, 4, 27, 5)
        history, target = latent[:, :3], latent[:, 3]
        noise = torch.randn(target.shape, device=device, generator=generator)
        sigma = torch.rand(b, device=device, generator=generator)
        view = sigma[:, None, None, None]
        clean = (target if residual_scale is None
                 else (target - history[:, -1]) / residual_scale)
        noisy = (1 - view) * clean + view * noise
        value = torch.randn(target.shape, device=device, generator=generator)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = model(history, noisy, sigma).float()
            flow = (velocity - (noise - clean)).square().mean()
            predicted_next = to_next_latent(
                noisy - view * velocity, history[:, -1], residual_scale)
            latent_l1 = (predicted_next - target).abs().mean()
            for step in range(args.flow_steps):
                timestep = torch.full((b,), 1 - step / args.flow_steps, device=device)
                value = value - model(history, value, timestep).float() / args.flow_steps
        value = to_next_latent(value, history[:, -1], residual_scale)
        decoded = torch.cat([vae.decode(chunk / scale).sample.float()
                             for chunk in value.split(args.vae_chunk_size)])
        loss_sum += float(flow + 0.1 * latent_l1) * b
        flow_sum += float(flow) * b
        latent_l1_sum += float(latent_l1) * b
        for prediction, truth in zip(decoded.cpu().numpy(), images[:, 3].numpy()):
            rows.append(frame_metrics(prediction, truth, stage1.DIRECTIONS,
                                      range_m, args.mask_threshold))
            for threshold, threshold_rows in extra_rows.items():
                threshold_rows.append(frame_metrics(
                    prediction, truth, stage1.DIRECTIONS, range_m, threshold))
        print(json.dumps({"samples_done": len(rows), "total": len(dataset),
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    metrics = {
        "checkpoint_step": checkpoint_step, "samples": len(rows), "seed": args.seed,
        "validation_windows": dataset.total_windows,
        "eligible_independent_windows": dataset.eligible_windows,
        "validation_shards": dataset.held_out_shards,
        "selection": "4090 val intersected with L20 held-out shards",
        "checkpoint": str(args.run / "best.pt"), "vae_checkpoint": str(args.vae_checkpoint),
        "vae_scale": scale, "depth_range_m": range_m,
        "prediction_frames": 1, "prediction_interval_s": 0.16,
        "prediction_target": prediction_target,
        "residual_scale": (residual_scale.flatten().tolist()
                           if residual_scale is not None else None),
        "reused_indices_file": str(args.indices_file) if args.indices_file else None,
        "flow_steps": args.flow_steps, "sampling": "single_random_euler",
        "mask_threshold": args.mask_threshold, "vae_precision": "fp32",
        "latent_validation_loss": loss_sum / len(rows),
        "flow_mse": flow_sum / len(rows), "latent_x0_l1": latent_l1_sum / len(rows),
        **summarize_images(rows),
        "extra_mask_thresholds": {str(threshold): summarize_images(threshold_rows)
                                  for threshold, threshold_rows in extra_rows.items()},
        "empty_prediction_penalty_m2": range_m ** 2,
        "chamfer_definition": "0.5*(mean_P min_Q squared_distance+mean_Q min_P squared_distance)",
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out / "per_window_metrics.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps({"result": metrics}), flush=True)


if __name__ == "__main__":
    main()
