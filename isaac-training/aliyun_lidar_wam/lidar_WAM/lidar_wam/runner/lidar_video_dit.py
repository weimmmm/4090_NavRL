"""Train a history-only LiDAR Video DiT with rectified flow.

The model consumes frozen-VAE latents at t-2, t-1 and t and predicts t+1.
It intentionally has no action, state, goal or privileged simulator input.

The recommended parameterization predicts the normalized latent residual
``(z[t+1] - z[t]) / residual_scale``.  Reconstructing the next latent as
``z[t] + residual_scale * predicted_residual`` gives the short-horizon model
an explicit copy-current-frame prior.  Legacy checkpoints that generate the
absolute next latent remain supported.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from lidar_wam.coordinates import numpy_goal_frame_causal_features
from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner import stage1
from lidar_wam.runner.world_direct_horizon import frame_metrics


FORMAT = "navrl-lidar-video-dit-v2"
INDEX_FORMAT = "navrl-lidar-history3-next1-index-v1"
ABSOLUTE_TARGET = "absolute_latent"
RESIDUAL_TARGET = "normalized_residual"


def _decode(values):
    return [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in values]


def _atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _entries_from_archive(path: Path):
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    return archive, metadata["entries"], metadata


def build_video_index(dataset_root: Path, base_index_root: Path,
                      output_root: Path, split: str):
    """Materialize strict t-2,t-1,t,t+1 row chains shard by shard."""
    output = output_root / split
    output.mkdir(parents=True, exist_ok=True)
    archive, entries, base_metadata = _entries_from_archive(
        base_index_root / f"{split}_windows.npz")
    all_shards = np.asarray(archive["shard"])
    base_rows = np.asarray(archive["rows"])
    counts, rejected = [], 0
    for shard_id, entry in enumerate(entries):
        destination = output / f"rows_{shard_id:04d}.npy"
        candidates = base_rows[all_shards == shard_id, :2]
        if destination.exists():
            counts.append(int(np.load(destination, mmap_mode="r").shape[0]))
            continue
        path = dataset_root / entry["dataset"]
        with h5py.File(path, "r") as handle:
            frames = handle["frames"] if "frames" in handle else handle
            tokens = _decode(frames["token"][:])
            previous = _decode(frames["prev_token"][:])
        token_to_row = {token: row for row, token in enumerate(tokens)}
        chains = []
        for current, following in candidates.tolist():
            first_token = previous[current]
            first = token_to_row.get(first_token, -1) if first_token else -1
            second_token = previous[first] if first >= 0 else ""
            second = token_to_row.get(second_token, -1) if second_token else -1
            if first < 0 or second < 0:
                rejected += 1
                continue
            chains.append((second, first, current, following))
        value = np.asarray(chains, dtype=np.int64).reshape(-1, 4)
        np.save(destination, value)
        counts.append(len(value))
        print(json.dumps({"split": split, "shard": shard_id,
                          "windows": len(value)}), flush=True)
    metadata = {
        "format": INDEX_FORMAT,
        "split": split,
        "counts": counts,
        "total_windows": int(sum(counts)),
        "rejected_missing_history": int(rejected),
        "entries": entries,
        "dataset_manifest_sha256": base_metadata["dataset_manifest_sha256"],
        "alignment": ["t-2", "t-1", "t", "t+1"],
        "conditions": [],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class HistoryLatentDataset(Dataset):
    def __init__(self, dataset_root: Path, latent_root: Path,
                 video_index_root: Path, split: str, return_target_image=False):
        self.dataset_root = Path(dataset_root)
        self.latent_root = Path(latent_root)
        self.root = Path(video_index_root) / split
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("format") != INDEX_FORMAT:
            raise ValueError("incompatible history index")
        self.entries = self.metadata["entries"]
        self.counts = np.asarray(self.metadata["counts"], dtype=np.int64)
        self.ends = np.cumsum(self.counts)
        self._rows = {}
        self._latents = {}
        self._files = {}
        self.return_target_image = bool(return_target_image)
        latent_metadata = json.loads((self.latent_root / "metadata.json").read_text())
        self.vae_metadata = {
            key: latent_metadata.get(key) for key in (
                "scaling_factor", "vae_variant", "vae_step",
                "vae_checkpoint", "vae_sha256")}

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getstate__(self):
        value = dict(self.__dict__)
        value["_rows"] = {}
        value["_latents"] = {}
        value["_files"] = {}
        return value

    def _open(self, shard: int):
        if shard not in self._rows:
            self._rows[shard] = np.load(
                self.root / f"rows_{shard:04d}.npy", mmap_mode="r")
            filename = self.entries[shard].get("latent_cache")
            if not filename:
                raise ValueError(f"manifest entry {shard} has no latent_cache")
            self._latents[shard] = np.load(
                self.latent_root / filename, mmap_mode="r")
        return self._rows[shard], self._latents[shard]

    def _target_image(self, shard: int, row: int) -> np.ndarray:
        """Read raw GT only for the small validation subset.

        Training stays entirely on the frozen-VAE latent cache.  Chamfer is
        deliberately measured against the physical range image, rather than a
        VAE reconstruction of it.
        """
        if shard not in self._files:
            self._files[shard] = h5py.File(
                self.dataset_root / self.entries[shard]["dataset"], "r")
        handle = self._files[shard]
        frames = handle["frames"] if "frames" in handle else handle
        return np.asarray(frames["range_values"][row], dtype=np.float32)

    def __getitem__(self, index):
        shard = bisect.bisect_right(self.ends, int(index))
        start = 0 if shard == 0 else int(self.ends[shard - 1])
        rows, latents = self._open(shard)
        chain = rows[int(index) - start]
        history = np.asarray(latents[chain[:3]], dtype=np.float32)
        target = np.asarray(latents[int(chain[3])], dtype=np.float32)
        if self.return_target_image:
            return (torch.from_numpy(history), torch.from_numpy(target),
                    torch.from_numpy(self._target_image(shard, int(chain[3]))))
        return torch.from_numpy(history), torch.from_numpy(target)


class MotionHistoryLatentDataset(HistoryLatentDataset):
    """History latents plus strictly causal motion observations.

    The action history is read from the three observed rows ``[t-2,t-1,t]``
    and flattened to ``[30,3]``.  Proprioception is computed from the current
    row ``t`` only.  In particular, no successor action or future state is
    exposed to the Future DiT.
    """

    past_horizon = 30

    def __init__(self, *args, past_horizon=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_horizon = int(past_horizon)
        if self.past_horizon != 30:
            raise ValueError("dataset action history currently requires 30 steps")
        self._action_files = {}

    def __getstate__(self):
        value = super().__getstate__()
        value["_action_files"] = {}
        return value

    def _action_frames(self, shard: int):
        if shard not in self._action_files:
            handle = h5py.File(
                self.dataset_root / self.entries[shard]["dataset"], "r")
            self._action_files[shard] = (
                handle["frames"] if "frames" in handle else handle)
        return self._action_files[shard]

    def _location(self, index: int):
        shard = bisect.bisect_right(self.ends, int(index))
        start = 0 if shard == 0 else int(self.ends[shard - 1])
        rows, _ = self._open(shard)
        return shard, rows[int(index) - start]

    def __getitem__(self, index):
        base = super().__getitem__(index)
        history, target = base[:2]
        shard, chain = self._location(int(index))
        frames = self._action_frames(shard)
        rows = np.asarray(chain[:3], dtype=np.int64)
        actions = np.asarray(
            frames["normalized_action_sequence"][rows], dtype=np.float32)
        masks = np.asarray(frames["action_mask"][rows], dtype=np.float32)
        finite = np.isfinite(actions).all(axis=-1)
        actions = np.where(np.isfinite(actions), actions, 0.0)
        masks = masks * finite.astype(np.float32)
        actions = actions.reshape(self.past_horizon, 3)
        masks = masks.reshape(self.past_horizon)
        current = np.asarray([int(chain[2])], dtype=np.int64)
        state = np.asarray(frames["drone_state"][current], dtype=np.float32)
        target_position = np.asarray(
            frames["target_position"][current], dtype=np.float32)
        target_direction = np.asarray(
            frames["target_dir_2d"][current], dtype=np.float32)
        goal, proprio = numpy_goal_frame_causal_features(
            state, target_position, target_direction)
        result = (history, target, torch.from_numpy(actions),
                  torch.from_numpy(masks), torch.from_numpy(proprio[0]),
                  torch.from_numpy(goal[0]))
        if self.return_target_image:
            result = result + (base[2],)
        return result


def estimate_residual_scale(dataset: HistoryLatentDataset, samples: int,
                            seed: int) -> dict:
    """Estimate a per-channel RMS without moving the zero-residual origin.

    Standard centering would make a zero normalized target decode to the mean
    motion rather than to an exact copy of the current frame.  RMS-only
    normalization preserves zero as the copy baseline while keeping the four
    latent channels at comparable scales.
    """
    count = min(max(int(samples), 1), len(dataset))
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(dataset), count, replace=False)
    sum_value = np.zeros(4, dtype=np.float64)
    sum_square = np.zeros(4, dtype=np.float64)
    elements = 0
    for index in indices.tolist():
        sample = dataset[int(index)]
        # Occupancy-supervised training returns the physical target image as a
        # third item; latent residual statistics use only the first two.
        history, target = sample[:2]
        residual = (
            target.numpy().astype(np.float64, copy=False)
            - history[-1].numpy().astype(np.float64, copy=False))
        sum_value += residual.sum(axis=(1, 2))
        sum_square += np.square(residual).sum(axis=(1, 2))
        elements += int(residual.shape[1] * residual.shape[2])
    mean = sum_value / max(elements, 1)
    rms = np.sqrt(sum_square / max(elements, 1))
    rms = np.maximum(rms, 1e-4)
    return {
        "samples": int(count),
        "elements_per_channel": int(elements),
        "mean": mean.astype(np.float32).tolist(),
        "scale_rms": rms.astype(np.float32).tolist(),
    }


def estimate_motion_stats(dataset: MotionHistoryLatentDataset, samples: int,
                          seed: int) -> dict:
    """Estimate training-set proprioception statistics without future data."""
    count = min(max(int(samples), 1), len(dataset))
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(dataset), count, replace=False)
    values = []
    goals = []
    action_values = []
    action_masks = []
    for index in indices.tolist():
        sample = dataset[int(index)]
        # Validation appends target_image after proprio and goal.
        proprio_index = -3 if dataset.return_target_image else -2
        goal_index = -2 if dataset.return_target_image else -1
        values.append(sample[proprio_index].numpy())
        goals.append(sample[goal_index].numpy())
        action_values.append(sample[2].numpy())
        action_masks.append(sample[3].numpy())
    value = np.asarray(values, dtype=np.float64)
    goal_value = np.asarray(goals, dtype=np.float64)
    action_value = np.asarray(action_values, dtype=np.float64).reshape(-1, 3)
    action_mask = np.asarray(action_masks, dtype=np.float64).reshape(-1)
    valid_action = action_value[action_mask > 0.0]
    if len(valid_action):
        valid_action = np.clip(valid_action, 1e-5, 1.0 - 1e-5)
        action_logit = np.log(valid_action) - np.log1p(-valid_action)
        action_mean = action_logit.mean(0)
        action_std = np.maximum(action_logit.std(0), 1e-4)
    else:
        action_mean = np.zeros(3, dtype=np.float64)
        action_std = np.ones(3, dtype=np.float64)
    return {
        "samples": int(count),
        "mean": value.mean(0).astype(np.float32).tolist(),
        "std": np.maximum(value.std(0), 1e-4).astype(np.float32).tolist(),
        "goal_mean": goal_value.mean(0).astype(np.float32).tolist(),
        "goal_std": np.maximum(goal_value.std(0), 1e-4).astype(np.float32).tolist(),
        "action_logit_mean": action_mean.astype(np.float32).tolist(),
        "action_logit_std": action_std.astype(np.float32).tolist(),
    }


def configure_prediction_target(model, prediction_target: str,
                                residual_scale=None):
    """Attach checkpoint-level target semantics without changing state keys."""
    source = model.module if isinstance(model, DistributedDataParallel) else model
    prediction_target = str(prediction_target)
    if prediction_target not in (ABSOLUTE_TARGET, RESIDUAL_TARGET):
        raise ValueError(f"unknown prediction target {prediction_target!r}")
    if residual_scale is None:
        residual_scale = [1.0] * 4
    scale = torch.as_tensor(
        residual_scale, device=next(source.parameters()).device,
        dtype=torch.float32).reshape(1, 4, 1, 1)
    if tuple(scale.shape) != (1, 4, 1, 1) or not bool(
            torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError(f"invalid residual scale {residual_scale!r}")
    source.prediction_target = prediction_target
    # This is deliberately not a persistent buffer: the values live in the
    # architecture metadata, so legacy model state dictionaries stay strict-
    # load compatible.
    if "prediction_residual_scale" in source._buffers:
        source._buffers["prediction_residual_scale"] = scale
    else:
        source.register_buffer(
            "prediction_residual_scale", scale, persistent=False)
    return source


def configure_prediction_from_payload(model, payload):
    architecture = dict(payload.get("architecture", {}))
    target = architecture.get("prediction_target", ABSOLUTE_TARGET)
    scale = architecture.get("residual_scale", [1.0] * 4)
    return configure_prediction_target(model, target, scale)


def _prediction_spec(model):
    source = model.module if isinstance(model, DistributedDataParallel) else model
    target = getattr(source, "prediction_target", ABSOLUTE_TARGET)
    scale = getattr(source, "prediction_residual_scale", None)
    if scale is None:
        scale = next(source.parameters()).new_ones((1, 4, 1, 1))
    return target, scale


def _flow_target(model, history, target):
    prediction_target, scale = _prediction_spec(model)
    if prediction_target == ABSOLUTE_TARGET:
        return target
    if prediction_target == RESIDUAL_TARGET:
        return (target - history[:, -1]) / scale.to(target.dtype)
    raise ValueError(f"unknown prediction target {prediction_target!r}")


def _to_next_latent(model, history, generated):
    prediction_target, scale = _prediction_spec(model)
    if prediction_target == ABSOLUTE_TARGET:
        return generated
    if prediction_target == RESIDUAL_TARGET:
        return history[:, -1] + generated * scale.to(generated.dtype)
    raise ValueError(f"unknown prediction target {prediction_target!r}")


def _normalize_motion_inputs(past_actions, past_mask, proprio, goal,
                             motion_stats):
    """Match Action Expert's causal action/state representation."""
    if motion_stats is None:
        return past_actions, past_mask, proprio, goal
    mean = torch.as_tensor(motion_stats["mean"], device=proprio.device,
                           dtype=proprio.dtype)
    std = torch.as_tensor(motion_stats["std"], device=proprio.device,
                          dtype=proprio.dtype)
    proprio = (proprio - mean) / std
    goal_mean = torch.as_tensor(
        motion_stats["goal_mean"], device=goal.device, dtype=goal.dtype)
    goal_std = torch.as_tensor(
        motion_stats["goal_std"], device=goal.device, dtype=goal.dtype)
    goal = (goal - goal_mean) / goal_std
    if "action_logit_mean" in motion_stats:
        value = past_actions.clamp(1e-5, 1.0 - 1e-5)
        value = value.log() - torch.log1p(-value)
        action_mean = torch.as_tensor(
            motion_stats["action_logit_mean"], device=value.device,
            dtype=value.dtype)
        action_std = torch.as_tensor(
            motion_stats["action_logit_std"], device=value.device,
            dtype=value.dtype)
        past_actions = (value - action_mean) / action_std
        past_actions = past_actions * past_mask[..., None].to(past_actions.dtype)
    return past_actions, past_mask, proprio, goal


def flow_batch(model, history, target, *, past_actions=None, past_mask=None,
               proprio=None, goal=None, vae=None, vae_scale=None,
               target_image=None, occupancy_weight=0.0,
               occupancy_threshold=1.5, motion_stats=None):
    """Compute latent rectified-flow loss and optional decoded hit loss.

    The auxiliary target is the physical binary hit map from the dataset, not
    a VAE reconstruction.  The VAE remains frozen, but gradients are allowed
    through its decoder into ``predicted_next``.  Centering the decoder mask
    activation at the inference threshold makes zero the BCE decision
    boundary.  A small ``1-sigma`` weight emphasizes reliable x0 estimates
    without changing the rectified-flow objective at high noise levels.
    """
    clean_target = _flow_target(model, history, target)
    noise = torch.randn_like(clean_target)
    sigma = torch.rand(len(clean_target), device=target.device,
                       dtype=target.dtype)
    noisy = (1 - sigma[:, None, None, None]) * clean_target + (
        sigma[:, None, None, None] * noise)
    velocity_target = noise - clean_target
    if motion_stats is not None:
        if (past_actions is None or past_mask is None or proprio is None
                or goal is None):
            raise ValueError("motion-conditioned flow requires causal motion inputs")
        past_actions, past_mask, proprio, goal = _normalize_motion_inputs(
            past_actions, past_mask, proprio, goal, motion_stats)
    velocity = model(history, noisy, sigma, past_actions=past_actions,
                     past_mask=past_mask, proprio=proprio, goal=goal)
    flow_mse = (velocity - velocity_target).square().mean()
    predicted_clean = noisy - sigma[:, None, None, None] * velocity
    predicted_next = _to_next_latent(model, history, predicted_clean)
    # Keep this metric and auxiliary loss in actual scaled-VAE latent units so
    # absolute and residual experiments remain directly comparable.
    x0_l1 = (predicted_next - target).abs().mean()
    occupancy = predicted_next.new_zeros(())
    if float(occupancy_weight) > 0.0:
        if vae is None or vae_scale is None or target_image is None:
            raise ValueError(
                "decoded occupancy loss requires vae, vae_scale and "
                "target_image")
        decoded = vae.decode(predicted_next.float() / float(vae_scale)).sample
        mask_logits = decoded[:, 1, :, :18].float() - float(
            occupancy_threshold)
        target_hit = (target_image[:, 1, :, :18] > 0.0).to(mask_logits.dtype)
        per_pixel = F.binary_cross_entropy_with_logits(
            mask_logits, target_hit, reduction="none")
        per_sample = per_pixel.flatten(1).mean(1)
        reliability = (1.0 - sigma.float()).clamp_min(0.05)
        occupancy = (per_sample * reliability).sum() / reliability.sum()
    loss = flow_mse + 0.1 * x0_l1 + float(occupancy_weight) * occupancy
    return loss, flow_mse.detach(), x0_l1.detach(), occupancy.detach()


@torch.no_grad()
def _sample_next_once(model, history, steps: int, generator=None,
                      past_actions=None, past_mask=None, proprio=None, goal=None,
                      motion_stats=None, initial_value=None):
    if initial_value is None:
        value = torch.randn(
            len(history), 4, 27, 5, device=history.device,
            dtype=history.dtype, generator=generator)
    else:
        expected = (len(history), 4, 27, 5)
        if tuple(initial_value.shape) != expected:
            raise ValueError(
                f"Expected initial flow state {expected}, got "
                f"{tuple(initial_value.shape)}")
        value = initial_value.to(device=history.device, dtype=history.dtype)
    condition = proprio
    goal_condition = goal
    if motion_stats is not None:
        if (past_actions is None or past_mask is None or proprio is None
                or goal is None):
            raise ValueError("motion-conditioned sampling requires causal motion inputs")
        past_actions, past_mask, condition, goal_condition = (
            _normalize_motion_inputs(
                past_actions, past_mask, proprio, goal, motion_stats))
    schedule = torch.linspace(1, 0, steps + 1, device=history.device,
                              dtype=history.dtype)
    for current, following in zip(schedule[:-1], schedule[1:]):
        timestep = torch.full((len(history),), current, device=history.device,
                              dtype=history.dtype)
        velocity = model(history, value, timestep,
                         past_actions=past_actions, past_mask=past_mask,
                         proprio=condition, goal=goal_condition)
        value = value + (following - current) * velocity
    return _to_next_latent(model, history, value)


@torch.no_grad()
def sample_next(model, history, steps: int, generator=None, num_samples=1,
                past_actions=None, past_mask=None, proprio=None, goal=None,
                motion_stats=None, sampling_mode="independent"):
    """Sample the next latent with independent, zero, or antithetic noise.

    Independent endpoints are averaged in scaled-VAE latent space.  ``zero``
    integrates once from the Gaussian mean.  ``antithetic`` integrates paired
    +noise/-noise endpoints and averages them, cancelling most sampling noise
    with two trajectories while preserving nonlinear conditional motion.
    """
    sampling_mode = str(sampling_mode)
    if sampling_mode not in ("independent", "zero", "antithetic"):
        raise ValueError(f"unknown flow sampling mode {sampling_mode!r}")
    if sampling_mode == "zero":
        initial = history.new_zeros((len(history), 4, 27, 5))
        return _sample_next_once(
            model, history, steps, generator, past_actions, past_mask,
            proprio, goal, motion_stats, initial)
    if sampling_mode == "antithetic":
        initial = torch.randn(
            len(history), 4, 27, 5, device=history.device,
            dtype=history.dtype, generator=generator)
        positive = _sample_next_once(
            model, history, steps, generator, past_actions, past_mask,
            proprio, goal, motion_stats, initial)
        negative = _sample_next_once(
            model, history, steps, generator, past_actions, past_mask,
            proprio, goal, motion_stats, -initial)
        return 0.5 * (positive + negative)
    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    total = None
    for _ in range(num_samples):
        value = _sample_next_once(
            model, history, steps, generator, past_actions, past_mask,
            proprio, goal, motion_stats)
        total = value if total is None else total + value
    return total / num_samples


@torch.no_grad()
def validate(model, loader, vae, scale, device, precision, flow_steps,
             flow_ensemble_size=1, motion_stats=None,
             flow_sampling_mode="independent"):
    """Validate a fixed random subset, including decoded squared Chamfer.

    ``cd_paper_m2`` includes every frame using the project-wide finite empty
    cloud convention. ``nonempty_gt_cd_paper_m2`` excludes only frames whose
    *ground truth* has no return; false-empty predictions on non-empty scenes
    are still penalised, so this metric cannot be gamed by predicting nothing.
    """
    model.eval()
    # [flow, x0, sampled latent, all-sample CD, nonempty CD,
    #  valid-depth absolute error, valid-depth count, sample count,
    #  nonempty-GT sample count, decoded range-image absolute error,
    #  copy-current CD, copy-current depth absolute error]
    totals = torch.zeros(12, device=device, dtype=torch.float64)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if precision == "bf16" else contextlib.nullcontext())
    generator = torch.Generator(device=device).manual_seed(12345)
    for batch in loader:
        if motion_stats is None:
            history, target, target_image = batch
            past_actions = past_mask = proprio = goal = None
        else:
            history, target, past_actions, past_mask, proprio, goal, target_image = batch
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if motion_stats is not None:
            past_actions = past_actions.to(device, non_blocking=True)
            past_mask = past_mask.to(device, non_blocking=True)
            proprio = proprio.to(device, non_blocking=True)
            goal = goal.to(device, non_blocking=True)
        with amp:
            loss, flow_mse, x0_l1, _ = flow_batch(
                model, history, target, past_actions=past_actions,
                past_mask=past_mask, proprio=proprio, goal=goal,
                motion_stats=motion_stats)
            predicted = sample_next(
                model, history, flow_steps, generator,
                num_samples=flow_ensemble_size, past_actions=past_actions,
                past_mask=past_mask, proprio=proprio, goal=goal,
                motion_stats=motion_stats,
                sampling_mode=flow_sampling_mode)
        # Decode in fp32: VAE mask logits around the hit threshold are too
        # sensitive for an evaluation metric in bf16.
        decoded = vae.decode(torch.cat(
            (predicted.float(), history[:, -1].float()), dim=0) / scale
        ).sample.cpu().numpy()
        prediction_image, copy_image = np.split(decoded, 2, axis=0)
        target_image = target_image.numpy()
        cd_sum = nonempty_cd_sum = range_image_l1_sum = 0.0
        copy_cd_sum = copy_depth_error_sum = 0.0
        depth_error_sum = depth_count = 0.0
        nonempty_count = 0
        for prediction, copied, truth in zip(
                prediction_image, copy_image, target_image):
            # The physical LiDAR image occupies the first 18 elevation bins.
            # Decoder mask values are unbounded logits-like activations, so a
            # raw two-channel L1 is not meaningful.  Evaluate the range image
            # in physical/binary form instead: normalized range MAE on GT hits
            # plus binary hit-map error, with equal weight.  The result is a
            # stable, dimensionless image-domain generation score.
            pred_hit = prediction[1, :, :18] > 1.5
            gt_hit = truth[1, :, :18] > 0.0
            pred_range = np.clip(
                (prediction[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
            gt_range = np.clip(
                (truth[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
            hit_map_l1 = np.abs(
                pred_hit.astype(np.float32) - gt_hit.astype(np.float32)).mean()
            range_l1_normalized = (
                float(np.abs(pred_range - gt_range)[gt_hit].mean()) / 10.0
                if bool(gt_hit.any()) else 0.0)
            range_image_l1_sum += 0.5 * (
                float(hit_map_l1) + range_l1_normalized)
            row = frame_metrics(prediction, truth)
            copy_row = frame_metrics(copied, truth)
            cd_sum += float(row["cd_paper_m2"])
            copy_cd_sum += float(copy_row["cd_paper_m2"])
            depth_error_sum += float(row["valid_range_abs_error_sum_m"])
            copy_depth_error_sum += float(
                copy_row["valid_range_abs_error_sum_m"])
            depth_count += float(row["valid_range_count"])
            if not row["gt_empty"]:
                nonempty_cd_sum += float(row["cd_paper_m2"])
                nonempty_count += 1
        count = len(history)
        totals += torch.tensor(
            (float(flow_mse) * count, float(x0_l1) * count,
             float((predicted-target).abs().mean()) * count, cd_sum,
             nonempty_cd_sum, depth_error_sum, depth_count,
             float(count), float(nonempty_count), range_image_l1_sum,
             copy_cd_sum, copy_depth_error_sum),
            device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(totals)
    count = max(float(totals[7]), 1.0)
    depth_count = max(float(totals[6]), 1.0)
    nonempty_count = max(float(totals[8]), 1.0)
    model.train()
    latent_generation_l1 = float(totals[2] / count)
    generation_loss = float(totals[9] / count)
    depth_estimate_loss = float(totals[5] / depth_count)
    cd_paper_m2 = float(totals[3] / count)
    copy_cd_paper_m2 = float(totals[10] / count)
    return {"flow_mse": float(totals[0] / count),
            "x0_l1": float(totals[1] / count),
            "sample_l1": latent_generation_l1,
            "latent_generation_l1": latent_generation_l1,
            "range_image_l1": generation_loss,
            "generation_loss": generation_loss,
            "selection_score": generation_loss,
            "cd_paper_m2": cd_paper_m2,
            "nonempty_gt_cd_paper_m2": float(totals[4] / nonempty_count),
            "depth_estimate_loss": depth_estimate_loss,
            "copy_cd_paper_m2": copy_cd_paper_m2,
            "copy_depth_estimate_loss": float(totals[11] / depth_count),
            "cd_gain_vs_copy_m2": copy_cd_paper_m2 - cd_paper_m2,
            "cd_relative_improvement_vs_copy": (
                (copy_cd_paper_m2 - cd_paper_m2)
                / max(copy_cd_paper_m2, 1e-12)),
            "depth_valid_samples": int(totals[6]),
            "samples": int(count),
            "nonempty_gt_samples": int(totals[8])}


def _load_initial_checkpoint(model, checkpoint: Path,
                             allow_motion_missing=False):
    """Load model weights only; optimizer state is intentionally reset."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_target = payload.get("architecture", {}).get(
        "prediction_target", ABSOLUTE_TARGET)
    requested_target = getattr(model, "prediction_target", ABSOLUTE_TARGET)
    if checkpoint_target != requested_target:
        raise ValueError(
            "Initial Future Model checkpoint target is incompatible: "
            f"checkpoint={checkpoint_target!r}, requested={requested_target!r}. "
            "Residual training must start from a residual checkpoint or from "
            "scratch.")
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if allow_motion_missing:
        allowed = ("action_history_encoder.", "action_position",
                   "proprio_encoder.", "goal_encoder.",
                   "motion_projection.")
        missing = [key for key in missing
                   if not key.startswith(allowed)]
    if missing or unexpected:
        raise RuntimeError(
            "Initial Future Model checkpoint is incompatible: "
            f"missing={missing}, unexpected={unexpected}")
    return payload


def configure_motion_trainability(model, freeze_base: bool,
                                  unfreeze_last_blocks: int = 0):
    """Configure staged motion adaptation without drifting the full backbone.

    ``freeze_base`` first freezes the pretrained Future DiT and then enables
    the causal motion branch.  ``unfreeze_last_blocks`` additionally enables
    only the tail DiT blocks and output modulation/projection, allowing the
    frozen representation to adapt to motion without rewriting early spatial
    features.
    """
    if not freeze_base:
        return
    source = model.module if isinstance(model, DistributedDataParallel) else model
    unfreeze_last_blocks = int(unfreeze_last_blocks)
    if not 0 <= unfreeze_last_blocks <= len(source.blocks):
        raise ValueError(
            f"unfreeze_last_blocks must be in [0,{len(source.blocks)}], "
            f"got {unfreeze_last_blocks}")
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    for name, parameter in source.named_parameters():
        if name.startswith(("action_history_encoder.", "action_position",
                            "proprio_encoder.", "goal_encoder.",
                            "motion_projection.")):
            parameter.requires_grad_(True)
    if unfreeze_last_blocks:
        for block in source.blocks[-unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for module in (source.final_modulation, source.output_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def checkpoint_payload(model, optimizer, step, bests, args, dataset):
    source = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "format": FORMAT,
        "step": int(step), "best_metrics": {key: float(value)
                                                for key, value in bests.items()},
        "model": source.state_dict(), "optimizer": optimizer.state_dict(),
        "architecture": {"history_frames": 3, "prediction_frames": 1,
                         "latent_shape": [4, 27, 5], "width": args.width,
                         "depth": args.depth, "heads": args.heads,
                         "mlp_ratio": args.mlp_ratio,
                         "conditioning": (
                             "history_lidar_motion_conditioned"
                             if args.motion_conditioning else "history_lidar_only"),
                         "motion_conditioning": bool(args.motion_conditioning),
                         "past_horizon": int(args.past_horizon),
                         "proprio_dim": int(args.proprio_dim),
                         "goal_dim": int(args.goal_dim),
                         "freeze_base": bool(args.freeze_base),
                         "unfreeze_last_blocks": int(
                             args.unfreeze_last_blocks),
                         "motion_stats": getattr(args, "motion_stats", None),
                         "prediction_target": args.prediction_target,
                         "residual_scale": list(args.residual_scale),
                         "residual_stat_samples": int(
                             args.residual_stat_samples),
                         "decoded_occupancy_weight": float(
                             args.occupancy_weight),
                         "decoded_occupancy_threshold": float(
                             args.occupancy_threshold),
                         "recommended_sampling_mode": str(
                             args.flow_sampling_mode),
                         "recommended_flow_steps": int(args.flow_steps),
                         "recommended_ensemble_size": int(
                             args.flow_ensemble_size)},
        "vae": dataset.vae_metadata,
        "config": vars(args),
    }


def _scheduled_learning_rate(step, total_steps, peak_lr, min_lr,
                             warmup_steps, initial_lr=0.0):
    if warmup_steps > 0 and step <= warmup_steps:
        return initial_lr + (peak_lr - initial_lr) * step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def train(args):
    if args.restart_lr_schedule and not args.resume_optimizer:
        raise ValueError("--restart-lr-schedule requires --resume-optimizer")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    if rank == 0:
        for split in ("train", "val"):
            path = args.video_index_root / split / "metadata.json"
            if not path.exists():
                build_video_index(args.dataset_root, args.base_index_root,
                                  args.video_index_root, split)
    if dist.is_initialized():
        dist.barrier()
    dataset_type = (MotionHistoryLatentDataset
                    if args.motion_conditioning else HistoryLatentDataset)
    training = dataset_type(
        args.dataset_root, args.latent_root, args.video_index_root, "train",
        return_target_image=args.occupancy_weight > 0.0)
    validation_full = dataset_type(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=True)
    motion_stats = None
    if args.motion_conditioning:
        if rank == 0:
            motion_stats = estimate_motion_stats(
                training, args.motion_stat_samples, args.seed)
            motion_mean = torch.tensor(motion_stats["mean"], device=device,
                                       dtype=torch.float32)
            motion_std = torch.tensor(motion_stats["std"], device=device,
                                      dtype=torch.float32)
        else:
            motion_mean = torch.empty(args.proprio_dim, device=device,
                                      dtype=torch.float32)
            motion_std = torch.empty(args.proprio_dim, device=device,
                                     dtype=torch.float32)
        if dist.is_initialized():
            dist.broadcast(motion_mean, src=0)
            dist.broadcast(motion_std, src=0)
        if rank != 0:
            motion_stats = {"samples": int(args.motion_stat_samples),
                            "mean": motion_mean.cpu().tolist(),
                            "std": motion_std.cpu().tolist()}
        args.motion_stats = motion_stats
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(
        len(validation_full), min(args.eval_samples, len(validation_full)),
        replace=False))
    validation = Subset(validation_full, selected.tolist())
    sampler = DistributedSampler(
        training, world_size, rank, shuffle=True, seed=args.seed,
        drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(
        training, batch_size=args.micro_batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0)
    val_sampler = DistributedSampler(
        validation, world_size, rank, shuffle=False, drop_last=False
    ) if world_size > 1 else None
    val_loader = DataLoader(
        validation, batch_size=args.eval_batch_size, sampler=val_sampler,
        num_workers=max(0, args.workers // 2), pin_memory=True)
    model = LiDARVideoDiT(
        args.width, args.depth, args.heads, args.mlp_ratio,
        motion_conditioning=args.motion_conditioning,
        past_horizon=args.past_horizon,
        proprio_dim=args.proprio_dim,
        goal_dim=args.goal_dim).to(device)
    residual_stats = None
    if args.prediction_target == RESIDUAL_TARGET:
        if rank == 0:
            residual_stats = estimate_residual_scale(
                training, args.residual_stat_samples, args.seed)
            scale_tensor = torch.tensor(
                residual_stats["scale_rms"], device=device,
                dtype=torch.float32)
        else:
            scale_tensor = torch.empty(4, device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.broadcast(scale_tensor, src=0)
        args.residual_scale = scale_tensor.cpu().tolist()
    else:
        args.residual_scale = [1.0] * 4
    configure_prediction_target(
        model, args.prediction_target, args.residual_scale)
    initial_payload = None
    if args.init_checkpoint is not None:
        initial_payload = _load_initial_checkpoint(
            model, args.init_checkpoint,
            allow_motion_missing=args.motion_conditioning)
        if rank == 0:
            print(json.dumps({
                "init_checkpoint": str(args.init_checkpoint),
                "init_step": int(initial_payload.get("step", -1)),
            }), flush=True)
    configure_motion_trainability(
        model, args.freeze_base, args.unfreeze_last_blocks)
    if rank == 0 and residual_stats is not None:
        print(json.dumps({"residual_stats": residual_stats}), flush=True)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters after motion freeze setup")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95))
    start_step = 0
    if args.resume_optimizer:
        if initial_payload is None:
            raise ValueError("--resume-optimizer requires --init-checkpoint")
        if "optimizer" not in initial_payload:
            raise ValueError("resume checkpoint does not contain optimizer state")
        optimizer.load_state_dict(initial_payload["optimizer"])
        start_step = int(initial_payload.get("step", 0))
        if start_step >= int(args.steps):
            raise ValueError(
                f"resume step {start_step} must be smaller than --steps "
                f"{args.steps}")
        if rank == 0:
            print(json.dumps({"resume_optimizer": True,
                              "start_step": start_step,
                              "target_step": int(args.steps)}), flush=True)
    schedule_offset = start_step if args.restart_lr_schedule else 0
    schedule_initial_lr = (
        float(optimizer.param_groups[0]["lr"])
        if args.restart_lr_schedule else 0.0)
    if rank == 0 and args.restart_lr_schedule:
        print(json.dumps({"restart_lr_schedule": True,
                          "start_step": start_step,
                          "initial_lr": schedule_initial_lr,
                          "peak_lr": args.lr,
                          "warmup_steps": args.warmup_steps,
                          "target_step": args.steps}), flush=True)
    # VAE weights always remain frozen.  With decoded occupancy supervision,
    # its decoder is differentiable only with respect to the predicted latent;
    # otherwise it is used exclusively for validation.
    vae = stage1.load_circular_vae().to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    scale = float(training.vae_metadata["scaling_factor"])
    run_dir = args.out / args.run_name
    writer = None
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir / "tensorboard")
        config = vars(args).copy()
        config.update({"global_batch_size": args.micro_batch_size * world_size,
                       "train_windows": len(training),
                       "val_windows": len(validation_full),
                       "val_subset": len(validation),
                       "residual_stats": residual_stats,
                       "motion_stats": motion_stats,
                       "vae": training.vae_metadata})
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n")
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.precision == "bf16" else contextlib.nullcontext())
    iterator, epoch = iter(train_loader), 0
    bests = {"selection_score": float("inf"), "cd_paper_m2": float("inf"),
             "nonempty_gt_cd_paper_m2": float("inf")}
    if args.resume_optimizer and initial_payload is not None:
        for key, value in initial_payload.get("best_metrics", {}).items():
            if key in bests:
                bests[key] = float(value)
    no_improve_evals = 0
    started = time.time()
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        if args.motion_conditioning:
            if args.occupancy_weight > 0.0:
                history, target, past_actions, past_mask, proprio, goal, target_image = batch
                target_image = target_image.to(device, non_blocking=True)
            else:
                history, target, past_actions, past_mask, proprio, goal = batch
                target_image = None
            past_actions = past_actions.to(device, non_blocking=True)
            past_mask = past_mask.to(device, non_blocking=True)
            proprio = proprio.to(device, non_blocking=True)
            goal = goal.to(device, non_blocking=True)
        elif args.occupancy_weight > 0.0:
            history, target, target_image = batch
            target_image = target_image.to(device, non_blocking=True)
            past_actions = past_mask = proprio = goal = None
        else:
            history, target = batch
            target_image = None
            past_actions = past_mask = proprio = goal = None
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        lr = _scheduled_learning_rate(
            step - schedule_offset, args.steps - schedule_offset,
            args.lr, args.min_lr, args.warmup_steps, schedule_initial_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with amp:
            loss, flow_mse, x0_l1, occupancy_loss = flow_batch(
                model, history, target, past_actions=past_actions,
                past_mask=past_mask, proprio=proprio, goal=goal,
                motion_stats=motion_stats,
                vae=vae, vae_scale=scale,
                target_image=target_image,
                occupancy_weight=args.occupancy_weight,
                occupancy_threshold=args.occupancy_threshold)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if rank == 0 and step % args.log_every == 0:
            row = {"step": step, "loss": float(loss),
                   "flow_mse": float(flow_mse), "x0_l1": float(x0_l1),
                   "occupancy_loss": float(occupancy_loss),
                   "grad_norm": float(grad_norm),
                   "lr": float(lr),
                   "peak_allocated_gib": (
                       torch.cuda.max_memory_allocated(device) / 2**30),
                   "elapsed_min": (time.time()-started)/60}
            print(json.dumps(row), flush=True)
            for key, value in row.items():
                if key not in ("step",):
                    writer.add_scalar(f"train/{key}", value, step)
        # A validation is also a checkpoint boundary. Keep the two intervals
        # equal by default so all four checkpoint files are coherent.
        # With external validation, the training ranks only write atomic
        # checkpoints.  A separate evaluator can then consume those files on
        # another GPU without synchronizing or allocating validation batches
        # on the training devices.
        evaluate = (not args.external_validation
                    and step % args.eval_every == 0)
        save_latest = step % args.checkpoint_every == 0
        metrics = None
        stop_after_eval = False
        if evaluate:
            metrics = validate(model, val_loader, vae, scale, device,
                               args.precision, args.flow_steps,
                               args.flow_ensemble_size, motion_stats,
                               args.flow_sampling_mode)
            metrics["selection_score"] = metrics[args.selection_metric]
            if rank == 0:
                print(json.dumps({"step": step, "validation": metrics}), flush=True)
                for key, value in metrics.items():
                    if key != "samples":
                        writer.add_scalar(f"validation/{key}", value, step)
                checkpoint = checkpoint_payload(
                    model, optimizer, step, bests, args, training)
                updates = {
                    "selection_score": ("best.pt", "best_metrics.json"),
                    "cd_paper_m2": ("best_chamfer_m2.pt",
                                     "best_chamfer_m2_metrics.json"),
                    "nonempty_gt_cd_paper_m2": (
                        "best_nonempty_chamfer_m2.pt",
                        "best_nonempty_chamfer_m2_metrics.json"),
                }
                previous_selection = bests["selection_score"]
                for metric, (weight_name, json_name) in updates.items():
                    if metrics[metric] < bests[metric]:
                        bests[metric] = metrics[metric]
                        checkpoint = checkpoint_payload(
                            model, optimizer, step, bests, args, training)
                        _atomic_torch_save(checkpoint, run_dir / weight_name)
                        (run_dir / json_name).write_text(
                            json.dumps({"step": step, "optimized_metric": metric,
                                        "best_value": metrics[metric], **metrics},
                                       indent=2) + "\n")
                if metrics["selection_score"] < previous_selection - args.early_stop_min_delta:
                    no_improve_evals = 0
                else:
                    no_improve_evals += 1
                if (args.early_stop_patience > 0
                        and no_improve_evals >= args.early_stop_patience):
                    print(json.dumps({
                        "early_stop": True,
                        "step": step,
                        "patience_evals": no_improve_evals,
                        "best_selection_score": bests["selection_score"],
                    }), flush=True)
                    stop_after_eval = True
                else:
                    stop_after_eval = False
        if dist.is_initialized():
            stop_flag = torch.tensor(
                [int(stop_after_eval)], device=device, dtype=torch.int32)
            dist.broadcast(stop_flag, src=0)
            stop_after_eval = bool(stop_flag.item())
        if rank == 0 and (save_latest or evaluate):
            _atomic_torch_save(
                checkpoint_payload(model, optimizer, step, bests, args, training),
                run_dir / "latest.pt")
            writer.flush()
            if stop_after_eval:
                break
    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-index")
    training = sub.add_parser("train")
    for item in (prepare, training):
        item.add_argument("--dataset-root", type=Path, required=True)
        item.add_argument("--latent-root", type=Path, required=True)
        item.add_argument("--base-index-root", type=Path, required=True)
        item.add_argument("--video-index-root", type=Path, required=True)
    prepare.add_argument("--splits", nargs="+", default=("train", "val"))
    training.add_argument("--out", type=Path, default=Path("outputs"))
    training.add_argument("--run-name", default="lidar_video_dit_history3_next1")
    training.add_argument("--steps", type=int, default=50000)
    training.add_argument("--micro-batch-size", type=int, default=16)
    training.add_argument("--eval-batch-size", type=int, default=16)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--width", type=int, default=512)
    training.add_argument("--depth", type=int, default=8)
    training.add_argument("--heads", type=int, default=8)
    training.add_argument("--mlp-ratio", type=float, default=4.0)
    training.add_argument(
        "--motion-conditioning", action="store_true",
        help=("Condition Future DiT on causal 30-step action history and "
              "current proprioception. No successor action is used."))
    training.add_argument("--past-horizon", type=int, default=30)
    training.add_argument("--proprio-dim", type=int, default=10)
    training.add_argument("--goal-dim", type=int, default=4)
    training.add_argument(
        "--freeze-base", action="store_true",
        help=("With motion conditioning, freeze the pretrained LiDAR/DiT "
              "backbone and optimize only the new action/state/goal branch."))
    training.add_argument(
        "--unfreeze-last-blocks", type=int, default=0,
        help=("With --freeze-base, also fine-tune this many tail DiT blocks "
              "plus the final modulation/output head. Zero keeps the entire "
              "pretrained backbone frozen."))
    training.add_argument(
        "--motion-stat-samples", type=int, default=5000,
        help="Training windows used to normalize current proprioception.")
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--min-lr", type=float, default=2e-7)
    training.add_argument("--warmup-steps", type=int, default=200)
    training.add_argument("--init-checkpoint", type=Path, default=None)
    training.add_argument(
        "--resume-optimizer", action="store_true",
        help=("Restore optimizer state and continue step numbering from the "
              "initial checkpoint. --steps is then the final total step."))
    training.add_argument(
        "--restart-lr-schedule", action="store_true",
        help=("With --resume-optimizer, warm up from the saved optimizer LR "
              "to --lr over --warmup-steps, then decay across the remaining "
              "training steps. Global checkpoint step numbers are preserved."))
    training.add_argument(
        "--occupancy-weight", type=float, default=0.0,
        help=("Weight of the frozen-VAE decoded hit-mask BCE. Zero preserves "
              "the original latent-only objective."))
    training.add_argument(
        "--occupancy-threshold", type=float, default=1.5,
        help="Decoder mask activation used as the binary hit boundary.")
    training.add_argument(
        "--selection-metric",
        choices=("generation_loss", "cd_paper_m2",
                 "nonempty_gt_cd_paper_m2"),
        default="generation_loss",
        help="Validation metric used for best.pt and early stopping.")
    training.add_argument(
        "--prediction-target",
        choices=(ABSOLUTE_TARGET, RESIDUAL_TARGET),
        default=RESIDUAL_TARGET,
        help=("Generate the absolute next latent or the RMS-normalized "
              "residual from the current latent. Residual is recommended."))
    training.add_argument(
        "--residual-stat-samples", type=int, default=20000,
        help="Training windows used to estimate per-channel residual RMS.")
    training.add_argument("--early-stop-patience", type=int, default=15)
    training.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    training.add_argument("--weight-decay", type=float, default=1e-2)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    training.add_argument("--flow-steps", type=int, default=20)
    training.add_argument(
        "--flow-ensemble-size", type=int, default=1,
        help=("Independent next-latent samples averaged before VAE decoding. "
              "Use 4 for the higher-accuracy validation/inference setting."))
    training.add_argument(
        "--flow-sampling-mode",
        choices=("independent", "zero", "antithetic"),
        default="independent",
        help=("Initial-noise rule used for validation sampling. zero is "
              "deterministic; antithetic averages paired +noise and -noise "
              "endpoints using two trajectories."))
    training.add_argument("--eval-samples", type=int, default=1024)
    training.add_argument(
        "--external-validation", action="store_true",
        help=("Skip in-process validation. A separate validation worker reads "
              "latest.pt and evaluates on another GPU."))
    training.add_argument("--eval-every", type=int, default=200)
    training.add_argument("--checkpoint-every", type=int, default=200)
    training.add_argument("--log-every", type=int, default=20)
    training.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare-index":
        for split in args.splits:
            result = build_video_index(
                args.dataset_root, args.base_index_root,
                args.video_index_root, split)
            print(json.dumps(result, indent=2), flush=True)
    else:
        train(args)


if __name__ == "__main__":
    main()
