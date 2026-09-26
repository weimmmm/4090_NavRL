"""Joint history-only World DiT and Action DiT training.

This runner deliberately has one causal observation path:

    three cached VAE latents -> World-DiT tokenizer -> history tokens
                                      |                  |
                                      +--> World DiT      +--> Action DiT

The World DiT is initialized from a trained history-3/next-1 checkpoint.
The Action DiT is created from scratch and is trained at the same time.  No
future LiDAR, future state, simulator outcome, or privileged obstacle feature
is passed to the action branch.  The VAE is not part of the graph: the input
latents are the frozen-VAE cache produced by ``cache-latents``.

Example::

  python -m lidar_wam.runner.history_action_dit smoke \
      --dataset-root DATA --latent-root LATENTS --action-index-root ACTION_INDEX \
      --world-checkpoint outputs/lidar_video_dit_history3_next1/best.pt

  torchrun --nproc_per_node=4 -m lidar_wam.runner.history_action_dit train \
      --dataset-root DATA --latent-root LATENTS --action-index-root ACTION_INDEX \
      --world-checkpoint outputs/lidar_video_dit_history3_next1/best.pt \
      --out outputs --run-name history_world_action_dit
"""

from __future__ import annotations

import argparse
import contextlib
import bisect
import hashlib
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from lidar_wam.coordinates import numpy_goal_frame_causal_features
from lidar_wam.models.action_dit import (
    ActionDiT, JointHistoryWorldActionDiT, flow_sample_action)
from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner.lidar_video_dit import HistoryLatentDataset, INDEX_FORMAT


FORMAT = "navrl-history-world-action-dit-v2-masked-terminal"
ACTION_INDEX_FORMAT = "navrl-history3-action-next1-index-v2"


def _atomic_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(dataset, rows, dtype=np.float32):
    """Read HDF5 rows while respecting h5py's increasing-index requirement."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        shape = (0,) + tuple(dataset.shape[1:])
        return np.empty(shape, dtype=dtype)
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    values = np.asarray(dataset[sorted_rows], dtype=dtype)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return values[inverse]


class JointHistoryDataset(HistoryLatentDataset):
    """History latent windows plus strictly causal action conditions.

    The video index chain is ``[t-2,t-1,t,t+1]``.  The World DiT target is
    latent ``t+1``.  The Action DiT target is the PPO action chunk stored on
    the successor row ``t+1``; the only past chunk is the chunk stored on the
    current row ``t``.  This is the same ten-step execution protocol used by
    the evaluator and avoids the common off-by-one action leak.
    """

    def __init__(self, dataset_root: Path, latent_root: Path,
                 action_index_root: Path, split: str):
        # This index is a superset of the ordinary Video-DiT index: it retains
        # successful terminal chunks with 1--9 valid actions.  Reconstruct the
        # small amount of HistoryLatentDataset state here because that class
        # intentionally rejects index formats other than its world-only one.
        self.dataset_root = Path(dataset_root)
        self.latent_root = Path(latent_root)
        self.root = Path(action_index_root) / split
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("format") != ACTION_INDEX_FORMAT:
            raise ValueError("incompatible terminal-aware action index")
        self.entries = self.metadata["entries"]
        self.counts = np.asarray(self.metadata["counts"], dtype=np.int64)
        self.ends = np.cumsum(self.counts)
        self._rows = {}
        self._latents = {}
        self._files = {}
        self.return_target_image = False
        self._action_files = {}
        latent_metadata = json.loads(
            (self.latent_root / "metadata.json").read_text())
        self.vae_metadata = {
            key: latent_metadata.get(key) for key in (
                "scaling_factor", "vae_variant", "vae_step",
                "vae_checkpoint", "vae_sha256")}
        if not len(self):
            raise RuntimeError(f"No action windows in split {split!r}")

    def __getstate__(self):
        value = super().__getstate__()
        value["_action_files"] = {}
        return value

    def _action_frames(self, shard: int):
        if shard not in self._action_files:
            handle = h5py.File(self.dataset_root / self.entries[shard]["dataset"],
                               "r")
            self._action_files[shard] = handle["frames"] if "frames" in handle else handle
        return self._action_files[shard]

    def _location(self, index: int):
        shard = bisect.bisect_right(self.ends, int(index))
        start = 0 if shard == 0 else int(self.ends[shard - 1])
        rows, _ = self._open(shard)
        return shard, rows[int(index) - start]

    def _raw_conditions(self, indices):
        """Vectorized goal/proprio read used for stats and diagnostics."""
        indices = np.asarray(indices, dtype=np.int64)
        result_goal = np.empty((len(indices), 4), np.float32)
        result_proprio = np.empty((len(indices), 10), np.float32)
        shards = np.asarray([self._location(int(i))[0] for i in indices])
        for shard in np.unique(shards):
            positions = np.flatnonzero(shards == shard)
            rows = np.asarray([self._location(int(indices[p]))[1][2]
                               for p in positions], dtype=np.int64)
            frames = self._action_frames(int(shard))
            state = _read_rows(frames["drone_state"], rows)
            target = _read_rows(frames["target_position"], rows)
            direction = _read_rows(frames["target_dir_2d"], rows)
            goal, proprio = numpy_goal_frame_causal_features(state, target, direction)
            result_goal[positions] = goal
            result_proprio[positions] = proprio
        return result_goal, result_proprio

    def __getitem__(self, index):
        history, target = super().__getitem__(int(index))
        shard, chain = self._location(int(index))
        frames = self._action_frames(shard)
        current, successor = int(chain[2]), int(chain[3])
        action = np.asarray(frames["normalized_action_sequence"][successor],
                            dtype=np.float32)
        action_mask = np.asarray(frames["action_mask"][successor],
                                 dtype=np.float32)
        valid = action_mask.astype(bool)
        if not valid.any() or not bool(np.isfinite(action[valid]).all()):
            raise ValueError(f"invalid successor action at row {successor}")
        # Invalid suffix values were padding in the collector, not executed
        # zero velocity.  A neutral constant plus attention masking ensures
        # they neither receive loss nor become keys for valid action queries.
        action = np.where(valid[:, None] & np.isfinite(action), action, 0.5)
        # Three ten-step chunks (t-2, t-1, t) provide the requested 30-step
        # causal action history.  Each chunk carries its own validity mask.
        history_rows = np.asarray(chain[:3], dtype=np.int64)
        history_action = np.asarray(
            frames["normalized_action_sequence"][history_rows], dtype=np.float32)
        history_mask = np.asarray(frames["action_mask"][history_rows],
                                  dtype=np.float32)
        finite = np.isfinite(history_action).all(axis=-1)
        history_action = np.where(np.isfinite(history_action), history_action, 0.0)
        history_mask = history_mask * finite.astype(np.float32)
        history_action = history_action.reshape(30, 3)
        history_mask = history_mask.reshape(30)
        goal, proprio = self._raw_conditions([int(index)])
        return (history, target, torch.from_numpy(action),
                torch.from_numpy(goal[0]), torch.from_numpy(proprio[0]),
                torch.from_numpy(history_action), torch.from_numpy(history_mask),
                torch.from_numpy(action_mask),
                torch.tensor(bool(valid.all()), dtype=torch.bool))


def _condition_stats(dataset: JointHistoryDataset, samples: int, seed: int):
    count = min(int(samples), len(dataset))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), count, replace=False))
    goal, proprio = dataset._raw_conditions(indices)
    return {
        "goal_mean": goal.mean(0).tolist(),
        "goal_std": np.maximum(goal.std(0), 1e-4).tolist(),
        "proprio_mean": proprio.mean(0).tolist(),
        "proprio_std": np.maximum(proprio.std(0), 1e-4).tolist(),
        "samples": int(count),
    }


def _normalize(value, stats, name, device):
    mean = torch.as_tensor(stats[f"{name}_mean"], device=device,
                           dtype=value.dtype)
    std = torch.as_tensor(stats[f"{name}_std"], device=device,
                          dtype=value.dtype)
    return (value - mean) / std


def _load_world(checkpoint: Path, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = dict(payload.get("architecture", {}))
    model = LiDARVideoDiT(
        width=int(architecture.get("width", 512)),
        depth=int(architecture.get("depth", 8)),
        heads=int(architecture.get("heads", 8)),
        mlp_ratio=float(architecture.get("mlp_ratio", 4.0)),
    )
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload, architecture


def _flow_batch(model, batch, device, precision, stats, action_weight,
                world_weight):
    (history, target, action_target, goal, proprio, past, past_mask,
     action_mask, world_valid) = batch
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    action_target = action_target.to(device, non_blocking=True)
    goal = _normalize(goal.to(device, non_blocking=True), stats, "goal", device)
    proprio = _normalize(proprio.to(device, non_blocking=True), stats, "proprio", device)
    past = past.to(device, non_blocking=True)
    past_mask = past_mask.to(device, non_blocking=True)
    action_mask = action_mask.to(device, non_blocking=True)
    world_valid = world_valid.to(device, non_blocking=True)
    future_noise = torch.randn_like(target)
    future_sigma = torch.rand(len(target), device=device, dtype=target.dtype)
    noisy_future = ((1 - future_sigma[:, None, None, None]) * target
                    + future_sigma[:, None, None, None] * future_noise)
    action_noise = torch.randn_like(action_target)
    action_sigma = torch.rand(len(action_target), device=device,
                              dtype=action_target.dtype)
    noisy_action = ((1 - action_sigma[:, None, None]) * action_target
                    + action_sigma[:, None, None] * action_noise)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if precision == "bf16" else contextlib.nullcontext())
    with amp:
        world_velocity, action_velocity = model(
            history, noisy_future, future_sigma, noisy_action, action_sigma,
            goal, proprio, past, past_mask, action_mask=action_mask)
        world_target = future_noise - target
        action_flow_target = action_noise - action_target
        world_weight_mask = world_valid.to(world_velocity.dtype)
        world_denominator = world_weight_mask.sum().clamp_min(1.0)
        world_flow_per_sample = (world_velocity - world_target).square().flatten(1).mean(1)
        world_x0_per_sample = (
            noisy_future - future_sigma[:, None, None, None]
            * world_velocity - target).abs().flatten(1).mean(1)
        world_flow = (world_flow_per_sample * world_weight_mask).sum() / world_denominator
        world_x0 = (world_x0_per_sample * world_weight_mask).sum() / world_denominator
        action_weight_mask = action_mask[:, :, None].to(action_velocity.dtype)
        action_denominator = (
            action_weight_mask.sum() * action_velocity.shape[-1]).clamp_min(1.0)
        action_flow = ((action_velocity - action_flow_target).square()
                       * action_weight_mask).sum() / action_denominator
        action_x0 = ((noisy_action - action_sigma[:, None, None]
                      * action_velocity - action_target).abs()
                     * action_weight_mask).sum() / action_denominator
        loss = world_weight * (world_flow + 0.1 * world_x0) + action_weight * (
            action_flow + 0.1 * action_x0)
    metrics = {
        "loss": loss.detach(), "world_loss": world_flow.detach(),
        "action_loss": action_flow.detach(), "world_x0_l1": world_x0.detach(),
        "action_x0_l1": action_x0.detach(),
        "terminal_partial_fraction": (~world_valid).float().mean().detach(),
        "valid_action_steps": action_mask.sum(1).float().mean().detach(),
    }
    return loss, metrics


@torch.no_grad()
def _validate(model, loader, device, precision, stats, world_weight,
              action_weight, sample_steps=10, sample_windows=128):
    model.eval()
    sums = {key: 0.0 for key in
            ("loss", "world_loss", "action_loss", "world_x0_l1",
             "action_x0_l1", "terminal_partial_fraction",
             "valid_action_steps", "action_sample_l1")}
    count = 0
    sample_window_count = 0
    sample_value_count = 0
    generator = torch.Generator(device=device).manual_seed(271828)
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    for batch in loader:
        loss, metrics = _flow_batch(model, batch, device, precision, stats,
                                    action_weight, world_weight)
        n = len(batch[0])
        count += n
        for key, value in metrics.items():
            sums[key] += float(value) * n
        if sample_window_count < int(sample_windows):
            take = min(n, int(sample_windows) - sample_window_count)
            history = batch[0][:take].to(device, non_blocking=True)
            target_action = batch[2][:take].to(device, non_blocking=True)
            goal = _normalize(batch[3][:take].to(device, non_blocking=True),
                              stats, "goal", device)
            proprio = _normalize(batch[4][:take].to(device, non_blocking=True),
                                 stats, "proprio", device)
            past = batch[5][:take].to(device, non_blocking=True)
            past_mask = batch[6][:take].to(device, non_blocking=True)
            target_mask = batch[7][:take].to(device, non_blocking=True)
            features = raw.encode_history_features(history)
            amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if precision == "bf16" else contextlib.nullcontext())
            with amp:
                sampled = flow_sample_action(
                    raw.action, features, goal, proprio, past, past_mask,
                    steps=int(sample_steps), generator=generator)
                absolute = (sampled.float() - target_action.float()).abs()
                valid_values = target_mask.sum() * absolute.shape[-1]
                sample_l1 = (absolute * target_mask[:, :, None]).sum()
            sums["action_sample_l1"] += float(sample_l1)
            sample_value_count += int(valid_values)
            sample_window_count += take
    if dist.is_initialized():
        value = torch.tensor(
            [sums[k] for k in sums] +
            [count, sample_window_count, sample_value_count], device=device,
                             dtype=torch.float64)
        dist.all_reduce(value)
        for i, key in enumerate(sums):
            sums[key] = float(value[i])
        count = float(value[-3])
        sample_window_count = float(value[-2])
        sample_value_count = float(value[-1])
    count = max(float(count), 1.0)
    sample_value_count = max(float(sample_value_count), 1.0)
    model.train()
    output = {key: value / count for key, value in sums.items()
              if key != "action_sample_l1"}
    output["action_sample_l1"] = sums["action_sample_l1"] / sample_value_count
    output["action_sample_windows"] = int(sample_window_count)
    return output


def _payload(model, optimizer, step, args, stats, world_payload, metrics):
    source = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "format": FORMAT,
        "step": int(step),
        "model": source.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "architecture": {
            "history_frames": 3, "prediction_frames": 1,
            "latent_shape": [4, 27, 5], "width": args.width,
            "depth": args.depth, "heads": args.heads,
            "mlp_ratio": args.mlp_ratio, "action_horizon": 10,
            "action_dim": 3, "past_horizon": args.past_horizon,
            "shared_world_depth": args.shared_world_depth,
            "history_tokens": 3 * 27 * 5,
            "conditioning": "causal_history_goal_proprio_past_action",
            "world_backbone": "history_video_dit",
            "action_backbone": "action_dit_same_video_blocks",
        },
        "condition_stats": stats,
        "source_world_checkpoint": str(args.world_checkpoint),
        "source_world_sha256": _sha256(args.world_checkpoint),
        "source_world_step": int(world_payload.get("step", -1)),
        "resume_checkpoint": (str(args.resume_checkpoint)
                              if args.resume_checkpoint else None),
        "config": vars(args),
    }


def _build_loader(dataset, batch_size, workers, sampler=None, shuffle=False,
                  drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=shuffle if sampler is None else False,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, drop_last=drop_last)


def train(args):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Joint DiT training requires CUDA")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    train_data = JointHistoryDataset(
        args.dataset_root, args.latent_root, args.action_index_root, "train")
    val_full = JointHistoryDataset(
        args.dataset_root, args.latent_root, args.action_index_root, "val")
    rng = np.random.default_rng(args.seed)
    val_indices = np.sort(rng.choice(len(val_full),
                                     min(args.eval_samples, len(val_full)),
                                     replace=False))
    val_data = Subset(val_full, val_indices.tolist())
    stats = _condition_stats(train_data, args.condition_stat_samples, args.seed)
    train_sampler = (DistributedSampler(train_data, world_size, rank,
                                         shuffle=True, seed=args.seed,
                                         drop_last=True)
                     if world_size > 1 else None)
    val_sampler = (DistributedSampler(val_data, world_size, rank,
                                      shuffle=False, drop_last=False)
                   if world_size > 1 else None)
    train_loader = _build_loader(train_data, args.micro_batch_size, args.workers,
                                 train_sampler, shuffle=True, drop_last=True)
    val_loader = _build_loader(val_data, args.eval_batch_size,
                               max(0, args.workers // 2), val_sampler)
    world, world_payload, world_arch = _load_world(args.world_checkpoint, device)
    # New Action DiT and the pretrained World DiT are both trainable.  The
    # cached VAE never enters this graph, so it is frozen by construction.
    model = JointHistoryWorldActionDiT(
        world, width=args.width, depth=args.depth, heads=args.heads,
        mlp_ratio=args.mlp_ratio, past_horizon=args.past_horizon,
        shared_world_depth=args.shared_world_depth).to(device)
    resume_payload = None
    initial_step = 0
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload.get("model", resume_payload), strict=True)
        initial_step = int(resume_payload.get("step", 0))
        if "condition_stats" in resume_payload:
            # Preserve the exact deployment normalization learned by the
            # checkpoint.  Optimizer state is deliberately not restored for
            # this terminal-aware fine-tuning stage.
            stats = resume_payload["condition_stats"]
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        find_unused_parameters=False)
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    optimizer = torch.optim.AdamW([
        {"params": raw.world.parameters(), "lr": args.world_lr,
         "name": "world"},
        {"params": raw.action.parameters(), "lr": args.action_lr,
         "name": "action"},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.95))
    run_dir = args.out / args.run_name
    writer = None
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir / "tensorboard")
        config = vars(args).copy()
        config.update({"world_size": world_size,
                       "global_batch_size": args.micro_batch_size * world_size,
                       "train_windows": len(train_data),
                       "val_windows": len(val_full),
                       "val_subset": len(val_data), "condition_stats": stats,
                       "world_architecture": world_arch})
        (run_dir / "config.json").write_text(json.dumps(
            config, indent=2, default=str) + "\n")
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.precision == "bf16" else contextlib.nullcontext())
    iterator, epoch = iter(train_loader), 0
    # Validation data and masking semantics changed in this stage, so old
    # checkpoint metrics are not comparable.  Establish all three best files
    # from the first new validation instead of silently retaining stale bars.
    best_score = float("inf")
    best_world = float("inf")
    best_action = float("inf")
    started = time.time()
    model.train()
    for local_step in range(1, args.steps + 1):
        step = initial_step + local_step
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _flow_batch(model, batch, device, args.precision,
                                    stats, args.action_weight,
                                    args.world_weight)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if rank == 0 and step % args.log_every == 0:
            row = {"step": step, **{k: float(v) for k, v in metrics.items()},
                   "grad_norm": float(grad_norm),
                   "elapsed_min": (time.time() - started) / 60.0,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}
            print(json.dumps(row), flush=True)
            if writer:
                for key, value in row.items():
                    if key != "step" and math.isfinite(float(value)):
                        writer.add_scalar(f"train/{key}", value, step)
        if step % args.eval_every == 0:
            metrics_val = _validate(
                model, val_loader, device, args.precision, stats,
                args.world_weight, args.action_weight, args.action_sample_steps,
                args.action_sample_windows)
            if rank == 0:
                print(json.dumps({"step": step, "validation": metrics_val}),
                      flush=True)
                if writer:
                    for key, value in metrics_val.items():
                        writer.add_scalar(f"validation/{key}", value, step)
                # The action branch is deployed from pure-noise flow sampling;
                # rank checkpoints with that metric rather than only the
                # teacher-forced x0 estimate.
                score = (0.5 * metrics_val["world_x0_l1"]
                         + 0.5 * metrics_val["action_sample_l1"])
                checkpoint = _payload(model, optimizer, step, args, stats,
                                      world_payload, metrics_val)
                _atomic_save(checkpoint, run_dir / "latest.pt")
                if score < best_score:
                    best_score = score
                    _atomic_save(checkpoint, run_dir / "best.pt")
                if metrics_val["world_x0_l1"] < best_world:
                    best_world = metrics_val["world_x0_l1"]
                    _atomic_save(checkpoint, run_dir / "best_world.pt")
                if metrics_val["action_sample_l1"] < best_action:
                    best_action = metrics_val["action_sample_l1"]
                    _atomic_save(checkpoint, run_dir / "best_action.pt")
                (run_dir / "best_metrics.json").write_text(json.dumps({
                    "selection_score": best_score,
                    "best_world_x0_l1": best_world,
                    "best_action_sample_l1": best_action,
                    "last_validation": metrics_val,
                    "step": step}, indent=2) + "\n")
                if writer:
                    writer.add_scalar("validation/selection_score", score, step)
                    writer.flush()
            if dist.is_initialized():
                dist.barrier()
    if writer:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def smoke(args):
    args.steps = 1
    args.eval_every = 1
    args.eval_samples = min(args.eval_samples, 2)
    args.micro_batch_size = min(args.micro_batch_size, 2)
    args.eval_batch_size = min(args.eval_batch_size, 2)
    train(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (sub.add_parser("train"), sub.add_parser("smoke")):
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--latent-root", type=Path, required=True)
        command.add_argument("--action-index-root", type=Path, required=True)
        command.add_argument("--world-checkpoint", type=Path, required=True)
        command.add_argument("--resume-checkpoint", type=Path)
        command.add_argument("--out", type=Path, default=Path("outputs"))
        command.add_argument("--run-name", default="history_world_action_dit")
        command.add_argument("--steps", type=int, default=20000)
        command.add_argument("--micro-batch-size", type=int, default=8)
        command.add_argument("--eval-batch-size", type=int, default=8)
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--width", type=int, default=512)
        command.add_argument("--depth", type=int, default=8)
        command.add_argument("--heads", type=int, default=8)
        command.add_argument("--mlp-ratio", type=float, default=4.0)
        command.add_argument("--past-horizon", type=int, default=30)
        command.add_argument("--shared-world-depth", type=int, default=6)
        command.add_argument("--world-lr", type=float, default=1e-5)
        command.add_argument("--action-lr", type=float, default=1e-4)
        command.add_argument("--weight-decay", type=float, default=1e-2)
        command.add_argument("--grad-clip", type=float, default=1.0)
        command.add_argument("--world-weight", type=float, default=1.0)
        command.add_argument("--action-weight", type=float, default=1.0)
        command.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
        command.add_argument("--eval-samples", type=int, default=1024)
        command.add_argument("--condition-stat-samples", type=int, default=20000)
        command.add_argument("--action-sample-steps", type=int, default=10)
        command.add_argument("--action-sample-windows", type=int, default=128)
        command.add_argument("--eval-every", type=int, default=200)
        command.add_argument("--log-every", type=int, default=20)
        command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
