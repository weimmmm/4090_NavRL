"""Add frozen autoregressive future-LiDAR conditioning to an Action Expert.

This runner is a controlled ablation, not another shared World/Action joint
model.  It loads the standalone Action Expert and a history-3 Future-DiT,
freezes both pretrained networks, and trains only:

* ``FutureChangeAdapter``;
* Future cross-attention/norm in the final Action blocks;
* the corresponding zero-initialized residual gates.

The action path receives three Future-DiT predictions sampled autoregressively
from noise.  Predicted ``t+1`` is fed back to predict ``t+2``, and both are fed
back to predict ``t+3``.  Ground-truth future LiDAR is never returned by the
dataset or passed to the model.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lidar_wam.coordinates import semantics as coordinate_semantics
from lidar_wam.data_v2 import V2WindowDataset
from lidar_wam.models.action_expert import FutureConditionedActionModel
from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner.action_expert_joint import (
    action_to_flow, flow_to_action, normalize_condition)


FORMAT = "navrl-future-conditioned-action-training-v1"


def _atomic_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FutureActionDataset(Dataset):
    """Action windows augmented with causal ``[t-2,t-1,t]`` latents."""

    def __init__(self, split: str, dataset_root: Path, latent_root: Path,
                 index_root: Path, limit: int | None = None, seed: int = 42,
                 past_horizon: int = 30):
        self.base = V2WindowDataset(
            split, dataset_root, latent_root, index_root,
            random_seed=seed, limit=limit, all_future_targets=False,
            past_horizon=past_horizon)
        if self.base.past_rows is None:
            raise ValueError(
                "Future conditioning requires a v2 index containing past_rows")
        if not np.array_equal(self.base.past_rows[:, -1], self.base.rows[:, 0]):
            raise RuntimeError("history/current row alignment mismatch")
        # Future-DiT was trained only on real consecutive history.  Do not turn
        # scene-boundary zero padding into a fake LiDAR observation.
        self.selection = np.flatnonzero((self.base.past_rows >= 0).all(axis=1))
        if not len(self.selection):
            raise RuntimeError(f"No complete three-frame histories in {split}")
        self.clearance = self.base.clearance[self.selection]
        self.turn_score = self.base.turn_score[self.selection]

    def __len__(self):
        return len(self.selection)

    def __getitem__(self, index):
        base_index = int(self.selection[int(index)])
        item = self.base[base_index]
        current, _, actions, _, _, _, goal, proprio, past, past_mask = item
        shard = int(self.base.shard[base_index])
        rows = np.asarray(self.base.past_rows[base_index], dtype=np.int64)
        _, latent = self.base._handles(shard)
        history = torch.from_numpy(
            np.asarray(latent[rows], dtype=np.float32).copy())
        # The index stores three future ten-action chunks.  Deployment executes
        # only the first chunk before replanning.
        return (history, current, actions[0], goal, proprio, past, past_mask)

    def close(self):
        self.base.close()


def _load_model(action_path: Path, future_path: Path, device):
    action_payload = torch.load(action_path, map_location="cpu", weights_only=False)
    future_payload = torch.load(future_path, map_location="cpu", weights_only=False)
    action_arch = dict(action_payload.get("architecture", {}))
    future_arch = dict(future_payload.get("architecture", {}))
    future = LiDARVideoDiT(
        width=int(future_arch.get("width", 512)),
        depth=int(future_arch.get("depth", 8)),
        heads=int(future_arch.get("heads", 8)),
        mlp_ratio=float(future_arch.get("mlp_ratio", 4.0)))
    future.load_state_dict(future_payload.get("model", future_payload), strict=True)
    model = FutureConditionedActionModel(
        future,
        width=int(action_arch.get("width", 512)),
        depth=int(action_arch.get("depth", 8)),
        heads=int(action_arch.get("heads", 8)),
        ffn_width=int(action_arch.get("ffn_width", 2048)),
        past_horizon=int(action_arch.get("past_horizon", 30)),
        future_attention_layers=2)
    action_state = {
        key: value for key, value in action_payload["model"].items()
        if key.startswith(("observation.", "action_expert."))}
    incompatible = model.load_state_dict(action_state, strict=False)
    allowed = {
        key for key in model.state_dict()
        if (key.startswith(("future_model.", "future_adapter."))
            or ".future_attn." in key or ".norm_future." in key
            or key.endswith(".future_gate"))}
    if set(incompatible.missing_keys) != allowed or incompatible.unexpected_keys:
        raise ValueError(
            "Action checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}")

    model.requires_grad_(False)
    model.future_adapter.requires_grad_(True)
    for block in model.action_expert.blocks:
        if block.future_attention:
            block.norm_future.requires_grad_(True)
            block.future_attn.requires_grad_(True)
            block.future_gate.requires_grad_(True)
    model.to(device=device, dtype=torch.float32)
    return model, action_payload, future_payload, action_arch, future_arch


def _conditions(batch, device, stats):
    history, current, action, goal, proprio, past, past_mask = batch
    history = history.to(device, non_blocking=True)
    current = current.to(device, non_blocking=True)
    action = action.to(device, non_blocking=True)
    goal = normalize_condition(
        goal.to(device, non_blocking=True), stats, "goal")
    proprio = normalize_condition(
        proprio.to(device, non_blocking=True), stats, "proprio")
    past_mask = past_mask.to(device, non_blocking=True)
    past = action_to_flow(
        past.to(device, non_blocking=True), stats) * past_mask.unsqueeze(-1)
    return history, current, action, goal, proprio, past, past_mask


def _flow_loss(model, batch, device, stats, future_steps, future_horizon,
               precision):
    history, current, action, goal, proprio, past, past_mask = _conditions(
        batch, device, stats)
    target = action_to_flow(action, stats)
    noise = torch.randn_like(target)
    sigma = torch.rand(len(target), device=device, dtype=target.dtype)
    noisy = ((1-sigma[:, None, None])*target
             + sigma[:, None, None]*noise)
    future_noise = (
        torch.randn_like(current) if int(future_horizon) == 1 else
        torch.randn(
            len(current), int(future_horizon), *current.shape[1:],
            device=current.device, dtype=current.dtype))
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if precision == "bf16" else contextlib.nullcontext())
    with amp:
        velocity, _ = model(
            noisy, sigma*1000.0, current, goal, proprio, past, past_mask,
            history, future_noise, future_steps, "correct")
        flow = F.mse_loss(velocity, noise-target)
        clean = noisy-sigma[:, None, None]*velocity
        endpoint = (clean-target).abs().mean()
        delta = F.l1_loss(
            clean[:, 1:]-clean[:, :-1], target[:, 1:]-target[:, :-1])
        loss = flow + 0.20*delta
    return loss, {
        "loss": loss.detach(), "action_flow_mse": flow.detach(),
        "action_x0_l1": endpoint.detach(),
        "action_delta_l1": delta.detach()}


@torch.no_grad()
def _sample_actions(model, history, current, goal, proprio, past, past_mask,
                    stats, action_noise, future_noise, future_steps,
                    action_steps, mode, future_tokens=None):
    observation = model.encode_current(current)
    if future_tokens is None and mode != "disabled":
        future_tokens = model.encode_predicted_future(
            history, future_noise, future_steps, mode)
    value = action_noise.clone()
    schedule = torch.linspace(
        1, 0, action_steps+1, device=value.device, dtype=value.dtype)
    for now, following in zip(schedule[:-1], schedule[1:]):
        timestep = torch.full(
            (len(value),), now*1000.0, device=value.device, dtype=value.dtype)
        velocity = model.action_expert(
            value, timestep, observation, goal, proprio, past, past_mask,
            future_tokens=future_tokens)
        value = value+(following-now)*velocity
    return flow_to_action(value, stats)


@torch.no_grad()
def _validate(model, loader, device, stats, args):
    model.eval()
    modes = ("correct", "disabled", "zero", "shuffled")
    sums = {mode: 0.0 for mode in modes}
    count = 0
    generator = torch.Generator(device=device).manual_seed(271828)
    for batch in loader:
        values = _conditions(batch, device, stats)
        history, current, target, goal, proprio, past, past_mask = values
        action_noise = torch.randn(
            target.shape, device=device, dtype=target.dtype,
            generator=generator)
        future_shape = (
            current.shape if int(args.future_horizon) == 1 else
            (len(current), int(args.future_horizon), *current.shape[1:]))
        future_noise = torch.randn(
            future_shape, device=device, dtype=current.dtype,
            generator=generator)
        correct_tokens = model.encode_predicted_future(
            history, future_noise, args.future_steps, "correct")
        tokens_by_mode = {
            "correct": correct_tokens,
            "disabled": None,
            "zero": torch.zeros_like(correct_tokens),
            "shuffled": (correct_tokens.roll(1, dims=0)
                         if len(correct_tokens) > 1 else correct_tokens),
        }
        for mode in modes:
            predicted = _sample_actions(
                model, history, current, goal, proprio, past, past_mask,
                stats, action_noise, future_noise, args.future_steps,
                args.action_steps, mode, tokens_by_mode[mode])
            sums[mode] += float((predicted-target).abs().mean())*len(target)
        count += len(target)
    count = max(count, 1)
    result = {f"{mode}_action_mae": sums[mode]/count for mode in modes}
    correct = result["correct_action_mae"]
    result["gain_vs_disabled"] = (
        result["disabled_action_mae"]-correct)
    result["gain_vs_zero"] = result["zero_action_mae"]-correct
    result["gain_vs_shuffled"] = result["shuffled_action_mae"]-correct
    result["selection_score"] = correct
    result["samples"] = count
    model.train()
    return result


def _checkpoint(model, optimizer, step, stats, args, metrics,
                action_payload, future_payload):
    model = _unwrap(model)
    return {
        "format": FORMAT, "step": int(step), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "stats": stats,
        "validation": metrics, "semantics": coordinate_semantics(),
        "architecture": {
            "width": model.action_expert.width,
            "depth": len(model.action_expert.blocks),
            "heads": args.action_heads, "ffn_width": args.action_ffn_width,
            "action_horizon": model.action_expert.horizon,
            "past_horizon": model.action_expert.past_horizon,
            "future_attention_layers": 2, "history_frames": 3,
            "future_width": model.future_model.width,
            "future_depth": model.future_model.depth,
            "future_heads": model.future_model.heads,
            "future_mlp_ratio": args.future_mlp_ratio,
            "future_flow_steps": args.future_steps,
            "future_prediction_frames": args.future_horizon,
            "fusion": "one_way_autoregressive_future_to_action_blocks_7_8",
        },
        "source_action_checkpoint": str(args.action_checkpoint),
        "source_action_step": int(action_payload.get("step", -1)),
        "source_action_sha256": _sha256(args.action_checkpoint),
        "source_future_checkpoint": str(args.future_checkpoint),
        "source_future_step": int(future_payload.get("step", -1)),
        "source_future_sha256": _sha256(args.future_checkpoint),
        "config": vars(args),
    }


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Future-conditioned Action training requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed+rank)
    np.random.seed(args.seed+rank)
    model, action_payload, future_payload, action_arch, future_arch = _load_model(
        args.action_checkpoint, args.future_checkpoint, device)
    args.action_heads = int(action_arch.get("heads", 8))
    args.action_ffn_width = int(action_arch.get("ffn_width", 2048))
    args.future_mlp_ratio = float(future_arch.get("mlp_ratio", 4.0))
    stats = action_payload["stats"]
    past_horizon = int(action_arch.get("past_horizon", 30))
    train_data = FutureActionDataset(
        "train", args.dataset_root, args.latent_root, args.index_root,
        seed=args.seed, past_horizon=past_horizon)
    val_data = FutureActionDataset(
        "val", args.dataset_root, args.latent_root, args.index_root,
        limit=args.eval_samples, seed=args.seed, past_horizon=past_horizon)
    train_sampler = (DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True,
        seed=args.seed, drop_last=True) if distributed else None)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0)
    val_loader = DataLoader(
        val_data, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=max(0, args.workers//2), pin_memory=True,
        persistent_workers=args.workers > 1)
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False)
    raw = _unwrap(model)
    adapter = list(raw.future_adapter.parameters())
    attention, gates = [], []
    for block in raw.action_expert.blocks:
        if block.future_attention:
            attention.extend(block.norm_future.parameters())
            attention.extend(block.future_attn.parameters())
            gates.append(block.future_gate)
    optimizer = torch.optim.AdamW([
        {"params": adapter, "lr": args.adapter_lr, "name": "adapter"},
        {"params": attention+gates, "lr": args.attention_lr,
         "name": "future_attention"},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.95))
    run_dir = args.out/args.run_name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir/"tensorboard")
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({"train_windows": len(train_data), "world_size": world_size,
                   "global_batch_size": args.batch_size*world_size,
                   "val_windows": len(val_data),
                   "trainable_parameters": sum(
                       p.numel() for p in raw.parameters() if p.requires_grad)})
    if rank == 0:
        (run_dir/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    first_step = 1
    best = math.inf
    if args.resume_from is not None:
        resume_payload = torch.load(
            args.resume_from, map_location="cpu", weights_only=False)
        resume_horizon = int(resume_payload.get(
            "architecture", {}).get("future_prediction_frames", 1))
        if resume_horizon != int(args.future_horizon):
            raise ValueError(
                "resume checkpoint future horizon does not match the current "
                f"run: checkpoint={resume_horizon}, requested={args.future_horizon}")
        raw.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        first_step = int(resume_payload.get("step", 0)) + 1
        best = float(resume_payload.get("validation", {}).get(
            "selection_score", math.inf))
        if rank == 0:
            print(json.dumps({
                "resume_from": str(args.resume_from),
                "first_step": first_step, "previous_best": best}),
                  flush=True)
    iterator = iter(train_loader)
    started = time.time()
    model.train()
    try:
        epoch = 0
        for step in range(first_step, args.steps+1):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _flow_loss(
                model, batch, device, stats, args.future_steps,
                args.future_horizon,
                args.precision)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip)
            optimizer.step()
            if rank == 0 and step % args.log_every == 0:
                row = {key: float(value) for key, value in metrics.items()}
                row.update({"step": step, "grad_norm": float(grad_norm),
                            "elapsed_min": (time.time()-started)/60.0,
                            "peak_allocated_gib": (
                                torch.cuda.max_memory_allocated()/2**30)})
                print(json.dumps(row), flush=True)
                for key, value in row.items():
                    if key != "step" and math.isfinite(float(value)):
                        writer.add_scalar(f"train/{key}", value, step)
            if step % args.eval_every == 0:
                if distributed:
                    dist.barrier()
                if rank == 0:
                    validation = _validate(raw, val_loader, device, stats, args)
                    print(json.dumps({"step": step, "validation": validation}),
                          flush=True)
                    for key, value in validation.items():
                        writer.add_scalar(f"validation/{key}", value, step)
                    payload = _checkpoint(
                        model, optimizer, step, stats, args, validation,
                        action_payload, future_payload)
                    _atomic_save(payload, run_dir/"latest.pt")
                    if (args.closed_loop_every > 0
                            and step >= args.closed_loop_start
                            and step % args.closed_loop_every == 0):
                        _atomic_save(
                            payload,
                            run_dir/"closed_loop_queue"/f"step_{step:06d}.pt")
                    if validation["selection_score"] < best:
                        best = validation["selection_score"]
                        _atomic_save(payload, run_dir/"best.pt")
                    gates_now = [float(torch.tanh(g).detach()) for g in gates]
                    (run_dir/"best_metrics.json").write_text(json.dumps({
                        "best_selection_score": best, "last_step": step,
                        "future_gates": gates_now,
                        "last_validation": validation}, indent=2)+"\n")
                    for index, value in enumerate(gates_now):
                        writer.add_scalar(f"fusion/gate_{index}", value, step)
                    writer.flush()
                if distributed:
                    dist.barrier()
    finally:
        if writer is not None:
            writer.close()
        train_data.close()
        val_data.close()
        if distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--future-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", default="action_expert_future_fusion")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--adapter-lr", type=float, default=3e-6)
    parser.add_argument("--attention-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--future-steps", type=int, default=4)
    parser.add_argument(
        "--future-horizon", type=int, default=3,
        help=("Number of autoregressively predicted future latent frames. "
              "The default predicts t+1, t+2, and t+3."))
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument(
        "--closed-loop-every", type=int, default=500,
        help="Write an immutable checkpoint for asynchronous Isaac evaluation.")
    parser.add_argument(
        "--closed-loop-start", type=int, default=0,
        help="Do not enqueue Isaac closed-loop checkpoints before this step.")
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Resume model and optimizer state from a future-fusion checkpoint.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.future_horizon <= 0:
        parser.error("--future-horizon must be positive")
    if args.closed_loop_start < 0:
        parser.error("--closed-loop-start must be nonnegative")
    train(args)


if __name__ == "__main__":
    main()
