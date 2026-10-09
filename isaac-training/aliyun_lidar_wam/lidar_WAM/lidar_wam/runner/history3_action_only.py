"""Train the causal Action Expert from three observed LiDAR VAE latents.

The policy inputs are strictly deployment-available::

    LiDAR latent [t-2, t-1, t] + goal(t) + proprio(t)
    + executed action chunks [t-2, t-1, t]
        -> ActionOnlyModel -> action chunk for t+1 (10 low-level actions)

There is no World/Future model and the target latent at t+1 is discarded.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, Subset

from lidar_wam.coordinates import semantics as coordinate_semantics
from lidar_wam.models.action_expert import ActionOnlyModel
from lidar_wam.runner.history_action_dit import (
    JointHistoryDataset,
    _atomic_save,
    _build_loader,
    _condition_stats,
    _read_rows,
)


FORMAT = "navrl-history3-action-only-v1"
FLOW_EPS = 1e-4


def _tensor_stat(stats, key, like):
    return torch.as_tensor(stats[key], device=like.device, dtype=like.dtype)


def _normalize(value, stats, name):
    return ((value - _tensor_stat(stats, f"{name}_mean", value)) /
            _tensor_stat(stats, f"{name}_std", value))


def _action_to_flow(action, stats):
    clipped = action.clamp(FLOW_EPS, 1.0-FLOW_EPS)
    logit = torch.log(clipped) - torch.log1p(-clipped)
    return ((logit - _tensor_stat(stats, "action_logit_mean", logit)) /
            _tensor_stat(stats, "action_logit_std", logit))


def _flow_to_action(value, stats):
    logit = (value * _tensor_stat(stats, "action_logit_std", value)
             + _tensor_stat(stats, "action_logit_mean", value))
    return logit.sigmoid()


def _training_stats(dataset: JointHistoryDataset, samples: int, seed: int):
    """Compute condition/action normalization from causal training rows only."""
    result = _condition_stats(dataset, samples, seed)
    count = min(int(samples), len(dataset))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), count, replace=False))
    shards = np.asarray([dataset._location(int(i))[0] for i in indices])
    valid_actions = []
    for shard in np.unique(shards):
        positions = np.flatnonzero(shards == shard)
        successor_rows = np.asarray([
            dataset._location(int(indices[p]))[1][3] for p in positions
        ], dtype=np.int64)
        frames = dataset._action_frames(int(shard))
        action = _read_rows(frames["normalized_action_sequence"], successor_rows)
        mask = _read_rows(frames["action_mask"], successor_rows).astype(bool)
        finite = np.isfinite(action).all(axis=-1)
        valid_actions.append(action[mask & finite])
    action = np.concatenate(valid_actions, axis=0).astype(np.float64)
    clipped = np.clip(action, FLOW_EPS, 1.0-FLOW_EPS)
    logit = np.log(clipped) - np.log1p(-clipped)
    result.update({
        "action_logit_mean": logit.mean(0).tolist(),
        "action_logit_std": np.maximum(logit.std(0), 1e-4).tolist(),
        "action_raw_mean": action.mean(0).tolist(),
        "action_stat_steps": int(len(action)),
        "logit_clip_epsilon": FLOW_EPS,
    })
    return result


def _amp(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _prepare(batch, device, stats):
    history, _unused_target, action, goal, proprio, past, past_mask, action_mask, _ = batch
    history = history.to(device, non_blocking=True)
    action = action.to(device, non_blocking=True)
    goal = _normalize(goal.to(device, non_blocking=True), stats, "goal")
    proprio = _normalize(proprio.to(device, non_blocking=True), stats, "proprio")
    past = past.to(device, non_blocking=True)
    past_mask = past_mask.to(device, non_blocking=True)
    action_mask = action_mask.to(device, non_blocking=True)
    target = _action_to_flow(action, stats)
    past = _action_to_flow(past, stats) * past_mask[:, :, None]
    return history, action, target, goal, proprio, past, past_mask, action_mask


def _flow_loss(model, batch, device, precision, stats, delta_weight):
    history, _action, target, goal, proprio, past, past_mask, action_mask = (
        _prepare(batch, device, stats))
    noise = torch.randn_like(target)
    sigma = torch.rand(len(target), device=device, dtype=target.dtype)
    noisy = (1-sigma[:, None, None])*target + sigma[:, None, None]*noise
    timestep = sigma * 1000.0
    with _amp(device, precision):
        velocity, _ = model(
            noisy, timestep, history, goal, proprio, past, past_mask)
        target_velocity = noise-target
        mask = action_mask[:, :, None].to(velocity.dtype)
        denominator = (mask.sum()*velocity.shape[-1]).clamp_min(1.0)
        flow = ((velocity-target_velocity).square()*mask).sum()/denominator
        x0 = noisy-sigma[:, None, None]*velocity
        endpoint = ((x0-target).abs()*mask).sum()/denominator
        pair_mask = (action_mask[:, 1:]*action_mask[:, :-1])[:, :, None]
        pair_denominator = (pair_mask.sum()*velocity.shape[-1]).clamp_min(1.0)
        delta_error = ((x0[:, 1:]-x0[:, :-1])
                       -(target[:, 1:]-target[:, :-1])).abs()
        delta = (delta_error*pair_mask).sum()/pair_denominator
        loss = flow + float(delta_weight)*delta
    return loss, {
        "loss": loss.detach(), "action_flow_mse": flow.detach(),
        "action_x0_l1": endpoint.detach(), "action_delta_l1": delta.detach(),
        "valid_action_steps": action_mask.sum(1).float().mean().detach(),
    }


@torch.no_grad()
def _sample(model, history, goal, proprio, past, past_mask, stats,
            steps, generator):
    raw = _unwrap(model)
    value = torch.randn(
        (len(history), 10, 3), device=history.device, dtype=history.dtype,
        generator=generator)
    observation = raw.encode_current(history)
    schedule = torch.linspace(
        1, 0, int(steps)+1, device=history.device, dtype=history.dtype)
    for current, following in zip(schedule[:-1], schedule[1:]):
        timestep = torch.full(
            (len(history),), float(current*1000), device=history.device,
            dtype=history.dtype)
        velocity = raw.predict_action_velocity(
            value, timestep, history, goal, proprio, past, past_mask,
            observation_tokens=observation)
        value = value + (following-current)*velocity
    return _flow_to_action(value, stats)


@torch.no_grad()
def _validate(model, loader, device, precision, stats, delta_weight,
              sample_steps, sample_windows):
    model.eval()
    totals = {key: 0.0 for key in (
        "loss", "action_flow_mse", "action_x0_l1", "action_delta_l1",
        "valid_action_steps")}
    examples = values = 0
    sample_sum = 0.0
    sampled = 0
    generator = torch.Generator(device=device).manual_seed(271828)
    for batch in loader:
        _, metrics = _flow_loss(
            model, batch, device, precision, stats, delta_weight)
        n = len(batch[0])
        examples += n
        for key, value in metrics.items():
            totals[key] += float(value)*n
        if sampled < int(sample_windows):
            take = min(n, int(sample_windows)-sampled)
            prepared = _prepare(tuple(v[:take] for v in batch), device, stats)
            history, action, _target, goal, proprio, past, past_mask, action_mask = prepared
            with _amp(device, precision):
                prediction = _sample(
                    model, history, goal, proprio, past, past_mask, stats,
                    sample_steps, generator)
            mask = action_mask[:, :, None]
            sample_sum += float(((prediction.float()-action.float()).abs()*mask).sum())
            values += int(mask.sum().item())*action.shape[-1]
            sampled += take
    if dist.is_initialized():
        packed = torch.tensor(
            [totals[k] for k in totals] + [examples, sample_sum, values, sampled],
            device=device, dtype=torch.float64)
        dist.all_reduce(packed)
        for index, key in enumerate(totals):
            totals[key] = float(packed[index])
        examples, sample_sum, values, sampled = map(float, packed[-4:])
    output = {key: value/max(examples, 1) for key, value in totals.items()}
    output["action_sample_l1"] = sample_sum/max(values, 1)
    output["action_sample_windows"] = int(sampled)
    model.train()
    return output


def _payload(model, optimizer, step, args, stats, metrics):
    return {
        "format": FORMAT, "step": int(step),
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "architecture": {
            "width": args.width, "depth": args.depth, "heads": args.heads,
            "ffn_width": args.ffn_width, "observation_history_frames": 3,
            "history_tokens": 405, "past_horizon": 30,
            "action_horizon": 10, "action_dim": 3,
            "world_model": None, "future_cross_attention": False,
            "conditioning": "lidar_t-2:t_goal_proprio_executed_action_t-2:t",
        },
        "stats": stats, "semantics": coordinate_semantics(),
        "config": vars(args),
    }


def _deployment_payload(checkpoint):
    """Strip optimizer state for asynchronous Isaac-Sim transfer."""
    return {key: value for key, value in checkpoint.items()
            if key != "optimizer"}


def train(args):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("history3 ActionOnly training requires CUDA")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed+rank)
    train_data = JointHistoryDataset(
        args.dataset_root, args.latent_root, args.action_index_root, "train")
    val_full = JointHistoryDataset(
        args.dataset_root, args.latent_root, args.action_index_root, "val")
    rng = np.random.default_rng(args.seed)
    val_indices = np.sort(rng.choice(
        len(val_full), min(args.eval_samples, len(val_full)), replace=False))
    val_data = Subset(val_full, val_indices.tolist())
    resume_payload = None
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False)
        if resume_payload.get("format") != FORMAT:
            raise ValueError(
                f"resume checkpoint format is {resume_payload.get('format')!r}, "
                f"expected {FORMAT!r}")
    stats_holder = [(
        resume_payload["stats"] if resume_payload is not None else
        _training_stats(train_data, args.condition_stat_samples, args.seed)
    ) if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(stats_holder, src=0)
    stats = stats_holder[0]
    train_sampler = (DistributedSampler(
        train_data, world_size, rank, shuffle=True, seed=args.seed, drop_last=True)
        if world_size > 1 else None)
    val_sampler = (DistributedSampler(
        val_data, world_size, rank, shuffle=False, drop_last=False)
        if world_size > 1 else None)
    train_loader = _build_loader(
        train_data, args.micro_batch_size, args.workers, train_sampler,
        shuffle=True, drop_last=True)
    val_loader = _build_loader(
        val_data, args.eval_batch_size, max(0, args.workers//2), val_sampler)
    model = ActionOnlyModel(
        width=args.width, depth=args.depth, heads=args.heads,
        ffn_width=args.ffn_width, past_horizon=30,
        observation_history_frames=3).to(device)
    initial_step = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"], strict=True)
        initial_step = int(resume_payload.get("step", 0))
        if initial_step >= args.steps:
            raise ValueError(
                f"resume step {initial_step} is not below final step {args.steps}")
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    if resume_payload is not None and not args.reset_optimizer:
        optimizer.load_state_dict(resume_payload["optimizer"])
    run_dir = args.out/args.run_name
    writer = None
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir/"tensorboard")
        config = vars(args).copy()
        config.update({
            "world_size": world_size,
            "global_batch_size": args.micro_batch_size*world_size,
            "train_windows": len(train_data), "val_windows": len(val_full),
            "val_subset": len(val_data), "stats": stats,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad),
        })
        (run_dir/"config.json").write_text(
            json.dumps(config, indent=2, default=str)+"\n")
    iterator = iter(train_loader)
    epoch = 0
    best = (float(resume_payload.get("metrics", {}).get(
        "action_sample_l1", float("inf")))
        if resume_payload is not None else float("inf"))
    started = time.time()
    model.train()
    for step in range(initial_step+1, args.steps+1):
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
            model, batch, device, args.precision, stats, args.delta_weight)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if rank == 0 and step % args.log_every == 0:
            row = {
                "step": step, **{key: float(value) for key, value in metrics.items()},
                "grad_norm": float(grad_norm),
                "elapsed_min": (time.time()-started)/60,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/2**30,
            }
            print(json.dumps(row), flush=True)
            if writer:
                for key, value in row.items():
                    if key != "step" and math.isfinite(float(value)):
                        writer.add_scalar(f"train/{key}", value, step)
        if step % args.eval_every == 0:
            validation = _validate(
                model, val_loader, device, args.precision, stats,
                args.delta_weight, args.sample_steps, args.sample_windows)
            if rank == 0:
                print(json.dumps({"step": step, "validation": validation}), flush=True)
                checkpoint = _payload(
                    model, optimizer, step, args, stats, validation)
                _atomic_save(checkpoint, run_dir/f"checkpoint_step_{step:06d}.pt")
                _atomic_save(checkpoint, run_dir/"latest.pt")
                if (step >= args.closed_loop_start
                        and step % args.closed_loop_every == 0):
                    _atomic_save(
                        _deployment_payload(checkpoint),
                        run_dir/"closed_loop_queue"/f"step_{step:06d}.pt")
                if validation["action_sample_l1"] < best:
                    best = validation["action_sample_l1"]
                    _atomic_save(checkpoint, run_dir/"best.pt")
                (run_dir/"best_metrics.json").write_text(json.dumps({
                    "best_action_sample_l1": best,
                    "last_validation": validation, "step": step,
                }, indent=2)+"\n")
                if writer:
                    for key, value in validation.items():
                        writer.add_scalar(f"validation/{key}", value, step)
                    writer.flush()
            if dist.is_initialized():
                dist.barrier()
    if writer:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command_name in ("train", "smoke"):
        command = sub.add_parser(command_name)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--latent-root", type=Path, required=True)
        command.add_argument("--action-index-root", type=Path, required=True)
        command.add_argument("--out", type=Path, default=Path("outputs"))
        command.add_argument("--run-name", default="history3_action_only")
        command.add_argument("--resume-checkpoint", type=Path)
        command.add_argument("--reset-optimizer", action="store_true")
        command.add_argument("--steps", type=int, default=10000)
        command.add_argument("--micro-batch-size", type=int, default=128)
        command.add_argument("--eval-batch-size", type=int, default=128)
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--width", type=int, default=512)
        command.add_argument("--depth", type=int, default=8)
        command.add_argument("--heads", type=int, default=8)
        command.add_argument("--ffn-width", type=int, default=2048)
        command.add_argument("--lr", type=float, default=1e-4)
        command.add_argument("--weight-decay", type=float, default=1e-2)
        command.add_argument("--delta-weight", type=float, default=0.20)
        command.add_argument("--grad-clip", type=float, default=1.0)
        command.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
        command.add_argument("--eval-samples", type=int, default=1024)
        command.add_argument("--condition-stat-samples", type=int, default=20000)
        command.add_argument("--sample-steps", type=int, default=10)
        command.add_argument("--sample-windows", type=int, default=128)
        command.add_argument("--eval-every", type=int, default=200)
        command.add_argument("--closed-loop-start", type=int, default=2000)
        command.add_argument("--closed-loop-every", type=int, default=400)
        command.add_argument("--log-every", type=int, default=20)
        command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "smoke":
        args.steps = 1
        args.eval_every = 1
        args.eval_samples = min(args.eval_samples, 4)
        args.condition_stat_samples = min(args.condition_stat_samples, 32)
        args.micro_batch_size = min(args.micro_batch_size, 2)
        args.eval_batch_size = min(args.eval_batch_size, 2)
        args.sample_windows = min(args.sample_windows, 2)
    train(args)


if __name__ == "__main__":
    main()
