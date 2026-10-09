"""Jointly fine-tune Future-DiT and Action Expert with one predicted frame.

The action branch is strictly causal:

    observed LiDAR latents [t-2, t-1, t]
        -> Future-DiT rollout from pure noise -> predicted latent t+1
        -> FutureChangeAdapter -> Action-DiT future cross-attention
        -> next ten actions

Ground-truth latent ``t+1`` is used only by an auxiliary Future-DiT
flow-matching loss.  It is never converted to action-conditioning tokens.
Unlike ``action_expert_future.py``, this runner updates both pretrained
backbones as well as the new fusion layers.
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
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from lidar_wam.coordinates import semantics as coordinate_semantics
from lidar_wam.models.action_expert import FutureConditionedActionModel
from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner.action_expert_joint import (
    action_to_flow,
    flow_to_action,
    normalize_condition,
)
from lidar_wam.runner.history_action_dit import JointHistoryDataset


FORMAT = "navrl-single-future-action-joint-training-v1"


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


def _unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _close_dataset(dataset):
    """Close lazy HDF5 handles used by the historical dataset implementation."""
    closed = set()
    for name in ("_files", "_action_files"):
        for handle in getattr(dataset, name, {}).values():
            owner = getattr(handle, "file", handle)
            if id(owner) in closed:
                continue
            owner.close()
            closed.add(id(owner))
        getattr(dataset, name, {}).clear()


def _amp(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _load_model(action_path: Path, future_path: Path, device):
    action_payload = torch.load(
        action_path, map_location="cpu", weights_only=False)
    future_payload = torch.load(
        future_path, map_location="cpu", weights_only=False)
    action_arch = dict(action_payload.get("architecture", {}))
    future_arch = dict(future_payload.get("architecture", {}))
    if int(action_arch.get("action_horizon", 10)) != 10:
        raise ValueError("Joint runner requires a ten-action checkpoint")
    if int(future_arch.get("history_frames", 3)) != 3:
        raise ValueError("Joint runner requires a history-three Future-DiT")
    if int(future_arch.get("prediction_frames", 1)) != 1:
        raise ValueError("Joint runner requires a next-one-frame Future-DiT")

    future = LiDARVideoDiT(
        width=int(future_arch.get("width", 512)),
        depth=int(future_arch.get("depth", 8)),
        heads=int(future_arch.get("heads", 8)),
        mlp_ratio=float(future_arch.get("mlp_ratio", 4.0)))
    future.load_state_dict(
        future_payload.get("model", future_payload), strict=True)
    model = FutureConditionedActionModel(
        future,
        width=int(action_arch.get("width", 512)),
        depth=int(action_arch.get("depth", 8)),
        heads=int(action_arch.get("heads", 8)),
        ffn_width=int(action_arch.get("ffn_width", 2048)),
        past_horizon=int(action_arch.get("past_horizon", 30)),
        future_attention_layers=2,
        freeze_future_model=False)
    action_state = {
        key: value for key, value in action_payload["model"].items()
        if key.startswith(("observation.", "action_expert."))
    }
    incompatible = model.load_state_dict(action_state, strict=False)
    allowed_missing = {
        key for key in model.state_dict()
        if (key.startswith(("future_model.", "future_adapter."))
            or ".future_attn." in key or ".norm_future." in key
            or key.endswith(".future_gate"))
    }
    if (set(incompatible.missing_keys) != allowed_missing
            or incompatible.unexpected_keys):
        raise ValueError(
            "Action checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}")

    # True joint optimization: both pretrained backbones and fusion train.
    model.requires_grad_(True)
    # This auxiliary head is unrelated to the deployed Action output.
    model.action_expert.candidate_norm.requires_grad_(False)
    model.action_expert.candidate_out.requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    return model, action_payload, future_payload, action_arch, future_arch


def _prepare(batch, device, stats):
    (history, future_target, action, goal, proprio, past, past_mask,
     action_mask, future_valid) = batch
    history = history.to(device, non_blocking=True)
    future_target = future_target.to(device, non_blocking=True)
    action = action.to(device, non_blocking=True)
    goal = normalize_condition(
        goal.to(device, non_blocking=True), stats, "goal")
    proprio = normalize_condition(
        proprio.to(device, non_blocking=True), stats, "proprio")
    past_mask = past_mask.to(device, non_blocking=True)
    past = action_to_flow(
        past.to(device, non_blocking=True), stats) * past_mask[:, :, None]
    action_mask = action_mask.to(device, non_blocking=True)
    future_valid = future_valid.to(device, non_blocking=True)
    action_flow = action_to_flow(action, stats)
    return (history, history[:, -1], future_target, action, action_flow,
            goal, proprio, past, past_mask, action_mask, future_valid)


def _joint_loss(model, batch, device, stats, args):
    (history, current, future_target, action, action_target, goal, proprio,
     past, past_mask, action_mask, future_valid) = _prepare(
         batch, device, stats)

    action_noise = torch.randn_like(action_target)
    action_sigma = torch.rand(
        len(action_target), device=device, dtype=action_target.dtype)
    noisy_action = ((1-action_sigma[:, None, None])*action_target
                    + action_sigma[:, None, None]*action_noise)

    # This pure-noise rollout is the only Future information exposed to Action.
    rollout_noise = torch.randn_like(future_target)

    # An independent noised GT target supervises Future-DiT, but is not fed to
    # FutureChangeAdapter or Action-DiT.
    future_noise = torch.randn_like(future_target)
    future_sigma = torch.rand(
        len(future_target), device=device, dtype=future_target.dtype)
    noisy_future = ((1-future_sigma[:, None, None, None])*future_target
                    + future_sigma[:, None, None, None]*future_noise)

    with _amp(device, args.precision):
        action_velocity, future_velocity = model(
            noisy_action, action_sigma*1000.0, current, goal, proprio,
            past, past_mask, history, rollout_noise,
            args.future_steps, "correct",
            future_noisy_target=noisy_future,
            future_timestep=future_sigma,
            future_gradient_steps=args.future_gradient_steps)

        action_velocity_target = action_noise-action_target
        action_weight = action_mask[:, :, None].to(action_velocity.dtype)
        action_denominator = (
            action_weight.sum()*action_velocity.shape[-1]).clamp_min(1.0)
        action_flow = (((action_velocity-action_velocity_target).square()
                        * action_weight).sum()/action_denominator)
        action_x0 = noisy_action-action_sigma[:, None, None]*action_velocity
        action_x0_l1 = (((action_x0-action_target).abs()*action_weight).sum()
                        / action_denominator)
        pair_mask = (action_mask[:, 1:]*action_mask[:, :-1])[:, :, None]
        pair_denominator = (
            pair_mask.sum()*action_velocity.shape[-1]).clamp_min(1.0)
        action_delta_l1 = (((
            (action_x0[:, 1:]-action_x0[:, :-1])
            -(action_target[:, 1:]-action_target[:, :-1])).abs()
            * pair_mask).sum()/pair_denominator)

        future_velocity_target = future_noise-future_target
        future_per_sample = (
            future_velocity-future_velocity_target).square().flatten(1).mean(1)
        future_x0 = noisy_future-future_sigma[:, None, None, None]*future_velocity
        future_x0_per_sample = (
            future_x0-future_target).abs().flatten(1).mean(1)
        future_weight = future_valid.to(future_per_sample.dtype)
        future_denominator = future_weight.sum().clamp_min(1.0)
        future_flow = (future_per_sample*future_weight).sum()/future_denominator
        future_x0_l1 = (
            future_x0_per_sample*future_weight).sum()/future_denominator

        action_loss = action_flow + args.action_delta_weight*action_delta_l1
        future_loss = future_flow + args.future_x0_weight*future_x0_l1
        loss = action_loss + args.future_weight*future_loss

    return loss, {
        "loss": loss.detach(),
        "action_loss": action_loss.detach(),
        "action_flow_mse": action_flow.detach(),
        "action_x0_l1": action_x0_l1.detach(),
        "action_delta_l1": action_delta_l1.detach(),
        "future_loss": future_loss.detach(),
        "future_flow_mse": future_flow.detach(),
        "future_x0_l1": future_x0_l1.detach(),
        "valid_action_steps": action_mask.sum(1).float().mean().detach(),
    }


@torch.no_grad()
def _sample_actions(model, history, current, goal, proprio, past, past_mask,
                    stats, action_noise, future_noise, future_steps,
                    action_steps, future_tokens):
    observation = model.encode_current(current)
    value = action_noise.clone()
    schedule = torch.linspace(
        1, 0, int(action_steps)+1, device=value.device, dtype=value.dtype)
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
    metric_sums = {}
    metric_examples = 0
    modes = ("correct", "disabled", "zero", "shuffled")
    action_sums = {mode: 0.0 for mode in modes}
    action_values = 0
    future_l1_sum = 0.0
    future_values = 0
    generator = torch.Generator(device=device).manual_seed(271828)

    for batch in loader:
        _, metrics = _joint_loss(model, batch, device, stats, args)
        batch_size = len(batch[0])
        metric_examples += batch_size
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0)+float(value)*batch_size

        (history, current, future_target, action, _action_target, goal,
         proprio, past, past_mask, action_mask, future_valid) = _prepare(
             batch, device, stats)
        future_noise = torch.randn(
            future_target.shape, device=device, dtype=future_target.dtype,
            generator=generator)
        predicted_future = model.predict_next(
            history, future_noise, args.future_steps, gradient_steps=0)
        correct_tokens = model.future_adapter(current, predicted_future)
        tokens = {
            "correct": correct_tokens,
            "disabled": None,
            "zero": torch.zeros_like(correct_tokens),
            "shuffled": (correct_tokens.roll(1, dims=0)
                         if len(correct_tokens) > 1 else correct_tokens),
        }
        action_noise = torch.randn(
            action.shape, device=device, dtype=action.dtype,
            generator=generator)
        action_weight = action_mask[:, :, None]
        for mode in modes:
            prediction = _sample_actions(
                model, history, current, goal, proprio, past, past_mask,
                stats, action_noise, future_noise, args.future_steps,
                args.action_steps, tokens[mode])
            action_sums[mode] += float(
                ((prediction-action).abs()*action_weight).sum())
        action_values += int(action_weight.sum().item())*action.shape[-1]
        valid = future_valid[:, None, None, None]
        future_l1_sum += float(
            ((predicted_future-future_target).abs()*valid).sum())
        future_values += int(future_valid.sum().item())*int(
            np.prod(future_target.shape[1:]))

    result = {
        key: value/max(metric_examples, 1)
        for key, value in metric_sums.items()
    }
    for mode in modes:
        result[f"{mode}_action_mae"] = (
            action_sums[mode]/max(action_values, 1))
    result["future_rollout_l1"] = future_l1_sum/max(future_values, 1)
    result["gain_vs_disabled"] = (
        result["disabled_action_mae"]-result["correct_action_mae"])
    result["gain_vs_zero"] = (
        result["zero_action_mae"]-result["correct_action_mae"])
    result["gain_vs_shuffled"] = (
        result["shuffled_action_mae"]-result["correct_action_mae"])
    result["selection_score"] = result["correct_action_mae"]
    result["samples"] = metric_examples
    model.train()
    return result


def _optimizer(model, args):
    fusion = list(model.future_adapter.parameters())
    for block in model.action_expert.blocks:
        if block.future_attention:
            fusion.extend(block.norm_future.parameters())
            fusion.extend(block.future_attn.parameters())
            fusion.append(block.future_gate)
    future = list(model.future_model.parameters())
    excluded = {id(parameter) for parameter in fusion+future}
    action = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    groups = [
        {"params": action, "lr": args.action_lr, "name": "action"},
        {"params": future, "lr": args.future_lr, "name": "future"},
        {"params": fusion, "lr": args.fusion_lr, "name": "fusion"},
    ]
    return torch.optim.AdamW(
        groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))


def _checkpoint(model, optimizer, step, stats, args, metrics,
                action_payload, future_payload):
    model = _unwrap(model)
    semantics = dict(coordinate_semantics())
    semantics.update({"action_horizon": 10, "execution_horizon": 10})
    return {
        "format": FORMAT,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stats": stats,
        "validation": metrics,
        "semantics": semantics,
        "architecture": {
            "width": model.action_expert.width,
            "depth": len(model.action_expert.blocks),
            "heads": args.action_heads,
            "ffn_width": args.action_ffn_width,
            "action_horizon": 10,
            "past_horizon": model.action_expert.past_horizon,
            "future_attention_layers": 2,
            "history_frames": 3,
            "future_width": model.future_model.width,
            "future_depth": model.future_model.depth,
            "future_heads": model.future_model.heads,
            "future_mlp_ratio": args.future_mlp_ratio,
            "future_flow_steps": args.future_steps,
            "future_gradient_steps": args.future_gradient_steps,
            "future_prediction_frames": 1,
            "fusion": "one_way_single_future_to_action_blocks_7_8",
            "joint_optimization": True,
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
        raise RuntimeError("One-frame joint training requires CUDA")
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

    train_data = JointHistoryDataset(
        args.dataset_root, args.latent_root, args.index_root, "train")
    val_full = JointHistoryDataset(
        args.dataset_root, args.latent_root, args.index_root, "val")
    rng = np.random.default_rng(args.seed)
    val_indices = np.sort(rng.choice(
        len(val_full), min(args.eval_samples, len(val_full)), replace=False))
    val_data = Subset(val_full, val_indices.tolist())
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
    optimizer = _optimizer(raw, args)
    run_dir = args.out/args.run_name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({
        "format": FORMAT,
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "world_size": world_size,
        "global_batch_size": args.batch_size*world_size,
        "trainable_parameters": sum(
            parameter.numel() for parameter in raw.parameters()
            if parameter.requires_grad),
        "causal_action_conditioning": (
            "pure_noise_single_future_rollout; no GT future tokens"),
    })
    if rank == 0:
        (run_dir/"config.json").write_text(
            json.dumps(config, indent=2)+"\n")
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir/"tensorboard")
    else:
        writer = None

    first_step, best = 1, math.inf
    if args.resume_from is not None:
        payload = torch.load(
            args.resume_from, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT:
            raise ValueError("resume checkpoint is not a one-frame joint run")
        raw.load_state_dict(payload["model"], strict=True)
        if not args.reset_optimizer:
            optimizer.load_state_dict(payload["optimizer"])
        first_step = int(payload.get("step", 0))+1
        best = float(payload.get("validation", {}).get(
            "selection_score", math.inf))

    iterator = iter(train_loader)
    epoch = 0
    started = time.time()
    model.train()
    try:
        for step in range(first_step, args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            loss, metrics = _joint_loss(
                model, batch, device, stats, args)
            loss.backward()
            parameters = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, args.grad_clip)
            optimizer.step()

            if rank == 0 and step % args.log_every == 0:
                row = {key: float(value) for key, value in metrics.items()}
                row.update({
                    "step": step,
                    "grad_norm": float(grad_norm),
                    "elapsed_min": (time.time()-started)/60.0,
                    "peak_allocated_gib": (
                        torch.cuda.max_memory_allocated()/2**30),
                })
                print(json.dumps(row), flush=True)
                for key, value in row.items():
                    if key != "step" and math.isfinite(float(value)):
                        writer.add_scalar(f"train/{key}", value, step)

            if step % args.eval_every == 0:
                if distributed:
                    dist.barrier()
                if rank == 0:
                    validation = _validate(
                        raw, val_loader, device, stats, args)
                    print(json.dumps(
                        {"step": step, "validation": validation}),
                        flush=True)
                    for key, value in validation.items():
                        writer.add_scalar(f"validation/{key}", value, step)
                    payload = _checkpoint(
                        model, optimizer, step, stats, args, validation,
                        action_payload, future_payload)
                    _atomic_save(payload, run_dir/"latest.pt")
                    _atomic_save(payload, run_dir/f"checkpoint_step_{step:06d}.pt")
                    if (args.closed_loop_every > 0
                            and step >= args.closed_loop_start
                            and step % args.closed_loop_every == 0):
                        _atomic_save(
                            payload,
                            run_dir/"closed_loop_queue"/f"step_{step:06d}.pt")
                    if validation["selection_score"] < best:
                        best = validation["selection_score"]
                        _atomic_save(payload, run_dir/"best.pt")
                    gates = [
                        float(torch.tanh(block.future_gate).detach())
                        for block in raw.action_expert.blocks
                        if block.future_attention]
                    (run_dir/"best_metrics.json").write_text(json.dumps({
                        "best_selection_score": best,
                        "last_step": step,
                        "future_gates": gates,
                        "last_validation": validation,
                    }, indent=2)+"\n")
                    writer.flush()
                if distributed:
                    dist.barrier()
    finally:
        if writer is not None:
            writer.close()
        _close_dataset(train_data)
        _close_dataset(val_full)
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
    parser.add_argument(
        "--run-name", default="action_expert_future_joint_next1")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--action-lr", type=float, default=1e-5)
    parser.add_argument("--future-lr", type=float, default=1e-6)
    parser.add_argument("--fusion-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--action-delta-weight", type=float, default=0.20)
    parser.add_argument("--future-weight", type=float, default=0.25)
    parser.add_argument("--future-x0-weight", type=float, default=0.10)
    parser.add_argument("--future-steps", type=int, default=4)
    parser.add_argument(
        "--future-gradient-steps", type=int, default=1,
        help=("Final rollout steps retaining Action-loss gradients. Future "
              "flow loss always trains the complete Future-DiT."))
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--closed-loop-every", type=int, default=500)
    parser.add_argument("--closed-loop-start", type=int, default=2000)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.future_steps <= 0:
        parser.error("--future-steps must be positive")
    if not 0 <= args.future_gradient_steps <= args.future_steps:
        parser.error("--future-gradient-steps must be in [0, future-steps]")
    if args.eval_every <= 0:
        parser.error("--eval-every must be positive")
    train(args)


if __name__ == "__main__":
    main()
