"""Offline imitation learning for the v2 final-Lee-velocity dataset.

This is an Action-DiT-only runner: the frozen circular VAE encodes the current
LiDAR frame, the time-conditioned transformer learns a rectified-flow action
objective, and a deterministic transformer head learns the same clean
ten-step command with supervised behavior cloning.  No World/Future model or
future LiDAR enters either action path.  Deployment uses the deterministic
head in one forward pass, then rotates its ego-frame velocity into world ENU
before passing it to the preserved Lee controller.

Use ``--policy-type flow`` (or the ``flow_action_expert`` runner) for a
standalone flow policy with no Direct branch. It uses only flow MSE during
training, and ten Euler steps from Gaussian noise for validation/deployment.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
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

from lidar_wam.models.action_expert import SuperActionExpertModel
from lidar_wam.models.flow_action_expert import (
    FLOW_ACTION_EXPERT_FORMAT, FlowActionExpertModel,
)
from lidar_wam.runner import stage1
from lidar_wam.runner.super_lee_action_data import (
    ACTION_HORIZON,
    LIDAR_HISTORY_FRAMES,
    PAST_HORIZON,
    PROPRIO_DIM,
    SuperLeeActionDataset,
    discover_shard_names,
    fixed_eval_indices,
)


FORMAT = "navrl-action-dit-final-lee-ego-bc-history1-v3"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAE_CHECKPOINT = (
    PROJECT_ROOT / "outputs/vae_5m_finetune_v1/best/"
    "diffusion_pytorch_model.safetensors")


def _unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


class SkipInitialBatches:
    """Skip sampler indices once, without loading skipped HDF5/VAE samples.

    Length stays the full epoch length for global-step cursor arithmetic.
    After the first iterator, subsequent epochs use the complete sampler.
    """

    def __init__(self, batch_sampler, skip):
        self.batch_sampler, self.skip = batch_sampler, int(skip)

    def __iter__(self):
        skip, self.skip = self.skip, 0
        yield from itertools.islice(iter(self.batch_sampler), skip, None)

    def __len__(self):
        return len(self.batch_sampler)


def _restore_resume_state(payload, model, optimizer, args):
    """Restore parameters AND Adam moments; reject a different action contract."""
    flow_only = isinstance(_unwrap(model), FlowActionExpertModel)
    expected_format = FLOW_ACTION_EXPERT_FORMAT if flow_only else FORMAT
    if payload.get("format") != expected_format:
        raise ValueError("resume checkpoint has a different policy format")
    expected = {"width": args.width, "depth": args.depth, "heads": args.heads,
                "ffn_width": args.ffn_width, "action_horizon": ACTION_HORIZON,
                "proprio_dim": PROPRIO_DIM, "history_lidar_frames": LIDAR_HISTORY_FRAMES}
    if any(payload.get("architecture", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("resume architecture differs from this model")
    interface = payload.get("action_interface", {})
    if interface.get("model_output_scale_mps_xyz") != args.action_scale_mps_xyz:
        raise ValueError("resume action scale differs")
    if flow_only and (interface.get("normalized_output_constraint") != "unit_box_after_euler"
                      or interface.get("physical_axis_limit_mps_xyz") != [2., 2., 2.]):
        raise ValueError("resume Flow-only per-axis velocity contract differs")
    if "optimizer" not in payload:
        raise ValueError("resume requires the full optimizer state, not inference-only weights")
    _unwrap(model).load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])


def _checkpoint_rng_states(device):
    state = {"torch_cpu": torch.get_rng_state(), "numpy": np.random.get_state(),
             "torch_cuda": torch.cuda.get_rng_state(device)}
    if dist.is_initialized():
        states = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(state, states, dst=0)
        return states
    return [state]


def _build_model(args):
    kwargs = dict(width=args.width, depth=args.depth, heads=args.heads,
                  ffn_width=args.ffn_width, horizon=ACTION_HORIZON,
                  proprio_dim=PROPRIO_DIM,
                  lidar_history_frames=LIDAR_HISTORY_FRAMES)
    if getattr(args, "policy_type", "direct") == "flow":
        return FlowActionExpertModel(**kwargs, sample_steps=args.sample_steps)
    return SuperActionExpertModel(**kwargs, past_horizon=PAST_HORIZON)


def _autocast(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


def _atomic_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_vae(checkpoint: Path, device: torch.device):
    """Load the explicitly selected 5 m VAE and its matching latent scale."""
    from safetensors.torch import load_file

    checkpoint = checkpoint.expanduser().resolve()
    config_path = checkpoint.parent / "config.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LiDAR VAE weights missing: {checkpoint}")
    if not config_path.is_file():
        raise FileNotFoundError(f"LiDAR VAE config missing: {config_path}")
    config = json.loads(config_path.read_text())
    expected = {"in_channels": 2, "out_channels": 2,
                "latent_channels": 4}
    mismatch = {
        key: (config.get(key), value) for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatch:
        raise ValueError(f"incompatible LiDAR VAE config: {mismatch}")
    scale = float(config["scaling_factor"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid VAE scaling_factor {scale}")
    model = stage1.make_vae(circular=True)
    model.load_state_dict(
        load_file(str(checkpoint), device="cpu"), strict=True)
    model.register_to_config(scaling_factor=scale)
    model = model.to(device).float().eval()
    model.requires_grad_(False)
    return model, scale


def _encode_current(vae, image, scale, device, precision):
    image = image.to(device, non_blocking=True).float()
    if image.ndim == 5:
        batch, frames = image.shape[:2]
        image = image.reshape(batch * frames, *image.shape[2:])
        with torch.no_grad(), _autocast(device, precision):
            latent = vae.encode(image).latent_dist.mode() * float(scale)
        return latent.reshape(batch, frames, *latent.shape[1:])
    if image.ndim != 4:
        raise ValueError(
            f"Expected LiDAR image [B,2,108,20], got {tuple(image.shape)}")
    with torch.no_grad(), _autocast(device, precision):
        return vae.encode(image).latent_dist.mode() * float(scale)


def _condition_stats(dataset, samples, seed):
    count = min(int(samples), len(dataset))
    indices = fixed_eval_indices(len(dataset), count, seed)
    goal, proprio = dataset.conditions_many(indices)
    return {
        "goal_mean": goal.mean(0).tolist(),
        "goal_std": np.maximum(goal.std(0), 1.0e-4).tolist(),
        "proprio_mean": proprio.mean(0).tolist(),
        "proprio_std": np.maximum(proprio.std(0), 1.0e-4).tolist(),
        "past_velocity_scale_mps": 2.0,
        "samples": int(count),
    }


def _normalise(value, stats, name, device):
    mean = torch.as_tensor(
        stats[f"{name}_mean"], device=device, dtype=value.dtype)
    std = torch.as_tensor(
        stats[f"{name}_std"], device=device, dtype=value.dtype)
    return (value - mean) / std


def _prepare(batch, vae, scale, action_scale, stats, device, precision):
    if len(batch) not in (5, 7):
        raise ValueError("expected current-only or legacy history action batch")
    image, target_physical, goal, proprio = batch[:4]
    latent = _encode_current(vae, image, scale, device, precision)
    target_physical = target_physical.to(device, non_blocking=True)
    target = target_physical / action_scale
    goal = _normalise(
        goal.to(device, non_blocking=True), stats, "goal", device)
    proprio = _normalise(
        proprio.to(device, non_blocking=True),
        stats, "proprio", device)
    prepared = (latent, target, target_physical, goal, proprio)
    if len(batch) == 5:
        return prepared
    past, past_mask = batch[4:6]
    past = past.to(device, non_blocking=True) / float(
        stats["past_velocity_scale_mps"])
    past_mask = past_mask.to(device, non_blocking=True)
    return prepared + (past, past_mask)


def _loss_batch(model, vae, batch, scale, action_scale, stats, device,
                precision, flow_weight, direct_weight, delta_weight,
                hard_weight, hard_threshold, first_action_weight,
                terminal_loss_weight, terminal_distance_m, past_dropout):
    flow_only = isinstance(_unwrap(model), FlowActionExpertModel)
    prepared = _prepare(batch, vae, scale, action_scale, stats, device, precision)
    if flow_only:
        latent, target, target_physical, goal, proprio = prepared
    else:
        latent, target, target_physical, goal, proprio, past, past_mask = prepared
    noise = torch.randn_like(target)
    sigma = torch.rand(len(target), device=device, dtype=target.dtype)
    view = sigma[:, None, None]
    noisy = (1.0 - view) * target + view * noise
    flow_target = noise - target

    # The first command in a ten-step chunk is the only command that is
    # guaranteed to be applied before the next observation arrives.  A plain
    # mean over all ten labels therefore permits a small average error while
    # making the first command wrong enough to leave the expert manifold.
    # Randomly dropping the measured-velocity history also prevents the head
    # from depending on a teacher-forced signal that is noisy after a policy
    # starts deviating in closed loop.
    if not flow_only and float(past_dropout) > 0.0:
        drop = (torch.rand(len(past), device=device)
                < float(past_dropout))
        past = past.masked_fill(drop[:, None, None], 0.0)
        past_mask = past_mask.masked_fill(drop[:, None], 0.0)

    horizon_weights = torch.ones(
        target.shape[1], device=device, dtype=target.dtype)
    horizon_weights[0] = max(float(first_action_weight), 1.0)
    horizon_weights = horizon_weights / horizon_weights.mean().clamp_min(1.0e-6)

    with _autocast(device, precision):
        if flow_only:
            flow_velocity = model(
                latent, noisy, sigma * 1000.0, goal, proprio)
        else:
            flow_velocity, clean = model(
                latent, noisy, sigma * 1000.0, goal, proprio, past, past_mask)
        flow_error = (
            flow_velocity.float() - flow_target.float()
        ).square()
        flow_per_sample = (
            flow_error * horizon_weights[None, :, None].float()
        ).mean(dim=(1, 2))
        physical = target_physical.float()
        # Absolute Lee targets always contain forward cruise speed.  Weight
        # samples by lateral/vertical avoidance instead of total speed, or all
        # straight-flight samples would be classified as difficult.
        avoidance = torch.maximum(
            physical[..., 1].abs(), physical[..., 2].abs())
        difficult = avoidance.amax(dim=1) >= float(hard_threshold)
        sample_weight = 1.0 + float(hard_weight) * difficult.float()
        # Successful demonstrations contain a short terminal braking segment.
        # Without explicit weighting, the much longer cruise portion dominates
        # the direct head and it tends to overshoot the final 0.5 m/0.2 m/s
        # success window in closed loop.
        goal_mean = torch.as_tensor(
            stats["goal_mean"], device=device, dtype=goal.dtype)
        goal_std = torch.as_tensor(
            stats["goal_std"], device=device, dtype=goal.dtype)
        goal_distance = (
            goal[:, 3].float() * goal_std[3].float()
            + goal_mean[3].float())
        terminal = goal_distance <= float(terminal_distance_m)
        sample_weight = sample_weight + (
            float(terminal_loss_weight) * terminal.float())
        sample_weight = sample_weight / sample_weight.mean().clamp_min(1.0e-6)
        flow_loss = (flow_per_sample * sample_weight).mean()
        loss = float(flow_weight) * flow_loss
        if not flow_only:
            direct_error = F.smooth_l1_loss(
                clean.float(), target.float(), reduction="none", beta=0.25)
            direct_per_sample = (
                direct_error * horizon_weights[None, :, None].float()
            ).mean(dim=(1, 2))
            direct_loss = (direct_per_sample * sample_weight).mean()
            clean_physical = clean.float() * action_scale.float()
            delta_loss = F.smooth_l1_loss(
                clean_physical[:, 1:] - clean_physical[:, :-1],
                target_physical.float()[:, 1:] - target_physical.float()[:, :-1],
                beta=0.05)
            loss = (loss + float(direct_weight) * direct_loss
                    + float(delta_weight) * delta_loss)
    metrics = {
        "loss": loss.detach(),
        "flow_mse": flow_loss.detach(),
        "terminal_fraction": terminal.float().mean().detach(),
        "hard_fraction": difficult.float().mean().detach(),
    }
    if not flow_only:
        metrics.update({
            "direct_smooth_l1": direct_loss.detach(),
            "delta_smooth_l1_mps": delta_loss.detach(),
            "direct_l1_mps": (
                clean_physical - target_physical.float()).abs().mean().detach(),
            "first_action_l1_mps": (
                clean_physical[:, 0] - target_physical.float()[:, 0]
            ).abs().mean().detach(),
        })
    return loss, metrics


@torch.no_grad()
def _validate(model, vae, loader, scale, action_scale, stats, device,
              precision, hard_threshold):
    model.eval()
    raw = _unwrap(model)
    flow_only = isinstance(raw, FlowActionExpertModel)
    generator = torch.Generator(device=device).manual_seed(271828) if flow_only else None
    total = count = 0
    l1_sum = l2_sum = hard_sum = hard_count = sensitivity_sum = 0.0
    first_l1_sum = first_count = 0.0
    for batch in loader:
        prepared = _prepare(batch, vae, scale, action_scale, stats, device, precision)
        if flow_only:
            latent, _target_scaled, target, goal, proprio = prepared
            conditions = (goal, proprio)
        else:
            latent, _target_scaled, target, goal, proprio, past, past_mask = prepared
            conditions = (goal, proprio, past, past_mask)
        with _autocast(device, precision):
            sampling = {}
            if flow_only:
                sampling["initial_noise"] = torch.randn(
                    (len(target), ACTION_HORIZON, 3), device=device,
                    dtype=torch.float32, generator=generator)
            prediction = raw.predict_action(
                latent, *conditions, **sampling)
            # Use the SAME noise for the LiDAR permutation diagnostic, so its
            # change measures observation dependence rather than randomness.
            permuted = raw.predict_action(
                torch.roll(latent, 1, dims=0), *conditions, **sampling)
        prediction = prediction.float() * action_scale.float()
        error = prediction - target.float()
        batch_count = int(error.numel())
        l1_sum += float(error.abs().sum())
        first_l1_sum += float(error[:, 0].abs().sum())
        first_count += int(error[:, 0].numel())
        l2_sum += float(error.square().sum())
        total += batch_count
        difficult = torch.maximum(
            target.float()[..., 1].abs(),
            target.float()[..., 2].abs()).amax(dim=1) >= float(hard_threshold)
        if bool(difficult.any()):
            hard_sum += float(error[difficult].abs().sum())
            hard_count += int(error[difficult].numel())
        sensitivity_sum += float(
            (prediction - permuted.float() * action_scale.float())
            .abs().mean(dim=(1, 2)).sum())
        count += len(error)
    if dist.is_initialized():
        totals = torch.tensor(
            [total, count, l1_sum, l2_sum, hard_sum, hard_count,
             first_l1_sum, first_count, sensitivity_sum],
            dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        (total, count, l1_sum, l2_sum, hard_sum, hard_count,
         first_l1_sum, first_count, sensitivity_sum) = totals.tolist()
    model.train()
    result = {
        "action_l1_mps": l1_sum / max(total, 1),
        "action_rmse_mps": math.sqrt(l2_sum / max(total, 1)),
        "hard_action_l1_mps": hard_sum / max(hard_count, 1),
        "first_action_l1_mps": first_l1_sum / max(first_count, 1),
        # Closed-loop checkpoint selection emphasizes the command that is
        # actually applied immediately after each LiDAR replan.
        "selection_l1_mps": (
            0.7 * first_l1_sum / max(first_count, 1)
            + 0.3 * l1_sum / max(total, 1)),
        "lidar_permuted_action_delta_mps": sensitivity_sum / max(count, 1),
        "samples": int(count),
    }
    if flow_only:
        result.update(action_generation="flow", sample_steps=raw.sample_steps,
                      noise_seed=271828)
    return result


def _payload(model, optimizer, step, args, stats, metrics, vae_scale,
             vae_sha256):
    flow_only = isinstance(_unwrap(model), FlowActionExpertModel)
    result = {
        "format": FLOW_ACTION_EXPERT_FORMAT if flow_only else FORMAT,
        "step": int(step),
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "architecture": {
            "width": args.width, "depth": args.depth,
            "heads": args.heads, "ffn_width": args.ffn_width,
            "action_horizon": ACTION_HORIZON, "action_dim": 3,
            "proprio_dim": PROPRIO_DIM,
            "history_lidar_frames": LIDAR_HISTORY_FRAMES,
            "training_action_diffusion": True,
            "deployment_action_diffusion": flow_only,
            "policy_type": "flow" if flow_only else "direct",
            "has_direct_head": not flow_only,
            "sampling_steps": _unwrap(model).sample_steps if flow_only else 0,
            "sampling_method": "euler" if flow_only else "direct",
        },
        "action_interface": {
            "semantics": "actual_velocity_input_to_original_lee",
            "frame": "current_observation_yaw_only_ego_flu",
            "target_representation": "absolute_lee_target_velocity",
            "model_output_scale_mps_xyz": args.action_scale_mps_xyz,
            "physical_action_formula": (
                "model_output * model_output_scale_mps_xyz"),
            "normalized_output_constraint": (
                "unit_box_after_euler" if flow_only else "unit_l2_ball"),
            "lee_input": (
                "ego_to_world_yaw(model_output_mps, current_quaternion)"),
            "units_after_scale": "m/s",
        },
        "proprio_conditioning": {
            "semantics": "ego_velocity_ego_acceleration_quaternion",
            "deployment_use_proprio": True,
            "includes_world_position": False,
        },
        "condition_stats": ({key: value for key, value in stats.items()
                             if key != "past_velocity_scale_mps"} if flow_only else stats),
        "vae": {
            "variant": "circular",
            "checkpoint": str(args.vae_checkpoint),
            "sha256": vae_sha256,
            "scaling_factor": float(vae_scale),
            "frozen": True,
        },
        "config": vars(args),
    }
    if flow_only:
        result["action_interface"].update(
            physical_output_constraint="per_axis",
            physical_axis_limit_mps_xyz=[2.0, 2.0, 2.0],
            physical_norm_limit_mps=None,
        )
    if not flow_only:
        result["architecture"]["past_horizon"] = PAST_HORIZON
        result["past_action_conditioning"] = {
            "input_semantics": "measured_velocity_in_current_observation_ego_frame",
            "deployment_use_past_actions": bool(
                getattr(args, "deployment_use_past_actions", False)),
        }
    return result


def _resolve_data_split(args):
    """Select held-out shards without moving files or making new folders."""
    train_available = discover_shard_names(args.train_root)
    if not train_available:
        raise FileNotFoundError(
            f"no trajectory shards under {args.train_root}")
    validation_root = args.validation_root or args.train_root
    if validation_root == args.train_root:
        validation_names = tuple(args.validation_shards)
        if not validation_names:
            grouped: dict[str, list[str]] = {}
            for name in train_available:
                parts = Path(name).parts
                group = parts[0] if len(parts) > 1 else "__flat__"
                grouped.setdefault(group, []).append(name)
            # Hold out one complete trajectory shard from every obstacle
            # condition.  This prevents validation windows from sharing a
            # trajectory with training while preserving both 200/350 domains.
            validation_names = tuple(
                sorted(names)[-1] for _, names in sorted(grouped.items()))
        missing = sorted(set(validation_names) - set(train_available))
        if missing:
            raise FileNotFoundError(
                "validation shards are missing from train-root: "
                + ", ".join(missing))
        validation_set = set(validation_names)
        train_names = tuple(
            name for name in train_available if name not in validation_set)
        if not train_names:
            raise ValueError("validation shards consume the entire dataset")
    else:
        train_names = train_available
        validation_names = discover_shard_names(validation_root)
        if not validation_names:
            raise FileNotFoundError(
                f"no trajectory shards under {validation_root}")
    return train_names, validation_root, validation_names


def train(args):
    flow_only = getattr(args, "policy_type", "direct") == "flow"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if args.cuda_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    resume = (torch.load(args.resume_from, map_location="cpu", weights_only=False)
              if args.resume_from is not None else None)
    start_step = int(resume["step"]) if resume is not None else 0
    if args.steps <= start_step:
        raise ValueError("steps is the TOTAL target and must exceed the resume step")
    if resume is not None:
        old = resume.get("config", {})
        unchanged = ("batch_size", "seed", "precision", "lr", "weight_decay", "grad_clip",
                     "flow_weight", "direct_weight", "delta_weight", "first_action_weight",
                     "terminal_loss_weight", "terminal_distance_m", "hard_loss_weight",
                     "hard_action_threshold", "eval_windows", "sample_steps", "validation_shards")
        for key in unchanged:
            if key in old and json.dumps(old[key], default=str) != json.dumps(getattr(args, key), default=str):
                raise ValueError(f"resume would change {key}; use a separate experiment")
        for key in ("train_root", "vae_checkpoint"):
            if key in old and Path(old[key]).resolve() != getattr(args, key).resolve():
                raise ValueError(f"resume would change {key}")
        if resume.get("sampler_state", {}).get("world_size", world_size) != world_size:
            raise ValueError("resume requires the same world size to restore the data cursor")

    action_scale = torch.as_tensor(
        args.action_scale_mps_xyz, device=device,
        dtype=torch.float32).view(1, 1, 3)
    train_names, validation_root, validation_names = _resolve_data_split(args)
    train_data = SuperLeeActionDataset(
        args.train_root, shard_names=train_names,
        max_windows=args.max_train_windows, random_seed=args.seed,
        include_history=not flow_only)
    validation_full = SuperLeeActionDataset(
        validation_root, shard_names=validation_names,
        random_seed=args.seed, include_history=not flow_only)
    validation_indices = fixed_eval_indices(
        len(validation_full), args.eval_windows, args.seed + 17)
    validation_data = Subset(validation_full, validation_indices.tolist())
    if rank == 0:
        stats = (resume["condition_stats"] if resume is not None else _condition_stats(
            train_data, args.condition_stat_samples, args.seed + 31))
        if flow_only:
            stats.pop("past_velocity_scale_mps", None)
    else:
        stats = None
    if world_size > 1:
        values = [stats]
        dist.broadcast_object_list(values, src=0)
        stats = values[0]

    sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank,
        shuffle=True, seed=args.seed) if world_size > 1 else None
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=sampler is None, sampler=sampler,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, drop_last=True)
    batches_per_epoch = len(train_loader)
    if not batches_per_epoch:
        raise ValueError("training dataset has no complete batch")
    if (resume is not None and resume.get("sampler_state", {}).get(
            "batches_per_epoch", batches_per_epoch) != batches_per_epoch):
        raise ValueError("resume epoch length changed")
    epoch, batch_offset = divmod(start_step, batches_per_epoch)
    if start_step:
        if sampler is None:
            raise ValueError("cursor-preserving resume currently requires a DistributedSampler")
        sampler.set_epoch(epoch)
        train_loader = DataLoader(
            train_data, batch_sampler=SkipInitialBatches(train_loader.batch_sampler, batch_offset),
            num_workers=args.workers, pin_memory=True,
            persistent_workers=args.workers > 0)
    # Split the fixed validation subset without DistributedSampler padding:
    # every window is scored exactly once, then raw sums are all-reduced.
    local_validation = (Subset(validation_data, list(range(rank, len(validation_data), world_size)))
                        if world_size > 1 else validation_data)
    validation_loader = DataLoader(
        local_validation, batch_size=args.eval_batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)

    model = _build_model(args).to(device).float()
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95))
    if resume is not None:
        _restore_resume_state(resume, model, optimizer, args)

    vae, vae_scale = _load_frozen_vae(args.vae_checkpoint, device)
    vae_sha256 = _sha256(args.vae_checkpoint)
    if resume is not None and (resume["vae"]["sha256"] != vae_sha256
                               or resume["vae"]["scaling_factor"] != vae_scale):
        raise ValueError("resume must use the exact same frozen VAE and scaling factor")
    run_dir = args.out / args.run_name
    writer = None
    best = math.inf
    if resume is not None:
        best = float(resume.get("best_selection_score", resume["metrics"]["selection_l1_mps"]))
    if rank == 0:
        if resume is not None and run_dir.exists():
            old_config = json.loads((run_dir / "config.json").read_text())
            if old_config["world_size"] != world_size:
                raise ValueError("resume requires the original world size")
            if old_config["train_shards"] != list(train_names):
                raise ValueError("resume training shards changed")
            if old_config["train_windows"] != len(train_data):
                raise ValueError("resume training window count changed")
            if not np.array_equal(np.load(run_dir / "validation_indices.npy"), validation_indices):
                raise ValueError("resume fixed validation subset changed")
            newer = [path for path in run_dir.glob("step_*.pt")
                     if int(path.stem.split("_")[-1]) > start_step]
            if newer:
                raise ValueError("resume would overwrite later checkpoints; choose a new run name")
            previous_best = run_dir / "best_metrics.json"
            if previous_best.exists():
                best = min(best, float(json.loads(previous_best.read_text())["best_selection_l1_mps"]))
            config_path = run_dir / f"resume_step_{start_step:06d}_to_{args.steps:06d}.json"
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
            np.save(run_dir / "validation_indices.npy", validation_indices)
            config_path = run_dir / "config.json"
        config_path.write_text(json.dumps({
            **vars(args), "format": (
                FLOW_ACTION_EXPERT_FORMAT if getattr(args, "policy_type", "direct") == "flow"
                else FORMAT),
            "train_windows": len(train_data), "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
            "validation_windows": len(validation_full),
            "validation_subset": len(validation_data),
            "train_shards": list(train_names),
            "validation_shards": list(validation_names),
            "train_windows_by_shard": train_data.window_count_by_shard,
            "validation_windows_by_shard": (
                validation_full.window_count_by_shard),
            "condition_stats": stats,
            "resume_global_step": start_step, "resume_epoch": epoch,
            "resume_batch_offset": batch_offset,
            "resume_optimizer_restored": resume is not None,
        }, indent=2, default=str) + "\n")
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tensorboard"), flush_secs=10)
    if world_size > 1:
        dist.barrier()

    iterator = iter(train_loader)
    if resume is not None:
        states = resume.get("rng_states")
        if states is not None:
            if len(states) != world_size:
                raise ValueError("resume RNG state count differs from world size")
            torch.set_rng_state(states[rank]["torch_cpu"])
            torch.cuda.set_rng_state(states[rank]["torch_cuda"], device)
            np.random.set_state(states[rank]["numpy"])
        else:
            # Older checkpoints did not save RNG state. Resume weights, Adam,
            # stats and sampler exactly, but use a fresh reproducible noise
            # stream rather than replaying the first 20k steps' random draws.
            torch.manual_seed(args.seed + rank + start_step)
            np.random.seed(args.seed + rank + start_step)
        if rank == 0:
            print(json.dumps({"resumed_from_step": start_step, "target_total_steps": args.steps,
                              "optimizer_restored": True, "data_epoch": epoch,
                              "data_batch_offset": batch_offset,
                              "rng_restored": states is not None,
                              "lr": optimizer.param_groups[0]["lr"]}), flush=True)
        del resume
    started = time.monotonic()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _loss_batch(
            model, vae, batch, vae_scale, action_scale, stats, device,
            args.precision, args.flow_weight, args.direct_weight,
            args.delta_weight, args.hard_loss_weight,
            args.hard_action_threshold, args.first_action_weight,
            args.terminal_loss_weight, args.terminal_distance_m,
            getattr(args, "past_dropout", 0.0))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        if step == start_step + 1 or step % args.log_every == 0:
            values = torch.stack([value.float() for value in metrics.values()])
            if world_size > 1:
                dist.all_reduce(values)
                values /= world_size
            if rank == 0:
                row = {"step": step, **dict(zip(metrics, values.tolist())),
                       "grad_norm": float(grad), "lr": optimizer.param_groups[0]["lr"],
                       "global_batch_size": args.batch_size * world_size,
                       "elapsed_min": (time.monotonic() - started) / 60.0,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                       "reserved_gib": torch.cuda.memory_reserved(device) / 2**30}
                print(json.dumps(row), flush=True)
                for key, value in row.items():
                    if key != "step":
                        writer.add_scalar(f"train/{key}", value, step)

        if step % args.eval_every == 0 or step == args.steps:
            validation = _validate(
                model, vae, validation_loader, vae_scale, action_scale,
                stats, device, args.precision, args.hard_action_threshold)
            rng_states = _checkpoint_rng_states(device)
            if rank == 0:
                print(json.dumps({"step": step, "validation": validation}),
                      flush=True)
                for key, value in validation.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"validation/{key}", value, step)
                writer.flush()
                improved = validation["selection_l1_mps"] < best
                best = min(best, validation["selection_l1_mps"])
                payload = _payload(
                    model, optimizer, step, args, stats,
                    validation, vae_scale, vae_sha256)
                payload.update(best_selection_score=best, rng_states=rng_states,
                               sampler_state={"world_size": world_size,
                                              "batches_per_epoch": batches_per_epoch,
                                              "epoch": step // batches_per_epoch,
                                              "batch_offset": step % batches_per_epoch})
                _atomic_save(payload, run_dir / "latest.pt")
                if improved:
                    _atomic_save(payload, run_dir / "best.pt")
                # ``latest.pt`` is atomically replaced at every validation
                # point.  The closed-loop monitor reads its embedded step
                # number, so we do not create dozens of multi-hundred-MB
                # duplicate checkpoints while still allowing evaluation every
                # 500 updates, including before the nominal closed-loop start.
                if step >= args.closed_loop_start:
                    _atomic_save(payload, run_dir / f"step_{step}.pt")
                (run_dir / "best_metrics.json").write_text(json.dumps({
                    "step": step, "best_selection_l1_mps": best,
                    "best_action_l1_mps": best,
                    "last_validation": validation,
                }, indent=2) + "\n")
            if world_size > 1:
                dist.barrier()
    if writer:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


def parse_args(argv=None, *, default_policy_type="direct"):
    description = (
        "Train a current-observation-only Action Flow Expert. Frozen VAE, "
        "current LiDAR + goal + proprio; no historical-motion inputs or "
        "placeholder tokens. Flow MSE training and ten-step Euler generation."
        if default_policy_type == "flow" else __doc__)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--policy-type", choices=("direct", "flow"), default=default_policy_type)
    parser.add_argument("--sample-steps", type=int, default=10,
                        help="Euler steps for Flow-only validation and deployment (default: 10)")
    parser.add_argument("--resume-from", type=Path,
                        help="full checkpoint: restore model, optimizer, stats and global step")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument(
        "--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT,
        help="5 m circular VAE safetensors checkpoint (kept frozen)")
    parser.add_argument(
        "--validation-shards",
        default="",
        help=("comma-separated shard directories held out when validation-root "
              "is omitted or equals train-root; empty automatically holds out "
              "the last shard from every obstacle-condition directory"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--run-name", default=None)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=None,
                        help="optional per-process PyTorch allocator cap on shared GPUs")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-width", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--direct-weight", type=float, default=None)
    parser.add_argument("--delta-weight", type=float, default=None)
    parser.add_argument("--hard-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-action-threshold", type=float, default=0.5)
    parser.add_argument(
        "--first-action-weight", type=float, default=4.0,
        help="relative direct/flow loss weight for the first command in each 10-step chunk")
    parser.add_argument(
        "--terminal-loss-weight", type=float, default=2.0,
        help="extra sample weight for goal distances within terminal-distance-m")
    parser.add_argument(
        "--terminal-distance-m", type=float, default=3.0,
        help="goal-distance threshold for terminal braking samples")
    if default_policy_type == "direct":
        parser.add_argument(
            "--past-dropout", type=float, default=None,
            help="legacy Direct policy: measured velocity history dropout")
        parser.add_argument(
            "--deployment-use-past-actions", action="store_true",
            help="legacy Direct policy: enable measured velocity history")
    parser.add_argument(
        "--action-scale-mps-xyz", default="2,2,2",
        help="physical m/s per normalized output unit; Flow-only requires 2,2,2 (per-axis limit)")
    parser.add_argument("--condition-stat-samples", type=int, default=20_000)
    parser.add_argument("--eval-windows", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--closed-loop-start", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args(argv)
    if args.cuda_memory_fraction is not None and not (0.0 < args.cuda_memory_fraction <= 1.0):
        raise ValueError("cuda-memory-fraction must be in (0, 1]")
    flow_only = args.policy_type == "flow"
    if flow_only:
        if (getattr(args, "deployment_use_past_actions", False)
                or getattr(args, "past_dropout", None) is not None):
            raise ValueError("Flow-only has no historical motion input")
        for key in ("past_dropout", "deployment_use_past_actions"):
            if hasattr(args, key):
                delattr(args, key)
    else:
        if getattr(args, "past_dropout", None) is None:
            args.past_dropout = .25
        if not hasattr(args, "deployment_use_past_actions"):
            args.deployment_use_past_actions = False
    if args.run_name is None:
        args.run_name = ("action_expert_flow_history1_5m_v1" if flow_only
                         else "action_dit_final_lee_ego_history1_5m_v3")
    if args.direct_weight is None:
        args.direct_weight = 0.0 if flow_only else 4.0
    if args.delta_weight is None:
        args.delta_weight = 0.0 if flow_only else 0.2
    if flow_only and (args.direct_weight != 0.0 or args.delta_weight != 0.0):
        raise ValueError("Flow-only has no Direct/delta loss; their weights must be zero")
    if flow_only and (args.flow_weight <= 0 or not math.isfinite(args.flow_weight)):
        raise ValueError("Flow-only requires a finite positive flow-weight")
    if args.sample_steps <= 0:
        raise ValueError("sample-steps must be positive")
    args.train_root = args.train_root.expanduser().resolve()
    if args.validation_root is not None:
        args.validation_root = args.validation_root.expanduser().resolve()
    args.vae_checkpoint = args.vae_checkpoint.expanduser().resolve()
    if args.resume_from is not None:
        args.resume_from = args.resume_from.expanduser().resolve()
        if not args.resume_from.is_file():
            raise FileNotFoundError(args.resume_from)
    args.validation_shards = tuple(
        value.strip() for value in args.validation_shards.split(",")
        if value.strip())
    if not args.vae_checkpoint.is_file():
        raise FileNotFoundError(
            f"LiDAR VAE checkpoint missing: {args.vae_checkpoint}")
    args.out = args.out.expanduser().resolve()
    args.action_scale_mps_xyz = [
        float(value) for value in args.action_scale_mps_xyz.split(",")]
    if (len(args.action_scale_mps_xyz) != 3
            or not all(math.isfinite(value) and value > 0.0
                       for value in args.action_scale_mps_xyz)):
        raise ValueError("action-scale-mps-xyz requires 3 finite positive values")
    if flow_only and args.action_scale_mps_xyz != [2.0, 2.0, 2.0]:
        raise ValueError("Flow-only per-axis +/-2 m/s contract requires action-scale-mps-xyz=2,2,2")
    if args.closed_loop_start < args.eval_every:
        raise ValueError("closed-loop-start must be at least eval-every")
    return args


if __name__ == "__main__":
    train(parse_args())
