"""Jointly train a causal action expert with the existing LaGen LiDAR UNet.

The implementation follows Fast-WAM's training/inference split without using
Wan: during training, a world diffusion loss and an action flow-matching loss
share a clean-current-LiDAR encoder.  During action inference only the current
LiDAR, goal, current proprioception, and past actions are available.  Future
LiDAR targets are never visible to the action branch.

Examples (on the Aliyun PPU host)::

  python -m lidar_wam.runner.action_expert_joint inspect --data-root DATA
  python -m lidar_wam.runner.action_expert_joint smoke --data-root DATA \
      --world-checkpoint outputs/world_direct_t3/latest.pt
  python -m lidar_wam.runner.action_expert_joint train --data-root DATA \
      --world-checkpoint outputs/world_direct_t3/latest.pt --overfit
  python -m lidar_wam.runner.action_expert_joint train --data-root DATA \
      --world-checkpoint outputs/world_direct_t3/latest.pt --steps 30000
  python -m lidar_wam.runner.action_expert_joint evaluate --data-root DATA \
      --checkpoint outputs/action_expert_joint/best.pt --split val
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from lidar_wam.coordinates import semantics as coordinate_semantics
from lidar_wam.data_v2 import (
    StratifiedSampler,
    V2WindowDataset,
    compute_condition_stats,
    file_sha256,
)
from lidar_wam.models.action_expert import ActionOnlyModel, JointWorldActionModel
from lidar_wam.models.world import DirectHorizonWorldModel
from lidar_wam.runner import stage1
from lidar_wam.runner.world_decoded_aux import (
    decoded_losses,
    predicted_x0,
    select_aux_indices,
)
from lidar_wam.runner.world_direct_horizon import (
    DirectHorizonDataset,
    WorldV2View,
    evaluate_world_v2,
    frame_metrics,
    weighted_future_loss,
)


RUN_NAME = "action_expert_joint"
ACTION_HORIZON = 10
ACTION_DIM = 3
FLOW_EPS = 1e-4
TRAINING_FORMAT_V2 = "navrl-action-expert-training-v2"
JOINT_TRAINING_FORMAT_V3 = "navrl-joint-world-action-training-v3"
JOINT_POLICY_FORMAT_V3 = "navrl-joint-world-action-policy-v3"
JOINT_TRAINING_FORMAT_V4 = "navrl-bidirectional-world-action-training-v4"
JOINT_POLICY_FORMAT_V4 = "navrl-bidirectional-world-action-policy-v4"
JOINT_TRAINING_FORMAT_V5 = "navrl-single-future-action10-training-v5"
JOINT_POLICY_FORMAT_V5 = "navrl-single-future-action10-policy-v5"


def _decode(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def _rotation_wxyz(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        raise ValueError("Invalid zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float32)


def _causal_features(drone_state, target_world):
    """Return body-frame goal [xyz,distance] and 10-D causal proprioception."""
    state = np.asarray(drone_state, dtype=np.float32)
    if state.shape != (13,):
        raise ValueError(f"Expected 13-D drone state, got {state.shape}")
    rotation = _rotation_wxyz(state[3:7])
    world_to_body = rotation.T
    goal_body = world_to_body @ (np.asarray(target_world, np.float32) - state[:3])
    goal = np.concatenate([goal_body, [np.linalg.norm(goal_body)]], dtype=np.float32)
    velocity_body = world_to_body @ state[7:10]
    angular_body = world_to_body @ state[10:13]
    gravity_body = world_to_body @ np.asarray([0.0, 0.0, -1.0], np.float32)
    proprio = np.concatenate([
        state[2:3], velocity_body, angular_body, gravity_body
    ]).astype(np.float32)
    return goal, proprio


def _load_raw_metadata(raw_root: Path, split: str):
    rows = {}
    paths = sorted((raw_root / split).glob("seed_*/metadata/frames.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No {split}/seed_*/metadata/frames.jsonl under {raw_root}")
    for path in paths:
        with path.open() as stream:
            for line in stream:
                value = json.loads(line)
                token = str(value["token"])
                if token in rows:
                    raise ValueError(f"Duplicate raw token {token}")
                rows[token] = {
                    "target": value["target_position_world"],
                    "drone": value["drone_state"],
                    "collision": bool(value.get("collision", False)),
                    "out_of_bounds": bool(value.get("out_of_bounds", False)),
                    "reach_goal": bool(value.get("reach_goal", False)),
                    "termination_reason": value.get("termination_reason", ""),
                }
    return rows


class JointActionDataset(Dataset):
    """Direct-t+3 world samples plus strictly causal policy conditions."""

    def __init__(self, split: str, data_root: Path, latent_root: Path,
                 raw_root: Path, source_chains=None, limit_per_seed=None,
                 random_seed=42, overfit=False, drop_failure_windows=False):
        base = DirectHorizonDataset(
            split, data_root, latent_root, source_chains=source_chains,
            limit_per_seed=limit_per_seed, random_seed=random_seed,
            overfit=overfit, include_drone_state=True)
        metadata = _load_raw_metadata(raw_root, split)
        h5_path = Path(data_root) / f"navrl_static_{split}.h5"
        with h5py.File(h5_path, "r") as h5:
            tokens = [_decode(v) for v in h5["token"][:]]
            previous_tokens = [_decode(v) for v in h5["prev_token"][:]]
            scenes = [_decode(v) for v in h5["scene_token"][:]]
            action_mask = np.asarray(h5["action_mask"][:], dtype=bool)
            step_delta = np.asarray(h5["step_delta"][:])
            all_actions = np.asarray(h5["normalized_action_sequence"][:], np.float32)
        token_to_row = {token: row for row, token in enumerate(tokens) if token}

        goal, proprio, past, past_mask, keep, outcomes = [], [], [], [], [], []
        for local_index, chain in enumerate(base.source_indices):
            first_row = int(chain[0])
            current_token = previous_tokens[first_row]
            record = metadata.get(current_token)
            if record is None:
                raise KeyError(f"Raw metadata is missing initial token {current_token}")
            future_records = [metadata.get(tokens[int(row)]) for row in chain]
            if any(value is None for value in future_records):
                raise KeyError(f"Raw metadata is missing a target token in {chain.tolist()}")
            failed = any(value["collision"] or value["out_of_bounds"]
                         for value in future_records)
            if drop_failure_windows and failed:
                continue
            state = np.asarray(base.previous_drone_state[local_index], np.float32)
            current_goal, current_proprio = _causal_features(state, record["target"])
            previous_row = token_to_row.get(current_token)
            valid_past = (
                previous_row is not None
                and scenes[previous_row] == scenes[first_row]
                and int(step_delta[previous_row]) == 10
                and bool(action_mask[previous_row].all())
                and bool(np.isfinite(all_actions[previous_row]).all())
            )
            if valid_past:
                past.append(all_actions[previous_row])
                past_mask.append(np.ones(10, np.float32))
            else:
                past.append(np.zeros((10, 3), np.float32))
                past_mask.append(np.zeros(10, np.float32))
            goal.append(current_goal)
            proprio.append(current_proprio)
            keep.append(local_index)
            outcomes.append({
                "collision_or_oob": failed,
                "reach_goal": any(value["reach_goal"] for value in future_records),
            })
        if not keep:
            raise RuntimeError(f"No usable joint action samples in {split}")
        self.base = base
        self.keep = np.asarray(keep, np.int64)
        self.goal = torch.from_numpy(np.asarray(goal, np.float32))
        self.proprio = torch.from_numpy(np.asarray(proprio, np.float32))
        self.past = torch.from_numpy(np.asarray(past, np.float32))
        self.past_mask = torch.from_numpy(np.asarray(past_mask, np.float32))
        self.source_indices = base.source_indices[self.keep]
        self.seeds = base.seeds[self.keep]
        self.outcomes = outcomes
        self.scale = base.scale

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, index):
        base = self.base[int(self.keep[index])]
        return base + (self.goal[index], self.proprio[index], self.past[index],
                       self.past_mask[index])


def _read_manifest(path: Path | None):
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    return [row["source_indices"] for row in payload["rows"]]


def build_dataset(split, args, overfit=False):
    manifest = None
    if not overfit and split in ("val", "test"):
        candidate = args.manifest_root / f"fixed_{split}_manifest.json"
        manifest = _read_manifest(candidate)
    return JointActionDataset(
        split, args.data_root, args.latent_root, args.raw_data_root,
        source_chains=manifest,
        # Training must use every seed-0..15 trajectory. The per-seed cap is
        # only for deterministic held-out evaluation manifests.
        limit_per_seed=(None if split == "train" or manifest is not None
                        else args.samples_per_seed),
        random_seed=args.seed, overfit=overfit,
        drop_failure_windows=args.drop_failure_windows and split == "train")


def compute_stats(dataset: JointActionDataset):
    action = dataset.base.actions[dataset.keep].reshape(-1, 3).numpy()
    transformed = np.log(np.clip(action, FLOW_EPS, 1-FLOW_EPS)) - np.log1p(
        -np.clip(action, FLOW_EPS, 1-FLOW_EPS))
    return {
        "action_logit_mean": transformed.mean(0).tolist(),
        "action_logit_std": np.maximum(transformed.std(0), 1e-4).tolist(),
        "action_raw_mean": action.mean(0).tolist(),
        "goal_mean": dataset.goal.numpy().mean(0).tolist(),
        "goal_std": np.maximum(dataset.goal.numpy().std(0), 1e-4).tolist(),
        "proprio_mean": dataset.proprio.numpy().mean(0).tolist(),
        "proprio_std": np.maximum(dataset.proprio.numpy().std(0), 1e-4).tolist(),
        "logit_clip_epsilon": FLOW_EPS,
    }


def _tensor_stats(stats, key, like):
    return torch.as_tensor(stats[key], device=like.device, dtype=like.dtype)


def action_to_flow(action, stats):
    clipped = action.clamp(FLOW_EPS, 1-FLOW_EPS)
    value = torch.log(clipped) - torch.log1p(-clipped)
    return ((value - _tensor_stats(stats, "action_logit_mean", value)) /
            _tensor_stats(stats, "action_logit_std", value))


def flow_to_action(value, stats):
    value = (value * _tensor_stats(stats, "action_logit_std", value)
             + _tensor_stats(stats, "action_logit_mean", value))
    return value.sigmoid()


def normalize_condition(value, stats, prefix):
    return ((value - _tensor_stats(stats, f"{prefix}_mean", value)) /
            _tensor_stats(stats, f"{prefix}_std", value))


def load_joint_model(args, checkpoint=None):
    world = DirectHorizonWorldModel().to(stage1.DEVICE).float()
    if checkpoint is None:
        payload = torch.load(args.world_checkpoint, map_location="cpu", weights_only=False)
        world.load_state_dict(payload["model"], strict=True)
        world_step = int(payload.get("step", -1))
    else:
        world_step = int(checkpoint.get("world_initial_step", -1))
    model = JointWorldActionModel(
        world, width=args.action_width, depth=args.action_depth,
        heads=args.action_heads, ffn_width=args.action_ffn_width,
    ).to(stage1.DEVICE).float()
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
    return model, world_step


def action_flow_loss(model, previous, actions, goal, proprio, past, past_mask,
                     stats, observation_tokens=None):
    target = action_to_flow(actions.flatten(1, 2), stats)
    noise = torch.randn_like(target)
    sigma = torch.rand(len(target), device=target.device, dtype=target.dtype)
    noisy = (1-sigma[:, None, None])*target + sigma[:, None, None]*noise
    timestep = sigma * 1000.0
    goal = normalize_condition(goal, stats, "goal")
    proprio = normalize_condition(proprio, stats, "proprio")
    past_flow = action_to_flow(past, stats) * past_mask.unsqueeze(-1)
    velocity = model.predict_action_velocity(
        noisy, timestep, previous, goal, proprio, past_flow, past_mask,
        observation_tokens=observation_tokens)
    target_velocity = noise - target
    flow = F.mse_loss(velocity, target_velocity)
    x0 = noisy - sigma[:, None, None] * velocity
    delta = F.l1_loss(x0[:, 1:] - x0[:, :-1],
                      target[:, 1:] - target[:, :-1])
    return flow + 0.02 * delta, {"action_flow_mse": flow.detach(),
                                "action_delta_l1": delta.detach()}


def world_diffusion_loss(model, previous, target, actions, state, target_image,
                         vae, scale, args, observation_tokens=None):
    scheduler = stage1.DDPMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon")
    noise = torch.randn_like(target)
    timestep = torch.randint(0, 1000, (len(target),), device=target.device)
    noisy = scheduler.add_noise(target, noise, timestep)
    epsilon = model.predict_world(
        noisy, previous, actions, state, timestep,
        observation_tokens=observation_tokens)
    epsilon_loss = F.mse_loss(epsilon, noise)
    eligible = select_aux_indices(
        timestep, target_image, getattr(args, "aux_t_max", 500),
        getattr(args, "aux_batch_max", 64),
        getattr(args, "aux_empty_max", 16))
    zero = epsilon_loss * 0
    latent = mask = empty_mask = range_loss = presence = empty = zero
    if len(eligible):
        x0 = predicted_x0(noisy[eligible], epsilon[eligible],
                          timestep[eligible], scheduler)
        latent = F.l1_loss(x0, target[eligible])
        (mask, empty_mask, range_loss, presence, empty, _, _) = decoded_losses(
            vae, x0, target_image[eligible], scale, 1.5, 8, 0.5)
    total = (epsilon_loss + 0.1*latent + 0.1*mask + 0.1*empty_mask
             + 0.05*range_loss + 0.02*presence + 0.2*empty)
    return total, {
        "world_epsilon_mse": epsilon_loss.detach(), "world_x0_l1": latent.detach(),
        "world_mask_bce": mask.detach(), "world_range_l1": range_loss.detach(),
        "world_presence": presence.detach(), "world_empty": empty.detach(),
    }


@torch.no_grad()
def sample_actions(model, previous, goal, proprio, past, past_mask, stats,
                   steps=10, seed=42, world_state=None):
    generator = torch.Generator(device=previous.device).manual_seed(int(seed))
    value = torch.randn((len(previous), ACTION_HORIZON, ACTION_DIM),
                        device=previous.device, dtype=previous.dtype,
                        generator=generator)
    goal = normalize_condition(goal, stats, "goal")
    proprio = normalize_condition(proprio, stats, "proprio")
    past = action_to_flow(past, stats) * past_mask.unsqueeze(-1)
    observation = model.encode_current(previous)
    sigmas = torch.linspace(1, 0, steps+1, device=previous.device,
                            dtype=previous.dtype)
    joint = isinstance(model, JointWorldActionModel)
    if joint:
        if world_state is None:
            raise ValueError("joint action sampling requires the causal world state")
        scheduler = stage1.DDIMScheduler(
            num_train_timesteps=1000, prediction_type="epsilon",
            clip_sample=False)
        scheduler.set_timesteps(steps, device=previous.device)
        future = torch.randn(
            (len(previous), *previous.shape[1:]), generator=generator,
            device=previous.device,
            dtype=previous.dtype)
        world_timesteps = scheduler.timesteps
        action_mean = _tensor_stats(stats, "action_logit_mean", value)
        action_std = _tensor_stats(stats, "action_logit_std", value)
    else:
        world_timesteps = [None] * steps
    for (current, following, world_timestep) in zip(
            sigmas[:-1], sigmas[1:], world_timesteps):
        timestep = torch.full((len(previous),), float(current*1000),
                              device=previous.device, dtype=previous.dtype)
        if joint:
            velocity, epsilon, _ = model.predict_joint_velocity(
                value, timestep, future, world_timestep, previous,
                goal, proprio, past, past_mask, world_state,
                action_mean, action_std,
                scheduler.alphas_cumprod[world_timestep].expand(len(previous)),
                observation_tokens=observation)
            future = scheduler.step(
                epsilon, world_timestep, future, eta=0.0).prev_sample
        else:
            velocity = model.predict_action_velocity(
                value, timestep, previous, goal, proprio, past, past_mask,
                observation_tokens=observation)
        value = value + (following-current) * velocity
    return flow_to_action(value, stats)


def summarize_action(prediction, target):
    error = (prediction-target).abs()
    result = {
        "action_mae": float(error.mean()),
        "action_rmse": float((prediction-target).square().mean().sqrt()),
        "first_10_mae": float(error[:, :10].mean()),
    }
    for axis in range(3):
        result[f"axis_{axis}_mae"] = float(error[:, :, axis].mean())
    result["chunk_1_mae"] = float(error.mean())
    return result


@torch.no_grad()
def evaluate_actions(model, dataset, stats, args, save_rows=False):
    model.eval()
    predictions, targets, last_baseline, sources = [], [], [], []
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.workers)
    offset = 0
    for batch in loader:
        (previous, _target, actions, state, _image, source,
         goal, proprio, past, past_mask) = [v.to(stage1.DEVICE) for v in batch]
        prediction = sample_actions(
            model, previous, goal, proprio, past, past_mask, stats,
            args.flow_steps, args.seed + offset, world_state=state)
        last = torch.where(
            past_mask[:, -1:, None].bool(), past[:, -1:, :],
            torch.as_tensor(stats["action_raw_mean"], device=past.device)[None, None]
        ).expand(-1, ACTION_HORIZON, -1)
        predictions.append(prediction.cpu())
        targets.append(actions.flatten(1, 2).cpu())
        last_baseline.append(last.cpu())
        sources.append(source.cpu())
        offset += len(previous)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    last = torch.cat(last_baseline)
    mean = torch.as_tensor(stats["action_raw_mean"])[None, None].expand_as(target)
    summary = summarize_action(prediction, target)
    summary["mean_action_baseline_mae"] = float((mean-target).abs().mean())
    summary["repeat_last_baseline_mae"] = float((last-target).abs().mean())
    summary["relative_improvement_vs_repeat_last"] = (
        1-summary["action_mae"]/max(summary["repeat_last_baseline_mae"], 1e-8))
    rows = []
    if save_rows:
        source = torch.cat(sources)
        sample_mae = (prediction-target).abs().mean((1, 2))
        for i in range(len(target)):
            rows.append({
                "source_indices": [int(v) for v in source[i]],
                "seed": int(dataset.seeds[i]),
                "action_mae": float(sample_mae[i]),
                "prediction": prediction[i].tolist(),
                "target": target[i].tolist(),
            })
    return summary, rows


def _optimizer(model, args):
    action = list(model.action_expert.parameters())
    shared = list(model.observation.parameters()) + list(model.observation_to_world.parameters())
    world = list(model.world.parameters())
    if args.freeze_world:
        for parameter in world:
            parameter.requires_grad_(False)
        world = []
    groups = [
        {"params": action, "lr": args.action_lr, "name": "action"},
        {"params": shared, "lr": args.shared_lr, "name": "shared"},
    ]
    if world:
        groups.append({"params": world, "lr": args.world_lr, "name": "world"})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _save_checkpoint(path, model, optimizer, step, stats, world_step, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "stats": stats, "world_initial_step": world_step,
        "extra": extra,
    }, temporary)
    temporary.replace(path)


def train(args):
    run_dir = args.out / (RUN_NAME + "_overfit" if args.overfit else RUN_NAME)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_data = build_dataset("train", args, overfit=args.overfit)
    val_data = train_data if args.overfit else build_dataset("val", args)
    stats = compute_stats(train_data)
    stage1.save_json(run_dir / "normalization.json", stats)
    model, world_step = load_joint_model(args)
    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    optimizer = _optimizer(model, args)
    first_step = 1
    best = math.inf
    if args.resume:
        payload = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        stats = payload["stats"]
        first_step = int(payload["step"])+1
        best = float(payload.get("extra", {}).get("best_action_mae", math.inf))
    config = {
        "architecture": "LaGen direct-t3 UNet + shared LiDAR encoder + action flow Transformer",
        "fastwam_adaptation": "joint world/action losses; action attends current LiDAR only",
        "world_checkpoint": str(args.world_checkpoint), "world_initial_step": world_step,
        "action_shape": [30, 3], "output": "normalized PPO action in [0,1]",
        "causal_inputs": ["current_lidar_latent", "goal_in_body_frame",
                          "current_proprioception", "previous_10_actions"],
        "action_expert": {"width": args.action_width, "depth": args.action_depth,
                          "heads": args.action_heads, "ffn_width": args.action_ffn_width},
        "loss_weights": {"world": args.world_weight, "action": args.action_weight,
                         "action_delta": 0.02},
        "freeze_world": args.freeze_world, "steps": args.steps,
        "batch_size": args.batch_size, "normalization": stats,
    }
    stage1.save_json(run_dir / "config.json", config)
    loader = stage1.infinite(DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True))
    history = []
    started = time.monotonic()
    parameters = [p for p in model.parameters() if p.requires_grad]
    for step in range(first_step, args.steps+1):
        model.train()
        batch = [value.to(stage1.DEVICE, non_blocking=True) for value in next(loader)]
        (previous, target, actions, state, target_image, _source,
         goal, proprio, past, past_mask) = batch
        optimizer.zero_grad(set_to_none=True)
        observation = model.encode_current(previous)
        action_loss, action_parts = action_flow_loss(
            model, previous, actions, goal, proprio, past, past_mask, stats,
            observation_tokens=observation)
        if args.world_weight:
            world_loss, world_parts = world_diffusion_loss(
                model, previous, target, actions, state, target_image,
                vae, train_data.scale, args, observation_tokens=observation)
        else:
            world_loss = action_loss*0
            world_parts = {}
        loss = args.action_weight*action_loss + args.world_weight*world_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite joint loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            record = {
                "step": step, "loss": float(loss), "action_loss": float(action_loss),
                "world_loss": float(world_loss), "grad_norm": float(grad),
                **{k: float(v) for k, v in {**action_parts, **world_parts}.items()},
                "elapsed_min": round((time.monotonic()-started)/60, 2),
            }
            if stage1.DEVICE.type == "cuda":
                record["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
            history.append(record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            summary, _ = evaluate_actions(model, val_data, stats, args)
            print(json.dumps({"step": step, "validation": summary}), flush=True)
            extra = {"validation": summary, "best_action_mae": min(best, summary["action_mae"])}
            _save_checkpoint(run_dir/"latest.pt", model, optimizer, step,
                             stats, world_step, extra)
            if summary["action_mae"] < best:
                best = summary["action_mae"]
                extra["best_action_mae"] = best
                _save_checkpoint(run_dir/"best.pt", model, optimizer, step,
                                 stats, world_step, extra)
                stage1.save_json(run_dir/"best_metrics.json", {"step": step, **summary})
            stage1.save_json(run_dir/"history.json", history)


def inspect(args):
    report = {}
    for split in ("train", "val", "test"):
        data = build_dataset(split, args, overfit=False)
        report[split] = {
            "samples": len(data), "latent_shape": list(data.base.previous.shape[1:]),
            "future_action_shape": list(data.base.actions.shape[1:]),
            "goal_shape": list(data.goal.shape[1:]),
            "proprio_shape": list(data.proprio.shape[1:]),
            "past_action_shape": list(data.past.shape[1:]),
            "missing_past_fraction": float((data.past_mask.sum(1)==0).float().mean()),
            "collision_or_oob_windows": int(sum(v["collision_or_oob"] for v in data.outcomes)),
            "reach_goal_windows": int(sum(v["reach_goal"] for v in data.outcomes)),
        }
    train_small = build_dataset("train", args, overfit=True)
    report["normalization_preview"] = compute_stats(train_small)
    print(json.dumps(report, indent=2), flush=True)


def smoke(args):
    data = build_dataset("train", args, overfit=True)
    stats = compute_stats(data)
    model, world_step = load_joint_model(args)
    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    batch = [v[:2].to(stage1.DEVICE) for v in next(iter(DataLoader(data, batch_size=2)))]
    (previous, target, actions, state, target_image, _source,
     goal, proprio, past, past_mask) = batch
    observation = model.encode_current(previous)
    action_loss, _ = action_flow_loss(model, previous, actions, goal, proprio,
                                      past, past_mask, stats, observation)
    world_loss, _ = world_diffusion_loss(model, previous, target, actions, state,
                                         target_image, vae, data.scale, args, observation)
    loss = action_loss + world_loss
    loss.backward()
    with torch.no_grad():
        generated = sample_actions(model, previous, goal, proprio, past,
                                   past_mask, stats, steps=2, seed=args.seed)
    print(json.dumps({
        "world_checkpoint_step": world_step, "joint_loss": float(loss),
        "action_loss": float(action_loss), "world_loss": float(world_loss),
        "observation_tokens": list(observation.shape),
        "generated_actions": list(generated.shape),
        "generated_min": float(generated.min()), "generated_max": float(generated.max()),
        "finite_gradients": all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                for p in model.parameters()),
    }, indent=2), flush=True)


def evaluate(args):
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = load_joint_model(args, checkpoint=payload)
    data = build_dataset(args.split, args)
    summary, rows = evaluate_actions(model, data, payload["stats"], args, save_rows=True)
    result = {"split": args.split, "checkpoint": str(args.checkpoint),
              "checkpoint_step": int(payload["step"]), "summary": summary,
              "rows": rows}
    run_dir = args.out/RUN_NAME/f"evaluation_{args.split}"
    stage1.save_json(run_dir/"action_metrics.json", result)
    print(json.dumps(summary, indent=2), flush=True)


def _distributed_runtime(precision: str):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--precision bf16 requested but this GPU lacks BF16 support")
    stage1.DEVICE = device
    return distributed, rank, world_size, local_rank, device


def _autocast(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _sha_or_none(path):
    return file_sha256(path) if path is not None and Path(path).is_file() else None


def _v2_dataset(split, args, overfit=False, samples=None, limit=None):
    return V2WindowDataset(
        split, args.dataset_root, args.latent_root, args.index_root,
        overfit=overfit, samples_per_seed=samples, random_seed=args.seed,
        limit=limit, all_future_targets=args.command == "train-joint")


def _broadcast_stats(dataset, rank):
    values = [compute_condition_stats(dataset) if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(values, src=0)
    return values[0]


def _v2_action_terms(model, previous, actions, goal, proprio, past,
                     past_mask, stats, joint_inputs=None,
                     x0_weight=0.0, delta_weight=0.20,
                     candidate_weight=0.10):
    if actions.ndim == 4:
        actions = actions[:, 0]
    target = action_to_flow(actions, stats)
    noise = torch.randn_like(target)
    sigma = torch.rand(len(target), device=target.device, dtype=target.dtype)
    noisy = (1-sigma[:, None, None])*target + sigma[:, None, None]*noise
    timestep = sigma * 1000.0
    normalized_goal = normalize_condition(goal, stats, "goal")
    normalized_proprio = normalize_condition(proprio, stats, "proprio")
    past_flow = action_to_flow(past, stats) * past_mask.unsqueeze(-1)
    kwargs = {} if joint_inputs is None else joint_inputs
    velocity, world_epsilon = model(
        noisy, timestep, previous, normalized_goal, normalized_proprio,
        past_flow, past_mask, **kwargs)
    target_velocity = noise - target
    flow_error = (velocity-target_velocity).square()
    flow = flow_error.mean()
    x0 = noisy - sigma[:, None, None] * velocity
    endpoint_error = (x0-target).abs()
    endpoint = endpoint_error.mean()
    delta_error = ((x0[:, 1:] - x0[:, :-1])
                   - (target[:, 1:] - target[:, :-1])).abs()
    delta = delta_error.mean()
    candidate = flow*0
    if joint_inputs is not None:
        raw_model = _unwrap(model)
        candidate_velocity = raw_model._last_candidate_velocity
        candidate = F.mse_loss(candidate_velocity, target_velocity)
    total = (flow + float(x0_weight)*endpoint
             + float(delta_weight)*delta + float(candidate_weight)*candidate)
    metrics = {
        "action_flow_mse": flow.detach(),
        "action_x0_l1": endpoint.detach(),
        "action_delta_l1": delta.detach(),
        "candidate_flow_mse": candidate.detach(),
        "candidate_loss_weight": float(candidate_weight),
        "action_horizon_10_flow_mse": flow.detach(),
        "action_horizon_10_x0_l1": endpoint.detach(),
        "action_horizon_10_delta_l1": delta.detach(),
    }
    if joint_inputs is not None:
        feedback = raw_model._last_predicted_future.detach()
        metrics.update({
            "candidate_velocity_mean": candidate_velocity.detach().mean(),
            "candidate_velocity_std": candidate_velocity.detach().float().std(),
            "feedback_future_mean": feedback.mean(),
            "feedback_future_std": feedback.float().std(),
            "feedback_future_abs_max": feedback.abs().max(),
        })
    return total, world_epsilon, metrics


@torch.no_grad()
def evaluate_actions_v2(model, dataset, stats, args, device):
    model = _unwrap(model)
    model.eval()
    prediction_rows, target_rows, baseline_rows = [], [], []
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    offset = 0
    for batch in loader:
        (previous, _target, actions, state, _image, _source,
         goal, proprio, past, past_mask) = [v.to(device, non_blocking=True) for v in batch]
        with _autocast(device, args.precision):
            prediction = sample_actions(
                model, previous, goal, proprio, past, past_mask, stats,
                args.flow_steps, args.seed + offset, world_state=state)
        target = actions[:, 0] if actions.ndim == 4 else actions
        fallback = torch.as_tensor(
            stats["action_raw_mean"], device=device, dtype=past.dtype)[None, None]
        repeat = torch.where(past_mask[:, -1:, None].bool(), past[:, -1:, :],
                             fallback).expand(-1, ACTION_HORIZON, -1)
        prediction_rows.append(prediction.float().cpu())
        target_rows.append(target.float().cpu())
        baseline_rows.append(repeat.float().cpu())
        offset += len(previous)
    prediction, target, repeat = map(torch.cat,
                                     (prediction_rows, target_rows, baseline_rows))
    error = (prediction-target).abs()
    per_sample = error.mean((1, 2))
    first_per_sample = error[:, :10].mean((1, 2))
    repeat_error = (repeat-target).abs()
    repeat_per_sample = repeat_error.mean((1, 2))
    repeat_first_per_sample = repeat_error[:, :10].mean((1, 2))
    used = len(target)
    low = torch.from_numpy(
        dataset.clearance[:used] <= float(stats["clearance_q25"]))
    turning = torch.from_numpy(
        dataset.turn_score[:used] >= float(stats["turn_score_q75"]))
    summary = summarize_action(prediction, target)
    summary.update({
        "samples": used,
        "low_clearance_mae": float(per_sample[low].mean()),
        "turning_mae": float(per_sample[turning].mean()),
        "low_clearance_first_10_mae": float(first_per_sample[low].mean()),
        "turning_first_10_mae": float(first_per_sample[turning].mean()),
        "low_clearance_repeat_last_mae": float(repeat_per_sample[low].mean()),
        "turning_repeat_last_mae": float(repeat_per_sample[turning].mean()),
        "low_clearance_repeat_last_first_10_mae": float(
            repeat_first_per_sample[low].mean()),
        "turning_repeat_last_first_10_mae": float(
            repeat_first_per_sample[turning].mean()),
        "repeat_last_baseline_mae": float(repeat_error.mean()),
        "repeat_last_first_10_mae": float(repeat_error[:, :10].mean()),
        "mean_action_baseline_mae": float((
            torch.as_tensor(stats["action_raw_mean"])[None, None] - target
        ).abs().mean()),
    })
    summary["selection_score"] = (
        0.40*summary["first_10_mae"]
        + 0.25*summary["low_clearance_first_10_mae"]
        + 0.25*summary["turning_first_10_mae"]
        + 0.10*summary["action_mae"])
    summary["relative_improvement_vs_repeat_last"] = (
        1-summary["action_mae"]/max(summary["repeat_last_baseline_mae"], 1e-8))
    return summary


def _action_validation_gate(summary):
    """Hard gate for the commands that closed-loop control really executes."""
    improvements = {
        "first_10_improvement": 1-summary["first_10_mae"]/max(
            summary["repeat_last_first_10_mae"], 1e-8),
        "low_clearance_first_10_improvement": (
            1-summary["low_clearance_first_10_mae"]/max(
                summary["low_clearance_repeat_last_first_10_mae"], 1e-8)),
        "turning_first_10_improvement": (
            1-summary["turning_first_10_mae"]/max(
                summary["turning_repeat_last_first_10_mae"], 1e-8)),
    }
    improvements["passed"] = min(improvements.values()) >= 0.10-1e-8
    improvements["criterion"] = (
        "first-10, low-clearance first-10, and turning first-10 MAE must each "
        "improve at least 10% over repeat-last")
    return improvements


def _load_world_v2(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if any(key.startswith("world.") for key in state):
        state = {key[len("world."):]: value for key, value in state.items()
                 if key.startswith("world.")}
    model = DirectHorizonWorldModel()
    # Older direct-horizon checkpoints predate this non-learned conditioning
    # buffer.  Preserve the checkpoint's learnable tensors and use the model
    # default only for that deterministic compatibility field.
    expected = model.state_dict()
    if "condition.horizon" not in state and "condition.horizon" in expected:
        state = dict(state)
        state["condition.horizon"] = expected["condition.horizon"]
    model.load_state_dict(state, strict=True)
    return model, int(payload.get("step", -1))


def _make_v2_model(args):
    if args.command == "train-action":
        return ActionOnlyModel(
            args.action_width, args.action_depth, args.action_heads,
            args.action_ffn_width), -1
    world_payload = torch.load(
        args.world_checkpoint, map_location="cpu", weights_only=False)
    world_gate = world_payload.get("validation", {}).get("gate", {})
    if not bool(world_gate.get("passed", False)) and not args.allow_world_gate_failure:
        raise RuntimeError(
            "World checkpoint did not pass the required 10% copy improvement "
            "and 5% shuffled-action degradation gate; pass "
            "--allow-world-gate-failure only for an explicitly audited "
            "experimental joint run")
    world, world_step = _load_world_v2(args.world_checkpoint)
    model = JointWorldActionModel(
        world, width=args.action_width, depth=args.action_depth,
        heads=args.action_heads, ffn_width=args.action_ffn_width,
        future_attention_layers=args.future_attention_layers)
    action_payload = torch.load(
        args.action_checkpoint, map_location="cpu", weights_only=False)
    action_state = action_payload["model"]
    selected = {key: value for key, value in action_state.items()
                if key.startswith(("observation.", "action_expert."))}
    position_key = "action_expert.action_position"
    if (position_key in selected
            and selected[position_key].shape[1] != ACTION_HORIZON):
        selected[position_key] = selected[position_key][:, :ACTION_HORIZON].clone()
    candidate_missing = not any(
        key.startswith("action_expert.candidate_") for key in selected)
    incompatible = model.load_state_dict(selected, strict=False)
    allowed = {
        key for key in model.state_dict()
        if (key.startswith(("world.", "observation_to_world.",
                            "future_adapter.", "action_semantic_adapter."))
            or ".future_attn." in key
            or ".norm_future." in key
            or key.endswith(".future_gate")
            or (candidate_missing
                and key.startswith("action_expert.candidate_")))
    }
    if set(incompatible.missing_keys) != allowed or incompatible.unexpected_keys:
        raise ValueError(
            f"action checkpoint merge mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}")
    if candidate_missing:
        model.action_expert.initialize_candidate_head()
    return model, world_step


def _v2_optimizer(model, args):
    raw = _unwrap(model)
    fusion_parameters = []
    if isinstance(raw, JointWorldActionModel):
        fusion_parameters.extend(raw.future_adapter.parameters())
        fusion_parameters.extend(raw.action_semantic_adapter.parameters())
        fusion_parameters.extend(raw.action_expert.candidate_norm.parameters())
        fusion_parameters.extend(raw.action_expert.candidate_out.parameters())
        for block in raw.action_expert.blocks:
            if block.future_attention:
                fusion_parameters.extend(block.norm_future.parameters())
                fusion_parameters.extend(block.future_attn.parameters())
                fusion_parameters.append(block.future_gate)
    fusion_ids = {id(parameter) for parameter in fusion_parameters}
    action_parameters = [
        parameter for parameter in raw.action_expert.parameters()
        if parameter.requires_grad and id(parameter) not in fusion_ids]
    groups = [
        {"params": action_parameters, "lr": args.action_lr,
         "name": "action"},
        {"params": raw.observation.parameters(), "lr": args.shared_lr,
         "name": "observation"},
    ]
    if isinstance(raw, JointWorldActionModel):
        groups.extend([
            {"params": raw.observation_to_world.parameters(), "lr": args.shared_lr,
             "name": "observation_to_world"},
            {"params": fusion_parameters, "lr": args.fusion_lr,
             "name": "future_adapter"},
            # Keeping the parameters trainable avoids changing DDP's graph;
            # the step schedule freezes the pretrained UNet with LR=0 first.
            {"params": raw.world.parameters(), "lr": 0.0, "name": "world"},
        ])
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _freeze_joint_world_branch(model):
    """Freeze future generation/fusion while preserving its differentiable path.

    Gradients can still flow through the fixed world model and future adapter
    into the provisional action and shared observation tokens.  Excluding the
    fixed parameters from autograd keeps their gradients out of global clipping
    and the divergence guard.
    """
    if not isinstance(model, JointWorldActionModel):
        raise TypeError("world-branch freezing requires JointWorldActionModel")
    model.world.requires_grad_(False)
    model.observation_to_world.requires_grad_(False)
    model.future_adapter.requires_grad_(False)
    model.action_semantic_adapter.requires_grad_(False)
    model.action_expert.candidate_norm.requires_grad_(False)
    model.action_expert.candidate_out.requires_grad_(False)
    for block in model.action_expert.blocks:
        if block.future_attention:
            block.norm_future.requires_grad_(False)
            block.future_attn.requires_grad_(False)
            block.future_gate.requires_grad_(False)


def _joint_schedule(step, args):
    """Return the exact staged world-loss weight and world-model LR."""
    if step <= args.world_freeze_steps:
        fraction = (step-1) / max(args.world_freeze_steps-1, 1)
        weight = args.world_weight_start + fraction * (
            args.world_weight_mid-args.world_weight_start)
        world_lr = 0.0
    elif step <= args.world_ramp_end:
        fraction = (step-args.world_freeze_steps) / max(
            args.world_ramp_end-args.world_freeze_steps, 1)
        weight = args.world_weight_mid + fraction * (
            args.world_weight-args.world_weight_mid)
        world_lr = args.world_lr
    else:
        weight, world_lr = args.world_weight, args.world_lr
    return float(weight), float(world_lr)


def _cosine_lr_scale(step, args):
    """Cosine-decay multiplier for stable long joint fine-tuning."""
    start = int(getattr(args, "lr_decay_start", 0))
    minimum = float(getattr(args, "lr_min_scale", 1.0))
    if start <= 0 or step <= start:
        return 1.0
    progress = min(max((step-start) / max(args.steps-start, 1), 0.0), 1.0)
    return minimum + 0.5 * (1.0-minimum) * (1.0+math.cos(math.pi*progress))


def _set_optimizer_lr(optimizer, name, value):
    matches = [group for group in optimizer.param_groups
               if group.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one optimizer group named {name!r}")
    matches[0]["lr"] = float(value)


def _parameter_grad_norm(parameters):
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt())


def _tensorboard_write(writer, prefix, value, step):
    """Recursively write numeric training/validation values to TensorBoard."""
    if writer is None:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _tensorboard_write(writer, f"{prefix}/{key}", child, step)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _tensorboard_write(writer, f"{prefix}/{index}", child, step)
    elif isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
        writer.add_scalar(prefix, float(value), int(step))


class _JointWorldEvaluationView(torch.nn.Module):
    """Expose only the joint model's world branch to the standard evaluator."""

    def __init__(self, joint):
        super().__init__()
        self.joint = joint
        self._observation = None

    def prepare(self, previous):
        self._observation = self.joint.encode_current(previous)

    def forward(self, noisy, previous, actions, state, timestep):
        if self._observation is None or len(self._observation) != len(previous):
            raise RuntimeError("world evaluation view was not prepared for this batch")
        # The shared evaluator still allocates three horizon slots.  Only t+1
        # is part of this model; leave the other slots inert and discard their
        # metrics below.
        epsilon = self.joint.predict_world(
            noisy[:, 0], previous, actions[:, 0], state, timestep,
            observation_tokens=self._observation)
        result = torch.zeros_like(noisy)
        result[:, 0] = epsilon
        return result


def _combined_validation(model, vae, action_dataset, world_dataset,
                         stats, args, device):
    action = evaluate_actions_v2(model, action_dataset, stats, args, device)
    action_gate = _action_validation_gate(action)
    result = {"action": action, "action_gate": action_gate}
    if isinstance(_unwrap(model), JointWorldActionModel):
        world = evaluate_world_v2(
            _JointWorldEvaluationView(_unwrap(model)), vae,
            WorldV2View(world_dataset), args, device)
        t1 = world["by_horizon"]["t+1"]
        world["prediction"] = t1["prediction"]
        world["copy"] = t1["copy"]
        world["shuffled"] = t1["shuffled"]
        world["evaluated_horizons"] = ["t+1"]
        prediction_cd = float(t1["prediction"]["cd_paper_m2"])
        copy_cd = float(t1["copy"]["cd_paper_m2"])
        shuffled_cd = float(t1["shuffled"]["cd_paper_m2"])
        world["gate"] = {
            "metric": "cd_paper_m2",
            "prediction_improvement": 1-prediction_cd/max(copy_cd, 1e-8),
            "shuffle_degradation": shuffled_cd/max(prediction_cd, 1e-8)-1,
        }
        world["gate"]["passed"] = (
            world["gate"]["prediction_improvement"] >= 0.10
            and world["gate"]["shuffle_degradation"] >= 0.05)
        result["world"] = world
        result["passed"] = bool(action_gate["passed"] and world["gate"]["passed"])
    else:
        result["passed"] = bool(action_gate["passed"])
    result["selection_score"] = float(action["selection_score"])
    return result


def _checkpoint_v2(path, model, optimizer, step, stats, args, world_step,
                   validation, best_score, validation_step=None,
                   best_chamfer_m2=math.inf):
    is_joint = isinstance(_unwrap(model), JointWorldActionModel)
    payload = {
        "format": (JOINT_TRAINING_FORMAT_V5 if is_joint else TRAINING_FORMAT_V2),
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(), "step": int(step), "stats": stats,
        "world_initial_step": int(world_step), "validation": validation,
        "validation_step": int(
            step if validation_step is None else validation_step),
        "best_selection_score": float(best_score),
        # The value is meaningful for joint runs.  Keep it in every checkpoint
        # so latest/best snapshots can be compared without external logs.
        "best_prediction_cd_paper_m2": float(best_chamfer_m2),
        "semantics": coordinate_semantics(),
        "architecture": {
            "action_horizon": ACTION_HORIZON, "action_dim": ACTION_DIM,
            "width": args.action_width, "depth": args.action_depth,
            "heads": args.action_heads, "ffn_width": args.action_ffn_width,
            "future_attention_layers": (
                args.future_attention_layers if is_joint else 0),
            "world_horizon": 1 if is_joint else 0,
            "bidirectional_semantic_fusion": bool(is_joint),
        },
        "training_schedule": {
            "world_freeze_steps": getattr(args, "world_freeze_steps", 0),
            "world_ramp_end": getattr(args, "world_ramp_end", 0),
            "world_weight_start": getattr(args, "world_weight_start", 0.0),
            "world_weight_mid": getattr(args, "world_weight_mid", 0.0),
            "world_weight_final": getattr(args, "world_weight", 0.0),
            "fusion_warmup_steps": getattr(args, "fusion_warmup_steps", 0),
            "candidate_weight": getattr(args, "candidate_weight", 0.0),
            "feedback_ddim_steps": getattr(args, "feedback_ddim_steps", 0),
            "feedback_gradient_steps": getattr(
                args, "feedback_gradient_steps", 0),
            "action_lr": args.action_lr,
            "shared_lr": args.shared_lr,
            "fusion_lr": args.fusion_lr,
            "world_lr": args.world_lr,
            "lr_decay_start": args.lr_decay_start,
            "lr_min_scale": args.lr_min_scale,
            "max_preclip_grad_norm": args.max_preclip_grad_norm,
            "reset_optimizer": bool(args.reset_optimizer),
        },
        "provenance": {
            "dataset_manifest_sha256": file_sha256(args.dataset_root / "manifest.json"),
            "latent_metadata_sha256": file_sha256(args.latent_root / "metadata.json"),
            "world_checkpoint_sha256": _sha_or_none(getattr(args, "world_checkpoint", None)),
            "action_checkpoint_sha256": _sha_or_none(getattr(args, "action_checkpoint", None)),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _candidate_policy_v2(path, model, step, stats, args, validation):
    state = _unwrap(model).state_dict()
    selected = {key: value for key, value in state.items()
                if key.startswith(("observation.", "action_expert."))}
    payload = {
        "format": "navrl-action-expert-policy-v2", "model": selected,
        "step": int(step), "stats": stats, "validation": validation,
        "semantics": coordinate_semantics(),
        "architecture": {
            "action_horizon": ACTION_HORIZON, "action_dim": ACTION_DIM,
            "width": args.action_width, "depth": args.action_depth,
            "heads": args.action_heads, "ffn_width": args.action_ffn_width,
        },
        "provenance": {
            "dataset_manifest_sha256": file_sha256(args.dataset_root/"manifest.json"),
            "latent_metadata_sha256": file_sha256(args.latent_root/"metadata.json"),
            "world_checkpoint_sha256": _sha_or_none(getattr(args, "world_checkpoint", None)),
        },
    }
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _candidate_joint_policy_v2(path, model, step, stats, args, validation):
    raw = _unwrap(model)
    if not isinstance(raw, JointWorldActionModel):
        return
    payload = {
        "format": JOINT_POLICY_FORMAT_V5,
        "model": raw.state_dict(), "step": int(step), "stats": stats,
        "validation": validation, "semantics": coordinate_semantics(),
        "architecture": {
            "action_horizon": ACTION_HORIZON, "action_dim": ACTION_DIM,
            "width": args.action_width, "depth": args.action_depth,
            "heads": args.action_heads, "ffn_width": args.action_ffn_width,
            "world_horizon": 1,
            "future_attention_layers": args.future_attention_layers,
            "joint_denoising": True,
            "bidirectional_semantic_fusion": True,
        },
        "provenance": {
            "dataset_manifest_sha256": file_sha256(args.dataset_root/"manifest.json"),
            "latent_metadata_sha256": file_sha256(args.latent_root/"metadata.json"),
            "world_checkpoint_sha256": _sha_or_none(args.world_checkpoint),
            "action_checkpoint_sha256": _sha_or_none(args.action_checkpoint),
        },
    }
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _update_candidates(run_dir, score, step, checkpoint, joint_checkpoint=None,
                       keep=4):
    path = run_dir / "candidates.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows = [row for row in rows if int(row["step"]) != int(step)]
    rows.append({"step": int(step), "selection_score": float(score),
                 "checkpoint": checkpoint.name,
                 "joint_checkpoint": (joint_checkpoint.name
                                      if joint_checkpoint is not None else None)})
    rows.sort(key=lambda row: row["selection_score"])
    removed, rows = rows[keep:], rows[:keep]
    for row in removed:
        candidate = run_dir / row["checkpoint"]
        if candidate.exists() and candidate != checkpoint:
            candidate.unlink()
        joint_name = row.get("joint_checkpoint")
        if joint_name:
            joint_candidate = run_dir / joint_name
            if joint_candidate.exists() and joint_candidate != joint_checkpoint:
                joint_candidate.unlink()
    stage1.save_json(path, rows)


def train_v2(args):
    distributed, rank, world_size, local_rank, device = _distributed_runtime(args.precision)
    is_main = rank == 0
    stage1.seed_everything(args.seed + rank)
    run_name = (args.run_name or (
        "action_expert_action_only_v2" if args.command == "train-action"
        else "action_expert_joint_v2_future8500"))
    if args.overfit:
        run_name += "_overfit"
    run_dir = args.out / run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    train_data = _v2_dataset("train", args, overfit=args.overfit)
    # Validation limits are global random-window counts, not per-seed counts.
    # Keeping the seeded subset fixed across validation rounds makes the
    # TensorBoard curves comparable while avoiding a full validation scan.
    val_data = _v2_dataset(
        "train" if args.overfit else "val", args, overfit=args.overfit,
        limit=args.eval_samples)
    world_val_data = val_data
    if args.command == "train-joint":
        world_val_data = _v2_dataset(
            "train" if args.overfit else "val", args, overfit=args.overfit,
            limit=args.world_eval_samples)
    stats = _broadcast_stats(train_data, rank)
    # ``compute_condition_stats`` opens HDF5 handles on rank zero.  Close them
    # before DataLoader workers are forked; workers reopen independent handles.
    train_data.close()
    model, world_step = _make_v2_model(args)
    if (args.command == "train-joint"
            and getattr(args, "freeze_world_branch", False)):
        _freeze_joint_world_branch(model)
    model = model.to(device).float()
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                         output_device=local_rank,
                                         broadcast_buffers=False)
    optimizer = _v2_optimizer(model, args)
    first_step, best_score, best_accepted_score = 1, math.inf, math.inf
    best_chamfer_m2 = math.inf
    last_validation, last_validation_step = {}, -1
    resume_path = getattr(args, "resume_from", None)
    if args.resume and resume_path is None:
        resume_path = run_dir / "latest.pt"
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        resume_state = payload["model"]
        if args.command == "train-action":
            # A joint checkpoint is the usual source for an Action Expert
            # fine-tune.  Deliberately retain only the deployment-visible
            # current-observation/action branch; world and future tokens must
            # not leak into an action-only run.
            target_keys = set(_unwrap(model).state_dict())
            if set(resume_state) != target_keys:
                resume_state = {
                    key: value for key, value in resume_state.items()
                    if key in target_keys
                }
            position_key = "action_expert.action_position"
            if (position_key in resume_state
                    and resume_state[position_key].shape[1] != ACTION_HORIZON):
                resume_state = dict(resume_state)
                resume_state[position_key] = resume_state[position_key][
                    :, :ACTION_HORIZON].clone()
        if args.command == "train-action":
            incompatible = _unwrap(model).load_state_dict(
                resume_state, strict=False)
            allowed_missing = {
                key for key in _unwrap(model).state_dict()
                if key.startswith("action_expert.candidate_")
            }
            if (set(incompatible.missing_keys) not in (set(), allowed_missing)
                    or incompatible.unexpected_keys):
                raise ValueError(
                    f"action resume mismatch: missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}")
            if set(incompatible.missing_keys) == allowed_missing:
                _unwrap(model).action_expert.initialize_candidate_head()
        else:
            _unwrap(model).load_state_dict(resume_state, strict=True)
        if not args.reset_optimizer:
            optimizer.load_state_dict(payload["optimizer"])
        # Keep the model weights when the physical data distribution changes,
        # but let the new train split define its own goal/proprio/action
        # normalization.  This is essential for randomized-terrain fine-tunes.
        if not args.reset_stats:
            stats = payload["stats"]
        first_step = int(payload["step"])+1
        best_score = float(payload.get("best_selection_score", math.inf))
        best_chamfer_m2 = float(payload.get(
            "best_prediction_cd_paper_m2", math.inf))
        last_validation = payload.get("validation", {})
        last_validation_step = int(payload.get(
            "validation_step", payload.get("step", -1)))
        best_accepted_score = best_score
        if args.reset_best_score:
            # Start a run-local model-selection history while retaining the
            # resumed model/optimizer/step.  This is useful when the training
            # distribution changes: the previous checkpoint remains intact,
            # while best.pt in the new run records the best validation result
            # obtained on the new distribution.
            best_score = math.inf
            best_accepted_score = math.inf
            best_chamfer_m2 = math.inf

    config = {
            "format": (JOINT_TRAINING_FORMAT_V5
                       if args.command == "train-joint"
                       else TRAINING_FORMAT_V2),
            "dataset_root": str(args.dataset_root), "latent_root": str(args.latent_root),
            "index_root": str(args.index_root), "steps": args.steps,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation": args.grad_accumulation,
            "checkpoint_every": args.checkpoint_every,
            "resume_from": str(resume_path) if resume_path is not None else None,
            "reset_optimizer": bool(args.reset_optimizer),
            "reset_best_score": bool(args.reset_best_score),
            "reset_stats": bool(args.reset_stats),
            "world_size": world_size,
            "global_batch_size": args.micro_batch_size*args.grad_accumulation*world_size,
            "precision": args.precision, "semantics": coordinate_semantics(),
            "action_validation_samples": len(val_data),
            "world_validation_samples": len(world_val_data),
            "validation_sampling": {
                "method": "global_random_without_replacement",
                "fixed_across_validations": True,
                "random_seed": args.seed,
            },
            "sampling": {"uniform": 0.5, "low_clearance": 0.25, "turning": 0.25},
            "action_loss": {
                "flow_weight": 1.0,
                "x0_weight": args.action_x0_weight,
                "delta_weight": args.action_delta_weight,
                "candidate_weight": args.candidate_weight,
                "horizon": "commands 1-10",
            },
            "selection": (
                "0.40*first10 + 0.25*low_clearance_first10 + "
                "0.25*turning_first10 + 0.10*overall"),
            "checkpoint_selection": {
                "best.pt": "lowest action validation selection_score",
                "best_chamfer_m2.pt": (
                    "lowest joint validation world.prediction.cd_paper_m2"),
                "latest.pt": "most recently saved optimizer state",
            },
            "optimizer_stability": {
                "action_lr": args.action_lr,
                "shared_lr": args.shared_lr,
                "fusion_lr": args.fusion_lr,
                "world_lr": args.world_lr,
                "freeze_world_branch": bool(getattr(
                    args, "freeze_world_branch", False)),
                "lr_decay_start": args.lr_decay_start,
                "lr_min_scale": args.lr_min_scale,
                "max_preclip_grad_norm": args.max_preclip_grad_norm,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
            },
            "joint_schedule": ({
                "world_lr": args.world_lr,
                "world_lr_zero_through_step": args.world_freeze_steps,
                "world_weight_start": args.world_weight_start,
                "world_weight_mid": args.world_weight_mid,
                "world_weight_final": args.world_weight,
                "world_weight_ramp_end": args.world_ramp_end,
                "future_attention_layers": args.future_attention_layers,
                "fusion_warmup_steps": args.fusion_warmup_steps,
                "allow_world_gate_failure": bool(
                    args.allow_world_gate_failure),
                "future_targets": ["t+1"],
                "action_targets": ["1:10"],
                "coupling": (
                    "candidate action tokens -> single-future UNet; pure-noise "
                    "DDIM prediction -> final action blocks"),
                "feedback_ddim_steps": args.feedback_ddim_steps,
                "feedback_gradient_steps": args.feedback_gradient_steps,
            } if args.command == "train-joint" else None),
            "normalization": stats,
        }
    if is_main:
        stage1.save_json(run_dir / "config.json", config)

    writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard logging requires the tensorboard package in the "
                "training environment") from error
        writer = SummaryWriter(
            log_dir=str(run_dir / "tensorboard"),
            purge_step=(first_step if args.resume and args.resume_from is None
                        else None))
        writer.add_text(
            "run/config_json", json.dumps(config, indent=2, sort_keys=True),
            global_step=max(first_step-1, 0))

    sampler = StratifiedSampler(
        train_data,
        num_samples=max(args.micro_batch_size * args.grad_accumulation * 100,
                        len(train_data) // world_size),
        seed=args.seed, rank=rank)
    loader = stage1.infinite(DataLoader(
        train_data, batch_size=args.micro_batch_size, sampler=sampler,
        drop_last=True, num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0))
    vae = None
    noise_scheduler = None
    feedback_scheduler = None
    if args.command == "train-joint":
        vae = stage1.load_circular_vae()
        vae.requires_grad_(False)
        noise_scheduler = stage1.DDPMScheduler(
            num_train_timesteps=1000, prediction_type="epsilon")
        feedback_scheduler = stage1.DDIMScheduler(
            num_train_timesteps=1000, prediction_type="epsilon",
            clip_sample=False)
        feedback_scheduler.set_timesteps(
            args.feedback_ddim_steps, device=device)
        feedback_timesteps = feedback_scheduler.timesteps
        feedback_alphas = feedback_scheduler.alphas_cumprod.to(device)[
            feedback_timesteps]
        feedback_previous_alphas = torch.cat((
            feedback_alphas[1:], torch.ones(
                1, device=device, dtype=feedback_alphas.dtype)))

    history_path = run_dir/"history.json"
    history = (json.loads(history_path.read_text())
               if args.resume and args.resume_from is None
               and history_path.exists() else [])
    started = time.monotonic()
    validations_without_improvement = 0
    try:
        for step in range(first_step, args.steps+1):
            model.train()
            if (args.command == "train-joint"
                    and args.freeze_world_branch):
                _unwrap(model).world.eval()
            optimizer.zero_grad(set_to_none=True)
            totals = defaultdict(float)
            scheduled_world_weight = 0.0
            world_lr = 0.0
            lr_scale = _cosine_lr_scale(step, args)
            if args.command == "train-joint":
                scheduled_world_weight, world_lr = _joint_schedule(step, args)
                world_lr *= lr_scale
                if args.freeze_world_branch:
                    # Keep the joint forward path active so the Action Expert
                    # still consumes predicted-future tokens, but prevent the
                    # pretrained world/fusion parameters and auxiliary world
                    # objective from drifting during stabilization.
                    scheduled_world_weight = 0.0
                    world_lr = 0.0
                _set_optimizer_lr(optimizer, "world", world_lr)
                warmup = step <= args.fusion_warmup_steps
                _set_optimizer_lr(
                    optimizer, "action",
                    0.0 if warmup else args.action_lr*lr_scale)
                _set_optimizer_lr(
                    optimizer, "observation",
                    0.0 if warmup else args.shared_lr*lr_scale)
                _set_optimizer_lr(
                    optimizer, "observation_to_world",
                    0.0 if args.freeze_world_branch else (
                        args.fusion_lr*lr_scale if warmup
                        else args.shared_lr*lr_scale))
                _set_optimizer_lr(
                    optimizer, "future_adapter",
                    0.0 if args.freeze_world_branch
                    else args.fusion_lr*lr_scale)
            else:
                _set_optimizer_lr(optimizer, "action", args.action_lr*lr_scale)
                _set_optimizer_lr(
                    optimizer, "observation", args.shared_lr*lr_scale)
            for micro_step in range(args.grad_accumulation):
                batch = [value.to(device, non_blocking=True) for value in next(loader)]
                (previous, target, actions, state, target_image, _source,
                 goal, proprio, past, past_mask) = batch
                sync = micro_step == args.grad_accumulation-1
                no_sync = contextlib.nullcontext() if sync or not distributed else model.no_sync()
                with no_sync, _autocast(device, args.precision):
                    joint_inputs = None
                    noise = timesteps = noisy_future = None
                    if args.command == "train-joint":
                        target = target[:, 0]
                        target_image = target_image[:, 0]
                        actions = actions[:, 0]
                        noise = torch.randn_like(target)
                        timesteps = torch.randint(
                            0, 1000, (len(target),), device=device)
                        noisy_future = noise_scheduler.add_noise(
                            target, noise, timesteps)
                        joint_inputs = {
                            "noisy_future": noisy_future, "world_actions": actions,
                            "world_state": state, "world_timestep": timesteps,
                            "world_alpha_cumprod": (
                                noise_scheduler.alphas_cumprod.to(device)[timesteps]),
                            "feedback_noise": torch.randn_like(target),
                            "feedback_timesteps": feedback_timesteps,
                            "feedback_alphas": feedback_alphas,
                            "feedback_previous_alphas": feedback_previous_alphas,
                            "feedback_gradient_steps": args.feedback_gradient_steps,
                            "action_logit_mean": torch.as_tensor(
                                stats["action_logit_mean"], device=device),
                            "action_logit_std": torch.as_tensor(
                                stats["action_logit_std"], device=device),
                        }
                    action_loss, epsilon, action_parts = _v2_action_terms(
                        model, previous, actions, goal, proprio, past,
                        past_mask, stats, joint_inputs,
                        args.action_x0_weight, args.action_delta_weight,
                        args.candidate_weight)
                    world_loss = action_loss*0
                    world_parts = {}
                    if epsilon is not None:
                        epsilon_loss = F.mse_loss(epsilon, noise)
                        zero = epsilon_loss*0
                        latent = mask = empty_mask = range_loss = presence = empty = zero
                        eligible = select_aux_indices(
                            timesteps, target_image, args.aux_t_max,
                            args.aux_batch_max, args.aux_empty_max)
                        if len(eligible):
                            x0 = predicted_x0(
                                noisy_future[eligible], epsilon[eligible],
                                timesteps[eligible], noise_scheduler)
                            latent = F.l1_loss(x0, target[eligible])
                            decoded = decoded_losses(
                                vae, x0, target_image[eligible],
                                train_data.scale, 1.5, 8, 0.5)
                            mask, empty_mask, range_loss = decoded[:3]
                            presence, empty = decoded[3:5]
                        world_loss = (epsilon_loss + 0.1*latent + 0.1*mask
                                      + 0.1*empty_mask + 0.05*range_loss
                                      + 0.02*presence + 0.2*empty)
                        world_parts = {
                            "world_epsilon_mse": epsilon_loss.detach(),
                            "world_x0_l1": latent.detach(),
                            "world_mask_bce": mask.detach(),
                            "world_empty_mask_bce": empty_mask.detach(),
                            "world_range_l1": range_loss.detach(),
                            "world_presence_loss": presence.detach(),
                            "world_empty_loss": empty.detach(),
                        }
                    loss = (args.action_weight*action_loss
                            + scheduled_world_weight*world_loss) / args.grad_accumulation
                if (args.command == "train-joint"
                        and (step == 1 or step % args.log_every == 0)
                        and micro_step == args.grad_accumulation-1):
                    shared_gradient = torch.autograd.grad(
                        scheduled_world_weight*world_loss,
                        _unwrap(model)._last_observation_tokens,
                        retain_graph=True, allow_unused=False)[0]
                    totals["world_to_shared_grad_norm"] = float(
                        shared_gradient.detach().float().norm())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                loss.backward()
                totals["loss"] += float(loss.detach())
                totals["action_loss"] += float(action_loss.detach())/args.grad_accumulation
                totals["world_loss"] += float(world_loss.detach())/args.grad_accumulation
                for key, value in {**action_parts, **world_parts}.items():
                    totals[key] += float(value)/args.grad_accumulation
            # LR-zero groups remain in the graph so gradients can flow through
            # pretrained modules into the new fusion adapters.  They must not
            # dominate clipping during fusion warmup, however.
            active_parameters = [
                parameter
                for group in optimizer.param_groups if float(group["lr"]) > 0
                for parameter in group["params"]
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not active_parameters:
                raise RuntimeError("no optimizer parameters have a positive learning rate")
            grad = torch.nn.utils.clip_grad_norm_(
                active_parameters, args.grad_clip)
            if not torch.isfinite(grad):
                raise FloatingPointError(f"non-finite gradient at step {step}")
            skip_update = bool(
                args.max_preclip_grad_norm > 0
                and float(grad) > args.max_preclip_grad_norm)
            if distributed:
                skip_flag = torch.tensor(
                    int(skip_update), device=device, dtype=torch.int32)
                dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
                skip_update = bool(skip_flag.item())
            adapter_grad_norm = 0.0
            if args.command == "train-joint":
                adapter_grad_norm = _parameter_grad_norm(
                    _unwrap(model).observation_to_world.parameters())
            if skip_update:
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()

            if is_main and (step == 1 or step % args.log_every == 0):
                record = {"step": step, **totals, "grad_norm": float(grad),
                          "optimizer_step_skipped": skip_update,
                          "elapsed_min": round((time.monotonic()-started)/60, 2),
                          "world_weight": scheduled_world_weight,
                          "world_lr": world_lr,
                          "lr_scale": lr_scale,
                          "fusion_warmup": bool(
                              args.command == "train-joint"
                              and step <= args.fusion_warmup_steps)}
                for group in optimizer.param_groups:
                    record[f"lr/{group.get('name', 'unnamed')}"] = float(group["lr"])
                if args.command == "train-joint":
                    raw_model = _unwrap(model)
                    future_attention_parameters = []
                    future_gates = [
                        float(block.future_gate.detach())
                        for block in raw_model.action_expert.blocks
                        if block.future_attention]
                    for block in raw_model.action_expert.blocks:
                        if block.future_attention:
                            future_attention_parameters.extend(
                                block.norm_future.parameters())
                            future_attention_parameters.extend(
                                block.future_attn.parameters())
                            future_attention_parameters.append(block.future_gate)
                    record.update({
                        "world_adapter_grad_norm": adapter_grad_norm,
                        "world_adapter_weight_norm": float(
                            raw_model.observation_to_world.weight.detach()
                            .float().norm()),
                        "future_adapter_grad_norm": _parameter_grad_norm(
                            raw_model.future_adapter.parameters()),
                        "action_semantic_adapter_grad_norm": _parameter_grad_norm(
                            raw_model.action_semantic_adapter.parameters()),
                        "action_semantic_gate": float(
                            raw_model.action_semantic_adapter.gate.detach()),
                        "candidate_head_grad_norm": _parameter_grad_norm([
                            *raw_model.action_expert.candidate_norm.parameters(),
                            *raw_model.action_expert.candidate_out.parameters(),
                        ]),
                        "future_attention_grad_norm": _parameter_grad_norm(
                            future_attention_parameters),
                        "future_attention_gates": future_gates,
                    })
                if device.type == "cuda":
                    record["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device)/2**30
                history.append(record)
                print(json.dumps(record), flush=True)
                _tensorboard_write(writer, "train", record, step)
            if args.memory_smoke:
                # Run more than one iteration so DDP bucket rebuilding and
                # optimizer-state allocation are included in the peak.
                if distributed:
                    dist.barrier()
                continue
            evaluate_now = step % args.eval_every == 0 or step == args.steps
            checkpoint_now = (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0)
            stop_training = False
            if evaluate_now:
                if distributed:
                    dist.barrier()
                if is_main:
                    validation = _combined_validation(
                        model, vae, val_data, world_val_data,
                        stats, args, device)
                    last_validation = validation
                    last_validation_step = step
                    score = float(validation["selection_score"])
                    print(json.dumps({"step": step, "validation": validation}), flush=True)
                    _tensorboard_write(writer, "validation", validation, step)
                    writer.flush()
                    improved = score < best_score-args.early_stop_min_delta
                    if improved:
                        best_score = score
                        validations_without_improvement = 0
                    else:
                        validations_without_improvement += 1
                    prediction_chamfer_m2 = (
                        float(validation["world"]["prediction"]["cd_paper_m2"])
                        if "world" in validation else math.inf)
                    if prediction_chamfer_m2 < best_chamfer_m2:
                        best_chamfer_m2 = prediction_chamfer_m2
                        _checkpoint_v2(
                            run_dir/"best_chamfer_m2.pt", model, optimizer,
                            step, stats, args, world_step, validation,
                            best_score, step, best_chamfer_m2)
                        if args.command == "train-joint":
                            _candidate_joint_policy_v2(
                                run_dir/"best_chamfer_joint_policy.pt", model,
                                step, stats, args, validation)
                        stage1.save_json(
                            run_dir/"best_chamfer_metrics.json",
                            {"step": step,
                             "prediction_cd_paper_m2": best_chamfer_m2,
                             **validation})
                        print(json.dumps({
                            "step": step,
                            "best_chamfer_checkpoint": str(
                                run_dir/"best_chamfer_m2.pt"),
                            "best_prediction_cd_paper_m2": best_chamfer_m2,
                        }), flush=True)
                    _checkpoint_v2(run_dir/"latest.pt", model, optimizer, step,
                                   stats, args, world_step, validation,
                                   best_score, step, best_chamfer_m2)
                    if validation["passed"]:
                        joint_candidate = None
                        if args.command == "train-joint":
                            candidate = run_dir/f"candidate_joint_step_{step:06d}.pt"
                            _candidate_joint_policy_v2(
                                candidate, model, step, stats, args, validation)
                        else:
                            candidate = run_dir/f"candidate_step_{step:06d}.pt"
                            _candidate_policy_v2(
                                candidate, model, step, stats, args, validation)
                        _update_candidates(
                            run_dir, score, step, candidate, joint_candidate)
                    retain_best = bool(score < best_accepted_score)
                    if retain_best:
                        best_accepted_score = score
                        _checkpoint_v2(run_dir/"best.pt", model, optimizer, step,
                                       stats, args, world_step, validation,
                                       best_score, step, best_chamfer_m2)
                        if args.command == "train-joint":
                            _candidate_joint_policy_v2(
                                run_dir/"best_joint_policy.pt", model, step,
                                stats, args, validation)
                        else:
                            _candidate_policy_v2(
                                run_dir/"best_action_policy.pt", model, step,
                                stats, args, validation)
                        stage1.save_json(run_dir/"best_metrics.json",
                                         {"step": step, **validation})
                        print(json.dumps({
                            "step": step,
                            "best_checkpoint": str(run_dir/"best.pt"),
                            "best_selection_score": best_accepted_score,
                            "hard_gate_passed": bool(validation["passed"]),
                        }), flush=True)
                    stop_training = bool(
                        args.early_stop_patience > 0
                        and validations_without_improvement
                        >= args.early_stop_patience)
                    if stop_training:
                        early_stop = {
                            "step": step,
                            "reason": "validation_selection_score_plateau",
                            "validations_without_improvement": (
                                validations_without_improvement),
                            "patience": args.early_stop_patience,
                            "min_delta": args.early_stop_min_delta,
                            "best_selection_score": best_score,
                            "current_selection_score": score,
                        }
                        stage1.save_json(run_dir/"early_stop.json", early_stop)
                        print(json.dumps({"early_stop": early_stop}), flush=True)
                    if args.overfit and step == args.steps:
                        gate = {"step": step, **validation}
                        stage1.save_json(run_dir/"overfit_gate.json", gate)
                        print(json.dumps({"overfit_gate": gate}), flush=True)
                    stage1.save_json(run_dir/"history.json", history)
                if distributed:
                    stop_flag = torch.tensor(
                        int(stop_training) if is_main else 0,
                        device=device, dtype=torch.int32)
                    dist.broadcast(stop_flag, src=0)
                    stop_training = bool(stop_flag.item())
                    dist.barrier()
            elif checkpoint_now:
                # Keep every rank at the same optimizer step while rank zero
                # serializes the multi-gigabyte recovery checkpoint. Without
                # the barriers, the other ranks can enter the next all-reduce
                # and time out while rank zero is still writing the file.
                if distributed:
                    dist.barrier()
                if is_main:
                    _checkpoint_v2(
                        run_dir/"latest.pt", model, optimizer, step, stats,
                        args, world_step, last_validation, best_score,
                        last_validation_step, best_chamfer_m2)
                    _tensorboard_write(writer, "checkpoint/step", step, step)
                    writer.flush()
                    print(json.dumps({
                        "step": step,
                        "checkpoint": str(run_dir/"latest.pt"),
                        "validation_step": last_validation_step,
                    }), flush=True)
                if distributed:
                    dist.barrier()
            if stop_training:
                break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        train_data.close()
        if val_data is not train_data:
            val_data.close()
        if world_val_data not in (train_data, val_data):
            world_val_data.close()
        if distributed:
            dist.destroy_process_group()


def evaluate_v2(args):
    _, rank, _, _, device = _distributed_runtime(args.precision)
    if rank != 0:
        return
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") not in (
            TRAINING_FORMAT_V2, JOINT_TRAINING_FORMAT_V3,
            JOINT_TRAINING_FORMAT_V4, JOINT_TRAINING_FORMAT_V5):
        raise ValueError("evaluate-v2 requires a supported action/joint checkpoint")
    architecture = payload["architecture"]
    if payload.get("format") == JOINT_TRAINING_FORMAT_V5:
        model = JointWorldActionModel(
            DirectHorizonWorldModel(), architecture["width"],
            architecture["depth"], architecture["heads"],
            architecture["ffn_width"],
            architecture.get("future_attention_layers", 2))
        model.load_state_dict(payload["model"], strict=True)
    elif payload.get("format") in (JOINT_TRAINING_FORMAT_V3,
                                   JOINT_TRAINING_FORMAT_V4):
        raise ValueError(
            "30-action/3-future joint checkpoints must be evaluated with the "
            "legacy runner; initialize a v5 model from their component weights")
    else:
        model = ActionOnlyModel(
            architecture["width"], architecture["depth"], architecture["heads"],
            architecture["ffn_width"])
        selected = {key: value for key, value in payload["model"].items()
                    if key.startswith(("observation.", "action_expert."))}
        position_key = "action_expert.action_position"
        if selected[position_key].shape[1] != ACTION_HORIZON:
            selected[position_key] = selected[position_key][:, :ACTION_HORIZON]
        incompatible = model.load_state_dict(selected, strict=False)
        allowed = {key for key in model.state_dict()
                   if key.startswith("action_expert.candidate_")}
        if set(incompatible.missing_keys) not in (set(), allowed):
            raise ValueError(f"action checkpoint mismatch: {incompatible}")
    model.to(device).eval()
    dataset = _v2_dataset(args.split, args, limit=args.eval_samples)
    summary = evaluate_actions_v2(model, dataset, payload["stats"], args, device)
    result = {"checkpoint": str(args.checkpoint), "step": payload["step"],
              "split": args.split, "summary": summary}
    stage1.save_json(args.out/"evaluation_v2"/f"{args.split}.json", result)
    print(json.dumps(result, indent=2), flush=True)


def _v2_common(parser):
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=stage1.OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--action-width", type=int, default=512)
    parser.add_argument("--action-depth", type=int, default=8)
    parser.add_argument("--action-heads", type=int, default=8)
    parser.add_argument("--action-ffn-width", type=int, default=2048)
    parser.add_argument("--future-attention-layers", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--eval-samples", type=int, default=1024,
        help="Total random windows used for Action validation (not per seed).")
    parser.add_argument(
        "--world-eval-samples", type=int, default=256,
        help="Total random windows used for Future validation (not per seed).")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--mask-threshold", type=float, default=1.5)


def _v2_train_arguments(parser, joint=False):
    _v2_common(parser)
    parser.add_argument("--steps", type=int, default=10000 if joint else 20000)
    parser.add_argument("--micro-batch-size", type=int, default=1 if joint else 4)
    parser.add_argument("--grad-accumulation", type=int, default=4 if joint else 8)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help=("Save latest.pt at this optimizer-step interval without running "
              "validation; zero saves only on validation steps."))
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--run-name")
    parser.add_argument("--action-lr", type=float, default=3e-5 if joint else 1e-4)
    parser.add_argument("--shared-lr", type=float, default=1e-5 if joint else 5e-5)
    parser.add_argument("--fusion-lr", type=float, default=1e-4)
    parser.add_argument("--world-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--max-preclip-grad-norm", type=float, default=0.0,
        help=("Skip an optimizer update when the gradient norm before clipping "
              "exceeds this value; zero disables the guard."))
    parser.add_argument(
        "--lr-decay-start", type=int, default=0,
        help=("Optimizer step where cosine LR decay begins; zero disables "
              "decay."))
    parser.add_argument(
        "--lr-min-scale", type=float, default=0.1,
        help="Final cosine-decay learning-rate multiplier.")
    parser.add_argument(
        "--early-stop-patience", type=int, default=0,
        help=("Stop after this many consecutive validations without a lower "
              "selection score; zero disables early stopping."))
    parser.add_argument(
        "--early-stop-min-delta", type=float, default=0.0,
        help="Minimum selection-score decrease counted as an improvement.")
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--world-weight", type=float, default=0.25 if joint else 0.0)
    parser.add_argument("--world-weight-start", type=float, default=0.05)
    parser.add_argument("--world-weight-mid", type=float, default=0.15)
    parser.add_argument("--world-freeze-steps", type=int, default=1000)
    parser.add_argument("--world-ramp-end", type=int, default=1500)
    parser.add_argument("--action-x0-weight", type=float, default=0.0)
    parser.add_argument("--action-delta-weight", type=float, default=0.20)
    parser.add_argument(
        "--candidate-weight", type=float, default=0.10,
        help="Auxiliary flow loss weight for the pre-fusion candidate head.")
    parser.add_argument("--aux-t-max", type=int, default=500)
    parser.add_argument("--aux-batch-max", type=int, default=64)
    parser.add_argument("--aux-empty-max", type=int, default=16)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-from", type=Path,
        help="Resume model/statistics/step from an explicit checkpoint path.")
    parser.add_argument(
        "--reset-optimizer", action="store_true",
        help="When resuming, initialize a fresh AdamW optimizer state.")
    parser.add_argument(
        "--reset-best-score", action="store_true",
        help=("When resuming, keep model/optimizer/step but start a new "
              "run-local best.pt selection history."))
    parser.add_argument(
        "--reset-stats", action="store_true",
        help=("When resuming on a changed dataset, retain model weights but "
              "recompute normalization statistics from the new train split."))
    parser.add_argument(
        "--memory-smoke", action="store_true",
        help="Run steady-state optimizer steps and skip validation/checkpoints.")
    parser.add_argument("--memory-smoke-steps", type=int, default=2)
    if joint:
        parser.add_argument("--action-checkpoint", type=Path, required=True)
        parser.add_argument("--world-checkpoint", type=Path, required=True)
        parser.add_argument(
            "--allow-world-gate-failure", action="store_true",
            help=("Allow an experimental joint run from a world checkpoint that "
                  "failed its offline world-model quality gate. The override is "
                  "recorded in the run configuration."))
        parser.add_argument(
            "--fusion-warmup-steps", type=int, default=1000,
            help=("Only train FutureTokenAdapter and future cross-attention "
                  "during this zero-gated warmup."))
        parser.add_argument(
            "--feedback-ddim-steps", type=int, default=4,
            help="Pure-noise DDIM steps used by the action-feedback path.")
        parser.add_argument(
            "--feedback-gradient-steps", type=int, default=1,
            help="Final feedback DDIM steps retained for backpropagation.")
        parser.add_argument(
            "--freeze-world-branch", action="store_true",
            help=("Keep joint future-token conditioning active while freezing "
                  "the world UNet, world adapter, and future-fusion weights; "
                  "also disable the auxiliary world loss."))


def _common(parser):
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--raw-data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=stage1.OUT)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--samples-per-seed", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--action-width", type=int, default=512)
    parser.add_argument("--action-depth", type=int, default=8)
    parser.add_argument("--action-heads", type=int, default=8)
    parser.add_argument("--action-ffn-width", type=int, default=2048)
    parser.add_argument("--drop-failure-windows", action="store_true")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    _common(inspect_parser)
    smoke_parser = commands.add_parser("smoke")
    _common(smoke_parser)
    smoke_parser.add_argument("--world-checkpoint", type=Path, required=True)
    train_parser = commands.add_parser("train")
    _common(train_parser)
    train_parser.add_argument("--world-checkpoint", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--eval-batch-size", type=int, default=64)
    train_parser.add_argument("--eval-every", type=int, default=500)
    train_parser.add_argument("--log-every", type=int, default=20)
    train_parser.add_argument("--flow-steps", type=int, default=10)
    train_parser.add_argument("--action-lr", type=float, default=1e-4)
    train_parser.add_argument("--shared-lr", type=float, default=5e-5)
    train_parser.add_argument("--world-lr", type=float, default=1e-6)
    train_parser.add_argument("--weight-decay", type=float, default=1e-2)
    train_parser.add_argument("--action-weight", type=float, default=1.0)
    train_parser.add_argument("--world-weight", type=float, default=1.0)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--aux-t-max", type=int, default=500)
    train_parser.add_argument("--aux-batch-max", type=int, default=64)
    train_parser.add_argument("--aux-empty-max", type=int, default=16)
    train_parser.add_argument("--freeze-world", action="store_true")
    train_parser.add_argument("--overfit", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    evaluate_parser = commands.add_parser("evaluate")
    _common(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--world-checkpoint", type=Path, default=None)
    evaluate_parser.add_argument("--split", choices=("val", "test"), default="val")
    evaluate_parser.add_argument("--eval-batch-size", type=int, default=64)
    evaluate_parser.add_argument("--flow-steps", type=int, default=10)

    action_v2_parser = commands.add_parser("train-action")
    _v2_train_arguments(action_v2_parser, joint=False)
    joint_v2_parser = commands.add_parser("train-joint")
    _v2_train_arguments(joint_v2_parser, joint=True)
    evaluate_v2_parser = commands.add_parser("evaluate-v2")
    _v2_common(evaluate_v2_parser)
    evaluate_v2_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_v2_parser.add_argument(
        "--split", choices=("val", "test", "unseen"), default="val")

    args = parser.parse_args()
    if args.command in ("train-action", "train-joint", "evaluate-v2"):
        args.dataset_root = args.dataset_root.expanduser().resolve()
        args.latent_root = args.latent_root.expanduser().resolve()
        args.index_root = ((args.latent_root.parent / "index") if args.index_root is None
                           else args.index_root.expanduser().resolve())
        args.out = args.out.expanduser().resolve()
        if getattr(args, "checkpoint", None) is not None:
            args.checkpoint = args.checkpoint.expanduser().resolve()
        if getattr(args, "action_checkpoint", None) is not None:
            args.action_checkpoint = args.action_checkpoint.expanduser().resolve()
        if getattr(args, "world_checkpoint", None) is not None:
            args.world_checkpoint = args.world_checkpoint.expanduser().resolve()
        if getattr(args, "resume_from", None) is not None:
            args.resume_from = args.resume_from.expanduser().resolve()
            if not args.resume_from.is_file():
                parser.error(f"resume checkpoint does not exist: {args.resume_from}")
        if args.command == "evaluate-v2":
            evaluate_v2(args)
        else:
            if args.grad_accumulation <= 0 or args.micro_batch_size <= 0:
                parser.error("batch sizes and gradient accumulation must be positive")
            if args.eval_every <= 0 or args.checkpoint_every < 0:
                parser.error(
                    "--eval-every must be positive and --checkpoint-every "
                    "must be nonnegative")
            if (args.reset_optimizer or args.reset_stats) and not (
                    args.resume or args.resume_from):
                parser.error("--reset-optimizer/--reset-stats requires --resume or --resume-from")
            if args.lr_decay_start < 0 or args.lr_decay_start >= args.steps:
                parser.error("--lr-decay-start must satisfy 0 <= start < steps")
            if not 0 < args.lr_min_scale <= 1:
                parser.error("--lr-min-scale must be in (0, 1]")
            if args.max_preclip_grad_norm < 0:
                parser.error("--max-preclip-grad-norm must be nonnegative")
            if args.early_stop_patience < 0:
                parser.error("--early-stop-patience must be nonnegative")
            if args.early_stop_min_delta < 0:
                parser.error("--early-stop-min-delta must be nonnegative")
            if args.candidate_weight < 0:
                parser.error("--candidate-weight must be nonnegative")
            if args.command == "train-joint":
                if (args.feedback_ddim_steps <= 0
                        or not 1 <= args.feedback_gradient_steps
                        <= args.feedback_ddim_steps):
                    parser.error(
                        "feedback steps require 1 <= gradient_steps <= ddim_steps")
                if (args.world_freeze_steps < 0
                        or args.world_ramp_end < args.world_freeze_steps):
                    parser.error(
                        "world schedule requires 0 <= freeze_steps <= ramp_end")
                if not (0 <= args.world_weight_start <= args.world_weight_mid
                        <= args.world_weight):
                    parser.error(
                        "world weights must be nonnegative and monotonic")
            if args.overfit and args.steps == (10000 if args.command == "train-joint" else 20000):
                args.steps = 500 if args.command == "train-joint" else 1000
                args.eval_every = min(args.eval_every, 200)
            if args.memory_smoke:
                if args.memory_smoke_steps < 2:
                    parser.error("--memory-smoke-steps must be at least 2")
                args.steps = args.memory_smoke_steps
                args.log_every = 1
            train_v2(args)
        return
    args.data_root = args.data_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.latent_root = ((args.out/stage1.LATENT_DIR) if args.latent_root is None
                        else args.latent_root.expanduser().resolve())
    args.raw_data_root = (args.data_root.parent if args.raw_data_root is None
                          else args.raw_data_root.expanduser().resolve())
    args.manifest_root = (args.out/"world_direct_t3" if args.manifest_root is None
                          else args.manifest_root.expanduser().resolve())
    if getattr(args, "world_checkpoint", None) is not None:
        args.world_checkpoint = args.world_checkpoint.expanduser().resolve()
    if getattr(args, "checkpoint", None) is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    stage1.seed_everything(args.seed)
    if args.command == "inspect":
        inspect(args)
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "train":
        if args.overfit:
            args.steps = 1000 if args.steps is None else args.steps
            args.batch_size = 128 if args.batch_size is None else args.batch_size
            args.eval_every = min(args.eval_every, 200)
        else:
            args.steps = 30000 if args.steps is None else args.steps
            args.batch_size = 256 if args.batch_size is None else args.batch_size
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
