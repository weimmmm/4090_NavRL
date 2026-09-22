"""Action-conditioned residual diffusion using finite executed NavRL commands.

This keeps the existing LaGen Diffusers UNet and circular NavRL VAE. The denoising
target is the scaled change from the previous latent, so a zero prediction means
copying the previous frame. Conditions use the recorded world-frame commands and
the previous drone altitude, orientation, linear velocity, and angular velocity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from lidar_wam.runner import stage1


VERSION = "finite_world_commands_prev_kinematics_residual_v1"
STATS = "executed_residual_condition_stats.json"
FULL_DIR = "world_circular_executed_residual_full"
OVERFIT_DIR = "world_circular_executed_residual_overfit"


def latent_metadata():
    meta = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())
    if any(meta.get(key) != value for key, value in stage1.circular_vae_identity().items()):
        raise ValueError("Circular VAE weights and latent cache do not match")
    return meta


def condition_stats():
    meta = latent_metadata()
    path = stage1.OUT / stage1.LATENT_DIR / STATS
    if not path.exists():
        raise FileNotFoundError(f"Run prepare first: {path}")
    stats = json.loads(path.read_text())
    if stats.get("version") != VERSION or stats.get("vae_sha256") != meta["vae_sha256"]:
        raise ValueError("Executed-action conditioning statistics are stale")
    return stats


def prepare():
    meta = latent_metadata()
    cached = np.load(stage1.OUT / stage1.LATENT_DIR / "train.npz")
    indices = cached["source_index"]
    with h5py.File(stage1.DATA / "navrl_static_train.h5", "r") as h5:
        actions = h5["action_sequence"][indices]
        previous_state = h5["prev_drone_state"][indices, 2:13]
    finite = np.isfinite(actions).all(axis=(1, 2)) & np.isfinite(previous_state).all(axis=1)
    if finite.sum() < 1000:
        raise ValueError("Too few transitions with finite executed commands")
    clean_actions = actions[finite].astype(np.float64)
    clean_state = previous_state[finite].astype(np.float64)
    action_mean = clean_actions.mean(axis=(0, 1))
    action_std = clean_actions.std(axis=(0, 1)).clip(1e-3)
    state_mean = clean_state.mean(axis=0)
    state_std = clean_state.std(axis=0).clip(1e-3)
    delta = ((cached["target"][finite] - cached["previous"][finite]) *
             meta["scaling_factor"])
    delta_std = float(delta.std())
    if not math.isfinite(delta_std) or delta_std < 1e-4:
        raise ValueError("Invalid latent residual scale")
    stats = {"version": VERSION, "vae_sha256": meta["vae_sha256"],
             "action_source": "action_sequence: finite executed world-frame velocity commands",
             "state_source": "prev_drone_state[2:13]: z, quaternion, linear and angular velocity",
             "train_candidates": len(indices), "train_finite": int(finite.sum()),
             "train_excluded_nonfinite_commands": int((~finite).sum()),
             "action_mean": action_mean.tolist(), "action_std": action_std.tolist(),
             "state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
             "latent_residual_std": delta_std, "residual_scale": 1.0 / delta_std}
    stage1.save_json(stage1.OUT / stage1.LATENT_DIR / STATS, stats)
    print(json.dumps(stats), flush=True)
    return stats


class ExecutedLatents(Dataset):
    def __init__(self, split, limit=None, overfit=False):
        stats = condition_stats()
        meta = latent_metadata()
        cached = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
        indices = cached["source_index"]
        seeds = cached["seeds"]
        with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
            actions = h5["action_sequence"][indices]
            previous_state = h5["prev_drone_state"][indices, 2:13]
        finite = np.isfinite(actions).all(axis=(1, 2)) & np.isfinite(previous_state).all(axis=1)
        if overfit:
            selection = np.flatnonzero(finite)[:128]
        elif limit is not None and split != "train":
            selection = np.concatenate([
                np.flatnonzero(finite & (seeds == seed))[:max(1, limit // 2)]
                for seed in sorted(stage1.EXPECTED_SEEDS[split])])[:limit]
        else:
            selection = np.flatnonzero(finite)
        if len(selection) == 0:
            raise ValueError(f"No finite executed-action transitions in {split}")
        latent_scale = meta["scaling_factor"]
        self.previous = torch.from_numpy(cached["previous"][selection].copy() * latent_scale)
        self.target = torch.from_numpy(cached["target"][selection].copy() * latent_scale)
        self.actions = torch.from_numpy(((actions[selection] - stats["action_mean"]) /
                                         stats["action_std"]).astype(np.float32))
        self.state = torch.from_numpy(((previous_state[selection] - stats["state_mean"]) /
                                       stats["state_std"]).astype(np.float32))
        self.indices = indices[selection].copy()
        self.seeds = seeds[selection].copy()
        self.candidates = len(indices)
        self.finite_candidates = int(finite.sum())

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return (self.previous[index], self.target[index],
                self.actions[index], self.state[index])


@torch.no_grad()
def noise_mse(model, scheduler, data, residual_scale):
    model.eval()
    total = count = 0
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]
                               if stage1.DEVICE.type == "cuda" else []):
        torch.manual_seed(2917)
        for previous, target, actions, state in DataLoader(data, batch_size=32):
            previous, target, actions, state = [value.to(stage1.DEVICE)
                                                for value in (previous, target, actions, state)]
            clean = (target - previous) * residual_scale
            noise = torch.randn_like(clean)
            t = torch.randint(0, 1000, (len(clean),), device=stage1.DEVICE,
                              dtype=torch.long)
            predicted = model(scheduler.add_noise(clean, noise, t), previous,
                              actions, state, t)
            total += F.mse_loss(predicted, noise, reduction="sum").item()
            count += noise.numel()
    return total / count


def train(args):
    stats_path = stage1.OUT / stage1.LATENT_DIR / STATS
    stats = condition_stats() if stats_path.exists() else prepare()
    train_data = ExecutedLatents("train", overfit=args.overfit)
    val_data = (ExecutedLatents("train", overfit=True) if args.overfit else
                ExecutedLatents("val", limit=512))
    run_dir = stage1.OUT / (OVERFIT_DIR if args.overfit else FULL_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if args.resume and config_path.exists():
        previous_config = json.loads(config_path.read_text())
        if (previous_config.get("version") != VERSION or
                previous_config.get("vae_sha256") != stats["vae_sha256"] or
                previous_config.get("residual_scale") != stats["residual_scale"]):
            raise ValueError("Cannot resume with different VAE or conditioning")
    stage1.save_json(config_path, {"version": VERSION, "steps": args.steps,
                     "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
                     "train_samples": len(train_data), "validation_samples": len(val_data),
                     "vae_sha256": stats["vae_sha256"],
                     "residual_scale": stats["residual_scale"],
                     "condition_stats": f"{stage1.LATENT_DIR}/{STATS}",
                     "prediction": "epsilon of scaled latent residual",
                     "diffusion_train_steps": 1000, "state_dim": 11})
    stage1.seed_everything(args.seed)
    model = stage1.WorldModel(state_dim=11).to(stage1.DEVICE).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = stage1.DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    loader = stage1.infinite(DataLoader(train_data, batch_size=args.batch_size,
                                       shuffle=True, drop_last=True))
    first = (stage1.resume_training(run_dir / "latest.pt", model, optimizer) + 1
             if args.resume else 1)
    metric_path = run_dir / "best_metrics.json"
    best = (json.loads(metric_path.read_text())["validation_noise_mse"]
            if args.resume and metric_path.exists() else math.inf)
    history_path = run_dir / "history.json"
    history = (json.loads(history_path.read_text())
               if args.resume and history_path.exists() else [])
    for step in range(first, args.steps + 1):
        model.train()
        previous, target, actions, state = [value.to(stage1.DEVICE)
                                            for value in next(loader)]
        clean = (target - previous) * stats["residual_scale"]
        noise = torch.randn_like(clean)
        t = torch.randint(0, 1000, (len(clean),), device=stage1.DEVICE, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        predicted = model(scheduler.add_noise(clean, noise, t), previous, actions, state, t)
        loss = F.mse_loss(predicted, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite residual diffusion loss at step {step}")
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            record = {"step": step, "loss": loss.item(), "grad_norm": grad.item()}
            history.append(record)
            print(json.dumps(record), flush=True)
        if step == args.steps or step % args.eval_every == 0:
            score = noise_mse(model, scheduler, val_data, stats["residual_scale"])
            print(json.dumps({"step": step, "validation_noise_mse": score}), flush=True)
            stage1.save_model(run_dir / "latest.pt", model, optimizer, step,
                              {"validation_noise_mse": score})
            if score < best:
                best = score
                stage1.save_model(run_dir / "best.pt", model, optimizer, step,
                                  {"validation_noise_mse": score})
                stage1.save_json(metric_path, {"step": step, "validation_noise_mse": score})
            stage1.save_json(history_path, history)


@torch.no_grad()
def evaluate(args):
    stats = condition_stats()
    run_dir = stage1.OUT / FULL_DIR
    config = json.loads((run_dir / "config.json").read_text())
    if (config.get("version") != VERSION or
            config.get("vae_sha256") != stats["vae_sha256"] or
            config.get("residual_scale") != stats["residual_scale"]):
        raise ValueError("World checkpoint and executed-action statistics differ")
    data = ExecutedLatents(args.split, limit=args.samples)
    world = stage1.WorldModel(state_dim=11).to(stage1.DEVICE).float().eval()
    step = stage1.load_model(run_dir / "best.pt", world)["step"]
    vae = stage1.load_circular_vae()
    scale = latent_metadata()["scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                    prediction_type="epsilon", clip_sample=False)
    with h5py.File(stage1.DATA / f"navrl_static_{args.split}.h5", "r") as h5:
        previous_images = h5["prev_range_values"][data.indices]
        target_images = h5["range_values"][data.indices]
    rows = []
    for terrain_seed in sorted(stage1.EXPECTED_SEEDS[args.split]):
        positions = np.flatnonzero(data.seeds == terrain_seed)
        rng = np.random.default_rng(args.seed + int(terrain_seed))
        if len(positions) > 1:
            permutation = rng.permutation(len(positions))
            while np.any(permutation == np.arange(len(positions))):
                permutation = rng.permutation(len(positions))
            shuffled_positions = positions[permutation]
        else:
            shuffled_positions = positions
        for start in range(0, len(positions), args.batch_size):
            indices = positions[start:start + args.batch_size]
            previous = data.previous[indices].to(stage1.DEVICE)
            actions = data.actions[indices].to(stage1.DEVICE)
            state = data.state[indices].to(stage1.DEVICE)
            seed = args.seed + terrain_seed * 10000 + start
            generated = stage1.generate(world, scheduler, previous, actions, state,
                                        seed, args.init_strength, args.ddim_steps,
                                        residual_scale=stats["residual_scale"])
            shuffled = data.actions[shuffled_positions[start:start + len(indices)]].to(stage1.DEVICE)
            generated_shuffled = stage1.generate(
                world, scheduler, previous, shuffled, state, seed,
                args.init_strength, args.ddim_steps,
                residual_scale=stats["residual_scale"])
            true_delta = data.target[indices].to(stage1.DEVICE) - previous
            predicted_delta = generated - previous
            true_flat = true_delta.flatten(1)
            predicted_flat = predicted_delta.flatten(1)
            true_rms = true_flat.square().mean(dim=1).sqrt().cpu().numpy()
            predicted_rms = predicted_flat.square().mean(dim=1).sqrt().cpu().numpy()
            delta_cosine = F.cosine_similarity(true_flat, predicted_flat, dim=1).cpu().numpy()
            predictions = vae.decode(generated / scale).sample.cpu().numpy()
            shuffled_predictions = vae.decode(generated_shuffled / scale).sample.cpu().numpy()
            for j, index in enumerate(indices):
                truth = stage1.to_points(target_images[index])
                rows.append({"source_index": int(data.indices[index]),
                             "terrain_seed": int(terrain_seed),
                             "true_latent_delta_rms": float(true_rms[j]),
                             "predicted_latent_delta_rms": float(predicted_rms[j]),
                             "latent_delta_cosine": float(delta_cosine[j]),
                             "copy_chamfer_m": stage1.chamfer(
                                 stage1.to_points(previous_images[index]), truth),
                             "prediction_chamfer_m": stage1.chamfer(
                                 stage1.to_points(predictions[j], threshold), truth),
                             "shuffled_action_chamfer_m": stage1.chamfer(
                                 stage1.to_points(shuffled_predictions[j], threshold), truth)})
            if start == 0:
                path = (stage1.OUT / "evaluation" /
                        f"executed_residual_{args.split}_seed{terrain_seed}_step{step}_strength{args.init_strength:g}_ddim{args.ddim_steps}.png")
                stage1.save_preview(path, previous_images[indices[0]],
                                    target_images[indices[0]], predictions[0], threshold)
        print(f"evaluated seed {terrain_seed}: {len(positions)}", flush=True)
    means = {key: float(np.mean([row[key] for row in rows])) for key in
             ("copy_chamfer_m", "prediction_chamfer_m", "shuffled_action_chamfer_m",
              "true_latent_delta_rms", "predicted_latent_delta_rms", "latent_delta_cosine")}
    means["prediction_improvement"] = 1 - means["prediction_chamfer_m"] / means["copy_chamfer_m"]
    means["shuffle_degradation"] = means["shuffled_action_chamfer_m"] / means["prediction_chamfer_m"] - 1
    means["passed"] = means["prediction_improvement"] >= 0.10 and means["shuffle_degradation"] >= 0.05
    report = {"split": args.split, "world_step": step, "version": VERSION,
              "init_strength": args.init_strength, "ddim_steps": args.ddim_steps,
              "mask_threshold": threshold, "samples": len(rows),
              "finite_executed_action_only": True,
              "action_shuffle": "fixed per-seed derangement across selected samples",
              "summary": means, "rows": rows}
    path = (stage1.OUT / "evaluation" /
            f"executed_residual_{args.split}_step{step}_strength{args.init_strength:g}_n{len(rows)}_ddim{args.ddim_steps}.json")
    stage1.save_json(path, report)
    print(json.dumps({"samples": len(rows), **means}), flush=True)
    print(f"saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--init-strength", type=float, default=1.0)
    parser.add_argument("--ddim-steps", type=int, default=20)
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    stage1.seed_everything(args.seed)
    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        args.steps = args.steps or (200 if args.overfit else 5000)
        args.batch_size = args.batch_size or (128 if args.overfit else 256)
        train(args)
    else:
        args.batch_size = args.batch_size or 8
        evaluate(args)


if __name__ == "__main__":
    main()
