"""NavRL one-frame LiDAR world model using locally vendored Diffusers modules.

Run with a PPU-compatible Python environment and data root.
The input actions are ten consecutive simulator actions between two LiDAR frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("LAGEN_ATTENTION_BACKEND", "sdpa")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "third_party" / "diffusers" / "src"))
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from lidar_wam.runner import utils

DATA = Path(os.environ.get("LIDAR_WAM_DATA_ROOT", PROJECT / "data" / "lagen_cache"))
OUT = PROJECT / "outputs"
CIRCULAR_VAE = PROJECT / "lidar_wam" / "vae" / "circular"
CIRCULAR_VAE_DIR = "vae_circular"
LATENT_DIR = "latents_circular"
WORLD_FULL_DIR = "world_circular_causal_full"
WORLD_OVERFIT_DIR = "world_circular_causal_overfit"
EXPECTED_SEEDS = {"train": set(range(16)), "val": {16, 17}, "test": {18, 19}}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAUSAL_STATE_DEFINITION = "prev_ego_feats[0,1,4] with future-derived dims 2,3 zeroed"


def causal_state(state):
    """Remove features computed from the next frame's drone state."""
    result = np.array(state, dtype=np.float32, copy=True)
    result[..., 2:4] = 0.0
    return result


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def valid_indices(h5, split):
    """Reject incomplete transitions, non-finite conditions, and split leakage."""
    pieces = []
    counts = {"rows": len(h5["step_delta"]), "valid": 0}
    for start in range(0, counts["rows"], 4096):
        stop = min(start + 4096, counts["rows"])
        mask = h5["action_mask"][start:stop].all(axis=1)
        mask &= h5["step_delta"][start:stop] == 10
        actions = h5["normalized_action_sequence"][start:stop]
        state = h5["prev_ego_feats"][start:stop]
        mask &= np.isfinite(actions).all(axis=(1, 2))
        mask &= np.isfinite(state).all(axis=1)
        seeds = h5["terrain_seed"][start:stop]
        if not set(np.unique(seeds)).issubset(EXPECTED_SEEDS[split]):
            raise ValueError(f"Unexpected {split} seed: {set(np.unique(seeds))}")
        mask &= np.isin(seeds, tuple(EXPECTED_SEEDS[split]))
        for key in ("range_values", "prev_range_values"):
            mask &= np.isfinite(h5[key][start:stop]).all(axis=(1, 2, 3))
        pieces.append(np.flatnonzero(mask) + start)
    idx = np.concatenate(pieces).astype(np.int64)
    counts["valid"] = len(idx)
    counts["excluded"] = counts["rows"] - len(idx)
    if not len(idx):
        raise ValueError(f"No valid samples in {split}")
    return idx, counts


class Frames(Dataset):
    def __init__(self, split, limit=None, overfit=False, need_prev=False):
        path = DATA / f"navrl_static_{split}.h5"
        with h5py.File(path, "r") as h5:
            idx, self.counts = valid_indices(h5, split)
            if overfit:
                idx = idx[:128]
            elif limit is not None:
                # Cover both held-out seeds when evaluating a subset.
                if split != "train" and len(EXPECTED_SEEDS[split]) == 2:
                    seeds = h5["terrain_seed"][:]
                    halves = [idx[seeds[idx] == s][: max(1, limit // 2)]
                              for s in sorted(EXPECTED_SEEDS[split])]
                    idx = np.concatenate(halves)[:limit]
                else:
                    idx = idx[:limit]
            self.image = h5["range_values"][idx].astype(np.float32)
            self.prev = h5["prev_range_values"][idx].astype(np.float32) if need_prev else None
            self.actions = h5["normalized_action_sequence"][idx].astype(np.float32) if need_prev else None
            self.state = causal_state(h5["prev_ego_feats"][idx]) if need_prev else None
            self.seeds = h5["terrain_seed"][idx].astype(np.int16)
            self.indices = idx

    def __len__(self):
        return len(self.image)

    def __getitem__(self, i):
        if self.prev is None:
            return torch.from_numpy(self.image[i])
        return (torch.from_numpy(self.prev[i]), torch.from_numpy(self.image[i]),
                torch.from_numpy(self.actions[i]), torch.from_numpy(self.state[i]))


def make_vae(circular=True):
    config = CIRCULAR_VAE / "config.json" if circular else PROJECT / "lidar_wam" / "vae" / "config.json"
    cfg = json.loads(config.read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg["sample_size"] = [108, 20]
    model = AutoencoderKL(**cfg)
    if circular:
        utils.replace_down(model)
        utils.replace_conv(model)
        utils.replace_attn(model)
    return model


def circular_vae_identity():
    path = CIRCULAR_VAE / "diffusion_pytorch_model.safetensors"
    digest = hashlib.sha256()
    with path.open("rb") as weights:
        for chunk in iter(lambda: weights.read(1024 * 1024), b""):
            digest.update(chunk)
    step = json.loads((CIRCULAR_VAE / "best_validation.json").read_text())["step"]
    return {"vae_variant": "circular", "vae_step": step,
            "vae_checkpoint": "lidar_wam/vae/circular/diffusion_pytorch_model.safetensors",
            "vae_sha256": digest.hexdigest()}


def load_circular_vae():
    model = make_vae(circular=True)
    model.load_state_dict(load_file(str(CIRCULAR_VAE / "diffusion_pytorch_model.safetensors")),
                          strict=True)
    return model.to(DEVICE).float().eval()


def vae_loss(model, image):
    posterior = model.encode(image).latent_dist
    reconstruction = model.decode(posterior.sample()).sample
    valid = (image[:, 1:2, :, :18] > 0).float()
    distance = (reconstruction[:, :1, :, :18] - image[:, :1, :, :18]).abs()
    valid_l1 = (distance * valid).sum() / valid.sum().clamp_min(1)
    invalid_l1 = (distance * (1 - valid)).sum() / (1 - valid).sum().clamp_min(1)
    mask_bce = F.binary_cross_entropy_with_logits(
        reconstruction[:, 1:2, :, :18], valid, pos_weight=torch.tensor(4.0, device=image.device))
    kl = posterior.kl().mean()
    total = 3.0 * valid_l1 + 0.1 * invalid_l1 + mask_bce + 1e-6 * kl
    return total, {"valid_l1": valid_l1, "invalid_l1": invalid_l1,
                   "mask_bce": mask_bce, "kl": kl}, reconstruction


@torch.no_grad()
def evaluate_vae(model, dataset, batch_size=64, mask_threshold=0.0):
    model.eval()
    valid_sum = mask_tp = mask_fp = mask_fn = n_valid = 0.0
    for image in DataLoader(dataset, batch_size=batch_size):
        image = image.to(DEVICE)
        recon = model.decode(model.encode(image).latent_dist.mode()).sample
        target = image[:, 1:2, :, :18] > 0
        pred = recon[:, 1:2, :, :18] > mask_threshold
        valid_sum += ((recon[:, :1, :, :18] - image[:, :1, :, :18]).abs() * target).sum().item()
        n_valid += target.sum().item()
        mask_tp += (pred & target).sum().item()
        mask_fp += (pred & ~target).sum().item()
        mask_fn += (~pred & target).sum().item()
    return {"valid_range_mae": valid_sum / max(n_valid, 1),
            "mask_f1": 2 * mask_tp / max(2 * mask_tp + mask_fp + mask_fn, 1),
            "samples": len(dataset)}


def save_preview(path, prev, target, prediction, pred_mask_threshold=0.0):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for col, (label, frame) in enumerate(zip(("previous", "target", "prediction"),
                                              (prev, target, prediction))):
        rng = np.clip((frame[0, :, :18] + 1) * 5, 0, 10)
        mask = frame[1, :, :18] > (pred_mask_threshold if col == 2 else 0.0)
        axes[0, col].imshow(np.where(mask, rng, np.nan).T, origin="lower", vmin=0, vmax=10)
        axes[0, col].set_title(label + " range (m)")
        axes[1, col].imshow(mask.T, origin="lower", vmin=0, vmax=1, cmap="gray")
        axes[1, col].set_title(label + " valid mask")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_model(path, model, optimizer, step, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "extra": extra or {}}, temporary)
    temporary.replace(path)


def load_model(path, model):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt


def infinite(loader):
    while True:
        yield from loader


def resume_training(path, model, optimizer):
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint missing: {path}")
    ckpt = load_model(path, model)
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


def train_vae(args):
    train = Frames("train", overfit=args.overfit)
    val = Frames("train", overfit=True) if args.overfit else Frames("val")
    run_dir = args.out / ("vae_circular_train_overfit" if args.overfit else "vae_circular_train_full")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", {"stage": "vae", "steps": args.steps,
              "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
              "loss": "3*valid_l1 + .1*invalid_l1 + mask_bce(pos_weight=4) + 1e-6*kl",
              "train_filter": train.counts, "val_filter": val.counts})
    model = make_vae().to(DEVICE).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = infinite(DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True))
    first_step = resume_training(run_dir / "latest.pt", model, optimizer) + 1 if args.resume else 1
    best_path = run_dir / "best_metrics.json"
    best = (json.loads(best_path.read_text())["valid_range_mae"]
            + .2 * (1 - json.loads(best_path.read_text())["mask_f1"])) if args.resume and best_path.exists() else math.inf
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if args.resume and history_path.exists() else []
    for step in range(first_step, args.steps + 1):
        model.train()
        image = next(loader).to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss, components, _ = vae_loss(model, image)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"VAE loss non-finite at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"VAE grad non-finite at step {step}")
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            record = {"step": step, "loss": loss.item(),
                      **{k: v.item() for k, v in components.items()}, "grad_norm": grad.item()}
            history.append(record)
            print(json.dumps(record), flush=True)
        if step == args.steps or step % args.eval_every == 0:
            metrics = evaluate_vae(model, val)
            score = metrics["valid_range_mae"] + 0.2 * (1 - metrics["mask_f1"])
            print(json.dumps({"step": step, "validation": metrics}), flush=True)
            save_model(run_dir / "latest.pt", model, optimizer, step, metrics)
            if score < best:
                best = score
                save_model(run_dir / "best.pt", model, optimizer, step, metrics)
                with torch.no_grad():
                    x = torch.from_numpy(val.image[:1]).to(DEVICE)
                    y = model.decode(model.encode(x).latent_dist.mode()).sample[0].cpu().numpy()
                    save_preview(run_dir / "preview.png", val.image[0], val.image[0], y)
                save_json(run_dir / "best_metrics.json", {"step": step, **metrics})
            save_json(run_dir / "history.json", history)
    if not args.overfit:
        gate = json.loads((run_dir / "best_metrics.json").read_text())
        gate["passed"] = gate["valid_range_mae"] <= 0.10 and gate["mask_f1"] >= 0.70
        save_json(run_dir / "gate.json", gate)
    return run_dir / "best.pt"


class ActionCondition(nn.Module):
    def __init__(self, width=768, state_dim=5):
        super().__init__()
        self.actions = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.state = nn.Sequential(nn.Linear(state_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.zeros(1, 10, width))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, actions, state):
        return torch.cat([self.actions(actions) + self.position,
                          self.state(state).unsqueeze(1)], dim=1)


class WorldModel(nn.Module):
    def __init__(self, state_dim=5):
        super().__init__()
        self.condition = ActionCondition(state_dim=state_dim)
        self.unet = UNet2DConditionModel(
            sample_size=(27, 5), in_channels=8, out_channels=4,
            layers_per_block=4, block_out_channels=(256, 512, 512),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=768, attention_head_dim=8)

    def forward(self, noisy_next, previous, actions, state, timestep):
        tokens = self.condition(actions, state)
        return self.unet(torch.cat([noisy_next, previous], dim=1), timestep,
                         encoder_hidden_states=tokens).sample


def cache_latents(args):
    identity = circular_vae_identity()
    vae = load_circular_vae()
    validation = evaluate_vae(vae, Frames("val"), batch_size=args.batch_size,
                              mask_threshold=1.5)
    gate = {**identity, **validation, "mask_threshold": 1.5,
            "passed": validation["valid_range_mae"] <= 0.10
                      and validation["mask_f1"] >= 0.70}
    save_json(args.out / CIRCULAR_VAE_DIR / "gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError("Circular VAE validation gate did not pass")
    cache_dir = args.out / LATENT_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    scaling = float(vae.config.scaling_factor)
    stats = {}
    for split in ("train", "val", "test"):
        frames = Frames(split, need_prev=True)
        targets, previous = [], []
        with torch.no_grad():
            for prev, target, _, _ in DataLoader(frames, batch_size=args.batch_size):
                targets.append(vae.encode(target.to(DEVICE)).latent_dist.mode().cpu().numpy())
                previous.append(vae.encode(prev.to(DEVICE)).latent_dist.mode().cpu().numpy())
        target = np.concatenate(targets).astype(np.float32)
        prev = np.concatenate(previous).astype(np.float32)
        np.savez(cache_dir / f"{split}.npz", target=target, previous=prev,
                 actions=frames.actions, state=frames.state,
                 seeds=frames.seeds, source_index=frames.indices)
        stats[split] = {"samples": len(frames), "filter": frames.counts}
        print(f"Cached {split}: {len(frames)}", flush=True)
    save_json(cache_dir / "metadata.json", {"scaling_factor": scaling,
              **identity, "splits": stats,
              "latent_definition": "posterior mode * scaling_factor"})


class Latents(Dataset):
    def __init__(self, split, out, limit=None, overfit=False):
        data = np.load(out / LATENT_DIR / f"{split}.npz")
        metadata = json.loads((out / LATENT_DIR / "metadata.json").read_text())
        scale = metadata["scaling_factor"]
        if overfit:
            selection = np.arange(min(128, len(data["seeds"])))
        elif limit is not None and split != "train":
            seeds = data["seeds"]
            selection = np.concatenate([
                np.flatnonzero(seeds == seed)[:max(1, limit // len(EXPECTED_SEEDS[split]))]
                for seed in sorted(EXPECTED_SEEDS[split])])[:limit]
        else:
            selection = slice(None, limit)
        self.target = torch.from_numpy(data["target"][selection].copy() * scale)
        self.previous = torch.from_numpy(data["previous"][selection].copy() * scale)
        self.actions = torch.from_numpy(data["actions"][selection].copy())
        self.state = torch.from_numpy(causal_state(data["state"][selection]))
        self.seeds = data["seeds"][selection].copy()
        self.indices = data["source_index"][selection].copy()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        return self.previous[i], self.target[i], self.actions[i], self.state[i]


def train_world(args):
    train = Latents("train", args.out, overfit=args.overfit)
    val = Latents("train", args.out, overfit=True) if args.overfit else Latents("val", args.out, limit=512)
    run_dir = args.out / (WORLD_OVERFIT_DIR if args.overfit else WORLD_FULL_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    latent_meta = json.loads((args.out / LATENT_DIR / "metadata.json").read_text())
    identity = circular_vae_identity()
    if any(latent_meta.get(key) != value for key, value in identity.items()):
        raise ValueError("Circular VAE weights changed; rebuild latent cache")
    config_path = run_dir / "config.json"
    if args.resume and config_path.exists():
        old_config = json.loads(config_path.read_text())
        if (old_config.get("vae_sha256") != latent_meta["vae_sha256"] or
                old_config.get("ego_condition") != CAUSAL_STATE_DEFINITION):
            raise ValueError("Cannot resume world training after the VAE or state definition changed")
    save_json(run_dir / "config.json", {"stage": "world", "steps": args.steps,
              "max_hours": args.max_hours,
              "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
              "train_samples": len(train), "validation_samples": len(val),
              "vae_step": latent_meta["vae_step"],
              "vae_variant": latent_meta["vae_variant"],
              "vae_sha256": latent_meta["vae_sha256"],
              "ego_condition": CAUSAL_STATE_DEFINITION,
              "prediction": "epsilon", "diffusion_train_steps": 1000,
              "sampling": "DDIM 20 steps", "unet_block_channels": [256, 512, 512],
              "layers_per_block": 4, "cross_attention_dim": 768})
    model = WorldModel().to(DEVICE).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    loader = infinite(DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True))
    first_step = resume_training(run_dir / "latest.pt", model, optimizer) + 1 if args.resume else 1
    if args.resume:
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    best_path = run_dir / "best_metrics.json"
    best = json.loads(best_path.read_text())["validation_noise_mse"] if args.resume and best_path.exists() else math.inf
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if args.resume and history_path.exists() else []
    deadline = time.monotonic() + args.max_hours * 3600 if args.max_hours else None
    for step in range(first_step, args.steps + 1):
        model.train()
        prev, target, actions, state = [v.to(DEVICE) for v in next(loader)]
        noise = torch.randn_like(target)
        t = torch.randint(0, 1000, (len(target),), device=DEVICE, dtype=torch.long)
        noisy = scheduler.add_noise(target, noise, t)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, prev, actions, state, t)
        loss = F.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"World loss non-finite at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"World grad non-finite at step {step}")
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            record = {"step": step, "loss": loss.item(), "grad_norm": grad.item()}
            if DEVICE.type == "cuda":
                record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            history.append(record)
            print(json.dumps(record), flush=True)
        time_limit_reached = deadline is not None and time.monotonic() >= deadline
        if step == args.steps or step % args.eval_every == 0 or time_limit_reached:
            score = world_noise_mse(model, scheduler, val)
            print(json.dumps({"step": step, "validation_noise_mse": score}), flush=True)
            save_model(run_dir / "latest.pt", model, optimizer, step,
                       {"validation_noise_mse": score})
            if score < best:
                best = score
                save_model(run_dir / "best.pt", model, optimizer, step,
                           {"validation_noise_mse": score})
                save_json(run_dir / "best_metrics.json", {"step": step,
                          "validation_noise_mse": score})
            save_json(run_dir / "history.json", history)
        if time_limit_reached:
            print(json.dumps({"stopped_after_hours": args.max_hours, "step": step}), flush=True)
            break
    return run_dir / "best.pt"


@torch.no_grad()
def world_noise_mse(model, scheduler, dataset, shuffle_actions=False):
    model.eval()
    total = count = 0
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEVICE.type == "cuda" else []):
        torch.manual_seed(2917)
        for prev, target, actions, state in DataLoader(dataset, batch_size=32):
            prev, target, actions, state = [v.to(DEVICE) for v in (prev, target, actions, state)]
            if shuffle_actions:
                actions = torch.roll(actions, shifts=1, dims=0)
            noise = torch.randn_like(target)
            t = torch.randint(0, 1000, (len(target),), device=DEVICE, dtype=torch.long)
            prediction = model(scheduler.add_noise(target, noise, t), prev, actions, state, t)
            total += F.mse_loss(prediction, noise, reduction="sum").item()
            count += noise.numel()
    return total / count


@torch.no_grad()
def evaluate_conditioning(args):
    model = WorldModel().to(DEVICE).float().eval()
    checkpoint = load_model(args.out / WORLD_FULL_DIR / "best.pt", model)
    dataset = Latents("val", args.out, limit=512)
    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    correct = world_noise_mse(model, scheduler, dataset)
    wrong = world_noise_mse(model, scheduler, dataset, shuffle_actions=True)
    result = {"checkpoint_step": checkpoint["step"], "samples": len(dataset),
              "ego_condition": CAUSAL_STATE_DEFINITION,
              "correct_action_noise_mse": correct, "shifted_action_noise_mse": wrong,
              "relative_increase": wrong / correct - 1}
    save_json(args.out / WORLD_FULL_DIR / "conditioning_probe.json", result)
    print(json.dumps(result), flush=True)


@torch.no_grad()
def generate(model, scheduler, previous, actions, state, seed, init_strength=1.0,
             num_steps=20, residual_scale=None, initial_latent=None):
    if not 0 < init_strength <= 1:
        raise ValueError("init_strength must be in (0, 1]")
    if not 1 <= num_steps <= 1000:
        raise ValueError("num_steps must be in [1, 1000]")
    torch.manual_seed(seed)
    if init_strength == 1.0:
        scheduler.set_timesteps(num_steps, device=DEVICE)
        x = torch.randn_like(previous)
        timesteps = scheduler.timesteps
    else:
        scheduler.set_timesteps(min(1000, math.ceil(num_steps / init_strength)), device=DEVICE)
        timesteps = scheduler.timesteps[-num_steps:]
        initial_t = timesteps[0].repeat(len(previous))
        clean_initial = (torch.zeros_like(previous) if residual_scale is not None else
                         previous if initial_latent is None else initial_latent)
        if clean_initial.shape != previous.shape:
            raise ValueError("initial_latent and previous must have the same shape")
        x = scheduler.add_noise(clean_initial, torch.randn_like(previous), initial_t)
    for t in timesteps:
        pred = model(x, previous, actions, state, t)
        x = scheduler.step(pred, t, x, eta=0.0).prev_sample
    return previous + x / residual_scale if residual_scale is not None else x


AZIMUTH = np.deg2rad(np.arange(108, dtype=np.float32) * (360.0 / 108.0))
ELEVATION = np.deg2rad(np.linspace(-7, 52, 18, dtype=np.float32))
DIRECTIONS = np.stack((np.cos(AZIMUTH[:, None]) * np.cos(ELEVATION[None, :]),
                       np.sin(AZIMUTH[:, None]) * np.cos(ELEVATION[None, :]),
                       np.broadcast_to(np.sin(ELEVATION)[None, :], (108, 18))), axis=-1)


def to_points(frame, mask_threshold=0.0):
    valid = frame[1, :, :18] > mask_threshold
    distance = np.clip((frame[0, :, :18] + 1) * 5, 0, 10)
    return (DIRECTIONS[valid] * distance[valid, None]).astype(np.float32)


def chamfer(a, b):
    if not len(a) and not len(b):
        return 0.0
    if not len(a) or not len(b):
        return 20.0
    return float((cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean()) / 2)


def evaluate_baselines(args):
    dataset = Frames(args.split, limit=args.samples, need_prev=True)
    with h5py.File(DATA / f"navrl_static_{args.split}.h5", "r") as h5:
        transforms = h5["prev_trans_mat"][dataset.indices]
    records = []
    range_error_sum = valid_count = mask_tp = mask_fp = mask_fn = 0
    for i in range(len(dataset)):
        previous = to_points(dataset.prev[i])
        truth = to_points(dataset.image[i])
        prev_valid = dataset.prev[i, 1, :, :18] > 0
        target_valid = dataset.image[i, 1, :, :18] > 0
        range_error_sum += float((np.abs(dataset.prev[i, 0, :, :18]
                                         - dataset.image[i, 0, :, :18]) * 5
                                  * target_valid).sum())
        valid_count += int(target_valid.sum())
        mask_tp += int((prev_valid & target_valid).sum())
        mask_fp += int((prev_valid & ~target_valid).sum())
        mask_fn += int((~prev_valid & target_valid).sum())
        transform = transforms[i]
        if not np.isfinite(transform).all():
            raise ValueError(f"Non-finite pose transform at source index {dataset.indices[i]}")
        compensated = previous @ transform[:3, :3].T + transform[:3, 3]
        records.append({"source_index": int(dataset.indices[i]),
                        "seed": int(dataset.seeds[i]),
                        "copy_chamfer_m": chamfer(previous, truth),
                        "true_pose_compensated_chamfer_m": chamfer(compensated, truth)})
    report = {"split": args.split, "samples": len(records),
              "copy_chamfer_m": float(np.mean([r["copy_chamfer_m"] for r in records])),
              "copy_target_valid_range_mae_m": range_error_sum / max(valid_count, 1),
              "copy_mask_f1": 2 * mask_tp / max(2 * mask_tp + mask_fp + mask_fn, 1),
              "true_pose_compensated_chamfer_m": float(np.mean([
                  r["true_pose_compensated_chamfer_m"] for r in records])),
              "pose_baseline_uses_future_state": True,
              "records": records}
    save_json(args.out / "evaluation" / f"baselines_{args.split}_n{len(records)}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


@torch.no_grad()
def evaluate_vae_chamfer(args):
    dataset = Frames(args.split, limit=args.samples, need_prev=True)
    vae = load_circular_vae()
    identity = circular_vae_identity()
    records = []
    thresholds = (0.0, 0.5, 1.0, 1.5, 2.0)
    for start in range(0, len(dataset), args.batch_size):
        end = min(start + args.batch_size, len(dataset))
        images = dataset.image[start:end]
        reconstruction = vae.decode(
            vae.encode(torch.from_numpy(images).to(DEVICE)).latent_dist.mode()).sample.cpu().numpy()
        for j, image in enumerate(images):
            truth = to_points(image)
            row = {"seed": int(dataset.seeds[start + j]),
                   "copy_chamfer_m": chamfer(to_points(dataset.prev[start + j]), truth),
                   "true_points": len(truth)}
            for threshold in thresholds:
                prediction = to_points(reconstruction[j], threshold)
                row[f"vae_chamfer_threshold_{threshold}"] = chamfer(prediction, truth)
                row[f"vae_points_threshold_{threshold}"] = len(prediction)
            records.append(row)
    result = {"split": args.split, "samples": len(records), **identity,
              "copy_chamfer_m": float(np.mean([r["copy_chamfer_m"] for r in records])),
              "oracle_vae_chamfer_m": {
                  str(threshold): float(np.mean([r[f"vae_chamfer_threshold_{threshold}"]
                                                 for r in records])) for threshold in thresholds},
              "mean_true_points": float(np.mean([r["true_points"] for r in records])),
              "mean_vae_points": {
                  str(threshold): float(np.mean([r[f"vae_points_threshold_{threshold}"]
                                                 for r in records])) for threshold in thresholds},
              "records": records}
    if args.split == "val":
        result["selected_threshold"] = float(min(
            result["oracle_vae_chamfer_m"], key=result["oracle_vae_chamfer_m"].get))
    save_json(args.out / CIRCULAR_VAE_DIR / f"oracle_{args.split}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


@torch.no_grad()
def evaluate_world(args):
    gate = json.loads((args.out / CIRCULAR_VAE_DIR / "gate.json").read_text())
    if not gate["passed"]:
        raise RuntimeError("VAE validation gate did not pass")
    world_cfg = json.loads((args.out / WORLD_FULL_DIR / "config.json").read_text())
    latent_meta = json.loads((args.out / LATENT_DIR / "metadata.json").read_text())
    identity = circular_vae_identity()
    if (any(latent_meta.get(key) != value for key, value in identity.items()) or
            world_cfg.get("vae_sha256") != latent_meta.get("vae_sha256") or
            world_cfg.get("ego_condition") != CAUSAL_STATE_DEFINITION):
        raise ValueError("World checkpoint has stale VAE or noncausal state definition")
    model = WorldModel().to(DEVICE).float().eval()
    world_step = load_model(args.out / WORLD_FULL_DIR / "best.pt", model)["step"]
    vae = load_circular_vae()
    oracle_path = args.out / CIRCULAR_VAE_DIR / "oracle_val.json"
    if not oracle_path.exists():
        raise FileNotFoundError("Run evaluate-vae-chamfer on val to calibrate mask threshold")
    oracle = json.loads(oracle_path.read_text())
    if oracle.get("vae_sha256") != latent_meta.get("vae_sha256"):
        raise ValueError("VAE threshold calibration is stale; rerun evaluate-vae-chamfer on val")
    mask_threshold = (oracle["selected_threshold"] if args.mask_threshold is None
                      else args.mask_threshold)
    threshold_sweep = tuple(sorted(set((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, mask_threshold))))
    scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type="epsilon",
                              clip_sample=False)
    scale = latent_meta["scaling_factor"]
    report = {"split": args.split, "requested_samples": args.samples,
              "world_step": world_step, "vae_step": latent_meta["vae_step"],
              "vae_variant": "circular", "vae_sha256": latent_meta["vae_sha256"],
              "ego_condition": CAUSAL_STATE_DEFINITION,
              "ddim_steps": args.ddim_steps, "sampler_version": f"fixed_{args.ddim_steps}_steps",
              "init_strength": args.init_strength,
              "mask_threshold": mask_threshold,
              "vae_gate": gate, "per_seed": {}}
    for seed in sorted(EXPECTED_SEEDS[args.split]):
        with h5py.File(DATA / f"navrl_static_{args.split}.h5", "r") as h5:
            idx, _ = valid_indices(h5, args.split)
            idx = idx[h5["terrain_seed"][:][idx] == seed][:max(1, args.samples // 2)]
            prev_image = h5["prev_range_values"][idx]
            target_image = h5["range_values"][idx]
            actions = h5["normalized_action_sequence"][idx]
            state = h5["prev_ego_feats"][idx]
        records = []
        for start in range(0, len(idx), args.batch_size):
            sl = slice(start, start + args.batch_size)
            prev = torch.from_numpy(prev_image[sl]).to(DEVICE)
            act = torch.from_numpy(actions[sl]).to(DEVICE)
            ego = torch.from_numpy(causal_state(state[sl])).to(DEVICE)
            latent = vae.encode(prev).latent_dist.mode() * scale
            generated = generate(model, scheduler, latent, act, ego,
                                 args.seed + seed * 10000 + start, args.init_strength,
                                 args.ddim_steps)
            shuffled = act[torch.randperm(len(act), device=DEVICE)] if len(act) > 1 else act.flip(1)
            generated_shuffle = generate(model, scheduler, latent, shuffled, ego,
                                         args.seed + seed * 10000 + start, args.init_strength,
                                         args.ddim_steps)
            predicted = vae.decode(generated / scale).sample.cpu().numpy()
            predicted_shuffle = vae.decode(generated_shuffle / scale).sample.cpu().numpy()
            for j in range(len(predicted)):
                truth_cloud = to_points(target_image[sl][j])
                row = {"source_index": int(idx[start + j]),
                       "copy_chamfer_m": chamfer(to_points(prev_image[sl][j]), truth_cloud)}
                for threshold in threshold_sweep:
                    row[f"prediction_t{threshold:g}"] = chamfer(
                        to_points(predicted[j], threshold), truth_cloud)
                    row[f"shuffled_t{threshold:g}"] = chamfer(
                        to_points(predicted_shuffle[j], threshold), truth_cloud)
                row["prediction_chamfer_m"] = row[f"prediction_t{mask_threshold:g}"]
                row["shuffled_action_chamfer_m"] = row[f"shuffled_t{mask_threshold:g}"]
                records.append(row)
            if start == 0:
                save_preview(args.out / "evaluation" / f"{args.split}_seed_{seed}.png",
                             prev_image[0], target_image[0], predicted[0], mask_threshold)
                save_preview(args.out / "evaluation" /
                             f"{args.split}_seed_{seed}_step{world_step}_strength{args.init_strength:g}_circular_causal_ddim{args.ddim_steps}.png",
                             prev_image[0], target_image[0], predicted[0], mask_threshold)
        report["per_seed"][str(seed)] = records
    all_records = [r for group in report["per_seed"].values() for r in group]
    means = {key: float(np.mean([r[key] for r in all_records])) for key in
             ("copy_chamfer_m", "prediction_chamfer_m", "shuffled_action_chamfer_m")}
    means["prediction_improvement"] = 1 - means["prediction_chamfer_m"] / means["copy_chamfer_m"]
    means["shuffle_degradation"] = means["shuffled_action_chamfer_m"] / means["prediction_chamfer_m"] - 1
    means["passed"] = means["prediction_improvement"] >= .10 and means["shuffle_degradation"] >= .05
    report["summary"] = means
    report["threshold_sweep"] = {}
    for threshold in threshold_sweep:
        predicted_mean = float(np.mean([r[f"prediction_t{threshold:g}"] for r in all_records]))
        shuffled_mean = float(np.mean([r[f"shuffled_t{threshold:g}"] for r in all_records]))
        report["threshold_sweep"][str(threshold)] = {
            "prediction_chamfer_m": predicted_mean,
            "shuffled_action_chamfer_m": shuffled_mean,
            "prediction_improvement": 1 - predicted_mean / means["copy_chamfer_m"],
            "shuffle_degradation": shuffled_mean / predicted_mean - 1,
            "passed": predicted_mean <= .9 * means["copy_chamfer_m"]
                      and shuffled_mean >= 1.05 * predicted_mean}
    report["samples"] = len(all_records)
    save_json(args.out / "evaluation" /
              f"{args.split}_step{world_step}_strength{args.init_strength:g}_n{len(all_records)}_circular_causal_ddim{args.ddim_steps}.json",
              report)
    save_json(args.out / "evaluation" / f"{args.split}.json", report)
    print(json.dumps({"samples": len(all_records), **means}), flush=True)


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "benchmark-world", "train-vae",
                                            "cache-latents", "train-world", "evaluate-vae-chamfer",
                                            "evaluate-conditioning", "evaluate-baselines", "evaluate"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data-root", type=Path, default=DATA,
                        help="Directory containing navrl_static_{train,val,test}.h5")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Stop world training after this wall-clock duration, validating and saving the final step")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue from latest.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--init-strength", type=float, default=1.0,
                        help="1.0 samples from noise; lower values start from a noisy previous latent")
    parser.add_argument("--ddim-steps", type=int, default=20,
                        help="Number of DDIM sampling steps for evaluate")
    parser.add_argument("--mask-threshold", type=float, default=None,
                        help="Override the VAE-calibrated point-cloud mask logit threshold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    DATA = args.data_root.expanduser().resolve()
    seed_everything(args.seed)
    print(f"device={DEVICE}", flush=True)
    if args.command == "inspect":
        for split in ("train", "val", "test"):
            with h5py.File(DATA / f"navrl_static_{split}.h5") as h5:
                _, counts = valid_indices(h5, split)
            print(split, counts, flush=True)
    elif args.command == "benchmark-world":
        batch = args.batch_size or 32
        model = WorldModel().to(DEVICE).float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(batch, 4, 27, 5, device=DEVICE)
        actions = torch.randn(batch, 10, 3, device=DEVICE)
        state = torch.randn(batch, 5, device=DEVICE)
        t = torch.randint(0, 1000, (batch,), device=DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        loss = F.mse_loss(model(x, x, actions, state, t), x)
        loss.backward()
        optimizer.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()),
                          "batch_size": batch, "loss": loss.item(),
                          "seconds": time.monotonic() - start,
                          "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
                          if DEVICE.type == "cuda" else None}), flush=True)
    elif args.command == "train-vae":
        args.steps = args.steps or (200 if args.overfit else 20000)
        args.batch_size = args.batch_size or 64
        args.lr = args.lr or 1e-4
        train_vae(args)
    elif args.command == "cache-latents":
        args.batch_size = args.batch_size or 64
        cache_latents(args)
    elif args.command == "train-world":
        args.steps = args.steps or (200 if args.overfit else 20000)
        args.batch_size = args.batch_size or (128 if args.overfit else 256)
        args.lr = args.lr or 1e-4
        train_world(args)
    elif args.command == "evaluate":
        args.batch_size = args.batch_size or 8
        evaluate_world(args)
    elif args.command == "evaluate-conditioning":
        evaluate_conditioning(args)
    elif args.command == "evaluate-baselines":
        evaluate_baselines(args)
    elif args.command == "evaluate-vae-chamfer":
        args.batch_size = args.batch_size or 64
        evaluate_vae_chamfer(args)


if __name__ == "__main__":
    main()
