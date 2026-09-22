"""Fine-tune the existing causal UNet with action-predicted sparse LiDAR geometry.

Only the first convolution gains two channels: pooled projected range and hit
density. Its new weights start at zero, so step zero reproduces the original
UNet exactly. The VAE, old checkpoint, and source HDF5 are never modified.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch, manifest_hash
from try_hard_geometry_inpaint import (load_motion, load_rays, next_state,
    transform_from_initial, warp_frame, stage1)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GeometryWorldModel(stage1.WorldModel):
    def __init__(self):
        super().__init__()

    def enable_geometry(self):
        old = self.unet.conv_in
        if old.in_channels != 8:
            raise ValueError("Expected original eight-channel UNet")
        extended = nn.Conv2d(10, old.out_channels, old.kernel_size,
                             stride=old.stride, padding=old.padding,
                             dilation=old.dilation, groups=old.groups,
                             bias=old.bias is not None,
                             padding_mode=old.padding_mode).to(old.weight.device,
                                                               dtype=old.weight.dtype)
        with torch.no_grad():
            extended.weight.zero_()
            extended.weight[:, :8].copy_(old.weight)
            if old.bias is not None:
                extended.bias.copy_(old.bias)
        self.unet.conv_in = extended
        self.unet.register_to_config(in_channels=10)

    def forward(self, noisy_next, previous, actions, state, timestep, geometry):
        tokens = self.condition(actions, state)
        return self.unet(torch.cat((noisy_next, previous, geometry), dim=1),
                         timestep, encoder_hidden_states=tokens).sample


def geometry_features(warp):
    """[2,27,5] range mean and hit density, preserving the VAE grid axes."""
    hit = warp[1] > 0
    distance = np.clip(warp[0], -1, 1)
    hit_blocks = hit.reshape(27, 4, 5, 4)
    value_blocks = (distance * hit).reshape(27, 4, 5, 4)
    count = hit_blocks.sum(axis=(1, 3))
    mean = value_blocks.sum(axis=(1, 3)) / np.maximum(count, 1)
    density = count / 16.0
    return np.stack((mean, density)).astype(np.float32)


def split_positions(split, cached, manifest, samples_per_seed):
    if split == "train":
        rng = np.random.default_rng(42)
        with h5py.File(stage1.DATA / "navrl_static_train.h5", "r") as h5:
            source = cached["source_index"][:]
            finite = (np.isfinite(fetch(h5, "prev_drone_state", source)).all(axis=1) &
                      np.isfinite(fetch(h5, "action_sequence", source)).all(axis=(1, 2)))
        positions = np.concatenate([rng.choice(np.flatnonzero((cached["seeds"] == seed) & finite),
                                               size=samples_per_seed, replace=False)
                                    for seed in sorted(stage1.EXPECTED_SEEDS[split])])
        return np.sort(positions), None
    rows = manifest["splits"][split]
    lookup = {int(index): i for i, index in enumerate(cached["source_index"])}
    positions = np.array([lookup[int(row["source_index"])] for row in rows], dtype=np.int64)
    return positions, rows


def prepare_split(args, split, motion, manifest, scale):
    path = args.out / f"{split}_cache.npz"
    if path.exists() and not args.rebuild_cache:
        return path
    cached = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
    positions, rows = split_positions(split, cached, manifest, args.train_per_seed)
    source = cached["source_index"][positions].astype(np.int64)
    seeds = cached["seeds"][positions]
    geometry = np.empty((len(source), 2, 27, 5), dtype=np.float32)
    frames = np.empty((len(source), 2, 108, 20), dtype=np.float32) if split != "train" else None
    warped = np.empty_like(frames) if frames is not None else None
    rays = {int(seed): load_rays(args.raw_root, split, int(seed))
            for seed in np.unique(seeds)}
    with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
        if not np.all(fetch(h5, "step_delta", source) == 10):
            raise ValueError("Non-ten-step sample")
        if not np.all(fetch(h5, "action_mask", source)):
            raise ValueError("Incomplete ten-step action sequence")
        for start in range(0, len(source), 128):
            stop = min(start + 128, len(source))
            idx = source[start:stop]
            previous = fetch(h5, "prev_range_values", idx)
            states = fetch(h5, "prev_drone_state", idx).astype(np.float64)
            commands = fetch(h5, "action_sequence", idx).astype(np.float64)
            if not np.isfinite(states).all() or not np.isfinite(commands).all():
                raise ValueError("Nonfinite causal geometry input")
            if frames is not None:
                frames[start:stop] = fetch(h5, "range_values", idx)
            for j in range(stop - start):
                k = start + j
                next_pose = next_state(states[j], commands[j], motion)
                grid, azimuth, elevation = rays[int(seeds[k])]
                warp = warp_frame(previous[j],
                                  transform_from_initial(states[j], next_pose),
                                  grid, azimuth, elevation)
                geometry[k] = geometry_features(warp)
                if warped is not None:
                    warped[k] = warp
            if start % 2048 == 0:
                print(json.dumps({"cache_split": split, "done": stop,
                                  "total": len(source)}), flush=True)
    if rows is not None and any(int(row["source_index"]) != int(index)
                                for row, index in zip(rows, source)):
        raise ValueError("Manifest order mismatch")
    payload = {"source_index": source, "seeds": seeds,
               "previous": cached["previous"][positions].copy() * scale,
               "target": cached["target"][positions].copy() * scale,
               "actions": cached["actions"][positions].copy(),
               "state": stage1.causal_state(cached["state"][positions]),
               "geometry": geometry}
    if frames is not None:
        payload.update(frames=frames, warped=warped)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    temporary.replace(path)
    print(json.dumps({"cached": str(path), "samples": len(source)}), flush=True)
    return path


def dataset_from_cache(path):
    with np.load(path) as cached:
        return TensorDataset(*(torch.from_numpy(cached[name].copy()).float()
                               for name in ("previous", "target", "actions", "state",
                                            "geometry")))


@torch.no_grad()
def generate(model, scheduler, previous, actions, state, geometry, seed, steps=20):
    model.eval()
    torch.manual_seed(seed)
    scheduler.set_timesteps(steps, device=stage1.DEVICE)
    latent = torch.randn_like(previous)
    for timestep in scheduler.timesteps:
        noise = model(latent, previous, actions, state, timestep, geometry)
        latent = scheduler.step(noise, timestep, latent, eta=0.0).prev_sample
    return latent


@torch.no_grad()
def evaluate(model, vae, path, scale, threshold, raw_root, split,
             limit=None, batch_size=16):
    with np.load(path) as cached:
        size = len(cached["source_index"])
        selected = np.arange(size if limit is None else min(limit, size))
        seeds = cached["seeds"][selected]
        rays = {int(seed): load_rays(raw_root, split, int(seed))[0]
                for seed in np.unique(seeds)}
        records = []
        scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                         prediction_type="epsilon", clip_sample=False)
        for seed in sorted(rays):
            locations = selected[seeds == seed]
            for start in range(0, len(locations), batch_size):
                chosen = locations[start:start + batch_size]
                arrays = [torch.from_numpy(cached[name][chosen].copy()).float().to(stage1.DEVICE)
                          for name in ("previous", "actions", "state", "geometry")]
                latent = generate(model, scheduler, *arrays,
                                  seed=42 + int(seed) * 10000 + start)
                prediction = vae.decode(latent / scale).sample.cpu().numpy()
                for i, position in enumerate(chosen):
                    target = cached["frames"][position]
                    warp = cached["warped"][position]
                    grid = rays[int(seed)]
                    records.append({"source_index": int(cached["source_index"][position]),
                                    "seed": int(seed),
                                    "warp": lidar_metric(warp, target, grid, 0),
                                    "model": lidar_metric(prediction[i], target,
                                                          grid, threshold)})
    paired = [r for r in records if not r["warp"]["empty_cloud"]]
    summary = {"samples": len(records), "paired_nonempty": len(paired),
               "paired_cd_paper_m2": {name: float(np.mean([
                   r[name]["cd_paper_m2"] for r in paired])) for name in ("warp", "model")},
               "all_samples": {name: summarize([r[name] for r in records])
                               for name in ("warp", "model")}}
    for seed in sorted({r["seed"] for r in records}):
        subset = [r for r in paired if r["seed"] == seed]
        summary[f"seed_{seed}"] = {name: float(np.mean([
            r[name]["cd_paper_m2"] for r in subset])) for name in ("warp", "model")}
    return {"split": split, "summary": summary, "rows": records}


def train(args, model, train_path, val_path, vae, scale, threshold):
    training = dataset_from_cache(train_path)
    first = model.unet.conv_in
    optimizer = torch.optim.AdamW([
        {"params": list(first.parameters()), "lr": args.input_lr},
        {"params": [p for name, p in model.named_parameters()
                    if not name.startswith("unet.conv_in.")], "lr": args.lr}],
        weight_decay=0.01)
    scheduler = stage1.DDPMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon")
    loader = stage1.infinite(DataLoader(training, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True, num_workers=0))
    history = []
    best = math.inf
    for step in range(1, args.steps + 1):
        model.train()
        previous, target, actions, state, geometry = [
            tensor.to(stage1.DEVICE) for tensor in next(loader)]
        noise = torch.randn_like(target)
        timestep = torch.randint(0, 1000, (len(target),),
                                 device=stage1.DEVICE, dtype=torch.long)
        noisy = scheduler.add_noise(target, noise, timestep)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, previous, actions, state, timestep, geometry)
        loss = F.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        if step == 1 or step % 25 == 0:
            record = {"step": step, "noise_mse": float(loss.item()),
                      "grad_norm": float(norm.item())}
            history.append(record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(model, vae, val_path, scale, threshold,
                              args.raw_root, "val", limit=None,
                              batch_size=args.eval_batch_size)
            score = result["summary"]["paired_cd_paper_m2"]["model"]
            print(json.dumps({"step": step, "validation_cd_paper_m2": score,
                              "warp_cd_paper_m2": result["summary"]
                              ["paired_cd_paper_m2"]["warp"]}), flush=True)
            stage1.save_json(args.out / "validation_latest.json", result)
            if score < best:
                best = score
                checkpoint = args.out / "best.pt"
                temp = checkpoint.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "step": step,
                            "validation_cd_paper_m2": score}, temp)
                temp.replace(checkpoint)
                stage1.save_json(args.out / "validation_best.json", result)
                stage1.save_json(args.out / "best_metrics.json", {
                    "step": step, "validation_cd_paper_m2": score})
            stage1.save_json(args.out / "history.json", history)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=stage1.OUT / "geometry_conditioned_unet")
    parser.add_argument("--manifest", type=Path, default=stage1.OUT / "representative_baseline")
    parser.add_argument("--base", type=Path,
                        default=stage1.OUT / "world_circular_causal_8h" / "best.pt")
    parser.add_argument("--motion", type=Path,
                        default=stage1.OUT / "epona_probe" / "motion_ridge.npz")
    parser.add_argument("--train-per-seed", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--input-lr", type=float, default=1e-4)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    start_time = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    stage1.DATA = args.data_root.expanduser().resolve()
    stage1.seed_everything(42)
    manifest = json.loads((args.manifest / "sample_manifest.json").read_text())
    latent_metadata = json.loads((stage1.OUT / stage1.LATENT_DIR /
                                  "metadata.json").read_text())
    identity = stage1.circular_vae_identity()
    if any(latent_metadata.get(key) != value for key, value in identity.items()):
        raise ValueError("VAE and latent cache mismatch")
    scale = latent_metadata["scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    config = {"base_checkpoint": str(args.base), "base_sha256": sha256(args.base),
              "motion_checkpoint": str(args.motion), "motion_sha256": sha256(args.motion),
              "vae_sha256": identity["vae_sha256"], "sample_manifest_sha256": manifest_hash(manifest),
              "train_per_seed": args.train_per_seed, "train_seeds": list(range(16)),
              "validation_seeds": [16, 17], "test_seeds": [18, 19],
              "steps": args.steps, "eval_every": args.eval_every,
              "batch_size": args.batch_size, "lr": args.lr, "input_lr": args.input_lr,
              "condition": "pooled action-predicted warp normalized hit range and hit density",
              "geometry_shape": [2, 27, 5], "input_channels": 10,
              "selection": "minimum validation paired decoded squared Chamfer",
              "future_pose_input": False, "sampling": "DDIM 20 steps"}
    stage1.save_json(args.out / "config.json", config)
    motion = load_motion(args.motion)
    paths = {split: prepare_split(args, split, motion, manifest, scale)
             for split in ("train", "val", "test")}
    if args.prepare_only:
        return
    vae = stage1.load_circular_vae()
    model = GeometryWorldModel().to(stage1.DEVICE).float()
    base = stage1.load_model(args.base, model)
    model.enable_geometry()
    print(json.dumps({"initialized_from_step": base["step"],
                      "train_samples": len(np.load(paths["train"])["source_index"])}), flush=True)
    train(args, model, paths["train"], paths["val"], vae, scale, threshold)
    checkpoint = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = evaluate(model, vae, paths["test"], scale, threshold,
                      args.raw_root, "test", batch_size=args.eval_batch_size)
    stage1.save_json(args.out / "test.json", result)
    stage1.save_json(args.out / "selection.json", {"selected_step": checkpoint["step"],
        "validation_cd_paper_m2": checkpoint["validation_cd_paper_m2"],
        "test_cd_paper_m2": result["summary"]["paired_cd_paper_m2"],
        "elapsed_seconds": time.monotonic() - start_time})
    print(json.dumps({"test": result["summary"]["paired_cd_paper_m2"],
                      "selected_step": checkpoint["step"]}), flush=True)


if __name__ == "__main__":
    main()
