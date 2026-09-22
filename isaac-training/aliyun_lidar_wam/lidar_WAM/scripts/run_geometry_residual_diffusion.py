"""Train a causal geometry-conditioned residual LiDAR diffusion model.

The action-to-pose model predicts a sensor transform from the previous state
and ten executed commands. Reprojection supplies a latent anchor; the UNet
diffuses the next latent minus that anchor. No future pose enters inference.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_geometry import load_rays, warp_frame
from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch, identity_check, manifest_hash
from try_epona_guided_diffusion import load_motion
from try_epona_motion import next_state, transform_from_initial


VERSION = "causal_geometry_residual_diffusion_v1"


def paths(args, split):
    return args.out / "geometry_cache" / f"{split}.npz"


def latent_metadata():
    meta = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())
    if any(meta.get(k) != v for k, v in stage1.circular_vae_identity().items()):
        raise ValueError("Circular VAE weights do not match the latent cache")
    return meta


class GeometryResidualUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = stage1.ActionCondition()
        self.unet = stage1.UNet2DConditionModel(
            sample_size=(27, 5), in_channels=12, out_channels=4,
            layers_per_block=4, block_out_channels=(256, 512, 512),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=768, attention_head_dim=8)

    def forward(self, noisy_residual, previous, geometry, actions, state, timestep):
        tokens = self.condition(actions, state)
        inputs = torch.cat((noisy_residual, previous, geometry), dim=1)
        return self.unet(inputs, timestep, encoder_hidden_states=tokens).sample


def initialize_from_unet(model, checkpoint):
    original = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    saved = stage1.load_model(checkpoint, original)
    model.condition.load_state_dict(original.condition.state_dict())
    old = original.unet.state_dict()
    new = model.unet.state_dict()
    for key, value in old.items():
        if key == "conv_in.weight":
            new[key].zero_()
            new[key][:, :8] = value
        else:
            new[key].copy_(value)
    model.unet.load_state_dict(new)
    del original
    return saved["step"]


@torch.no_grad()
def prepare_split(args, split, vae, motion, meta):
    cache = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
    source = cache["source_index"].astype(np.int64)
    with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
        commands = fetch(h5, "action_sequence", source)
        drone = fetch(h5, "prev_drone_state", source)
    finite = np.isfinite(commands).all(axis=(1, 2)) & np.isfinite(drone).all(axis=1)
    indices = source[finite]
    seeds = cache["seeds"][finite]
    commands = commands[finite]
    drone = drone[finite]
    if not len(indices):
        raise ValueError(f"No finite geometry rows in {split}")
    rays = {int(seed): load_rays(args.raw_root, split, int(seed))
            for seed in sorted(set(seeds))}
    guides = []
    weights = []
    identity_max = 0.0
    with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
        for start in range(0, len(indices), args.prepare_batch_size):
            stop = min(start + args.prepare_batch_size, len(indices))
            previous = fetch(h5, "prev_range_values", indices[start:stop])
            target = fetch(h5, "range_values", indices[start:stop])
            warped = []
            for j, position in enumerate(range(start, stop)):
                ray_grid, azimuth, elevation = rays[int(seeds[position])]
                if int(seeds[position]) not in getattr(prepare_split, "_checked", set()):
                    identity_max = max(identity_max, identity_check(
                        previous[j], ray_grid, azimuth, elevation))
                    checked = getattr(prepare_split, "_checked", set())
                    checked.add(int(seeds[position]))
                    prepare_split._checked = checked
                state = drone[position].astype(np.float64)
                command = commands[position].astype(np.float64)
                prediction = next_state(state, command, motion)
                warped.append(warp_frame(previous[j], transform_from_initial(
                    state, prediction), ray_grid, azimuth, elevation))
            warped = np.stack(warped)
            guide = vae.encode(torch.from_numpy(warped).to(stage1.DEVICE)).latent_dist.mode()
            guides.append((guide * meta["scaling_factor"]).cpu().numpy())
            true_new = ((target[:, 1, :, :18] > 0) & (warped[:, 1, :, :18] <= 0))
            mask = np.zeros((len(warped), 1, 108, 20), dtype=np.float32)
            mask[:, 0, :, :18] = true_new.astype(np.float32)
            weight = F.max_pool2d(torch.from_numpy(mask), kernel_size=4, stride=4)
            weights.append(weight.numpy().astype(np.uint8))
            if stop % 8192 < args.prepare_batch_size or stop == len(indices):
                print(json.dumps({"stage": "prepare", "split": split,
                                  "done": stop, "total": len(indices)}), flush=True)
    destination = paths(args, split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, source_index=indices, seeds=seeds,
             geometry=np.concatenate(guides).astype(np.float32),
             new_visible=np.concatenate(weights).astype(np.uint8))
    stage1.save_json(destination.with_suffix(".json"), {
        "version": VERSION, "split": split, "rows": len(indices),
        "excluded_nonfinite_executed_commands": int((~finite).sum()),
        "vae_sha256": meta["vae_sha256"], "motion_checkpoint": str(args.motion),
        "identity_max_distance_error_m": identity_max,
        "new_visible": "GT hit and predicted warp miss; training weight only"})


class GeometryLatents(Dataset):
    def __init__(self, args, split, overfit=False):
        geometry_info = json.loads(paths(args, split).with_suffix(".json").read_text())
        meta = latent_metadata()
        if geometry_info["vae_sha256"] != meta["vae_sha256"]:
            raise ValueError("Stale geometry latent cache")
        geometry = np.load(paths(args, split))
        cache = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
        lookup = {int(src): i for i, src in enumerate(cache["source_index"])}
        selected = np.array([lookup[int(src)] for src in geometry["source_index"]])
        take = slice(0, min(128, len(selected))) if overfit else slice(None)
        selected = selected[take]
        self.previous = torch.from_numpy(cache["previous"][selected].copy() * meta["scaling_factor"])
        self.target = torch.from_numpy(cache["target"][selected].copy() * meta["scaling_factor"])
        self.actions = torch.from_numpy(cache["actions"][selected].copy())
        self.state = torch.from_numpy(stage1.causal_state(cache["state"][selected]))
        self.geometry = torch.from_numpy(geometry["geometry"][take].copy())
        self.new_visible = torch.from_numpy(geometry["new_visible"][take].copy().astype(np.float32))
        self.source_index = geometry["source_index"][take].copy()
        self.seeds = geometry["seeds"][take].copy()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return (self.previous[index], self.target[index], self.actions[index],
                self.state[index], self.geometry[index], self.new_visible[index])


def residual_scale(data):
    value = float((data.target - data.geometry).std().item())
    if not math.isfinite(value) or value < 1e-4:
        raise ValueError("Invalid geometry residual standard deviation")
    return 1.0 / value


def region_scores(prediction, target, warped, threshold):
    """Radial error and hit recall on reprojectable and newly visible GT rays."""
    true_hit = target[1, :, :18] > 0
    warp_hit = warped[1, :, :18] > 0
    predicted_hit = prediction[1, :, :18] > threshold
    truth_range = np.clip((target[0, :, :18] + 1) * 5, 0, 10)
    predicted_range = np.clip((prediction[0, :, :18] + 1) * 5, 0, 10)
    result = {}
    for label, region in (("reprojectable", true_hit & warp_hit),
                          ("new_visible", true_hit & ~warp_hit)):
        result[label] = {"count": int(region.sum()),
                         "range_abs_sum_m": float(np.abs(
                             predicted_range[region] - truth_range[region]).sum()),
                         "hit_recalled": int(predicted_hit[region].sum())}
    return result


@torch.no_grad()
def noise_score(model, scheduler, data, scale, batch_size):
    model.eval()
    torch.manual_seed(2917)
    total = elements = 0.0
    for previous, target, actions, state, geometry, new_visible in DataLoader(
            data, batch_size=batch_size):
        previous, target, actions, state, geometry = (
            x.to(stage1.DEVICE) for x in (previous, target, actions, state, geometry))
        clean = (target - geometry) * scale
        noise = torch.randn_like(clean)
        timestep = torch.randint(0, 1000, (len(clean),),
                                 device=stage1.DEVICE, dtype=torch.long)
        prediction = model(scheduler.add_noise(clean, noise, timestep), previous,
                           geometry, actions, state, timestep)
        total += F.mse_loss(prediction, noise, reduction="sum").item()
        elements += noise.numel()
    return total / elements


def train(args):
    meta = latent_metadata()
    stage1.seed_everything(args.seed)
    data = GeometryLatents(args, "train", overfit=args.overfit)
    if args.overfit:
        validation = data
    else:
        all_validation = GeometryLatents(args, "val")
        manifest = json.loads((args.manifest / "sample_manifest.json").read_text())
        positions = {int(src): i for i, src in enumerate(all_validation.source_index)}
        validation = Subset(all_validation, [positions[int(row["source_index"])]
            for row in manifest["splits"]["val"]])
    scale = residual_scale(data if args.overfit else GeometryLatents(args, "train"))
    run = args.out / ("overfit_128" if args.overfit else "full")
    run.mkdir(parents=True, exist_ok=True)
    model = GeometryResidualUNet().to(stage1.DEVICE).float()
    source_step = initialize_from_unet(model, args.unet)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = stage1.DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    loader = stage1.infinite(DataLoader(data, batch_size=args.batch_size,
                                       shuffle=True, drop_last=True))
    config = {"version": VERSION, "steps": args.steps, "batch_size": args.batch_size,
              "lr": args.lr, "source_unet": str(args.unet), "source_unet_step": source_step,
              "vae_sha256": meta["vae_sha256"], "motion": str(args.motion),
              "residual_scale": scale, "train_samples": len(data),
              "validation_samples": len(validation),
              "inputs": "noisy geometry residual, previous latent, action-predicted geometry latent, normalized future 10x3 actions, causal previous ego",
              "target": "DDPM epsilon for (next latent - geometry latent) * residual_scale",
              "new_visible_weight": args.new_visible_weight}
    stage1.save_json(run / "config.json", config)
    best = math.inf
    history = []
    began = time.monotonic()
    for step in range(1, args.steps + 1):
        previous, target, actions, state, geometry, new_visible = (
            x.to(stage1.DEVICE) for x in next(loader))
        clean = (target - geometry) * scale
        noise = torch.randn_like(clean)
        timestep = torch.randint(0, 1000, (len(clean),),
                                 device=stage1.DEVICE, dtype=torch.long)
        model.train()
        prediction = model(scheduler.add_noise(clean, noise, timestep), previous,
                           geometry, actions, state, timestep)
        weight = 1.0 + args.new_visible_weight * new_visible
        loss = ((prediction - noise).square() * weight).mean() / weight.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "train_weighted_noise_mse": loss.item(),
                              "elapsed_min": round((time.monotonic() - began) / 60, 2)}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            score = noise_score(model, scheduler, validation, scale, args.eval_batch_size)
            history.append({"step": step, "validation_noise_mse": score})
            if score < best:
                best = score
                temp = run / "best.tmp"
                torch.save({"model": model.state_dict(), "step": step,
                            "validation_noise_mse": score, "config": config}, temp)
                temp.replace(run / "best.pt")
            stage1.save_json(run / "history.json", history)
            print(json.dumps({"step": step, "validation_noise_mse": score,
                              "best": best}), flush=True)


@torch.no_grad()
def generate(model, scheduler, previous, geometry, actions, state, scale, seed, steps=20):
    torch.manual_seed(seed)
    scheduler.set_timesteps(steps, device=stage1.DEVICE)
    residual = torch.randn_like(geometry)
    for timestep in scheduler.timesteps:
        noise = model(residual, previous, geometry, actions, state, timestep)
        residual = scheduler.step(noise, timestep, residual, eta=0.0).prev_sample
    return geometry + residual / scale


@torch.no_grad()
def evaluate(args, split):
    saved = torch.load(args.out / "full" / "best.pt", map_location="cpu", weights_only=False)
    meta = latent_metadata()
    if saved["config"]["vae_sha256"] != meta["vae_sha256"]:
        raise ValueError("Checkpoint and VAE differ")
    data = GeometryLatents(args, split)
    manifest = json.loads((args.manifest / "sample_manifest.json").read_text())
    lookup = {int(src): i for i, src in enumerate(data.source_index)}
    selected = [(row, lookup[int(row["source_index"])]) for row in manifest["splits"][split]
                if int(row["source_index"]) in lookup]
    if len(selected) != len(manifest["splits"][split]):
        raise ValueError("Fixed representative manifest is not fully in geometry cache")
    if split == "val":
        alphas = tuple(sorted(set(args.residual_alphas + [1.0])))
    else:
        chosen = json.loads((args.out / "validation_alpha_selection.json").read_text())
        alphas = tuple(sorted({float(chosen["selected_alpha"]), 1.0}))
    model = GeometryResidualUNet().to(stage1.DEVICE).float().eval()
    model.load_state_dict(saved["model"])
    vae = stage1.load_circular_vae()
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    old_unet = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    old_checkpoint = stage1.load_model(args.unet, old_unet)
    motion = load_motion(args.motion)
    rays = {seed: load_rays(args.raw_root, split, seed)
            for seed in sorted({row["seed"] for row, _ in selected})}
    source = np.array([row["source_index"] for row, _ in selected], dtype=np.int64)
    with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
        current = fetch(h5, "prev_range_values", source)
        target = fetch(h5, "range_values", source)
        drone = fetch(h5, "prev_drone_state", source)
        commands = fetch(h5, "action_sequence", source)
    rows = []
    for seed in sorted(rays):
        positions = [i for i, (row, _) in enumerate(selected) if row["seed"] == seed]
        for start in range(0, len(positions), args.eval_batch_size):
            chosen = positions[start:start + args.eval_batch_size]
            indices = [selected[i][1] for i in chosen]
            previous = data.previous[indices].to(stage1.DEVICE)
            geometry = data.geometry[indices].to(stage1.DEVICE)
            actions = data.actions[indices].to(stage1.DEVICE)
            state = data.state[indices].to(stage1.DEVICE)
            generated = generate(model, scheduler, previous, geometry, actions, state,
                                 saved["config"]["residual_scale"],
                                 42 + seed * 10000 + start, args.ddim_steps)
            ordinary = stage1.generate(old_unet, scheduler, previous, actions, state,
                                       42 + seed * 10000 + start, num_steps=args.ddim_steps)
            variants = {alpha: geometry + alpha * (generated - geometry)
                        for alpha in alphas}
            decoded_variants = {alpha: vae.decode(z / meta["scaling_factor"]).sample.cpu().numpy()
                                for alpha, z in variants.items()}
            ordinary_decoded = vae.decode(ordinary / meta["scaling_factor"]).sample.cpu().numpy()
            geo_decoded = vae.decode(geometry / meta["scaling_factor"]).sample.cpu().numpy()
            latent_pred = {alpha: z.cpu().numpy() for alpha, z in variants.items()}
            for local, location in enumerate(chosen):
                index = indices[local]
                ray, azimuth, elevation = rays[seed]
                truth = target[location]
                state_np = drone[location].astype(np.float64)
                predicted_pose = next_state(state_np, commands[location].astype(np.float64), motion)
                warped = warp_frame(current[location], transform_from_initial(
                    state_np, predicted_pose), ray, azimuth, elevation)
                record = {**selected[location][0],
                          "copy": lidar_metric(current[location], truth, ray, 0),
                          "geometry_direct": lidar_metric(warped, truth, ray, 0),
                          "geometry_vae": lidar_metric(geo_decoded[local], truth, ray, threshold),
                          "pure_diffusion": lidar_metric(ordinary_decoded[local], truth, ray, threshold),
                          "latent_mse": {f"alpha_{alpha:g}": float(np.square(
                              latent_pred[alpha][local] - data.target[index].numpy()).mean())
                                         for alpha in alphas}}
                for alpha in alphas:
                    record[f"prediction_alpha_{alpha:g}"] = lidar_metric(
                        decoded_variants[alpha][local], truth, ray, threshold)
                record["regions"] = {
                    "geometry_direct": region_scores(warped, truth, warped, 0),
                    "pure_diffusion": region_scores(ordinary_decoded[local], truth,
                                                    warped, threshold)}
                for alpha in alphas:
                    record["regions"][f"prediction_alpha_{alpha:g}"] = region_scores(
                        decoded_variants[alpha][local], truth, warped, threshold)
                rows.append(record)
        print(json.dumps({"split": split, "seed": seed, "evaluated": len(positions)}), flush=True)
    methods = ("copy", "geometry_direct", "geometry_vae", "pure_diffusion",
               *(f"prediction_alpha_{alpha:g}" for alpha in alphas))
    summaries = {}
    for label, group in (("all", rows), *(
            (f"seed_{seed}", [r for r in rows if r["seed"] == seed])
            for seed in sorted(rays))):
        paired = [r for r in group if all(not r[m]["empty_cloud"] for m in methods)]
        summaries[label] = {"samples": len(group), "paired_nonempty": len(paired),
            "latent_mse": {f"alpha_{alpha:g}": float(np.mean([
                r["latent_mse"][f"alpha_{alpha:g}"] for r in group])) for alpha in alphas},
            "metrics": {m: summarize([r[m] for r in group]) for m in methods},
            "paired_cd_paper_m2": {m: float(np.mean([r[m]["cd_paper_m2"] for r in paired]))
                                   if paired else None for m in methods}}
        summaries[label]["regions"] = {}
        for region in ("reprojectable", "new_visible"):
            count = sum(r["regions"]["pure_diffusion"][region]["count"] for r in group)
            summaries[label]["regions"][region] = {
                "gt_hit_count": count,
                "methods": {m: {
                    "radial_mae_m": sum(r["regions"][m][region]["range_abs_sum_m"]
                                        for r in group) / max(count, 1),
                    "hit_recall": sum(r["regions"][m][region]["hit_recalled"]
                                      for r in group) / max(count, 1)}
                    for m in ("geometry_direct", "pure_diffusion",
                              *(f"prediction_alpha_{alpha:g}" for alpha in alphas))}}
    selected_alpha = None
    if split == "val":
        selected_alpha = min(alphas, key=lambda alpha:
            summaries["all"]["paired_cd_paper_m2"][f"prediction_alpha_{alpha:g}"])
        stage1.save_json(args.out / "validation_alpha_selection.json", {
            "selected_alpha": selected_alpha, "candidate_alphas": alphas,
            "selection": "minimum paired validation squared Chamfer",
            "validation_paired_cd_paper_m2": summaries["all"]["paired_cd_paper_m2"]})
    else:
        selected_alpha = float(json.loads((args.out /
            "validation_alpha_selection.json").read_text())["selected_alpha"])
    report = {"split": split, "samples": len(rows), "checkpoint_step": saved["step"],
              "source_unet_checkpoint_step": old_checkpoint["step"],
              "manifest_sha256": manifest_hash(manifest), "vae_sha256": meta["vae_sha256"],
              "causal_geometry": True, "future_pose_used": False,
              "ddim_steps": args.ddim_steps, "mask_threshold": threshold,
              "residual_alphas": alphas, "selected_alpha_from_validation": selected_alpha,
              "cd_paper_m2": "sum of bidirectional squared nearest-point means; m^2",
              "summary": summaries, "rows": rows}
    stage1.save_json(args.out / f"{split}_one_step.json", report)
    print(json.dumps({"split": split, "summary": summaries["all"]["paired_cd_paper_m2"]}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--out", type=Path, default=stage1.OUT / "geometry_residual_diffusion")
    parser.add_argument("--manifest", type=Path, default=stage1.OUT / "representative_baseline")
    parser.add_argument("--motion", type=Path, default=stage1.OUT / "epona_probe" / "motion_ridge.npz")
    parser.add_argument("--unet", type=Path, default=stage1.OUT / "world_circular_causal_8h" / "best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prepare-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--new-visible-weight", type=float, default=2.0)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--residual-alphas", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(not 0 < alpha <= 1 for alpha in args.residual_alphas):
        parser.error("Residual alphas must be in (0, 1]")
    stage1.DATA = args.data_root.expanduser().resolve()
    args.raw_root = (args.raw_root or stage1.DATA.parent).expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if args.command == "prepare":
        meta = latent_metadata()
        vae = stage1.load_circular_vae()
        motion = load_motion(args.motion)
        for split in ("train", "val", "test"):
            prepare_split(args, split, vae, motion, meta)
    elif args.command == "train":
        train(args)
    else:
        evaluate(args, args.split)


if __name__ == "__main__":
    main()
