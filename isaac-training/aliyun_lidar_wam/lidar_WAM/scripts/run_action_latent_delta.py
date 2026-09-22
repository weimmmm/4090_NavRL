"""Deterministic, action-conditioned next-LiDAR latent baseline.

Uses the same scaled circular-VAE latents, normalized PPO action chunks, and
causal ego features as the existing LaGen-style UNet. This is a diagnostic
baseline; it does not replace or modify the diffusion generator.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_geometry import frame_points, load_rays, warp_frame
from evaluate_autoregressive_5 import eligible_starts
from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch, manifest_hash
from try_epona_guided_diffusion import load_motion
from try_epona_motion import next_state, transform_from_initial


class ActionLatentDelta(nn.Module):
    """Small MLP that predicts a residual in the 540-dimensional scaled latent."""

    def __init__(self, width=1024):
        super().__init__()
        self.width = width
        self.features = nn.Sequential(
            nn.Linear(540 + 30 + 5, width), nn.LayerNorm(width), nn.SiLU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU(),
        )
        self.output = nn.Linear(width, 540)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, previous, actions, state):
        if previous.ndim != 4 or previous.shape[1:] != (4, 27, 5):
            raise ValueError("Expected scaled previous latent [B,4,27,5]")
        if actions.shape != (len(previous), 10, 3) or state.shape != (len(previous), 5):
            raise ValueError("Expected actions [B,10,3] and causal state [B,5]")
        features = torch.cat((previous.flatten(1), actions.flatten(1), state), dim=1)
        return previous + self.output(self.features(features)).reshape_as(previous)


def latent_metadata():
    meta = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())
    if any(meta.get(k) != v for k, v in stage1.circular_vae_identity().items()):
        raise ValueError("Circular VAE and latent cache weights do not match")
    return meta


@torch.no_grad()
def latent_mse(model, dataset, batch_size):
    model.eval()
    total = copied = elements = 0.0
    for previous, target, actions, state in DataLoader(dataset, batch_size=batch_size):
        previous, target, actions, state = (
            x.to(stage1.DEVICE) for x in (previous, target, actions, state))
        predicted = model(previous, actions, state)
        total += F.mse_loss(predicted, target, reduction="sum").item()
        copied += F.mse_loss(previous, target, reduction="sum").item()
        elements += target.numel()
    return {"predicted": total / elements, "copy": copied / elements}


def train(args):
    meta = latent_metadata()
    stage1.seed_everything(args.seed)
    all_train = stage1.Latents("train", stage1.OUT)
    train_data = Subset(all_train, range(min(128, len(all_train)))) if args.overfit else all_train
    validation = train_data if args.overfit else stage1.Latents("val", stage1.OUT)
    run_dir = args.out / ("overfit_128" if args.overfit else "full")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = ActionLatentDelta(args.width).to(stage1.DEVICE).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)
    if len(loader) == 0:
        raise ValueError("Batch size exceeds number of training examples")
    batches = iter(loader)
    config = {"model": "two-hidden-layer MLP residual", "width": args.width,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size,
              "max_steps": args.steps, "overfit": args.overfit,
              "train_samples": len(train_data), "validation_samples": len(validation),
              "action": "normalized_action_sequence [10,3] in recorded order",
              "state": stage1.CAUSAL_STATE_DEFINITION,
              "target": "scaled next latent minus scaled current latent, MSE",
              "vae_sha256": meta["vae_sha256"], "scaling_factor": meta["scaling_factor"]}
    stage1.save_json(run_dir / "config.json", config)
    best = math.inf
    history = []
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        previous, target, actions, state = (x.to(stage1.DEVICE) for x in batch)
        model.train()
        predicted = model(previous, actions, state)
        loss = F.mse_loss(predicted - previous, target - previous)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "train_delta_mse": loss.item(),
                              "elapsed_min": round((time.monotonic() - start) / 60, 2)}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            score = latent_mse(model, validation, args.eval_batch_size)
            entry = {"step": step, "validation_latent_mse": score["predicted"],
                     "validation_copy_mse": score["copy"]}
            history.append(entry)
            if score["predicted"] < best:
                best = score["predicted"]
                temp = run_dir / "best.tmp"
                torch.save({"model": model.state_dict(), "step": step,
                            "width": args.width, "config": config,
                            "validation": score}, temp)
                temp.replace(run_dir / "best.pt")
            stage1.save_json(run_dir / "history.json", history)
            print(json.dumps({**entry, "best_validation_latent_mse": best}), flush=True)
    return run_dir / "best.pt"


def load_delta(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    meta = latent_metadata()
    if saved["config"]["vae_sha256"] != meta["vae_sha256"]:
        raise ValueError("Deterministic checkpoint uses a different VAE")
    model = ActionLatentDelta(saved["width"]).to(stage1.DEVICE).float().eval()
    model.load_state_dict(saved["model"])
    return model, saved


def paired_summary(rows, methods):
    groups = {"all": rows,
              "early_frame_1_2": [r for r in rows if r["frame_idx"] <= 2],
              "later_frame_3_plus": [r for r in rows if r["frame_idx"] > 2]}
    groups.update({f"seed_{seed}": [r for r in rows if r["seed"] == seed]
                   for seed in sorted({r["seed"] for r in rows})})
    result = {}
    for name, group in groups.items():
        if not group:
            continue
        paired = [r for r in group if all(not r[m]["empty_cloud"] for m in methods)]
        result[name] = {"samples": len(group), "paired_nonempty": len(paired),
                        "latent_mse": {m: float(np.mean([r[m]["latent_mse"] for r in group]))
                                       for m in ("copy", "deterministic", "diffusion")},
                        "methods": {m: summarize([r[m] for r in group]) for m in methods},
                        "paired_cd_paper_m2": {m: float(np.mean([r[m]["cd_paper_m2"]
                                                              for r in paired])) if paired else None
                                               for m in methods}}
    return result


def add_metrics(record, name, image, target, rays, threshold, latent=None, gt_latent=None):
    values = lidar_metric(image, target, rays, threshold)
    if latent is not None:
        values["latent_mse"] = float(np.square(latent - gt_latent).mean())
    record[name] = values


def plot_comparison(out, examples):
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 4, figsize=(16, 4 * len(examples)), squeeze=False)
    for row, example in enumerate(examples):
        rays, frames, title = example
        for col, (label, frame) in enumerate(zip(("Input", "GT", "Deterministic", "Diffusion"), frames)):
            points = frame_points(frame, rays, 1.5 if col >= 2 else 0)
            ax = axes[row, col]
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], s=0.3)
            ax.set(xlim=(-10, 10), ylim=(-10, 10), aspect="equal", title=f"{title}: {label}")
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate_one(args, split, model, saved, meta):
    manifest = json.loads((args.manifest / "sample_manifest.json").read_text())
    rows = manifest["splits"][split]
    cache = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
    lookup = {int(index): i for i, index in enumerate(cache["source_index"])}
    source = np.array([r["source_index"] for r in rows], dtype=np.int64)
    if any(int(index) not in lookup for index in source):
        raise ValueError("Fixed manifest includes an invalid transition")
    pos = np.array([lookup[int(index)] for index in source])
    with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
        if not np.all(h5["step_delta"][source] == 10) or not np.all(h5["action_mask"][source]):
            raise ValueError("Fixed manifest contains a non-ten-step transition")
        arrays = {key: fetch(h5, key, source) for key in (
            "prev_range_values", "range_values", "normalized_action_sequence",
            "prev_ego_feats", "prev_drone_state", "action_sequence")}
    if not np.isfinite(arrays["normalized_action_sequence"]).all():
        raise ValueError("Nonfinite normalized action in fixed manifest")
    rays = {seed: load_rays(args.raw_root, split, seed)
            for seed in sorted({r["seed"] for r in rows})}
    motion = load_motion(args.motion)
    vae = stage1.load_circular_vae()
    unet = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    unet_saved = stage1.load_model(args.unet, unet)
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    scale = meta["scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    report_rows = [dict(row) for row in rows]
    examples = []
    for seed in sorted(rays):
        locations = np.array([i for i, r in enumerate(rows) if r["seed"] == seed])
        for offset in range(0, len(locations), args.eval_batch_size):
            chosen = locations[offset:offset + args.eval_batch_size]
            previous = torch.from_numpy(cache["previous"][pos[chosen]].copy() * scale).to(stage1.DEVICE)
            truth = cache["target"][pos[chosen]] * scale
            actions = torch.from_numpy(arrays["normalized_action_sequence"][chosen]).to(stage1.DEVICE)
            state = torch.from_numpy(stage1.causal_state(arrays["prev_ego_feats"][chosen])).to(stage1.DEVICE)
            det = model(previous, actions, state)
            diff = stage1.generate(unet, scheduler, previous, actions, state,
                                   seed=42 + seed * 10000 + offset, num_steps=20)
            det_image = vae.decode(det / scale).sample.cpu().numpy()
            diff_image = vae.decode(diff / scale).sample.cpu().numpy()
            det_latent, diff_latent, previous_latent = (
                x.cpu().numpy() for x in (det, diff, previous))
            for j, i in enumerate(chosen):
                ray_grid, azimuth, elevation = rays[seed]
                target = arrays["range_values"][i]
                current = arrays["prev_range_values"][i]
                predicted_state = next_state(arrays["prev_drone_state"][i].astype(np.float64),
                                             arrays["action_sequence"][i].astype(np.float64), motion)
                warp = warp_frame(current, transform_from_initial(
                    arrays["prev_drone_state"][i], predicted_state), ray_grid, azimuth, elevation)
                record = report_rows[i]
                add_metrics(record, "copy", current, target, ray_grid, 0,
                            previous_latent[j], truth[j])
                add_metrics(record, "geometry", warp, target, ray_grid, 0)
                add_metrics(record, "deterministic", det_image[j], target, ray_grid,
                            threshold, det_latent[j], truth[j])
                add_metrics(record, "diffusion", diff_image[j], target, ray_grid,
                            threshold, diff_latent[j], truth[j])
                if len(examples) < 6 and (offset == 0 or len(examples) < 2):
                    examples.append((ray_grid, (current, target, det_image[j], diff_image[j]),
                                     f"seed {seed}, frame {record['frame_idx']}"))
        print(json.dumps({"split": split, "seed": seed, "evaluated": len(locations)}), flush=True)
    methods = ("copy", "geometry", "deterministic", "diffusion")
    report = {"split": split, "samples": len(rows), "manifest_sha256": manifest_hash(manifest),
              "checkpoint_step": saved["step"], "unet_checkpoint_step": unet_saved["step"],
              "vae_sha256": meta["vae_sha256"], "mask_logit_threshold": threshold,
              "latent_mse_unit": "squared scaled latent value; not metres",
              "cd_paper_m2": "bidirectional squared Chamfer, sum of direction means; empty clouds penalized",
              "paired_cd_paper_m2": "same rows with all four clouds nonempty",
              "summary": paired_summary(report_rows, methods), "rows": report_rows}
    stage1.save_json(args.out / f"{split}_one_step.json", report)
    plot_comparison(args.out / f"{split}_comparison.png", examples)
    return report


@torch.no_grad()
def evaluate_five(args, split, model, saved, meta):
    manifest = json.loads((args.manifest / "sample_manifest.json").read_text())
    cache = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
    with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
        # The shared helper names its selected split "test" internally. Give it
        # the requested fixed rows so validation and test use the same rules.
        selected_manifest = {"splits": {"test": manifest["splits"][split]}}
        starts, chains, rejection = eligible_starts(selected_manifest,
                                                    cache["source_index"], h5)
        if not starts:
            raise ValueError(f"No linked five-frame sequences: {rejection}")
        source = np.array([row["source_index"] for row in starts], dtype=np.int64)
        actions = np.stack([fetch(h5, "normalized_action_sequence", chains[:, h])
                            for h in range(5)], axis=1)
        targets = np.stack([fetch(h5, "range_values", chains[:, h])
                            for h in range(5)], axis=1)
        state = stage1.causal_state(fetch(h5, "prev_ego_feats", source))
        initial_frames = fetch(h5, "prev_range_values", source)
        initial_drone_states = fetch(h5, "prev_drone_state", source)
        world_actions = np.stack([fetch(h5, "action_sequence", chains[:, h])
                                  for h in range(5)], axis=1)
    finite_geometry = (np.isfinite(world_actions).all(axis=(1, 2, 3)) &
                       np.isfinite(initial_drone_states).all(axis=1))
    excluded_geometry = int((~finite_geometry).sum())
    keep = np.flatnonzero(finite_geometry)
    starts = [starts[int(i)] for i in keep]
    chains, source, actions, targets, state = (x[keep] for x in
        (chains, source, actions, targets, state))
    initial_frames, initial_drone_states, world_actions = (x[keep] for x in
        (initial_frames, initial_drone_states, world_actions))
    if not starts:
        raise ValueError("No five-frame sequences with finite executed commands")
    lookup = {int(index): i for i, index in enumerate(cache["source_index"])}
    previous = cache["previous"][[lookup[int(i)] for i in source]] * meta["scaling_factor"]
    truth_latents = np.stack([cache["target"][[lookup[int(i)] for i in chains[:, h]]]
                              for h in range(5)], axis=1) * meta["scaling_factor"]
    rays = {seed: load_rays(args.raw_root, split, seed)
            for seed in sorted({row["seed"] for row in starts})}
    vae = stage1.load_circular_vae()
    unet = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    unet_saved = stage1.load_model(args.unet, unet)
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    motion = load_motion(args.motion)
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    report_rows = []
    for seed in sorted(rays):
        locations = np.array([i for i, r in enumerate(starts) if r["seed"] == seed])
        for offset in range(0, len(locations), args.eval_batch_size):
            chosen = locations[offset:offset + args.eval_batch_size]
            det = torch.from_numpy(previous[chosen]).to(stage1.DEVICE)
            diff = det.clone()
            ego = torch.from_numpy(state[chosen]).to(stage1.DEVICE)
            predicted_states = initial_drone_states[chosen].astype(np.float64).copy()
            for horizon in range(5):
                action = torch.from_numpy(actions[chosen, horizon]).to(stage1.DEVICE)
                det = model(det, action, ego)
                diff = stage1.generate(unet, scheduler, diff, action, ego,
                                       seed=42 + seed * 10000 + offset + horizon * 100000,
                                       num_steps=20)
                det_image = vae.decode(det / meta["scaling_factor"]).sample.cpu().numpy()
                diff_image = vae.decode(diff / meta["scaling_factor"]).sample.cpu().numpy()
                det_z, diff_z = det.cpu().numpy(), diff.cpu().numpy()
                for j, i in enumerate(chosen):
                    target = targets[i, horizon]
                    truth = truth_latents[i, horizon]
                    record = {"source_index": int(source[i]), "seed": seed,
                              "frame_idx": starts[i]["frame_idx"], "horizon": horizon + 1,
                              "target_source_index": int(chains[i, horizon])}
                    predicted_states[j] = next_state(predicted_states[j],
                                                      world_actions[i, horizon].astype(np.float64), motion)
                    ray_grid, azimuth, elevation = rays[seed]
                    warp = warp_frame(initial_frames[i],
                                      transform_from_initial(initial_drone_states[i],
                                                             predicted_states[j]),
                                      ray_grid, azimuth, elevation)
                    add_metrics(record, "copy_initial", initial_frames[i], target, ray_grid, 0,
                                previous[i], truth)
                    add_metrics(record, "geometry", warp, target, ray_grid, 0)
                    add_metrics(record, "deterministic", det_image[j], target, ray_grid,
                                threshold, det_z[j], truth)
                    add_metrics(record, "diffusion", diff_image[j], target, ray_grid,
                                threshold, diff_z[j], truth)
                    report_rows.append(record)
        print(json.dumps({"split": split, "seed": seed, "five_step_starts": len(locations)}), flush=True)
    summary = {}
    for horizon in range(1, 6):
        selected = [r for r in report_rows if r["horizon"] == horizon]
        group = {"all": selected}
        group.update({f"seed_{seed}": [r for r in selected if r["seed"] == seed]
                      for seed in sorted(rays)})
        summary[str(horizon)] = {}
        for name, rows in group.items():
            methods = ("copy_initial", "geometry", "deterministic", "diffusion")
            paired = [r for r in rows if all(not r[m]["empty_cloud"] for m in methods)]
            summary[str(horizon)][name] = {
                "samples": len(rows), "paired_nonempty": len(paired),
                "latent_mse": {m: float(np.mean([r[m]["latent_mse"] for r in rows]))
                               for m in ("copy_initial", "deterministic", "diffusion")},
                "methods": {m: summarize([r[m] for r in rows])
                            for m in methods},
                "paired_cd_paper_m2": {m: float(np.mean([r[m]["cd_paper_m2"] for r in paired]))
                                        if paired else None for m in methods}}
    report = {"split": split, "starts": len(starts), "manifest_sha256": manifest_hash(manifest),
              "rejected": rejection, "excluded_nonfinite_geometry_commands": excluded_geometry,
              "checkpoint_step": saved["step"],
              "unet_checkpoint_step": unet_saved["step"],
              "state_rollout": "initial causal state held fixed for both predictors",
              "actions": "five successive, distinct recorded normalized 10x3 chunks",
              "future_observations": "ground truth used only for scoring",
              "latent_mse_unit": "squared scaled latent value; not metres",
              "cd_paper_m2": "bidirectional squared Chamfer, sum of direction means",
              "summary": summary, "rows": report_rows}
    stage1.save_json(args.out / f"{split}_autoregressive_5.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--out", type=Path, default=stage1.OUT / "action_latent_delta")
    parser.add_argument("--manifest", type=Path, default=stage1.OUT / "representative_baseline")
    parser.add_argument("--unet", type=Path, default=stage1.OUT / "world_circular_causal_8h" / "best.pt")
    parser.add_argument("--motion", type=Path, default=stage1.OUT / "epona_probe" / "motion_ridge.npz")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    args.raw_root = (args.raw_root or stage1.DATA.parent).expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if args.command == "train":
        train(args)
    else:
        model, saved = load_delta(args.out / "full" / "best.pt")
        meta = latent_metadata()
        one = evaluate_one(args, args.split, model, saved, meta)
        five = evaluate_five(args, args.split, model, saved, meta)
        print(json.dumps({"split": args.split,
                          "one_step": one["summary"]["all"]["paired_cd_paper_m2"],
                          "five_step": five["summary"]["5"]["all"]["paired_cd_paper_m2"]}),
              flush=True)


if __name__ == "__main__":
    main()
