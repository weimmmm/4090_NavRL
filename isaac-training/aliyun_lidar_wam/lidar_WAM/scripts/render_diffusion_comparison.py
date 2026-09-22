"""Render matched test-set point clouds from existing diffusion checkpoints.

Uses the same sample manifest, batch membership, DDIM noise, and metrics as
evaluate_representative.py. No training or HDF5 writes occur.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_wam.runner import stage1
from lidar_wam.runner.executed_residual import condition_stats
from lidar_wam.runner.lidar_geometry import frame_points, load_rays, predict_transform, warp_frame
from lidar_wam.runner.nwm_predictor import NavRLCDiT, model_kwargs
from scripts.evaluate_representative import load_selected, metric
from third_party.nwm.diffusion import create_diffusion


METHODS = ("lagen_unet", "lagen_residual", "nwm")
LABELS = ("Previous", "True next", "Velocity warp", "LaGen UNet", "LaGen residual", "NWM CDiT")


def matched_batch(rows, sample_index, batch_size):
    item = next(i for i, row in enumerate(rows) if row["source_index"] == sample_index)
    seed = rows[item]["seed"]
    locations = np.array([i for i, row in enumerate(rows) if row["seed"] == seed])
    offset = int(np.flatnonzero(locations == item)[0])
    start = offset // batch_size * batch_size
    return item, seed, start, locations[start:start + batch_size], offset - start


@torch.no_grad()
def predict_batch(method, data, chosen, positions, arrays, seed, start, vae, scale):
    source = positions[chosen]
    previous = data.previous[source].to(stage1.DEVICE)
    state = data.state[source].to(stage1.DEVICE)
    if method == "lagen_unet":
        actions = torch.from_numpy(arrays["normalized_action_sequence"][chosen]).to(stage1.DEVICE)
        state = torch.from_numpy(stage1.causal_state(arrays["prev_ego_feats"][chosen])).to(stage1.DEVICE)
    else:
        actions = data.actions[source].to(stage1.DEVICE)
    noise_seed = 42 + seed * 10000 + start
    if method == "nwm":
        model = NavRLCDiT("CDiT-B/2").to(stage1.DEVICE).float().eval()
        saved = torch.load(stage1.OUT / "world_nwm_cdit_full" / "best.pt",
                           map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        diffusion = create_diffusion("ddim20")
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(noise_seed)
            noise = torch.randn_like(NavRLCDiT.pad_latent(previous))
        predicted = diffusion.ddim_sample_loop(
            model, noise.shape, noise=noise, clip_denoised=False,
            model_kwargs=model_kwargs(previous, actions, state),
            device=stage1.DEVICE, progress=False)
        latent = NavRLCDiT.crop_latent(predicted)
    else:
        residual = method == "lagen_residual"
        model = stage1.WorldModel(state_dim=11 if residual else 5).to(stage1.DEVICE).float().eval()
        run = "world_circular_executed_residual_full" if residual else "world_circular_causal_full"
        stage1.load_model(stage1.OUT / run / "best.pt", model)
        scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                          prediction_type="epsilon", clip_sample=False)
        latent = stage1.generate(
            model, scheduler, previous, actions, state, noise_seed,
            0.05 if residual else 1.0, 20,
            residual_scale=condition_stats()["residual_scale"] if residual else None)
    images = vae.decode(latent / scale).sample.cpu().numpy()
    del model
    return images


def draw(path, frames, rays, thresholds, title, scores):
    fig, axes = plt.subplots(3, len(LABELS), figsize=(23, 10), constrained_layout=True)
    for col, (label, frame) in enumerate(zip(LABELS, frames)):
        xyz = frame_points(frame, rays, thresholds[col])
        label = f"{label}\nChamfer {scores[col]:.3f} m" if col != 1 else f"{label}\nreference"
        axes[0, col].set_title(label, fontsize=12)
        for row, horizontal, vertical, ybounds in ((0, 0, 1, (-10, 10)),
                                                    (1, 0, 2, (-2, 9))):
            axis = axes[row, col]
            axis.scatter(xyz[:, horizontal], xyz[:, vertical], c=xyz[:, 2],
                         vmin=-2, vmax=9, cmap="viridis", s=2, linewidths=0,
                         alpha=0.85, rasterized=True)
            axis.set_xlim(-10, 10)
            axis.set_ylim(*ybounds)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.15)
            if col == 0:
                axis.set_ylabel("Y / m" if row == 0 else "Z / m")
            axis.set_xlabel("X / m")
        valid = frame[1, :, :18] > thresholds[col]
        distance = np.clip((frame[0, :, :18] + 1) * 5, 0, 10)
        axes[2, col].imshow(np.where(valid, distance, np.nan).T, origin="lower",
                            cmap="viridis", vmin=0, vmax=10, aspect="auto")
        axes[2, col].set_xlabel("Azimuth bin (0 to 360°)")
        if col == 0:
            axes[2, col].set_ylabel("Elevation bin")
    fig.suptitle(title + " | top: XY point cloud; middle: XZ point cloud; bottom: range image",
                 fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_overlay(path, frames, rays, thresholds, title, scores):
    truth = frame_points(frames[1], rays)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    for axis, col in zip(axes.flat, (2, 3, 4, 5)):
        predicted = frame_points(frames[col], rays, thresholds[col])
        axis.scatter(truth[:, 0], truth[:, 1], s=9, c="#222222", alpha=0.65,
                     linewidths=0, label="True next")
        axis.scatter(predicted[:, 0], predicted[:, 1], s=5, c="#ee5335",
                     alpha=0.55, linewidths=0, label="Prediction")
        axis.set_title(f"{LABELS[col]} — Chamfer {scores[col]:.3f} m")
        axis.set_xlim(-10, 10)
        axis.set_ylim(-10, 10)
        axis.set_aspect("equal")
        axis.set_xlabel("X / m")
        axis.set_ylabel("Y / m")
        axis.grid(alpha=0.15)
        axis.legend(loc="upper right", markerscale=2)
    fig.suptitle(title + " | XY overlay: true next in black, prediction in orange", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs="+", type=int, default=[4772, 9097])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--output", type=Path,
                        default=stage1.OUT / "representative_baseline" / "figures")
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    manifest = json.loads((stage1.OUT / "representative_baseline" / "sample_manifest.json").read_text())
    rows, data, positions, arrays = load_selected(manifest, args.split)
    expected = json.loads((stage1.OUT / "representative_baseline" /
                           f"summary_{args.split}.json").read_text())
    report_rows = {r["source_index"]: r for r in expected["rows"]}
    vae = stage1.load_circular_vae()
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())["scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR / "oracle_val.json").read_text())["selected_threshold"]
    for sample_index in args.indices:
        item, seed, start, chosen, within = matched_batch(rows, sample_index, args.batch_size)
        rays, azimuth, elevation = load_rays(stage1.DATA.parent, args.split, seed)
        previous = arrays["prev_range_values"][item]
        target = arrays["range_values"][item]
        velocity = warp_frame(previous, predict_transform(arrays["prev_drone_state"][item]),
                              rays, azimuth, elevation)
        predictions = {method: predict_batch(
            method, data, chosen, positions, arrays, seed, start, vae, scale)[within]
            for method in METHODS}
        frames = [previous, target, velocity] + [predictions[method] for method in METHODS]
        thresholds = [0, 0, 0, threshold, threshold, threshold]
        names = ["copy", None, "velocity_pose", *METHODS]
        scores = [None if name is None else metric(frame, target, rays, limit)["chamfer_m"]
                  for name, frame, limit in zip(names, frames, thresholds)]
        for name, score in zip(names, scores):
            if name and abs(score - report_rows[sample_index]["methods"][name]["chamfer_m"]) > 1e-4:
                raise ValueError(f"Saved metric mismatch for sample {sample_index}, {name}: {score}")
        path = args.output / f"{args.split}_seed{seed}_source{sample_index}_pointcloud.png"
        draw(path, frames, rays, thresholds,
             f"{args.split} seed {seed}, frame {rows[item]['frame_idx']}, source {sample_index}", scores)
        overlay = args.output / f"{args.split}_seed{seed}_source{sample_index}_overlay.png"
        draw_overlay(overlay, frames, rays, thresholds,
                     f"{args.split} seed {seed}, frame {rows[item]['frame_idx']}, source {sample_index}",
                     scores)
        print(json.dumps({"figure": str(path), "overlay": str(overlay), "chamfer_m": dict(zip(
            ("copy", "velocity_pose", *METHODS), (scores[0], *scores[2:])))}), flush=True)


if __name__ == "__main__":
    main()
