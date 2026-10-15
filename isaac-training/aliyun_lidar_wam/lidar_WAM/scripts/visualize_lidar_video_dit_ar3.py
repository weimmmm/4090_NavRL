"""Visualize strict t+1:t+3 autoregressive LiDAR Video-DiT rollouts.

Only the initial three GT latents are supplied.  Predictions at t+1 and t+2
are shifted into the history used for later horizons; future GT is used only
for visualization and metrics.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_video_dit import (
    HistoryLatentDataset, configure_prediction_from_payload, sample_next)
from lidar_wam.runner.world_direct_horizon import frame_metrics


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["architecture"]
    model = LiDARVideoDiT(
        width=int(cfg["width"]), depth=int(cfg["depth"]),
        heads=int(cfg["heads"]), mlp_ratio=float(cfg["mlp_ratio"]),
    ).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    configure_prediction_from_payload(model, payload)
    return model, payload


def _location(dataset: HistoryLatentDataset, index: int):
    shard = bisect.bisect_right(dataset.ends, int(index))
    start = 0 if shard == 0 else int(dataset.ends[shard - 1])
    rows, latents = dataset._open(shard)
    return shard, rows[int(index) - start], latents


def _successor_map(dataset: HistoryLatentDataset, shard: int, cache: dict):
    if shard not in cache:
        if shard not in dataset._files:
            dataset._target_image(shard, 0)
        handle = dataset._files[shard]
        frames = handle["frames"] if "frames" in handle else handle
        tokens = frames["token"][:]
        cache[shard] = (
            frames,
            {token: row for row, token in enumerate(tokens)},
        )
    return cache[shard]


def select_sequences(dataset: HistoryLatentDataset, count: int, seed: int):
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(len(dataset))
    successor_cache = {}
    selected = []
    for index in candidates.tolist():
        shard, chain, latents = _location(dataset, index)
        frames, token_to_row = _successor_map(dataset, shard, successor_cache)
        future_rows = [int(chain[3])]
        valid = True
        for _ in range(2):
            token = frames["next_token"][future_rows[-1]]
            following = token_to_row.get(token)
            if following is None:
                valid = False
                break
            future_rows.append(int(following))
        if not valid:
            continue
        truth = np.asarray(frames["range_values"][future_rows], np.float32)
        # Keep examples whose forward hemisphere contains enough returns to
        # make a camera-like front-view comparison visually meaningful.
        azimuth_deg = np.arange(108, dtype=np.float32) * (360.0 / 108.0)
        signed_azimuth = (azimuth_deg + 180.0) % 360.0 - 180.0
        front = np.abs(signed_azimuth) <= 90.0
        if not all(int((frame[1, front, :18] > 0).sum()) >= 12
                   for frame in truth):
            continue
        selected.append({
            "index": int(index),
            "shard": int(shard),
            "history": np.asarray(latents[chain[:3]], np.float32),
            "future_rows": future_rows,
            "truth": truth,
        })
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError(f"found only {len(selected)} valid sequences")
    return selected


def physical_range(frame, prediction: bool):
    hit_threshold = 1.5 if prediction else 0.0
    hit = frame[1, :, :18] > hit_threshold
    distance = np.clip((frame[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
    return np.where(hit, distance, np.nan)


def front_view_range(frame, prediction: bool):
    """Conventional forward depth map and its azimuth coordinates."""
    azimuth_deg = np.arange(108, dtype=np.float32) * (360.0 / 108.0)
    signed = (azimuth_deg + 180.0) % 360.0 - 180.0
    chosen = np.flatnonzero(np.abs(signed) <= 60.0)
    chosen = chosen[np.argsort(signed[chosen])]
    return physical_range(frame, prediction)[chosen].T, signed[chosen]


def point_overlay(ax, truth, prediction, title, cd):
    gt = stage1.to_points(truth, 0.0)
    pred = stage1.to_points(prediction, 1.5)
    if len(gt):
        ax.scatter(gt[:, 0], gt[:, 1], s=7, c="#2878b5", alpha=.75)
    if len(pred):
        ax.scatter(pred[:, 0], pred[:, 1], s=8, c="#d84a3a", marker="x",
                   alpha=.75)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.18)
    ax.set_title(f"{title} | CD={cd:.3f} m²", fontsize=9)
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    dataset = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=False)
    sequences = select_sequences(dataset, args.count, args.seed)
    history = torch.from_numpy(np.stack([
        row["history"] for row in sequences])).to(device)
    model, payload = load_model(args.checkpoint, device)
    vae = stage1.load_circular_vae().to(device).eval()
    scale = float(dataset.vae_metadata["scaling_factor"])
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)

    predictions = []
    rollout_history = history
    for _ in range(3):
        prediction = sample_next(
            model, rollout_history, args.flow_steps, generator=generator,
            num_samples=args.flow_ensemble_size)
        predictions.append(prediction)
        rollout_history = torch.cat(
            (rollout_history[:, 1:], prediction[:, None]), dim=1)
    predicted_latents = torch.stack(predictions, dim=1)
    flat = predicted_latents.flatten(0, 1).float() / scale
    predicted_images = vae.decode(flat).sample.float().cpu().numpy().reshape(
        args.count, 3, 2, 108, 20)
    truth_images = np.stack([row["truth"] for row in sequences])

    records = []
    for sample, sequence in enumerate(sequences):
        horizons = []
        for horizon in range(3):
            metric = frame_metrics(
                predicted_images[sample, horizon], truth_images[sample, horizon])
            horizons.append({
                "horizon": horizon + 1,
                "cd_paper_m2": float(metric["cd_paper_m2"]),
                "valid_range_mae_m": (
                    float(metric["valid_range_abs_error_sum_m"])
                    / max(float(metric["valid_range_count"]), 1.0)),
                "gt_empty": bool(metric["gt_empty"]),
            })
        records.append({
            "index": sequence["index"], "shard": sequence["shard"],
            "future_rows": sequence["future_rows"], "horizons": horizons,
        })

    args.output.mkdir(parents=True, exist_ok=True)
    point_path = args.output / "ar3_pointcloud_overlay.png"
    figure, axes = plt.subplots(args.count, 3,
                                figsize=(12, 3.55 * args.count), dpi=170)
    axes = np.asarray(axes).reshape(args.count, 3)
    for sample in range(args.count):
        for horizon in range(3):
            point_overlay(
                axes[sample, horizon], truth_images[sample, horizon],
                predicted_images[sample, horizon],
                f"sample {records[sample]['index']} · t+{horizon+1}",
                records[sample]["horizons"][horizon]["cd_paper_m2"])
    handles = [
        plt.Line2D([], [], color="#2878b5", marker="o", linestyle="",
                   label="GT"),
        plt.Line2D([], [], color="#d84a3a", marker="x", linestyle="",
                   label="autoregressive prediction"),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2,
                  bbox_to_anchor=(.5, .968), frameon=False)
    figure.suptitle(
        "Future-DiT strict autoregressive rollout: point-cloud overlay\n"
        "t+2 and t+3 receive predicted history only (GT is comparison only)",
        y=.997, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .935))
    figure.savefig(point_path, bbox_inches="tight")
    plt.close(figure)

    range_path = args.output / "ar3_range_image_comparison.png"
    figure, axes = plt.subplots(
        2 * args.count, 3, figsize=(14, 2.15 * args.count), dpi=170,
        sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(2 * args.count, 3)
    image = None
    for sample in range(args.count):
        for horizon in range(3):
            for offset, (frame, label, prediction) in enumerate((
                    (truth_images[sample, horizon], "GT", False),
                    (predicted_images[sample, horizon], "Prediction", True))):
                ax = axes[2 * sample + offset, horizon]
                image = ax.imshow(
                    physical_range(frame, prediction).T, origin="lower",
                    aspect="auto", vmin=0, vmax=10, cmap="viridis")
                ax.set_title(
                    f"sample {records[sample]['index']} · t+{horizon+1} · {label}",
                    fontsize=8)
                ax.set_ylabel("elevation bin", fontsize=7)
                if sample == args.count - 1 and offset == 1:
                    ax.set_xlabel("azimuth bin", fontsize=7)
    color_axis = figure.add_axes((.92, .13, .015, .72))
    figure.colorbar(image, cax=color_axis, label="range (m)")
    figure.suptitle(
        "Future-DiT strict autoregressive rollout: GT vs predicted range image",
        y=.995, fontsize=13)
    figure.subplots_adjust(top=.945, right=.895, hspace=.58, wspace=.2)
    figure.savefig(range_path, bbox_inches="tight")
    plt.close(figure)

    front_paths = []
    # Conventional grayscale depth visualization: near returns are dark,
    # distant returns are bright, and missing returns remain pure black.  Do not interpolate
    # missing rays or invent obstacle surfaces.
    front_cmap = plt.get_cmap("gray").copy()
    front_cmap.set_bad("black")
    for sample, record in enumerate(records):
        figure, axes = plt.subplots(2, 3, figsize=(15, 6.5), dpi=180,
                                    sharex=True, sharey=True)
        depth_plot = None
        for horizon in range(3):
            for row, (frame, label, prediction) in enumerate((
                    (truth_images[sample, horizon], "GT", False),
                    (predicted_images[sample, horizon], "Prediction", True))):
                depth, signed = front_view_range(frame, prediction)
                depth_plot = axes[row, horizon].imshow(
                    depth, origin="lower", aspect="auto", cmap=front_cmap,
                    vmin=.5, vmax=10,
                    extent=(float(signed.min()), float(signed.max()), -7, 52),
                    interpolation="nearest")
                axes[row, horizon].axvline(0, color="white", linewidth=.7,
                                           alpha=.65)
                axes[row, horizon].set_title(
                    f"t+{horizon+1} · {label}"
                    + (f" · CD={record['horizons'][horizon]['cd_paper_m2']:.3f} m²"
                       if prediction else ""), fontsize=11)
                axes[row, horizon].set_xlabel(
                    "azimuth (deg; 0° = forward)")
                axes[row, horizon].set_ylabel("elevation angle (deg)")
        color_axis = figure.add_axes((.925, .16, .014, .68))
        figure.colorbar(depth_plot, cax=color_axis, label="obstacle range (m)")
        figure.suptitle(
            f"Forward LiDAR depth map · validation sample {record['index']}\n"
            "top: ground truth · bottom: strict autoregressive prediction",
            y=.985, fontsize=14)
        figure.subplots_adjust(left=.07, right=.9, bottom=.1, top=.86,
                               hspace=.42, wspace=.18)
        front_path = args.output / (
            f"ar3_depth_map_sample_{record['index']:06d}.png")
        figure.savefig(front_path, bbox_inches="tight")
        plt.close(figure)
        front_paths.append(str(front_path))

    means = {}
    for horizon in range(3):
        rows = [record["horizons"][horizon] for record in records]
        means[f"t+{horizon+1}"] = {
            "cd_paper_m2": float(np.mean([row["cd_paper_m2"] for row in rows])),
            "valid_range_mae_m": float(np.mean([
                row["valid_range_mae_m"] for row in rows])),
        }
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "protocol": (
            "strict autoregressive: initial GT history only; t+1/t+2 predictions "
            "are fed back for later horizons; future GT is metrics/visualization only"),
        "count": args.count, "flow_steps": args.flow_steps,
        "flow_ensemble_size": args.flow_ensemble_size,
        "seed": args.seed, "noise_seed": args.noise_seed,
        "means": means, "samples": records,
        "pointcloud_figure": str(point_path),
        "range_image_figure": str(range_path),
        "front_view_figures": front_paths,
    }
    report_path = args.output / "ar3_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--video-index-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--noise-seed", type=int, default=271828)
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument("--flow-ensemble-size", type=int, default=1)
    main(parser.parse_args())
