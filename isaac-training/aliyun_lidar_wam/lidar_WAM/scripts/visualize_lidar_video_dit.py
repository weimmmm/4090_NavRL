"""Visualize one history-only Video-DiT validation window.

The figure uses the same frozen circular VAE and LiDAR geometry as validation:
the top row is a reprojected XY point-cloud view, while the bottom row shows
the decoded range image.  No Isaac Sim state or privileged information is used.
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
    HistoryLatentDataset, sample_next,
)


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    architecture = payload["architecture"]
    model = LiDARVideoDiT(
        width=int(architecture["width"]),
        depth=int(architecture["depth"]),
        heads=int(architecture["heads"]),
        mlp_ratio=float(architecture["mlp_ratio"]),
    ).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def point_cloud(ax, frame, title, threshold):
    points = stage1.to_points(frame, threshold)
    if len(points):
        image = ax.scatter(points[:, 0], points[:, 1], c=points[:, 2],
                           s=10, cmap="viridis", vmin=-2.0, vmax=8.0)
    else:
        image = ax.scatter([], [])
    ax.set_title(f"{title}\n{len(points)} points")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)
    return points, image


def range_image(ax, frame, title):
    # Show the return mask and normalized range with the same azimuth/elevation
    # layout as the VAE input.  This is supplementary to the point-cloud row.
    value = np.clip((frame[0] + 1.0) * 5.0, 0.0, 10.0)
    mask = frame[1] > 1.5
    display = np.where(mask, value, np.nan)
    image = ax.imshow(display[:, :18].T, origin="lower", aspect="auto",
                      cmap="magma", vmin=0.0, vmax=10.0)
    ax.set_title(title)
    ax.set_xlabel("azimuth bin")
    ax.set_ylabel("elevation bin")
    return image


@torch.no_grad()
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=True)
    if not len(dataset):
        raise RuntimeError("validation video index is empty")
    index = int(args.index)
    history, target, target_image = dataset[index]
    shard = bisect.bisect_right(dataset.ends, index)
    start = 0 if shard == 0 else int(dataset.ends[shard - 1])
    rows, _ = dataset._open(shard)
    chain = rows[index - start]
    current_image = dataset._target_image(shard, int(chain[2]))

    model, payload = load_model(args.checkpoint, device)
    vae = stage1.load_circular_vae().to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    scale = float(dataset.vae_metadata["scaling_factor"])
    history_device = history[None].to(device)
    predicted = sample_next(model, history_device, args.flow_steps)[0]
    current_latent = history[-1:].to(device)
    current_reconstruction = vae.decode(current_latent / scale).sample[0].float().cpu().numpy()
    predicted_image = vae.decode(predicted[None] / scale).sample[0].float().cpu().numpy()
    target_image = target_image.numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(18, 9), dpi=150)
    grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 0.72),
                               hspace=0.32, wspace=0.22)
    point_rows = [
        (current_image, "GT current", 0.0),
        (current_reconstruction, "VAE current reconstruction", 1.5),
        (target_image, "GT next", 0.0),
        (predicted_image, "Video-DiT predicted next", 1.5),
    ]
    point_counts = {}
    for column, (frame, title, threshold) in enumerate(point_rows):
        ax = figure.add_subplot(grid[0, column])
        points, _ = point_cloud(ax, frame, title, threshold)
        point_counts[title] = int(len(points))
    for column, (frame, title, _) in enumerate(point_rows):
        ax = figure.add_subplot(grid[1, column])
        range_image(ax, frame, title + " range")
    figure.suptitle(
        f"History-only LiDAR Video-DiT | val window {index} | "
        f"checkpoint step {payload.get('step', '?')}", fontsize=14)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "validation_index": index,
        "flow_steps": int(args.flow_steps),
        "point_counts": point_counts,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--video-index-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=20)
    main(parser.parse_args())
