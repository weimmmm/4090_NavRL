"""Batch visualization of random validation windows for the history-only DiT."""

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
from lidar_wam.runner.lidar_video_dit import HistoryLatentDataset, sample_next
from lidar_wam.runner.world_direct_horizon import frame_metrics


def raw_current(dataset, index):
    shard = bisect.bisect_right(dataset.ends, int(index))
    start = 0 if shard == 0 else int(dataset.ends[shard - 1])
    rows, _ = dataset._open(shard)
    chain = rows[int(index) - start]
    return dataset._target_image(shard, int(chain[2]))


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = payload["architecture"]
    model = LiDARVideoDiT(
        width=int(cfg["width"]), depth=int(cfg["depth"]),
        heads=int(cfg["heads"]), mlp_ratio=float(cfg["mlp_ratio"]),
    ).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def points(frame, threshold):
    return stage1.to_points(frame, threshold)


def overlay(ax, truth, prediction, title, cd):
    gt = points(truth, 0.0)
    pred = points(prediction, 1.5)
    if len(gt):
        ax.scatter(gt[:, 0], gt[:, 1], s=10, c="#2878b5", label="GT")
    if len(pred):
        ax.scatter(pred[:, 0], pred[:, 1], s=12, c="#d84a3a", marker="x",
                   label="pred")
    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\nGT {len(gt)} / pred {len(pred)} | CD {cd:.3f} m²",
                 fontsize=8)
    ax.grid(alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([])


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=True)
    rng = np.random.default_rng(args.seed)
    count = min(int(args.count), len(dataset))
    indices = np.sort(rng.choice(len(dataset), count, replace=False))
    histories, targets, target_images, current_images = [], [], [], []
    for index in indices.tolist():
        history, target, target_image = dataset[int(index)]
        histories.append(history)
        targets.append(target)
        target_images.append(target_image.numpy())
        current_images.append(raw_current(dataset, int(index)))

    model, payload = load_model(args.checkpoint, device)
    vae = stage1.load_circular_vae().to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    scale = float(dataset.vae_metadata["scaling_factor"])
    history = torch.stack(histories).to(device)
    target = torch.stack(targets).to(device)
    predicted = sample_next(model, history, args.flow_steps)
    current_reconstruction = vae.decode(history[:, -1] / scale).sample.float().cpu().numpy()
    predicted_images = vae.decode(predicted / scale).sample.float().cpu().numpy()
    target_images = np.asarray(target_images)
    current_images = np.asarray(current_images)

    rows = []
    for index, truth, prediction in zip(indices.tolist(), target_images,
                                        predicted_images):
        metric = frame_metrics(prediction, truth)
        rows.append({"index": int(index), "cd_paper_m2": float(metric["cd_paper_m2"]),
                     "gt_points": int(len(points(truth, 0.0))),
                     "pred_points": int(len(points(prediction, 1.5))),
                     "gt_empty": bool(metric["gt_empty"])})
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 4, figsize=(12, 12), dpi=150)
    for ax, row, truth, prediction in zip(axes.flat, rows, target_images,
                                          predicted_images):
        overlay(ax, truth, prediction, f"val index {row['index']}",
                row["cd_paper_m2"])
    handles = [plt.Line2D([], [], color="#2878b5", marker="o", linestyle="",
                          label="GT next"),
               plt.Line2D([], [], color="#d84a3a", marker="x", linestyle="",
                          label="Video-DiT prediction")]
    figure.legend(handles=handles, loc="upper center", ncol=2)
    figure.suptitle(
        f"Random validation next-frame point clouds | step {payload.get('step', '?')} | "
        f"N={count}", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)

    vae_output = output.with_name(output.stem + "_vae.png")
    figure, axes = plt.subplots(4, 4, figsize=(12, 12), dpi=150)
    vae_rows = []
    for ax, index, truth, reconstruction in zip(
            axes.flat, indices.tolist(), current_images, current_reconstruction):
        metric = frame_metrics(reconstruction, truth)
        vae_rows.append(float(metric["cd_paper_m2"]))
        overlay(ax, truth, reconstruction, f"val index {index}", vae_rows[-1])
    figure.suptitle(
        f"Random validation current-frame VAE reconstruction | N={count}", y=0.98)
    figure.tight_layout()
    figure.savefig(vae_output, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "checkpoint": str(args.checkpoint), "checkpoint_step": int(payload.get("step", -1)),
        "count": int(count), "indices": indices.tolist(), "flow_steps": int(args.flow_steps),
        "prediction_cd_mean_m2": float(np.mean([row["cd_paper_m2"] for row in rows])),
        "prediction_cd_median_m2": float(np.median([row["cd_paper_m2"] for row in rows])),
        "prediction_nonempty_gt_cd_mean_m2": float(np.mean([
            row["cd_paper_m2"] for row in rows if not row["gt_empty"]])),
        "vae_current_cd_mean_m2": float(np.mean(vae_rows)),
        "rows": rows, "output": str(output), "vae_output": str(vae_output),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--video-index-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--flow-steps", type=int, default=20)
    main(parser.parse_args())
