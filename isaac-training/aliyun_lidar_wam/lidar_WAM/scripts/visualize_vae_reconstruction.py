"""Visualize a same-frame reconstruction from the frozen circular LiDAR VAE.

Run from the lidar_WAM project root with its PPU-compatible Python environment:
    python -m scripts.visualize_vae_reconstruction --split val --data-root /path/to/lagen_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_geometry import frame_points, load_rays


def plot_comparison(output: Path, gt: np.ndarray, recon: np.ndarray,
                    seed: int, index: int, threshold: float,
                    rays: np.ndarray) -> dict:
    gt_hit = gt[1, :, :18] > 0
    pred_hit = recon[1, :, :18] > threshold
    gt_range = np.clip((gt[0, :, :18] + 1) * 5, 0, 10)
    pred_range = np.clip((recon[0, :, :18] + 1) * 5, 0, 10)
    error = np.abs(pred_range - gt_range)
    tp = int(np.count_nonzero(gt_hit & pred_hit))
    fp = int(np.count_nonzero(~gt_hit & pred_hit))
    fn = int(np.count_nonzero(gt_hit & ~pred_hit))
    target_cloud = frame_points(gt, rays)
    reconstructed_cloud = frame_points(recon, rays, threshold)
    if len(target_cloud) and len(reconstructed_cloud):
        squared_cd = float(
            np.square(cKDTree(target_cloud).query(reconstructed_cloud)[0]).mean()
            + np.square(cKDTree(reconstructed_cloud).query(target_cloud)[0]).mean()
        )
    else:
        squared_cd = 0.0 if not len(target_cloud) and not len(reconstructed_cloud) else 200.0
    metrics = {
        "terrain_seed": seed,
        "hdf5_index": index,
        "mask_logit_threshold": threshold,
        "gt_hit_count": int(gt_hit.sum()),
        "reconstruction_hit_count": int(pred_hit.sum()),
        "valid_range_mae_m": float(error[gt_hit].mean()) if gt_hit.any() else None,
        "mask_precision": tp / max(tp + fp, 1),
        "mask_recall": tp / max(tp + fn, 1),
        "mask_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "cd_paper_m2": squared_cd,
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 6.3), constrained_layout=True)
    values = (np.ma.masked_where(~gt_hit.T, gt_range.T),
              np.ma.masked_where(~pred_hit.T, pred_range.T),
              np.ma.masked_where(~gt_hit.T, error.T))
    titles = ("Original range (m)", "VAE reconstruction (m)",
              "Absolute range error on GT hits (m)")
    for col, (value, title) in enumerate(zip(values, titles)):
        image = axes[0, col].imshow(
            value, origin="lower", aspect="auto", interpolation="nearest",
            vmin=0, vmax=10 if col < 2 else 1,
            cmap="viridis" if col < 2 else "magma")
        axes[0, col].set_title(title)
        fig.colorbar(image, ax=axes[0, col], shrink=0.85, label="m")
    for col, (mask, title) in enumerate(zip(
            (gt_hit, pred_hit, gt_hit != pred_hit),
            ("Original hit mask", "VAE reconstructed hit mask", "Hit disagreement"))):
        axes[1, col].imshow(mask.T, origin="lower", aspect="auto",
                            interpolation="nearest", cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(title)
    for row in axes:
        for ax in row:
            ax.set_xlabel("Horizontal ray index (0–107; full 360°)")
            ax.set_ylabel("Vertical ray index (0–17)")
    fig.suptitle(
        f"Circular VAE same-frame reconstruction | seed {seed}, HDF5 row {index} | "
        f"hit MAE {metrics['valid_range_mae_m']:.3f} m, "
        f"mask F1 {metrics['mask_f1']:.4f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--raw-root", type=Path,
                        help="Raw dataset root containing val/seed_0016/calibration/lidar.json")
    parser.add_argument("--output", type=Path,
                        default=stage1.OUT / "vae_same_frame_comparison")
    parser.add_argument("--mask-threshold", type=float, default=1.5)
    args = parser.parse_args()

    stage1.DATA = args.data_root
    raw_root = args.raw_root or args.data_root.parent
    data = stage1.Frames(args.split, limit=512)
    vae = stage1.load_circular_vae()
    summary = []
    for seed in sorted(set(data.seeds.tolist())):
        rays = load_rays(raw_root, args.split, int(seed))[0]
        candidates = np.flatnonzero(data.seeds == seed)
        counts = (data.image[candidates, 1, :, :18] > 0).sum(axis=(1, 2))
        chosen = int(candidates[np.argsort(counts)[len(candidates) // 2]])
        gt = data.image[chosen]
        with torch.inference_mode():
            tensor = torch.from_numpy(gt[None]).to(stage1.DEVICE)
            recon = vae.decode(vae.encode(tensor).latent_dist.mode()).sample
        output = args.output / f"{args.split}_seed{seed}_vae_reconstruction.png"
        metrics = plot_comparison(output, gt, recon[0].cpu().numpy(),
                                  int(seed), int(data.indices[chosen]),
                                  args.mask_threshold, rays)
        metrics["image"] = str(output)
        summary.append(metrics)
    stage1.save_json(args.output / "metrics.json", {
        "split": args.split, "vae": stage1.circular_vae_identity(),
        "samples": summary,
    })
    for sample in summary:
        print(sample, flush=True)


if __name__ == "__main__":
    main()
