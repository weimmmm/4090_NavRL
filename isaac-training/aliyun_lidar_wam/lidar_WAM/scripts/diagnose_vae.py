"""Audit NavRL VAE reconstruction quality beyond global MAE and mask F1."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from lidar_wam.runner import stage1


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()

    stage1.seed_everything(42)
    dataset = stage1.Frames(args.split, limit=args.limit)
    model = stage1.load_circular_vae()
    checkpoint = {"step": stage1.circular_vae_identity()["vae_step"]}
    rows = []
    bin_errors = defaultdict(list)
    latent_mean, posterior_sigma, kl = [], [], []
    sample_mode_mae = []
    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            end = min(start + args.batch_size, len(dataset))
            source = dataset.image[start:end]
            x = torch.from_numpy(source).to(stage1.DEVICE)
            posterior = model.encode(x).latent_dist
            mode = posterior.mode()
            reconstruction = model.decode(mode).sample.cpu().numpy()
            latent_mean.append(mode.mean().item())
            posterior_sigma.append(posterior.std.mean().item())
            kl.append(posterior.kl().mean().item())
            if start < 512:
                sampled = model.decode(posterior.sample()).sample.cpu().numpy()
            for j, (target, pred) in enumerate(zip(source, reconstruction)):
                valid = target[1, :, :18] > 0
                predicted_valid = pred[1, :, :18] > 0
                distance_m = np.abs(pred[0, :, :18] - target[0, :, :18]) * 5
                radial = float(distance_m[valid].mean()) if valid.any() else 0.0
                tp = int(np.logical_and(valid, predicted_valid).sum())
                fp = int(np.logical_and(~valid, predicted_valid).sum())
                fn = int(np.logical_and(valid, ~predicted_valid).sum())
                f1 = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
                truth_cloud = stage1.to_points(target)
                row = {"source_index": int(dataset.indices[start + j]),
                       "seed": int(dataset.seeds[start + j]),
                       "valid_points": int(valid.sum()),
                       "reconstructed_points": int(predicted_valid.sum()),
                       "radial_mae_m": radial, "mask_f1": f1,
                       "chamfer_t0_m": stage1.chamfer(stage1.to_points(pred), truth_cloud),
                       "chamfer_t2_m": stage1.chamfer(stage1.to_points(pred, 2), truth_cloud)}
                rows.append(row)
                if start < 512:
                    sample_mode_mae.append(float((np.abs(sampled[j, 0, :, :18] -
                                                         pred[0, :, :18]) * 5)[valid].mean()))
                true_distance = (target[0, :, :18] + 1) * 5
                for low, high in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10.01)):
                    in_bin = valid & (true_distance >= low) & (true_distance < high)
                    if in_bin.any():
                        bin_errors[f"{low}-{high:g}m"].extend(distance_m[in_bin].tolist())
            if (end % 1024 < args.batch_size) or end == len(dataset):
                print(f"audited {end}/{len(dataset)}", flush=True)

    report = {"split": args.split, "samples": len(rows), "vae_step": checkpoint["step"],
              "mask_f1_logit_threshold": 0.0,
              "empty_truth_frames": sum(r["valid_points"] == 0 for r in rows),
              "empty_reconstruction_frames": sum(r["reconstructed_points"] == 0 for r in rows),
              "both_empty_frames": sum(r["valid_points"] == 0 and
                                       r["reconstructed_points"] == 0 for r in rows),
              "per_frame_radial_mae_m": describe([r["radial_mae_m"] for r in rows]),
              "per_frame_mask_f1": describe([r["mask_f1"] for r in rows]),
              "per_frame_chamfer_t0_m": describe([r["chamfer_t0_m"] for r in rows]),
              "per_frame_chamfer_t2_m": describe([r["chamfer_t2_m"] for r in rows]),
              "valid_points_per_frame": describe([r["valid_points"] for r in rows]),
              "radial_mae_by_true_range_m": {k: {"points": len(v), "mae_m": float(np.mean(v))}
                                              for k, v in bin_errors.items()},
              "latent_batch_mean": describe(latent_mean),
              "posterior_sigma_batch_mean": describe(posterior_sigma),
              "posterior_kl_batch_mean": describe(kl),
              "sample_vs_mode_radial_mae_m_first512": describe(sample_mode_mae),
              "per_seed": {},
              "worst_16_chamfer": sorted(rows, key=lambda r: r["chamfer_t2_m"], reverse=True)[:16]}
    for seed in sorted(set(dataset.seeds)):
        subset = [r for r in rows if r["seed"] == seed]
        report["per_seed"][str(seed)] = {
            "samples": len(subset),
            "chamfer_t2_m": describe([r["chamfer_t2_m"] for r in subset]),
            "radial_mae_m": describe([r["radial_mae_m"] for r in subset]),
            "mask_f1": describe([r["mask_f1"] for r in subset])}
    suffix = f"_n{len(dataset)}" if args.limit is not None else ""
    path = stage1.OUT / stage1.CIRCULAR_VAE_DIR / f"diagnostics_{args.split}{suffix}.json"
    stage1.save_json(path, report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("worst_16_chamfer", "per_seed")}), flush=True)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
