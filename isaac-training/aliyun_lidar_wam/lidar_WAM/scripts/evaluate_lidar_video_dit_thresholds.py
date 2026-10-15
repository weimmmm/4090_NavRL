"""Calibrate the decoded hit-mask threshold of a LiDAR Video-DiT.

The Future Model predicts a frozen-VAE latent.  The VAE decoder's second
channel is an unbounded hit-mask activation, so the threshold used to turn it
into a point cloud has a large effect on Chamfer distance, especially for
empty ground-truth scans.  This script samples each next latent once and then
evaluates the exact same decoded predictions at every requested threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_video_dit import (
    HistoryLatentDataset,
    configure_prediction_from_payload,
    sample_next,
)
from lidar_wam.runner.world_direct_horizon import frame_metrics


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = dict(payload.get("architecture", {}))
    model = LiDARVideoDiT(
        width=int(architecture.get("width", 512)),
        depth=int(architecture.get("depth", 8)),
        heads=int(architecture.get("heads", 8)),
        mlp_ratio=float(architecture.get("mlp_ratio", 4.0)),
    ).to(device).eval()
    model.load_state_dict(payload.get("model", payload), strict=True)
    configure_prediction_from_payload(model, payload)
    return model, payload


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    nonempty = [row for row in rows if not row["gt_empty"]]
    empty = [row for row in rows if row["gt_empty"]]
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    valid_count = sum(int(row["valid_range_count"]) for row in rows)
    valid_error = sum(float(row["valid_range_abs_error_sum_m"])
                      for row in rows)
    false_hit = sum(bool(row["false_hit_frame"]) for row in rows)
    false_empty = sum(bool(row["false_empty_frame"]) for row in rows)
    return {
        "samples": count,
        "nonempty_gt_samples": len(nonempty),
        "empty_gt_samples": len(empty),
        "cd_paper_m2": float(np.mean([
            row["cd_paper_m2"] for row in rows])),
        "median_cd_paper_m2": float(np.median([
            row["cd_paper_m2"] for row in rows])),
        "nonempty_gt_cd_paper_m2": float(np.mean([
            row["cd_paper_m2"] for row in nonempty])) if nonempty else 0.0,
        "empty_gt_cd_paper_m2": float(np.mean([
            row["cd_paper_m2"] for row in empty])) if empty else 0.0,
        "valid_range_mae_m": valid_error / max(valid_count, 1),
        "mask_precision": tp / max(tp + fp, 1),
        "mask_recall": tp / max(tp + fn, 1),
        "mask_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_hit_frames": int(false_hit),
        "false_hit_rate_on_empty": false_hit / max(len(empty), 1),
        "false_empty_frames": int(false_empty),
        "false_empty_rate_on_nonempty": false_empty / max(len(nonempty), 1),
        "mean_pred_hits_on_empty": float(np.mean([
            row["pred_hit_count"] for row in empty])) if empty else 0.0,
        "gt_hit_count": int(sum(row["gt_hit_count"] for row in rows)),
        "pred_hit_count": int(sum(row["pred_hit_count"] for row in rows)),
    }


@torch.no_grad()
def evaluate(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Threshold calibration requires CUDA")
    device = torch.device(args.device)
    dataset = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=True)
    count = min(int(args.eval_samples), len(dataset))
    rng = np.random.default_rng(int(args.seed))
    indices = np.sort(rng.choice(len(dataset), count, replace=False))
    loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=args.eval_batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)

    model, payload = load_model(args.checkpoint, device)
    vae = stage1.load_circular_vae().to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    scale = float(dataset.vae_metadata["scaling_factor"])
    generator = torch.Generator(device=device).manual_seed(
        int(args.sampling_seed))
    prediction_images, copy_images, target_images = [], [], []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    for history, _, target_image in loader:
        history = history.to(device, non_blocking=True)
        with autocast:
            predicted = sample_next(
                model, history, int(args.flow_steps), generator,
                num_samples=args.flow_ensemble_size)
        decoded = vae.decode(torch.cat(
            (predicted.float(), history[:, -1].float()), dim=0) / scale
        ).sample.float().cpu().numpy()
        prediction, copied = np.split(decoded, 2, axis=0)
        prediction_images.extend(prediction)
        copy_images.extend(copied)
        target_images.extend(target_image.numpy())

    results = []
    for threshold in args.thresholds:
        predicted_rows = [
            frame_metrics(prediction, target, float(threshold))
            for prediction, target in zip(prediction_images, target_images)
        ]
        copy_rows = [
            frame_metrics(copied, target, float(threshold))
            for copied, target in zip(copy_images, target_images)
        ]
        row = {
            "threshold": float(threshold),
            "prediction": summarize(predicted_rows),
            "copy_current_vae": summarize(copy_rows),
        }
        results.append(row)
        metric = row["prediction"]
        print(json.dumps({
            "threshold": float(threshold),
            "cd_paper_m2": metric["cd_paper_m2"],
            "nonempty_gt_cd_paper_m2": metric["nonempty_gt_cd_paper_m2"],
            "mask_f1": metric["mask_f1"],
            "false_hit_frames": metric["false_hit_frames"],
            "false_empty_frames": metric["false_empty_frames"],
        }), flush=True)

    best_all = min(results, key=lambda row: row["prediction"]["cd_paper_m2"])
    best_nonempty = min(
        results,
        key=lambda row: row["prediction"]["nonempty_gt_cd_paper_m2"])
    best_f1 = max(results, key=lambda row: row["prediction"]["mask_f1"])
    report = {
        "format": "navrl-lidar-video-dit-threshold-sweep-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "prediction_target": payload.get("architecture", {}).get(
            "prediction_target", "absolute_latent"),
        "dataset_root": str(args.dataset_root),
        "eval_samples": count,
        "seed": int(args.seed),
        "sampling_seed": int(args.sampling_seed),
        "flow_steps": int(args.flow_steps),
        "flow_ensemble_size": int(args.flow_ensemble_size),
        "indices": indices.tolist(),
        "best_all_cd": best_all,
        "best_nonempty_cd": best_nonempty,
        "best_mask_f1": best_f1,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "best_all_cd_threshold": best_all["threshold"],
        "best_all_cd_paper_m2": best_all["prediction"]["cd_paper_m2"],
        "best_nonempty_threshold": best_nonempty["threshold"],
        "best_mask_f1_threshold": best_f1["threshold"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--video-index-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument("--flow-ensemble-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=12345)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[
        -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
        2.0, 2.25, 2.5, 3.0, 3.5, 4.0,
    ])
    evaluate(parser.parse_args())
