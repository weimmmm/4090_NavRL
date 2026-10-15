"""Compare inference solvers for a trained one-step LiDAR Video-DiT.

All methods use the same validation windows and deterministic random seed.
Only non-empty ground-truth frames determine the primary score.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_video_dit import (
    HistoryLatentDataset,
    _to_next_latent,
    configure_prediction_from_payload,
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


@torch.no_grad()
def integrate(model, history, steps: int, method: str,
              generator: torch.Generator):
    value = torch.randn(
        len(history), 4, 27, 5, device=history.device,
        dtype=history.dtype, generator=generator)
    schedule = torch.linspace(
        1, 0, int(steps) + 1, device=history.device, dtype=history.dtype)
    for current, following in zip(schedule[:-1], schedule[1:]):
        timestep = torch.full(
            (len(history),), current, device=history.device,
            dtype=history.dtype)
        velocity = model(history, value, timestep)
        delta = following - current
        if method == "euler":
            value = value + delta * velocity
        elif method == "heun":
            provisional = value + delta * velocity
            next_timestep = torch.full(
                (len(history),), following, device=history.device,
                dtype=history.dtype)
            next_velocity = model(history, provisional, next_timestep)
            value = value + delta * 0.5 * (velocity + next_velocity)
        else:
            raise ValueError(f"unknown solver {method!r}")
    return _to_next_latent(model, history, value)


@torch.no_grad()
def predict(model, history, method: str, steps: int, samples: int,
            generator: torch.Generator):
    values = [integrate(model, history, steps, method, generator)
              for _ in range(int(samples))]
    return torch.stack(values).mean(0)


@torch.no_grad()
def predict_decoded_median(model, vae, history, method: str, steps: int,
                           samples: int, scale: float,
                           generator: torch.Generator):
    """Decode independent flow samples, then take a per-ray median.

    This is the metric-range-image counterpart of inference-time ensembling
    used by diffusion depth estimators such as Marigold.  Our LiDAR range is
    already metrically calibrated, so no per-sample scale/shift alignment is
    needed.  Both the normalized range channel and hit-mask logit channel are
    aggregated in decoded output space.
    """
    decoded = []
    for _ in range(int(samples)):
        latent = integrate(model, history, steps, method, generator)
        decoded.append(
            vae.decode(latent.float() / float(scale)).sample.float())
    ordered = torch.stack(decoded, dim=0).sort(dim=0).values
    middle = len(decoded) // 2
    if len(decoded) % 2:
        return ordered[middle]
    # torch.median returns the lower middle value for an even sample count;
    # use the conventional statistical median used by ensemble estimators.
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def summarize(rows):
    nonempty = [row for row in rows if not row["gt_empty"]]
    tp = sum(int(row["tp"]) for row in nonempty)
    fp = sum(int(row["fp"]) for row in nonempty)
    fn = sum(int(row["fn"]) for row in nonempty)
    valid_count = sum(int(row["valid_range_count"]) for row in nonempty)
    valid_error = sum(float(row["valid_range_abs_error_sum_m"])
                      for row in nonempty)
    return {
        "samples": len(rows),
        "nonempty_gt_samples": len(nonempty),
        "nonempty_gt_cd_paper_m2": float(np.mean([
            row["cd_paper_m2"] for row in nonempty])),
        "nonempty_gt_median_cd_paper_m2": float(np.median([
            row["cd_paper_m2"] for row in nonempty])),
        "valid_range_mae_m": valid_error / max(valid_count, 1),
        "mask_precision_nonempty": tp / max(tp + fp, 1),
        "mask_recall_nonempty": tp / max(tp + fn, 1),
        "mask_f1_nonempty": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_empty_frames": int(sum(
            bool(row["false_empty_frame"]) for row in nonempty)),
    }


@torch.no_grad()
def evaluate(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Sampler evaluation requires CUDA")
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
    specifications = [
        ("euler20", "euler", 20, 1, "latent_mean"),
        ("euler20_mean2", "euler", 20, 2, "latent_mean"),
        ("euler40", "euler", 40, 1, "latent_mean"),
        ("heun20", "heun", 20, 1, "latent_mean"),
        ("euler20_mean4", "euler", 20, 4, "latent_mean"),
        ("euler20_mean8", "euler", 20, 8, "latent_mean"),
        ("euler20_mean10", "euler", 20, 10, "latent_mean"),
        ("euler20_decoded_median10", "euler", 20, 10,
         "decoded_median"),
    ]
    if args.methods:
        requested = set(args.methods)
        known = {name for name, _, _, _, _ in specifications}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
        specifications = [row for row in specifications
                          if row[0] in requested]
    results = {}
    for name, method, steps, samples, aggregation in specifications:
        started = time.time()
        generator = torch.Generator(device=device).manual_seed(
            int(args.sampling_seed))
        rows = []
        for history, _, target_image in loader:
            history = history.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if aggregation == "decoded_median":
                    decoded = predict_decoded_median(
                        model, vae, history, method, steps, samples, scale,
                        generator)
                else:
                    predicted = predict(
                        model, history, method, steps, samples, generator)
                    decoded = vae.decode(
                        predicted.float() / scale).sample
            decoded = decoded.float().cpu().numpy()
            for prediction, truth in zip(decoded, target_image.numpy()):
                rows.append(frame_metrics(
                    prediction, truth, float(args.mask_threshold)))
        result = summarize(rows)
        result.update({
            "solver": method,
            "steps": int(steps),
            "samples": int(samples),
            "aggregation": aggregation,
            "elapsed_seconds": time.time() - started,
        })
        results[name] = result
        print(json.dumps({"method": name, **result}), flush=True)

    # Decode the current latent once as the no-learning lower-cost reference.
    copy_rows = []
    for history, _, target_image in loader:
        current = history[:, -1].to(device, non_blocking=True)
        decoded = vae.decode(current.float() / scale).sample.cpu().numpy()
        for prediction, truth in zip(decoded, target_image.numpy()):
            copy_rows.append(frame_metrics(
                prediction, truth, float(args.mask_threshold)))
    results["copy_current_vae"] = summarize(copy_rows)
    best_name = min(
        (name for name, _, _, _, _ in specifications),
        key=lambda name: results[name]["nonempty_gt_cd_paper_m2"])
    report = {
        "format": "navrl-lidar-video-dit-sampler-ablation-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "eval_samples": count,
        "nonempty_gt_samples": results[best_name]["nonempty_gt_samples"],
        "seed": int(args.seed),
        "sampling_seed": int(args.sampling_seed),
        "mask_threshold": float(args.mask_threshold),
        "best_method": best_name,
        "best_nonempty_gt_cd_paper_m2": results[best_name][
            "nonempty_gt_cd_paper_m2"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "best_method": best_name,
        "best_nonempty_gt_cd_paper_m2": report[
            "best_nonempty_gt_cd_paper_m2"],
        "copy_current_nonempty_gt_cd_paper_m2": results[
            "copy_current_vae"]["nonempty_gt_cd_paper_m2"],
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=12345)
    parser.add_argument("--mask-threshold", type=float, default=1.5)
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Optional subset, for example euler20_mean2 euler20_mean8.")
    evaluate(parser.parse_args())
