"""Ablate geometry-preserving post-processing for decoded Future-DiT LiDAR.

The model prediction is unchanged: eight independent rectified-flow endpoints
are averaged in VAE latent space and decoded once.  Every method below sees
only that decoded prediction; ground truth is used exclusively for metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evaluate_lidar_video_dit_samplers import load_model, predict, summarize
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_video_dit import HistoryLatentDataset
from lidar_wam.runner.world_direct_horizon import frame_metrics


MASK_THRESHOLD = 1.5


def _range_and_hit(frame):
    hit = frame[1, :, :18] > MASK_THRESHOLD
    distance = np.clip((frame[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
    return distance, hit


def remove_small_components(frame, min_size: int, range_delta: float | None):
    """Remove tiny 4-connected components, optionally split at depth edges."""
    output = frame.copy()
    distance, hit = _range_and_hit(output)
    visited = np.zeros_like(hit, dtype=bool)
    keep = np.zeros_like(hit, dtype=bool)
    height, width = hit.shape
    for azimuth, elevation in np.argwhere(hit):
        azimuth, elevation = int(azimuth), int(elevation)
        if visited[azimuth, elevation]:
            continue
        queue = deque(((azimuth, elevation),))
        visited[azimuth, elevation] = True
        component = []
        while queue:
            row, column = queue.popleft()
            component.append((row, column))
            for following_row, following_column in (
                    ((row - 1) % height, column),
                    ((row + 1) % height, column),
                    (row, column - 1), (row, column + 1)):
                if not 0 <= following_column < width:
                    continue
                if visited[following_row, following_column] or not hit[
                        following_row, following_column]:
                    continue
                if (range_delta is not None and abs(
                        float(distance[row, column])
                        - float(distance[following_row, following_column]))
                        > float(range_delta)):
                    continue
                visited[following_row, following_column] = True
                queue.append((following_row, following_column))
        if len(component) >= int(min_size):
            rows, columns = zip(*component)
            keep[rows, columns] = True
    output[1, :, :18][hit & ~keep] = -10.0
    return output


def edge_preserving_median(frame, range_delta: float):
    """Median-filter ranges only among adjacent hits on the same surface."""
    output = frame.copy()
    distance, hit = _range_and_hit(output)
    filtered = distance.copy()
    height, width = hit.shape
    for row, column in np.argwhere(hit):
        row, column = int(row), int(column)
        values = []
        center = float(distance[row, column])
        for row_offset in (-1, 0, 1):
            following_row = (row + row_offset) % height
            for column_offset in (-1, 0, 1):
                following_column = column + column_offset
                if not 0 <= following_column < width:
                    continue
                if not hit[following_row, following_column]:
                    continue
                value = float(distance[following_row, following_column])
                if abs(value - center) <= float(range_delta):
                    values.append(value)
        if len(values) >= 3:
            filtered[row, column] = float(np.median(values))
    output[0, :, :18] = filtered / 5.0 - 1.0
    return output


def fill_vertical_one_ray_gaps(frame, range_delta: float):
    """Fill one missing elevation ray between two consistent surface hits."""
    output = frame.copy()
    distance, hit = _range_and_hit(output)
    for row in range(hit.shape[0]):
        for column in range(1, hit.shape[1] - 1):
            if hit[row, column] or not (
                    hit[row, column - 1] and hit[row, column + 1]):
                continue
            before = float(distance[row, column - 1])
            after = float(distance[row, column + 1])
            if abs(before - after) > float(range_delta):
                continue
            filled = 0.5 * (before + after)
            output[0, row, column] = filled / 5.0 - 1.0
            output[1, row, column] = max(
                float(output[1, row, column - 1]),
                float(output[1, row, column + 1]),
                MASK_THRESHOLD + 1e-3)
    return output


def apply_method(frame, method):
    name, arguments = method
    if name == "none":
        return frame
    if name == "component":
        return remove_small_components(frame, **arguments)
    if name == "median":
        return edge_preserving_median(frame, **arguments)
    if name == "component_median":
        value = remove_small_components(
            frame, arguments["min_size"], arguments["component_delta"])
        return edge_preserving_median(value, arguments["median_delta"])
    if name == "vertical_fill":
        return fill_vertical_one_ray_gaps(frame, **arguments)
    if name == "vertical_fill_median":
        value = fill_vertical_one_ray_gaps(
            frame, arguments["fill_delta"])
        return edge_preserving_median(value, arguments["median_delta"])
    raise ValueError(name)


def method_name(method):
    name, arguments = method
    if not arguments:
        return name
    suffix = "_".join(
        f"{key}{'none' if value is None else format(value, 'g')}"
        for key, value in arguments.items())
    return f"{name}_{suffix}"


@torch.no_grad()
def main(args):
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
    scale = float(dataset.vae_metadata["scaling_factor"])
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)

    methods = [
        ("none", {}),
        ("component", {"min_size": 2, "range_delta": None}),
        ("component", {"min_size": 3, "range_delta": None}),
        ("component", {"min_size": 4, "range_delta": None}),
        ("component", {"min_size": 2, "range_delta": 1.0}),
        ("component", {"min_size": 3, "range_delta": 1.0}),
        ("component", {"min_size": 4, "range_delta": 1.0}),
        ("median", {"range_delta": 0.5}),
        ("median", {"range_delta": 1.0}),
        ("median", {"range_delta": 2.0}),
        ("component_median", {
            "min_size": 3, "component_delta": 1.0,
            "median_delta": 1.0}),
        ("vertical_fill", {"range_delta": 1.0}),
        ("vertical_fill_median", {
            "fill_delta": 1.0, "median_delta": 1.0}),
    ]
    rows = {method_name(method): [] for method in methods}
    started = time.time()
    for history, _, target in loader:
        history = history.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = predict(
                model, history, "euler", args.flow_steps, 8, generator)
            decoded = vae.decode(latent.float() / scale).sample
        decoded = decoded.float().cpu().numpy()
        target = target.numpy()
        for prediction, truth in zip(decoded, target):
            for method in methods:
                name = method_name(method)
                processed = apply_method(prediction, method)
                rows[name].append(frame_metrics(
                    processed, truth, MASK_THRESHOLD))

    results = {name: summarize(value) for name, value in rows.items()}
    baseline = results["none"]["nonempty_gt_cd_paper_m2"]
    for result in results.values():
        result["cd_change_vs_raw"] = (
            result["nonempty_gt_cd_paper_m2"] - baseline)
    best = min(results, key=lambda key: results[key][
        "nonempty_gt_cd_paper_m2"])
    report = {
        "format": "navrl-future-dit-decoded-postprocess-ablation-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "prediction": "euler20_latent_mean8",
        "eval_samples": count,
        "nonempty_gt_samples": results[best]["nonempty_gt_samples"],
        "selection_seed": int(args.seed),
        "noise_seed": int(args.noise_seed),
        "best_method": best,
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)


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
    parser.add_argument("--noise-seed", type=int, default=12345)
    parser.add_argument("--flow-steps", type=int, default=20)
    main(parser.parse_args())
