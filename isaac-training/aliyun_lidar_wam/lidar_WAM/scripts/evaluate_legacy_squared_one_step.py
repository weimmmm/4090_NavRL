"""Re-evaluate fixed one-step legacy methods with squared point-cloud Chamfer.

The old representative reports stored linear Chamfer only. This script uses
the exact same samples, checkpoints, DDIM seeds, and mask threshold, while
computing the paper-style sum of directional squared nearest-neighbor means.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from evaluate_lagen_metrics_10 import lidar_metric
from evaluate_representative import load_selected
from lidar_wam.runner import stage1
from lidar_wam.runner.executed_residual import condition_stats
from lidar_wam.runner.lidar_geometry import (frame_points, load_rays,
                                              predict_transform, warp_frame)


def linear_chamfer(prediction, target, rays, threshold):
    a = frame_points(prediction, rays, threshold)
    b = frame_points(target, rays)
    if not len(a) and not len(b):
        return 0.0
    if not len(a) or not len(b):
        return 20.0
    return float((cKDTree(a).query(b)[0].mean() +
                  cKDTree(b).query(a)[0].mean()) / 2)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path,
                        default=stage1.OUT / "representative_baseline" /
                                "test_legacy_squared_one_step.json")
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    manifest = json.loads((stage1.OUT / "representative_baseline" /
                           "sample_manifest.json").read_text())
    rows, data, positions, arrays = load_selected(manifest, args.split)
    rays = {seed: load_rays(args.raw_root, args.split, seed)
            for seed in sorted({r["seed"] for r in rows})}
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR /
                        "metadata.json").read_text())["scaling_factor"]
    outputs = [{"source_index": row["source_index"], "seed": row["seed"]}
               for row in rows]
    for index, row in enumerate(rows):
        grid, azimuth, elevation = rays[row["seed"]]
        previous = arrays["prev_range_values"][index]
        target = arrays["range_values"][index]
        velocity = warp_frame(previous,
                              predict_transform(arrays["prev_drone_state"][index]),
                              grid, azimuth, elevation)
        for name, frame in (("copy", previous), ("velocity_pose", velocity)):
            metric = lidar_metric(frame, target, grid, 0)
            metric["chamfer_linear_m"] = linear_chamfer(frame, target, grid, 0)
            outputs[index][name] = metric
    vae = stage1.load_circular_vae()
    for name, state_dim, run, strength in (
        ("lagen_unet_8h", 5, "world_circular_causal_8h", 1.0),
        ("lagen_residual", 11, "world_circular_executed_residual_full", 0.05),
    ):
        model = stage1.WorldModel(state_dim=state_dim).to(stage1.DEVICE).float().eval()
        checkpoint = stage1.load_model(stage1.OUT / run / "best.pt", model)
        scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                         prediction_type="epsilon", clip_sample=False)
        for seed in sorted(rays):
            locations = np.array([i for i, row in enumerate(rows) if row["seed"] == seed])
            for start in range(0, len(locations), args.batch_size):
                chosen = locations[start:start + args.batch_size]
                source = positions[chosen]
                previous = data.previous[source].to(stage1.DEVICE)
                if name == "lagen_unet_8h":
                    actions = torch.from_numpy(arrays["normalized_action_sequence"][chosen]).to(stage1.DEVICE)
                    state = torch.from_numpy(stage1.causal_state(
                        arrays["prev_ego_feats"][chosen])).to(stage1.DEVICE)
                    residual_scale = None
                else:
                    actions = data.actions[source].to(stage1.DEVICE)
                    state = data.state[source].to(stage1.DEVICE)
                    residual_scale = condition_stats()["residual_scale"]
                latent = stage1.generate(model, scheduler, previous, actions, state,
                                         seed=42 + seed * 10000 + start,
                                         init_strength=strength, num_steps=20,
                                         residual_scale=residual_scale)
                predicted = vae.decode(latent / scale).sample.cpu().numpy()
                for local, index in enumerate(chosen):
                    grid = rays[seed][0]
                    target = arrays["range_values"][index]
                    metric = lidar_metric(predicted[local], target, grid, threshold)
                    metric["chamfer_linear_m"] = linear_chamfer(
                        predicted[local], target, grid, threshold)
                    outputs[index][name] = metric
            print(json.dumps({"method": name, "seed": seed,
                              "checkpoint_step": checkpoint["step"]}), flush=True)
        del model
        torch.cuda.empty_cache()
    methods = ("copy", "velocity_pose", "lagen_unet_8h", "lagen_residual")
    paired = [row for row in outputs if all(not row[m]["empty_cloud"] for m in methods)]
    summary = {}
    for group_name, group in (("all", outputs), ("paired_nonempty", paired)):
        summary[group_name] = {"samples": len(group), "methods": {name: {
            "cd_paper_m2": float(np.mean([row[name]["cd_paper_m2"] for row in group])),
            "cd_released_code_m2": float(np.mean([
                row[name]["cd_released_code_m2"] for row in group])),
            "chamfer_linear_m": float(np.mean([
                row[name]["chamfer_linear_m"] for row in group])),
            "empty_cloud_cases": sum(row[name]["empty_cloud"] for row in group),
        } for name in methods}}
    report = {"split": args.split, "samples": len(outputs),
              "metric": "paper-style bidirectional squared Chamfer sum, m^2; empty one-sided cloud penalized 200 m^2",
              "summary": summary, "rows": outputs}
    stage1.save_json(args.out, report)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
