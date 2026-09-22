"""GT-only diagnosis: can selected diffusion hole-fill points beat reprojection?

Greedily add diffusion hits only where the action-predicted warp has no hit.
At every addition, evaluate the exact change in the full bidirectional squared
Chamfer used by evaluate_lagen_metrics_10.py. GT is used only for this oracle
diagnostic; this script does not train or produce a deployable gate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import manifest_hash
from try_hard_geometry_inpaint import (load_selected, load_rays, next_state,
    transform_from_initial, warp_frame, stage1, load_motion, frame_points)


def oracle_additions(warp, diffusion, target, rays, threshold):
    """Exact greedy marginal Chamfer; fixed warp points are never removed."""
    warp_hit = warp[1, :, :18] > 0
    candidate_mask = ~warp_hit & (diffusion[1, :, :18] > threshold)
    positions = np.argwhere(candidate_mask)
    warp_points = frame_points(warp, rays)
    target_points = frame_points(target, rays)
    candidate_range = np.clip((diffusion[0, :, :18] + 1) * 5, 0, 10)
    candidate_points = rays[candidate_mask] * candidate_range[candidate_mask, None]
    chosen = np.zeros(len(positions), dtype=bool)
    if not len(warp_points) or not len(target_points) or not len(candidate_points):
        return chosen, positions, {"candidates": len(positions), "accepted": 0,
                                   "initial_improving": 0, "initial_worsening": 0,
                                   "initial_best_gain_m2": None}
    # All GT-to-candidate squared distances are computed once. The first
    # Chamfer direction is updated after each accepted point; the second has
    # an exact running sum and point count. No raywise proxy is used.
    gt_to_candidate = cdist(target_points, candidate_points, "sqeuclidean")
    candidate_to_gt = gt_to_candidate.min(axis=0)
    gt_nearest = np.square(cKDTree(warp_points).query(target_points)[0])
    pred_to_gt = np.square(cKDTree(target_points).query(warp_points)[0])
    pred_sum = float(pred_to_gt.sum())
    pred_count = len(warp_points)
    initial_cd = float(gt_nearest.mean() + pred_sum / pred_count)
    active = np.ones(len(positions), dtype=bool)
    first_gain = None
    while active.any():
        # Shape [GT points, candidate points]. Each column is the exact
        # first directional term if that one candidate is added next.
        first_term = np.minimum(gt_nearest[:, None], gt_to_candidate).mean(axis=0)
        second_term = (pred_sum + candidate_to_gt) / (pred_count + 1)
        proposed = first_term + second_term
        proposed[~active] = np.inf
        current = float(gt_nearest.mean() + pred_sum / pred_count)
        if first_gain is None:
            gains = current - proposed
            first_gain = {"initial_improving": int((gains > 1e-7).sum()),
                          "initial_worsening": int((gains < -1e-7).sum()),
                          "initial_best_gain_m2": float(max(gains.max(), 0))}
        best = int(proposed.argmin())
        if not np.isfinite(proposed[best]) or proposed[best] >= current - 1e-7:
            break
        chosen[best] = True
        active[best] = False
        gt_nearest = np.minimum(gt_nearest, gt_to_candidate[:, best])
        pred_sum += candidate_to_gt[best]
        pred_count += 1
    stats = {"candidates": len(positions), "accepted": int(chosen.sum()),
             **first_gain, "initial_cd_m2": initial_cd,
             "oracle_cd_m2_computed": float(gt_nearest.mean() + pred_sum / pred_count)}
    return chosen, positions, stats


def compose_gate(warp, diffusion, chosen, positions, threshold):
    result = warp.copy()
    if len(positions):
        selected = positions[chosen]
        if len(selected):
            x, y = selected[:, 0], selected[:, 1]
            result[0, x, y] = diffusion[0, x, y]
            result[1, x, y] = max(float(threshold) + 1, 2)
    return result


def aggregate(rows):
    groups = {"all": rows}
    groups.update({f"seed_{seed}": [r for r in rows if r["seed"] == seed]
                   for seed in sorted({r["seed"] for r in rows})})
    result = {}
    for key, group in groups.items():
        paired = [r for r in group if all(not r[m]["empty_cloud"]
                                         for m in ("warp", "all_hits", "oracle_add"))]
        result[key] = {
            "samples": len(group), "paired_nonempty": len(paired),
            "paired_cd_paper_m2": {m: float(np.mean([r[m]["cd_paper_m2"] for r in paired]))
                                   for m in ("warp", "all_hits", "oracle_add")},
            "all_samples": {m: summarize([r[m] for r in group])
                            for m in ("warp", "all_hits", "oracle_add")},
            "candidates": sum(r["gate"]["candidates"] for r in group),
            "accepted": sum(r["gate"]["accepted"] for r in group),
            "initial_improving": sum(r["gate"]["initial_improving"] for r in group),
            "initial_worsening": sum(r["gate"]["initial_worsening"] for r in group),
        }
    return result


@torch.no_grad()
def evaluate(args, split, model, vae, scheduler, motion, threshold):
    manifest, rows, arrays, scale = load_selected(args, split)
    rays = {seed: load_rays(args.raw_root, split, seed)
            for seed in sorted({r["seed"] for r in rows})}
    warped = []
    for i, row in enumerate(rows):
        state = arrays["prev_drone_state"][i].astype(np.float64)
        next_pose = next_state(state, arrays["action_sequence"][i].astype(np.float64), motion)
        grid, azimuth, elevation = rays[row["seed"]]
        warped.append(warp_frame(arrays["prev_range_values"][i],
                                 transform_from_initial(state, next_pose),
                                 grid, azimuth, elevation))
    warped = np.stack(warped)
    output = [None] * len(rows)
    for seed in sorted(rays):
        locations = np.array([i for i, row in enumerate(rows) if row["seed"] == seed])
        for start in range(0, len(locations), args.batch_size):
            selected = locations[start:start + args.batch_size]
            previous = torch.from_numpy(arrays["previous_latent"][selected]).to(stage1.DEVICE)
            actions = torch.from_numpy(arrays["normalized_actions"][selected]).to(stage1.DEVICE)
            states = torch.from_numpy(arrays["causal_state"][selected]).to(stage1.DEVICE)
            latent = stage1.generate(model, scheduler, previous, actions, states,
                                     seed=42 + seed * 10000 + start,
                                     init_strength=1.0, num_steps=20)
            generated = vae.decode(latent / scale).sample.cpu().numpy()
            for local, index in enumerate(selected):
                target = arrays["range_values"][index]
                warp = warped[index]
                diffusion = generated[local]
                grid = rays[seed][0]
                chosen, positions, stats = oracle_additions(warp, diffusion, target,
                                                             grid, threshold)
                oracle = compose_gate(warp, diffusion, chosen, positions, threshold)
                all_hits = compose_gate(warp, diffusion, np.ones(len(positions), bool),
                                        positions, threshold)
                output[index] = {**rows[index], "gate": stats,
                                 "warp": lidar_metric(warp, target, grid, 0),
                                 "all_hits": lidar_metric(all_hits, target, grid, 0),
                                 "oracle_add": lidar_metric(oracle, target, grid, 0)}
                if "oracle_cd_m2_computed" in stats:
                    error = abs(output[index]["oracle_add"]["cd_paper_m2"] -
                                stats["oracle_cd_m2_computed"])
                    if error > 1e-4:
                        raise AssertionError(f"Incremental Chamfer mismatch: {error}")
        print(json.dumps({"split": split, "seed": seed, "evaluated": len(locations)}),
              flush=True)
    result = {"split": split, "manifest_sha256": manifest_hash(manifest),
              "method": "GT-guided greedy exact marginal squared Chamfer, holes only; diagnostic",
              "samples": len(output), "summary": aggregate(output), "rows": output}
    stage1.save_json(args.out / f"{split}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=stage1.OUT / "completion_oracle")
    parser.add_argument("--manifest", type=Path, default=stage1.OUT / "representative_baseline")
    parser.add_argument("--unet", type=Path,
                        default=stage1.OUT / "world_circular_causal_8h" / "best.pt")
    parser.add_argument("--motion", type=Path,
                        default=stage1.OUT / "epona_probe" / "motion_ridge.npz")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    model = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    stage1.load_model(args.unet, model)
    vae = stage1.load_circular_vae()
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    motion = load_motion(args.motion)
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    for split in ("val", "test"):
        report = evaluate(args, split, model, vae, scheduler, motion, threshold)
        print(json.dumps({"split": split, "summary": report["summary"]["all"]
                          ["paired_cd_paper_m2"]}), flush=True)


if __name__ == "__main__":
    main()
