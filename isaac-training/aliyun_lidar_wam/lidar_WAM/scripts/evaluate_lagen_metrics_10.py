"""Evaluate five and ten recursive LiDAR frames with LaGen-style metric units.

Selects 256 ten-frame-eligible starts per held-out test seed with RNG 42.
All horizons share the same starts; only the initial observation and ego state
plus the recorded action block for each interval are supplied to the model.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.executed_residual import ExecutedLatents
from lidar_wam.runner.lidar_geometry import frame_points, load_rays, predict_transform, warp_frame
from evaluate_representative import fetch, save_json


RANGE_MAX_M = 10.0


def choose_trajectories(h5, valid_indices, horizons, per_seed=256, random_seed=42):
    """Follow token links: HDF5 rows interleave concurrent Isaac environments."""
    tokens = h5["token"][:]
    previous_tokens = h5["prev_token"][:]
    scene = h5["scene_token"][:]
    frame = h5["frame_idx"][:]
    seeds = h5["terrain_seed"][:]
    successor = {token: i for i, token in enumerate(previous_tokens)}
    valid = set(map(int, valid_indices))
    candidates = {18: [], 19: []}
    for index in valid_indices:
        index = int(index)
        chain = [index]
        for _ in range(1, horizons):
            next_index = successor.get(tokens[chain[-1]])
            if next_index is None or next_index not in valid:
                break
            chain.append(next_index)
        if len(chain) != horizons:
            continue
        if any(scene[j] != scene[index] or frame[j] != frame[index] + h
               for h, j in enumerate(chain)):
            continue
        candidates[int(seeds[index])].append(chain)
    rng = np.random.default_rng(random_seed)
    selected = []
    checked = {seed: 0 for seed in candidates}
    for seed in sorted(candidates):
        for position in rng.permutation(len(candidates[seed])):
            chain = candidates[seed][position]
            checked[seed] += 1
            if not all(np.array_equal(h5["prev_range_values"][chain[h]],
                                      h5["range_values"][chain[h - 1]])
                       for h in range(1, horizons)):
                continue
            selected.append(chain)
            if sum(int(seeds[chain[0]]) == seed for chain in selected) == per_seed:
                break
        if sum(int(seeds[chain[0]]) == seed for chain in selected) < per_seed:
            raise ValueError(f"Only {len(candidates[seed])} candidates for seed {seed}")
    selected = np.array(sorted(selected, key=lambda chain: chain[0]), dtype=np.int64)
    starts = selected[:, 0]
    manifest = {"version": 1, "split": "test", "random_seed": random_seed,
                "samples_per_seed": per_seed, "horizons": horizons,
                "candidate_counts": {str(seed): len(rows) for seed, rows in candidates.items()},
                "checked_counts": {str(seed): count for seed, count in checked.items()},
                "selection": f"Uniform without replacement among valid {horizons}-frame same-episode token-linked sequences",
                "rows": [{"source_index": int(i), "seed": int(seeds[i]),
                          "frame_idx": int(frame[i]), "scene_token": scene[i].decode(),
                          "trajectory_indices": list(map(int, chain))}
                         for i, chain in zip(starts, selected)]}
    return selected, manifest


def lidar_metric(prediction, target, rays, threshold):
    predicted = frame_points(prediction, rays, threshold)
    truth = frame_points(target, rays)
    if not len(predicted) and not len(truth):
        directional_a = directional_b = 0.0
    elif not len(predicted) or not len(truth):
        # Chamfer is undefined for an empty cloud. Penalize the absent side
        # by the squared sensor range in each direction and count such cases.
        directional_a = directional_b = RANGE_MAX_M ** 2
    else:
        directional_a = float(np.square(cKDTree(predicted).query(truth)[0]).mean())
        directional_b = float(np.square(cKDTree(truth).query(predicted)[0]).mean())
    actual_hit = target[1, :, :18] > 0
    predicted_hit = prediction[1, :, :18] > threshold
    actual_distance = np.clip((target[0, :, :18] + 1) * 5, 0, RANGE_MAX_M)
    predicted_distance = np.clip((prediction[0, :, :18] + 1) * 5, 0, RANGE_MAX_M)
    actual_depth = np.where(actual_hit, actual_distance, RANGE_MAX_M)
    predicted_depth = np.where(predicted_hit, predicted_distance, RANGE_MAX_M)
    error = np.abs(actual_depth - predicted_depth)
    hit_error = error[actual_hit]
    relative = hit_error / np.maximum(actual_depth[actual_hit], 1e-6)
    return {"cd_paper_m2": directional_a + directional_b,
            "cd_released_code_m2": (directional_a + directional_b) / 2,
            "gt_hit_l1_sum_m": float(hit_error.sum()),
            "gt_hit_absrel_sum": float(relative.sum()),
            "gt_hit_count": int(actual_hit.sum()),
            "all_ray_l1_sum_m": float(error.sum()),
            "all_ray_count": int(error.size),
            "tp": int((predicted_hit & actual_hit).sum()),
            "fp": int((predicted_hit & ~actual_hit).sum()),
            "fn": int((~predicted_hit & actual_hit).sum()),
            "empty_cloud": bool(not len(predicted) or not len(truth))}


def summarize(records):
    n = len(records)
    count = sum(r["gt_hit_count"] for r in records)
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    return {"samples": n,
            "cd_paper_m2": sum(r["cd_paper_m2"] for r in records) / n,
            "cd_released_code_m2": sum(r["cd_released_code_m2"] for r in records) / n,
            "gt_hit_l1_m": sum(r["gt_hit_l1_sum_m"] for r in records) / max(count, 1),
            "gt_hit_absrel_percent": 100 * sum(r["gt_hit_absrel_sum"] for r in records) /
                                     max(count, 1),
            "all_ray_l1_m": sum(r["all_ray_l1_sum_m"] for r in records) /
                            max(sum(r["all_ray_count"] for r in records), 1),
            "mask_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "empty_cloud_cases": sum(r["empty_cloud"] for r in records)}


@torch.no_grad()
def evaluate(args):
    data = ExecutedLatents("test")
    with h5py.File(stage1.DATA / "navrl_static_test.h5", "r") as h5:
        trajectories, manifest = choose_trajectories(
            h5, data.indices, args.horizons, args.samples_per_seed, args.random_seed)
        indices = trajectories[:, 0]
        actions = np.stack([fetch(h5, "normalized_action_sequence", trajectories[:, h])
                            for h in range(args.horizons)], axis=1)
        targets = np.stack([fetch(h5, "range_values", trajectories[:, h])
                            for h in range(args.horizons)], axis=1)
        initial_frames = fetch(h5, "prev_range_values", indices)
        initial_states = fetch(h5, "prev_ego_feats", indices)
        initial_drone_states = fetch(h5, "prev_drone_state", indices)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / f"test_{args.horizons}frame_sample_manifest.json"
    save_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path),
                      "candidate_counts": manifest["candidate_counts"],
                      "selected": len(indices)}), flush=True)
    if not np.isfinite(actions).all() or not np.isfinite(targets).all():
        raise ValueError("Non-finite condition or target in selected trajectories")

    lookup = {int(index): position for position, index in enumerate(data.indices)}
    previous_latents = data.previous[[lookup[int(i)] for i in indices]]
    states = torch.from_numpy(stage1.causal_state(initial_states))
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())[
        "scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR / "oracle_val.json").read_text())[
        "selected_threshold"]
    model = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    checkpoint_path = args.checkpoint or stage1.OUT / "world_circular_causal_8h" / "best.pt"
    checkpoint = stage1.load_model(checkpoint_path, model)
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    vae = stage1.load_circular_vae()
    rays = {seed: load_rays(args.raw_root, "test", seed)
            for seed in sorted({row["seed"] for row in manifest["rows"]})}
    output = []
    for seed in sorted(rays):
        locations = np.array([i for i, row in enumerate(manifest["rows"])
                              if row["seed"] == seed])
        ray_grid, azimuth, elevation = rays[seed]
        for batch_start in range(0, len(locations), args.batch_size):
            chosen = locations[batch_start:batch_start + args.batch_size]
            previous = previous_latents[chosen].to(stage1.DEVICE)
            state = states[chosen].to(stage1.DEVICE)
            for h in range(args.horizons):
                action = torch.from_numpy(actions[chosen, h]).to(stage1.DEVICE)
                previous = stage1.generate(model, scheduler, previous, action,
                                           state, args.random_seed + seed * 10000 +
                                           batch_start + h * 100000,
                                           init_strength=1.0, num_steps=20)
                predicted = vae.decode(previous / scale).sample.cpu().numpy()
                for j, location in enumerate(chosen):
                    target = targets[location, h]
                    initial = initial_frames[location]
                    warp = warp_frame(initial, predict_transform(
                        initial_drone_states[location], dt=0.16 * (h + 1)),
                        ray_grid, azimuth, elevation)
                    output.append({"source_index": int(indices[location]),
                                   "target_source_index": int(trajectories[location, h]),
                                   "seed": seed, "frame_idx": manifest["rows"][location]["frame_idx"],
                                   "horizon": h + 1, "time_s": round(0.16 * (h + 1), 2),
                                   "model": lidar_metric(predicted[j], target, ray_grid, threshold),
                                   "copy": lidar_metric(initial, target, ray_grid, 0),
                                   "velocity": lidar_metric(warp, target, ray_grid, 0)})
        print(f"Seed {seed}: {len(locations)} starts, {args.horizons} horizons complete", flush=True)
    summary = {str(h): {method: {"all": summarize([r[method] for r in output
                                                  if r["horizon"] == h]),
                                 **{f"seed_{seed}": summarize([r[method] for r in output
                                                                if r["horizon"] == h and
                                                                r["seed"] == seed])
                                    for seed in sorted(rays)}}
                        for method in ("model", "copy", "velocity")}
               for h in range(1, args.horizons + 1)}
    report = {"version": 1, "split": "test", "checkpoint_step": checkpoint.get(
                  "fine_tune_step", checkpoint.get("step")),
              "checkpoint": str(checkpoint_path),
              "ddim_steps_per_frame": 20, "mask_logit_threshold": threshold,
              "sample_manifest": str(manifest_path), "starts": len(indices),
              "definition": {
                  "cd_paper_m2": "mean GT-to-pred squared nearest-neighbor distance plus mean pred-to-GT squared nearest-neighbor distance",
                  "cd_released_code_m2": "one-half of paper-formula Chamfer, matching LaGen inference_autoregressive4D_nus.py function",
                  "gt_hit_l1_m": "GT-hit query rays only; a predicted miss is assigned sensor max range 10 m",
                  "gt_hit_absrel_percent": "GT-hit ray absolute relative range error times 100",
                  "all_ray_l1_m": "all 1944 physical rays; misses assigned sensor max range 10 m",
                  "empty_cloud": "when one cloud is empty, each directional squared distance is assigned 100 m^2",
                  "causal_input": "initial latent and ego state, ten different recorded actions per 0.16 s interval, recursively generated latent only afterward"},
              "summary": summary, "rows": output}
    path = args.output / args.report_name
    save_json(path, report)
    requested = {str(h): summary[str(h)] for h in args.report_horizons
                 if 1 <= h <= args.horizons}
    print(json.dumps({"report": str(path), "starts": len(indices),
                      "requested_horizons": requested}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizons", type=int, default=10)
    parser.add_argument("--report-horizons", type=int, nargs="+", default=[3, 6, 9, 10])
    parser.add_argument("--samples-per-seed", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, default=stage1.OUT / "representative_baseline")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report-name", default="test_lagen_style_10frame_metrics.json")
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    args.raw_root = (args.raw_root or stage1.DATA.parent).expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    evaluate(args)


if __name__ == "__main__":
    main()
