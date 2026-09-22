"""Five-frame autoregressive evaluation of the selected causal LaGen UNet.

Uses the fixed representative manifest. No future LiDAR frame or future ego
state is provided to the predictor; each horizon has its own recorded ten
actions. Results are paired against initial-frame copy and constant-velocity
ray reprojection on exactly the same sequence starts.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.executed_residual import ExecutedLatents
from lidar_wam.runner.lidar_geometry import load_rays, predict_transform, warp_frame
from evaluate_representative import fetch, manifest_hash, metric, save_json, summarize_metrics


def eligible_starts(manifest, valid_indices, h5, horizons=5):
    valid = set(map(int, valid_indices))
    tokens = h5["token"][:]
    previous_tokens = h5["prev_token"][:]
    scenes = h5["scene_token"][:]
    frames = h5["frame_idx"][:]
    successors = {token: i for i, token in enumerate(previous_tokens)}
    selected, trajectories = [], []
    rejection = {"missing_valid_transition": 0, "different_episode": 0,
                 "nonconsecutive_frame": 0, "disconnected_range": 0}
    for row in manifest["splits"]["test"]:
        i = row["source_index"]
        chain = [i]
        for _ in range(1, horizons):
            successor = successors.get(tokens[chain[-1]])
            if successor is None:
                break
            chain.append(successor)
        if len(chain) != horizons or any(j not in valid for j in chain):
            rejection["missing_valid_transition"] += 1
            continue
        if any(scenes[j] != scenes[i] for j in chain):
            rejection["different_episode"] += 1
            continue
        if not np.array_equal(frames[chain], frames[i] + np.arange(horizons)):
            rejection["nonconsecutive_frame"] += 1
            continue
        # The stored previous frame of each segment must equal the preceding
        # segment's target frame, including both range and hit channels.
        connected = all(np.array_equal(h5["prev_range_values"][chain[h]],
                                           h5["range_values"][chain[h - 1]])
                        for h in range(1, horizons))
        if not connected:
            rejection["disconnected_range"] += 1
            continue
        selected.append(row)
        trajectories.append(chain)
    return selected, np.asarray(trajectories, dtype=np.int64), rejection


def summarise_rows(rows, field, horizon):
    selected = [r for r in rows if r["horizon"] == horizon]
    result = {"all": summarize_metrics([r[field] for r in selected])}
    for seed in sorted({r["seed"] for r in selected}):
        result[f"seed_{seed}"] = summarize_metrics(
            [r[field] for r in selected if r["seed"] == seed])
    return result


@torch.no_grad()
def evaluate(args):
    manifest = json.loads((args.output / "sample_manifest.json").read_text())
    data = ExecutedLatents("test")
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())["scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR / "oracle_val.json").read_text())["selected_threshold"]
    with h5py.File(stage1.DATA / "navrl_static_test.h5", "r") as h5:
        starts, trajectories, rejection = eligible_starts(manifest, data.indices, h5)
        if not starts:
            raise ValueError(f"No valid five-step trajectories: {rejection}")
        indices = np.array([r["source_index"] for r in starts], dtype=np.int64)
        actions = np.stack([fetch(h5, "normalized_action_sequence", trajectories[:, h])
                            for h in range(5)], axis=1)
        targets = np.stack([fetch(h5, "range_values", trajectories[:, h])
                            for h in range(5)], axis=1)
        initial_frames = fetch(h5, "prev_range_values", indices)
        initial_states = fetch(h5, "prev_ego_feats", indices)
        initial_drone_states = fetch(h5, "prev_drone_state", indices)
    if not np.isfinite(actions).all() or not np.isfinite(targets).all():
        raise ValueError("Non-finite trajectory data passed the eligibility filter")

    lookup = {int(index): position for position, index in enumerate(data.indices)}
    previous_latents = data.previous[[lookup[int(i)] for i in indices]]
    causal_states = torch.from_numpy(stage1.causal_state(initial_states))
    model = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    checkpoint = stage1.load_model(stage1.OUT / args.checkpoint_run / args.checkpoint_name, model)
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    vae = stage1.load_circular_vae()
    rays = {seed: load_rays(args.raw_root, "test", seed)
            for seed in sorted({r["seed"] for r in starts})}

    records = []
    for seed in sorted(rays):
        locations = np.array([i for i, r in enumerate(starts) if r["seed"] == seed])
        ray_grid, azimuth, elevation = rays[seed]
        for batch_start in range(0, len(locations), args.batch_size):
            chosen = locations[batch_start:batch_start + args.batch_size]
            previous = previous_latents[chosen].to(stage1.DEVICE)
            state = causal_states[chosen].to(stage1.DEVICE)
            for h in range(5):
                action = torch.from_numpy(actions[chosen, h]).to(stage1.DEVICE)
                # Generated latent is passed directly to the next step.
                previous = stage1.generate(model, scheduler, previous, action,
                                           state, 42 + seed * 10000 + batch_start + h * 100000,
                                           init_strength=1.0, num_steps=20)
                predicted = vae.decode(previous / scale).sample.cpu().numpy()
                for j, location in enumerate(chosen):
                    target = targets[location, h]
                    frame = initial_frames[location]
                    warp = warp_frame(frame, predict_transform(
                        initial_drone_states[location], dt=0.16 * (h + 1)),
                        ray_grid, azimuth, elevation)
                    records.append({**starts[location], "horizon": h + 1,
                                    "target_source_index": int(trajectories[location, h]),
                                    "autoregressive": metric(predicted[j], target, ray_grid,
                                                             threshold),
                                    "copy_initial": metric(frame, target, ray_grid),
                                    "velocity_initial": metric(warp, target, ray_grid)})
        print(f"Seed {seed}: {len(locations)} five-frame trajectories complete", flush=True)

    summary = {str(h): {name: summarise_rows(records, name, h)
                         for name in ("autoregressive", "copy_initial", "velocity_initial")}
               for h in range(1, 6)}
    report = {"version": 1, "split": "test", "manifest_sha256": manifest_hash(manifest),
              "checkpoint_run": args.checkpoint_run, "checkpoint_name": args.checkpoint_name,
              "checkpoint_step": checkpoint["step"], "ddim_steps_per_frame": 20,
              "mask_logit_threshold": threshold, "selected_starts": len(starts),
              "manifest_starts": len(manifest["splits"]["test"]), "rejected": rejection,
              "definition": "Five recursively generated latents, five distinct recorded 10x3 action blocks, fixed initial causal ego state; no future frame or future ego state to model",
              "caveat": "Recorded future PPO actions are supplied controls, not proven counterfactual action plans",
              "summary": summary, "rows": records}
    destination = args.output / "test_lagen_unet_8h_autoregressive_5.json"
    save_json(destination, report)
    print(json.dumps({"report": str(destination), "starts": len(starts),
                      "rejected": rejection,
                      "chamfer_m": {h: {name: summary[str(h)][name]["all"]["chamfer_m"]
                                         for name in summary[str(h)]}
                                    for h in range(1, 6)}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, default=stage1.OUT / "representative_baseline")
    parser.add_argument("--checkpoint-run", default="world_circular_causal_8h")
    parser.add_argument("--checkpoint-name", default="best.pt")
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    args.raw_root = (args.raw_root or stage1.DATA.parent).expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    evaluate(args)


if __name__ == "__main__":
    main()
