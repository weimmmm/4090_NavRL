"""Separate recursive-LiDAR feedback error from stale ego-state error.

Uses the exact 512 test starts, actions, checkpoint, batch membership, and DDIM
noise from evaluate_lagen_metrics_10.py. Ground-truth previous LiDAR/state
variants are oracle diagnostics, never deployable prediction results.
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
from lidar_wam.runner.lidar_geometry import load_rays
from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch, save_json


HORIZONS = (5, 10)
METHODS = ("ar_frozen", "ar_oracle_state", "teacher_frozen",
           "teacher_oracle_state")


def summary(rows, horizon, seeds):
    horizon_rows = [r for r in rows if r["horizon"] == horizon]
    aggregate = {method: {"all": summarize([r[method] for r in horizon_rows]),
                          **{f"seed_{seed}": summarize([r[method] for r in horizon_rows
                                                        if r["seed"] == seed])
                             for seed in seeds}}
                 for method in METHODS}
    paired = [r for r in horizon_rows
              if all(not r[method]["empty_cloud"] for method in METHODS)]
    aggregate["paired_nonempty_cd_paper_m2"] = {
        method: sum(r[method]["cd_paper_m2"] for r in paired) / len(paired)
        for method in METHODS}
    aggregate["paired_nonempty_samples"] = len(paired)
    return aggregate


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=stage1.OUT / "representative_baseline")
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    args.raw_root = (args.raw_root or stage1.DATA.parent).expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    manifest = json.loads((args.output / "test_10frame_sample_manifest.json").read_text())
    original = json.loads((args.output / "test_lagen_style_10frame_metrics.json").read_text())
    if original["starts"] != len(manifest["rows"]):
        raise ValueError("Saved test manifest and original report disagree")
    rows = manifest["rows"]
    trajectories = np.asarray([r["trajectory_indices"] for r in rows], dtype=np.int64)
    starts = trajectories[:, 0]
    data = ExecutedLatents("test")
    lookup = {int(index): i for i, index in enumerate(data.indices)}
    if any(int(i) not in lookup for i in trajectories.flatten()):
        raise ValueError("Trajectory contains transition missing from latent cache")
    latent_positions = np.asarray([[lookup[int(i)] for i in chain]
                                   for chain in trajectories])
    original_lookup = {(r["source_index"], r["horizon"]): r["model"]
                       for r in original["rows"]}
    with h5py.File(stage1.DATA / "navrl_static_test.h5", "r") as h5:
        actions = np.stack([fetch(h5, "normalized_action_sequence", trajectories[:, h])
                            for h in range(10)], axis=1)
        states = np.stack([fetch(h5, "prev_ego_feats", trajectories[:, h])
                           for h in range(10)], axis=1)
        targets = {h: fetch(h5, "range_values", trajectories[:, h - 1])
                   for h in HORIZONS}
        first_targets = fetch(h5, "range_values", starts)
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Non-finite action or state")
    causal_states = torch.from_numpy(stage1.causal_state(states))
    latent_previous = data.previous[latent_positions]
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())[
        "scaling_factor"]
    threshold = original["mask_logit_threshold"]
    model = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    checkpoint = stage1.load_model(stage1.OUT / "world_circular_causal_8h" / "best.pt", model)
    if checkpoint["step"] != original["checkpoint_step"]:
        raise ValueError("Checkpoint changed since original evaluation")
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    vae = stage1.load_circular_vae()
    seeds = sorted({row["seed"] for row in rows})
    rays = {seed: load_rays(args.raw_root, "test", seed)[0] for seed in seeds}
    output = []
    for seed in seeds:
        locations = np.array([i for i, row in enumerate(rows) if row["seed"] == seed])
        ray_grid = rays[seed]
        for batch_start in range(0, len(locations), args.batch_size):
            chosen = locations[batch_start:batch_start + args.batch_size]
            generated = latent_previous[chosen, 0].to(stage1.DEVICE)
            frozen_state = causal_states[chosen, 0].to(stage1.DEVICE)
            for h in range(10):
                action = torch.from_numpy(actions[chosen, h]).to(stage1.DEVICE)
                actual_state = causal_states[chosen, h].to(stage1.DEVICE)
                noise_seed = manifest["random_seed"] + seed * 10000 + batch_start + h * 100000
                generated = stage1.generate(model, scheduler, generated, action,
                                            actual_state, noise_seed,
                                            init_strength=1.0, num_steps=20)
                if h == 0:
                    decoded = vae.decode(generated / scale).sample.cpu().numpy()
                    for j, index in enumerate(chosen):
                        score = lidar_metric(decoded[j], first_targets[index], ray_grid,
                                             threshold)["cd_paper_m2"]
                        expected = original_lookup[(int(starts[index]), 1)]["cd_paper_m2"]
                        if abs(score - expected) > 1e-4:
                            raise ValueError(f"First-frame replay differs: {score} vs {expected}")
                if h + 1 not in HORIZONS:
                    continue
                true_previous = latent_previous[chosen, h].to(stage1.DEVICE)
                teacher_frozen = stage1.generate(model, scheduler, true_previous,
                                                 action, frozen_state, noise_seed,
                                                 init_strength=1.0, num_steps=20)
                teacher_oracle = stage1.generate(model, scheduler, true_previous,
                                                 action, actual_state, noise_seed,
                                                 init_strength=1.0, num_steps=20)
                decoded = {
                    "ar_oracle_state": vae.decode(generated / scale).sample.cpu().numpy(),
                    "teacher_frozen": vae.decode(teacher_frozen / scale).sample.cpu().numpy(),
                    "teacher_oracle_state": vae.decode(teacher_oracle / scale).sample.cpu().numpy()}
                for j, index in enumerate(chosen):
                    target = targets[h + 1][index]
                    record = {"seed": seed, "source_index": int(starts[index]),
                              "target_source_index": int(trajectories[index, h]),
                              "horizon": h + 1, "time_s": round((h + 1) * .16, 2),
                              "ar_frozen": original_lookup[(int(starts[index]), h + 1)]}
                    for method, image in decoded.items():
                        record[method] = lidar_metric(image[j], target, ray_grid, threshold)
                    output.append(record)
        print(f"Seed {seed}: {len(locations)} starts diagnosed", flush=True)
    report = {"version": 1, "split": "test", "starts": len(starts),
              "horizons": list(HORIZONS), "checkpoint_step": checkpoint["step"],
              "ddim_steps_per_frame": 20, "batch_size": args.batch_size,
              "definition": {
                  "ar_frozen": "Original deployable evaluation: recursive generated latent; initial ego state held fixed",
                  "ar_oracle_state": "Recursive generated latent; TRUE per-frame previous ego state, unavailable without future execution",
                  "teacher_frozen": "TRUE previous LiDAR latent at the requested horizon; initial ego state held fixed; oracle diagnostic",
                  "teacher_oracle_state": "TRUE previous LiDAR latent and TRUE previous ego state at requested horizon; oracle one-step diagnostic",
                  "paired_cd": "Paper-formula squared Chamfer on identical samples where all four methods and GT are nonempty"},
              "summary": {str(h): summary(output, h, seeds) for h in HORIZONS},
              "rows": output}
    path = args.output / "test_rollout_input_diagnostics.json"
    save_json(path, report)
    print(json.dumps({"report": str(path), "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
