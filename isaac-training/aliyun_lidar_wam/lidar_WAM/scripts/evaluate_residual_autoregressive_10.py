"""Roll out the existing single-frame executed-action residual diffusion for ten frames."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.executed_residual import ExecutedLatents, condition_stats, FULL_DIR
from lidar_wam.runner.lidar_geometry import load_rays
from evaluate_lagen_metrics_10 import HORIZONS, choose_trajectories, lidar_metric, summarize
from evaluate_representative import fetch, save_json


@torch.no_grad()
def evaluate(args):
    stats = condition_stats()
    data = ExecutedLatents("test")
    with h5py.File(stage1.DATA / "navrl_static_test.h5", "r") as h5:
        trajectories, manifest = choose_trajectories(
            h5, data.indices, args.samples_per_seed, args.random_seed)
        initial = fetch(h5, "prev_range_values", trajectories[:, 0])
        targets = np.stack([fetch(h5, "range_values", trajectories[:, h])
                            for h in range(HORIZONS)], axis=1)
        deltas = np.stack([fetch(h5, "step_delta", trajectories[:, h])
                           for h in range(HORIZONS)], axis=1)
    if not np.all(deltas == 10):
        raise ValueError("Every predicted interval must contain exactly ten simulation steps")
    lookup = {int(source): i for i, source in enumerate(data.indices)}
    positions = np.array([[lookup[int(source)] for source in chain]
                          for chain in trajectories], dtype=np.int64)
    actions = data.actions[positions]
    if not torch.isfinite(actions).all():
        raise ValueError("Non-finite executed action in selected trajectories")
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "test_10frame_sample_manifest.json", manifest)

    checkpoint_path = stage1.OUT / FULL_DIR / "best.pt"
    config = json.loads((stage1.OUT / FULL_DIR / "config.json").read_text())
    if config["vae_sha256"] != stats["vae_sha256"] or config["residual_scale"] != stats["residual_scale"]:
        raise ValueError("Residual checkpoint configuration differs from the latent cache")
    model = stage1.WorldModel(state_dim=11).to(stage1.DEVICE).float().eval()
    checkpoint = stage1.load_model(checkpoint_path, model)
    vae = stage1.load_circular_vae()
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())["scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR / "oracle_val.json").read_text())["selected_threshold"]
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    rays = {seed: load_rays(args.raw_root, "test", seed)[0]
            for seed in sorted(set(int(x) for x in data.seeds[positions[:, 0]]))}
    rows = []
    for seed in sorted(rays):
        locations = np.flatnonzero(data.seeds[positions[:, 0]] == seed)
        for batch_start in range(0, len(locations), args.batch_size):
            chosen = locations[batch_start:batch_start + args.batch_size]
            latent = data.previous[positions[chosen, 0]].to(stage1.DEVICE)
            initial_state = data.state[positions[chosen, 0]].to(stage1.DEVICE)
            for h in range(HORIZONS):
                action = actions[chosen, h].to(stage1.DEVICE)
                latent = stage1.generate(
                    model, scheduler, latent, action, initial_state,
                    args.random_seed + seed * 10000 + batch_start + h * 100000,
                    init_strength=args.init_strength, num_steps=args.ddim_steps,
                    residual_scale=stats["residual_scale"])
                predicted = vae.decode(latent / scale).sample.cpu().numpy()
                true_latent = data.target[positions[chosen, h]].to(stage1.DEVICE)
                latent_mse = (latent - true_latent).square().flatten(1).mean(1).cpu().numpy()
                for j, location in enumerate(chosen):
                    target = targets[location, h]
                    rows.append({
                        "source_index": int(trajectories[location, 0]),
                        "target_source_index": int(trajectories[location, h]),
                        "seed": seed,
                        "frame_idx": manifest["rows"][location]["frame_idx"],
                        "horizon": h + 1,
                        "time_s": round(0.16 * (h + 1), 2),
                        "latent_mse": float(latent_mse[j]),
                        "model": lidar_metric(predicted[j], target, rays[seed], threshold),
                        "copy_initial": lidar_metric(initial[location], target, rays[seed], 0),
                    })
        print(f"seed={seed}: {len(locations)} starts, all ten horizons complete", flush=True)
        save_json(args.output / "test_per_sample.json", rows)

    summary = {}
    for h in range(1, HORIZONS + 1):
        horizon_rows = [row for row in rows if row["horizon"] == h]
        summary[str(h)] = {
            "latent_mse": float(np.mean([row["latent_mse"] for row in horizon_rows])),
            "model": {"all": summarize([row["model"] for row in horizon_rows]),
                      **{f"seed_{seed}": summarize([row["model"] for row in horizon_rows
                                                     if row["seed"] == seed]) for seed in rays}},
            "copy_initial": {"all": summarize([row["copy_initial"] for row in horizon_rows]),
                             **{f"seed_{seed}": summarize([row["copy_initial"] for row in horizon_rows
                                                            if row["seed"] == seed]) for seed in rays}},
        }
    report = {"version": 1, "split": "test", "samples": len(trajectories),
              "checkpoint": str(checkpoint_path), "checkpoint_step": checkpoint["step"],
              "vae_sha256": stats["vae_sha256"], "ddim_steps_per_frame": args.ddim_steps,
              "init_strength": args.init_strength, "mask_logit_threshold": threshold,
              "state_protocol": "The initial measured 11D drone state is held fixed for every future step; no future measured state is supplied.",
              "rollout_protocol": "The generated latent is fed recursively to the next interval; each interval uses its recorded ten executed world-frame commands.",
              "chamfer_units": "cd_paper_m2 is sum of bidirectional mean squared nearest-neighbor distances (m^2); cd_released_code_m2 is half that value.",
              "manifest": str(args.output / "test_10frame_sample_manifest.json"),
              "per_sample": str(args.output / "test_per_sample.json"),
              "summary": summary}
    save_json(args.output / "test_summary.json", report)
    print(json.dumps({"report": str(args.output / "test_summary.json"),
                      "horizon_5": summary["5"], "horizon_10": summary["10"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-seed", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--init-strength", type=float, default=0.05)
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, default=stage1.OUT / "residual_autoregressive_10")
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    args.raw_root = (args.raw_root or stage1.DATA.parent).expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    evaluate(args)


if __name__ == "__main__":
    main()
