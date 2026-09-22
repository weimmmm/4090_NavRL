"""Causal geometry-guided sampling with the existing NavRL diffusion UNet.

The final future LiDAR frame always comes from DDIM and the circular VAE.
Predicted motion supplies only a noised latent initialization; it is never a
future ground-truth pose.  Choose strength on validation maps, then test once.
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
from lidar_wam.runner.lidar_geometry import load_rays, warp_frame
from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch, save_json
from try_epona_motion import next_state, transform_from_initial

ROOT = Path(__file__).resolve().parents[1]


def load_motion(path):
    with np.load(path) as archive:
        return {k: archive[k] for k in ("mean", "std", "weight", "bias")}


def selected(split, per_seed, data_root):
    manifest = json.loads((ROOT / "outputs" / "representative_baseline" /
                           "sample_manifest.json").read_text())
    rows = []
    for seed in sorted(stage1.EXPECTED_SEEDS[split]):
        candidates = [r for r in manifest["splits"][split] if r["seed"] == seed]
        if len(candidates) < per_seed:
            raise ValueError(f"Only {len(candidates)} fixed {split} samples for seed {seed}")
        rng = np.random.default_rng(42 + seed)
        chosen = rng.choice(len(candidates), size=per_seed, replace=False)
        rows.extend(sorted((candidates[i] for i in chosen),
                           key=lambda item: item["source_index"]))
    source = np.asarray([r["source_index"] for r in rows], dtype=np.int64)
    with h5py.File(data_root / f"navrl_static_{split}.h5", "r") as h5:
        arrays = {key: fetch(h5, key, source) for key in
                  ("prev_range_values", "range_values", "prev_drone_state",
                   "action_sequence")}
    cache = np.load(ROOT / "outputs" / "latents_circular" / f"{split}.npz")
    lookup = {int(value): i for i, value in enumerate(cache["source_index"])}
    index = [lookup[int(value)] for value in source]
    scale = json.loads((ROOT / "outputs" / "latents_circular" /
                        "metadata.json").read_text())["scaling_factor"]
    arrays["previous_latent"] = cache["previous"][index].copy() * scale
    arrays["normalized_action"] = cache["actions"][index].copy()
    arrays["causal_state"] = stage1.causal_state(cache["state"][index])
    if not np.isfinite(arrays["action_sequence"]).all():
        raise ValueError("Fixed manifest contains nonfinite executed command")
    return rows, arrays


@torch.no_grad()
def evaluate(split, strengths, args, model, vae, scheduler, motion, scale, threshold):
    rows, arrays = selected(split, args.per_seed, args.data_root)
    ray_cache = {seed: load_rays(args.raw_root, split, seed)
                 for seed in sorted({r["seed"] for r in rows})}
    warped = []
    for i, row in enumerate(rows):
        state = arrays["prev_drone_state"][i].astype(np.float64)
        command = arrays["action_sequence"][i].astype(np.float64)
        predicted = next_state(state, command, motion)
        rays, azimuth, elevation = ray_cache[row["seed"]]
        warped.append(warp_frame(arrays["prev_range_values"][i],
                                 transform_from_initial(state, predicted),
                                 rays, azimuth, elevation))
    warped = np.stack(warped)
    records = []
    for start in range(0, len(rows), args.batch_size):
        stop = min(start + args.batch_size, len(rows))
        previous = torch.from_numpy(arrays["previous_latent"][start:stop]).to(stage1.DEVICE)
        actions = torch.from_numpy(arrays["normalized_action"][start:stop]).to(stage1.DEVICE)
        states = torch.from_numpy(arrays["causal_state"][start:stop]).to(stage1.DEVICE)
        guide = vae.encode(torch.from_numpy(warped[start:stop]).to(
            stage1.DEVICE)).latent_dist.mode() * scale
        predictions = {}
        for strength in (1.0, *strengths):
            z = stage1.generate(model, scheduler, previous, actions, states,
                                seed=args.seed + start,
                                init_strength=strength, num_steps=20,
                                initial_latent=guide if strength < 1 else None)
            predictions[f"diffusion_{strength:g}"] = vae.decode(z / scale).sample.cpu().numpy()
        for i in range(start, stop):
            rays = ray_cache[rows[i]["seed"]][0]
            target = arrays["range_values"][i]
            result = {"source_index": rows[i]["source_index"],
                      "seed": rows[i]["seed"],
                      "copy": lidar_metric(arrays["prev_range_values"][i], target, rays, 0),
                      "warp": lidar_metric(warped[i], target, rays, 0)}
            for name, images in predictions.items():
                result[name] = lidar_metric(images[i - start], target, rays, threshold)
            records.append(result)
    methods = ("copy", "warp", "diffusion_1", *[f"diffusion_{s:g}" for s in strengths])
    summary = {method: {"all": summarize([r[method] for r in records]),
                        **{f"seed_{seed}": summarize([r[method] for r in records
                                                     if r["seed"] == seed])
                           for seed in sorted(ray_cache)}} for method in methods}
    return {"split": split, "samples": len(rows), "source_indices": [
                r["source_index"] for r in rows], "summary": summary, "rows": records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--per-seed", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--strengths", type=float, nargs="+", default=[0.35, 0.6, 0.8])
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "epona_probe" /
                   "guided_diffusion")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    motion = load_motion(ROOT / "outputs" / "epona_probe" / "motion_ridge.npz")
    model = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    checkpoint_path = ROOT / "outputs" / "world_circular_causal_8h" / "best.pt"
    checkpoint = stage1.load_model(checkpoint_path, model)
    vae = stage1.load_circular_vae()
    scale = json.loads((ROOT / "outputs" / "latents_circular" /
                        "metadata.json").read_text())["scaling_factor"]
    threshold = json.loads((ROOT / "outputs" / "vae_circular" /
                            "oracle_val.json").read_text())["selected_threshold"]
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    val = evaluate("val", args.strengths, args, model, vae, scheduler,
                   motion, scale, threshold)
    save_json(args.out / "validation.json", val)
    best = min(args.strengths, key=lambda strength: val["summary"][
        f"diffusion_{strength:g}"]["all"]["cd_paper_m2"])
    test = evaluate("test", [best], args, model, vae, scheduler,
                    motion, scale, threshold)
    save_json(args.out / "test.json", test)
    result = {"checkpoint": str(checkpoint_path), "checkpoint_step": checkpoint["step"],
              "final_output": "diffusion-generated LiDAR; geometry only initializes a noised latent",
              "sampling_steps": 20, "chosen_strength_val_only": best,
              "validation": {name: score["all"] for name, score in val["summary"].items()},
              "test": {name: score["all"] for name, score in test["summary"].items()},
              "note": "One-step fixed representative subset; no weight training or future-pose input."}
    save_json(args.out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
