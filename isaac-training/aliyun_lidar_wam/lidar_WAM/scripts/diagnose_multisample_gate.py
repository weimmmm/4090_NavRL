"""No-training K=4/8 diffusion consensus gate for reprojection holes.

Gate thresholds are selected on validation squared Chamfer, then reported on
test maps. No GT enters gate features or the predicted point cloud.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import manifest_hash
from try_hard_geometry_inpaint import (load_selected, load_rays, next_state,
    transform_from_initial, warp_frame, stage1, load_motion)


SETTINGS = tuple((k, c, sigma) for k, counts in ((4, (2, 3, 4)),
                                                  (8, (4, 6, 8)))
                 for c in counts for sigma in (0.25, 0.5, 1.0, 2.0, float("inf")))
METHODS = ("warp", "single_all_hits", *(f"k{k}_c{c}_s{sigma:g}"
                                      for k, c, sigma in SETTINGS))


def ensemble_frame(warp, samples, threshold, count, minimum_hits, max_sigma):
    hit = samples[:count, 1, :, :18] > threshold
    ranges = np.clip((samples[:count, 0, :, :18] + 1) * 5, 0, 10)
    hit_count = hit.sum(axis=0)
    summed = (ranges * hit).sum(axis=0)
    mean = summed / np.maximum(hit_count, 1)
    variance = ((ranges * ranges * hit).sum(axis=0) /
                np.maximum(hit_count, 1) - mean * mean)
    deviation = np.sqrt(np.maximum(variance, 0))
    accept = ((warp[1, :, :18] <= 0) & (hit_count >= minimum_hits) &
              (deviation <= max_sigma))
    result = warp.copy()
    distance = result[0, :, :18]
    mask = result[1, :, :18]
    distance[accept] = mean[accept] / 5 - 1
    mask[accept] = 1
    return result, {"candidate_rays": int(((warp[1, :, :18] <= 0) &
                                            (hit_count > 0)).sum()),
                    "accepted_rays": int(accept.sum())}


def single_frame(warp, sample, threshold):
    hit = (warp[1, :, :18] <= 0) & (sample[1, :, :18] > threshold)
    result = warp.copy()
    result[0, :, :18][hit] = sample[0, :, :18][hit]
    result[1, :, :18][hit] = 1
    return result


def aggregate(rows):
    groups = {"all": rows}
    groups.update({f"seed_{seed}": [r for r in rows if r["seed"] == seed]
                   for seed in sorted({r["seed"] for r in rows})})
    output = {}
    for name, group in groups.items():
        # Same nonempty warp+GT sample set for every gate; gates always retain
        # the warp, so this excludes only baseline-empty geometry.
        paired = [r for r in group if not r["warp"]["empty_cloud"]]
        output[name] = {"samples": len(group), "paired_nonempty": len(paired),
                        "paired_cd_paper_m2": {m: float(np.mean([
                            r[m]["cd_paper_m2"] for r in paired])) for m in METHODS},
                        "all_samples": {m: summarize([r[m] for r in group])
                                        for m in METHODS},
                        "accepted_rays": {m: sum(r["gate"][m]["accepted_rays"]
                                                for r in group) for m in METHODS[2:]}}
    return output


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
            chosen = locations[start:start + args.batch_size]
            previous = torch.from_numpy(arrays["previous_latent"][chosen]).to(stage1.DEVICE)
            actions = torch.from_numpy(arrays["normalized_actions"][chosen]).to(stage1.DEVICE)
            states = torch.from_numpy(arrays["causal_state"][chosen]).to(stage1.DEVICE)
            samples = []
            for draw in range(8):
                latent = stage1.generate(model, scheduler, previous, actions, states,
                                         seed=42 + seed * 10000 + start + draw * 100000,
                                         init_strength=1.0, num_steps=20)
                samples.append(vae.decode(latent / scale).sample.cpu().numpy())
            samples = np.stack(samples, axis=0)
            for local, index in enumerate(chosen):
                grid = rays[seed][0]
                target = arrays["range_values"][index]
                warp = warped[index]
                frames = {"warp": warp,
                          "single_all_hits": single_frame(warp, samples[0, local], threshold)}
                gate_stats = {}
                for k, count, sigma in SETTINGS:
                    name = f"k{k}_c{count}_s{sigma:g}"
                    frames[name], gate_stats[name] = ensemble_frame(
                        warp, samples[:, local], threshold, k, count, sigma)
                record = {**rows[index], "gate": gate_stats}
                for name, frame in frames.items():
                    record[name] = lidar_metric(frame, target, grid, 0)
                output[index] = record
        print(json.dumps({"split": split, "seed": seed, "evaluated": len(locations)}),
              flush=True)
    report = {"split": split, "manifest_sha256": manifest_hash(manifest),
              "samples": len(output), "sampling": "8 independent DDIM draws; first 4 used for K=4; first draw matches previous benchmark",
              "gate": "warp hit preserved; on warp-missing rays accept mean of hit ranges if hit count and conditional range std pass validation-chosen thresholds",
              "methods": METHODS, "summary": aggregate(output), "rows": output}
    stage1.save_json(args.out / f"{split}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=stage1.OUT / "completion_multisample")
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
    validation = evaluate(args, "val", model, vae, scheduler, motion, threshold)
    scores = validation["summary"]["all"]["paired_cd_paper_m2"]
    selected = min(METHODS[2:], key=lambda name: scores[name])
    stage1.save_json(args.out / "selection.json", {
        "selected_gate": selected, "criterion": "lowest validation paired squared Chamfer",
        "validation_paired_cd_paper_m2": scores})
    print(json.dumps({"selected": selected, "validation_cd_m2": scores[selected]}), flush=True)
    test = evaluate(args, "test", model, vae, scheduler, motion, threshold)
    print(json.dumps({"selected": selected, "test_cd_m2": test["summary"]["all"]
                      ["paired_cd_paper_m2"][selected]}), flush=True)


if __name__ == "__main__":
    main()
