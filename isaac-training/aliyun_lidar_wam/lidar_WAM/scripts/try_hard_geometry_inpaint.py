"""No-training test of causal hard geometry preservation with diffusion fill.

The existing UNet samples a full candidate future LiDAR. Reliable projected
hits from the current frame replace its output in range-image space. An
oracle mask using the next ground truth is diagnostic only and is never a
candidate for deployment or a test-set selection rule.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_geometry import frame_points, load_rays, warp_frame
from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch, manifest_hash
from try_epona_guided_diffusion import load_motion
from try_epona_motion import next_state, transform_from_initial


CAUSAL_RULES = (
    "all_hits", "support3_depth025", "support3_depth05", "support5_depth05",
    "support3_depth1", "support5_depth1",
)
METHODS = ("copy", "warp", "diffusion", *CAUSAL_RULES, "oracle_gt_mask")


def shifted(array, horizontal, vertical, fill):
    """Horizontal angle wraps; the vertical LiDAR axis never wraps."""
    rolled = np.roll(array, horizontal, axis=0)
    output = np.full_like(array, fill)
    if vertical >= 0:
        output[:, vertical:] = rolled[:, :array.shape[1] - vertical]
    else:
        output[:, :vertical] = rolled[:, -vertical:]
    return output


def masks_from_warp(warp):
    hit = warp[1, :, :18] > 0
    distance = np.clip((warp[0, :, :18] + 1) * 5, 0, 10)
    support = np.zeros_like(distance, dtype=np.int16)
    smallest = np.full_like(distance, np.inf)
    largest = np.full_like(distance, -np.inf)
    for horizontal in (-1, 0, 1):
        for vertical in (-1, 0, 1):
            nearby = shifted(hit, horizontal, vertical, False)
            nearby_distance = shifted(distance, horizontal, vertical, 10.0)
            support += nearby
            smallest = np.minimum(smallest, np.where(nearby, nearby_distance, np.inf))
            largest = np.maximum(largest, np.where(nearby, nearby_distance, -np.inf))
    spread = largest - smallest
    return {
        "all_hits": hit,
        "support3_depth025": hit & (support >= 3) & (spread <= 0.25),
        "support3_depth05": hit & (support >= 3) & (spread <= 0.5),
        "support5_depth05": hit & (support >= 5) & (spread <= 0.5),
        "support3_depth1": hit & (support >= 3) & (spread <= 1.0),
        "support5_depth1": hit & (support >= 5) & (spread <= 1.0),
    }


def oracle_mask(warp, diffusion, target, threshold):
    """GT-informed per-ray selector for hit-aware L1, not a Chamfer bound."""
    warp_hit = warp[1, :, :18] > 0
    diffusion_hit = diffusion[1, :, :18] > threshold
    true_hit = target[1, :, :18] > 0
    warp_range = np.clip((warp[0, :, :18] + 1) * 5, 0, 10)
    diffusion_range = np.clip((diffusion[0, :, :18] + 1) * 5, 0, 10)
    true_range = np.clip((target[0, :, :18] + 1) * 5, 0, 10)
    warp_error = np.where(true_hit, np.abs(warp_range - true_range), 10.0)
    diffusion_error = np.where(diffusion_hit == true_hit,
                               np.where(true_hit, np.abs(diffusion_range - true_range), 0.0),
                               10.0)
    return warp_hit & (warp_error < diffusion_error)


def compose(warp, diffusion, lock, threshold):
    result = diffusion.copy()
    if not np.all(warp[1, :, :18][lock] > 0):
        raise ValueError("Only projected hits may be hard-locked")
    distance = result[0, :, :18]
    distance[lock] = warp[0, :, :18][lock]
    # The raw warp encodes hit as +1, while generated frames are evaluated
    # with a calibrated mask-logit threshold of 1.5. Preserve the hit meaning.
    mask_logit = result[1, :, :18]
    mask_logit[lock] = max(float(threshold) + 1.0, 2.0)
    return result


def lock_diagnostics(lock, warp, target):
    true_hit = target[1, :, :18] > 0
    warp_range = np.clip((warp[0, :, :18] + 1) * 5, 0, 10)
    true_range = np.clip((target[0, :, :18] + 1) * 5, 0, 10)
    correct = true_hit & (np.abs(warp_range - true_range) <= 0.3)
    return {"locked": int(lock.sum()), "correct_0p3m": int((lock & correct).sum()),
            "false_0p3m": int((lock & ~correct).sum()),
            "gt_hit_locked": int((lock & true_hit).sum())}


def new_visible_scores(prediction, warp, target, threshold):
    new = (target[1, :, :18] > 0) & (warp[1, :, :18] <= 0)
    predicted = prediction[1, :, :18] > threshold
    return {"gt_new_hits": int(new.sum()), "predicted_new_hits": int((new & predicted).sum())}


def load_selected(args, split):
    manifest = json.loads((args.manifest / "sample_manifest.json").read_text())
    rows = manifest["splits"][split]
    source = np.array([row["source_index"] for row in rows], dtype=np.int64)
    cached = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
    lookup = {int(index): i for i, index in enumerate(cached["source_index"])}
    if any(int(index) not in lookup for index in source):
        raise ValueError("Fixed manifest includes a stale or invalid transition")
    positions = np.array([lookup[int(index)] for index in source])
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR /
                        "metadata.json").read_text())["scaling_factor"]
    with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
        if not np.all(fetch(h5, "step_delta", source) == 10):
            raise ValueError("Selected transition does not span ten simulator steps")
        if not np.all(fetch(h5, "action_mask", source)):
            raise ValueError("Selected transition has incomplete action sequence")
        arrays = {key: fetch(h5, key, source) for key in (
            "prev_range_values", "range_values", "prev_drone_state", "action_sequence")}
    if not np.isfinite(arrays["action_sequence"]).all() or not np.isfinite(
            arrays["prev_drone_state"]).all():
        raise ValueError("Nonfinite physical command or previous state")
    arrays["previous_latent"] = cached["previous"][positions].copy() * scale
    arrays["normalized_actions"] = cached["actions"][positions].copy()
    arrays["causal_state"] = stage1.causal_state(cached["state"][positions])
    return manifest, rows, arrays, scale


def aggregate(rows, methods):
    result = {}
    groups = {"all": rows}
    groups.update({f"seed_{seed}": [r for r in rows if r["seed"] == seed]
                   for seed in sorted({r["seed"] for r in rows})})
    for label, group in groups.items():
        paired = [r for r in group if all(not r[m]["empty_cloud"] for m in methods)]
        count = sum(r["new_visible"]["diffusion"]["gt_new_hits"] for r in group)
        result[label] = {
            "samples": len(group), "paired_nonempty": len(paired),
            "all_samples": {m: summarize([r[m] for r in group]) for m in methods},
            "paired_cd_paper_m2": {m: float(np.mean([r[m]["cd_paper_m2"] for r in paired]))
                                    if paired else None for m in methods},
            "new_visible_gt_hits": count,
            "new_visible_recall": {m: sum(r["new_visible"][m]["predicted_new_hits"]
                                         for r in group) / max(count, 1) for m in methods},
            "lock": {m: {
                "fraction_of_rays": sum(r["lock"][m]["locked"] for r in group) /
                                    (len(group) * 108 * 18),
                "false_lock_fraction": sum(r["lock"][m]["false_0p3m"] for r in group) /
                                       max(sum(r["lock"][m]["locked"] for r in group), 1),
                "locked": sum(r["lock"][m]["locked"] for r in group),
            } for m in (*CAUSAL_RULES, "oracle_gt_mask")},
        }
    return result


def render_examples(path, examples, selected):
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 5, figsize=(18, 3.5 * len(examples)),
                             squeeze=False)
    labels = ("Input", "GT", "Warp", "Diffusion", selected)
    for row, (rays, images, title, threshold) in enumerate(examples):
        for col, (label, frame) in enumerate(zip(labels, images)):
            points = frame_points(frame, rays, threshold if col >= 3 else 0)
            ax = axes[row, col]
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], s=0.3)
            ax.set(xlim=(-10, 10), ylim=(-10, 10), aspect="equal",
                   title=f"{title}: {label}")
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def evaluate(args, split, model, vae, scheduler, motion, threshold):
    manifest, rows, arrays, scale = load_selected(args, split)
    rays = {seed: load_rays(args.raw_root, split, seed)
            for seed in sorted({row["seed"] for row in rows})}
    warped = []
    for i, row in enumerate(rows):
        state = arrays["prev_drone_state"][i].astype(np.float64)
        command = arrays["action_sequence"][i].astype(np.float64)
        next_pose = next_state(state, command, motion)
        ray_grid, azimuth, elevation = rays[row["seed"]]
        warped.append(warp_frame(arrays["prev_range_values"][i],
                                 transform_from_initial(state, next_pose),
                                 ray_grid, azimuth, elevation))
    warped = np.stack(warped)
    results = [None] * len(rows)
    examples = []
    for seed in sorted(rays):
        locations = np.array([i for i, row in enumerate(rows) if row["seed"] == seed])
        for start in range(0, len(locations), args.batch_size):
            chosen = locations[start:start + args.batch_size]
            previous = torch.from_numpy(arrays["previous_latent"][chosen]).to(stage1.DEVICE)
            actions = torch.from_numpy(arrays["normalized_actions"][chosen]).to(stage1.DEVICE)
            states = torch.from_numpy(arrays["causal_state"][chosen]).to(stage1.DEVICE)
            latent = stage1.generate(model, scheduler, previous, actions, states,
                                     seed=42 + seed * 10000 + start,
                                     init_strength=1.0, num_steps=20)
            generated = vae.decode(latent / scale).sample.cpu().numpy()
            for local, index in enumerate(chosen):
                ray_grid = rays[seed][0]
                target = arrays["range_values"][index]
                warp = warped[index]
                prediction = generated[local]
                masks = masks_from_warp(warp)
                masks["oracle_gt_mask"] = oracle_mask(warp, prediction, target, threshold)
                images = {"copy": arrays["prev_range_values"][index],
                          "warp": warp, "diffusion": prediction}
                images.update({name: compose(warp, prediction, masks[name], threshold)
                               for name in (*CAUSAL_RULES, "oracle_gt_mask")})
                record = {**rows[index], "lock": {}, "new_visible": {}}
                for name, image in images.items():
                    record[name] = lidar_metric(image, target, ray_grid,
                                                threshold if name not in ("copy", "warp") else 0)
                    record["new_visible"][name] = new_visible_scores(
                        image, warp, target, threshold if name not in ("copy", "warp") else 0)
                for name, mask in masks.items():
                    record["lock"][name] = lock_diagnostics(mask, warp, target)
                results[index] = record
                if len(examples) < 4 and (start == 0 or len(examples) < 2):
                    examples.append((ray_grid, (images["copy"], target, warp,
                                                      prediction, images["all_hits"]),
                                     f"seed {seed}, frame {rows[index]['frame_idx']}", threshold))
        print(json.dumps({"split": split, "seed": seed, "evaluated": len(locations)}), flush=True)
    report = {"split": split, "manifest_sha256": manifest_hash(manifest),
              "samples": len(results), "checkpoint": str(args.unet),
              "motion_checkpoint": str(args.motion), "methods": METHODS,
              "oracle_gt_mask": "uses next ground truth to choose the lower hit-aware L1 candidate per ray; not a Chamfer upper bound or deployable mask",
              "causal_masks": "computed solely from action-predicted warp and local support/depth spread",
              "false_lock": "locked ray with GT miss or GT range error greater than 0.3m",
              "cd_paper_m2": "sum of two squared nearest-neighbor direction means; m^2",
              "summary": aggregate(results, METHODS), "rows": results}
    stage1.save_json(args.out / f"{split}.json", report)
    return report, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=stage1.OUT / "hard_geometry_inpaint")
    parser.add_argument("--manifest", type=Path,
                        default=stage1.OUT / "representative_baseline")
    parser.add_argument("--unet", type=Path,
                        default=stage1.OUT / "world_circular_causal_8h" / "best.pt")
    parser.add_argument("--motion", type=Path,
                        default=stage1.OUT / "epona_probe" / "motion_ridge.npz")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    model = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    checkpoint = stage1.load_model(args.unet, model)
    vae = stage1.load_circular_vae()
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    motion = load_motion(args.motion)
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    val, val_examples = evaluate(args, "val", model, vae, scheduler, motion, threshold)
    scores = val["summary"]["all"]["paired_cd_paper_m2"]
    best = min(CAUSAL_RULES, key=lambda name: scores[name])
    stage1.save_json(args.out / "selection.json", {
        "checkpoint_step": checkpoint["step"], "selected_causal_mask": best,
        "selection": "lowest validation paired squared Chamfer among causal masks only",
        "validation_paired_cd_paper_m2": scores})
    render_examples(args.out / "validation_examples.png", val_examples, "all_hits")
    print(json.dumps({"validation_selected_mask": best, "validation": scores}), flush=True)
    test, test_examples = evaluate(args, "test", model, vae, scheduler, motion, threshold)
    render_examples(args.out / "test_examples.png", test_examples, "all_hits")
    print(json.dumps({"test_selected_mask": best,
                      "test": test["summary"]["all"]["paired_cd_paper_m2"]}), flush=True)


if __name__ == "__main__":
    main()
