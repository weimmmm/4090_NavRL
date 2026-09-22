"""Autoregress the direct-t+3 model to 3/6/9/12/15 future frames.

The model consumes three consecutive 10-action chunks per generation.  Its
generated latent is fed back as the previous latent for the next generation.
Only the initial causal ego state is used throughout the rollout.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_wam.runner import stage1
from lidar_wam.runner.world_direct_horizon import (
    DirectHorizonWorldModel,
    frame_metrics,
    generate,
)


FRAME_HORIZONS = (3, 6, 9, 12, 15)
CHUNK_SIZE = 3


def read_rows(h5, key, indices):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(h5[key][indices[order]])[inverse]


def select_scenes(h5, valid_sources, scene_count):
    """Match the old LaGen evaluator's 16-scene selection where possible.

    Unlike the old evaluator, reject gaps and rows absent from the validated
    latent cache.  Every selected scene therefore supplies exactly 15 valid,
    consecutive transitions.
    """
    valid = set(map(int, valid_sources))
    grouped = defaultdict(list)
    scenes = h5["scene_token"].asstr()[:]
    frames = h5["frame_idx"][:]
    seeds = h5["terrain_seed"][:]
    for row, (scene, frame) in enumerate(zip(scenes, frames)):
        if row in valid:
            grouped[scene].append((int(frame), row))

    candidates = []
    for scene, entries in grouped.items():
        entries.sort()
        for start in range(max(0, len(entries) - FRAME_HORIZONS[-1] + 1)):
            window = entries[start:start + FRAME_HORIZONS[-1]]
            if len(window) != FRAME_HORIZONS[-1]:
                continue
            first_frame = window[0][0]
            if all(frame == first_frame + offset
                   for offset, (frame, _) in enumerate(window)):
                candidates.append((scene, [row for _, row in window]))
                break
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < scene_count:
        raise RuntimeError(
            f"Only {len(candidates)} valid 15-frame scenes; requested {scene_count}")
    positions = np.linspace(0, len(candidates) - 1, scene_count, dtype=np.int64)
    selected = [candidates[int(position)] for position in positions]
    return selected, {
        "candidate_scenes": len(candidates),
        "selected_scenes": len(selected),
        "scene_tokens": [scene for scene, _ in selected],
        "seeds": [int(seeds[rows[0]]) for _, rows in selected],
        "trajectory_indices": [list(map(int, rows)) for _, rows in selected],
    }


def summarize(rows):
    values = np.asarray([row["cd_released_code_m2"] for row in rows])
    nonempty = np.asarray([
        row["cd_released_code_m2"] for row in rows
        if not row["pred_empty"] and not row["gt_empty"]
    ])
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    return {
        "samples": len(rows),
        "cd_released_code_m2": float(values.mean()),
        "median_cd_released_code_m2": float(np.median(values)),
        "p95_cd_released_code_m2": float(np.percentile(values, 95)),
        "max_cd_released_code_m2": float(values.max()),
        "common_nonempty_samples": int(len(nonempty)),
        "common_nonempty_cd_released_code_m2": (
            float(nonempty.mean()) if len(nonempty) else None),
        "mask_precision": tp / max(tp + fp, 1),
        "mask_recall": tp / max(tp + fn, 1),
        "mask_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_empty_frames": sum(row["false_empty_frame"] for row in rows),
        "false_hit_frames": sum(row["false_hit_frame"] for row in rows),
        "mean_pred_hit_count": float(np.mean([
            row["pred_hit_count"] for row in rows])),
    }


@torch.no_grad()
def evaluate(args):
    active_horizons = tuple(h for h in FRAME_HORIZONS if h <= args.max_horizon)
    if not active_horizons or active_horizons[-1] != args.max_horizon:
        raise ValueError(f"max-horizon must be one of {FRAME_HORIZONS}")
    stage1.DATA = args.data_root
    cache = np.load(args.latent_root / "test.npz")
    source = np.asarray(cache["source_index"], dtype=np.int64)
    source_position = {int(row): position for position, row in enumerate(source)}
    metadata = json.loads((args.latent_root / "metadata.json").read_text())
    scale = float(metadata["scaling_factor"])

    with h5py.File(args.data_root / "navrl_static_test.h5", "r") as h5:
        selected, manifest = select_scenes(h5, source, args.scenes)
        trajectory = np.asarray([rows for _, rows in selected], dtype=np.int64)
        target_rows = trajectory[:, np.asarray(active_horizons) - 1]
        target_images = np.stack([
            read_rows(h5, "range_values", rows) for rows in target_rows.T
        ], axis=1).astype(np.float32)

    positions = np.asarray([
        [source_position[int(row)] for row in rows] for rows in trajectory
    ], dtype=np.int64)
    previous = torch.from_numpy(
        np.asarray(cache["previous"][positions[:, 0]]) * scale).float()
    actions = torch.from_numpy(np.asarray(cache["actions"][positions])).float()
    state = torch.from_numpy(stage1.causal_state(
        np.asarray(cache["state"][positions[:, 0]]))).float()
    if tuple(actions.shape[1:]) != (15, 10, 3):
        raise ValueError(f"Unexpected action shape: {tuple(actions.shape)}")

    model = DirectHorizonWorldModel().to(stage1.DEVICE).float().eval()
    checkpoint = stage1.load_model(args.checkpoint, model)
    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    scheduler = stage1.DDIMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon", clip_sample=False)

    latent = previous.to(stage1.DEVICE)
    state = state.to(stage1.DEVICE)
    rows_by_horizon = {}
    for chunk in range(len(active_horizons)):
        chunk_actions = actions[:, chunk * CHUNK_SIZE:(chunk + 1) * CHUNK_SIZE]
        latent = generate(
            model, scheduler, latent, chunk_actions.to(stage1.DEVICE), state,
            args.seed + chunk * 100000, args.ddim_steps)
        decoded = vae.decode(latent / scale).sample.cpu().numpy()
        horizon = active_horizons[chunk]
        horizon_rows = []
        for index, (prediction, target) in enumerate(
                zip(decoded, target_images[:, chunk])):
            row = frame_metrics(prediction, target, args.mask_threshold)
            # world_direct_horizon uses the two directional sum.  LaGen's
            # released inference code reports their average.
            row["cd_released_code_m2"] = row["cd_paper_m2"] / 2.0
            row.update({
                "scene_token": selected[index][0],
                "seed": manifest["seeds"][index],
                "horizon": horizon,
                "time_s": horizon * 0.16,
                "target_source_index": int(target_rows[index, chunk]),
            })
            horizon_rows.append(row)
        rows_by_horizon[str(horizon)] = horizon_rows
        print(json.dumps({
            "horizon": horizon,
            "summary": summarize(horizon_rows),
        }), flush=True)

    result = {
        "version": 1,
        "mode": "direct_t3_model_autoregressive_in_3-frame_chunks",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "ddim_steps_per_chunk": args.ddim_steps,
        "seed": args.seed,
        "mask_threshold": args.mask_threshold,
        "state": "initial causal state fixed throughout rollout",
        "actions": "15 recorded action chunks, grouped as five [3,10,3] conditions",
        "chamfer_definition": (
            "0.5 * (mean squared GT-to-Pred NN + mean squared Pred-to-GT NN)"),
        "empty_policy": "an empty side receives 100 m^2 in released-code CD units",
        "manifest": manifest,
        "summary": {horizon: summarize(rows)
                    for horizon, rows in rows_by_horizon.items()},
        "rows": rows_by_horizon,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage1.save_json(args.output, result)
    print("DIRECT_T3_AUTOREGRESSIVE_RESULT=" + json.dumps({
        "checkpoint_step": result["checkpoint_step"],
        "summary": result["summary"],
        "output": str(args.output),
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--max-horizon", type=int, default=15)
    parser.add_argument("--mask-threshold", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.latent_root = (args.latent_root or
                        stage1.OUT / stage1.LATENT_DIR).expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    evaluate(args)


if __name__ == "__main__":
    main()
