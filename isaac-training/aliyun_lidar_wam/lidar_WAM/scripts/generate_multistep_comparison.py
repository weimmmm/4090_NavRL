"""Generate test-set GT/VAE/autoregressive Future Model comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lidar_wam.coordinates import numpy_goal_frame_causal_features
from lidar_wam.data_v2 import split_entries
from lidar_wam.runner import stage1
from lidar_wam.runner.world_direct_horizon import (
    DirectHorizonWorldModel,
    frame_metrics,
    generate,
)


HORIZONS = (0, 3, 6, 9, 12, 15, 18)
PREDICTION_HORIZONS = HORIZONS[1:]


def _decode(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def _frames(handle):
    return handle["frames"] if "frames" in handle else handle


def _valid_chains(frames):
    tokens = [_decode(value) for value in frames["token"][:]]
    following = [_decode(value) for value in frames["next_token"][:]]
    scenes = [_decode(value) for value in frames["scene_token"][:]]
    frame_index = np.asarray(frames["frame_index"][:], np.int64)
    step_delta = np.asarray(frames["step_delta"][:], np.int16)
    action_mask = np.asarray(frames["action_mask"][:], bool)
    collision = np.asarray(frames["collision"][:], bool)
    out_of_bounds = np.asarray(frames["out_of_bounds"][:], bool)
    token_to_row = {token: row for row, token in enumerate(tokens)}
    chains = []
    for start in range(len(tokens)):
        chain = [start]
        for _ in range(HORIZONS[-1]):
            next_token = following[chain[-1]]
            if not next_token or next_token not in token_to_row:
                break
            chain.append(token_to_row[next_token])
        if len(chain) != HORIZONS[-1] + 1:
            continue
        if any(scenes[row] != scenes[start] for row in chain):
            continue
        if any(frame_index[row] != frame_index[start] + offset
               for offset, row in enumerate(chain)):
            continue
        future = np.asarray(chain[1:], np.int64)
        if np.any(step_delta[future] != 10):
            continue
        if not bool(action_mask[future].all()):
            continue
        if bool(collision[future].any() or out_of_bounds[future].any()):
            continue
        chains.append(np.asarray(chain, np.int64))
    return chains


def _select_diverse_chains(frames, chains, count, seed):
    by_scene = {}
    scene_ids = np.asarray(frames["scene_id"][:], np.int32)
    for chain in chains:
        by_scene.setdefault(int(scene_ids[chain[0]]), []).append(chain)
    if len(by_scene) < count:
        raise RuntimeError(f"Only {len(by_scene)} scenes have valid 18-frame chains")
    rng = np.random.default_rng(seed)
    selected_scenes = np.sort(rng.choice(sorted(by_scene), count, replace=False))
    selected = []
    for scene in selected_scenes:
        candidates = by_scene[int(scene)]
        # Choose a deterministic random point inside the trajectory instead of
        # showing only episode starts.
        selected.append(candidates[int(rng.integers(0, len(candidates)))])
    return selected


def _world_condition(frames, rows):
    state = np.stack([np.asarray(frames["drone_state"][row], np.float32)
                      for row in rows])
    target = np.stack([np.asarray(frames["target_position"][row], np.float32)
                       for row in rows])
    direction = np.stack([np.asarray(frames["target_dir_2d"][row], np.float32)
                          for row in rows])
    _, proprio = numpy_goal_frame_causal_features(state, target, direction)
    zeros = np.zeros(len(rows), np.float32)
    return np.stack((proprio[:, 1], proprio[:, 2], zeros, zeros,
                     proprio[:, 6]), axis=1).astype(np.float32)


def _range_and_points(frame, threshold):
    valid = frame[1, :, :18] > threshold
    ranges = np.clip((frame[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
    return valid, ranges, stage1.to_points(frame, threshold)


def _render_range(axis, frame, threshold, title):
    valid, ranges, _ = _range_and_points(frame, threshold)
    axis.imshow(np.ma.masked_where(~valid, ranges).T, origin="lower",
                vmin=0, vmax=10, cmap="turbo", aspect="auto")
    axis.set_facecolor("black")
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def _render_xy(axis, frame, threshold):
    _, _, points = _range_and_points(frame, threshold)
    if len(points):
        axis.scatter(points[:, 0], points[:, 1], s=2,
                     c=np.linalg.norm(points, axis=1), cmap="turbo",
                     vmin=0, vmax=10, rasterized=True)
    axis.set_xlim(-10, 10)
    axis.set_ylim(-10, 10)
    axis.set_aspect("equal")
    axis.grid(alpha=0.15)
    axis.set_xticks((-10, 0, 10))
    axis.set_yticks((-10, 0, 10))


def _save_figure(path, gt, predicted, scene_id, start_frame, checkpoint_step):
    fig, axes = plt.subplots(4, len(HORIZONS), figsize=(22, 12),
                             constrained_layout=True)
    for column, horizon in enumerate(HORIZONS):
        _render_range(axes[0, column], gt[column], 0.0, f"GT t+{horizon}")
        _render_xy(axes[1, column], gt[column], 0.0)
        name = "VAE t+0" if horizon == 0 else f"Future t+{horizon}"
        _render_range(axes[2, column], predicted[column], 1.5, name)
        _render_xy(axes[3, column], predicted[column], 1.5)
    for row, label in enumerate(("GT range", "GT point cloud XY",
                                 "VAE/Future range", "VAE/Future point cloud XY")):
        axes[row, 0].set_ylabel(label, fontsize=11)
    fig.suptitle(
        f"Test seed 9 | scene {scene_id} | start frame {start_frame} | "
        f"checkpoint step {checkpoint_step} | horizons in saved LiDAR frames",
        fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim-steps", type=int, default=20)
    args = parser.parse_args()

    entries = split_entries(args.dataset_root, "test")
    if len(entries) != 1 or int(entries[0]["seed"]) != 9:
        raise ValueError("Expected the fixed test split to contain only seed 9")
    h5_path = args.dataset_root / entries[0]["dataset"]
    latent_path = args.latent_root / "seed_0009.npy"
    metadata = json.loads((args.latent_root / "metadata.json").read_text())
    scale = float(metadata["scaling_factor"])

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_step = int(payload.get("step", -1))
    model = DirectHorizonWorldModel().to(stage1.DEVICE).float().eval()
    model.load_state_dict(payload["model"], strict=True)
    del payload
    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    scheduler = stage1.DDIMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon", clip_sample=False)

    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as handle:
        frames = _frames(handle)
        chains = _select_diverse_chains(
            frames, _valid_chains(frames), args.samples, args.seed)
        latent = np.load(latent_path, mmap_mode="r")
        starts = np.asarray([chain[0] for chain in chains], np.int64)
        current = torch.from_numpy(
            np.stack([np.asarray(latent[row], np.float32) for row in starts])
        ).to(stage1.DEVICE)
        reconstructed_zero = vae.decode(current / scale).sample.cpu().numpy()
        predictions = [reconstructed_zero]

        for rollout_index, horizon in enumerate(PREDICTION_HORIZONS, start=1):
            action_rows = [chain[horizon - 2:horizon + 1] for chain in chains]
            actions = np.stack([
                np.stack([np.asarray(frames["normalized_action_sequence"][int(row)],
                                     np.float32) for row in rows])
                for rows in action_rows
            ])
            condition_rows = [int(chain[horizon - 3]) for chain in chains]
            state = _world_condition(frames, condition_rows)
            current = generate(
                model, scheduler, current,
                torch.from_numpy(actions).to(stage1.DEVICE),
                torch.from_numpy(state).to(stage1.DEVICE),
                args.seed + rollout_index * 100000, args.ddim_steps)
            predictions.append(vae.decode(current / scale).sample.cpu().numpy())

        predictions = np.stack(predictions, axis=1)
        gt = np.stack([
            np.stack([np.asarray(frames["range_values"][int(chain[horizon])],
                                 np.float32) for horizon in HORIZONS])
            for chain in chains
        ])
        scene_ids = np.asarray(frames["scene_id"][:], np.int32)
        frame_indices = np.asarray(frames["frame_index"][:], np.int64)

        report = {
            "format": "navrl-future-multistep-comparison-v1",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": checkpoint_step,
            "split": "test", "terrain_seed": 9,
            "horizons": list(HORIZONS),
            "horizon_seconds": [float(value * 0.16) for value in HORIZONS],
            "ddim_steps": args.ddim_steps,
            "rollout": "autoregressive latent; GT actions and causal boundary state",
            "samples": [],
        }
        for sample_index, chain in enumerate(chains):
            scene_id = int(scene_ids[chain[0]])
            start_frame = int(frame_indices[chain[0]])
            filename = f"comparison_{sample_index:02d}_scene{scene_id:04d}_frame{start_frame:04d}.png"
            _save_figure(args.output / filename, gt[sample_index],
                         predictions[sample_index], scene_id, start_frame,
                         checkpoint_step)
            np.savez_compressed(
                args.output / filename.replace(".png", ".npz"),
                gt=gt[sample_index], prediction=predictions[sample_index],
                rows=chain[np.asarray(HORIZONS)], horizons=np.asarray(HORIZONS))
            metrics = {}
            for column, horizon in enumerate(HORIZONS):
                metrics[str(horizon)] = frame_metrics(
                    predictions[sample_index, column], gt[sample_index, column],
                    mask_threshold=1.5)
            report["samples"].append({
                "sample": sample_index, "scene_id": scene_id,
                "start_frame": start_frame, "image": filename,
                "rows": [int(chain[horizon]) for horizon in HORIZONS],
                "metrics": metrics,
            })
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "checkpoint_step": checkpoint_step,
        "images": [row["image"] for row in report["samples"]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
