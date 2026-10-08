"""Prepare the settled-goal 40x40 shards for Action Expert training.

The collector writes one HDF5 file per obstacle condition/shard and a
``summary.json`` beside it.  This utility publishes the v2 manifest expected
by :mod:`lidar_wam.data_v2`, then creates a whole-trajectory validation split
without copying the 17 GB of HDF5 data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from lidar_wam.data_v2 import (
    _trajectory_inventory,
    build_split_index,
    validation_trajectory_keys,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def build_manifest(root: Path):
    summaries = sorted(root.glob("train/obs_*/shard_*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no shard summaries under {root / 'train'}")

    entries = []
    condition_totals = {}
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())["summary"]
        # Summary files were written inside the container and therefore carry
        # a ``/workspace/NavRL`` absolute path.  Resolve the actual host path
        # from the summary's directory instead of trusting that prefix.
        dataset_path = summary_path.parent / "trajectories.h5"
        dataset = dataset_path.relative_to(root)
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        obstacles = int(summary["static_obstacles"])
        label = f"map40_obs{obstacles:04d}"
        shard_id = int(summary["shard_id"])
        condition_totals.setdefault(label, 0)
        condition_totals[label] += int(summary["frames"])
        entries.append({
            "condition": label,
            "condition_index": 0 if obstacles == 200 else 1,
            "dataset": str(dataset),
            "dataset_sha256": str(summary["dataset_sha256"]),
            "frames": int(summary["frames"]),
            "latent_cache": f"obs{obstacles:04d}_shard_{shard_id:05d}.npy",
            "map_size_m": 40.0,
            "max_steps": int(summary["max_steps"]),
            "route_seed": int(summary["route_seed"]),
            "scenes": int(summary["scenes"]),
            "seed": int(summary["terrain_seed"]),
            "split": "train",
            "static_obstacles": obstacles,
        })

    entries.sort(key=lambda row: (row["condition_index"], row["seed"],
                                  row["route_seed"]))
    if len({row["dataset"] for row in entries}) != len(entries):
        raise ValueError("duplicate dataset path in shard summaries")
    if len({row["dataset_sha256"] for row in entries}) != len(entries):
        raise ValueError("duplicate HDF5 SHA256 in shard summaries")
    if len({(row["seed"], row["route_seed"]) for row in entries}) != len(entries):
        raise ValueError("duplicate terrain/route seed pair in shard summaries")

    conditions = []
    for index, obstacles in enumerate((200, 350)):
        label = f"map40_obs{obstacles:04d}"
        rows = [row for row in entries if row["static_obstacles"] == obstacles]
        conditions.append({
            "accepted_frames": sum(row["frames"] for row in rows),
            "condition_index": index,
            "label": label,
            "map_size_m": 40.0,
            "max_steps": 2500,
            "representative_environment":
                f"environments/map40_obs{obstacles:04d}.pt",
            "static_obstacles": obstacles,
            "target_frames": 2_000_000,
        })
    manifest = {
        "format": "navrl-isaac-dataset-manifest-v2",
        "dataset_name": "wam_40x40_obs200_350_settled_4m",
        "collection_protocol": "ppo_full_horizon_final_goal_hold",
        "conditions": conditions,
        "entries": entries,
        "target_frames": 4_000_000,
        "total_frames": sum(row["frames"] for row in entries),
        "total_scenes": sum(row["scenes"] for row in entries),
        "num_envs_per_shard": 512,
        "opposite_routes_per_shard": 256,
        "sample_interval": 10,
        "full_horizon": True,
        "success_only": True,
        "success_definition": "final_goal_hold",
        "goal_radius_m": 0.5,
        "success_max_speed_mps": 0.1,
        "collision_episodes_kept": False,
        "world_seed_base": min(row["seed"] for row in entries),
        "route_seed_base": min(row["route_seed"] for row in entries),
        "reset_seed_increment": 100,
        "split_fractions": {"train": 1.0 - 0.01, "val": 0.01, "test": 0.0},
        "split_frames": {"train": 0, "val": 0, "test": 0},
        "logical_validation": {
            "fraction": 0.01,
            "random_seed": 42,
            "stratification": "per_shard",
            "unit": "whole_trajectory",
        },
        "expected_windows": {},
    }
    _write_json(root / "manifest.json", manifest)
    return entries, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    index_root = args.index_root.expanduser().resolve()
    entries, manifest = build_manifest(root)
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0,1)")
    inventory = _trajectory_inventory(root, entries)
    held_out = validation_trajectory_keys(
        inventory, fraction=args.val_fraction, random_seed=args.seed)
    protocol = {
        "name": "settled-40x40-trajectory-holdout-v1",
        "validation_fraction": float(args.val_fraction),
        "selection_random_seed": int(args.seed),
        "validation_unit": "whole scene_token trajectory",
        "trajectory_count": sum(len(set(value)) for value in inventory.values()),
        "validation_trajectory_count": len(held_out),
    }
    train_meta = build_split_index(
        root, "train", index_root, strict_expected=False,
        entries_override=entries, exclude_trajectories=held_out,
        protocol={**protocol, "role": "train"})
    val_meta = build_split_index(
        root, "val", index_root, strict_expected=False,
        entries_override=entries, include_trajectories=held_out,
        protocol={**protocol, "role": "validation"})
    # Do not mutate the manifest after writing the indices: the index embeds
    # its manifest SHA and the latent cache uses the same identity.  Window
    # counts are printed below and remain discoverable in the NPZ metadata.
    print(json.dumps({
        "dataset_root": str(root),
        "manifest_sha256": _sha256(root / "manifest.json"),
        "shards": len(entries),
        "frames": manifest["total_frames"],
        "train_windows": train_meta["windows"],
        "val_windows": val_meta["windows"],
        "validation_trajectories": len(held_out),
        "index_root": str(index_root),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
