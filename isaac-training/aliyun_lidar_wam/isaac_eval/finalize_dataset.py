"""Validate completed seeds and write dataset/split manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from isaac_eval.dataset_io import validate_dataset
from isaac_eval.environment_file import file_sha256, load_environment


ROOT = Path(__file__).resolve().parents[1]


def split_for_seed(seed: int) -> str:
    return "train" if seed <= 7 else ("val" if seed == 8 else "test")


def parse_seed_list(value: str):
    result = []
    for part in value.split(","):
        if "-" in part:
            begin, end = map(int, part.split("-", 1))
            result.extend(range(begin, end + 1))
        else:
            result.append(int(part))
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("seed list contains duplicates")
    return result


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "datasets" / "wam_static_seed00_09")
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("0-9"))
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--sample-interval", type=int, default=10)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    entries, totals = [], Counter()
    total_frames = total_bytes = 0
    weighted_steps = 0.0
    for seed in args.seeds:
        split = split_for_seed(seed)
        env_path = root / "environments" / f"static350_seed{seed:02d}_n{args.num_envs}.pt"
        dataset_path = root / split / f"seed_{seed:04d}" / "trajectories.h5"
        summary_path = root / split / f"seed_{seed:04d}" / "summary.json"
        for path in (env_path, dataset_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        environment = load_environment(env_path)
        validation = validate_dataset(
            dataset_path, environment, args.num_envs, args.sample_interval)
        totals.update(validation["termination_counts"])
        total_frames += validation["frames"]
        total_bytes += validation["size_bytes"] + env_path.stat().st_size
        weighted_steps += validation["mean_episode_steps"] * validation["scenes"]
        entries.append({
            **validation, "seed": seed, "split": split,
            "environment": str(env_path.relative_to(root)),
            "environment_sha256": file_sha256(env_path),
            "terrain_mesh_sha256": environment["terrain_mesh"]["sha256"],
            "dataset": str(dataset_path.relative_to(root)),
            "dataset_sha256": file_sha256(dataset_path),
        })
    manifest = {
        "format": "navrl-isaac-dataset-manifest-v2",
        "seeds": list(args.seeds), "static_obstacles": 350,
        "dynamic_obstacles": 0, "num_envs_per_seed": args.num_envs,
        "sample_interval": args.sample_interval, "total_scenes": len(entries) * args.num_envs,
        "total_frames": total_frames, "termination_counts": dict(totals),
        "mean_episode_steps": weighted_steps / max(len(entries) * args.num_envs, 1),
        "total_size_bytes": total_bytes, "entries": entries,
    }
    _write(root / "manifest.json", manifest)
    for split in ("train", "val", "test"):
        selected = [entry for entry in entries if entry["split"] == split]
        _write(root / split / "manifest.json", {
            "format": "navrl-isaac-split-manifest-v2", "split": split,
            "seeds": [entry["seed"] for entry in selected], "entries": selected,
            "scenes": sum(entry["scenes"] for entry in selected),
            "frames": sum(entry["frames"] for entry in selected),
        })
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
