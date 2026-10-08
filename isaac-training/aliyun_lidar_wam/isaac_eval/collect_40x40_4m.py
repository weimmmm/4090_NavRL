"""Schedule the 4M-frame 40x40 settled-goal PPO collection protocol.

The two conditions are balanced by accepted frame count:

* 200 static obstacles: 2,000,000 frames
* 350 static obstacles: 2,000,000 frames

Each Isaac process creates a fresh obstacle layout and fresh routes.  Failed
collision/out-of-bounds trajectories are discarded by ``collect_dataset``;
only retained frames count toward the targets.  This scheduler is resumable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (200, 350)


def _parse_gpus(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("GPU list must contain unique IDs")
    return result


def _summary_path(root: Path, obstacles: int, shard: int) -> Path:
    return root / "train" / f"obs_{obstacles:04d}" / f"shard_{shard:05d}" / "summary.json"


def _read_frames(path: Path) -> int:
    payload = json.loads(path.read_text())
    return int(payload["summary"]["frames"])


def _scan(root: Path, obstacles: int) -> tuple[int, set[int]]:
    total, shards = 0, set()
    directory = root / "train" / f"obs_{obstacles:04d}"
    for shard_dir in sorted(directory.glob("shard_*")):
        shard = int(shard_dir.name.split("_")[-1])
        shards.add(shard)
        path = shard_dir / "summary.json"
        if path.is_file():
            total += _read_frames(path)
    return total, shards


def _run_shard(*, root: Path, checkpoint: Path, gpu: int, obstacles: int,
               shard: int, num_envs: int, keep_environment: bool) -> tuple[int, Path]:
    condition_offset = 0 if obstacles == 200 else 100_000
    terrain_seed = 4_000_000 + condition_offset + shard * 100
    route_seed = 6_000_000 + condition_offset + shard * 100
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"obs{obstacles}_shard{shard:05d}_gpu{gpu}.log"
    command = [
        sys.executable, "-m", "isaac_eval.collect_dataset",
        "--seed", str(terrain_seed), "--route-seed", str(route_seed),
        "--shard-id", str(shard), "--static-obstacles", str(obstacles),
        "--num-envs", str(num_envs), "--max-steps", "2500",
        "--sample-interval", "10", "--goal-radius", "0.5",
        "--settle-speed", "0.1", "--settle-steps", "10",
        "--opposite-fraction", "0.5",
        "--split", "train", "--device", f"cuda:{gpu}",
        "--output-root", str(root), "--checkpoint", str(checkpoint),
    ]
    if keep_environment:
        command.append("--keep-environment")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(
            f"obs={obstacles} shard={shard} failed on GPU {gpu}; see {log_path}")
    frames = _read_frames(_summary_path(root, obstacles, shard))
    return frames, log_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=_parse_gpus, default=_parse_gpus("0,1,2,3"))
    parser.add_argument("--target-per-condition", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--expected-frames-per-shard", type=int, default=50_000,
                        help="Scheduling estimate only; completion uses actual accepted frames.")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "datasets" / "wam_40x40_obs200_350_settled_4m")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "isaac_eval" / "checkpoints" /
                        "ppo_dataset_collector.pt")
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if args.target_per_condition <= 0 or args.num_envs <= 0:
        parser.error("frame target and num-envs must be positive")
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")

    totals, used, next_shard = {}, {}, {}
    for obstacles in CONDITIONS:
        totals[obstacles], used[obstacles] = _scan(root, obstacles)
        next_shard[obstacles] = max(used[obstacles], default=-1) + 1
        print(f"[resume] obs={obstacles} accepted_frames={totals[obstacles]:,} "
              f"target={args.target_per_condition:,}", flush=True)

    available_gpus = list(args.gpus)
    pending: dict[concurrent.futures.Future, tuple[int, int, int, bool]] = {}
    reserved = {condition: 0 for condition in CONDITIONS}
    keep_reserved = {condition: False for condition in CONDITIONS}
    workers = len(available_gpus)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or any(totals[c] < args.target_per_condition for c in CONDITIONS):
            while available_gpus:
                candidates = [
                    c for c in CONDITIONS
                    if totals[c] + reserved[c] < args.target_per_condition
                ]
                if not candidates:
                    break
                condition = min(
                    candidates,
                    key=lambda c: (totals[c] + reserved[c]) / args.target_per_condition)
                gpu = available_gpus.pop(0)
                shard = next_shard[condition]
                next_shard[condition] += 1
                representative = root / "environments" / f"map40_obs{condition:04d}.pt"
                keep = not representative.exists() and not keep_reserved[condition]
                keep_reserved[condition] |= keep
                future = pool.submit(
                    _run_shard, root=root, checkpoint=checkpoint, gpu=gpu,
                    obstacles=condition, shard=shard, num_envs=args.num_envs,
                    keep_environment=keep)
                pending[future] = (condition, shard, gpu, keep)
                reserved[condition] += args.expected_frames_per_shard
                print(f"[launch] obs={condition} shard={shard} gpu={gpu} "
                      f"representative_map={keep}", flush=True)
            if not pending:
                break
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                condition, shard, gpu, kept = pending.pop(future)
                available_gpus.append(gpu)
                reserved[condition] -= args.expected_frames_per_shard
                frames, log_path = future.result()
                totals[condition] += frames
                print(f"[done] obs={condition} shard={shard} gpu={gpu} "
                      f"frames={frames:,} total={totals[condition]:,}/"
                      f"{args.target_per_condition:,} log={log_path}", flush=True)

    status = {
        "format": "navrl-40x40-settled-4m-v1",
        "map_size_m": [40.0, 40.0],
        "target_total_frames": 2 * args.target_per_condition,
        "conditions": {
            str(c): {"target_frames": args.target_per_condition,
                     "accepted_frames": totals[c]} for c in CONDITIONS},
        "max_steps": 2500, "sample_interval": 10,
        "goal_radius_m": 0.5, "settle_speed_mps": 0.1,
        "settle_steps": 10, "num_envs_per_shard": args.num_envs,
        "opposite_route_fraction": 0.5,
        "discarded_terminations": ["collision", "out_of_bounds"],
        "test_split": None,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "collection_status.json").write_text(
        json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
