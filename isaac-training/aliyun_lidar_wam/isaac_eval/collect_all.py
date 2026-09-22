"""Run 16/128/512 smoke tests or schedule production seeds across GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_ints(value: str):
    values = []
    for part in value.split(","):
        if "-" in part:
            begin, end = map(int, part.split("-", 1))
            values.extend(range(begin, end + 1))
        else:
            values.append(int(part))
    return values


def run_one(seed: int, gpu: int, output_root: Path, num_envs: int, split: str):
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed{seed:02d}_n{num_envs}_gpu{gpu}.log"
    command = [
        sys.executable, "-m", "isaac_eval.collect_dataset", "--seed", str(seed),
        "--num-envs", str(num_envs), "--device", f"cuda:{gpu}",
        "--split", split, "--output-root", str(output_root),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"seed {seed} failed on GPU {gpu}; see {log_path}")
    return str(log_path)


def run_capacity(seed: int, gpu: int, output_root: Path, num_envs: int):
    log_dir = output_root / "smoke_tests"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"capacity_n{num_envs:04d}_gpu{gpu}.log"
    command = [sys.executable, "-m", "isaac_eval.capacity_smoke",
               "--seed", str(seed), "--num-envs", str(num_envs),
               "--device", f"cuda:{gpu}"]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"capacity test n={num_envs} failed; see {log_path}")
    return str(log_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "collect"), required=True)
    parser.add_argument("--gpus", type=parse_ints, default=parse_ints("0,1,2,3,4,5,6,7"))
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("0-9"))
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "datasets" / "wam_static_seed00_09")
    args = parser.parse_args()
    if not args.gpus:
        parser.error("at least one GPU is required")
    output = args.output_root.expanduser().resolve()
    if args.mode == "smoke":
        for size in (16, 128, 512):
            print(run_capacity(args.seeds[0], args.gpus[0], output, size), flush=True)
        return
    if args.seeds != list(range(10)):
        print(f"[collect_all] non-standard seed selection: {args.seeds}", flush=True)
    workers = args.max_parallel or len(args.gpus)
    workers = max(1, min(workers, len(args.gpus), len(args.seeds)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        seed_iter = iter(args.seeds)
        pending = {}
        # One process per GPU.  A completed GPU immediately receives seed 8/9.
        for gpu in args.gpus[:workers]:
            seed = next(seed_iter, None)
            if seed is None:
                break
            split = "train" if seed <= 7 else ("val" if seed == 8 else "test")
            future = pool.submit(run_one, seed, gpu, output, 512, split)
            pending[future] = (seed, gpu)
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                seed, gpu = pending.pop(future)
                print(f"[collect_all] seed={seed} gpu={gpu} log={future.result()}", flush=True)
                following = next(seed_iter, None)
                if following is not None:
                    split = "train" if following <= 7 else ("val" if following == 8 else "test")
                    next_future = pool.submit(run_one, following, gpu, output, 512, split)
                    pending[next_future] = (following, gpu)
    subprocess.run([
        sys.executable, "-m", "isaac_eval.finalize_dataset",
        "--output-root", str(output)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
