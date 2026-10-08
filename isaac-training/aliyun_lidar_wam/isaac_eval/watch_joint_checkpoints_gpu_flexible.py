"""Evaluate immutable joint checkpoints on two Isaac-Sim GPUs.

The trainer writes ``closed_loop_queue/step_XXXXXX.pt`` atomically.  This
watcher waits until both requested GPUs are idle, evaluates half of a fixed
128-route environment on each GPU, and writes one compact aggregate summary.
Queued checkpoints are retained for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def gpu_memory_used() -> dict[int, int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    result = {}
    for line in output.splitlines():
        index, used = line.split(",", 1)
        result[int(index.strip())] = int(used.strip())
    return result


def container_path(path: Path, host_root: Path, container_root: Path) -> Path:
    return container_root / path.resolve().relative_to(host_root.resolve())


def evaluation_command(args, checkpoint: Path, output: Path, gpu: int,
                       route_start: int, route_limit: int) -> list[str]:
    checkpoint_in = container_path(
        checkpoint, args.host_root, args.container_root)
    output_in = container_path(output, args.host_root, args.container_root)
    environment_in = container_path(
        args.environment, args.host_root, args.container_root)
    command = (
        f"cd {args.container_root / 'isaac-training/aliyun_lidar_wam'} && "
        f"/isaac-sim/python.sh isaac_eval/evaluate.py "
        f"--checkpoint {checkpoint_in} --output {output_in} "
        f"--environment {environment_in} --device cuda:0 "
        f"--route-start {route_start} --route-limit {route_limit} "
        f"--flow-steps {args.flow_steps}"
    )
    return [
        "docker", "exec", "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
        args.container, "bash", "-lc", command,
    ]


def result_path(output: Path, step: int, route_start: int,
                route_limit: int) -> Path:
    # The terrain seed belongs to the serialized environment (for settled4m
    # it is 6100000), so do not hard-code the seed used by an older snapshot.
    matches = sorted(output.glob(
        f"action_expert_step_{step:06d}_seed*_n{route_limit}"
        f"_routes{route_start:03d}_policy42.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one evaluator result for step={step}, "
            f"route_start={route_start}, route_limit={route_limit}; "
            f"found {matches}")
    return matches[0]


def aggregate(step: int, paths: list[Path], destination: Path):
    payloads = [json.loads(path.read_text()) for path in paths]
    episodes = [row for payload in payloads
                for row in payload["episode_results"]]
    reasons = [row["termination_reason"] for row in episodes]
    count = max(len(episodes), 1)
    summary = {
        "checkpoint_step": int(step),
        "episodes": len(episodes),
        "reach_goal": reasons.count("reach_goal"),
        "collision": reasons.count("collision"),
        "timeout": reasons.count("timeout"),
        "out_of_bounds": reasons.count("out_of_bounds"),
        "success_rate": reasons.count("reach_goal") / count,
        "collision_rate": reasons.count("collision") / count,
        "timeout_rate": reasons.count("timeout") / count,
        "out_of_bounds_rate": reasons.count("out_of_bounds") / count,
        "source_results": [str(path) for path in paths],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(summary), flush=True)


def evaluate(args, checkpoint: Path, step: int):
    step_output = args.output / f"step_{step:06d}"
    step_output.mkdir(parents=True, exist_ok=True)
    aggregate_path = step_output / "summary_128.json"
    if aggregate_path.exists():
        return
    if len(args.gpus) == 1:
        assignments = ((args.gpus[0], 0, args.routes),)
    else:
        half = args.routes // 2
        assignments = (
            (args.gpus[0], 0, half),
            (args.gpus[1], half, args.routes-half),
        )
    jobs = []
    logs = []
    for gpu, start, count in assignments:
        log_path = step_output / f"gpu{gpu}_routes{start:03d}.log"
        stream = log_path.open("a")
        logs.append(stream)
        jobs.append(subprocess.Popen(
            evaluation_command(args, checkpoint, step_output, gpu, start,
                               count), stdout=stream,
            stderr=subprocess.STDOUT, text=True))
    codes = [job.wait() for job in jobs]
    for stream in logs:
        stream.close()
    if any(code != 0 for code in codes):
        raise RuntimeError(
            f"Isaac evaluation failed at step {step}: exit codes {codes}")
    paths = [result_path(step_output, step, start, count)
             for _, start, count in assignments]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing evaluation results: {missing}")
    aggregate(step, paths, aggregate_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--host-root", type=Path,
                        default=Path("/home/yimingwei/NavRL"))
    parser.add_argument("--container-root", type=Path,
                        default=Path("/workspace/NavRL"))
    parser.add_argument("--container", default="navrl-train")
    parser.add_argument("--gpus", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--routes", type=int, default=128)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument(
        "--min-step", type=int, default=0,
        help="Ignore queued checkpoints earlier than this optimizer step.")
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--max-used-mib", type=int, default=2000)
    args = parser.parse_args()
    if len(args.gpus) not in (1, 2):
        parser.error("--gpus accepts one GPU or two GPUs")
    if args.routes < 2:
        parser.error("--routes must be at least two")
    queue = args.run_dir / "closed_loop_queue"
    while True:
        snapshots = sorted(queue.glob("step_*.pt")) if queue.exists() else []
        pending = []
        for checkpoint in snapshots:
            step = int(checkpoint.stem.split("_")[-1])
            if step < args.min_step:
                continue
            summary = args.output / f"step_{step:06d}" / "summary_128.json"
            if not summary.exists():
                pending.append((step, checkpoint))
        if pending:
            used = gpu_memory_used()
            if all(used.get(gpu, 10**9) <= args.max_used_mib
                   for gpu in args.gpus):
                step, checkpoint = pending[0]
                print(json.dumps({
                    "event": "start_evaluation", "step": step,
                    "checkpoint": str(checkpoint), "gpus": args.gpus,
                }), flush=True)
                try:
                    evaluate(args, checkpoint, step)
                except Exception as error:
                    print(json.dumps({
                        "event": "evaluation_error", "step": step,
                        "error": repr(error),
                    }), flush=True)
                    time.sleep(args.poll_seconds)
                continue
        final_summary = (
            args.output / f"step_{args.final_step:06d}" / "summary_128.json")
        if final_summary.exists():
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
