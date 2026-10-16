"""Evaluate saved Flow checkpoints sequentially without interrupting training."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from run_flow_action_training import gpu_free_mib, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--start-step", type=int, default=30500)
    parser.add_argument("--end-step", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=500)
    parser.add_argument("--min-free-mib", type=int, default=9000)
    parser.add_argument("--wait-for-lock", action="store_true",
                        help="wait for an existing evaluation queue to finish")
    parser.add_argument("--host-root", type=Path,
                        default=Path("/home/yimingwei/NavRL"))
    args = parser.parse_args()
    if args.interval <= 0 or args.start_step > args.end_step:
        parser.error("require positive interval and start <= end")
    output = args.host_root / "isaac-training/training/lidar_WAM/outputs"
    run = output / args.run_name
    job = output / (args.run_name + "_job")
    if not run.is_dir() or not job.is_dir():
        raise FileNotFoundError("Existing training run and job are required")
    lock = (job / "evaluation_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_lock else fcntl.LOCK_NB))
    steps = list(range(args.start_step, args.end_step + 1, args.interval))
    state = {"pid": os.getpid(), "state": "starting", "gpu": 0,
             "steps": steps, "completed": [], "failed": [],
             "started_unix": time.time(), "training_interrupted": False}

    def save(**values):
        state.update(values, heartbeat_unix=time.time())
        write_json(job / "evaluation_queue_status.json", state)

    def summarize():
        results = []
        for path in job.glob("manual_step_*/status.json"):
            item = json.loads(path.read_text())
            if item.get("state") == "complete":
                results.append({key: item[key] for key in ("step", "counts", "rates")})
        results.sort(key=lambda row: row["step"])
        write_json(job / "closed_loop_summary.json", {
            "results": results,
            "best": max(results, key=lambda row: row["rates"]["sr"]) if results else None,
            "updated_unix": time.time()})

    flag_path = job / "manual_evaluation_only.json"
    flag = json.loads(flag_path.read_text()) if flag_path.exists() else {}
    flag.update(automatic_closed_loop_enabled=True, mode="every_500_steps_queue",
                enabled_by_user_unix=time.time(), start_step=args.start_step,
                end_step=args.end_step, interval=args.interval, queue_pid=os.getpid(),
                training_kept_running=True, tensorboard_kept_running=True)
    write_json(flag_path, flag)
    try:
        for step in steps:
            save(current_step=step, state="waiting_checkpoint")
            checkpoint = run / f"step_{step}.pt"
            deadline = time.monotonic() + 4 * 3600
            while not checkpoint.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Checkpoint did not appear: {checkpoint}")
                time.sleep(10)
                save()
            folder = job / f"manual_step_{step:06d}"
            if folder.exists():
                status = json.loads((folder / "status.json").read_text())
                if status.get("state") != "complete":
                    raise RuntimeError(f"Existing incomplete evaluation requires inspection: {folder}")
                if status.get("step") != step or sum(status["counts"].values()) != 128:
                    raise ValueError(f"Invalid existing evaluation: {folder}")
                state["completed"].append(step)
                summarize()
                continue
            save(state="waiting_gpu_memory")
            while gpu_free_mib(0) < args.min_free_mib:
                time.sleep(10)
                save(gpu_free_mib=gpu_free_mib(0))
            save(state="evaluating")
            command = ["python3", str(Path(__file__).with_name("evaluate_flow_checkpoint.py")),
                       "--run-name", args.run_name, "--step", str(step),
                       "--host-root", str(args.host_root),
                       "--min-free-mib", str(args.min_free_mib)]
            print(f"START step={step}", flush=True)
            # The helper saves trajectories and records TensorBoard metrics.
            result = subprocess.run(command, check=False)
            path = folder / "status.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            if result.returncode == 0 and status.get("state") == "complete":
                state["completed"].append(step)
            else:
                state["failed"].append({"step": step, "exit_code": result.returncode,
                                        "error": status.get("error")})
            summarize()
            save()
            print(f"FINISH step={step} exit={result.returncode}", flush=True)
        save(state="complete", current_step=None, finished_unix=time.time())
    except BaseException as error:
        save(state="failed", error=str(error), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()
