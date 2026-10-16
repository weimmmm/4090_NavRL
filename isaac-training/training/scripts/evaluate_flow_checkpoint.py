"""Prioritize one Flow checkpoint on GPU0 without pausing the trainer.

An optional SIGSTOP affects ONLY the validated host evaluation-dispatch loop,
not its separately launched trainer, TensorBoard, or in-flight Isaac process.
The dispatcher is resumed in finally, including when evaluation fails.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_flow_action_training import (
    container_path, docker_prefix, evaluation_command, gpu_free_mib, spawn, write_json,
)


def validate_dispatcher(pid, run_name):
    command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    words = [word.decode() for word in command if word]
    script = "/home/yimingwei/NavRL/isaac-training/training/scripts/run_flow_action_training.py"
    if (script not in words or "run" not in words or "--run-name" not in words
            or words[words.index("--run-name") + 1] != run_name):
        raise ValueError("PID is not this run's evaluation dispatcher")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--min-free-mib", type=int, default=9000)
    parser.add_argument("--host-root", type=Path, default=Path("/home/yimingwei/NavRL"))
    args = parser.parse_args()
    root = args.host_root.resolve()
    output = root / "isaac-training/training/lidar_WAM/outputs"
    run = output / args.run_name
    checkpoint = run / f"step_{args.step}.pt"
    folder = output / (args.run_name + "_job") / f"manual_step_{args.step:06d}"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    folder.mkdir(exist_ok=False)
    paused = False
    status = {"step": args.step, "checkpoint": str(checkpoint), "routes": 128,
              "gpu": 0, "started_unix": time.time(), "state": "initializing"}

    def save(**values):
        status.update(values)
        write_json(folder / "status.json", status)

    try:
        if args.supervisor_pid is not None:
            validate_dispatcher(args.supervisor_pid, args.run_name)
            proc_status = (Path("/proc") / str(args.supervisor_pid) / "status").read_text()
            state = next(line for line in proc_status.splitlines() if line.startswith("State:"))
            if "T" not in state.split()[1]:
                os.kill(args.supervisor_pid, signal.SIGSTOP)
                paused = True
            save(dispatcher_pid=args.supervisor_pid, dispatcher_paused_by_this_job=paused)
        deadline = time.monotonic() + 300
        while gpu_free_mib(0) < args.min_free_mib:
            save(state="waiting_for_gpu0", gpu0_free_mib=gpu_free_mib(0))
            if time.monotonic() > deadline:
                raise TimeoutError("GPU0 memory did not become available; no other process was stopped")
            time.sleep(5)
        command = evaluation_command(checkpoint, folder, root, routes=128)
        process = spawn(command, folder / "eval.log")
        save(state="evaluating", pid=process.pid, command=command)
        print(f"Evaluating step {args.step} on GPU0; trainer continues", flush=True)
        code = process.wait(timeout=1900)
        metrics_file = folder / "metrics.json"
        if not metrics_file.is_file():
            raise RuntimeError(f"Isaac exit {code}; no metrics; inspect {folder / 'eval.log'}")
        metrics = json.loads(metrics_file.read_text())
        if metrics["checkpoint_step"] != args.step or sum(metrics["counts"].values()) != 128:
            raise ValueError("checkpoint step or route count mismatch")
        if not (folder / "trajectories.npz").is_file():
            raise FileNotFoundError("trajectory file was not saved")
        command = docker_prefix("lidar-wam-train", gpu="") + ["python",
            "training/scripts/run_flow_action_training.py", "record", "--metrics",
            str(container_path(metrics_file, root)), "--tensorboard",
            str(container_path(run / "tensorboard", root))]
        subprocess.run(command, check=True, timeout=120)
        save(state="complete", counts=metrics["counts"], rates=metrics["rates"],
             finished_unix=time.time(), tensorboard_recorded=True)
        print(json.dumps({"step": args.step, "counts": metrics["counts"],
                          "rates": metrics["rates"]}), flush=True)
    except BaseException as error:
        save(state="failed", error=str(error), finished_unix=time.time())
        raise
    finally:
        if paused:
            validate_dispatcher(args.supervisor_pid, args.run_name)
            os.kill(args.supervisor_pid, signal.SIGCONT)
            save(dispatcher_resumed=True)


if __name__ == "__main__":
    main()
