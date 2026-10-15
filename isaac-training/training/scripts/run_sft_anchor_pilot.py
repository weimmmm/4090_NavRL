"""Run one conservative online-SFT pilot only after its wiring smoke passed.

Both student and trust reference start from the ORIGINAL 7500-step expert.
The reference sees only the same freshly visited student observations. It
never receives future data and no offline samples/replay are used for grads.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from online_sft_jobs import read_json
from run_action_dagger_sft import HOST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-pid", type=int, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--smoke-name", default="action_sft_anchor_wiring_smoke_20261014")
    parser.add_argument("--run-name", default="action_sft_original_current_anchor_20261014")
    parser.add_argument("--supervision-horizon", type=int, choices=(1, 10), default=1)
    parser.add_argument("--anchor-weight", type=float, default=1.)
    parser.add_argument("--selective-correction-threshold-mps", type=float, default=0.)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--freeze-policy-output-head", action="store_true")
    parser.add_argument("--unclipped-training-output", action="store_true")
    args = parser.parse_args()
    if min(args.updates, args.eval_every) <= 0 or any(Path(name).name != name for name in (args.smoke_name, args.run_name)):
        parser.error("Positive update count and path-free run names required")
    smoke = HOST/"training/lidar_WAM/outputs"/args.smoke_name
    deadline = time.monotonic()+600
    while read_json(smoke/"status.json").get("stage") != "complete":
        stage = read_json(smoke/"status.json").get("stage")
        if stage in ("failed", "complete_with_evaluation_errors"):
            raise RuntimeError(f"Anchor smoke failed: {stage}")
        try:
            os.kill(args.smoke_pid, 0)  # Read-only check of the CONFIRMED test handle.
        except ProcessLookupError:
            raise RuntimeError("Smoke handle disappeared without successful completion")
        if time.monotonic() > deadline:
            raise TimeoutError("Smoke gate exceeded ten minutes; no formal training launched")
        time.sleep(1)
    command = [sys.executable, str(HOST/"training/scripts/run_action_online_sft.py"),
               "--run-name", args.run_name,
               "--teacher-target-mode", "student-norm", "--anchor-weight", str(args.anchor_weight),
               "--supervision-horizon", str(args.supervision_horizon),
               "--selective-correction-threshold-mps", str(args.selective_correction_threshold_mps),
               "--start-step", "0", "--updates", str(args.updates), "--eval-every", str(args.eval_every),
               "--evaluate-start", "--num-envs", "512", "--per-gpu-batch-size", "128",
               "--fixed-windows", "4096", "--map-seed-base", "8500000", "--route-seed-base", "9500000"]
    if args.unclipped_training_output:
        command.append("--unclipped-training-output")
    if args.freeze_policy_output_head:
        command.append("--freeze-policy-output-head")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
