"""Selective pure-online SFT of the 85.9375% Flow-only baseline.

Run on the host. Defaults to a 500-update pilot, not a long unvalidated run.
No existing training, evaluators, or external GPU jobs are stopped.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from run_action_dagger_sft import HOST, PROJECT


def build_command(args):
    source = PROJECT / ("outputs/action_flow_only_history1_axis2_b128_6gpu_20261015/"
                        "step_57500.pt")
    command = [sys.executable, HOST/"training/scripts/run_action_online_sft.py",
               "--flow-only", "--run-name", args.run_name,
               "--checkpoint", source, "--best-checkpoint", source,
               "--baseline-sr", str(110/128), "--evaluate-start",
               "--teacher-target-mode", "student-box", "--anchor-weight", "0",
               "--supervision-horizon", "1", "--freeze-policy-output-head",
               "--selective-correction-threshold-mps", ".3",
               "--updates", str(args.updates), "--eval-every", str(args.eval_every),
               "--num-envs", str(args.num_envs), "--per-gpu-batch-size", str(args.per_gpu_batch_size),
               "--inference-batch-size", str(args.inference_batch_size),
               "--min-global-batch-size", str(args.min_global_batch_size),
               "--lr", "5e-6", "--cuda-memory-fraction", ".12",
               "--actor-cuda-memory-fraction", str(args.actor_cuda_memory_fraction),
               "--sim-min-free-mib", str(args.sim_min_free_mib),
               "--low-memory-actor-physx",
               "--map-seed-base", "8500000", "--route-seed-base", "9500000",
               "--training-gpus", args.training_gpus,
               "--sim-gpu", str(args.sim_gpu), "--eval-gpu", str(args.eval_gpu),
               "--sim-container", f"navrl-collect-gpu{args.sim_gpu}",
               "--eval-container", f"navrl-collect-gpu{args.eval_gpu}",
               "--skip-fixed-validation"]
    if args.smoke:
        command += ["--smoke", "--smoke-updates", "2", "--smoke-num-envs", "32",
                    "--smoke-eval-num-envs", "16"]
    return list(map(str, command))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="action_flow_base57500_selective_sft_v1")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--training-gpus", default="2,3,4,5,7")
    parser.add_argument("--sim-gpu", type=int, default=1)
    parser.add_argument("--eval-gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--per-gpu-batch-size", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--min-global-batch-size", type=int, default=32)
    parser.add_argument("--actor-cuda-memory-fraction", type=float, default=.06)
    parser.add_argument("--sim-min-free-mib", type=int, default=4000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    command = build_command(args)
    if args.print_command:
        import shlex
        print(shlex.join(command))
    else:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
