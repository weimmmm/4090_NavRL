"""Independent immutable-snapshot evaluator; it never connects to the actor.

Run one worker for closed loop and another for fixed validation. Slow or
failed evaluation cannot reset, interrupt, or roll back the ongoing learner.
Every scheduled snapshot is queued, not overwritten with newer weights.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_action_dagger_sft import ROOT, HOST, PROJECT, TRAIN, PY, DEPS
from online_sft_jobs import (atomic_json, docker_command, read_json,
                             run_logged, snapshot_steps, wait_snapshot)


def record_closed_loop(metrics, tensorboard, step, log):
    command = [PY, "-c", "import json,sys; from torch.utils.tensorboard import SummaryWriter; "
               "m=json.load(open(sys.argv[1])); w=SummaryWriter(sys.argv[2]); "
               "[w.add_scalar('closed_loop/'+k,v,int(sys.argv[3])) for k,v in m['rates'].items()]; "
               "[w.add_scalar('closed_loop/count_'+k,v,int(sys.argv[3])) for k,v in m['counts'].items()]; w.close()",
               metrics, tensorboard, step]
    run_logged(docker_command(TRAIN, command, {}, ROOT), log, timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("closed-loop", "fixed"), required=True)
    parser.add_argument("--host-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--fixed-windows", type=int, default=4096)
    parser.add_argument("--fixed-seed", type=int, default=20261014)
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--include-start", action="store_true")
    parser.add_argument("--baseline-sr", type=float, default=.84375)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--flow-only", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=0)
    parser.add_argument("--tensorboard", type=Path,
                        help="Optional separate event directory for an algorithm ablation")
    args = parser.parse_args()
    status_path = args.host_out / f"{args.kind}_status.json"
    steps = snapshot_steps(args.start_step, args.final_step, args.every,
                           include_start=args.include_start or args.kind == "fixed")
    completed, errors = [], []
    best_sr = args.baseline_sr
    fixed_identity = None
    for index, step in enumerate(steps):
        checkpoint_name = f"sft_step_{step}.pt"
        atomic_json(status_path, {"stage": "waiting_for_snapshot", "gpu": args.gpu,
                                  "next_step": step, "completed": completed,
                                  "pending": steps[index:], "errors": errors})
        if not wait_snapshot(args.host_out/checkpoint_name, args.host_out):
            errors.append({"step": step, "error": "Training ended without this snapshot"})
            continue
        gpu_lock = None
        if args.flow_only and args.kind == "closed-loop":
            lock_dir = HOST.parent/".runtime"
            lock_dir.mkdir(exist_ok=True)
            gpu_lock = (lock_dir/f"sft_gpu{args.gpu}_eval.lock").open("a")
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        while args.min_free_mib > 0:
            free = int(subprocess.check_output(
                ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"], text=True).strip())
            if free >= args.min_free_mib:
                break
            atomic_json(status_path, {"stage": "waiting_for_gpu_memory", "gpu": args.gpu,
                                      "free_mib": free, "snapshot_step": step,
                                      "completed": completed, "errors": errors})
            time.sleep(5)
        before = read_json(args.host_out / "progress.json")
        started = time.time()
        checkpoint = args.out / checkpoint_name
        folder = (f"closed_loop_step_{step}" if args.kind == "closed-loop"
                  else f"fixed_validation_step_{step}")
        host_job, job = args.host_out/folder, args.out/folder
        host_job.mkdir(parents=True, exist_ok=True)
        atomic_json(status_path, {"stage": "running", "gpu": args.gpu, "snapshot_step": step,
                                  "training_before": before, "completed": completed,
                                  "pending": steps[index+1:], "errors": errors})
        try:
            if args.kind == "closed-loop":
                environment = {"ISAAC_ACTIVE_GPU": "0", "PYTHONUNBUFFERED": "1",
                               "PYTHONPATH": f"{PROJECT}:{DEPS}",
                               "NAVRL_ASSETS_ROOT": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0"}
                command = ["/opt/navrl/bin/isaac-python", "training/scripts/wam_closed_loop_eval.py",
                           "headless=true", "device=cuda:0", "seed=20261014",
                           f"env.num_envs={args.eval_num_envs}", "env.max_episode_length=2500",
                           "env.num_obstacles=350", "env_dyn.num_obstacles=0",
                           "env.scene_path=training/datasets/wam_40x40_obs200_350_settled_5m/environments/map40_obs0350.pt",
                           f"wam_eval.checkpoint={checkpoint}", "wam_eval.bootstrap_forward_speed_mps=1.0",
                           "wam_eval.bootstrap_action_chunks=1", "wam_eval.goal_stop_adapter.enabled=true",
                           "wam_eval.goal_stop_adapter.enter_distance_m=0.5",
                           "wam_eval.goal_stop_adapter.enter_speed_mps=0.2", "wam_eval.safety_filter.enabled=false",
                           f"wam_eval.metrics_path={job}/metrics.json",
                           f"wam_eval.trajectory_path={job}/trajectories.npz"]
                if args.flow_only:
                    command = ["seed=0" if item == "seed=20261014" else item
                               for item in command]
                    command = ["env.scene_path=aliyun_lidar_wam/datasets/"
                               "wam_40x40_obs200_350_settled_5m/environments/map40_obs0350.pt"
                               if item.startswith("env.scene_path=") else item for item in command]
                    command += ["env.fixed_route_start_index=0", "env.success_distance=0.5",
                                "env.success_speed=0.2", "env.success_stop_steps=10",
                                "wam_eval.action_generation=flow", "wam_eval.sample_steps=10",
                                "wam_eval.action_seed=271828", "wam_eval.action_noise_scale=1.0",
                                "wam_eval.use_past_actions=false", "sensor.lidar_range=5.0",
                                "wam_eval.cuda_memory_fraction=0.12"]
            else:
                environment = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONUNBUFFERED": "1",
                               "PYTHONPATH": f"{PROJECT}:{DEPS}"}
                command = [PY, "-m", "lidar_wam.runner.action_expert_fixed_validation",
                           "--checkpoint", checkpoint, "--cache", args.cache,
                           "--output", job/"metrics.json", "--tensorboard", args.out/"tensorboard",
                           "--windows", args.fixed_windows, "--seed", args.fixed_seed]
            # Timeout lives INSIDE the container so an unresponsive Isaac app
            # is stopped along with its own children, not merely docker exec.
            bounded = ["timeout", "--signal=TERM", "--kill-after=15s", str(args.timeout), *command]
            if args.flow_only and args.kind == "closed-loop":
                # Match the historical baseline's interpreter, allocator,
                # BLAS/OMP threads and all environment/control overrides.
                from run_flow_action_training import evaluation_command
                exact = evaluation_command(args.host_out/checkpoint_name,
                                           host_job, HOST.parent, routes=args.eval_num_envs)
                exact = [args.container if x == "navrl-collect-gpu0" else x for x in exact]
                exact[exact.index("1800")] = str(args.timeout)
                run_logged(exact, host_job/"worker.log", timeout=args.timeout+60)
            else:
                run_logged(docker_command(args.container, bounded, environment, ROOT),
                           host_job/"worker.log", timeout=args.timeout+60)
            metrics = read_json(host_job/"metrics.json")
            if not metrics:
                raise RuntimeError("Evaluator exited without valid metrics")
            if args.kind == "closed-loop":
                if int(metrics["checkpoint_step"]) != step:
                    raise RuntimeError("Closed-loop snapshot step differs from requested step")
                record_closed_loop(job/"metrics.json", args.tensorboard or args.out/"tensorboard", step,
                                   host_job/"tensorboard.log")
                sr = float(metrics["rates"]["sr"])
                if sr > best_sr:
                    best_sr = sr
                    temporary = args.host_out/"best_closed_loop.pt.tmp"
                    shutil.copyfile(args.host_out/checkpoint_name, temporary)
                    os.replace(temporary, args.host_out/"best_closed_loop.pt")
                    atomic_json(args.host_out/"best_closed_loop_metrics.json",
                                {"step": step, "rates": metrics["rates"], "counts": metrics["counts"]})
            else:
                if int(metrics["step"]) != step:
                    raise RuntimeError("Fixed-validation snapshot step differs from requested step")
                identity = metrics["fixed_samples"]
                if fixed_identity is not None and identity != fixed_identity:
                    raise RuntimeError("Fixed validation samples changed between snapshots")
                fixed_identity = identity
            completed.append(step)
            after = read_json(args.host_out/"progress.json")
            info = {"snapshot_step": step, "gpu": args.gpu, "started_at": started,
                    "finished_at": time.time(), "training_before": before, "training_after": after,
                    "training_updates_during_job": int(after.get("step", step))-int(before.get("step", step)),
                    "controls_training_environment": False, "paused_training": False}
            atomic_json(host_job/"concurrency.json", info)
            print(json.dumps({"kind": args.kind, "step": step,
                              "training_updates_during_job": info["training_updates_during_job"],
                              "metrics": metrics.get("rates", metrics.get("metrics"))}), flush=True)
        except (subprocess.SubprocessError, RuntimeError, OSError, KeyError) as exc:
            failure = {"step": step, "error": f"{type(exc).__name__}: {exc}"}
            errors.append(failure)
            atomic_json(host_job/"error.json", failure)
            print(json.dumps(failure), flush=True)
            # Continue draining snapshots; NEVER stop/modify the actor or learner.
        finally:
            if gpu_lock is not None:
                gpu_lock.close()
    atomic_json(status_path, {"stage": "complete" if not errors else "complete_with_errors",
                              "gpu": args.gpu, "completed": completed, "errors": errors,
                              "best_sr": best_sr if args.kind == "closed-loop" else None})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
