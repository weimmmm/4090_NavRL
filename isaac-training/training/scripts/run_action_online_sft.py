"""One persistent Isaac SFT actor and two independent GPU evaluation workers.

Run on the 4090 HOST. The training actor stays alive from start to finish;
only terminated drone slots refill. Immutable snapshots queue on GPU4 for
closed loop and GPU5 for fixed validation, without pausing training or mixing
validation samples into its gradients. Previous experiments are preserved.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_action_dagger_sft import ROOT, HOST, PROJECT, TRAIN, SIM, PY, DEPS
from online_sft_jobs import (atomic_json, docker_command, read_json,
                             snapshot_steps, start_logged, validate_gpu_roles)


def check_gpus(args, training_gpus):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,uuid", "--format=csv,noheader,nounits"], text=True)
    cards = {}
    for line in output.splitlines():
        index, free, uuid = (part.strip() for part in line.split(","))
        cards[int(index)] = {"free_mib": int(free), "uuid": uuid}
    roles = ([(gpu, 2048) for gpu in training_gpus]
             + [(args.sim_gpu, 8000), (args.eval_gpu, 6000)])
    if not args.skip_fixed_validation:
        roles.append((args.validation_gpu, 2048))
    for gpu, minimum in roles:
        if gpu not in cards or cards[gpu]["free_mib"] < minimum:
            raise RuntimeError(f"GPU {gpu} needs {minimum} MiB free; external jobs will not be stopped")
    # A dedicated Isaac container exposes one PHYSICAL card as logical cuda:0.
    for container, gpu in ((args.sim_container, args.sim_gpu), (args.eval_container, args.eval_gpu)):
        visible = subprocess.check_output(
            ["docker", "exec", container, "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
        if visible.strip().splitlines() != [cards[gpu]["uuid"]]:
            raise RuntimeError(f"{container} must expose ONLY physical GPU {gpu}")
    return cards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="action_continuous_online_sft")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--per-gpu-batch-size", type=int, default=128)
    parser.add_argument("--mode", choices=("pure-online",), default="pure-online")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--smoke-num-envs", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--refill", action="store_true", help="Compatibility flag; refill is always enabled")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--best-checkpoint", type=Path)
    parser.add_argument("--obstacle-count", type=int, choices=(350,), default=350)
    parser.add_argument("--map-seed-base", type=int, default=8500000)
    parser.add_argument("--route-seed-base", type=int, default=9500000)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--baseline-sr", type=float, default=.8125)
    parser.add_argument("--smoke-updates", type=int, default=8)
    parser.add_argument("--smoke-eval-num-envs", type=int, default=16)
    parser.add_argument("--updates", type=int, default=10000, help="Additional optimizer updates")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--evaluate-start", action="store_true",
                        help="Also evaluate the untouched source policy on the same 128 routes")
    parser.add_argument("--teacher-target-mode", choices=("raw", "student-norm"), default="raw",
                        help="Label-only ablation; never changes deployed velocity bounds")
    parser.add_argument("--anchor-weight", type=float, default=0.,
                        help="Frozen original policy trust penalty on current online states only")
    parser.add_argument("--anchor-checkpoint", type=Path)
    parser.add_argument("--selective-correction-threshold-mps", type=float, default=0.,
                        help="PPO supervision only on significant disagreement with original; "
                             "otherwise preserve original whole chunk on the same online state")
    parser.add_argument("--freeze-policy-output-head", action="store_true")
    parser.add_argument("--supervision-horizon", type=int, choices=(1, 10), default=1)
    parser.add_argument("--online-tail-weight", type=float, default=1.)
    parser.add_argument("--unclipped-training-output", action="store_true")
    parser.add_argument("--min-global-batch-size", type=int, default=256)
    parser.add_argument("--training-gpus", default="1,2,3,7")
    parser.add_argument("--sim-gpu", type=int, default=0)
    parser.add_argument("--eval-gpu", type=int, default=4)
    parser.add_argument("--validation-gpu", type=int, default=5)
    parser.add_argument("--skip-fixed-validation", action="store_true",
                        help="Closed-loop-only pilot; reserve no extra GPU for offline diagnostics")
    parser.add_argument("--sim-container", default=SIM)
    parser.add_argument("--eval-container", default="navrl-collect-gpu4")
    parser.add_argument("--fixed-cache", type=Path,
                        default=PROJECT/"outputs/action_closed_loop_sft_first_step_20261014/offline_cache.pt")
    parser.add_argument("--fixed-windows", type=int, default=4096)
    parser.add_argument("--job-timeout", type=int, default=900)
    args = parser.parse_args()
    if (not 0 <= args.selective_correction_threshold_mps < float("inf")
            or (args.selective_correction_threshold_mps > 0
                and (args.anchor_weight != 0 or args.supervision_horizon != 1
                     or args.unclipped_training_output or args.teacher_target_mode != "student-norm"))):
        parser.error("Selective correction requires a finite positive threshold, feasible first labels, "
                     "bounded output and no simultaneous anchor/tail ablation")
    training_gpus = [int(value) for value in args.training_gpus.split(",")]
    validate_gpu_roles(training_gpus, args.sim_gpu, args.eval_gpu,
                       None if args.skip_fixed_validation else args.validation_gpu)
    if len(training_gpus) != 4 or min(args.updates, args.smoke_updates, args.eval_every,
            args.per_gpu_batch_size, args.fixed_windows, args.inference_batch_size, args.job_timeout) <= 0:
        parser.error("Expected four GPUs and positive update/interval/window/batch counts")
    fleet = args.smoke_num_envs if args.smoke else args.num_envs
    if not 4 <= fleet <= 2048 or fleet % 4:
        parser.error("Fleet must be a multiple of four and at most 2048")
    minimum = min(args.min_global_batch_size, fleet, args.per_gpu_batch_size*4)
    if (minimum <= 0 or args.run_name in ("", ".", "..")
            or Path(args.run_name).name != args.run_name):
        parser.error("Minimum fresh batch must be positive; run name must not contain a path")
    gpu_state = check_gpus(args, training_gpus)
    out, host_out = PROJECT/"outputs"/args.run_name, HOST/"training/lidar_WAM/outputs"/args.run_name
    if host_out.exists() and any((host_out/name).exists() for name in ("config.json", "launch_config.json")):
        raise FileExistsError("Use a new run name; old experiments are never overwritten")
    host_out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or PROJECT/"outputs/action_dit_final_lee_ego_history1_5m_v3/best_closed_loop.pt"
    best_checkpoint = args.best_checkpoint or checkpoint
    shutil.copyfile(HOST/best_checkpoint.relative_to(ROOT), host_out/"best_closed_loop.pt")
    updates = args.smoke_updates if args.smoke else args.updates
    final_step = args.start_step+updates
    token = hashlib.sha256(args.run_name.encode()).hexdigest()[:12]
    socket = Path(f"/workspace/NavRL/.online_sft_{token}.sock")
    host_socket = HOST.parent/socket.name
    if host_socket.exists():
        raise FileExistsError(host_socket)
    atomic_json(host_out/"launch_config.json", {**vars(args), "final_step": final_step,
                "refill": True, "simulator_sessions": 1, "gpu_preflight": gpu_state,
                "evaluation_steps": snapshot_steps(args.start_step, final_step, args.eval_every,
                                                     include_start=args.evaluate_start)})
    state = {"stage": "starting", "training_gpus": training_gpus, "simulation_gpu": args.sim_gpu,
             "evaluation_gpu": args.eval_gpu,
             "fixed_validation_gpu": None if args.skip_fixed_validation else args.validation_gpu,
             "fixed_validation_enabled": not args.skip_fixed_validation,
             "mode": "pure-online", "uses_offline_replay": False, "num_parallel_drones": fleet,
             "per_gpu_total_batch_size": args.per_gpu_batch_size, "start_step": args.start_step,
             "final_step": final_step, "additional_updates": updates, "training_obstacle_count": 350,
             "training_map_seed": args.map_seed_base, "training_route_seed": args.route_seed_base,
             "continuous_actor": True, "evaluation_blocks_training": False, "refill_enabled": True,
             "fixed_validation_windows": 0 if args.skip_fixed_validation else args.fixed_windows,
             "minimum_global_online_batch": minimum,
             "ppo_query_every_physics_steps": 10 if args.supervision_horizon == 1 else 1,
             "supervised_action_index": 0 if args.supervision_horizon == 1 else "0:9",
             "supervision_horizon": args.supervision_horizon,
             "selective_correction_threshold_mps": args.selective_correction_threshold_mps,
             "teacher_target_mode": args.teacher_target_mode, "anchor_weight": args.anchor_weight}
    state["unclipped_training_output"] = args.unclipped_training_output
    state["freeze_policy_output_head"] = args.freeze_policy_output_head
    atomic_json(host_out/"status.json", state)
    train_env = {"CUDA_VISIBLE_DEVICES": args.training_gpus, "OMP_NUM_THREADS": "4",
                 "PYTHONPATH": f"{PROJECT}:{DEPS}", "PYTHONUNBUFFERED": "1"}
    command = [PY, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "-m",
               "lidar_wam.runner.action_expert_online_sft", "serve", "--socket", socket,
               "--checkpoint", checkpoint, "--out", out, "--max-updates", final_step,
               "--start-step", args.start_step, "--lr", "0.000005", "--save-every", args.eval_every,
               "--per-gpu-batch-size", args.per_gpu_batch_size, "--mode", "pure-online",
               "--online-weight", "1", "--refill-enabled",
               "--teacher-target-mode", args.teacher_target_mode, "--anchor-weight", args.anchor_weight,
               "--supervision-horizon", args.supervision_horizon,
               "--selective-correction-threshold-mps", args.selective_correction_threshold_mps,
               "--online-tail-weight", args.online_tail_weight]
    if args.anchor_checkpoint:
        command += ["--anchor-checkpoint", args.anchor_checkpoint]
    if args.unclipped_training_output:
        command.append("--unclipped-training-output")
    if args.freeze_policy_output_head:
        command.append("--freeze-policy-output-head")
    learner = start_logged(docker_command(TRAIN, command, train_env, ROOT), host_out/"learner.log")
    workers, actor = [], None
    try:
        for _ in range(120):
            if learner.poll() is not None:
                raise RuntimeError("Learner exited before IPC readiness")
            if host_socket.exists():
                break
            time.sleep(1)
        else:
            raise TimeoutError("Learner IPC readiness timeout")
        for kind, gpu, container in (("closed-loop", args.eval_gpu, args.eval_container),
                                     ("fixed", args.validation_gpu, TRAIN)):
            if kind == "fixed" and args.skip_fixed_validation:
                continue
            command = [sys.executable, HOST/"training/scripts/monitor_online_sft.py",
                       "--kind", kind, "--host-out", host_out, "--out", out,
                       "--start-step", args.start_step, "--final-step", final_step, "--every", args.eval_every,
                       "--gpu", gpu, "--container", container, "--cache", args.fixed_cache,
                       "--fixed-windows", args.fixed_windows, "--timeout", args.job_timeout,
                       "--eval-num-envs", args.smoke_eval_num_envs if args.smoke else 128,
                       "--baseline-sr", 1.0 if args.smoke else args.baseline_sr]
            if args.evaluate_start:
                command.append("--include-start")
            workers.append(start_logged(list(map(str, command)), host_out/f"{kind}_monitor.log"))
        sim_env = {"ISAAC_ACTIVE_GPU": "0", "PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{PROJECT}:{DEPS}",
                   "NAVRL_ASSETS_ROOT": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0"}
        command = ["/opt/navrl/bin/isaac-python", "training/scripts/online_action_sft.py",
                   "headless=true", "device=cuda:0", "seed=20261014", f"env.num_envs={fleet}",
                   "env.max_episode_length=2500", "env_dyn.num_obstacles=0", "env.num_obstacles=350",
                   "env.runtime_scene_id=continuous_online_sft", f"env.map_seed={args.map_seed_base}",
                   f"env.route_seed={args.route_seed_base}", f"wam_eval.checkpoint={checkpoint}",
                   "wam_eval.bootstrap_forward_speed_mps=1.0", "wam_eval.bootstrap_action_chunks=1",
                   "wam_eval.goal_stop_adapter.enabled=true", "wam_eval.goal_stop_adapter.enter_distance_m=0.5",
                   "wam_eval.goal_stop_adapter.enter_speed_mps=0.2", "wam_eval.safety_filter.enabled=false",
                   f"+online.teacher_checkpoint={PROJECT}/outputs/privileged_ppo_checkpoint_27000.pt",
                   f"+online.socket={socket}", f"+online.output={out}/actor.json",
                   f"+online.target_updates={final_step}", f"+online.inference_batch_size={args.inference_batch_size}",
                   "+online.refill_enabled=true", f"+online.learn_batch_capacity={args.per_gpu_batch_size*4}",
                   f"+online.minimum_batch_size={minimum}", f"+online.stop_file={out}/stop_requested",
                   f"+online.teacher_target_mode={args.teacher_target_mode}",
                   f"+online.supervision_horizon={args.supervision_horizon}",
                   f"+online.test_force_refill={'true' if args.smoke else 'false'}", "+online.final_session=true"]
        actor = start_logged(docker_command(args.sim_container, command, sim_env, ROOT), host_out/"actor.log")
        state.update(stage="continuous_online_sft", actor_host_pid=actor.pid, learner_host_pid=learner.pid)
        atomic_json(host_out/"status.json", state)
        print(state, flush=True)
        # Wait only for TRAINING here. The independent evaluators continue
        # consuming snapshots, never awaited at a training step boundary.
        if actor.wait() != 0:
            raise RuntimeError("Training actor failed; inspect actor.log")
        if learner.wait(timeout=60) != 0:
            raise RuntimeError("Online learner failed; inspect learner.log")
        step = int(read_json(host_out/"progress.json").get("step", args.start_step))
        atomic_json(host_out/"training_finished.json", {"step": step, "success": True})
        state.update(stage="draining_evaluation_queue", training_finished_step=step)
        atomic_json(host_out/"status.json", state)
        codes = [worker.wait() for worker in workers]
        state["stage"] = "complete" if all(code == 0 for code in codes) else "complete_with_evaluation_errors"
        atomic_json(host_out/"status.json", state)
        if any(codes):
            print("Training finished; see independent evaluator errors", flush=True)
            return 1
    except BaseException as exc:
        (host_out/"stop_requested").touch()
        atomic_json(host_out/"training_finished.json", {"success": False,
                    "step": read_json(host_out/"progress.json").get("step", args.start_step),
                    "error": f"{type(exc).__name__}: {exc}"})
        state.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(host_out/"status.json", state)
        if actor is not None and actor.poll() is None:
            actor.wait(timeout=120)  # Our own actor's graceful K=10 stop.
        if learner.poll() is None:
            command = ["timeout", "30", PY, "-c", "import sys; from multiprocessing.connection import Client; "
                       "c=Client(sys.argv[1],family='AF_UNIX'); c.recv(); "
                       "c.send({'command':'finish'}); c.recv(); c.close()", socket]
            try:
                subprocess.run(docker_command(TRAIN, command, {}, ROOT), check=True, timeout=40)
            except subprocess.SubprocessError:
                pass
            learner.wait(timeout=60)
        raise


if __name__ == "__main__":
    raise SystemExit(main() or 0)
