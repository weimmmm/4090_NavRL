"""Host supervisor: one shared-model Isaac/SFT process per requested GPU.

Never uses GPU0 for training, never stops unrelated jobs, and keeps all
original flight/control/success code read-only. Evaluation is independent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from online_sft_jobs import atomic_json, docker_command, start_logged
from run_action_dagger_sft import ROOT, HOST, PROJECT, DEPS


PROTECTED = ("training/scripts/env.py", "training/scripts/wam_closed_loop_eval.py",
             "third_party/OmniDrones/omni_drones/controllers/lee_position_controller.py",
             "training/cfg/sim.yaml", "training/cfg/train.yaml")


def ensure_container(gpu):
    name = f"navrl-sft-local-gpu{gpu}"
    exists = subprocess.run(["docker", "inspect", name], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0
    if not exists:
        subprocess.run(["docker", "create", "--name", name, "--gpus", f"device={gpu}",
                        "--shm-size=2g", "--mount",
                        "type=bind,source=/home/yimingwei/NavRL,target=/workspace/NavRL",
                        "-e", "ACCEPT_EULA=Y", "-e", "PRIVACY_CONSENT=Y",
                        "-e", "NVIDIA_DRIVER_CAPABILITIES=all", "--entrypoint", "/bin/sleep",
                        "navrl-collect-ready:20260928", "infinity"], check=True)
    running = subprocess.check_output(["docker", "inspect", name, "--format",
                                       "{{.State.Running}}"], text=True).strip()
    if running != "true":
        subprocess.run(["docker", "start", name], check=True)
    visible = subprocess.check_output(["docker", "exec", name, "nvidia-smi",
                                      "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    expected = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid",
                                       "--format=csv,noheader"], text=True).strip()
    if visible != [expected]:
        raise RuntimeError(f"{name} does not exclusively expose GPU{gpu}")
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpus", default="2,3,4,5,7")
    parser.add_argument("--envs-per-worker", type=int, default=16)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--correction-tail-weight", type=float, default=.5)
    parser.add_argument("--threshold-mps", type=float, default=.3)
    parser.add_argument("--failure-window-chunks", type=int, default=0,
                        help="Train PPO corrections only before observed collision/OOB; zero disables")
    parser.add_argument("--tensorboard", type=Path, help="Container event directory; keep runs separate")
    parser.add_argument("--no-evaluation", action="store_true", help="only for short integration probes")
    parser.add_argument("--smoke-refill", action="store_true")
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    if (0 in gpus or 6 in gpus or len(set(gpus)) != len(gpus)
            or not 4 <= args.envs_per_worker <= 32 or args.updates < 1
            or args.eval_every < 1 or args.lr <= 0 or not 0 <= args.failure_window_chunks <= 10):
        parser.error("Use distinct training GPUs excluding 0/6, 4..32 drones and positive counts/lr")
    out = PROJECT/"outputs"/args.run_name
    host_out = HOST/"training/lidar_WAM/outputs"/args.run_name
    host_out.mkdir(parents=True, exist_ok=False)
    invariants = {name: hashlib.sha256((HOST/name).read_bytes()).hexdigest() for name in PROTECTED}
    atomic_json(host_out/"protected_logic_sha256.json", invariants)
    atomic_json(host_out/"launch_config.json", vars(args))
    source = PROJECT/"outputs/action_flow_only_history1_axis2_b128_6gpu_20261015/step_57500.pt"
    shutil.copyfile(HOST/source.relative_to(ROOT), host_out/"best_closed_loop.pt")
    state = {"stage": "starting", "training_gpus": gpus, "evaluation_gpu": 0,
             "simulators_per_training_gpu": 1, "shared_actor_learner_process": True,
             "drones_per_gpu": args.envs_per_worker, "total_drones": args.envs_per_worker*len(gpus),
             "target_updates": args.updates, "source_step": 57500,
             "source_historical_sr": 110/128, "protected_logic_sha256": invariants,
             "worker_pids": [], "started_unix": time.time()}
    atomic_json(host_out/"status.json", state)
    workers = []
    evaluator = None
    try:
        for rank, gpu in enumerate(gpus):
            free = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}",
                       "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
            if free < 5000:
                raise RuntimeError(f"GPU{gpu}: {free} MiB free, need 5000; external jobs untouched")
            container = ensure_container(gpu)
            env = {"RANK": str(rank), "WORLD_SIZE": str(len(gpus)), "GLOO_SOCKET_IFNAME": "eth0",
                   "ISAAC_ACTIVE_GPU": "0", "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
                   "PYTHONPATH": f"{PROJECT}:{DEPS}",
                   "NAVRL_ASSETS_ROOT": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0"}
            command = ["timeout", "--signal=TERM", "--kill-after=20s", "7200",
                       "/opt/navrl/bin/isaac-python", "training/scripts/online_action_sft.py",
                       "headless=true", "device=cuda:0", f"seed={20261016+rank}",
                       f"env.num_envs={args.envs_per_worker}", "env.max_episode_length=2500",
                       "env_dyn.num_obstacles=0", "env.num_obstacles=350",
                       f"env.runtime_scene_id=local_flow_sft_rank{rank}",
                       f"env.map_seed={8501000+rank}", f"env.route_seed={9501000+rank}",
                       "sensor.lidar_range=5.0", "env.success_distance=0.5",
                       "env.success_speed=0.2", "env.success_stop_steps=10",
                       f"wam_eval.checkpoint={source}", "wam_eval.action_generation=flow",
                       "wam_eval.sample_steps=10", f"wam_eval.action_seed={271828+rank*1009}",
                       "wam_eval.action_noise_scale=1.0", "wam_eval.use_past_actions=false",
                       "wam_eval.cuda_memory_fraction=0.08", "wam_eval.bootstrap_forward_speed_mps=1.0",
                       "wam_eval.bootstrap_action_chunks=1", "wam_eval.goal_stop_adapter.enabled=true",
                       "wam_eval.goal_stop_adapter.enter_distance_m=0.5",
                       "wam_eval.goal_stop_adapter.enter_speed_mps=0.2", "wam_eval.safety_filter.enabled=false",
                       f"+online.teacher_checkpoint={PROJECT}/outputs/privileged_ppo_checkpoint_27000.pt",
                       "+online.local_learner=true", f"+online.output={out}/actor_rank_{rank}.json",
                       f"+online.init_method=file://{out}/gloo_init",
                       f"+online.invariants={out}/protected_logic_sha256.json",
                       f"+online.lr={args.lr}", f"+online.correction_threshold_mps={args.threshold_mps}",
                       f"+online.correction_tail_weight={args.correction_tail_weight}",
                       f"+online.failure_window_chunks={args.failure_window_chunks}",
                       f"+online.save_every={args.eval_every}", f"+online.target_updates={args.updates}",
                       "+online.inference_batch_size=8", "+online.refill_enabled=true",
                       f"+online.learn_batch_capacity={args.envs_per_worker}", "+online.minimum_batch_size=1",
                       f"+online.stop_file={out}/stop_requested", "+online.teacher_target_mode=student-box",
                       "+online.supervision_horizon=1", "+online.final_session=true",
                       f"+online.test_force_refill={'true' if args.smoke_refill else 'false'}",
                       "sim.gpu_max_rigid_contact_count=65536", "sim.gpu_max_rigid_patch_count=16384",
                       "sim.gpu_found_lost_pairs_capacity=262144",
                       "sim.gpu_found_lost_aggregate_pairs_capacity=1048576",
                       "sim.gpu_total_aggregate_pairs_capacity=262144",
                       "sim.gpu_max_soft_body_contacts=1024", "sim.gpu_max_particle_contacts=1024"]
            if args.tensorboard is not None:
                command.append(f"+online.tensorboard={args.tensorboard}")
            process = start_logged(docker_command(container, command, env, ROOT),
                                   host_out/f"worker_rank_{rank}_gpu{gpu}.log")
            workers.append(process)
            state["worker_pids"].append(process.pid)
            atomic_json(host_out/f"worker_rank_{rank}_launch.json", {
                "pid": process.pid, "gpu": gpu, "container": container,
                "command": command, "environment": env})
        if not args.no_evaluation:
            command = [sys.executable, HOST/"training/scripts/monitor_online_sft.py",
                       "--kind", "closed-loop", "--host-out", host_out, "--out", out,
                       "--start-step", "0", "--final-step", str(args.updates),
                       "--every", str(args.eval_every), "--gpu", "0", "--container", "navrl-collect-gpu0",
                       "--include-start", "--flow-only", "--eval-num-envs", "128",
                       "--baseline-sr", str(110/128), "--min-free-mib", "6500", "--timeout", "900"]
            if args.tensorboard is not None:
                command += ["--tensorboard", str(args.tensorboard)]
            evaluator = start_logged(list(map(str, command)), host_out/"closed-loop_monitor.log")
            state["evaluator_pid"] = evaluator.pid
        state["stage"] = "training"
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None, 0) for p in workers):
                raise RuntimeError("An integrated Isaac/SFT worker failed; inspect per-worker logs")
            state["heartbeat_unix"] = time.time()
            atomic_json(host_out/"status.json", state)
            time.sleep(5)
        if any(p.returncode != 0 for p in workers):
            raise RuntimeError("Integrated worker exited unsuccessfully")
        state["stage"] = "draining_evaluation_queue"
        atomic_json(host_out/"status.json", state)
        if evaluator is not None and evaluator.wait() != 0:
            raise RuntimeError("Training completed but independent evaluation reported errors")
        state.update(stage="complete", finished_unix=time.time())
        atomic_json(host_out/"status.json", state)
    except BaseException as error:
        (host_out/"stop_requested").touch()
        atomic_json(host_out/"training_finished.json", {"success": False, "error": str(error)})
        state.update(stage="failed", error=str(error), finished_unix=time.time())
        atomic_json(host_out/"status.json", state)
        raise


if __name__ == "__main__":
    main()
