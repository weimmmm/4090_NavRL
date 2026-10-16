"""Host supervisor: six shared training GPUs, queued GPU0 evaluations, TensorBoard.

Only processes created by this job are launched or timed out. No existing
process is stopped. Immutable 500-step checkpoints are evaluated in order;
evaluation never resets the trainer, changes its weights, or skips a queued step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


PROJECT = Path("/workspace/NavRL/isaac-training")
PYTHONPATH = str(PROJECT / "training/lidar_WAM") + ":" + str(
    PROJECT / "aliyun_lidar_wam/lidar_WAM/third_party/diffusers/src")


def container_path(path, root):
    return Path("/workspace/NavRL") / Path(path).resolve().relative_to(root.resolve())


def gpu_free_mib(index):
    result = subprocess.check_output([
        "nvidia-smi", f"--id={index}", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], text=True)
    return int(result.strip())


def docker_prefix(container, *, gpu=None):
    result = ["docker", "exec", "-w", str(PROJECT), "-e", "PYTHONUNBUFFERED=1",
              "-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=1",
              "-e", "PYTHONPATH=" + PYTHONPATH,
              "-e", "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64"]
    if gpu is not None:
        result += ["-e", "CUDA_VISIBLE_DEVICES=" + str(gpu)]
    return result + [container]


def evaluation_command(checkpoint, folder, root, routes=128):
    scene = PROJECT / "aliyun_lidar_wam/datasets/wam_40x40_obs200_350_settled_5m/environments/map40_obs0350.pt"
    # This container exposes physical GPU0 only. Keep Isaac's graphics and
    # PhysX devices on that GPU as well, not on a training/VLM GPU.
    prefix = docker_prefix("navrl-collect-gpu0")
    prefix[-1:-1] = ["-e", "ISAAC_ACTIVE_GPU=0", "-e",
        "NAVRL_ASSETS_ROOT=http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0"]
    return prefix + ["timeout", "--signal=TERM", "--kill-after=20s", "1800",
        "/isaac-sim/python.sh", "training/scripts/wam_closed_loop_eval.py",
        "device=cuda:0", "headless=true", "seed=0", f"env.num_envs={routes}",
        "env.max_episode_length=2500", "env_dyn.num_obstacles=0",
        "env.scene_path=" + str(scene), "env.fixed_route_start_index=0",
        "env.success_distance=0.5", "env.success_speed=0.2", "env.success_stop_steps=10",
        "wam_eval.checkpoint=" + str(container_path(checkpoint, root)),
        "wam_eval.action_generation=flow", "wam_eval.sample_steps=10",
        "wam_eval.cuda_memory_fraction=0.12", "wam_eval.action_seed=271828",
        "wam_eval.action_noise_scale=1.0", "wam_eval.use_past_actions=false",
        "wam_eval.bootstrap_forward_speed_mps=1.0", "wam_eval.bootstrap_action_chunks=1",
        "wam_eval.safety_filter.enabled=false", "wam_eval.goal_stop_adapter.enabled=true",
        "wam_eval.goal_stop_adapter.enter_distance_m=0.5",
        "wam_eval.goal_stop_adapter.enter_speed_mps=0.2",
        "wam_eval.metrics_path=" + str(container_path(folder / "metrics.json", root)),
        "wam_eval.trajectory_path=" + str(container_path(folder / "trajectories.npz", root))]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def spawn(command, log_path):
    with log_path.open("a") as stream:
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)


def record(args):
    from torch.utils.tensorboard import SummaryWriter
    value = json.loads(args.metrics.read_text())
    step = int(value["checkpoint_step"])
    with SummaryWriter(str(args.tensorboard), filename_suffix=".closed_loop") as writer:
        for key, rate in value["rates"].items():
            writer.add_scalar("closed_loop/" + key, rate, step)
        for key, count in value["counts"].items():
            writer.add_scalar("closed_loop/count_" + key, count, step)
        writer.add_scalar("closed_loop/checkpoint_step", step, step)
        writer.add_scalar("closed_loop/wall_time_s", value["wall_time_s"], step)


def run(args):
    root = args.host_root.resolve()
    outputs = root / "isaac-training/training/lidar_WAM/outputs"
    run_dir = outputs / args.run_name
    job = outputs / (args.run_name + "_job")
    # Refuse to overwrite any prior run or duplicate its processes.
    if run_dir.exists() or job.exists():
        raise FileExistsError(f"choose a fresh run name: {run_dir}")
    for gpu in (1, 2, 3, 4, 5, 7):
        if gpu_free_mib(gpu) < args.min_train_free_mib:
            raise RuntimeError(f"GPU{gpu} no longer has {args.min_train_free_mib} MiB free")
    job.mkdir(parents=True)
    training_command = docker_prefix("lidar-wam-train", gpu="1,2,3,4,5,7")
    training_command += ["torchrun", "--nnodes=1", "--nproc_per_node=6",
        "--master-addr=127.0.0.1", f"--master-port={args.ddp_port}",
        "-m", "lidar_wam.runner.flow_action_expert",
        "--train-root", str(PROJECT / "training/datasets/wam_40x40_obs200_350_settled_5m/train"),
        "--vae-checkpoint", str(PROJECT / "training/lidar_WAM/outputs/vae_5m_finetune_v1/best/diffusion_pytorch_model.safetensors"),
        "--out", str(PROJECT / "training/lidar_WAM/outputs"), "--run-name", args.run_name,
        "--steps", str(args.steps), "--batch-size", str(args.batch_size),
        "--eval-batch-size", "32", "--workers", "2", "--precision", "bf16",
        "--cuda-memory-fraction", str(args.memory_fraction),
        "--lr", "0.0001", "--sample-steps", "10", "--action-scale-mps-xyz", "2,2,2",
        "--eval-every", "500", "--closed-loop-start", "500", "--eval-windows", "4096",
        "--log-every", "25"]
    trainer = spawn(training_command, job / "train.log")
    status = {"state": "running", "started_unix": time.time(),
        "training_pid": trainer.pid, "training_command": training_command,
        "run_dir": str(run_dir), "batch_per_gpu": args.batch_size,
        "global_batch": args.batch_size * 6, "training_gpus": [1, 2, 3, 4, 5, 7],
        "evaluation_gpu": 0, "evaluation_routes": 128, "evaluation_every": 500,
        "steps": args.steps, "tensorboard_port_in_container": args.tb_port,
        "evaluations": {}}
    evaluator = None
    tb = None
    current_step = None
    best_sr = -1.
    waiting_until = 0.
    while True:
        status["heartbeat_unix"] = time.time()
        status["training_exit_code"] = trainer.poll()
        if run_dir.is_dir() and tb is None:
            command = docker_prefix("lidar-wam-train", gpu="") + ["python",
                str(PROJECT / "aliyun_lidar_wam/lidar_WAM/scripts/run_tensorboard_compatible.py"),
                "--logdir", str(container_path(run_dir / "tensorboard", root)),
                "--host", "0.0.0.0", "--port", str(args.tb_port), "--reload_interval", "10"]
            tb = spawn(command, job / "tensorboard.log")
            status["tensorboard_pid"] = tb.pid
        if tb is not None:
            status["tensorboard_exit_code"] = tb.poll()
        if evaluator is not None and evaluator.poll() is not None:
            info = status["evaluations"][str(current_step)]
            folder = job / "closed_loop" / f"step_{current_step:06d}"
            metrics_path = folder / "metrics.json"
            try:
                metrics = json.loads(metrics_path.read_text())
                if int(metrics["checkpoint_step"]) != current_step:
                    raise ValueError("evaluation checkpoint step mismatch")
                if sum(metrics["counts"].values()) != 128:
                    raise ValueError("evaluation route count mismatch")
                if status.get("initial_conditions_sha256", metrics["initial_conditions_sha256"]) != metrics["initial_conditions_sha256"]:
                    raise ValueError("evaluation routes/initial poses changed")
                status["initial_conditions_sha256"] = metrics["initial_conditions_sha256"]
                command = docker_prefix("lidar-wam-train", gpu="") + ["python",
                    "training/scripts/run_flow_action_training.py", "record",
                    "--metrics", str(container_path(metrics_path, root)),
                    "--tensorboard", str(container_path(run_dir / "tensorboard", root))]
                subprocess.run(command, check=True, timeout=120)
                info.update(state="complete", counts=metrics["counts"], rates=metrics["rates"],
                            finished_unix=time.time(), result=str(metrics_path))
                if metrics["rates"]["sr"] > best_sr:
                    best_sr = metrics["rates"]["sr"]
                    target = run_dir / "best_closed_loop.pt"
                    temporary = target.with_suffix(".tmp")
                    shutil.copyfile(run_dir / f"step_{current_step}.pt", temporary)
                    temporary.replace(target)
                    write_json(run_dir / "best_closed_loop_metrics.json", metrics)
                    status.update(best_closed_loop_step=current_step, best_closed_loop_sr=best_sr)
                print(json.dumps({"step": current_step, "closed_loop": metrics["counts"]}), flush=True)
            except Exception as error:
                info.update(state="retry" if info["attempts"] < 2 else "failed",
                            error=str(error), exit_code=evaluator.returncode)
                waiting_until = time.time() + 60
                print(f"Evaluation {current_step}: {error}", flush=True)
            evaluator = None
        queued = sorted(int(path.stem.split("_")[-1]) for path in run_dir.glob("step_*.pt"))
        pending = [step for step in queued if status["evaluations"].get(str(step), {}).get("state")
                   not in {"complete", "failed"}]
        status["pending_steps"] = pending
        status["current_evaluation_step"] = current_step if evaluator is not None else None
        if evaluator is None and pending and time.time() >= waiting_until:
            free = gpu_free_mib(0)
            status["evaluation_gpu_free_mib"] = free
            if free >= args.min_eval_free_mib:
                current_step = pending[0]
                folder = job / "closed_loop" / f"step_{current_step:06d}"
                folder.mkdir(parents=True, exist_ok=True)
                command = evaluation_command(run_dir / f"step_{current_step}.pt", folder, root)
                info = status["evaluations"].setdefault(str(current_step), {"attempts": 0})
                info.update(state="running", attempts=info["attempts"] + 1,
                            started_unix=time.time(), command=command)
                evaluator = spawn(command, folder / "eval.log")
                info["pid"] = evaluator.pid
                print(f"GPU0: evaluating step {current_step}", flush=True)
        if trainer.returncode is not None and evaluator is None and not pending:
            failures = any(value["state"] == "failed" for value in status["evaluations"].values())
            status["state"] = ("complete" if trainer.returncode == 0 and not failures else "failed")
            status["finished_unix"] = time.time()
            write_json(job / "status.json", status)
            break
        write_json(job / "status.json", status)
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    job = sub.add_parser("run")
    job.add_argument("--host-root", type=Path, default=Path("/home/yimingwei/NavRL"))
    job.add_argument("--run-name", required=True)
    job.add_argument("--batch-size", type=int, required=True)
    job.add_argument("--steps", type=int, default=20_000)
    job.add_argument("--memory-fraction", type=float, default=.16)
    job.add_argument("--min-train-free-mib", type=int, default=4800)
    job.add_argument("--min-eval-free-mib", type=int, default=5000)
    job.add_argument("--ddp-port", type=int, default=29584)
    job.add_argument("--tb-port", type=int, default=6024)
    item = sub.add_parser("record")
    item.add_argument("--metrics", type=Path, required=True)
    item.add_argument("--tensorboard", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "record":
        record(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
