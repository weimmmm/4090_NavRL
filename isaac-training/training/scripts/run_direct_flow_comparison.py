"""Host-side, resumable GPU0-only Direct/Flow audit; never starts training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-root", type=Path, default=Path("/home/yimingwei/NavRL"))
    parser.add_argument("--eval-container", default="navrl-collect-gpu0")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    project = Path("/workspace/NavRL/isaac-training")
    container_output = Path("/workspace/NavRL") / output.relative_to(args.host_root)
    checkpoint = project / "training/lidar_WAM/outputs/action_dit_final_lee_ego_history1_5m_v3/best_closed_loop.pt"
    scene = project / "aliyun_lidar_wam/datasets/wam_40x40_obs200_350_settled_5m/environments/map40_obs0350.pt"
    status_path = output / "status.json"
    started = time.time()
    status = {"started_unix": started, "gpu": 0, "training_started": False,
              "checkpoint": str(checkpoint), "checkpoint_step": 7500,
              "scene": str(scene), "phases": {}}

    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    def run(phase, command, expected, timeout=1800):
        if expected.is_file():
            status["phases"][phase] = {"state": "already_complete", "result": str(expected)}
            save()
            return
        status["current_phase"] = phase
        status["phases"][phase] = {"state": "running", "command": command,
                                    "started_unix": time.time()}
        save()
        print(f"Starting {phase}", flush=True)
        with (output / f"{phase}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        if result.returncode != 0 or not expected.is_file():
            raise RuntimeError(f"{phase} failed: exit {result.returncode}; inspect {phase}.log")
        status["phases"][phase].update(state="complete", finished_unix=time.time(),
                                       result=str(expected))
        save()

    try:
        run("offline", ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=0",
            "-e", "OMP_NUM_THREADS=4", "-e", "OPENBLAS_NUM_THREADS=1",
            "lidar-wam-train", "python", str(project / "training/scripts/compare_direct_flow.py"),
            "--checkpoint", str(checkpoint), "--output", str(container_output / "offline_metrics.json"),
            "--windows", "4096", "--batch-size", "128", "--workers", "4"],
            output / "offline_metrics.json")
        results = {}
        for mode, steps in (("direct", 10), ("flow10", 10), ("flow20", 20)):
            folder = output / mode
            folder.mkdir(exist_ok=True)
            container_folder = container_output / mode
            command = ["docker", "exec", "-e", "ISAAC_ACTIVE_GPU=0",
                "-e", "NAVRL_ASSETS_ROOT=http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0",
                "-e", "PYTHONUNBUFFERED=1",
                "-e", "OMP_NUM_THREADS=4", "-e", "OPENBLAS_NUM_THREADS=1",
                "-e", "PYTHONPATH=" + str(project / "training/lidar_WAM") + ":" + str(
                    project / "aliyun_lidar_wam/lidar_WAM/third_party/diffusers/src"),
                "-w", str(project), args.eval_container,
                "timeout", "--signal=TERM", "--kill-after=15s", "900",
                "/isaac-sim/python.sh", "training/scripts/wam_closed_loop_eval.py",
                "device=cuda:0", "headless=true", "seed=0", "env.num_envs=128",
                "env.max_episode_length=2500", "env_dyn.num_obstacles=0",
                "env.scene_path=" + str(scene), "env.fixed_route_start_index=0",
                "env.success_distance=0.5", "env.success_speed=0.2", "env.success_stop_steps=10",
                "wam_eval.checkpoint=" + str(checkpoint),
                "wam_eval.action_generation=" + ("direct" if mode == "direct" else "flow"),
                f"wam_eval.sample_steps={steps}", "wam_eval.action_seed=271828",
                "wam_eval.action_noise_scale=1.0", "wam_eval.use_past_actions=false",
                "wam_eval.bootstrap_forward_speed_mps=1.0", "wam_eval.bootstrap_action_chunks=1",
                "wam_eval.safety_filter.enabled=false", "wam_eval.goal_stop_adapter.enabled=true",
                "wam_eval.goal_stop_adapter.enter_distance_m=0.5",
                "wam_eval.goal_stop_adapter.enter_speed_mps=0.2",
                "wam_eval.metrics_path=" + str(container_folder / "metrics.json"),
                "wam_eval.trajectory_path=" + str(container_folder / "trajectories.npz")]
            run(mode, command, folder / "metrics.json")
            results[mode] = json.loads((folder / "metrics.json").read_text())
        if len({value["map_sha256"] for value in results.values()}) != 1:
            raise RuntimeError("scene maps did not match")
        if len({value["initial_conditions_sha256"] for value in results.values()}) != 1:
            raise RuntimeError("initial routes/poses did not match")
        offline = json.loads((output / "offline_metrics.json").read_text())
        digest = hashlib.sha256()
        with (args.host_root / checkpoint.relative_to("/workspace/NavRL")).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != offline["checkpoint_sha256"]:
            raise RuntimeError("checkpoint weights changed during the comparison")
        combined = {"checkpoint": str(checkpoint), "checkpoint_step": 7500,
                    "map_sha256": results["direct"]["map_sha256"],
                    "initial_conditions_sha256": results["direct"]["initial_conditions_sha256"],
                    "routes": 128, "max_physics_steps": 2500,
                    "offline": offline, "closed_loop": results,
                    "elapsed_s": time.time() - started}
        (output / "summary_metrics.json").write_text(json.dumps(combined, indent=2) + "\n")
        status.update(state="complete", current_phase="complete", finished_unix=time.time())
        save()
        print(json.dumps({name: value["counts"] for name, value in results.items()}), flush=True)
    except BaseException as exc:
        status.update(state="failed", error=str(exc), finished_unix=time.time())
        save()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
