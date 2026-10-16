"""Recheck the SFT label source without changing the deployed student.

The K=1 runs diagnose the teacher's native feedback capability, not the
student's K=10 algorithm. Their results are never promoted as student weights.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import time

from monitor_online_sft import record_closed_loop
from online_sft_jobs import atomic_json, read_json, run_logged
from run_action_dagger_sft import HOST, PROJECT
from run_flow_action_training import evaluation_command, container_path, gpu_free_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing formal SFT host directory")
    args = parser.parse_args()
    out = args.run/"teacher_audit_current_box"
    out.mkdir(exist_ok=False)
    reference = read_json(args.run/"closed_loop_step_0/metrics.json")
    if reference.get("num_envs") != 128 or reference.get("checkpoint_step") != 0:
        raise ValueError("Require the formal run's completed baseline evaluation")
    source = HOST/"training/lidar_WAM/outputs/action_flow_only_history1_axis2_b128_6gpu_20261015/step_57500.pt"
    teacher = PROJECT/"outputs/privileged_ppo_checkpoint_27000.pt"
    invariant = read_json(args.run/"protected_logic_sha256.json")
    results = {}
    status = {"stage": "waiting_for_gpu0_lock", "pid": os.getpid(), "completed": results}
    atomic_json(out/"status.json", status)
    lock = (HOST.parent/".runtime/sft_gpu0_eval.lock").open("a")
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for mode in ("native", "student-box"):
            while gpu_free_mib(0) < 6500:
                time.sleep(5)
            for name, expected in invariant.items():
                if hashlib.sha256((HOST/name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Protected logic changed: {name}")
            name = mode.replace("-", "_")+"_k1"
            folder = out/name
            folder.mkdir()
            status.update(stage="running", current=name)
            atomic_json(out/"status.json", status)
            command = evaluation_command(source, folder, HOST.parent, routes=128)
            entry = command.index("training/scripts/wam_closed_loop_eval.py")
            command[entry] = "training/scripts/evaluate_online_sft_teacher.py"
            command += [f"+wam_eval.teacher_checkpoint={teacher}",
                        f"+wam_eval.teacher_velocity_domain={mode}",
                        "+wam_eval.teacher_query_interval_physics_steps=1"]
            atomic_json(folder/"command.json", command)
            run_logged(command, folder/"worker.log", timeout=1860)
            metrics = read_json(folder/"metrics.json")
            for field in ("map_sha256", "initial_conditions_sha256", "num_envs", "max_physics_steps"):
                if metrics.get(field) != reference[field]:
                    raise RuntimeError(f"Teacher audit identity differs: {field}")
            if (sum(metrics["counts"].values()) != 128
                    or metrics.get("teacher_velocity_domain") != mode
                    or metrics.get("teacher_query_interval_physics_steps") != 1):
                raise RuntimeError("Teacher diagnostic contract differs")
            results[name] = {"counts": metrics["counts"], "rates": metrics["rates"],
                             "teacher_velocity_domain": mode, "teacher_query_interval": 1}
            record_closed_loop(container_path(folder/"metrics.json", HOST.parent),
                               container_path(args.run/"tensorboard"/f"teacher_{name}", HOST.parent),
                               27000, folder/"tensorboard.log")
            atomic_json(out/"results.json", results)
        status.update(stage="complete", current=None)
        atomic_json(out/"status.json", status)


if __name__ == "__main__":
    main()
