"""Two matched online SFT pilots from the ORIGINAL imported Action Expert.

Raw PPO labels are the control; projected labels are the only objective
change. Both use 512 online drones, the same training map/routes, four learner
GPUs, unchanged K=10 and separately running complete 128-route evaluations.
No old trajectory samples enter training. All previous runs remain intact.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

from online_sft_jobs import atomic_json, docker_command, read_json, run_logged
from run_action_dagger_sft import ROOT, HOST, PROJECT, TRAIN, PY, DEPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="action_sft_target_ablation_original_20261014")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--summarize-only", action="store_true",
                        help="Recover report generation without restarting any finished training")
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT/"outputs/action_dit_final_lee_ego_history1_5m_v3/best_closed_loop.pt")
    args = parser.parse_args()
    if args.updates <= 0 or args.every <= 0 or Path(args.name).name != args.name:
        parser.error("Positive update/interval counts and a path-free name are required")
    host_root = HOST/"training/lidar_WAM/outputs"/args.name
    container_root = PROJECT/"outputs"/args.name
    if args.summarize_only:
        state = read_json(host_root/"status.json")
        if not state or not state.get("runs"):
            raise FileNotFoundError("No completed ablation to summarize")
        for mode in ("raw", "student-norm"):
            run_dir = HOST/Path(state["runs"][mode]).relative_to(ROOT)
            if read_json(run_dir/"status.json").get("stage") != "complete":
                raise RuntimeError(f"{mode} has not completed training AND evaluation")
        finish_report(state, host_root, container_root)
        return
    if host_root.exists():
        raise FileExistsError("Previous experiments cannot be overwritten")
    host_checkpoint = HOST/args.checkpoint.relative_to(ROOT)
    source_sha = hashlib.sha256(host_checkpoint.read_bytes()).hexdigest()
    # Inspect the SOURCE before launching anything; never resume an SFT model.
    validate = ("import torch,sys,json; d=torch.load(sys.argv[1],map_location='cpu'); "
                "assert not d.get('online_sft') and not d.get('closed_loop_sft'), 'Use the original expert'; "
                "assert d['architecture']['history_lidar_frames']==1; "
                "assert not d['architecture']['deployment_action_diffusion']; "
                "print(json.dumps({'source_pretraining_step':d['step'],'vae':d['vae']}))")
    host_root.mkdir(parents=True)
    run_logged(docker_command(TRAIN, [PY, "-c", validate, args.checkpoint], {}, ROOT),
               host_root/"source_validation.log", timeout=60)
    state = {"stage": "starting", "checkpoint": str(args.checkpoint), "source_sha256": source_sha,
             "updates_per_run": args.updates, "original_weights_preserved": True,
             "only_objective_difference": "raw versus unit-speed-ball teacher labels", "runs": {}}
    atomic_json(host_root/"status.json", state)
    for mode, suffix in (("raw", "raw"), ("student-norm", "projected")):
        name = f"{args.name}_{suffix}"
        state.update(stage="running", current_mode=mode)
        state["runs"][mode] = str(PROJECT/"outputs"/name)
        atomic_json(host_root/"status.json", state)
        command = [sys.executable, HOST/"training/scripts/run_action_online_sft.py",
                   "--run-name", name, "--checkpoint", args.checkpoint, "--start-step", "0",
                   "--teacher-target-mode", mode, "--updates", str(args.updates),
                   "--eval-every", str(args.every), "--evaluate-start", "--baseline-sr", ".8125",
                   "--num-envs", "512", "--per-gpu-batch-size", "128",
                   "--fixed-windows", "4096", "--map-seed-base", "8500000", "--route-seed-base", "9500000"]
        run_logged(list(map(str, command)), host_root/f"{suffix}_supervisor.log")
        if read_json(HOST/"training/lidar_WAM/outputs"/name/"status.json").get("stage") != "complete":
            raise RuntimeError(f"{mode} did not finish all training and evaluations")
    finish_report(state, host_root, container_root)


def finish_report(state, host_root, container_root):
    command = [PY, "training/scripts/summarize_sft_target_ablation.py",
               "--raw", state["runs"]["raw"], "--projected", state["runs"]["student-norm"],
               "--output", container_root/"comparison.json"]
    run_logged(docker_command(TRAIN, command, {"PYTHONPATH": f"{PROJECT}:{DEPS}"}, ROOT),
               host_root/"comparison.log", timeout=120)
    host_checkpoint = HOST/Path(state["checkpoint"]).relative_to(ROOT)
    if hashlib.sha256(host_checkpoint.read_bytes()).hexdigest() != state["source_sha256"]:
        raise RuntimeError("Original expert weights changed during the experiment")
    state.update(stage="complete", comparison=str(container_root/"comparison.json"))
    atomic_json(host_root/"status.json", state)


if __name__ == "__main__":
    main()
