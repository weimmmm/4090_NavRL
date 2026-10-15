"""Matched, independent 500-update selective-SFT threshold experiments.

Run on the 4090 host. Every variant starts from the ORIGINAL step-7500
expert, uses the same fresh-state online seeds and frozen output head, and
executes K=10. Only the original/PPO disagreement threshold changes.
GPU0/6 are deliberately left untouched. No offline validation GPU is used;
the requested endpoint is a complete, independent 128-route closed loop.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from online_sft_jobs import atomic_json, docker_command, read_json, run_logged
from run_action_dagger_sft import ROOT, HOST, PROJECT, TRAIN, PY


def threshold_values(raw):
    values = [float(part) for part in raw.split(",")]
    if (not values or any(not math.isfinite(x) or x <= 0 for x in values)
            or len(set(values)) != len(values)):
        raise ValueError("Thresholds must be distinct, finite, positive m/s values")
    return values


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_for(name, threshold, checkpoint, updates, evaluate_start=False, smoke=False):
    command = [sys.executable, HOST/"training/scripts/run_action_online_sft.py",
               "--run-name", name, "--checkpoint", checkpoint,
               "--teacher-target-mode", "student-norm", "--anchor-weight", "0",
               "--supervision-horizon", "1", "--freeze-policy-output-head",
               "--selective-correction-threshold-mps", str(threshold),
               "--start-step", "0", "--updates", str(updates), "--eval-every", str(updates),
               "--num-envs", "512", "--per-gpu-batch-size", "128",
               "--map-seed-base", "8500000", "--route-seed-base", "9500000",
               "--training-gpus", "1,2,5,7", "--sim-gpu", "4", "--eval-gpu", "3",
               "--sim-container", "navrl-collect-gpu4", "--eval-container", "navrl-collect-gpu3",
               "--skip-fixed-validation", "--baseline-sr", ".8125"]
    if evaluate_start:
        command.append("--evaluate-start")
    if smoke:
        command += ["--smoke", "--smoke-updates", "8", "--smoke-num-envs", "32",
                    "--smoke-eval-num-envs", "16"]
    return list(map(str, command))


def endpoint(run_dir, step):
    data = read_json(run_dir/f"closed_loop_step_{step}"/"metrics.json")
    if int(data.get("checkpoint_step", -1)) != step:
        raise RuntimeError(f"Missing or mismatched evaluation step: {run_dir}")
    if sum(data["counts"].values()) != 128:
        raise RuntimeError(f"Expected all 128 trajectories: {run_dir}")
    return {"step": step, "counts": data["counts"], "rates": data["rates"],
            "metrics_path": str(run_dir/f"closed_loop_step_{step}"/"metrics.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="action_sft_threshold_ablation_20261015")
    # Repeat the known working control FIRST, before trying new thresholds.
    parser.add_argument("--thresholds", default="0.3,0.1,0.2,0.4")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    thresholds = threshold_values(args.thresholds)
    if args.updates <= 0 or args.name in ("", ".", "..") or Path(args.name).name != args.name:
        parser.error("Positive update count and a path-free experiment name required")
    root = HOST/"training/lidar_WAM/outputs"/args.name
    checkpoint = PROJECT/"outputs/action_dit_final_lee_ego_history1_5m_v3/best_closed_loop.pt"
    source = HOST/checkpoint.relative_to(ROOT)
    if args.summarize_only:
        state = read_json(root/"status.json")
        if not state.get("runs"):
            raise FileNotFoundError("No existing threshold study")
        finish(root, state, source)
        return
    if root.exists():
        raise FileExistsError("Previous experiments are never overwritten")
    root.mkdir(parents=True)
    scene = HOST/"training/datasets/wam_40x40_obs200_350_settled_5m/environments/map40_obs0350.pt"
    state = {"stage": "validating_source", "source_checkpoint": str(checkpoint),
             "source_sha256": sha256(source), "scene_sha256": sha256(scene),
             "updates_per_run": args.updates, "thresholds_mps": thresholds,
             "training_map_seed": 8500000, "training_route_seed": 9500000,
             "training_gpus": [1, 2, 5, 7], "simulation_gpu": 4, "evaluation_gpu": 3,
             "untouched_gpus": [0, 6], "execution_k": 10, "max_physics_steps": 2500,
             "only_training_difference": "original/PPO command discrepancy threshold in m/s",
             "uses_offline_replay": False, "offline_validation_skipped": True,
             "comparison_endpoint": "step 500, not best-selected intermediate checkpoint",
             "runs": {}}
    atomic_json(root/"status.json", state)
    validation = ("import torch,sys,json; d=torch.load(sys.argv[1],map_location='cpu'); "
                  "assert d['step']==7500; "
                  "assert not d.get('online_sft') and not d.get('closed_loop_sft'); "
                  "assert d['architecture']['history_lidar_frames']==1; "
                  "assert not d['architecture']['deployment_action_diffusion']; "
                  "print(json.dumps({'step':d['step'],'vae':d['vae'],'architecture':d['architecture']}))")
    try:
        run_logged(docker_command(TRAIN, [PY, "-c", validation, checkpoint], {}, ROOT),
                   root/"source_validation.log", timeout=120)
        state["stage"] = "wiring_smoke"
        atomic_json(root/"status.json", state)
        smoke_name = args.name+"_smoke"
        run_logged(command_for(smoke_name, min(thresholds), checkpoint, 8, smoke=True),
                   root/"smoke_supervisor.log")
        if read_json(root.parent/smoke_name/"status.json").get("stage") != "complete":
            raise RuntimeError("New GPU-role wiring smoke did not pass")
        for index, threshold in enumerate(thresholds):
            name = args.name+"_t"+format(round(threshold*1000), "04d")
            state.update(stage="running", current_threshold_mps=threshold)
            state["runs"][str(threshold)] = str(PROJECT/"outputs"/name)
            atomic_json(root/"status.json", state)
            print(json.dumps({"stage": "launching", "threshold_mps": threshold, "run": name}), flush=True)
            run_logged(command_for(name, threshold, checkpoint, args.updates, evaluate_start=index == 0),
                       root/f"threshold_{threshold}_supervisor.log")
            folder = root.parent/name
            if read_json(folder/"status.json").get("stage") != "complete":
                raise RuntimeError(f"Threshold {threshold} did not finish training AND evaluation")
            if index == 0:
                state["baseline"] = endpoint(folder, 0)
                if state["baseline"]["counts"]["success"] != 104:
                    raise RuntimeError("Original 104/128 baseline did not reproduce; stop before comparing thresholds")
            state.setdefault("completed_results", {})[str(threshold)] = endpoint(folder, args.updates)
            atomic_json(root/"status.json", state)
        finish(root, state, source)
    except BaseException as exc:
        state.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(root/"status.json", state)
        raise


def finish(root, state, source):
    rows = []
    for threshold, value in state["runs"].items():
        folder = HOST/Path(value).relative_to(ROOT)
        if read_json(folder/"status.json").get("stage") != "complete":
            raise RuntimeError(f"Threshold {threshold} remains incomplete")
        config = read_json(folder/"config.json")
        if (config["selective_correction_threshold_mps"] != float(threshold)
                or not config["freeze_policy_output_head"] or config["lr"] != 5e-6
                or config["uses_offline_replay"] or len(config["trainable_names"]) != 36):
            raise RuntimeError(f"Unexpected fine-tuning configuration: {folder}")
        row = {"threshold_mps": float(threshold), "run": str(folder),
               **endpoint(folder, state["updates_per_run"])}
        progress = read_json(folder/"progress.json")
        row["final_correction_fraction"] = progress.get("selective_correction_fraction")
        rows.append(row)
    if len(rows) != len(state["thresholds_mps"]):
        raise RuntimeError("Not all requested thresholds have completed")
    if sha256(source) != state["source_sha256"]:
        raise RuntimeError("Original expert changed during the experiments")
    comparison = {"baseline": state["baseline"], "results": sorted(rows, key=lambda r: r["threshold_mps"]),
                  "only_training_difference": state["only_training_difference"],
                  "selection_scope": "same 128 routes; not an independent held-out acceptance test",
                  "source_sha256": state["source_sha256"]}
    atomic_json(root/"comparison.json", comparison)
    state.update(stage="complete", comparison=str(root/"comparison.json"))
    atomic_json(root/"status.json", state)
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
