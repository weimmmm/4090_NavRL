"""Compare matched 128-route SFT runs, retaining per-route gains AND losses."""
import argparse
import json
from pathlib import Path

import numpy as np


MATCHED_FIELDS = ("map_sha256", "num_envs", "physics_dt_s", "replan_interval_steps",
                  "max_physics_steps", "action_frame", "action_semantics",
                  "goal_stop_adapter_enabled", "safety_filter_enabled",
                  "bootstrap_forward_speed_mps", "bootstrap_action_chunks")


def load_evaluation(folder):
    metrics = json.loads((folder/"metrics.json").read_text())
    # The existing Isaac logger writes its own outcome strings as an object
    # array. These are trusted, locally generated experiment artifacts, not
    # arbitrary downloaded NPZ inputs. Do not modify/rerun the evaluation.
    with np.load(folder/"trajectories.npz", allow_pickle=True) as value:
        arrays = {key: value[key].copy() for key in
                  ("start_position", "target_position", "route_id", "outcome")}
    if metrics["num_envs"] != 128 or len(arrays["outcome"]) != 128:
        raise ValueError("Require the complete 128-route closed-loop evaluation")
    if not set(arrays["outcome"]).issubset(
            {"success", "timeout", "collision", "out_of_bounds", "goal_brake_failed"}):
        raise ValueError("Unknown trajectory terminal labels")
    if int(np.sum(arrays["outcome"] == "success")) != metrics["counts"]["success"]:
        raise ValueError("Trajectory outcomes disagree with reported successful routes")
    return metrics, arrays


def paired_changes(reference, candidate):
    reference_metrics, reference_arrays = reference
    metrics, arrays = candidate
    for key in MATCHED_FIELDS:
        if reference_metrics[key] != metrics[key]:
            raise ValueError(f"Evaluation setting changed: {key}")
    for key in ("start_position", "target_position", "route_id"):
        if not np.array_equal(reference_arrays[key], arrays[key]):
            raise ValueError(f"Not the same routes: {key}")
    before = reference_arrays["outcome"] == "success"
    after = arrays["outcome"] == "success"
    return {"sr_gain_percentage_points": float((after.mean()-before.mean())*100),
            "gained_routes": np.flatnonzero(~before & after).tolist(),
            "lost_routes": np.flatnonzero(before & ~after).tolist()}


def summarize(raw_root, projected_root):
    raw_zero = load_evaluation(raw_root/"closed_loop_step_0")
    projected_zero = load_evaluation(projected_root/"closed_loop_step_0")
    initial_changes = paired_changes(raw_zero, projected_zero)
    result = {"baseline": {"rates": raw_zero[0]["rates"], "counts": raw_zero[0]["counts"]},
              "second_baseline": {"rates": projected_zero[0]["rates"],
                                  "counts": projected_zero[0]["counts"], **initial_changes},
              "metric_scope": "same fixed 128 routes; not an unseen-map generalization claim",
              "runs": {}}
    models = {}
    for mode, root in (("raw", raw_root), ("student-norm", projected_root)):
        launch = json.loads((root/"launch_config.json").read_text())
        if launch["start_step"] != 0 or launch["teacher_target_mode"] != mode:
            raise ValueError("Ablation must begin from the original, non-SFT expert")
        entries = []
        for folder in sorted(root.glob("closed_loop_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])):
            current = load_evaluation(folder)
            step = current[0]["checkpoint_step"]
            models[mode, step] = current
            entries.append({"step": step, "rates": current[0]["rates"], "counts": current[0]["counts"],
                            **paired_changes(raw_zero, current)})
        result["runs"][mode] = entries
    raw_launch = json.loads((raw_root/"launch_config.json").read_text())
    norm_launch = json.loads((projected_root/"launch_config.json").read_text())
    for key in ("checkpoint", "updates", "eval_every", "num_envs", "per_gpu_batch_size",
                "map_seed_base", "route_seed_base", "training_gpus", "fixed_cache", "fixed_windows"):
        if raw_launch[key] != norm_launch[key]:
            raise ValueError(f"Training settings not matched: {key}")
    steps = sorted(set(step for mode, step in models if mode == "raw")
                   & set(step for mode, step in models if mode == "student-norm"))
    result["projected_vs_raw_at_same_update"] = [
        {"step": step, **paired_changes(models["raw", step], models["student-norm", step])}
        for step in steps]
    return result


def include_anchor(result, raw_root, projected_root, anchor_root):
    """Check that the conservative pilot changes only the current-state prior."""
    launch = json.loads((anchor_root/"launch_config.json").read_text())
    reference_launch = json.loads((projected_root/"launch_config.json").read_text())
    for key in ("checkpoint", "updates", "eval_every", "num_envs", "per_gpu_batch_size",
                "map_seed_base", "route_seed_base", "training_gpus", "fixed_cache", "fixed_windows",
                "teacher_target_mode", "start_step"):
        if key == "checkpoint":
            # The CLI can omit the checkpoint and use the original default.
            # Compare the learner's RESOLVED source, not None vs a path string.
            value = json.loads((anchor_root/"config.json").read_text())[key]
            reference_value = json.loads((projected_root/"config.json").read_text())[key]
        else:
            value, reference_value = launch[key], reference_launch[key]
        if value != reference_value:
            raise ValueError(f"Anchored pilot changed another training setting: {key}")
    if launch.get("anchor_weight", 0.) <= 0:
        raise ValueError("No conservative constraint in the anchored pilot")
    baseline = load_evaluation(raw_root/"closed_loop_step_0")
    entries, paired = [], []
    for step in launch["evaluation_steps"]:
        folder = anchor_root/f"closed_loop_step_{step}"
        current = load_evaluation(folder)  # Never accept missing/partial results.
        entries.append({"step": step, "rates": current[0]["rates"], "counts": current[0]["counts"],
                        **paired_changes(baseline, current)})
        matched = load_evaluation(projected_root/f"closed_loop_step_{step}")
        paired.append({"step": step, **paired_changes(matched, current)})
    result["runs"]["student-norm-current-anchor"] = entries
    result["anchor_weight"] = launch["anchor_weight"]
    result["anchored_vs_projected_at_same_update"] = paired
    return result


def include_full_chunk(result, raw_root, projected_root, chunk_root):
    launch = json.loads((chunk_root/"launch_config.json").read_text())
    control = json.loads((projected_root/"launch_config.json").read_text())
    if launch["supervision_horizon"] != 10 or launch.get("anchor_weight", 0.) != 0.:
        raise ValueError("Require the isolated full-chunk, no-anchor ablation")
    source = json.loads((chunk_root/"config.json").read_text())["checkpoint"]
    original = json.loads((projected_root/"config.json").read_text())["checkpoint"]
    if source != original:
        raise ValueError("Full-chunk trial did not start from the same original expert")
    for key in ("updates", "eval_every", "num_envs", "per_gpu_batch_size", "map_seed_base",
                "route_seed_base", "training_gpus", "fixed_cache", "fixed_windows", "teacher_target_mode"):
        if launch[key] != control[key]:
            raise ValueError(f"Unexpected full-chunk training difference: {key}")
    baseline = load_evaluation(raw_root/"closed_loop_step_0")
    entries, changes = [], []
    for step in launch["evaluation_steps"]:
        current = load_evaluation(chunk_root/f"closed_loop_step_{step}")
        entries.append({"step": step, "rates": current[0]["rates"], "counts": current[0]["counts"],
                        **paired_changes(baseline, current)})
        matched = load_evaluation(projected_root/f"closed_loop_step_{step}")
        changes.append({"step": step, **paired_changes(matched, current)})
    result["runs"]["student-norm-full-chunk"] = entries
    result["full_chunk_vs_first_only_at_same_update"] = changes
    result["full_chunk_label_semantics"] = (
        "PPO corrections at student-visited states, represented in chunk-start yaw; "
        "not a counterfactual PPO rollout; future observations are labels only")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--projected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchored", type=Path)
    parser.add_argument("--full-chunk", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.raw, args.projected)
    if args.anchored is not None:
        result = include_anchor(result, args.raw, args.projected, args.anchored)
    if args.full_chunk is not None:
        result = include_full_chunk(result, args.raw, args.projected, args.full_chunk)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
