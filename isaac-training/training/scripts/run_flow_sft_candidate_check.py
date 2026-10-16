"""Independent paired checks of a frozen SFT candidate, not training feedback.

Repeat routes 0..127 and use previously unselected routes 128..255. No
controller, terminal rule, noise setting, map or student parameter is changed.
All GPU0 checks share the existing evaluation lock and wait for free memory.
"""
import argparse
import fcntl
import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np

from monitor_online_sft import record_closed_loop
from online_sft_jobs import atomic_json, read_json, run_logged
from run_action_dagger_sft import HOST
from run_flow_action_training import evaluation_command, container_path, gpu_free_mib


def paired_outcomes(base, candidate):
    for key in ("source_route_index", "start_position", "target_position"):
        if not np.array_equal(base[key], candidate[key]):
            raise ValueError(f"Paired routes differ: {key}")
    a, b = base["outcome"], candidate["outcome"]
    if a.shape != (128,) or b.shape != (128,):
        raise ValueError("Require exactly 128 completed routes per policy")
    allowed = {"success", "collision", "out_of_bounds", "goal_brake_failed", "timeout"}
    if not set(a).issubset(allowed) or not set(b).issubset(allowed):
        raise ValueError("Incomplete/unrecognized terminal outcomes")
    wins = base["source_route_index"][(a != "success") & (b == "success")].tolist()
    losses = base["source_route_index"][(a == "success") & (b != "success")].tolist()
    n = len(wins)+len(losses)
    # An auxiliary paired statistic, not a claim that a selection set is an
    # unbiased final test. The held-out result must be read separately.
    p = sum(math.comb(n, k) for k in range(len(wins), n+1))/2**n if n else 1.
    return dict(base_success=int((a == "success").sum()),
                candidate_success=int((b == "success").sum()),
                repaired_routes=wins, regressed_routes=losses,
                delta_success=len(wins)-len(losses), one_sided_paired_p_value=p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    source = HOST/"training/lidar_WAM/outputs/action_flow_only_history1_axis2_b128_6gpu_20261015/step_57500.pt"
    candidate = args.run/f"sft_step_{args.step}.pt"
    if not candidate.is_file() or args.step <= 0:
        raise ValueError("Require a published immutable trained snapshot")
    out = args.run/f"verify_step_{args.step}_holdout128"
    out.mkdir(exist_ok=False)
    invariants = read_json(args.run/"protected_logic_sha256.json")
    if not invariants:
        raise ValueError("Missing protected-flight-logic manifest")
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, candidate)}
    state = dict(stage="starting", pid=os.getpid(), candidate_step=args.step,
                 checkpoints_sha256=hashes, completed=[], control_changes=False)
    atomic_json(out/"status.json", state)
    baseline_metrics = read_json(args.run/"closed_loop_step_0/metrics.json")
    metrics_by_job = {}
    tb = HOST/"training/lidar_WAM/outputs/action_flow57500_local5_sft_tail05_lr1e6_v1/tensorboard"
    try:
        for split, offset in (("repeat", 0), ("heldout", 128)):
            for label, checkpoint in (("base", source), ("candidate", candidate)):
                name = f"{split}_{label}"
                folder = out/name
                folder.mkdir()
                state.update(stage="waiting_for_gpu0", current=name)
                atomic_json(out/"status.json", state)
                with (HOST.parent/".runtime/sft_gpu0_eval.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    while gpu_free_mib(0) < 6500:
                        time.sleep(5)
                    for relative, expected in invariants.items():
                        if hashlib.sha256((HOST/relative).read_bytes()).hexdigest() != expected:
                            raise RuntimeError(f"Protected logic changed: {relative}")
                    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != hashes[str(checkpoint)]:
                        raise RuntimeError("An immutable verification checkpoint changed")
                    command = evaluation_command(checkpoint, folder, HOST.parent, routes=128)
                    command[command.index("env.fixed_route_start_index=0")] = f"env.fixed_route_start_index={offset}"
                    atomic_json(folder/"command.json", command)
                    state.update(stage="running")
                    atomic_json(out/"status.json", state)
                    run_logged(command, folder/"worker.log", timeout=1860)
                    metrics = read_json(folder/"metrics.json")
                    if (metrics.get("num_envs") != 128 or sum(metrics["counts"].values()) != 128
                            or metrics.get("map_sha256") != baseline_metrics["map_sha256"]
                            or metrics.get("fixed_route_start_index") != offset
                            or metrics.get("checkpoint_step") != (57500 if label == "base" else args.step)):
                        raise RuntimeError("Verification identity/terminal contract differs")
                    metrics_by_job[name] = metrics
                    record_closed_loop(container_path(folder/"metrics.json", HOST.parent),
                                       container_path(tb/f"failure_gate_check_{args.step}_{name}", HOST.parent),
                                       0 if label == "base" else args.step, folder/"tensorboard.log")
                state["completed"].append(name)
                atomic_json(out/"status.json", state)
            a, b = metrics_by_job[f"{split}_base"], metrics_by_job[f"{split}_candidate"]
            if a["initial_conditions_sha256"] != b["initial_conditions_sha256"]:
                raise RuntimeError("Base and candidate initial states differ")
            # These NPZs are generated by this evaluator, not untrusted input.
            with np.load(out/f"{split}_base/trajectories.npz", allow_pickle=True) as old:
                with np.load(out/f"{split}_candidate/trajectories.npz", allow_pickle=True) as new:
                    pair = paired_outcomes(old, new)
            atomic_json(out/f"{split}_comparison.json", pair)
        comparison = {split: read_json(out/f"{split}_comparison.json") for split in ("repeat", "heldout")}
        comparison["pooled_256"] = {
            "base_success": sum(comparison[x]["base_success"] for x in ("repeat", "heldout")),
            "candidate_success": sum(comparison[x]["candidate_success"] for x in ("repeat", "heldout")),
            "same_map_unselected_routes_not_new_map": True,
        }
        atomic_json(out/"results.json", comparison)
        state.update(stage="complete", current=None)
        atomic_json(out/"status.json", state)
    except BaseException as error:
        state.update(stage="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(out/"status.json", state)
        raise


if __name__ == "__main__":
    main()
