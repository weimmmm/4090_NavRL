"""Bounded standard-SFT length/gate ablation; no change to the learner or flight.

Three runs start from the same original offline checkpoint. Only the existing
teacher-disagreement gate and snapshot length vary. GPU0 evaluates immutable
snapshots independently. A winner must beat the already verified SFT100, not
merely beat the original offline policy; any winner is then checked on fresh
same-map routes against SFT100 (not a claim of new-map generalization).
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from online_sft_jobs import atomic_json, read_json, run_logged, start_logged
from run_action_dagger_sft import HOST, PROJECT
from run_flow_action_training import container_path, evaluation_command, gpu_free_mib
from run_flow_sft_candidate_check import paired_outcomes
from monitor_online_sft import record_closed_loop


STEPS = (50, 100, 150)


def trial_specs():
    # First replicate the existing gate to isolate the training-length effect.
    return (("gate030", .3), ("gate020", .2), ("gate040", .4))


def verify_metrics(metrics, anchor, step, offset=0):
    if (metrics.get("num_envs") != 128
            or sum(metrics.get("counts", {}).values()) != 128
            or metrics.get("checkpoint_step") != step
            or metrics.get("fixed_route_start_index") != offset
            or metrics.get("map_sha256") != anchor["map_sha256"]):
        raise ValueError("Snapshot/map/routes/terminal counts do not match the experiment")
    if offset == 0 and metrics.get("initial_conditions_sha256") != anchor["initial_conditions_sha256"]:
        raise ValueError("Selection routes differ from the verified SFT100 routes")


def select_candidate(rows, incumbent_success):
    # Strict improvement; ties retain the validated incumbent. Stable order
    # makes the policy explicit and never uses the later holdout to select.
    better = [x for x in rows if x["counts"]["success"] > incumbent_success]
    return max(better, key=lambda x: x["counts"]["success"]) if better else None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confirm(winner, incumbent, out, anchor, invariants, tb, state):
    """Locked winner vs the prior SFT100, routes256..383, exactly one pair."""
    folder = out/"confirmation_routes256_383"
    folder.mkdir()
    checkpoints = {"incumbent_sft100": incumbent, "candidate": Path(winner["checkpoint"])}
    hashes = {name: sha(path) for name, path in checkpoints.items()}
    metrics = {}
    atomic_json(folder/"locked_selection.json", dict(winner=winner,
                reference=str(incumbent), checkpoint_sha256=hashes,
                route_offset=256, note="Same map; these routes do not select the winner"))
    for name, checkpoint in checkpoints.items():
        job = folder/name
        job.mkdir()
        state.update(stage="confirming_on_fresh_routes", confirmation_job=name)
        atomic_json(out/"status.json", state)
        with (HOST.parent/".runtime/sft_gpu0_eval.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while gpu_free_mib(0) < 6500:
                time.sleep(5)
            for relative, expected in invariants.items():
                if sha(HOST/relative) != expected:
                    raise RuntimeError(f"Protected flight logic changed: {relative}")
            if sha(checkpoint) != hashes[name]:
                raise RuntimeError("A locked confirmation checkpoint changed")
            command = evaluation_command(checkpoint, job, HOST.parent, routes=128)
            command[command.index("env.fixed_route_start_index=0")] = "env.fixed_route_start_index=256"
            atomic_json(job/"command.json", command)
            run_logged(command, job/"worker.log", timeout=1860)
            metrics[name] = read_json(job/"metrics.json")
            verify_metrics(metrics[name], anchor, 100 if name == "incumbent_sft100" else winner["step"], 256)
            record_closed_loop(container_path(job/"metrics.json", HOST.parent),
                               tb/f"confirmation_{name}", 0, job/"tensorboard.log")
    if metrics["incumbent_sft100"]["initial_conditions_sha256"] != metrics["candidate"]["initial_conditions_sha256"]:
        raise RuntimeError("Confirmation initial states differ")
    with np.load(folder/"incumbent_sft100/trajectories.npz", allow_pickle=True) as a:
        with np.load(folder/"candidate/trajectories.npz", allow_pickle=True) as b:
            paired = paired_outcomes(a, b)
    result = dict(paired=paired, metrics=metrics, same_map_not_new_map=True,
                  winner_selected_before_these_results=True,
                  confirmed_strict_improvement=paired["delta_success"] > 0)
    atomic_json(folder/"results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        parser.error("Name must be a single directory component")
    outputs = HOST/"training/lidar_WAM/outputs"
    out = outputs/args.name
    out.mkdir(exist_ok=False)
    previous = outputs/"action_flow57500_local5_failure3_lr1e6_v2"
    incumbent = previous/"sft_step_100.pt"
    anchor = read_json(previous/"closed_loop_step_100/metrics.json")
    invariants = read_json(previous/"protected_logic_sha256.json")
    if not invariants or anchor.get("counts", {}).get("success") != 112:
        raise RuntimeError("Verified SFT100 evidence is unavailable")
    for relative, expected in invariants.items():
        if sha(HOST/relative) != expected:
            raise RuntimeError(f"Protected flight logic differs from prior evaluation: {relative}")
    tb = PROJECT/"outputs/action_flow57500_local5_sft_tail05_lr1e6_v1/tensorboard"/args.name
    state = dict(stage="starting", pid=os.getpid(), trial_steps=STEPS,
                 gates_mps=[gate for _, gate in trial_specs()],
                 training_gpus=[2, 3, 4, 5, 7], evaluation_gpu=0,
                 incumbent=str(incumbent), incumbent_sr=112/128,
                 incumbent_sha256=sha(incumbent), runs=[], results=[], started_unix=time.time(),
                 changed_training_algorithm=False, control_changes=False)
    atomic_json(out/"status.json", state)
    evaluators = []
    try:
        for label, threshold in trial_specs():
            name = f"{args.name}_{label}"
            run = outputs/name
            train_command = [sys.executable, "-u", str(HOST/"training/scripts/run_local_flow_online_sft.py"),
                "--run-name", name, "--updates", "150", "--eval-every", "50", "--lr", "1e-6",
                "--threshold-mps", str(threshold), "--failure-window-chunks", "3",
                "--correction-tail-weight", ".5", "--envs-per-worker", "16",
                "--tensorboard", str(tb/label), "--no-evaluation"]
            # Use a separate monitor so the next training run need not wait
            # for GPU0 to drain the preceding immutable-snapshot queue.
            trainer = start_logged(train_command, out/f"{label}_training.log")
            state.update(stage="training_and_independent_evaluation", current_trial=label)
            state["runs"].append(dict(name=name, threshold_mps=threshold,
                                      training_pid=trainer.pid, path=str(run), command=train_command))
            atomic_json(out/"status.json", state)
            while not (run/"launch_config.json").is_file():
                if trainer.poll() is not None:
                    raise RuntimeError(f"{label} launcher failed before initialization")
                time.sleep(1)
            command = [sys.executable, "-u", str(HOST/"training/scripts/monitor_online_sft.py"),
                "--kind", "closed-loop", "--host-out", str(run), "--out", str(PROJECT/"outputs"/name),
                "--start-step", "0", "--final-step", "150", "--every", "50",
                "--gpu", "0", "--container", "navrl-collect-gpu0", "--flow-only",
                "--eval-num-envs", "128", "--baseline-sr", str(112/128),
                "--min-free-mib", "6500", "--timeout", "900", "--tensorboard", str(tb/label)]
            evaluator = start_logged(command, out/f"{label}_evaluation.log")
            evaluators.append((label, threshold, run, evaluator))
            state["runs"][-1].update(evaluator_pid=evaluator.pid, evaluation_command=command)
            atomic_json(out/"status.json", state)
            while trainer.poll() is None:
                state["heartbeat_unix"] = time.time()
                atomic_json(out/"status.json", state)
                time.sleep(5)
            if trainer.returncode != 0 or not read_json(run/"training_finished.json").get("success"):
                raise RuntimeError(f"{label} training failed; inspect its logs")
            if read_json(run/"protected_logic_sha256.json") != invariants:
                raise RuntimeError("Protected-flight manifest differs across trials")
        state.update(stage="draining_all_evaluations", current_trial=None)
        atomic_json(out/"status.json", state)
        for label, threshold, run, process in evaluators:
            if process.wait() != 0:
                raise RuntimeError(f"{label} evaluation failed; no unsupported result will be selected")
            for step in STEPS:
                metric = read_json(run/f"closed_loop_step_{step}/metrics.json")
                verify_metrics(metric, anchor, step)
                state["results"].append(dict(trial=label, threshold_mps=threshold, step=step,
                    counts=metric["counts"], rates=metric["rates"],
                    checkpoint=str(run/f"sft_step_{step}.pt")))
            atomic_json(out/"status.json", state)
        winner = select_candidate(state["results"], 112)
        state["selection_winner"] = winner
        atomic_json(out/"status.json", state)
        confirmation = confirm(winner, incumbent, out, anchor, invariants, tb, state) if winner else None
        if sha(incumbent) != state["incumbent_sha256"]:
            raise RuntimeError("Prior recommended checkpoint changed")
        result = dict(incumbent_counts=anchor["counts"], trials=state["results"],
                      selection_winner=winner, confirmation=confirmation,
                      recommended_checkpoint=(winner["checkpoint"] if confirmation and
                          confirmation["confirmed_strict_improvement"] else str(incumbent)),
                      no_claim_of_statistical_significance=True,
                      same_map_not_new_map=True)
        atomic_json(out/"results.json", result)
        state.update(stage="complete", finished_unix=time.time())
        atomic_json(out/"status.json", state)
    except BaseException as error:
        # Only this sweep's workers receive their existing cooperative stop
        # signal; unrelated GPU processes and previous experiments are untouched.
        for item in state["runs"]:
            run = Path(item["path"])
            if run.is_dir() and not read_json(run/"training_finished.json"):
                (run/"stop_requested").touch()
        state.update(stage="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(out/"status.json", state)
        raise


if __name__ == "__main__":
    main()
