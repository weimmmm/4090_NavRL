"""Aggregate fixed-route candidate evaluations and write the action-only gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-success-rate", type=float, default=0.50)
    parser.add_argument("--max-collision-rate", type=float, default=0.50)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        value = json.loads(path.expanduser().resolve().read_text())
        summary = value["summary"]
        if int(summary["episodes"]) != 128 or int(summary.get("route_limit") or 0) != 128:
            raise ValueError(f"{path}: candidate gate requires exactly 128 fixed routes")
        rows.append({
            "path": str(path), "checkpoint": summary["checkpoint"],
            "policy_seed": int(summary["policy_seed"]),
            "environment_sha256": summary["environment_sha256"],
            "success_rate": float(summary["success_rate"]),
            "collision_rate": float(summary["collision_rate"]),
            "timeout_rate": float(summary["timeout_rate"]),
            "out_of_bounds_rate": float(summary["out_of_bounds_rate"]),
        })
    environments = {row["environment_sha256"] for row in rows}
    if len(environments) != 1:
        raise ValueError("candidate evaluations do not use the same fixed environment")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["checkpoint"], []).append(row)
    candidates = []
    for checkpoint, group in grouped.items():
        seeds = {row["policy_seed"] for row in group}
        if seeds != {42, 43, 44}:
            raise ValueError(
                f"{checkpoint}: expected policy seeds 42,43,44, found {sorted(seeds)}")
        candidates.append({
            "checkpoint": checkpoint,
            "mean_success_rate": sum(row["success_rate"] for row in group)/3,
            "mean_collision_rate": sum(row["collision_rate"] for row in group)/3,
            "mean_timeout_rate": sum(row["timeout_rate"] for row in group)/3,
            "mean_out_of_bounds_rate": sum(row["out_of_bounds_rate"] for row in group)/3,
            "runs": group,
        })
    candidates.sort(key=lambda row: (
        -row["mean_success_rate"], row["mean_collision_rate"],
        row["mean_timeout_rate"], row["mean_out_of_bounds_rate"]))
    best = candidates[0]
    result = {
        "format": "navrl-action-closed-loop-gate-v2",
        "summary": {**{key: value for key, value in best.items() if key != "runs"},
                    "passed": (best["mean_success_rate"] >= args.min_success_rate
                               and best["mean_collision_rate"] <= args.max_collision_rate),
                    "dagger_required": best["mean_success_rate"] < 0.50,
                    "criteria": {
                        "min_success_rate": args.min_success_rate,
                        "max_collision_rate": args.max_collision_rate,
                    }},
        "environment_sha256": next(iter(environments)),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2), flush=True)
    if not result["summary"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
