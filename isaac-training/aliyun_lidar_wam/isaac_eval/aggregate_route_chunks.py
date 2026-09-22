"""Aggregate four deterministic 128-route runs from one 512-route snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-routes", type=int, default=512)
    args = parser.parse_args()

    payloads = [json.loads(path.expanduser().resolve().read_text())
                for path in args.inputs]
    if not payloads:
        parser.error("at least one input is required")
    identity_keys = (
        "checkpoint", "checkpoint_step", "environment_sha256", "terrain_seed",
        "policy_seed", "flow_steps")
    reference = payloads[0]["summary"]
    for value in payloads[1:]:
        summary = value["summary"]
        mismatch = [key for key in identity_keys
                    if summary.get(key) != reference.get(key)]
        if mismatch:
            raise ValueError(f"chunk identity mismatch: {mismatch}")

    episodes = [row for value in payloads for row in value["episode_results"]]
    route_ids = [int(row["route_id"]) for row in episodes]
    expected = list(range(args.expected_routes))
    if sorted(route_ids) != expected:
        raise ValueError(
            "route chunks must contain every source route exactly once; "
            f"got {len(route_ids)} rows and {len(set(route_ids))} unique IDs")

    reasons = ("reach_goal", "collision", "out_of_bounds", "timeout")
    counts = {reason: sum(row["termination_reason"] == reason for row in episodes)
              for reason in reasons}
    count = len(episodes)
    result = {
        "format": "navrl-fixed-route-aggregate-v2",
        "summary": {
            **{key: reference.get(key) for key in identity_keys},
            "episodes": count,
            **{f"{reason}_rate": counts[reason]/count for reason in reasons},
            "success_rate": counts["reach_goal"]/count,
            "termination_counts": counts,
            "mean_episode_steps": sum(row["steps"] for row in episodes)/count,
            "mean_path_length_m": sum(row["path_length_m"] for row in episodes)/count,
            "mean_min_clearance_m": sum(row["min_clearance_m"] for row in episodes)/count,
            "chunk_files": [str(path.expanduser().resolve()) for path in args.inputs],
        },
        "episode_results": sorted(episodes, key=lambda row: int(row["route_id"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
