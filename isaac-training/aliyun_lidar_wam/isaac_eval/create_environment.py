"""Create a fixed .pt environment for repeatable Isaac navigation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaac_eval.environment_file import generate_environment, save_environment


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "isaac_eval" / "environments" / "static350_seed18_n256.pt")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--terrain-seed", type=int, default=18)
    parser.add_argument("--route-seed", type=int, default=18)
    parser.add_argument("--static-obstacles", type=int, default=350)
    parser.add_argument("--max-steps", type=int, default=2200)
    args = parser.parse_args()
    value = generate_environment(
        num_envs=args.num_envs,
        terrain_seed=args.terrain_seed,
        route_seed=args.route_seed,
        static_obstacles=args.static_obstacles,
        max_steps=args.max_steps)
    output = args.output.expanduser().resolve()
    save_environment(output, value)
    same_side = int((value["start_sides"] == value["target_sides"]).sum())
    opposite_side = int(((value["start_sides"] ^ 1) == value["target_sides"]).sum())
    print(json.dumps({
        "output": str(output),
        "format": value["format"],
        "num_envs": value["num_envs"],
        "terrain_seed": value["terrain_seed"],
        "route_seed": value["route_seed"],
        "static_obstacles": value["static_obstacles"],
        "same_side_routes": same_side,
        "opposite_side_routes": opposite_side,
        "adjacent_side_routes": value["num_envs"] - same_side - opposite_side,
        "size_bytes": output.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
