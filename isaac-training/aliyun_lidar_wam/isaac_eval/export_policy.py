"""Strip the world model and optimizer from a joint Action Expert checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPACT_FORMAT = "navrl-action-expert-policy-v2"


def torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def export_compact_checkpoint(source: Path, destination: Path,
                              allow_legacy_body: bool = False):
    import torch

    payload = torch_load(source)
    if not isinstance(payload, dict) or "model" not in payload or "stats" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {source}")
    if "semantics" not in payload and not allow_legacy_body:
        raise ValueError(
            "Legacy checkpoint has no coordinate metadata; pass --allow-legacy-body "
            "only to export it for historical comparison")
    prefixes = ("observation.", "action_expert.")
    selected = {key: value for key, value in payload["model"].items()
                if key.startswith(prefixes)}
    if not selected:
        raise ValueError("Checkpoint contains no Action Expert policy parameters")
    compact = {
        "format": COMPACT_FORMAT,
        "model": selected,
        "stats": payload["stats"],
        "step": int(payload.get("step", -1)),
        "world_initial_step": int(payload.get("world_initial_step", -1)),
        "source_checkpoint": str(source),
        "architecture": payload.get("architecture", {
            "action_horizon": 30, "action_dim": 3, "width": 512,
            "depth": 8, "heads": 8, "ffn_width": 2048,
        }),
        "semantics": payload.get("semantics", {
            "condition_frame": "body", "action_frame": "fixed_start_to_target_goal_frame",
            "lidar_frame": "sensor_yaw", "legacy": True,
        }),
        "provenance": payload.get("provenance", {}),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, destination)
    return {
        "source": str(source), "destination": str(destination),
        "step": compact["step"], "parameter_tensors": len(selected),
        "size_bytes": destination.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "lidar_WAM" / "outputs" / "action_expert_joint" / "best.pt")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "lidar_WAM" / "outputs" / "action_expert_joint" / "policy.pt")
    parser.add_argument("--allow-legacy-body", action="store_true")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")
    if source == output:
        parser.error("--source and --output must differ")
    report = export_compact_checkpoint(
        source, output, allow_legacy_body=args.allow_legacy_body)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
