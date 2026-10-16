"""Audit recorded routes, controls and paired outcomes after the comparison."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    combined = json.loads((args.output / "summary_metrics.json").read_text())
    payload = torch.load(combined["checkpoint"], map_location="cpu", weights_only=False)
    scene = torch.load(combined["closed_loop"]["direct"]["scene_path"],
                       map_location="cpu", weights_only=False)
    original = None
    results = {}
    for mode, steps in (("direct", 0), ("flow10", 10), ("flow20", 20)):
        metrics = combined["closed_loop"][mode]
        data = np.load(args.output / mode / "trajectories.npz", allow_pickle=True)
        assert metrics["sample_steps"] == steps
        assert metrics["num_envs"] == 128 and metrics["max_physics_steps"] == 2500
        assert metrics["replan_interval_steps"] == 10 and metrics["physics_dt_s"] == .016
        assert metrics["checkpoint_step"] == 7500
        np.testing.assert_array_equal(data["source_route_index"], np.arange(128))
        np.testing.assert_array_equal(data["start_position"], scene["start_position"][:128].numpy())
        np.testing.assert_array_equal(data["target_position"], scene["goal_position"][:128].numpy())
        expected_q = scene["initial_orientation_wxyz"][:128].numpy()
        np.testing.assert_allclose(abs((data["root_state_wxyz"][0, :, 3:7] * expected_q).sum(-1)),
                                   np.ones(128), atol=1e-5, rtol=0)
        outcomes = data["outcome"]
        assert dict(Counter(outcomes)) == {key: value for key, value in metrics["counts"].items() if value}
        commands = data["ego_velocity_command"]
        assert np.isfinite(commands).all()
        max_speed = float(np.linalg.norm(commands, axis=-1).max())
        assert max_speed <= 2.00001
        expected_bootstrap = np.zeros_like(commands[:10])
        expected_bootstrap[..., 0] = 1.
        np.testing.assert_array_equal(commands[:10], expected_bootstrap)
        success = outcomes == "success"
        if original is None:
            original = success
        results[mode] = {
            "counts": metrics["counts"], "rates": metrics["rates"],
            "max_command_norm_mps": max_speed,
            "direct_success_to_failure": np.flatnonzero(original & ~success).tolist(),
            "direct_failure_to_success": np.flatnonzero(~original & success).tolist(),
            "offline": combined["offline"]["metrics"][mode]}
    verified = {
        "checkpoint": combined["checkpoint"], "checkpoint_step": 7500,
        "checkpoint_sha256": combined["offline"]["checkpoint_sha256"],
        "routes_start_goal_orientation_verified": True,
        "identical_initial_conditions_sha256": combined["initial_conditions_sha256"],
        "raw_outcomes_and_counts_verified": True, "bootstrap_and_speed_bound_verified": True,
        "training_action_diffusion": bool(payload["architecture"]["training_action_diffusion"]),
        "flow_out_weight_norm": float(payload["model"]["diffusion.out.weight"].float().norm()),
        "metadata_note": "Training flag is read from checkpoint architecture. Older evaluator logs used an incomplete format whitelist; no outcome or rate is corrected.",
        "results": results}
    target = args.output / "verified_comparison_metrics.json"
    target.write_text(json.dumps(verified, indent=2) + "\n")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
