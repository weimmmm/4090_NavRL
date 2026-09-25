"""Pure helpers for diagnosing closed-loop navigation timeouts."""

from __future__ import annotations

from collections import Counter
from typing import Any


def classify_timeout(row: dict[str, Any]) -> str | None:
    """Assign an interpretable failure mode using only measured trajectory data."""
    if row.get("termination_reason") != "timeout":
        return None
    initial = float(row["initial_goal_distance_m"])
    final = float(row["final_goal_distance_m"])
    closest = float(row["min_goal_distance_m"])
    path = float(row.get("path_length_m", 0.0))

    # The reach threshold is 0.5 m.  One metre deliberately gives a narrow
    # diagnostic band for trajectories which nearly succeeded but crossed or
    # orbited the terminal ball.
    if closest < 1.0:
        return "near_goal_overshoot_or_oscillation"

    improvement = initial - closest
    retreat = final - closest
    if improvement >= max(2.0, 0.20 * initial) and retreat >= max(2.0, 0.10 * initial):
        return "approached_then_retreated"

    net_progress = initial - final
    if path >= 5.0 and net_progress <= max(1.0, 0.10 * initial):
        return "moving_without_goal_progress"
    return "slow_progress_or_long_detour"


def diagnostic_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the added diagnostics, with a timeout-only breakdown."""
    timeout_rows = [row for row in episodes
                    if row.get("termination_reason") == "timeout"]
    classes = Counter(row.get("timeout_class") for row in timeout_rows)

    def mean(rows, key):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "timeout_count": len(timeout_rows),
        "timeout_class_counts": {
            key: int(classes.get(key, 0)) for key in (
                "near_goal_overshoot_or_oscillation",
                "approached_then_retreated",
                "moving_without_goal_progress",
                "slow_progress_or_long_detour")
        },
        "timeout_mean_initial_goal_distance_m": mean(
            timeout_rows, "initial_goal_distance_m"),
        "timeout_mean_final_goal_distance_m": mean(
            timeout_rows, "final_goal_distance_m"),
        "timeout_mean_min_goal_distance_m": mean(
            timeout_rows, "min_goal_distance_m"),
        "timeout_mean_net_goal_progress_m": mean(
            timeout_rows, "net_goal_progress_m"),
        "timeout_mean_goalward_velocity_mps": mean(
            timeout_rows, "mean_goalward_velocity_mps"),
        "timeout_mean_negative_goalward_velocity_fraction": mean(
            timeout_rows, "negative_goalward_velocity_fraction"),
        "timeout_mean_plan_direction_reversals": mean(
            timeout_rows, "plan_goal_direction_reversals"),
    }
