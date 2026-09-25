from isaac_eval.timeout_diagnostics import classify_timeout, diagnostic_summary


def _row(initial, closest, final, path=40.0, reason="timeout"):
    return {
        "termination_reason": reason,
        "initial_goal_distance_m": initial,
        "min_goal_distance_m": closest,
        "final_goal_distance_m": final,
        "path_length_m": path,
        "net_goal_progress_m": initial-final,
        "mean_goalward_velocity_mps": 0.1,
        "negative_goalward_velocity_fraction": 0.2,
        "plan_goal_direction_reversals": 3,
    }


def test_timeout_classes():
    assert classify_timeout(_row(30, 0.8, 4)) == "near_goal_overshoot_or_oscillation"
    assert classify_timeout(_row(30, 8, 15)) == "approached_then_retreated"
    assert classify_timeout(_row(30, 25, 29)) == "moving_without_goal_progress"
    assert classify_timeout(_row(30, 8, 9)) == "slow_progress_or_long_detour"
    assert classify_timeout(_row(30, 1, 1, reason="reach_goal")) is None


def test_summary_counts_only_timeouts():
    rows = [_row(30, 0.8, 4), _row(30, 25, 29),
            _row(30, 1, 1, reason="reach_goal")]
    for row in rows:
        row["timeout_class"] = classify_timeout(row)
    result = diagnostic_summary(rows)
    assert result["timeout_count"] == 2
    assert result["timeout_class_counts"][
        "near_goal_overshoot_or_oscillation"] == 1
    assert result["timeout_class_counts"]["moving_without_goal_progress"] == 1
