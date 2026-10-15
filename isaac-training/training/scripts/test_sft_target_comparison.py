"""Reject misleading success-rate comparisons with unmatched evaluation routes."""
import copy
import unittest

import numpy as np

from summarize_sft_target_ablation import MATCHED_FIELDS, paired_changes


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {key: 1 for key in MATCHED_FIELDS}
        self.arrays = {"start_position": np.zeros((128, 3)),
                       "target_position": np.ones((128, 3)), "route_id": np.arange(128)%4,
                       "outcome": np.full(128, "timeout", dtype="U32")}
        self.arrays["outcome"][:100] = "success"

    def test_reports_gains_and_regressions_not_only_net_sr(self):
        arrays = copy.deepcopy(self.arrays)
        arrays["outcome"][3] = "collision"
        arrays["outcome"][100:103] = "success"
        change = paired_changes((self.metrics, self.arrays), (self.metrics, arrays))
        self.assertEqual(change["gained_routes"], [100, 101, 102])
        self.assertEqual(change["lost_routes"], [3])
        self.assertAlmostEqual(change["sr_gain_percentage_points"], 2/128*100)

    def test_rejects_different_maps_or_execution_horizons(self):
        for key in ("map_sha256", "max_physics_steps", "replan_interval_steps"):
            metrics = self.metrics.copy()
            metrics[key] = 2
            with self.assertRaises(ValueError):
                paired_changes((self.metrics, self.arrays), (metrics, self.arrays))

    def test_rejects_different_start_or_goal_positions(self):
        for key in ("start_position", "target_position"):
            arrays = copy.deepcopy(self.arrays)
            arrays[key][0, 0] += .1
            with self.assertRaises(ValueError):
                paired_changes((self.metrics, self.arrays), (self.metrics, arrays))


if __name__ == "__main__":
    unittest.main()
