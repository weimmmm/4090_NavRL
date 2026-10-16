import unittest

from run_sft_length_gate_sweep import STEPS, select_candidate, trial_specs, verify_metrics


class LengthGateSweepTests(unittest.TestCase):
    def test_only_existing_gate_and_short_lengths_are_swept(self):
        self.assertEqual(STEPS, (50, 100, 150))
        self.assertEqual([x[1] for x in trial_specs()], [.3, .2, .4])

    def test_ties_do_not_replace_the_verified_incumbent(self):
        self.assertIsNone(select_candidate([dict(counts=dict(success=112))], 112))
        rows = [dict(counts=dict(success=113)), dict(counts=dict(success=115))]
        self.assertIs(select_candidate(rows, 112), rows[1])

    def test_route_and_snapshot_identity_are_checked(self):
        anchor = dict(map_sha256="map", initial_conditions_sha256="routes")
        good = dict(num_envs=128, counts=dict(success=112, collision=16),
                    checkpoint_step=50, fixed_route_start_index=0, **anchor)
        verify_metrics(good, anchor, 50)
        with self.assertRaises(ValueError):
            verify_metrics({**good, "initial_conditions_sha256": "other"}, anchor, 50)
        with self.assertRaises(ValueError):
            verify_metrics(good, anchor, 100)
        with self.assertRaises(ValueError):
            verify_metrics({**good, "counts": dict(success=112)}, anchor, 50)


if __name__ == "__main__":
    unittest.main()
