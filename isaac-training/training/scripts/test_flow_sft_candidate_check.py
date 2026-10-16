import unittest
import numpy as np
from run_flow_sft_candidate_check import paired_outcomes


def outcomes(values):
    return dict(outcome=np.asarray(values), source_route_index=np.arange(128)+128,
                start_position=np.zeros((128, 3)), target_position=np.ones((128, 3)))


class CandidateCheckTests(unittest.TestCase):
    def test_pair_uses_real_global_route_ids(self):
        a = outcomes(["collision"]*4+["success"]*124)
        b = outcomes(["success"]*128)
        pair = paired_outcomes(a, b)
        self.assertEqual(pair["repaired_routes"], [128, 129, 130, 131])
        self.assertEqual(pair["regressed_routes"], [])
        self.assertEqual(pair["delta_success"], 4)
        self.assertEqual(pair["one_sided_paired_p_value"], 1/16)

    def test_same_counts_are_not_confused_with_no_route_regressions(self):
        a = outcomes(["collision", "success"]+["success"]*126)
        b = outcomes(["success", "collision"]+["success"]*126)
        pair = paired_outcomes(a, b)
        self.assertEqual(pair["delta_success"], 0)
        self.assertEqual(pair["repaired_routes"], [128])
        self.assertEqual(pair["regressed_routes"], [129])

    def test_reject_unmatched_initial_states_or_unfinished_routes(self):
        a, b = outcomes(["success"]*128), outcomes(["success"]*128)
        b["target_position"][0, 0] = 8.
        with self.assertRaises(ValueError): paired_outcomes(a, b)
        b = outcomes(["active"]+["success"]*127)
        with self.assertRaises(ValueError): paired_outcomes(a, b)


if __name__ == "__main__":
    unittest.main()
