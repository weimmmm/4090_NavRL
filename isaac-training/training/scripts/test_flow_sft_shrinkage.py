import copy
import unittest

import torch

from shrink_flow_sft_checkpoint import shrink_checkpoint


class ShrinkageTests(unittest.TestCase):
    def setUp(self):
        self.base = {"step": 57500, "format": "flow", "architecture": {"depth": 8},
                     "model": {"encoder.weight": torch.ones(2),
                               "diffusion.blocks.6.weight": torch.ones(2)}}
        self.tuned = copy.deepcopy(self.base)
        self.tuned.update(step=100, online_sft_optimizer={"state": 1})
        self.tuned["model"]["diffusion.blocks.6.weight"].add_(2)

    def test_linear_shrinkage_and_frozen_exactness(self):
        result = shrink_checkpoint(self.base, self.tuned, .25)
        self.assertTrue(torch.equal(result["model"]["diffusion.blocks.6.weight"], torch.full((2,), 1.5)))
        self.assertTrue(torch.equal(result["model"]["encoder.weight"], self.base["model"]["encoder.weight"]))
        self.assertNotIn("online_sft_optimizer", result)
        self.assertFalse(result["sft_update_shrinkage"]["closed_loop_improvement_verified"])
        self.assertEqual(float(self.base["model"]["diffusion.blocks.6.weight"][0]), 1.)
        self.assertEqual(float(self.tuned["model"]["diffusion.blocks.6.weight"][0]), 3.)

    def test_endpoints_exact(self):
        for alpha, source in ((0., self.base), (1., self.tuned)):
            result = shrink_checkpoint(self.base, self.tuned, alpha)
            for name in source["model"]:
                self.assertTrue(torch.equal(result["model"][name], source["model"][name]))

    def test_reject_frozen_drift(self):
        self.tuned["model"]["encoder.weight"].add_(1)
        with self.assertRaisesRegex(ValueError, "frozen"):
            shrink_checkpoint(self.base, self.tuned, .25)

    def test_reject_contract_or_nonfinite(self):
        self.tuned["architecture"]["depth"] = 9
        with self.assertRaisesRegex(ValueError, "contract"):
            shrink_checkpoint(self.base, self.tuned, .25)
        self.tuned["architecture"]["depth"] = 8
        self.tuned["model"]["diffusion.blocks.6.weight"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            shrink_checkpoint(self.base, self.tuned, .25)


if __name__ == "__main__":
    unittest.main()
