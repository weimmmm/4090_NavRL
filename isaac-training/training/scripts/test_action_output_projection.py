"""Training radial gradients without changing original deployment behavior."""
import unittest

import torch
from torch.nn import functional as F

from lidar_wam.models.action_expert import DeterministicActionHead


class OutputProjectionTests(unittest.TestCase):
    def setUp(self):
        self.model = DeterministicActionHead(width=8, depth=0, heads=2, ffn_width=16)
        with torch.no_grad():
            self.model.out.weight.zero_()
            self.model.out.bias.copy_(torch.tensor([2., 0., 0.]))
        self.inputs = (torch.zeros(2, 135, 8), torch.zeros(2, 4), torch.zeros(2, 10),
                       torch.zeros(2, 30, 3), torch.zeros(2, 30))

    def test_default_deployment_is_identical_and_bounded(self):
        default = self.model(*self.inputs)
        explicit = self.model(*self.inputs, project_output=True)
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(torch.allclose(default.norm(dim=-1), torch.ones(2, 10)))
        self.assertTrue(torch.allclose(self.model(*self.inputs, project_output=False).norm(dim=-1),
                                      torch.full((2, 10), 2.)))

    def test_training_only_option_restores_radial_learning_signal(self):
        target = torch.zeros(2, 10, 3)
        target[..., 0] = .25
        F.smooth_l1_loss(self.model(*self.inputs), target, beta=.25).backward()
        self.assertEqual(float(self.model.out.bias.grad.abs().sum()), 0.)
        self.model.zero_grad(set_to_none=True)
        F.smooth_l1_loss(self.model(*self.inputs, project_output=False), target, beta=.25).backward()
        self.assertGreater(float(self.model.out.bias.grad[0]), 0.)

    def test_option_does_not_change_checkpoint_keys_or_weights(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.model(*self.inputs, project_output=False)
        after = self.model.state_dict()
        self.assertEqual(set(before), set(after))
        self.assertTrue(all(torch.equal(value, after[key]) for key, value in before.items()))


if __name__ == "__main__":
    unittest.main()
