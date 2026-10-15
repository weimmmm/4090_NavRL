"""Tail-only adaptation keeps original output calibration and deployment keys."""
import unittest

import torch

from lidar_wam.models.action_expert import SuperActionExpertModel
from lidar_wam.runner.action_expert_closed_loop_sft import DirectPolicy, build_model
from lidar_wam.runner.action_expert_online_sft import weight_packet


class OutputHeadFreezeTests(unittest.TestCase):
    def setUp(self):
        self.arch = dict(width=8, depth=8, heads=2, ffn_width=16,
                         action_horizon=10, action_dim=3, proprio_dim=10, past_horizon=30)
        original = SuperActionExpertModel(width=8, depth=8, heads=2, ffn_width=16)
        with torch.no_grad():
            original.policy.out.weight.normal_(std=.1)
        self.payload = dict(architecture=self.arch, model=original.state_dict())

    def test_default_freeze_split_remains_compatible(self):
        model = build_model(self.payload)
        self.assertTrue(model.policy.out.weight.requires_grad)
        self.assertTrue(model.policy.norm.weight.requires_grad)
        self.assertFalse(next(model.observation.parameters()).requires_grad)
        self.assertFalse(next(model.policy.blocks[5].parameters()).requires_grad)

    def test_only_last_two_blocks_can_change_and_gradient_crosses_frozen_head(self):
        model = build_model(self.payload, freeze_policy_output_head=True)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad,
                             name.startswith(("policy.blocks.6.", "policy.blocks.7.")))
        head = {name: p.detach().clone() for name, p in model.named_parameters()
                if name.startswith(("policy.norm.", "policy.out."))}
        direct = DirectPolicy(model)
        prediction = direct(torch.randn(2, 4, 27, 5), torch.randn(2, 4), torch.randn(2, 10))
        prediction[:, 0].square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()
                               if p.requires_grad and p.grad is not None), 0.)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)
        optimizer.step()
        current = dict(model.named_parameters())
        for name, value in head.items():
            self.assertIsNone(current[name].grad)
            self.assertTrue(torch.equal(value, current[name]))
        self.assertEqual(set(model.state_dict()), set(self.payload['model']))

    def test_actor_packet_is_explicit_about_frozen_output_head(self):
        model = build_model(self.payload, freeze_policy_output_head=True)
        model.online_freeze_policy_output_head = True
        packet = weight_packet(model, 7)
        self.assertTrue(packet['freeze_policy_output_head'])
        self.assertTrue(all(name.startswith(("policy.blocks.6.", "policy.blocks.7."))
                            for name in packet['names']))
        self.assertEqual(packet['weights'].size,
                         sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    unittest.main()
