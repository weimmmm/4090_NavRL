"""CPU regression checks for Flow-only selective online SFT."""
import unittest

import torch

from lidar_wam.models.flow_action_expert import FlowActionExpertModel, FLOW_ACTION_EXPERT_FORMAT
from lidar_wam.models.flow_online_sft import DifferentiableFlowPolicy
from lidar_wam.runner.action_expert_closed_loop_sft import build_model
from lidar_wam.runner.action_expert_online_sft import policy_wrapper, weight_packet
from lidar_wam.runner.online_supervision import online_teacher_target, selective_online_correction_loss


class FlowOnlineSFTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        model = FlowActionExpertModel(width=8, depth=8, heads=2, ffn_width=16)
        with torch.no_grad():
            model.diffusion.out.weight.normal_(std=.025)
        self.payload = {
            "format": FLOW_ACTION_EXPERT_FORMAT, "model": model.state_dict(),
            "architecture": dict(width=8, depth=8, heads=2, ffn_width=16,
                                 action_horizon=10, action_dim=3, proprio_dim=10,
                                 sampling_steps=10),
            "action_interface": dict(normalized_output_constraint="unit_box_after_euler",
                                     model_output_scale_mps_xyz=[2., 2., 2.])}
        self.inputs = (torch.randn(2, 4, 27, 5), torch.randn(2, 4), torch.randn(2, 10))
        self.noise = torch.randn(2, 10, 3)*.2

    def test_sampling_is_deployment_equivalent(self):
        expert = build_model(self.payload, freeze_policy_output_head=True)
        prediction = policy_wrapper(expert)(*self.inputs, self.noise)
        deployed = expert.predict_action(*self.inputs, initial_noise=self.noise)
        self.assertTrue(torch.equal(prediction.detach(), deployed))

    def test_all_ten_steps_are_in_gradient_graph_and_frozen_head_unchanged(self):
        expert = build_model(self.payload, freeze_policy_output_head=True)
        before = {k: p.detach().clone() for k, p in expert.named_parameters()}
        calls = []
        handle = expert.diffusion.register_forward_hook(lambda m, i, o: calls.append(o))
        prediction = DifferentiableFlowPolicy(expert, checkpoint_steps=False)(*self.inputs, self.noise)
        self.assertEqual(len(calls), 10)
        for out in calls:
            out.retain_grad()
        prediction[:, 0].square().mean().backward()
        self.assertTrue(all(out.grad is not None and out.grad.abs().sum() > 0 for out in calls))
        optimizer = torch.optim.AdamW([p for p in expert.parameters() if p.requires_grad], lr=5e-6)
        optimizer.step()
        changed = False
        for name, param in expert.named_parameters():
            trainable = name.startswith(("diffusion.blocks.6.", "diffusion.blocks.7."))
            self.assertEqual(param.requires_grad, trainable)
            if not trainable:
                self.assertTrue(torch.equal(param, before[name]), name)
                self.assertIsNone(param.grad)
            else:
                changed |= not torch.equal(param, before[name])
        self.assertTrue(changed)
        handle.remove()

    def test_checkpoint_recomputation_matches_full_graph_gradients(self):
        a = build_model(self.payload, freeze_policy_output_head=True)
        b = build_model(self.payload, freeze_policy_output_head=True)
        x = DifferentiableFlowPolicy(a, checkpoint_steps=False)(*self.inputs, self.noise)
        y = DifferentiableFlowPolicy(b, checkpoint_steps=True)(*self.inputs, self.noise)
        x[:, 0].square().mean().backward()
        y[:, 0].square().mean().backward()
        self.assertTrue(torch.equal(x, y))
        for p, q in zip(a.parameters(), b.parameters()):
            if p.requires_grad:
                self.assertTrue(torch.allclose(p.grad, q.grad, atol=1e-7, rtol=1e-5))

    def test_box_labels_keep_valid_large_combined_speed(self):
        value = torch.tensor([[.9, .9, .9], [1.3, -.4, -1.4]])
        target = online_teacher_target(value, "student-box")
        self.assertTrue(torch.equal(target[0], value[0]))
        self.assertGreater(float(target[0].norm()*2), 3.)
        self.assertTrue(torch.equal(target[1], torch.tensor([1., -.4, -1.])))

    def test_same_noise_reference_has_no_spurious_preservation_loss(self):
        student = policy_wrapper(build_model(self.payload, freeze_policy_output_head=True))
        reference = policy_wrapper(build_model(self.payload)).requires_grad_(False)
        prediction = student(*self.inputs, self.noise)
        with torch.no_grad():
            original = reference(*self.inputs, self.noise)
        loss, stats = selective_online_correction_loss(prediction, original[:, 0], original, .3, 1, 2)
        self.assertAlmostEqual(float(loss), 0., places=12)
        self.assertEqual(float(stats[0]), 0.)

    def test_packet_and_checkpoint_stay_flow_only(self):
        model = build_model(self.payload, freeze_policy_output_head=True)
        model.online_freeze_policy_output_head = True
        packet = weight_packet(model, 1)
        self.assertTrue(all(n.startswith(("diffusion.blocks.6.", "diffusion.blocks.7."))
                            for n in packet["names"]))
        self.assertEqual(set(model.state_dict()), set(self.payload["model"]))
        self.assertFalse(hasattr(model, "policy"))

    def test_wrong_action_contract_rejected(self):
        self.payload["action_interface"]["model_output_scale_mps_xyz"] = [1., 1., 1.]
        with self.assertRaises(ValueError):
            build_model(self.payload)

    def test_uneven_five_gpu_weighting_matches_global_objective(self):
        original = torch.zeros(7, 10, 3)
        target = torch.ones(7, 3)*.4
        prediction = torch.randn(7, 10, 3, requires_grad=True)
        full, _ = selective_online_correction_loss(prediction, target, original, .3, 1, 7)
        parts = [selective_online_correction_loss(prediction[i::5], target[i::5],
                 original[i::5], .3, 5, 7)[0] for i in range(5)]
        averaged = torch.stack(parts).mean()
        self.assertTrue(torch.allclose(full, averaged))
        expected = torch.autograd.grad(full, prediction, retain_graph=True)[0]
        actual = torch.autograd.grad(averaged, prediction)[0]
        self.assertTrue(torch.allclose(expected, actual))

    def test_requested_gpu_roles_exclude_zero_from_training(self):
        from online_sft_jobs import validate_gpu_roles
        validate_gpu_roles([2, 3, 4, 5, 7], 1, 0, None)
        with self.assertRaises(ValueError):
            validate_gpu_roles([0, 2, 3, 4, 5, 7], 1, 0, None)


if __name__ == "__main__":
    unittest.main()
