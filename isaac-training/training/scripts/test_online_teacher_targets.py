"""CPU tests for the teacher-label-only velocity-domain ablation."""
import unittest

import torch

from lidar_wam.runner.online_supervision import (
    online_teacher_target, student_feasible_target, target_diagnostics, current_state_anchor_loss)


class TeacherTargetTests(unittest.TestCase):
    def test_feasible_values_and_zero_remain_unchanged(self):
        raw = torch.tensor([[0., 0., 0.], [.2, -.3, .4], [1., 0., 0.]])
        self.assertTrue(torch.equal(student_feasible_target(raw), raw))
        self.assertIs(online_teacher_target(raw, "raw"), raw)

    def test_projection_preserves_direction_and_input(self):
        raw = torch.tensor([[1., 1., 0.], [-1.2, 1.8, .7]])
        saved = raw.clone()
        projected = student_feasible_target(raw)
        self.assertTrue(torch.equal(raw, saved))
        self.assertTrue(torch.allclose(projected.norm(dim=-1), torch.ones(2)))
        self.assertTrue(torch.allclose(torch.nn.functional.cosine_similarity(raw, projected), torch.ones(2)))
        self.assertTrue(torch.equal(student_feasible_target(projected), projected))

    def test_projection_commutes_with_yaw_coordinate_change(self):
        yaw = torch.tensor(.9)
        rotation = torch.tensor([[yaw.cos(), -yaw.sin(), 0.],
                                 [yaw.sin(), yaw.cos(), 0.], [0., 0., 1.]])
        velocity = torch.randn(100, 3)*2
        self.assertTrue(torch.allclose(student_feasible_target(velocity @ rotation.T),
                                      student_feasible_target(velocity) @ rotation.T, atol=2e-7))

    def test_supports_complete_action_chunks(self):
        projected = student_feasible_target(torch.randn(7, 10, 3)*2)
        self.assertEqual(projected.shape, (7, 10, 3))
        self.assertTrue(bool((projected.norm(dim=-1)<=1.000001).all()))

    def test_invalid_labels_are_not_silently_replaced(self):
        raw = torch.tensor([[float("nan"), 1., 0.]])
        self.assertFalse(bool(torch.isfinite(student_feasible_target(raw)).all()))
        with self.assertRaises(ValueError):
            online_teacher_target(torch.zeros(2, 3), "unknown")

    def test_diagnostics_use_only_selected_fresh_states(self):
        raw = torch.tensor([[1., 1., 0.], [.2, 0., 0.], [float("nan"), 0., 0.]])
        target = student_feasible_target(raw)
        stats = target_diagnostics(raw, target, torch.tensor([True, True, False]))
        self.assertEqual(stats["teacher_raw_speed_over_2_fraction"], .5)
        self.assertGreater(stats["teacher_label_projection_mae_mps"], 0.)
        self.assertLessEqual(stats["supervision_mean_speed_mps"], 2.)

    def test_current_state_anchor_starts_at_zero_and_freezes_reference(self):
        prediction = torch.randn(7, 10, 3, requires_grad=True)
        reference = prediction.detach().clone().requires_grad_(True)
        self.assertEqual(float(current_state_anchor_loss(prediction, reference, 1, 7)), 0.)
        prediction = torch.zeros(7, 10, 3, requires_grad=True)
        current_state_anchor_loss(prediction, reference, 1, 7).backward()
        self.assertIsNone(reference.grad)
        self.assertTrue(bool((prediction.grad[:, 1:].abs().sum((1, 2))>0).all()))

    def test_current_state_anchor_uneven_ddp_matches_global_mean(self):
        prediction = torch.randn(7, 10, 3)
        reference = torch.randn_like(prediction)
        global_loss = current_state_anchor_loss(prediction, reference, 1, 7)
        rank_losses = [current_state_anchor_loss(prediction[rank::4], reference[rank::4], 4, 7)
                       for rank in range(4)]
        self.assertTrue(torch.allclose(torch.stack(rank_losses).mean(), global_loss))


if __name__ == "__main__":
    unittest.main()
