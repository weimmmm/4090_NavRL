"""CPU checks for conditional, current-state-only online supervision."""
import unittest

import torch
from torch.nn import functional as F

from lidar_wam.runner.online_supervision import selective_online_correction_loss


class SelectiveCorrectionTests(unittest.TestCase):
    def test_agreement_preserves_original_instead_of_learning_small_bias(self):
        original = torch.zeros(3, 10, 3, requires_grad=True)
        student = original.detach().clone().requires_grad_(True)
        teacher = torch.full((3, 3), .025, requires_grad=True)
        loss, metrics = selective_online_correction_loss(student, teacher, original, .3, 1, 3)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(metrics[0]), 0.)
        self.assertEqual(float(student.grad.abs().sum()), 0.)
        self.assertIsNone(original.grad)
        self.assertIsNone(teacher.grad)

    def test_correction_has_no_competing_original_anchor(self):
        original = torch.zeros(1, 10, 3, requires_grad=True)
        teacher = torch.tensor([[.5, 0., 0.]], requires_grad=True)
        student = torch.ones(1, 10, 3, requires_grad=True)
        loss, metrics = selective_online_correction_loss(student, teacher, original, .3, 1, 1)
        expected = F.smooth_l1_loss(student[:, 0], teacher, beta=.25)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(float(metrics[0]), 1.)
        loss.backward()
        self.assertGreater(float(student.grad[:, 0].abs().sum()), 0.)
        self.assertEqual(float(student.grad[:, 1:].abs().sum()), 0.)
        self.assertIsNone(original.grad)
        self.assertIsNone(teacher.grad)

    def test_preservation_reaches_all_ten_commands(self):
        original = torch.zeros(1, 10, 3)
        student = torch.ones(1, 10, 3, requires_grad=True)
        loss, _ = selective_online_correction_loss(student, original[:, 0], original, .3, 1, 1)
        loss.backward()
        self.assertTrue(bool((student.grad.abs().sum(-1)>0).all()))

    def test_selector_does_not_disappear_when_student_matches_teacher(self):
        original = torch.zeros(1, 10, 3)
        teacher = torch.tensor([[.5, 0., 0.]])
        student = teacher[:, None].expand(-1, 10, -1).clone()
        _, metrics = selective_online_correction_loss(student, teacher, original, .3, 1, 1)
        self.assertEqual(float(metrics[0]), 1.)

    def test_physical_scale_is_two_mps_and_gate_uses_vector_norm(self):
        original = torch.zeros(2, 10, 3)
        teacher = torch.tensor([[.125, 0., 0.], [.125, .125, 0.]])
        _, metrics = selective_online_correction_loss(original, teacher, original, .3, 1, 2)
        self.assertEqual(float(metrics[0]), .5)

    def test_uneven_ddp_loss_metrics_and_gradients_match_global_mean(self):
        generator = torch.Generator().manual_seed(27)
        original = torch.randn(7, 10, 3, generator=generator)*.3
        teacher = original[:, 0].clone()
        teacher[::2, 1] += .5
        student = torch.randn(7, 10, 3, generator=generator).requires_grad_(True)
        full, stats = selective_online_correction_loss(student, teacher, original, .3, 1, 7)
        grad = torch.autograd.grad(full, student, retain_graph=True)[0]
        parts = [selective_online_correction_loss(student[i::4], teacher[i::4], original[i::4], .3, 4, 7)
                 for i in range(4)]
        mean = torch.stack([item[0] for item in parts]).mean()
        self.assertTrue(torch.allclose(mean, full))
        self.assertTrue(torch.allclose(torch.stack([item[1] for item in parts]).mean(0), stats))
        self.assertTrue(torch.allclose(torch.autograd.grad(mean, student)[0], grad))

    def test_empty_rank_has_zero_loss_and_zero_metrics(self):
        student = torch.ones(1, 10, 3, requires_grad=True)
        loss, stats = selective_online_correction_loss(student, torch.zeros(1, 3), student.detach(), .3, 4, 0)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertTrue(torch.equal(stats, torch.zeros(4)))
        self.assertEqual(float(student.grad.abs().sum()), 0.)

    def test_invalid_shape_or_threshold_rejected(self):
        student = torch.zeros(2, 10, 3)
        for threshold in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                selective_online_correction_loss(student, student[:, 0], student, threshold, 1, 2)
        with self.assertRaises(ValueError):
            selective_online_correction_loss(student, student, student, .3, 1, 2)


if __name__ == "__main__":
    unittest.main()
