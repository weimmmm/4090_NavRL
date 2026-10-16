"""Algorithm-only regression checks: no changes to the flight controller."""
import unittest
import torch
from lidar_wam.runner.online_supervision import (
    selective_online_correction_loss, selective_flow_correction_loss)


class FlowTailTests(unittest.TestCase):
    def test_zero_weight_is_exact_previous_objective(self):
        p, r, t = torch.randn(5, 10, 3), torch.randn(5, 10, 3), torch.randn(5, 3)
        old, metrics = selective_online_correction_loss(p, t, r, .3, 1, 5)
        new, new_metrics, _ = selective_flow_correction_loss(p, t, r, .3, 1, 5, 0.)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(metrics, new_metrics))

    def test_correct_first_action_has_no_competing_first_anchor(self):
        r = torch.zeros(1, 10, 3, requires_grad=True)
        t = torch.tensor([[.5, 0., 0.]], requires_grad=True)
        p = torch.ones(1, 10, 3, requires_grad=True)
        with torch.no_grad():
            p[:, 0] = t
        loss, _, _ = selective_flow_correction_loss(p, t, r, .3, 1, 1)
        loss.backward()
        self.assertEqual(float(p.grad[:, 0].abs().sum()), 0.)
        self.assertGreater(float(p.grad[:, 1:].abs().sum()), 0.)
        self.assertIsNone(r.grad)
        self.assertIsNone(t.grad)

    def test_agreement_uses_original_preservation_without_double_weight(self):
        p, r = torch.ones(2, 10, 3), torch.zeros(2, 10, 3)
        a, _ = selective_online_correction_loss(p, r[:, 0], r, .3, 1, 2)
        b, _, tail = selective_flow_correction_loss(p, r[:, 0], r, .3, 1, 2)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(float(tail), 0.)

    def test_five_uneven_ranks_equal_global_mean_and_gradient(self):
        p = torch.randn(7, 10, 3, requires_grad=True)
        r, t = torch.zeros(7, 10, 3), torch.ones(7, 3)*.4
        full, _, _ = selective_flow_correction_loss(p, t, r, .3, 1, 7)
        parts = [selective_flow_correction_loss(p[i::5], t[i::5], r[i::5], .3, 5, 7)[0]
                 for i in range(5)]
        average = torch.stack(parts).mean()
        self.assertTrue(torch.allclose(full, average))
        a = torch.autograd.grad(full, p, retain_graph=True)[0]
        b = torch.autograd.grad(average, p)[0]
        self.assertTrue(torch.allclose(a, b))

    def test_empty_rank_zero_gradient(self):
        p = torch.ones(1, 10, 3, requires_grad=True)
        loss, stats, tail = selective_flow_correction_loss(p, p[:, 0], p, .3, 5, 0)
        loss.backward()
        self.assertEqual(float(p.grad.abs().sum()), 0.)
        self.assertEqual(float(stats.abs().sum()), 0.)
        self.assertEqual(float(tail), 0.)


if __name__ == "__main__":
    unittest.main()
