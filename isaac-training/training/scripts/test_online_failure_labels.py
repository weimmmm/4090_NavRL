import unittest
import numpy as np
import torch

from online_failure_labels import DelayedFailureLabels
from lidar_wam.runner.online_supervision import selective_flow_correction_loss


def fresh(value=0., count=2):
    return {"latent": np.full((count, 4, 27, 5), value, np.float32),
            "goal": np.zeros((count, 4), np.float32),
            "proprio": np.zeros((count, 10), np.float32),
            "target": np.full((count, 3), .4, np.float32)}


class FailureLabelsTests(unittest.TestCase):
    def test_delay_failure_ownership_and_one_use(self):
        q = DelayedFailureLabels(30)
        first = fresh(1.)
        a = q.push(first, [0, 1], [0, 0], 10, [])
        self.assertEqual(len(a["target"]), 0)
        first["latent"][:] = 7.  # A reset or reused actor array must not mutate saved inputs.
        q.push(fresh(2.), [0, 1], [0, 0], 20, [])
        q.push(fresh(3.), [0, 1], [1, 0], 30, [(0, 0, 25)])
        a = q.push(fresh(4.), [0, 1], [1, 0], 40, [])
        self.assertTrue(np.array_equal(a["correction_eligible"], [True, False]))
        self.assertEqual(float(a["latent"][0, 0, 0, 0]), 1.)
        b = q.push(fresh(5.), [0, 1], [1, 0], 50, [])
        c = q.push(fresh(6.), [0, 1], [1, 0], 60, [])
        self.assertTrue(b["correction_eligible"][0])
        self.assertFalse(c["correction_eligible"].any())  # New episode is never marked.
        self.assertEqual(q.seen, q.emitted+q.remaining)
        self.assertEqual(q.emitted, 6)

    def test_failure_at_window_boundary_is_included(self):
        q = DelayedFailureLabels(10)
        q.push(fresh(), [0, 1], [0, 0], 10, [])
        a = q.push(fresh(), [0, 1], [0, 0], 20, [(1, 0, 20)])
        self.assertTrue(np.array_equal(a["correction_eligible"], [False, True]))

    def test_empty_worker_and_no_failure_are_valid(self):
        q = DelayedFailureLabels(10)
        q.push(fresh(count=0), np.zeros(0, np.int64), np.zeros(0, np.int64), 10, [])
        a = q.push(fresh(), [0, 1], [0, 0], 20, [])
        self.assertEqual(len(a["target"]), 0)
        a = q.push(fresh(), [0, 1], [0, 0], 30, [])
        self.assertFalse(a["correction_eligible"].any())

    def test_reject_future_or_repeated_feedback_and_duplicate_slots(self):
        q = DelayedFailureLabels(10)
        with self.assertRaises(ValueError): q.push(fresh(), [0, 0], [0, 0], 10, [])
        with self.assertRaises(ValueError): q.push(fresh(), [0, 1], [0, 0], 10, [(0, 0, 11)])
        q.push(fresh(), [0, 1], [0, 0], 10, [])
        with self.assertRaises(ValueError): q.push(fresh(), [0, 1], [0, 0], 10, [])
        with self.assertRaises(ValueError): q.push(fresh(), [0, 1], [0, 0], 20, [(0, 0, 9)])

    def test_future_outcome_is_not_an_inference_input(self):
        q = DelayedFailureLabels(10)
        q.push(fresh(), [0, 1], [0, 0], 10, [])
        a = q.push(fresh(), [0, 1], [0, 0], 20, [(0, 0, 20)])
        self.assertEqual(set(a), {"latent", "goal", "proprio", "target", "correction_eligible"})
        p = torch.zeros(2, 10, 3, requires_grad=True)
        reference = torch.zeros_like(p)
        target = torch.from_numpy(a["target"])
        loss, metrics, _ = selective_flow_correction_loss(
            p, target, reference, .3, 1, 2, correction_eligible=torch.from_numpy(a["correction_eligible"]))
        loss.backward()
        self.assertGreater(float(p.grad[0, 0].abs().sum()), 0.)
        self.assertEqual(float(p.grad[1].abs().sum()), 0.)  # Safe row keeps original, NOT teacher.
        self.assertAlmostEqual(float(metrics[0]), .5)

    def test_binary_gate_and_uneven_rank_weighting(self):
        p = torch.randn(7, 10, 3, requires_grad=True)
        r, t = torch.zeros_like(p), torch.ones(7, 3)*.4
        eligible = torch.tensor([0, 1, 0, 0, 1, 0, 1], dtype=torch.bool)
        full = selective_flow_correction_loss(p, t, r, .3, 1, 7, correction_eligible=eligible)[0]
        parts = [selective_flow_correction_loss(p[i::5], t[i::5], r[i::5], .3, 5, 7,
                 correction_eligible=eligible[i::5])[0] for i in range(5)]
        mean = torch.stack(parts).mean()
        self.assertTrue(torch.allclose(full, mean))
        self.assertTrue(torch.allclose(torch.autograd.grad(full, p, retain_graph=True)[0],
                                       torch.autograd.grad(mean, p)[0]))
        with self.assertRaises(ValueError):
            selective_flow_correction_loss(p, t, r, .3, 1, 7, correction_eligible=torch.ones(7)*.5)


if __name__ == "__main__":
    unittest.main()
