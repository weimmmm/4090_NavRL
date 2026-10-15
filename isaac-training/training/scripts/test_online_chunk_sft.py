"""CPU checks for online chunk masks, input causality and terminal exclusion."""
import unittest

import numpy as np
import torch

from online_chunk_sft import ChunkTeacherBuffer
from lidar_wam.runner.online_supervision import masked_online_tail_loss


class OnlineChunkTests(unittest.TestCase):
    def make_buffer(self):
        latent = torch.randn(4, 4, 27, 5)
        goal, proprio = torch.randn(4, 4), torch.randn(4, 10)
        quaternion = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
        selected = torch.tensor([True, True, False, True])
        return ChunkTeacherBuffer(latent, goal, proprio, quaternion, selected, 7, 100, "student-norm")

    def test_only_chunk_start_inputs_and_valid_targets_are_exposed(self):
        buffer = self.make_buffer()
        source = buffer.inputs["latent"].copy()
        for index in range(10):
            valid = torch.tensor([True, index < 4, True, True])
            raw = torch.ones(4, 3)*2.
            if index == 5:
                raw[3, 0] = float("nan")
            buffer.record(index, raw, valid)
        batch, _ = buffer.consume(7, 110)
        self.assertEqual(set(batch), {"latent", "goal", "proprio", "target", "target_mask"})
        self.assertEqual(batch["target"].shape, (3, 10, 3))
        self.assertTrue(np.array_equal(batch["latent"], source[[0, 1, 3]]))
        self.assertTrue(np.isfinite(batch["target"]).all())
        self.assertTrue((np.linalg.norm(batch["target"], axis=-1)<=1.000001).all())
        self.assertEqual(batch["target_mask"][1].sum(), 4.)
        self.assertEqual(batch["target_mask"][2, 5], 0.)
        self.assertTrue((batch["target"][1, 4:] == 0.).all())

    def test_rejects_partial_replayed_and_stale_chunks(self):
        buffer = self.make_buffer()
        with self.assertRaises(ValueError):
            buffer.consume(7, 110)
        for index in range(10):
            buffer.record(index, torch.zeros(4, 3), torch.ones(4, dtype=torch.bool))
        with self.assertRaises(ValueError):
            buffer.consume(8, 110)
        with self.assertRaises(ValueError):
            buffer.consume(7, 120)
        buffer.consume(7, 110)
        with self.assertRaises(ValueError):
            buffer.consume(7, 110)

    def test_source_pose_and_input_are_independent_of_future_mutation(self):
        buffer = self.make_buffer()
        state = torch.randn(4, 13)
        buffer.quaternion.copy_(state[:, 3:7])
        initial_quaternion = buffer.quaternion.clone()
        # A new/current pose is not substituted for the source pose of old labels.
        state[:, 3:7].zero_()
        self.assertTrue(torch.equal(buffer.quaternion, initial_quaternion))
        self.assertEqual(buffer.inputs["latent"].shape, (4, 4, 27, 5))

    def test_terminal_padding_has_zero_gradient_and_real_tail_has_gradient(self):
        prediction = torch.ones(2, 10, 3, requires_grad=True)
        target = torch.zeros_like(prediction)
        mask = torch.zeros(2, 10)
        mask[0, :4] = 1.
        mask[1, 0] = 1.
        masked_online_tail_loss(prediction, target, mask, 1, 1).backward()
        self.assertEqual(float(prediction.grad[:, 0].abs().sum()), 0.)
        self.assertGreater(float(prediction.grad[0, 1:4].abs().sum()), 0.)
        self.assertEqual(float(prediction.grad[0, 4:].abs().sum()), 0.)
        self.assertEqual(float(prediction.grad[1].abs().sum()), 0.)

    def test_uneven_four_gpu_tail_mean_is_exact(self):
        prediction, target = torch.randn(7, 10, 3), torch.randn(7, 10, 3)
        mask = torch.ones(7, 10)
        mask[1, 1:] = 0.
        mask[2, 5:] = 0.
        tail_samples = int(mask[:, 1:].any(-1).sum())
        global_loss = masked_online_tail_loss(prediction, target, mask, 1, tail_samples)
        ranks = [masked_online_tail_loss(prediction[rank::4], target[rank::4], mask[rank::4],
                                         4, tail_samples) for rank in range(4)]
        self.assertTrue(torch.allclose(torch.stack(ranks).mean(), global_loss))


if __name__ == "__main__":
    unittest.main()
