import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lidar_encoder import AzimuthCircularConv2d, RangeImageEncoder, min_pool_ranges


class RangeImageEncoderTests(unittest.TestCase):
    def test_wraps_azimuth_but_not_elevation(self):
        conv = AzimuthCircularConv2d(1, 1, kernel_size=(3, 3))
        with torch.no_grad():
            conv.weight.fill_(1)
            conv.bias.zero_()
        scan = torch.zeros(1, 1, 4, 6)
        scan[0, 0, 0, -1] = 1
        output = conv(scan)
        self.assertEqual(output.shape, scan.shape)
        self.assertEqual(output[0, 0, 0, 0].item(), 1)
        self.assertEqual(output[0, 0, -1, -1].item(), 0)

    def test_normalizes_distance_without_changing_range_image(self):
        for lidar_range in (4.0, 8.0, 10.0):
            encoder = RangeImageEncoder(lidar_range)
            values = torch.tensor([0., lidar_range / 2, lidar_range, lidar_range * 2])
            range_image = values.repeat(9).reshape(1, 1, 1, 36).expand(2, 1, 6, 36).clone()
            original = range_image.clone()
            inputs = []
            hook = encoder.network[0].register_forward_pre_hook(
                lambda module, args: inputs.append(args[0].detach().clone())
            )
            encoder(range_image)
            hook.remove()
            torch.testing.assert_close(inputs[0][0, 0, 0, :4], torch.tensor([-1., 0., 1., 1.]))
            torch.testing.assert_close(range_image, original)

    def test_shape_gradients_and_rollout_vmap(self):
        torch.manual_seed(0)
        encoder = RangeImageEncoder(10)
        range_image = (torch.rand(3, 1, 6, 36) * 10).requires_grad_()
        shapes = []
        hooks = [encoder.network[i].register_forward_hook(
            lambda module, args, output: shapes.append(tuple(output.shape))
        ) for i in (0, 2, 4)]
        features = encoder(range_image)
        for hook in hooks:
            hook.remove()
        self.assertEqual(shapes, [(3, 4, 6, 36), (3, 16, 6, 18), (3, 16, 3, 9)])
        self.assertEqual(features.shape, (3, 128))
        features[:, 0].sum().backward()
        for gradient in (range_image.grad, encoder.network[0].weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        rollout = torch.rand(3, 5, 1, 6, 36) * 10
        with torch.no_grad():
            mapped = torch.vmap(encoder)(rollout)
            expected = torch.stack([encoder(env_scan) for env_scan in rollout])
        self.assertEqual(mapped.shape, (3, 5, 128))
        torch.testing.assert_close(mapped, expected, rtol=1e-4, atol=1e-5)

    def test_dense_sampling_keeps_nearest_distance_per_block(self):
        distances = torch.arange(108 * 18, dtype=torch.float32).reshape(1, 1, 108, 18)
        distances = distances.expand(2, 1, 108, 18).clone()
        distances[0, 0, 107, 17] = 0.1
        original = distances.clone()
        pooled = min_pool_ranges(distances, 3, 3)
        self.assertEqual(pooled.shape, (2, 1, 36, 6))
        expected = torch.arange(0, 108, 3)[:, None] * 18 + torch.arange(0, 18, 3)[None, :]
        torch.testing.assert_close(pooled[1, 0], expected.float())
        self.assertAlmostEqual(pooled[0, 0, -1, -1].item(), 0.1)
        torch.testing.assert_close(distances, original)
        range_image = pooled.clamp_max(10).transpose(-2, -1).contiguous()
        self.assertEqual(range_image.shape, (2, 1, 6, 36))
        self.assertAlmostEqual(range_image[0, 0, -1, -1].item(), 0.1)
        features = RangeImageEncoder(10)(range_image)
        self.assertEqual(features.shape, (2, 128))
        self.assertTrue(torch.isfinite(features).all())

    def test_rejects_invalid_pooling_dimensions(self):
        distances = torch.ones(2, 1, 108, 18)
        for factors in ((0, 3), (3, -1), (5, 3), (3, 4)):
            with self.assertRaises(ValueError):
                min_pool_ranges(distances, *factors)

    def test_rejects_invalid_range(self):
        for lidar_range in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                RangeImageEncoder(lidar_range)


if __name__ == "__main__":
    unittest.main()
