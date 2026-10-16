"""Test integration sign, timestep units, noise pairing and baseline safety."""

import unittest

import torch

from lidar_wam.models.action_expert import SuperActionExpertModel
from lidar_wam.models.action_flow_sampling import sample_action_flow


class ConstantFlow:
    horizon = 10
    action_dim = 3

    def __init__(self, velocity):
        self.velocity = velocity
        self.times = []

    def __call__(self, action, timestep, *_conditions):
        self.times.append(timestep.clone())
        return self.velocity.expand_as(action)


class FlowSamplingTests(unittest.TestCase):
    def setUp(self):
        self.conditions = (torch.zeros(2, 135, 8), torch.zeros(2, 4),
                           torch.zeros(2, 10), torch.zeros(2, 30, 3),
                           torch.zeros(2, 30))

    def test_known_linear_path_has_correct_reverse_sign_and_time_scale(self):
        noise = torch.full((2, 10, 3), 0.6)
        clean = torch.full_like(noise, 0.2)
        for steps in (10, 20):
            expert = ConstantFlow(noise - clean)
            result = sample_action_flow(expert, *self.conditions, steps=steps,
                                        initial_noise=noise, project_output=False)
            self.assertTrue(torch.allclose(result, clean, atol=1e-6))
            self.assertEqual(len(expert.times), steps)
            self.assertEqual(float(expert.times[0][0]), 1000.)
            self.assertAlmostEqual(float(expert.times[-1][0]), 1000. / steps)
        self.assertTrue(torch.equal(noise, torch.full_like(noise, .6)))

    def test_equal_seed_and_final_unit_ball(self):
        expert = ConstantFlow(torch.tensor(0.))
        result = [sample_action_flow(
            expert, *self.conditions, steps=steps,
            generator=torch.Generator().manual_seed(18)) for steps in (10, 20)]
        self.assertTrue(torch.equal(*result))
        self.assertTrue(bool((result[0].norm(dim=-1) <= 1.000001).all()))

    def test_per_axis_unit_box_preserves_large_vector_norm(self):
        expert = ConstantFlow(torch.tensor(0.))
        noise = torch.full((2, 10, 3), .95)
        result = sample_action_flow(expert, *self.conditions, initial_noise=noise,
                                    output_constraint="unit_box")
        self.assertTrue(torch.equal(result, noise))
        self.assertGreater(float(result.norm(dim=-1).min()), 1.6)
        noise[:] = torch.tensor([1.4, -1.3, .7])
        result = sample_action_flow(expert, *self.conditions, initial_noise=noise,
                                    output_constraint="unit_box")
        self.assertTrue(torch.equal(result, torch.tensor([1., -1., .7]).expand_as(noise)))
        raw = sample_action_flow(expert, *self.conditions, initial_noise=noise,
                                 output_constraint="unit_box", project_output=False)
        self.assertTrue(torch.equal(raw, noise))

    def test_model_weights_and_direct_predictions_are_unchanged(self):
        torch.manual_seed(18)
        model = SuperActionExpertModel(width=8, depth=1, heads=2, ffn_width=16).eval()
        latent = torch.zeros(2, 4, 27, 5)
        _observation, goal, proprio, past, mask = self.conditions
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            direct_before = model.predict_action(latent, goal, proprio, past, mask)
            result = sample_action_flow(model.diffusion, model.encode_current(latent),
                                        goal, proprio, past, mask, steps=2)
            direct_after = model.predict_action(latent, goal, proprio, past, mask)
        self.assertEqual(tuple(result.shape), (2, 10, 3))
        self.assertTrue(torch.equal(direct_before, direct_after))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key])
                            for key, value in before.items()))

    def test_invalid_settings_are_rejected(self):
        expert = ConstantFlow(torch.tensor(0.))
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                sample_action_flow(expert, *self.conditions, steps=steps)
        with self.assertRaises(ValueError):
            sample_action_flow(expert, *self.conditions, noise_scale=float("nan"))
        with self.assertRaises(ValueError):
            sample_action_flow(expert, *self.conditions, output_constraint="unknown")


if __name__ == "__main__":
    unittest.main()
