from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from lidar_wam.coordinates import (
    goal_frame_causal_features,
    goal_to_world,
    numpy_goal_frame_causal_features,
    normalized_to_goal_velocity,
    normalized_to_world_velocity,
    world_to_goal,
)
from lidar_wam.data_v2 import random_window_subset, validation_trajectory_keys
from lidar_wam.runner.world_direct_horizon import (
    lagen_zero_pad_points,
    lagen_zero_pad_squared_chamfer_points,
    squared_chamfer_points,
    summarize_rows,
    voxel_anchor_squared_chamfer_points,
)
from lidar_wam.runner.action_expert_joint import (
    _action_validation_gate,
    _cosine_lr_scale,
    _joint_schedule,
)
from lidar_wam.models.action_expert import (
    ActionFlowExpert,
    JointWorldActionModel,
)


class _DummyCondition(torch.nn.Module):
    def forward(self, actions, state, horizon=None):
        value = actions.reshape(len(actions), 30, 3).mean(-1, keepdim=True)
        value = value.expand(-1, -1, 768)
        padding = torch.zeros(
            len(state), 768-state.shape[-1], device=state.device,
            dtype=state.dtype)
        state_token = torch.cat((state, padding), dim=-1).unsqueeze(1)
        return torch.cat((value, state_token), dim=1)


class _DummyUNet(torch.nn.Module):
    def forward(self, sample, timestep, encoder_hidden_states):
        condition = encoder_hidden_states.mean((1, 2))[:, None, None, None]
        return SimpleNamespace(sample=sample[:, :4] * 0.25 + condition)


class _DummyWorld(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = _DummyCondition()
        self.unet = _DummyUNet()


class CoordinateSemanticsTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(123)
        self.direction = torch.randn(64, 3, generator=generator)
        self.direction[:, 2] = 0

    def test_goal_world_round_trip(self):
        vector = torch.randn(64, 10, 3, generator=torch.Generator().manual_seed(2))
        recovered = world_to_goal(goal_to_world(vector, self.direction), self.direction)
        self.assertLessEqual(float((vector-recovered).abs().max()), 1e-5)

    def test_action_conversion_range(self):
        action = torch.tensor([[[0.0, 0.5, 1.0]]]).expand(64, -1, -1)
        goal = normalized_to_goal_velocity(action)
        self.assertTrue(torch.equal(goal[0, 0], torch.tensor([-2.0, 0.0, 2.0])))
        world = normalized_to_world_velocity(action, self.direction)
        self.assertEqual(tuple(world.shape), (64, 1, 3))

    def test_offline_online_features_are_identical(self):
        state = torch.randn(64, 13, generator=torch.Generator().manual_seed(3))
        state[:, 3:7] = torch.randn(64, 4, generator=torch.Generator().manual_seed(4))
        target = torch.randn(64, 3, generator=torch.Generator().manual_seed(5))
        expected = goal_frame_causal_features(state, target, self.direction)
        actual = numpy_goal_frame_causal_features(
            state.numpy(), target.numpy(), self.direction.numpy())
        self.assertTrue(np.array_equal(expected[0].numpy(), actual[0]))
        self.assertTrue(np.array_equal(expected[1].numpy(), actual[1]))


class DatasetSplitProtocolTest(unittest.TestCase):
    def test_validation_window_sampling_is_global_exact_and_deterministic(self):
        candidates = np.arange(8000, dtype=np.int64)
        first = random_window_subset(candidates, 1024, 42)
        second = random_window_subset(candidates, 1024, 42)
        future = random_window_subset(candidates, 256, 42)
        self.assertEqual(len(first), 1024)
        self.assertEqual(len(future), 256)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(np.unique(first)), len(first))

    def test_validation_selection_is_exact_per_seed_and_deterministic(self):
        inventory = {
            0: [("a.h5", f"scene-{index:03d}") for index in range(100)],
            1: [("b.h5", f"scene-{index:03d}") for index in range(41)],
        }
        first = validation_trajectory_keys(inventory, 0.05, 42)
        second = validation_trajectory_keys(inventory, 0.05, 42)
        self.assertEqual(first, second)
        self.assertEqual(sum(key[0] == "a.h5" for key in first), 5)
        self.assertEqual(sum(key[0] == "b.h5" for key in first), 2)
        self.assertTrue(first.isdisjoint(set(inventory[0] + inventory[1]) - first))

    def test_validation_selection_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            validation_trajectory_keys({0: [("a.h5", "scene")]}, 0.0, 42)


class SquaredChamferTest(unittest.TestCase):
    def test_matches_bidirectional_half_mean_definition(self):
        prediction = np.asarray([[0.0, 0.0, 0.0],
                                 [2.0, 0.0, 0.0]], dtype=np.float32)
        target = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

        # prediction -> target: (0^2 + 2^2) / 2 = 2
        # target -> prediction: 0^2 = 0
        # CD = 1/2 * (2 + 0) = 1 m^2
        self.assertEqual(squared_chamfer_points(prediction, target), 1.0)

    def test_identical_and_empty_point_clouds(self):
        points = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        empty = np.empty((0, 3), dtype=np.float32)
        self.assertEqual(squared_chamfer_points(points, points), 0.0)
        self.assertEqual(squared_chamfer_points(empty, empty), 0.0)
        self.assertEqual(squared_chamfer_points(points, empty), 100.0)

    def test_voxel_anchor_handles_empty_clouds_without_a_sentinel_penalty(self):
        empty = np.empty((0, 3), dtype=np.float32)
        self.assertEqual(voxel_anchor_squared_chamfer_points(empty, empty), 0.0)

        # Duplicates in one 20 cm voxel collapse to the same voxel center.
        prediction = np.asarray([[0.01, 0.01, 0.01],
                                 [0.09, 0.09, 0.09]], dtype=np.float32)
        target = np.asarray([[0.08, 0.08, 0.08]], dtype=np.float32)
        self.assertEqual(
            voxel_anchor_squared_chamfer_points(prediction, target), 0.0)

        one_voxel = np.asarray([[1.01, 0.01, 0.01]], dtype=np.float32)
        # Augmented prediction: origin + voxel center [1.1, 0.1, 0.1].
        # Augmented target: origin.  CD = 1/2 * (0 + 1.23/2) = 0.3075.
        self.assertAlmostEqual(
            voxel_anchor_squared_chamfer_points(one_voxel, empty),
            0.3075, places=7)

    def test_lagen_padding_uses_origin_padding_and_fixed_cardinality(self):
        points = np.asarray([[1.0, 0.0, 0.0],
                             [2.0, 0.0, 0.0]], dtype=np.float32)
        padded = lagen_zero_pad_points(points, count=5)
        self.assertEqual(tuple(padded.shape), (5, 3))
        self.assertTrue(np.array_equal(
            padded[:, 0], np.asarray([1.0, 2.0, 0.0, 0.0, 0.0])))

        empty = np.empty((0, 3), dtype=np.float32)
        self.assertEqual(
            lagen_zero_pad_squared_chamfer_points(empty, empty, count=5), 0.0)
        # Prediction -> empty: (4 + 0 + 0 + 0 + 0) / 5 = 0.8.
        # Empty -> prediction: every origin matches a padded origin = 0.
        # Bidirectional half mean = 0.4, exactly matching LaGen semantics.
        one_point = np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(
            lagen_zero_pad_squared_chamfer_points(one_point, empty, count=5),
            0.4, places=7)

    def test_nonempty_cd_summary_excludes_empty_cloud_penalties(self):
        base = {
            "voxel_anchor_cd_m2": 0.0,
            "lagen_zero_pad_1500_cd_m2": 0.0,
            "symmetric_nn_distance_m": 0.0,
            "valid_range_abs_error_sum_m": 0.0,
            "valid_range_count": 1,
            "tp": 1, "fp": 0, "fn": 0,
            "false_empty_frame": False, "false_hit_frame": False,
            "gt_hit_count": 1, "pred_hit_count": 1,
        }
        valid = {**base, "cd_paper_m2": 0.4,
                 "gt_empty": False, "pred_empty": False}
        empty = {**base, "cd_paper_m2": 100.0,
                 "gt_empty": False, "pred_empty": True,
                 "false_empty_frame": True, "pred_hit_count": 0}
        summary = summarize_rows([valid, empty])
        self.assertEqual(summary["cd_paper_m2"], 50.2)
        self.assertEqual(summary["nonempty_cd_paper_m2"], 0.4)
        self.assertEqual(summary["nonempty_cd_samples"], 1)
        self.assertEqual(summary["empty_cd_samples_excluded"], 1)


class JointTrainingPolicyTest(unittest.TestCase):
    def test_cosine_lr_scale_starts_at_one_and_ends_at_minimum(self):
        args = SimpleNamespace(
            lr_decay_start=6000, lr_min_scale=0.1, steps=100000)
        self.assertEqual(_cosine_lr_scale(6000, args), 1.0)
        self.assertAlmostEqual(_cosine_lr_scale(53000, args), 0.55)
        self.assertAlmostEqual(_cosine_lr_scale(100000, args), 0.1)

    def test_world_schedule_freezes_then_ramps(self):
        args = SimpleNamespace(
            world_freeze_steps=500, world_ramp_end=1500,
            world_weight_start=0.05, world_weight_mid=0.15,
            world_weight=0.25, world_lr=1e-6)
        for step, expected_weight, expected_lr in (
                (1, 0.05, 0.0), (500, 0.15, 0.0),
                (1000, 0.20, 1e-6), (1500, 0.25, 1e-6),
                (9000, 0.25, 1e-6)):
            weight, lr = _joint_schedule(step, args)
            self.assertAlmostEqual(weight, expected_weight)
            self.assertAlmostEqual(lr, expected_lr)

    def test_action_gate_uses_executed_first_ten_hard_slices(self):
        summary = {
            "first_10_mae": 0.09,
            "repeat_last_first_10_mae": 0.10,
            "low_clearance_first_10_mae": 0.18,
            "low_clearance_repeat_last_first_10_mae": 0.20,
            "turning_first_10_mae": 0.27,
            "turning_repeat_last_first_10_mae": 0.30,
        }
        gate = _action_validation_gate(summary)
        self.assertTrue(gate["passed"])
        summary["turning_first_10_mae"] = 0.285
        self.assertFalse(_action_validation_gate(summary)["passed"])

    def test_zero_future_gate_preserves_pretrained_action_output(self):
        expert = ActionFlowExpert(
            width=16, depth=2, heads=4, ffn_width=32,
            future_attention_layers=1)
        torch.nn.init.normal_(expert.out.weight, std=0.02)
        batch = 2
        values = (
            torch.randn(batch, 10, 3), torch.rand(batch) * 1000,
            torch.randn(batch, 135, 16), torch.randn(batch, 4),
            torch.randn(batch, 10), torch.randn(batch, 10, 3),
            torch.ones(batch, 10),
        )
        without_future = expert(*values)
        with_future = expert(*values, future_tokens=torch.randn(batch, 135, 16))
        self.assertTrue(torch.equal(without_future, with_future))

    @staticmethod
    def _joint_inputs(batch):
        return {
            "noisy_future": torch.randn(batch, 4, 27, 5),
            "world_actions": torch.rand(batch, 10, 3),
            "world_state": torch.randn(batch, 5),
            "world_timestep": torch.randint(0, 1000, (batch,)),
            "action_logit_mean": torch.zeros(3),
            "action_logit_std": torch.ones(3),
            "world_alpha_cumprod": torch.full((batch,), 0.5),
            "feedback_noise": torch.randn(batch, 4, 27, 5),
            "feedback_timesteps": torch.tensor([999, 500, 0]),
            "feedback_alphas": torch.tensor([0.01, 0.25, 0.99]),
            "feedback_previous_alphas": torch.tensor([0.25, 0.99, 1.0]),
            "feedback_gradient_steps": 1,
        }

    def test_joint_forward_is_bidirectional_and_shape_safe(self):
        model = JointWorldActionModel(
            _DummyWorld(), width=16, depth=2, heads=4, ffn_width=32,
            future_attention_layers=1)
        torch.nn.init.normal_(model.action_expert.out.weight, std=0.02)
        batch = 2
        refined, future = model(
            torch.randn(batch, 10, 3), torch.rand(batch) * 1000,
            torch.randn(batch, 4, 27, 5), torch.randn(batch, 4),
            torch.randn(batch, 10), torch.randn(batch, 10, 3),
            torch.ones(batch, 10),
            **self._joint_inputs(batch),
        )
        provisional = model._last_provisional_velocity
        self.assertEqual(tuple(refined.shape), (batch, 10, 3))
        self.assertEqual(tuple(provisional.shape), (batch, 10, 3))
        self.assertEqual(tuple(future.shape), (batch, 4, 27, 5))
        self.assertEqual(
            tuple(model.future_adapter(future).shape), (batch, 135, 16))
        self.assertEqual(
            tuple(model._last_action_semantic_tokens.shape), (batch, 10, 16))
        self.assertEqual(
            tuple(model._last_predicted_future.shape), (batch, 4, 27, 5))

    def test_action_semantic_condition_contains_only_executed_chunk(self):
        model = JointWorldActionModel(
            _DummyWorld(), width=16, depth=2, heads=4, ffn_width=32,
            future_attention_layers=1)
        model.action_semantic_adapter.gate.data.fill_(1.0)
        tokens = torch.randn(3, 10, 16)
        adapted = model.action_semantic_adapter(tokens)
        self.assertEqual(tuple(adapted.shape), (3, 10, 768))
        self.assertGreater(float(adapted.abs().sum()), 0.0)

    def test_action_feedback_is_independent_of_supervised_future_target(self):
        model = JointWorldActionModel(
            _DummyWorld(), width=16, depth=2, heads=4, ffn_width=32,
            future_attention_layers=1)
        model.action_expert.blocks[-1].future_gate.data.fill_(0.5)
        batch = 2
        common = (
            torch.randn(batch, 10, 3), torch.rand(batch) * 1000,
            torch.randn(batch, 4, 27, 5), torch.randn(batch, 4),
            torch.randn(batch, 10), torch.randn(batch, 10, 3),
            torch.ones(batch, 10),
        )
        kwargs = self._joint_inputs(batch)
        first = model(*common, **kwargs)[0]
        kwargs["noisy_future"] = torch.randn_like(kwargs["noisy_future"]) * 100
        second = model(*common, **kwargs)[0]
        self.assertTrue(torch.equal(first, second))

if __name__ == "__main__":
    unittest.main()
