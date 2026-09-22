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
from lidar_wam.runner.world_direct_horizon import (
    lagen_zero_pad_points,
    lagen_zero_pad_squared_chamfer_points,
    squared_chamfer_points,
    voxel_anchor_squared_chamfer_points,
)
from lidar_wam.runner.action_expert_joint import (
    _action_validation_gate,
    _joint_schedule,
)


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


class JointTrainingPolicyTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
