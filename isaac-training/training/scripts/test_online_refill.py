"""CPU checks for per-slot state isolation, route geometry and fresh selection."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch

from online_refill import refill_slots, reset_policy_slots, sample_routes, select_fresh


class RefillTests(unittest.TestCase):
    def test_route_geometry(self):
        routes = sample_routes(100, np.random.default_rng(123))
        delta = routes["goal_position"]-routes["start_position"]
        self.assertTrue(bool((delta.norm(dim=-1) >= 48.).all()))
        self.assertTrue(bool((routes["start_position"][:, :2].abs().amax(-1) == 24.).all()))
        self.assertTrue(torch.allclose(routes["initial_orientation_wxyz"].norm(dim=-1), torch.ones(100)))
        yaw = 2*torch.atan2(routes["initial_orientation_wxyz"][:, 3], routes["initial_orientation_wxyz"][:, 0])
        self.assertTrue(torch.allclose(yaw, torch.atan2(delta[:, 1], delta[:, 0]), atol=1e-6))

    def test_only_reset_selected_rows(self):
        n = 8
        p = SimpleNamespace(required_history_frames=1, use_past_actions=False,
                            bootstrap_action_chunks=1, history=[torch.randn(n, 2, 108, 20)],
                            previous_velocity=torch.randn(n, 3), step_index=120,
                            warmup_chunks=torch.zeros(n, dtype=torch.long))
        shapes = {"past": (30, 3), "past_world": (30, 3), "past_mask": (30,), "chunk": (10, 3),
                  "goal_braking": (), "reference_started": (), "previous_reference_velocity": (3,),
                  "safety_heading": (), "last_ego_command": (3,), "last_world_command": (3,),
                  "last_reference_velocity_world": (3,), "reference_position": (3,),
                  "reference_yaw": (), "chunk_frame_yaw": ()}
        for name, shape in shapes.items():
            setattr(p, name, torch.randn(n, *shape))
        p._quaternion_yaw = lambda q: torch.arange(len(q), dtype=torch.float)
        p._goal_frame_yaw = lambda pos: torch.zeros(len(pos))
        saved = {key: getattr(p, key).clone() for key in shapes}
        history = p.history[0].clone()
        state = torch.randn(n, 13)
        ids = torch.tensor([1, 6])
        keep = torch.tensor([0, 2, 3, 4, 5, 7])
        reset_policy_slots(p, ids, state)
        for key in shapes:
            self.assertTrue(torch.equal(getattr(p, key)[keep], saved[key][keep]), key)
        self.assertTrue(torch.equal(p.history[0][keep], history[keep]))
        self.assertEqual(p.step_index, 120)
        self.assertTrue(bool((p.warmup_chunks[ids] == 1).all()))
        self.assertTrue(bool((p.chunk[ids] == 0).all()))
        self.assertTrue(torch.equal(p.previous_velocity[ids], state[ids, 7:10]))

    def test_unique_fresh_selection(self):
        valid = torch.ones(4608, dtype=torch.bool)
        valid[:20] = False
        selected, available = select_fresh(valid, 4096, torch.Generator().manual_seed(7))
        self.assertEqual(available, 4588)
        self.assertEqual(int(selected.sum()), 4096)
        self.assertFalse(bool(selected[:20].any()))
        fewer = torch.zeros(4608, dtype=torch.bool)
        fewer[:3] = True
        selected, _ = select_fresh(fewer, 4096, torch.Generator().manual_seed(7))
        self.assertEqual(int(selected.sum()), 3)  # Never duplicate labels to fill a batch.

    def test_partial_reset_preserves_other_episode_bookkeeping(self):
        from tensordict import TensorDict

        class FakeEnv:
            num_envs = 4
            device = torch.device("cpu")
            dt = .016

            def __init__(self):
                self.fixed_scene = sample_routes(4, np.random.default_rng(3))
                self.state = torch.randn(4, 1, 13)
                self.stats = TensorDict({"return": torch.arange(4.).view(4, 1)+10,
                                         "reach_goal": torch.ones(4, 1)}, [4])
                self.goal_stop_count = torch.arange(4).view(4, 1)+3
                self.prev_drone_vel_w = torch.randn(4, 1, 3)
                self.reward = torch.randn(4, 1)
                self.terminated = torch.ones(4, 1, dtype=torch.bool)
                self.truncated = torch.ones(4, 1, dtype=torch.bool)
                self.progress_buf = torch.arange(4)+100
                self.lidar = SimpleNamespace(update=lambda *a, **k: None)

            def reset(self, td):
                ids = td["_reset"].flatten().nonzero().flatten()
                self.state[ids, 0, :3] = self.fixed_scene["start_position"][ids]
                self.progress_buf[ids] = 0
                self.stats[ids] = 0.
                return self._compute_state_and_obs()

            def _compute_state_and_obs(self):
                # Mirror the observation builder's otherwise unwanted
                # reward/stop-counter updates for ALL rows on partial reset.
                self.stats["return"] += 1
                self.goal_stop_count += 1
                self.prev_drone_vel_w.fill_(99)
                return TensorDict({"info": {"drone_state": self.state.clone()},
                                   "stats": self.stats.clone()}, [4])

            def _compute_reward_and_done(self):
                return TensorDict({"done": self.terminated | self.truncated,
                                   "terminated": self.terminated,
                                   "truncated": self.truncated}, [4])

        env = FakeEnv()
        td = TensorDict({"info": {"drone_state": env.state.clone()}}, [4])
        keys = ("goal_stop_count", "prev_drone_vel_w", "reward", "terminated", "truncated")
        saved = {key: getattr(env, key).clone() for key in keys}
        stats, age = env.stats.clone(), env.progress_buf.clone()
        ids, keep = torch.tensor([1, 3]), torch.tensor([0, 2])
        with patch("online_refill.reset_policy_slots") as clear_policy:
            fresh = refill_slots(env, None, td, ids, np.random.default_rng(4))
            clear_policy.assert_called_once()
        for key in keys:
            self.assertTrue(torch.equal(getattr(env, key)[keep], saved[key][keep]), key)
            self.assertTrue(bool((getattr(env, key)[ids] == 0).all()), key)
        self.assertTrue(torch.equal(env.stats["return"][keep], stats["return"][keep]))
        self.assertTrue(bool((env.stats["return"][ids] == 0).all()))
        self.assertTrue(torch.equal(env.progress_buf[keep], age[keep]))
        self.assertTrue(bool((env.progress_buf[ids] == 0).all()))
        self.assertTrue(torch.equal(fresh["info", "drone_state"][keep], td["info", "drone_state"][keep]))
        self.assertFalse(bool(fresh["done"][ids].any()))


if __name__ == "__main__":
    unittest.main()
