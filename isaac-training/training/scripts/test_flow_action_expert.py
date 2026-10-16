"""CPU-only training, sampling, checkpoint and Isaac adapter contract tests."""

import ast
import contextlib
import copy
import inspect
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import numpy as np

from lidar_wam.models.action_expert import SuperActionExpertModel
from lidar_wam.models.flow_action_expert import (
    FLOW_ACTION_EXPERT_FORMAT, FlowActionExpertModel,
)
from lidar_wam.runner import super_action_expert as runner
from lidar_wam.runner.super_lee_action_data import SuperLeeActionDataset


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class FakeVAE(torch.nn.Module):
    def register_to_config(self, **_kwargs):
        pass

    def encode(self, image):
        return SimpleNamespace(latent_dist=SimpleNamespace(
            mode=lambda: torch.zeros(len(image), 4, 27, 5)))


class FlowActionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        self.model = FlowActionExpertModel(width=8, depth=1, heads=2, ffn_width=16)
        self.latent = torch.randn(2, 4, 27, 5)
        self.goal = torch.zeros(2, 4)
        self.proprio = torch.zeros(2, 10)
        self.past = torch.randn(2, 30, 3)
        self.mask = torch.ones(2, 30)
        self.target = torch.full((2, 10, 3), .1)
        self.scale = torch.full((1, 1, 3), 2.)
        self.stats = {"goal_mean": [0.] * 4, "goal_std": [1.] * 4,
                      "proprio_mean": [0.] * 10, "proprio_std": [1.] * 10,
                      "past_velocity_scale_mps": 2.}
        self.prepared = (self.latent, self.target, self.target * 2.,
                         self.goal, self.proprio)

    def args(self, directory, mode="flow"):
        checkpoint = Path(directory) / "vae.safetensors"
        checkpoint.touch()
        return runner.parse_args([
            "--train-root", directory, "--vae-checkpoint", str(checkpoint),
            "--width", "8", "--depth", "1", "--heads", "2", "--ffn-width", "16"],
            default_policy_type=mode)

    def payload(self, args, model=None):
        model = self.model if model is None else model
        return runner._payload(model, torch.optim.AdamW(model.parameters()),
                               1, args, self.stats, {}, 1., "test")

    def test_no_direct_branch_and_ten_steps_by_default(self):
        self.assertFalse(hasattr(self.model, "policy"))
        self.assertFalse(any(key.startswith("policy.") for key in self.model.state_dict()))
        self.assertEqual(self.model.sample_steps, 10)
        with mock.patch.object(self.model.diffusion, "forward",
                               wraps=self.model.diffusion.forward) as forward:
            action = self.model.predict_action(
                self.latent, self.goal, self.proprio,
                generator=torch.Generator().manual_seed(18))
        self.assertEqual(forward.call_count, 10)
        self.assertEqual(tuple(action.shape), (2, 10, 3))
        self.assertTrue(bool((action.abs() <= 1.000001).all()))
        self.assertTrue(bool((action.norm(dim=-1) > 1.0).any()))

    def test_diagonal_speed_is_not_radially_rescaled(self):
        # Initial zero flow leaves this known integral unchanged. It is a
        # valid command of (1.9, 1.9, 1.9) m/s, total speed about 3.291 m/s.
        noise = torch.full((2, 10, 3), .95)
        action = self.model.predict_action(self.latent, self.goal, self.proprio,
                                            initial_noise=noise)
        self.assertTrue(torch.equal(action, noise))
        self.assertAlmostEqual(float((action * 2).norm(dim=-1)[0, 0]),
                               1.9 * math.sqrt(3), places=5)

    def test_history_modules_and_api_are_removed_and_context_has_137_tokens(self):
        self.assertFalse(hasattr(self.model, "use_past_actions"))
        for function in (self.model.__init__, self.model.forward, self.model.predict_action):
            self.assertFalse(any("past" in key for key in inspect.signature(function).parameters))
        self.assertFalse(any("past" in key or "candidate" in key for key in self.model.state_dict()))
        noisy, timestep = torch.randn(2, 10, 3), torch.full((2,), 500.)
        attention = self.model.diffusion.blocks[0].cross_attn
        with torch.no_grad(), mock.patch.object(attention, "forward", wraps=attention.forward) as attend:
            result = self.model(self.latent, noisy, timestep, self.goal, self.proprio)
        self.assertEqual(tuple(result.shape), (2, 10, 3))
        self.assertEqual(tuple(attend.call_args.args[1].shape), (2, 137, 8))
        with self.assertRaises(TypeError):
            self.model.predict_action(self.latent, self.goal, self.proprio, self.past, self.mask)

    def test_dataset_returns_only_current_inputs_and_never_reads_past_rows(self):
        class GuardedHandle(dict):
            def __getitem__(self, key):
                if key == "steps/world_velocity":
                    raise AssertionError("historical velocity must not be read")
                return super().__getitem__(key)

        state = np.zeros((2, 13), np.float32)
        state[:, 3] = 1.
        handle = GuardedHandle(frames={
            "range_values": np.zeros((2, 2, 108, 20), np.float32),
            "drone_state": state, "world_action_sequence": np.ones((2, 10, 3), np.float32),
            "frame_index": np.asarray([3, 4]),
        })
        data = SuperLeeActionDataset.__new__(SuperLeeActionDataset)
        data._handles = {}
        data.include_history = False
        data._location = lambda _index: (0, 0, 0, 1, 30, 0, handle)
        data.conditions = lambda _index: (np.zeros(4, np.float32), np.zeros(10, np.float32))
        item = data[0]
        self.assertEqual(len(item), 5)
        self.assertEqual([tuple(value.shape) for value in item],
                         [(2, 108, 20), (10, 3), (4,), (10,), (4,)])
        # The same guarded fixture proves that the legacy path would read
        # history, while the current-only path above does not even access it.
        data.include_history = True
        with self.assertRaisesRegex(AssertionError, "historical velocity"):
            data[0]

    def test_batch_preparation_has_no_history_values_or_normalization(self):
        batch = (torch.zeros(2, 2, 108, 20), self.target * 2,
                 self.goal, self.proprio, torch.zeros(2, 4, dtype=torch.long))
        stats = {key: value for key, value in self.stats.items() if "past" not in key}
        with mock.patch.object(runner, "_encode_current", return_value=self.latent):
            prepared = runner._prepare(batch, None, 1., self.scale, stats,
                                       torch.device("cpu"), "fp32")
        self.assertEqual(len(prepared), 5)
        self.assertTrue(torch.equal(prepared[1], self.target))

        # Training labels are only divided by the per-axis scale. Never
        # truncate diagonal speeds >2 m/s or rare rotated-axis outliers.
        physical = torch.tensor([2.14, -2.03, 1.95]).expand(2, 10, 3)
        batch = (batch[0], physical, *batch[2:])
        with mock.patch.object(runner, "_encode_current", return_value=self.latent):
            prepared = runner._prepare(batch, None, 1., self.scale, stats,
                                       torch.device("cpu"), "fp32")
        self.assertTrue(torch.equal(prepared[1], physical / 2.))
        self.assertTrue(torch.equal(prepared[2], physical))

    def test_loss_is_only_flow_mse_and_encoder_gets_gradients(self):
        torch.manual_seed(18)
        noise = torch.randn_like(self.target)
        weights = torch.ones(10)
        weights[0] = 4.
        weights /= weights.mean()
        expected = ((noise - self.target).square() * weights[None, :, None]).mean()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
        with mock.patch.object(runner, "_prepare", return_value=self.prepared):
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(18)
                loss, metrics = runner._loss_batch(
                    self.model, None, None, 1., self.scale, self.stats,
                    torch.device("cpu"), "fp32", 1., 0., 0.,
                    1., .5, 4., 2., 3., 0.)
                self.assertEqual(set(metrics), {"loss", "flow_mse", "terminal_fraction", "hard_fraction"})
                self.assertTrue(torch.equal(loss.detach(), metrics["flow_mse"]))
                if step == 0:
                    self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
                loss.backward()
                self.assertTrue(all(parameter.grad is not None for parameter in
                                    self.model.parameters() if parameter.requires_grad))
                if step == 1:
                    self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in
                                           self.model.observation.parameters()), 0.)
                optimizer.step()

    def test_validation_samples_without_teacher_actions_and_repeats_noise(self):
        with mock.patch.object(runner, "_prepare", return_value=self.prepared), \
                mock.patch.object(self.model, "predict_action", wraps=self.model.predict_action) as predict:
            first = runner._validate(self.model, None, [None], 1., self.scale,
                                     self.stats, torch.device("cpu"), "fp32", .5)
            torch.manual_seed(99)
            second = runner._validate(self.model, None, [None], 1., self.scale,
                                      self.stats, torch.device("cpu"), "fp32", .5)
        self.assertEqual(first, second)
        self.assertEqual(first["sample_steps"], 10)
        self.assertEqual(first["samples"], 2)
        self.assertTrue(torch.equal(predict.call_args_list[0].kwargs["initial_noise"],
                                    predict.call_args_list[1].kwargs["initial_noise"]))
        self.assertEqual(len(predict.call_args_list[0].args), 3)

    def test_validation_keeps_valid_diagonal_commands_above_2_mps(self):
        noise = torch.full((2, 10, 3), .95)
        prepared = (self.latent, noise, noise * 2., self.goal, self.proprio)
        with mock.patch.object(runner, "_prepare", return_value=prepared), \
                mock.patch("torch.randn", return_value=noise):
            result = runner._validate(self.model, None, [None], 1., self.scale,
                                      self.stats, torch.device("cpu"), "fp32", .5)
        self.assertAlmostEqual(result["action_l1_mps"], 0.)

    def test_distributed_validation_reduces_sums_not_batch_means(self):
        with mock.patch.object(runner, "_prepare", return_value=self.prepared):
            reference = runner._validate(self.model, None, [None], 1., self.scale,
                                         self.stats, torch.device("cpu"), "fp32", .5)
            with mock.patch.object(runner.dist, "is_initialized", return_value=True), \
                    mock.patch.object(runner.dist, "all_reduce", side_effect=lambda value: value.mul_(2)):
                distributed = runner._validate(self.model, None, [None], 1., self.scale,
                                               self.stats, torch.device("cpu"), "fp32", .5)
        self.assertEqual(distributed["samples"], reference["samples"] * 2)
        self.assertEqual({key: value for key, value in distributed.items() if key != "samples"},
                         {key: value for key, value in reference.items() if key != "samples"})

    def test_cli_metadata_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            self.assertEqual((args.policy_type, args.direct_weight, args.delta_weight), ("flow", 0., 0.))
            self.assertEqual(args.sample_steps, 10)
            self.assertFalse(hasattr(args, "past_dropout"))
            self.assertFalse(hasattr(args, "deployment_use_past_actions"))
            self.assertEqual(args.run_name, "action_expert_flow_history1_5m_v1")
            model = runner._build_model(args)
            payload = self.payload(args, model)
            self.assertEqual(payload["format"], FLOW_ACTION_EXPERT_FORMAT)
            self.assertTrue(payload["architecture"]["deployment_action_diffusion"])
            self.assertFalse(payload["architecture"]["has_direct_head"])
            self.assertEqual(payload["architecture"]["sampling_steps"], 10)
            self.assertNotIn("past_horizon", payload["architecture"])
            self.assertNotIn("past_action_conditioning", payload)
            self.assertNotIn("past_velocity_scale_mps", payload["condition_stats"])
            interface = payload["action_interface"]
            self.assertEqual(interface["normalized_output_constraint"], "unit_box_after_euler")
            self.assertEqual(interface["physical_output_constraint"], "per_axis")
            self.assertEqual(interface["physical_axis_limit_mps_xyz"], [2., 2., 2.])
            self.assertIsNone(interface["physical_norm_limit_mps"])
            for scale in ("1,2,2", "2,nan,2", "2,inf,2"):
                with self.assertRaises(ValueError):
                    runner.parse_args(["--train-root", directory, "--vae-checkpoint",
                                       str(args.vae_checkpoint), "--action-scale-mps-xyz", scale],
                                      default_policy_type="flow")
            restored = runner._build_model(args)
            restored.load_state_dict(payload["model"], strict=True)
            self.assertTrue(all(torch.equal(value, restored.state_dict()[key])
                                for key, value in model.state_dict().items()))
            legacy = self.args(directory, "direct")
            self.assertEqual((legacy.direct_weight, legacy.delta_weight), (4., .2))
            self.assertIsInstance(runner._build_model(legacy), SuperActionExpertModel)
            self.assertEqual(legacy.run_name, "action_dit_final_lee_ego_history1_5m_v3")
            self.assertEqual(legacy.past_dropout, .25)
            self.assertFalse(legacy.deployment_use_past_actions)
        help_text = io.StringIO()
        with contextlib.redirect_stdout(help_text), self.assertRaises(SystemExit):
            runner.parse_args(["--help"], default_policy_type="flow")
        self.assertNotIn("--past-dropout", help_text.getvalue())
        self.assertNotIn("--deployment-use-past-actions", help_text.getvalue())

    def test_resume_restores_adam_moments_and_next_update(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            original = runner._build_model(args)
            optimizer = torch.optim.AdamW(original.parameters(), lr=args.lr, betas=(.9, .95))

            def update(model, adam):
                adam.zero_grad(set_to_none=True)
                loss = model(self.latent, self.target, torch.full((2,), 500.),
                             self.goal, self.proprio).square().mean()
                # Ensure the initially zero output head receives a gradient.
                loss = loss + (model.diffusion.out.bias - .3).square().mean()
                loss.backward()
                adam.step()

            update(original, optimizer)
            payload = runner._payload(original, optimizer, 1, args, self.stats, {}, 1., "test")
            payload = copy.deepcopy(payload)
            restored = runner._build_model(args)
            restored_adam = torch.optim.AdamW(restored.parameters(), lr=args.lr, betas=(.9, .95))
            runner._restore_resume_state(payload, restored, restored_adam, args)
            self.assertTrue(all(int(value["step"]) == 1 for value in restored_adam.state.values()))
            update(original, optimizer)
            update(restored, restored_adam)
            self.assertTrue(all(torch.equal(value, restored.state_dict()[key])
                                for key, value in original.state_dict().items()))
            wrong = copy.deepcopy(payload)
            wrong["action_interface"]["normalized_output_constraint"] = "unit_l2_ball"
            with self.assertRaisesRegex(ValueError, "per-axis"):
                runner._restore_resume_state(wrong, restored, restored_adam, args)
            wrong = copy.deepcopy(payload)
            wrong.pop("optimizer")
            with self.assertRaisesRegex(ValueError, "optimizer"):
                runner._restore_resume_state(wrong, restored, restored_adam, args)

    def test_resume_skips_only_initial_sampler_batches_without_reading_data(self):
        base = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(range(20)), batch_size=3, drop_last=True)
        resumed = runner.SkipInitialBatches(base, 2)
        self.assertEqual(len(resumed), 6)
        self.assertEqual(list(resumed), [[6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]])
        self.assertEqual(list(resumed), list(base))

    def test_isaac_adapter_loads_flow_and_legacy_without_simulator(self):
        # Execute ONLY the policy class and literal format constants. Do not
        # import hydra/SimulationApp, initialize a GPU, or create a real scene.
        path = Path(__file__).with_name("wam_closed_loop_eval.py")
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body if
                    (isinstance(node, ast.ClassDef) and node.name == "WamLeePolicy") or
                    (isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id.endswith("FORMAT")
                        for target in node.targets))]
        namespace = {"Path": Path, "json": json, "math": math}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
        policy_class = namespace["WamLeePolicy"]
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("flow", "direct"):
                args = self.args(directory, mode)
                model = runner._build_model(args)
                payload = self.payload(args, model)
                (Path(directory) / "config.json").write_text('{"scaling_factor":1.0}')
                cfg = AttrDict(device="cpu", env=AttrDict(num_envs=2), sim=AttrDict(dt=.016),
                    wam_eval=AttrDict(replan_interval_steps=10, sample_steps=10,
                        action_generation="auto", action_noise_scale=1., action_seed=18,
                        yaw_speed_threshold=.2, max_yaw_rate=2.5, max_reference_acceleration=6.,
                        safety_filter=AttrDict(enabled=False, candidate_count=31,
                            max_heading_offset_deg=100., corridor_radius_m=.65, lookahead_m=3.,
                            vertical_half_extent_m=.6, obstacle_cost=4., heading_cost=.35,
                            continuity_cost=.2, boundary_margin_m=.8, boundary_cost=20.,
                            stop_clearance_m=.45, full_speed_clearance_m=1.5)))
                env = SimpleNamespace(target_pos=torch.tensor([[[10., 0., 0.]]] * 2))
                with mock.patch("torch.load", return_value=payload), \
                        mock.patch.object(runner.stage1, "make_vae", return_value=FakeVAE()), \
                        mock.patch.object(runner.stage1, "load_file", return_value={}):
                    policy = policy_class(cfg, env, None, Path(directory) / "model.pt",
                                          Path(directory) / "unused_world.pt")
                self.assertTrue(policy.is_standalone_action_expert)
                self.assertEqual(policy.is_flow_only_action_expert, mode == "flow")
                self.assertEqual(policy.use_flow_generation, mode == "flow")
                if mode == "flow":
                    self.assertFalse(hasattr(policy, "past"))
                    self.assertFalse(hasattr(policy, "past_world"))
                    self.assertFalse(hasattr(policy, "past_mask"))
                policy._lidar_image = lambda: torch.zeros(2, 2, 108, 20)
                state = torch.zeros(2, 13)
                state[:, 3] = 1.
                policy.lee = lambda *_args, **_kwargs: torch.zeros(2, 1, 4)

                class FakeTensorDict(dict):
                    def set(self, key, value):
                        self[key] = value
                        return self

                td = FakeTensorDict({("info", "drone_state"): state[:, None]})
                with mock.patch("torch.autocast", return_value=contextlib.nullcontext()), \
                        mock.patch.object(policy.model.diffusion, "forward",
                                          wraps=policy.model.diffusion.forward) as forward:
                    policy(td)
                self.assertEqual(forward.call_count, 10 if mode == "flow" else 0)
                self.assertEqual(tuple(policy.chunk.shape), (2, 10, 3))
                if mode == "flow":
                    self.assertTrue(bool((policy.chunk.abs() <= 2.000001).all()))
                else:
                    self.assertTrue(bool((policy.chunk.norm(dim=-1) <= 2.000001).all()))
                if mode == "direct":
                    self.assertTrue(bool((policy.past_mask[:, -1] == 1).all()))

                # A real policy call must send the complete diagonal command
                # through the original Lee target_vel interface, even after
                # yaw rotation makes one WORLD component exceed 2 m/s.
                state[:, 3] = math.cos(math.pi / 8)
                state[:, 6] = math.sin(math.pi / 8)
                for normalized in ([.95, .95, .95], [1.4, -1.3, .7]):
                    raw = torch.tensor(normalized).expand(2, 10, 3).clone()
                    physical = raw * 2.
                    expected = (physical.clamp(-2., 2.) if mode == "flow" else
                                physical * (2. / physical.norm(dim=-1, keepdim=True)).clamp_max(1.))
                    policy.step_index = 10
                    policy.lee = mock.Mock(return_value=torch.zeros(2, 1, 4))
                    with mock.patch("torch.autocast", return_value=contextlib.nullcontext()), \
                            mock.patch.object(policy.model, "predict_action", return_value=raw):
                        policy(td)
                    self.assertTrue(torch.allclose(policy.chunk, expected, atol=1e-6))
                    world = policy._ego_to_world(expected[:, 0], state[:, 3:7])
                    self.assertTrue(torch.allclose(policy.last_world_command, world, atol=1e-6))
                    self.assertTrue(torch.allclose(policy.lee.call_args.kwargs["target_vel"],
                                                   world[:, None], atol=1e-6))
                    self.assertTrue(torch.allclose(world.norm(dim=-1),
                                                   expected[:, 0].norm(dim=-1), atol=1e-6))
                    if mode == "flow":
                        self.assertGreater(float(world.norm(dim=-1).min()), 3.)
                        self.assertGreater(float(world.abs().max()), 2.)
                    else:
                        self.assertLessEqual(float(world.norm(dim=-1).max()), 2.000001)

                if mode == "flow":
                    # An optional first-chunk bootstrap must obey the same
                    # per-axis limit, without bypassing the final bound.
                    policy.bootstrap_action_chunks = policy.inference_calls + 1
                    policy.bootstrap_forward_speed = 3.
                    with mock.patch("torch.autocast", return_value=contextlib.nullcontext()):
                        policy._sample_chunk(state, torch.zeros(2, 3))
                    self.assertTrue(torch.equal(policy.chunk[..., 0], torch.full((2, 10), 2.)))
                    self.assertTrue(torch.equal(policy.chunk[..., 1:], torch.zeros(2, 10, 2)))


if __name__ == "__main__":
    unittest.main()
