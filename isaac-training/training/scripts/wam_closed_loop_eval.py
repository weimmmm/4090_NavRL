"""Evaluate the trained SUPER dual-DiT policy in closed-loop Isaac Sim.

No SUPER planner is created here.  The corrected interface interprets each
Action-DiT output as a goal-frame reference velocity.  A deterministic adapter
integrates reference position, differentiates/clamps reference acceleration,
derives yaw, and calls the original unmodified LeePositionController.
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


OMNIDRONES_SOURCE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "OmniDrones")
)
LIDAR_WAM_SOURCE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "lidar_WAM")
)
LIDAR_WAM_PYTHON_DEPS = os.path.join(
    LIDAR_WAM_SOURCE, "third_party", "python_deps")
for source_path in (
        OMNIDRONES_SOURCE, LIDAR_WAM_SOURCE, LIDAR_WAM_PYTHON_DEPS):
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

import hydra
from omni.isaac.kit import SimulationApp


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
FORMAT = "navrl-super-history-world-action-dit-t1-v1"
WORLD_FORMAT = "navrl-super-lidar-video-dit-t1-v1"
ACTION_EXPERT_FORMAT = "navrl-super-action-expert-deterministic-v1"
LEE_ACTION_EXPERT_FORMAT = "navrl-super-action-expert-lee-goalframe-v2"
FINAL_LEE_ACTION_EXPERT_FORMAT = "navrl-action-dit-final-lee-ego-bc-v1"
HISTORY3_FINAL_LEE_ACTION_EXPERT_FORMAT = "navrl-action-dit-final-lee-ego-bc-history3-v1"
HISTORY3_CURRENT_LEE_ACTION_EXPERT_FORMAT = (
    "navrl-action-dit-final-lee-ego-bc-history3-v2")
HISTORY1_CURRENT_LEE_ACTION_EXPERT_FORMAT = (
    "navrl-action-dit-final-lee-ego-bc-history1-v3")


def _resolve(project_dir: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


class WamLeePolicy:
    """Stateful Action-DiT policy driving the preserved Lee loop."""

    def __init__(self, cfg, env, lee_controller, checkpoint: Path,
                 world_checkpoint: Path):
        import torch

        from lidar_wam.runner import stage1

        self.torch = torch
        self.env = env
        self.lee = lee_controller
        self.device = torch.device(str(cfg.device))
        self.num_envs = int(cfg.env.num_envs)
        self.dt = float(cfg.sim.dt)
        self.interval = int(cfg.wam_eval.replan_interval_steps)
        self.sample_steps = int(cfg.wam_eval.sample_steps)
        self.action_noise_scale = float(cfg.wam_eval.action_noise_scale)
        self.bootstrap_forward_speed = float(
            cfg.wam_eval.get("bootstrap_forward_speed_mps", 0.0))
        self.bootstrap_action_chunks = int(
            cfg.wam_eval.get("bootstrap_action_chunks", 0))
        if self.bootstrap_forward_speed < 0.0:
            raise ValueError("bootstrap_forward_speed_mps must be non-negative")
        if self.bootstrap_action_chunks < 0:
            raise ValueError("bootstrap_action_chunks must be non-negative")
        requested_past_mode = cfg.wam_eval.get("use_past_actions", "auto")
        self.record_lidar = bool(cfg.wam_eval.get("record_lidar", False))
        self.recorded_lidar_steps = []
        self.recorded_lidar_images = []
        self.yaw_speed_threshold = float(cfg.wam_eval.yaw_speed_threshold)
        self.max_yaw_rate = float(cfg.wam_eval.max_yaw_rate)
        self.max_reference_acceleration = float(
            cfg.wam_eval.max_reference_acceleration)
        self.max_reference_speed = float(
            cfg.wam_eval.get("max_reference_speed", 2.0))
        if self.max_reference_speed <= 0.0:
            raise ValueError("max_reference_speed must be positive")
        goal_stop = cfg.wam_eval.get("goal_stop_adapter", {})
        self.goal_stop_enabled = bool(goal_stop.get("enabled", False))
        self.goal_stop_distance = float(
            goal_stop.get("enter_distance_m", 1.0))
        self.goal_stop_speed = float(goal_stop.get("enter_speed_mps", 1.0))
        self.goal_stop_interventions = 0
        self.goal_braking = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        if self.goal_stop_distance <= 0.0 or self.goal_stop_speed < 0.0:
            raise ValueError("invalid goal_stop_adapter thresholds")
        safety = cfg.wam_eval.safety_filter
        self.safety_enabled = bool(safety.enabled)
        self.safety_candidate_count = int(safety.candidate_count)
        self.safety_max_offset = math.radians(
            float(safety.max_heading_offset_deg))
        self.safety_corridor_radius = float(safety.corridor_radius_m)
        self.safety_lookahead = float(safety.lookahead_m)
        self.safety_vertical_half_extent = float(
            safety.vertical_half_extent_m)
        self.safety_obstacle_cost = float(safety.obstacle_cost)
        self.safety_heading_cost = float(safety.heading_cost)
        self.safety_continuity_cost = float(safety.continuity_cost)
        self.safety_boundary_margin = float(safety.boundary_margin_m)
        self.safety_boundary_cost = float(safety.boundary_cost)
        self.safety_stop_clearance = float(safety.stop_clearance_m)
        self.safety_full_speed_clearance = float(
            safety.full_speed_clearance_m)
        if self.interval != 10:
            raise ValueError("dual-DiT checkpoint requires replanning every 10 physics steps")

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_format = payload.get("format")
        self.is_deterministic_action_expert = checkpoint_format in {
            ACTION_EXPERT_FORMAT, LEE_ACTION_EXPERT_FORMAT,
            FINAL_LEE_ACTION_EXPERT_FORMAT,
            HISTORY3_FINAL_LEE_ACTION_EXPERT_FORMAT,
            HISTORY3_CURRENT_LEE_ACTION_EXPERT_FORMAT,
            HISTORY1_CURRENT_LEE_ACTION_EXPERT_FORMAT}
        self.required_history_frames = 3
        if checkpoint_format not in {
                FORMAT, ACTION_EXPERT_FORMAT, LEE_ACTION_EXPERT_FORMAT,
                FINAL_LEE_ACTION_EXPERT_FORMAT,
                HISTORY3_FINAL_LEE_ACTION_EXPERT_FORMAT,
                HISTORY3_CURRENT_LEE_ACTION_EXPERT_FORMAT,
                HISTORY1_CURRENT_LEE_ACTION_EXPERT_FORMAT}:
            raise ValueError(
                f"unsupported policy checkpoint: {checkpoint_format!r}")
        if str(requested_past_mode).lower() == "auto":
            self.use_past_actions = bool(payload.get(
                "past_action_conditioning", {}).get(
                    "deployment_use_past_actions", True))
        elif isinstance(requested_past_mode, bool):
            self.use_past_actions = requested_past_mode
        elif str(requested_past_mode).lower() in {"true", "1", "yes"}:
            self.use_past_actions = True
        elif str(requested_past_mode).lower() in {"false", "0", "no"}:
            self.use_past_actions = False
        else:
            raise ValueError(
                "wam_eval.use_past_actions must be auto, true, or false")
        self.past_input_semantics = str(payload.get(
            "past_action_conditioning", {}).get(
                "input_semantics", "reference_velocity"))
        if self.past_input_semantics not in {
            "reference_velocity", "actual_velocity", "actual_vehicle_velocity",
            "measured_ego_velocity",
            "measured_velocity_in_current_observation_ego_frame"}:
            raise ValueError(
                f"unsupported checkpoint past input {self.past_input_semantics!r}")
        self.mask_world_position = bool(payload.get(
            "proprio_conditioning", {}).get("mask_world_position", False))
        self.use_proprio = bool(payload.get(
            "proprio_conditioning", {}).get("deployment_use_proprio", True))
        architecture = dict(payload.get("architecture", {}))
        if self.is_deterministic_action_expert:
            from lidar_wam.models.action_expert import SuperActionExpertModel

            if bool(architecture.get("deployment_action_diffusion", True)):
                raise ValueError(
                    "Action Expert checkpoint is not marked for deterministic deployment")
            self.lidar_history_frames = int(
                architecture.get("history_lidar_frames", 1))
            if self.lidar_history_frames not in (1, 3):
                raise ValueError(
                    f"unsupported LiDAR history length {self.lidar_history_frames}")
            if (checkpoint_format in {
                    HISTORY3_FINAL_LEE_ACTION_EXPERT_FORMAT,
                    HISTORY3_CURRENT_LEE_ACTION_EXPERT_FORMAT}
                    and self.lidar_history_frames != 3):
                raise ValueError("history3 checkpoint must declare 3 LiDAR frames")
            if (checkpoint_format == HISTORY1_CURRENT_LEE_ACTION_EXPERT_FORMAT
                    and self.lidar_history_frames != 1):
                raise ValueError("history1 checkpoint must declare 1 LiDAR frame")
            model = SuperActionExpertModel(
                width=int(architecture.get("width", 512)),
                depth=int(architecture.get("depth", 8)),
                heads=int(architecture.get("heads", 8)),
                ffn_width=int(architecture.get("ffn_width", 2048)),
                horizon=int(architecture.get("action_horizon", 10)),
                action_dim=int(architecture.get("action_dim", 3)),
                proprio_dim=int(architecture.get("proprio_dim", 10)),
                past_horizon=int(architecture.get("past_horizon", 30)),
                lidar_history_frames=self.lidar_history_frames,
            )
            self.required_history_frames = self.lidar_history_frames
            self.flow_sample_action = None
        else:
            from lidar_wam.models.action_dit import (
                JointHistoryWorldActionDiT,
                flow_sample_action,
            )
            from lidar_wam.models.lidar_video_dit import LiDARVideoDiT

            self.flow_sample_action = flow_sample_action
            world_payload = torch.load(
                world_checkpoint, map_location="cpu", weights_only=False)
            if world_payload.get("format") != WORLD_FORMAT:
                raise ValueError(
                    f"unsupported World-DiT checkpoint: {world_payload.get('format')!r}")
            world_arch = dict(world_payload.get("architecture", {}))
            model = JointHistoryWorldActionDiT(
                LiDARVideoDiT(
                    width=int(world_arch.get("width", 512)),
                    depth=int(world_arch.get("depth", 8)),
                    heads=int(world_arch.get("heads", 8)),
                    mlp_ratio=float(world_arch.get("mlp_ratio", 4.0))),
                width=int(world_arch.get("width", 512)),
                depth=int(architecture.get("action_depth", 8)),
                heads=int(world_arch.get("heads", 8)),
                mlp_ratio=float(world_arch.get("mlp_ratio", 4.0)),
                past_horizon=int(architecture.get("past_horizon", 30)),
                shared_world_depth=int(architecture.get("shared_world_depth", 6)),
                action_horizon=int(architecture.get("action_horizon", 10)),
                action_dim=int(architecture.get("action_dim", 3)),
                proprio_dim=int(architecture.get("proprio_dim", 13)),
            )
        model.load_state_dict(payload["model"], strict=True)
        self.model = model.to(self.device).float().eval().requires_grad_(False)
        action_interface = dict(payload.get("action_interface", {}))
        self.action_semantics = str(action_interface.get(
            "semantics", "lee_velocity_only_equivalent"))
        self.action_frame = str(action_interface.get(
            "frame", "yaw_only_ego_flu"))
        self.action_target_representation = str(action_interface.get(
            "target_representation", "absolute_reference_velocity"))
        scale_key = ("model_output_scale_mps_xyz"
                     if "model_output_scale_mps_xyz" in action_interface
                     else "model_output_scale_xyz")
        self.action_target_scale = torch.as_tensor(
            action_interface.get(scale_key, [1.0, 1.0, 1.0]),
            device=self.device, dtype=torch.float32).view(1, 1, 3)
        if (self.action_target_scale.shape != (1, 1, 3)
                or not bool(torch.isfinite(self.action_target_scale).all())
                or bool((self.action_target_scale <= 0.0).any())):
            raise ValueError("invalid action model_output_scale_xyz")
        if self.action_semantics not in {
            "final_lee_target_velocity",
            "actual_velocity_input_to_original_lee",
            "lee_velocity_only_equivalent",
            "lee_velocity_only_equivalent_goal_frame",
            "super_reference_velocity",
            "super_reference_velocity_residual",
        }:
            raise ValueError(
                f"unsupported action semantics {self.action_semantics!r}")
        if (self.action_semantics in {
                "lee_velocity_only_equivalent_goal_frame",
                "super_reference_velocity",
                "super_reference_velocity_residual"}
                and self.action_frame != "current_position_to_goal_yaw_frame"):
            raise ValueError(
                f"unsupported reference-velocity frame {self.action_frame!r}")

        vae_metadata = dict(payload.get("vae", {}))
        vae_checkpoint = Path(str(vae_metadata.get("checkpoint", "")))
        if not vae_checkpoint.is_file():
            # Checkpoints may move between the host and the Isaac container.
            # Recover the selected VAE from its stable project-relative suffix.
            try:
                suffix = vae_checkpoint.parts[
                    vae_checkpoint.parts.index("training"):]
                vae_checkpoint = Path(__file__).resolve().parents[2].joinpath(
                    *suffix)
            except (ValueError, IndexError):
                pass
        config_path = vae_checkpoint.parent / "config.json"
        if not vae_checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"5 m VAE weights/config missing: {vae_checkpoint}")
        vae_config = json.loads(config_path.read_text())
        loaded_scale = float(vae_config["scaling_factor"])
        self.vae = stage1.make_vae(circular=True)
        self.vae.load_state_dict(
            stage1.load_file(str(vae_checkpoint), device="cpu"), strict=True)
        self.vae.register_to_config(scaling_factor=loaded_scale)
        self.vae = self.vae.to(self.device).float().eval().requires_grad_(False)
        self.scale = float(vae_metadata["scaling_factor"])
        if not math.isclose(self.scale, loaded_scale, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"checkpoint/VAE latent scale mismatch: {self.scale} vs "
                f"{loaded_scale}")
        self.stats = {
            key: torch.as_tensor(value, device=self.device, dtype=torch.float32)
            for key, value in payload["condition_stats"].items()
            if key in ("goal_mean", "goal_std", "proprio_mean", "proprio_std")
        }
        self.action_seed = int(cfg.wam_eval.action_seed)
        self.generator = (None if self.is_deterministic_action_expert else
                          torch.Generator(device=self.device).manual_seed(
                              self.action_seed))
        self.history = []
        self.past = torch.zeros(self.num_envs, 30, 3, device=self.device)
        self.past_world = torch.zeros_like(self.past)
        self.past_mask = torch.zeros(self.num_envs, 30, device=self.device)
        self.chunk = torch.zeros(self.num_envs, 10, 3, device=self.device)
        self.previous_velocity = None
        self.reference_position = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.previous_reference_velocity = torch.zeros_like(
            self.reference_position)
        self.reference_started = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self.reference_initialized = False
        self.reference_yaw = torch.zeros(
            self.num_envs, device=self.device)
        self.chunk_frame_yaw = torch.zeros_like(self.reference_yaw)
        self.safety_heading = torch.zeros_like(self.reference_yaw)
        self.safety_initialized = False
        self.safety_interventions = 0
        self.active = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device)
        self.step_index = 0
        self.checkpoint_step = int(payload.get("step", -1))
        self.last_ego_command = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.last_world_command = torch.zeros_like(self.last_ego_command)
        self.last_reference_velocity_world = torch.zeros_like(
            self.last_ego_command)
        self.command_abs_max = 0.0
        self.inference_calls = 0

    def set_active(self, active):
        self.active.copy_(active)

    def _lidar_image(self):
        """Exactly reproduce ``isaac_eval.collect_dataset._lidar_range_image``."""
        torch = self.torch
        horizontal, vertical = self.env.lidar_raw_resolution
        hits = self.env.lidar.data.ray_hits_w.reshape(
            self.num_envs, horizontal, vertical, 3)
        sensor = self.env.lidar.data.pos_w[:, None, None, :]
        distances = (hits - sensor).norm(dim=-1)
        valid = (
            torch.isfinite(hits).all(dim=-1)
            & (distances <= float(self.env.lidar_range)))
        distances = torch.where(
            valid, distances,
            torch.full_like(distances, float(self.env.lidar_range)))
        image = torch.ones(
            self.num_envs, 2, 108, 20, device=self.device, dtype=torch.float32)
        image[:, 1].fill_(-1.0)
        image[:, 0, :, :18] = (
            2.0 * distances / float(self.env.lidar_range) - 1.0)
        image[:, 1, :, :18] = (
            valid.to(torch.float32) * 2.0 - 1.0)
        if self.record_lidar:
            self.recorded_lidar_steps.append(int(self.step_index))
            self.recorded_lidar_images.append(
                image.detach().cpu().to(torch.float16).numpy())
        return image

    @staticmethod
    def _world_to_ego(vector, quaternion):
        torch = __import__("torch")
        w, x, y, z = quaternion.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y),
                          1.0 - 2.0 * (y * y + z * z))
        cosine, sine = yaw.cos(), yaw.sin()
        return torch.stack((
            cosine * vector[..., 0] + sine * vector[..., 1],
            -sine * vector[..., 0] + cosine * vector[..., 1],
            vector[..., 2],
        ), dim=-1)

    @staticmethod
    def _ego_to_world(vector, quaternion):
        torch = __import__("torch")
        w, x, y, z = quaternion.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y),
                          1.0 - 2.0 * (y * y + z * z))
        cosine, sine = yaw.cos(), yaw.sin()
        return torch.stack((
            cosine * vector[..., 0] - sine * vector[..., 1],
            sine * vector[..., 0] + cosine * vector[..., 1],
            vector[..., 2],
        ), dim=-1)

    @staticmethod
    def _world_to_yaw_frame(vector, frame_yaw):
        torch = __import__("torch")
        cosine, sine = frame_yaw.cos(), frame_yaw.sin()
        while cosine.ndim < vector.ndim - 1:
            cosine = cosine.unsqueeze(-1)
            sine = sine.unsqueeze(-1)
        return torch.stack((
            cosine * vector[..., 0] + sine * vector[..., 1],
            -sine * vector[..., 0] + cosine * vector[..., 1],
            vector[..., 2],
        ), dim=-1)

    @staticmethod
    def _yaw_frame_to_world(vector, frame_yaw):
        torch = __import__("torch")
        cosine, sine = frame_yaw.cos(), frame_yaw.sin()
        while cosine.ndim < vector.ndim - 1:
            cosine = cosine.unsqueeze(-1)
            sine = sine.unsqueeze(-1)
        return torch.stack((
            cosine * vector[..., 0] - sine * vector[..., 1],
            sine * vector[..., 0] + cosine * vector[..., 1],
            vector[..., 2],
        ), dim=-1)

    @staticmethod
    def _quaternion_yaw(quaternion):
        torch = __import__("torch")
        w, x, y, z = quaternion.unbind(-1)
        return torch.atan2(2.0 * (w * z + x * y),
                           1.0 - 2.0 * (y * y + z * z))

    def _goal_frame_yaw(self, position):
        delta = self.env.target_pos[:, 0] - position
        return self.torch.atan2(delta[:, 1], delta[:, 0])

    def _apply_safety_filter(self, reference_velocity, position):
        """VFH-style cylindrical-clearance filter over current LiDAR hits."""
        if not self.safety_enabled:
            return reference_velocity
        torch = self.torch
        horizontal_speed = reference_velocity[:, :2].norm(dim=-1)
        desired_heading = torch.atan2(
            reference_velocity[:, 1], reference_velocity[:, 0])
        fallback_heading = self._goal_frame_yaw(position)
        desired_heading = torch.where(
            horizontal_speed >= 0.05, desired_heading, fallback_heading)
        if not self.safety_initialized:
            self.safety_heading.copy_(desired_heading)
            self.safety_initialized = True

        relative = (
            self.env.lidar.data.ray_hits_w
            - self.env.lidar.data.pos_w.unsqueeze(1))
        distance = relative.norm(dim=-1)
        horizontal = relative[..., :2]
        valid = (
            torch.isfinite(relative).all(dim=-1)
            & (distance > 0.05)
            & (distance < float(self.env.lidar_range) - 1.0e-3)
            & (relative[..., 2].abs() <= self.safety_vertical_half_extent)
        )

        offsets = torch.linspace(
            -self.safety_max_offset,
            self.safety_max_offset,
            self.safety_candidate_count,
            device=self.device,
            dtype=reference_velocity.dtype)
        headings = desired_heading[:, None] + offsets[None]
        directions = torch.stack((headings.cos(), headings.sin()), dim=-1)
        longitudinal = torch.einsum("ncd,nrd->ncr", directions, horizontal)
        lateral = torch.abs(
            directions[..., 0, None] * horizontal[:, None, :, 1]
            - directions[..., 1, None] * horizontal[:, None, :, 0])
        corridor = (
            valid[:, None]
            & (longitudinal > 0.0)
            & (longitudinal <= self.safety_lookahead)
            & (lateral <= self.safety_corridor_radius))
        clearance = torch.where(
            corridor, longitudinal,
            torch.full_like(longitudinal, self.safety_lookahead)).amin(dim=-1)
        obstacle_penalty = self.safety_obstacle_cost * torch.relu(
            self.safety_lookahead - clearance).square()
        heading_penalty = self.safety_heading_cost * offsets.abs()[None]
        continuity_delta = torch.remainder(
            headings - self.safety_heading[:, None] + math.pi,
            2.0 * math.pi) - math.pi
        continuity_penalty = (
            self.safety_continuity_cost * continuity_delta.abs())
        boundary = (float(self.env.cfg.env.world_half_extent)
                    - self.safety_boundary_margin)
        projected_position = position[:, None, :2] + (
            directions * self.safety_lookahead)
        boundary_excess = torch.relu(
            projected_position.abs() - boundary).sum(dim=-1)
        boundary_penalty = (
            self.safety_boundary_cost * boundary_excess.square())
        cost = (obstacle_penalty + heading_penalty + continuity_penalty
                + boundary_penalty)
        choice = cost.argmin(dim=-1)
        selected_heading = headings.gather(1, choice[:, None]).squeeze(1)
        selected_clearance = clearance.gather(
            1, choice[:, None]).squeeze(1)
        moving = horizontal_speed >= 0.05
        selected_heading = torch.where(
            moving, selected_heading, self.safety_heading)
        self.safety_heading.copy_(selected_heading)
        speed_scale = ((selected_clearance - self.safety_stop_clearance)
                       / max(self.safety_full_speed_clearance
                             - self.safety_stop_clearance, 1.0e-3)).clamp(0.0, 1.0)
        speed_scale = torch.where(moving, speed_scale, torch.ones_like(speed_scale))
        filtered = reference_velocity.clone()
        filtered[:, 0] = horizontal_speed * speed_scale * selected_heading.cos()
        filtered[:, 1] = horizontal_speed * speed_scale * selected_heading.sin()
        changed = moving & (
            torch.abs(torch.remainder(
                selected_heading - desired_heading + math.pi,
                2.0 * math.pi) - math.pi) > math.radians(2.0))
        self.safety_interventions += int(changed.sum())
        return filtered

    def _normalise(self, value, name):
        return ((value - self.stats[f"{name}_mean"])
                / self.stats[f"{name}_std"])

    def _sample_chunk(self, state, acceleration_world):
        torch = self.torch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            position = state[:, :3]
            quaternion = state[:, 3:7]
            velocity_ego = self._world_to_ego(state[:, 7:10], quaternion)
            acceleration_ego = self._world_to_ego(
                acceleration_world, quaternion)
            goal_ego = self._world_to_ego(
                self.env.target_pos[:, 0] - position, quaternion)
            goal = torch.cat((goal_ego, goal_ego.norm(dim=-1, keepdim=True)), -1)
            quaternion = quaternion / quaternion.norm(
                dim=-1, keepdim=True).clamp_min(1.0e-6)
            if self.is_deterministic_action_expert:
                proprio = torch.cat(
                    (velocity_ego, acceleration_ego, quaternion), -1)
            else:
                proprio = torch.cat((position, velocity_ego,
                                      acceleration_ego, quaternion), -1)
                if self.mask_world_position:
                    proprio[:, :3] = 0.0
            if self.past_input_semantics == (
                    "measured_velocity_in_current_observation_ego_frame"):
                past = self._world_to_yaw_frame(
                    self.past_world, self._quaternion_yaw(quaternion))
            elif self.action_semantics in {
                    "lee_velocity_only_equivalent_goal_frame",
                    "super_reference_velocity",
                    "super_reference_velocity_residual"}:
                self.chunk_frame_yaw.copy_(self._goal_frame_yaw(position))
                past = self._world_to_yaw_frame(
                    self.past_world, self.chunk_frame_yaw)
            else:
                past = self.past
            if self.use_past_actions:
                model_past = past
                model_past_mask = self.past_mask
            else:
                # Diagnostic/deployment ablation for the train/deploy exposure
                # gap: training sees expert past actions, while closed loop sees
                # its own predictions.  Keeping both values and mask at zero
                # removes that autoregressive feedback without changing the
                # LiDAR, state, action target, Lee controller, or timing.
                model_past = self.torch.zeros_like(past)
                model_past_mask = self.torch.zeros_like(self.past_mask)
            model_proprio = self._normalise(proprio, "proprio")
            if not self.use_proprio:
                model_proprio = self.torch.zeros_like(model_proprio)
            if self.is_deterministic_action_expert:
                if self.lidar_history_frames == 1:
                    latent = (
                        self.vae.encode(self.history[-1]).latent_dist.mode()
                        * self.scale)
                else:
                    images = torch.stack(
                        self.history[-self.lidar_history_frames:], dim=1)
                    flat = images.reshape(
                        self.num_envs * self.lidar_history_frames,
                        2, 108, 20)
                    latent = self.vae.encode(flat).latent_dist.mode() * self.scale
                    latent = latent.reshape(
                        self.num_envs, self.lidar_history_frames, 4, 27, 5)
                past_scale = float(self.stats.get(
                    "past_velocity_scale_mps", 2.0))
                sampled = self.model.predict_action(
                    latent, self._normalise(goal, "goal"), model_proprio,
                    model_past / past_scale, model_past_mask)
            else:
                images = torch.stack(self.history[-3:], dim=1)
                flat = images.reshape(self.num_envs * 3, 2, 108, 20)
                latent = self.vae.encode(flat).latent_dist.mode() * self.scale
                latent = latent.reshape(self.num_envs, 3, 4, 27, 5)
                history_features = self.model.encode_history_features(latent)
                sampled = self.flow_sample_action(
                    self.model.action, history_features,
                    self._normalise(goal, "goal"), model_proprio,
                    model_past, model_past_mask, steps=self.sample_steps,
                    generator=self.generator, clamp_min=None, clamp_max=None,
                    noise_scale=self.action_noise_scale)
        sampled = sampled.float()
        if self.action_semantics in {
                "final_lee_target_velocity",
                "actual_velocity_input_to_original_lee"}:
            # The final-Lee Action Expert is trained in normalized action
            # units.  Its payload records the physical m/s scale explicitly.
            sampled *= self.action_target_scale
            # Keep the learned command inside the same physical domain as the
            # collector.  The labels were norm-clamped to SUPER's 2 m/s
            # maximum, while an unconstrained regression head can overshoot a
            # little at inference (especially after leaving the expert
            # manifold).  This is an adapter-side bound; Lee itself is not
            # modified.
            speed = sampled.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            sampled *= (self.max_reference_speed / speed).clamp_max(1.0)
        else:
            sampled /= self.action_target_scale
        if self.action_semantics == "super_reference_velocity_residual":
            # The network predicts only the visually driven correction.  Add
            # the measured velocity captured at this replanning instant to
            # recover the absolute SUPER reference-velocity contract.
            measured_goal_velocity = self._world_to_yaw_frame(
                state[:, 7:10], self.chunk_frame_yaw)
            sampled += measured_goal_velocity[:, None, :]
        if not bool(torch.isfinite(sampled).all()):
            raise FloatingPointError("Action-DiT generated a non-finite velocity")
        if self.inference_calls < self.bootstrap_action_chunks:
            # Give the first complete ten-step command chunk an exact forward
            # ego velocity.  A single 16 ms command is too short for Lee to
            # establish the motion history expected by the expert dataset.
            # The override expires before the next model-generated chunk.
            sampled.zero_()
            sampled[..., 0] = self.bootstrap_forward_speed
        self.chunk.copy_(sampled)
        self.inference_calls += 1

    def __call__(self, tensordict):
        torch = self.torch
        state = tensordict[("info", "drone_state")][:, 0, :13]
        velocity = state[:, 7:10]
        if self.previous_velocity is None:
            acceleration = torch.zeros_like(velocity)
        else:
            acceleration = (velocity - self.previous_velocity) / self.dt
        self.previous_velocity = velocity.clone()
        position = state[:, :3]
        quaternion = state[:, 3:7]
        if not self.reference_initialized:
            self.reference_position.copy_(position)
            self.reference_yaw.copy_(self._quaternion_yaw(quaternion))
            self.reference_initialized = True

        frame_due = self.step_index % self.interval == 0
        if frame_due:
            self.history.append(self._lidar_image())
            if len(self.history) > 3:
                self.history.pop(0)
            if len(self.history) >= self.required_history_frames:
                self._sample_chunk(state, acceleration)

        ready = len(self.history) >= self.required_history_frames
        action_command = (self.chunk[:, self.step_index % self.interval].clone()
                          if ready else torch.zeros_like(velocity))
        action_command[~self.active] = 0.0
        if (self.goal_stop_enabled
                and self.action_semantics in {
                    "final_lee_target_velocity",
                    "actual_velocity_input_to_original_lee"}):
            goal_distance = (
                self.env.target_pos[:, 0] - position).norm(dim=-1)
            newly_braking = (
                self.active & ready & ~self.goal_braking
                & (goal_distance <= self.goal_stop_distance))
            self.goal_braking |= newly_braking
            if bool(newly_braking.any()):
                self.goal_stop_interventions += int(newly_braking.sum())
            action_command[self.goal_braking] = 0.0
        if self.action_semantics == "lee_velocity_only_equivalent_goal_frame":
            world_command = self._yaw_frame_to_world(
                action_command, self.chunk_frame_yaw)
            world_command[~self.active] = 0.0
            reference_velocity = world_command
            motor_command = self.lee(
                state[:, None, :], target_vel=world_command[:, None, :],
                target_yaw=None)
        elif self.action_semantics in {
                "super_reference_velocity",
                "super_reference_velocity_residual"}:
            reference_velocity = self._yaw_frame_to_world(
                action_command, self.chunk_frame_yaw)
            reference_velocity[~self.active] = 0.0
            # Rectified-flow sampling is unbounded even though every SUPER
            # demonstration was generated with max_vel=2 m/s.  Project the
            # sampled command back onto that expert action domain before any
            # optional safety layer and before the preserved Lee adapter.
            reference_speed = reference_velocity.norm(
                dim=-1, keepdim=True).clamp_min(1.0e-6)
            reference_velocity *= (
                self.max_reference_speed / reference_speed).clamp_max(1.0)
            reference_velocity = self._apply_safety_filter(
                reference_velocity, position)
            reference_velocity[~self.active] = 0.0
            newly_started = self.active & ready & ~self.reference_started
            self.previous_reference_velocity[newly_started] = (
                reference_velocity[newly_started])
            self.reference_started |= self.active & ready
            reference_acceleration = (
                reference_velocity - self.previous_reference_velocity) / self.dt
            acceleration_norm = reference_acceleration.norm(
                dim=-1, keepdim=True).clamp_min(1.0e-6)
            acceleration_scale = (
                self.max_reference_acceleration / acceleration_norm
            ).clamp_max(1.0)
            reference_acceleration *= acceleration_scale

            inactive = ~self.active
            self.reference_position[inactive] = position[inactive]
            position_error = self.reference_position - position
            world_command = (
                reference_velocity
                + (4.0 / 2.2) * position_error
                + reference_acceleration / 2.2
            )
            world_command[inactive] = 0.0

            horizontal_speed = reference_velocity[:, :2].norm(dim=-1)
            desired_yaw = self.torch.atan2(
                reference_velocity[:, 1], reference_velocity[:, 0])
            use_heading = (
                self.active & ready
                & (horizontal_speed >= self.yaw_speed_threshold))
            desired_yaw = self.torch.where(
                use_heading, desired_yaw, self.reference_yaw)
            yaw_delta = self.torch.remainder(
                desired_yaw - self.reference_yaw + math.pi,
                2.0 * math.pi) - math.pi
            yaw_delta.clamp_(
                -self.max_yaw_rate * self.dt,
                self.max_yaw_rate * self.dt)
            self.reference_yaw += yaw_delta
            motor_command = self.lee(
                state[:, None, :],
                target_vel=world_command[:, None, :],
                target_yaw=self.reference_yaw[:, None, None])
            self.reference_position += reference_velocity * self.dt
            self.previous_reference_velocity.copy_(reference_velocity)
        else:
            reference_velocity = self._ego_to_world(
                action_command, quaternion)
            world_command = reference_velocity
            motor_command = self.lee(
                state[:, None, :], target_vel=world_command[:, None, :],
                target_yaw=None)
        torch.nan_to_num_(motor_command, 0.0)
        tensordict.set(("agents", "action"), motor_command)

        self.past = torch.roll(self.past, -1, dims=1)
        self.past_world = torch.roll(self.past_world, -1, dims=1)
        self.past_mask = torch.roll(self.past_mask, -1, dims=1)
        if self.past_input_semantics == "actual_vehicle_velocity":
            # New v2 demonstrations condition on measured ego velocity, not
            # teacher-forced commands.  Keep the same frame online.
            self.past[:, -1] = self._world_to_ego(velocity, quaternion)
        else:
            self.past[:, -1] = action_command
        if self.past_input_semantics in {
                "actual_velocity", "actual_vehicle_velocity",
                "measured_ego_velocity",
                "measured_velocity_in_current_observation_ego_frame"}:
            # This measured state exists with the same semantics both in the
            # HDF5 expert data and online, removing command-history teacher
            # forcing without changing the action target or Lee controller.
            self.past_world[:, -1] = velocity
            self.past_mask[:, -1] = self.active.to(torch.float32)
        else:
            self.past_world[:, -1] = reference_velocity
            self.past_mask[:, -1] = (self.active & ready).to(torch.float32)
        self.last_ego_command.copy_(action_command)
        self.last_world_command.copy_(world_command)
        self.last_reference_velocity_world.copy_(reference_velocity)
        self.command_abs_max = max(
            self.command_abs_max, float(action_command.abs().max()))
        self.step_index += 1
        return tensordict


@hydra.main(config_path=CONFIG_DIR, config_name="train", version_base=None)
def main(cfg):
    if float(cfg.sim.dt) != 0.016:
        raise ValueError("the original 0.016 s physics timestep must not change")
    if int(cfg.env_dyn.num_obstacles) != 0:
        raise ValueError("closed-loop WAM evaluation supports static obstacles only")
    if int(cfg.sensor.lidar_h_sample) != 3 or int(cfg.sensor.lidar_v_sample) != 3:
        raise ValueError("checkpoint requires raw 108x18 LiDAR")

    project_dir = Path(__file__).resolve().parents[2]
    scene_path = _resolve(project_dir, str(cfg.env.scene_path))
    route_scene_path = scene_path
    if scene_path.stem.endswith("_map"):
        route_scene_path = (
            scene_path.parent.parent / "environments"
            / f"{scene_path.stem[:-4]}.pt")
    checkpoint = _resolve(project_dir, str(cfg.wam_eval.checkpoint))
    world_checkpoint = _resolve(project_dir, str(cfg.wam_eval.world_checkpoint))
    metrics_path = _resolve(project_dir, str(cfg.wam_eval.metrics_path))
    trajectory_path = _resolve(project_dir, str(cfg.wam_eval.trajectory_path))
    import torch
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_format = torch.load(
        checkpoint, map_location="cpu", weights_only=False).get("format")
    required_files = [scene_path, route_scene_path]
    if checkpoint_format not in {
            ACTION_EXPERT_FORMAT, LEE_ACTION_EXPERT_FORMAT,
            FINAL_LEE_ACTION_EXPERT_FORMAT,
            HISTORY3_FINAL_LEE_ACTION_EXPERT_FORMAT,
            HISTORY3_CURRENT_LEE_ACTION_EXPERT_FORMAT,
            HISTORY1_CURRENT_LEE_ACTION_EXPERT_FORMAT}:
        required_files.append(world_checkpoint)
    for required in required_files:
        if not required.is_file():
            raise FileNotFoundError(required)
    cfg.env.scene_path = str(route_scene_path)

    device_index = int(str(cfg.device).split(":")[-1])
    active_gpu = int(os.environ.get("ISAAC_ACTIVE_GPU", device_index))
    sim_app = SimulationApp({
        "headless": bool(cfg.headless),
        "anti_aliasing": 1,
        "active_gpu": active_gpu,
        "physics_gpu": device_index,
        "multi_gpu": False,
    })
    if os.environ.get("NAVRL_ASSETS_ROOT"):
        import omni.isaac.core.utils.nucleus as nucleus
        nucleus.get_assets_root_path = lambda: os.environ["NAVRL_ASSETS_ROOT"]

    env = policy = lee_controller = None
    run_error = None
    try:
        import numpy as np
        import torch
        from env import NavigationEnv
        from omni_drones.controllers import LeePositionController

        requested_map = torch.load(
            scene_path, map_location="cpu", weights_only=False)
        route_scene = torch.load(
            route_scene_path, map_location="cpu", weights_only=False)
        if requested_map.get("map_sha256") != route_scene.get("map_sha256"):
            raise ValueError("map and route scene hashes differ")
        for key in ("obstacle_center_xyz", "obstacle_size_xyz"):
            if not torch.equal(requested_map[key], route_scene[key]):
                raise ValueError(f"map and route scene differ in {key}")

        env = NavigationEnv(cfg)
        env.eval()
        env.set_seed(int(cfg.seed))
        lee_controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        policy = WamLeePolicy(cfg, env, lee_controller, checkpoint, world_checkpoint)

        num_envs = int(cfg.env.num_envs)
        max_steps = int(cfg.env.max_episode_length)
        active = torch.ones(num_envs, dtype=torch.bool, device=cfg.device)
        outcomes = np.full(num_envs, "", dtype=object)
        terminal_steps = np.full(num_envs, -1, dtype=np.int32)
        td = env.reset()
        state_history = [
            td[("info", "drone_state")][:, 0, :13].cpu().float().numpy()]
        command_history = []
        reference_velocity_history = []
        lee_velocity_history = []
        start_time = time.perf_counter()

        for action_step in range(max_steps):
            policy.set_active(active)
            policy(td)
            command_history.append(
                policy.last_ego_command.detach().cpu().float().numpy().copy())
            reference_velocity_history.append(
                policy.last_reference_velocity_world.detach()
                .cpu().float().numpy().copy())
            lee_velocity_history.append(
                policy.last_world_command.detach()
                .cpu().float().numpy().copy())
            next_td = env.step(td)["next"]
            physics_step = action_step + 1
            state_history.append(
                next_td[("info", "drone_state")][:, 0, :13]
                .detach().cpu().float().numpy())
            stats = next_td["stats"]
            success = stats["reach_goal"].squeeze(-1).bool()
            collision = stats["collision"].squeeze(-1).bool()
            out_of_bounds = stats["out_of_bounds"].squeeze(-1).bool()
            next_position = next_td[("info", "drone_state")][:, 0, :3]
            next_goal_distance = (
                env.target_pos[:, 0] - next_position).norm(dim=-1)
            goal_brake_failed = (
                policy.goal_braking
                & (next_goal_distance > policy.goal_stop_distance))
            finished = active & (
                success | collision | out_of_bounds | goal_brake_failed)
            for env_id in finished.nonzero().squeeze(-1).tolist():
                outcomes[env_id] = (
                    "collision" if bool(collision[env_id]) else
                    "out_of_bounds" if bool(out_of_bounds[env_id]) else
                    "goal_brake_failed" if bool(goal_brake_failed[env_id]) else
                    "success")
                terminal_steps[env_id] = physics_step
            active &= ~finished
            td = next_td
            if physics_step % 100 == 0 or not bool(active.any()):
                print(
                    f"[WAM eval] step={physics_step} "
                    f"active={int(active.sum())}/{num_envs}", flush=True)
            if not bool(active.any()):
                break

        recorded_steps = len(command_history)
        for env_id in active.nonzero().squeeze(-1).tolist():
            outcomes[env_id] = "timeout"
            terminal_steps[env_id] = recorded_steps
        elapsed = time.perf_counter() - start_time
        counts = {
            name: int(np.count_nonzero(outcomes == name))
            for name in (
                "success", "timeout", "out_of_bounds", "collision",
                "goal_brake_failed")}
        route_ids = env.eval_route_id.detach().cpu().numpy()
        route_names = ["top_to_bottom", "bottom_to_top",
                       "left_to_right", "right_to_left"]
        by_route = {}
        for route_id, route_name in enumerate(route_names):
            indices = np.flatnonzero(route_ids == route_id)
            route_outcome = outcomes[indices]
            route_counts = {
                name: int(np.count_nonzero(route_outcome == name))
                for name in counts}
            by_route[route_name] = {
                "num_envs": int(len(indices)),
                "counts": route_counts,
                "sr": route_counts["success"] / max(1, len(indices)),
                "osr": 1.0 - route_counts["collision"] / max(1, len(indices)),
            }
        metrics = {
            "policy": ("deterministic_action_expert_to_original_lee"
                       if policy.is_deterministic_action_expert else
                       "dual_dit_velocity_to_original_lee"),
            "action_semantics": policy.action_semantics,
            "action_frame": policy.action_frame,
            "action_target_representation": policy.action_target_representation,
            "action_target_scale_xyz": (
                policy.action_target_scale.flatten().cpu().tolist()),
            "super_used": False,
            "checkpoint": str(checkpoint),
            "checkpoint_step": policy.checkpoint_step,
            "scene_path": str(scene_path),
            "route_scene_path": str(route_scene_path),
            "map_sha256": str(requested_map["map_sha256"]),
            "fixed_route_start_index": env.fixed_route_start_index,
            "num_envs": num_envs,
            "physics_dt_s": float(cfg.sim.dt),
            "replan_interval_steps": policy.interval,
            "sample_steps": (
                0 if policy.is_deterministic_action_expert else policy.sample_steps),
            "action_seed": policy.action_seed,
            "action_noise_scale": (
                0.0 if policy.is_deterministic_action_expert
                else policy.action_noise_scale),
            "training_action_diffusion": bool(
                checkpoint_format in {
                    ACTION_EXPERT_FORMAT, LEE_ACTION_EXPERT_FORMAT}),
            "deployment_action_diffusion": False,
            "bootstrap_forward_speed_mps": policy.bootstrap_forward_speed,
            "bootstrap_action_chunks": policy.bootstrap_action_chunks,
            "goal_stop_adapter_enabled": policy.goal_stop_enabled,
            "goal_stop_interventions": policy.goal_stop_interventions,
            "use_past_actions": policy.use_past_actions,
            "past_input_semantics": policy.past_input_semantics,
            "mask_world_position": policy.mask_world_position,
            "use_proprio": policy.use_proprio,
            "safety_filter_enabled": policy.safety_enabled,
            "safety_filter_interventions": policy.safety_interventions,
            "max_physics_steps": max_steps,
            "recorded_physics_steps": recorded_steps,
            "inference_calls": policy.inference_calls,
            "command_abs_max_mps": policy.command_abs_max,
            "wall_time_s": elapsed,
            "counts": counts,
            "rates": {
                "sr": counts["success"] / num_envs,
                "osr": 1.0 - counts["collision"] / num_envs,
                "timeout": counts["timeout"] / num_envs,
                "out_of_bounds": counts["out_of_bounds"] / num_envs,
                "collision": counts["collision"] / num_envs,
                "goal_brake_failed": (
                    counts["goal_brake_failed"] / num_envs),
            },
            "by_route": by_route,
            "terminal_steps": terminal_steps.tolist(),
        }
        # Optional standalone teacher audit supplies truthful policy metadata.
        # Ordinary student training/evaluations have no such attribute and
        # retain the exact existing schema, controller and termination logic.
        metrics.update(getattr(policy, "benchmark_metadata", {}))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_values = dict(
            root_state_wxyz=np.stack(state_history),
            ego_velocity_command=np.stack(command_history),
            reference_velocity_world=np.stack(reference_velocity_history),
            lee_target_velocity_world=np.stack(lee_velocity_history),
            start_position=env.eval_start_pos.detach().cpu().float().numpy(),
            target_position=env.target_pos[:, 0].detach().cpu().float().numpy(),
            route_id=route_ids,
            source_route_index=env.eval_source_route_index.detach().cpu().numpy(),
            outcome=outcomes,
            terminal_step=terminal_steps,
            physics_dt=np.asarray(float(cfg.sim.dt)),
            checkpoint_step=np.asarray(policy.checkpoint_step),
        )
        if policy.record_lidar:
            trajectory_values["lidar_step"] = np.asarray(
                policy.recorded_lidar_steps, dtype=np.int32)
            trajectory_values["lidar_image"] = np.stack(
                policy.recorded_lidar_images)
        np.savez_compressed(trajectory_path, **trajectory_values)
        print("[WAM eval] COMPLETE " + json.dumps(metrics["counts"]), flush=True)
        print("[WAM eval] rates=" + json.dumps(metrics["rates"]), flush=True)
        print(f"[WAM eval] metrics={metrics_path}", flush=True)
        print(f"[WAM eval] trajectory={trajectory_path}", flush=True)
    except BaseException as exc:
        run_error = exc
        print(f"[WAM eval] FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        policy = lee_controller = env = None
        gc.collect()
        # Isaac Sim 2023.1 intermittently segfaults while unloading plugins
        # after a successful headless run.  The process owns the simulation
        # context, so an OS-level exit below is both clean for GPU resources
        # and avoids turning a completed evaluation into a false failure.
        print("[WAM eval] skipping unstable Isaac 2023 plugin teardown", flush=True)
    if run_error is not None:
        raise run_error
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
