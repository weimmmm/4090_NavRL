"""Online Action Expert policy used by the standalone Isaac Sim evaluator."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAM_ROOT = PROJECT_ROOT / "lidar_WAM"
if not WAM_ROOT.is_dir():
    raise FileNotFoundError(f"Expected the sibling lidar_WAM directory at {WAM_ROOT}")
if str(WAM_ROOT) not in sys.path:
    sys.path.insert(0, str(WAM_ROOT))
DIFFUSERS_ROOT = WAM_ROOT / "third_party" / "diffusers" / "src"
if str(DIFFUSERS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSERS_ROOT))

from diffusers import AutoencoderKL, DDIMScheduler
from lidar_wam.models.action_expert import (
    ActionFlowExpert, JointWorldActionModel, LiDARObservationEncoder)
from lidar_wam.models.action_dit import (
    JointHistoryWorldActionDiT, flow_sample_action)
from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.models.world import DirectHorizonWorldModel
from lidar_wam.coordinates import (
    ACTION_FRAME,
    CONDITION_FRAME,
    LIDAR_FRAME,
    goal_frame_causal_features as canonical_goal_frame_causal_features,
    normalized_to_world_velocity,
    semantics as coordinate_semantics,
    yaw_error_radians,
)
from lidar_wam.runner import utils as wam_utils
from safetensors.torch import load_file

ACTION_HORIZON = 30
EXECUTION_HORIZON = 10
ACTION_DIM = 3
FLOW_EPS = 1e-4
COMPACT_FORMAT = "navrl-action-expert-policy-v2"
JOINT_POLICY_FORMAT = "navrl-joint-world-action-policy-v3"
JOINT_TRAINING_FORMAT = "navrl-joint-world-action-training-v3"
HISTORY_JOINT_FORMAT = "navrl-history-world-action-dit-v1"
HISTORY_JOINT_MASKED_FORMAT = "navrl-history-world-action-dit-v2-masked-terminal"
HISTORY_JOINT_GOAL_SPATIAL_FORMAT = "navrl-history-world-action-dit-v2-goal-spatial-attention"


def load_circular_vae(device: torch.device):
    """Load the exact trained Circular VAE without importing offline runners."""
    directory = WAM_ROOT / "lidar_wam" / "vae" / "circular"
    config = json.loads((directory / "config.json").read_text())
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    config["sample_size"] = [108, 20]
    model = AutoencoderKL(**config)
    wam_utils.replace_down(model)
    wam_utils.replace_conv(model)
    wam_utils.replace_attn(model)
    weights = load_file(str(directory / "diffusion_pytorch_model.safetensors"))
    model.load_state_dict(weights, strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """Load on both older Isaac torch releases and current PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class DeploymentActionModel(nn.Module):
    """The inference-only portion of the joint world/action checkpoint."""

    def __init__(self, width: int = 512, depth: int = 8, heads: int = 8,
                 ffn_width: int = 2048):
        super().__init__()
        self.observation = LiDARObservationEncoder(width)
        self.action_expert = ActionFlowExpert(
            width=width, depth=depth, heads=heads, ffn_width=ffn_width,
            future_attention_layers=0)

    def forward(self, noisy_action: torch.Tensor, timestep: torch.Tensor,
                current_latent: torch.Tensor, goal: torch.Tensor,
                proprio: torch.Tensor, past: torch.Tensor,
                past_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.observation(current_latent)
        return self.action_expert(
            noisy_action, timestep, tokens, goal, proprio, past, past_mask)


def _policy_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("observation.", "action_expert.")
    selected = {key: value for key, value in state.items()
                if key.startswith(prefixes)}
    if not selected:
        raise ValueError("Checkpoint contains no Action Expert policy parameters")
    return selected


def load_deployment_policy(checkpoint_path: Path, device: torch.device,
                           allow_legacy_body: bool = False):
    payload = torch_load(checkpoint_path, "cpu")
    # History World/Action DiT training stores causal condition statistics
    # under ``condition_stats`` and keeps the complete World+Action model in
    # the checkpoint.  Its action horizon is ten (the executed chunk), and
    # deployment must run the World-DiT history tokenizer before Action-DiT.
    if (isinstance(payload, dict)
            and payload.get("format") in (
                HISTORY_JOINT_FORMAT, HISTORY_JOINT_MASKED_FORMAT,
                HISTORY_JOINT_GOAL_SPATIAL_FORMAT)):
        architecture = dict(payload.get("architecture", {}))
        width = int(architecture.get("width", 512))
        depth = int(architecture.get("depth", 8))
        heads = int(architecture.get("heads", 8))
        mlp_ratio = float(architecture.get("mlp_ratio", 4.0))
        world = LiDARVideoDiT(width=width, depth=depth, heads=heads,
                               mlp_ratio=mlp_ratio)
        model = JointHistoryWorldActionDiT(
            world, width=width, depth=depth, heads=heads,
            mlp_ratio=mlp_ratio,
            past_horizon=int(architecture.get("past_horizon", 30)),
            shared_world_depth=int(architecture.get("shared_world_depth", 6)))
        model.load_state_dict(payload["model"], strict=True)
        model.to(device=device, dtype=torch.float32).eval()
        semantics = {
            "condition_frame": CONDITION_FRAME,
            "action_frame": ACTION_FRAME,
            "lidar_frame": LIDAR_FRAME,
            "action_limit_mps": 2.0,
            "action_horizon": int(architecture.get("action_horizon", 10)),
            "execution_horizon": 10,
            "history_frames": int(architecture.get("history_frames", 3)),
            "legacy": False,
        }
        return (model, payload.get("condition_stats", {}),
                int(payload.get("step", -1)), semantics)
    if not isinstance(payload, dict) or "model" not in payload or "stats" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    semantics = payload.get("semantics")
    if semantics is None:
        if not allow_legacy_body:
            raise ValueError(
                "Legacy checkpoint has no coordinate semantics. Pass "
                "--allow-legacy-body only for historical comparison; do not use it "
                "as a v2 policy.")
        semantics = {"condition_frame": "body", "action_frame": ACTION_FRAME,
                     "lidar_frame": LIDAR_FRAME, "legacy": True}
    else:
        expected = coordinate_semantics()
        for key in ("condition_frame", "action_frame", "lidar_frame"):
            if semantics.get(key) != expected[key]:
                raise ValueError(
                    f"checkpoint {key}={semantics.get(key)!r}, expected {expected[key]!r}")
    architecture = payload.get("architecture", {})
    width = int(architecture.get("width", 512))
    depth = int(architecture.get("depth", 8))
    heads = int(architecture.get("heads", 8))
    ffn_width = int(architecture.get("ffn_width", 2048))
    state = payload["model"]
    if payload.get("format") in (JOINT_POLICY_FORMAT, JOINT_TRAINING_FORMAT):
        model = JointWorldActionModel(
            DirectHorizonWorldModel(), width, depth, heads, ffn_width,
            int(architecture.get("future_attention_layers", 2)))
        model.load_state_dict(state, strict=True)
    else:
        model = DeploymentActionModel(width, depth, heads, ffn_width)
        if payload.get("format") != COMPACT_FORMAT:
            state = _policy_state_dict(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "checkpoint/model mismatch: "
                f"missing={missing}, unexpected={unexpected}")
    model.to(device=device, dtype=torch.float32).eval()
    return model, payload["stats"], int(payload.get("step", -1)), semantics


def export_compact_checkpoint(source: Path, destination: Path,
                              allow_legacy_body: bool = False) -> dict[str, Any]:
    payload = torch_load(source, "cpu")
    if not isinstance(payload, dict) or "model" not in payload or "stats" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {source}")
    if payload.get("format") in (JOINT_POLICY_FORMAT, JOINT_TRAINING_FORMAT):
        raise ValueError(
            "A v3 joint policy cannot be stripped to Action-only: its deployed "
            "action depends on the Future UNet and FutureTokenAdapter.")
    if "semantics" not in payload and not allow_legacy_body:
        raise ValueError("refusing to export legacy checkpoint without coordinate semantics")
    selected = _policy_state_dict(payload["model"])
    result = {
        "format": COMPACT_FORMAT,
        "model": selected,
        "stats": payload["stats"],
        "step": int(payload.get("step", -1)),
        "world_initial_step": int(payload.get("world_initial_step", -1)),
        "source_checkpoint": str(source),
        "architecture": payload.get("architecture", {
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            "width": 512,
            "depth": 8,
            "heads": 8,
            "ffn_width": 2048,
        }),
        "semantics": payload.get("semantics", {
            "condition_frame": "body", "action_frame": ACTION_FRAME,
            "lidar_frame": LIDAR_FRAME, "legacy": True}),
        "provenance": payload.get("provenance", {}),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "step": result["step"],
        "parameter_tensors": len(selected),
        "size_bytes": destination.stat().st_size,
    }


def _stat_tensor(stats: dict[str, Any], key: str, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(stats[key], device=like.device, dtype=like.dtype)


def normalize_condition(value: torch.Tensor, stats: dict[str, Any], prefix: str):
    return ((value - _stat_tensor(stats, f"{prefix}_mean", value)) /
            _stat_tensor(stats, f"{prefix}_std", value))


def action_to_flow(action: torch.Tensor, stats: dict[str, Any]):
    clipped = action.clamp(FLOW_EPS, 1.0 - FLOW_EPS)
    value = torch.log(clipped) - torch.log1p(-clipped)
    return ((value - _stat_tensor(stats, "action_logit_mean", value)) /
            _stat_tensor(stats, "action_logit_std", value))


def flow_to_action(value: torch.Tensor, stats: dict[str, Any]):
    value = (value * _stat_tensor(stats, "action_logit_std", value)
             + _stat_tensor(stats, "action_logit_mean", value))
    return value.sigmoid()


def quaternion_rotation_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(*quaternion.shape[:-1], 3, 3)


def causal_features(drone_state: torch.Tensor, target_world: torch.Tensor):
    """Construct exactly the body-frame conditions used during training."""
    state = drone_state.reshape(-1, 13)
    target = target_world.reshape(-1, 3)
    world_to_body = quaternion_rotation_wxyz(state[:, 3:7]).transpose(-1, -2)
    goal_body = torch.bmm(world_to_body, (target - state[:, :3]).unsqueeze(-1)).squeeze(-1)
    goal = torch.cat((goal_body, goal_body.norm(dim=-1, keepdim=True)), dim=-1)
    velocity = torch.bmm(world_to_body, state[:, 7:10].unsqueeze(-1)).squeeze(-1)
    angular = torch.bmm(world_to_body, state[:, 10:13].unsqueeze(-1)).squeeze(-1)
    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=state.device,
                                 dtype=state.dtype).expand(len(state), -1)
    gravity = torch.bmm(world_to_body, gravity_world.unsqueeze(-1)).squeeze(-1)
    proprio = torch.cat((state[:, 2:3], velocity, angular, gravity), dim=-1)
    return goal, proprio


def goal_frame_causal_features(drone_state: torch.Tensor,
                               target_world: torch.Tensor,
                               goal_direction: torch.Tensor):
    """Express all vector conditions in the PPO action coordinate frame.

    ``goal_direction`` is the fixed start-to-target direction stored by the
    environment at reset.  This is the same frame used by ``vec_to_world``
    when PPO normalized actions are converted into world velocity commands.
    """
    return canonical_goal_frame_causal_features(
        drone_state, target_world, goal_direction)


def lidar_range_image(env) -> torch.Tensor:
    """Read the live RayCaster and produce the VAE's [B,2,108,20] tensor."""
    horizontal, vertical = env.lidar_raw_resolution
    hits = env.lidar.data.ray_hits_w.reshape(env.num_envs, horizontal, vertical, 3)
    sensor = env.lidar.data.pos_w[:, None, None, :]
    distances = (hits - sensor).norm(dim=-1)
    valid = torch.isfinite(hits).all(dim=-1) & (distances <= float(env.lidar_range))
    distances = torch.where(valid, distances, torch.full_like(distances, env.lidar_range))
    image = torch.empty((env.num_envs, 2, horizontal, 20), device=distances.device,
                        dtype=torch.float32)
    image[:, 0].fill_(1.0)
    image[:, 1].fill_(-1.0)
    image[:, 0, :, :vertical] = distances / (float(env.lidar_range) / 2.0) - 1.0
    image[:, 1, :, :vertical] = valid.to(image.dtype) * 2.0 - 1.0
    return image


def route_stable_action_noise(seeds: torch.Tensor, device: torch.device,
                              dtype: torch.dtype, horizon: int = ACTION_HORIZON
                              ) -> torch.Tensor:
    """Generate one deterministic flow-noise tensor per route.

    A single CUDA generator for the whole batch is not route-stable: changing
    the number of parallel Isaac environments can change the CUDA normal RNG
    kernel and therefore the noise assigned to an otherwise identical route.
    Generate each small 30x3 tensor from its own CPU generator so a route gets
    exactly the same policy noise in 1-, 128-, and 512-environment runs.
    """
    rows = []
    for seed in seeds.detach().cpu().reshape(-1).tolist():
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        rows.append(torch.randn(
            (int(horizon), ACTION_DIM), generator=generator,
            device="cpu", dtype=torch.float32))
    return torch.stack(rows).to(device=device, dtype=dtype, non_blocking=True)


@torch.no_grad()
def sample_actions(model: nn.Module, latent: torch.Tensor,
                   goal: torch.Tensor, proprio: torch.Tensor,
                   past: torch.Tensor, past_mask: torch.Tensor,
                   stats: dict[str, Any], steps: int,
                   seeds: torch.Tensor,
                   history: torch.Tensor | None = None) -> torch.Tensor:
    if seeds.numel() != len(latent):
        raise ValueError(
            f"Expected one policy seed per route, got {seeds.numel()} for "
            f"batch {len(latent)}")
    if isinstance(model, JointHistoryWorldActionDiT):
        if history is None:
            raise ValueError("history latents are required by History World/Action DiT")
        goal = normalize_condition(goal, stats, "goal")
        proprio = normalize_condition(proprio, stats, "proprio")
        features = model.encode_history_features(history)
        value = route_stable_action_noise(
            seeds, latent.device, latent.dtype, model.action.action_horizon)
        schedule = torch.linspace(1, 0, int(steps) + 1,
                                  device=latent.device, dtype=latent.dtype)
        for current, following in zip(schedule[:-1], schedule[1:]):
            timestep = torch.full((len(latent),), current,
                                  device=latent.device, dtype=latent.dtype)
            velocity = model.action(
                features, value, timestep, goal, proprio, past, past_mask)
            value = value + (following - current) * velocity
        return value.clamp(0.0, 1.0)

    value = route_stable_action_noise(seeds, latent.device, latent.dtype)
    raw_proprio = proprio
    goal = normalize_condition(goal, stats, "goal")
    proprio = normalize_condition(proprio, stats, "proprio")
    past = action_to_flow(past, stats) * past_mask.unsqueeze(-1)
    observation = (model.encode_current(latent)
                   if isinstance(model, JointWorldActionModel)
                   else model.observation(latent))
    sigmas = torch.linspace(1, 0, steps + 1, device=latent.device, dtype=latent.dtype)
    joint = isinstance(model, JointWorldActionModel)
    if joint:
        scheduler = DDIMScheduler(
            num_train_timesteps=1000, prediction_type="epsilon",
            clip_sample=False)
        scheduler.set_timesteps(steps, device=latent.device)
        future_rows = []
        for seed in seeds.detach().cpu().reshape(-1).tolist():
            generator = torch.Generator(device="cpu").manual_seed(
                (int(seed) + 97_409) % (2**63 - 1))
            future_rows.append(torch.randn(
                (4, 27, 5), generator=generator, dtype=torch.float32))
        future = torch.stack(future_rows).to(
            device=latent.device, dtype=latent.dtype, non_blocking=True)
        zero = torch.zeros(len(latent), device=latent.device,
                           dtype=latent.dtype)
        world_state = torch.stack((
            raw_proprio[:, 1], raw_proprio[:, 2], zero, zero,
            raw_proprio[:, 6]), dim=-1)
        action_mean = _stat_tensor(stats, "action_logit_mean", value)
        action_std = _stat_tensor(stats, "action_logit_std", value)
        world_timesteps = scheduler.timesteps
    else:
        world_timesteps = [None] * steps
    for current, following, world_timestep in zip(
            sigmas[:-1], sigmas[1:], world_timesteps):
        timestep = torch.full((len(latent),), float(current * 1000),
                              device=latent.device, dtype=latent.dtype)
        if joint:
            velocity, epsilon, _ = model.predict_joint_velocity(
                value, timestep, future, world_timestep, latent, goal,
                proprio, past, past_mask, world_state, action_mean,
                action_std, observation_tokens=observation)
            future = scheduler.step(
                epsilon, world_timestep, future, eta=0.0).prev_sample
        else:
            velocity = model.action_expert(
                value, timestep, observation, goal, proprio, past, past_mask)
        value = value + (following - current) * velocity
    return flow_to_action(value, stats)


class RecedingHorizonPolicy:
    """Generate and execute one fixed ten-step action chunk."""

    def __init__(self, checkpoint: Path, num_envs: int, device: torch.device,
                 flow_steps: int = 10, seed: int = 42, action_limit: float = 2.0,
                 allow_legacy_body: bool = False,
                 route_ids: torch.Tensor | None = None):
        self.device = device
        self.flow_steps = int(flow_steps)
        self.seed = int(seed)
        self.action_limit = float(action_limit)
        (self.model, self.stats, self.checkpoint_step,
         self.semantics) = load_deployment_policy(
            checkpoint, device, allow_legacy_body=allow_legacy_body)
        self.history_joint = isinstance(self.model, JointHistoryWorldActionDiT)
        self.execution_horizon = EXECUTION_HORIZON
        self.past_horizon = (
            int(self.model.action.past_horizon)
            if self.history_joint else EXECUTION_HORIZON)
        self.condition_frame = self.semantics["condition_frame"]
        self.executed_head = (
            "history_world_action_dit_flow10"
            if self.history_joint else
            ("joint_world_action_flow30"
             if isinstance(self.model, JointWorldActionModel) else "flow30"))
        expected_limit = float(self.semantics.get("action_limit_mps", action_limit))
        if abs(expected_limit-self.action_limit) > 1e-6:
            raise ValueError(
                f"checkpoint action_limit_mps={expected_limit} but environment uses "
                f"{self.action_limit}")
        expected_horizon = 10 if self.history_joint else ACTION_HORIZON
        if int(self.semantics.get("action_horizon", expected_horizon)) != expected_horizon:
            raise ValueError("checkpoint action horizon is incompatible with deployment")
        if int(self.semantics.get(
                "execution_horizon", EXECUTION_HORIZON)) != EXECUTION_HORIZON:
            raise ValueError("checkpoint execution horizon is incompatible with deployment")
        self.vae = load_circular_vae(device)
        self.scale = float(self.vae.config.scaling_factor)
        self.actions = torch.zeros(num_envs, self.execution_horizon, ACTION_DIM,
                                    device=device)
        self.past = torch.zeros(num_envs, self.past_horizon, ACTION_DIM,
                                device=device)
        self.past_mask = torch.zeros(num_envs, self.past_horizon, device=device)
        self.history_latents = torch.zeros(
            num_envs, 3, 4, 27, 5, device=device, dtype=torch.float32)
        self.history_valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.action_index = torch.full((num_envs,), self.execution_horizon,
                                       device=device, dtype=torch.long)
        self.route_id = (torch.arange(num_envs, device=device, dtype=torch.long)
                         if route_ids is None else
                         torch.as_tensor(route_ids, device=device, dtype=torch.long))
        if tuple(self.route_id.shape) != (num_envs,):
            raise ValueError(f"route_ids must have shape [{num_envs}]")
        self.route_replan_count = torch.zeros(
            num_envs, device=device, dtype=torch.long)
        self.plan_counter = 0
        self.inference_seconds: list[float] = []
        self.yaw_error_abs: list[torch.Tensor] = []

    def reset(self, mask: torch.Tensor | None = None):
        if mask is None:
            mask = torch.ones(len(self.action_index), device=self.device, dtype=torch.bool)
        self.actions[mask] = 0
        self.past[mask] = 0
        self.past_mask[mask] = 0
        self.action_index[mask] = self.execution_horizon
        self.history_latents[mask] = 0
        self.history_valid[mask] = False
        self.route_replan_count[mask] = 0
    @torch.no_grad()
    def _replan(self, env, mask: torch.Tensor):
        indices = mask.nonzero().flatten()
        if not len(indices):
            return
        continuing = self.past_mask[indices].any(dim=-1) | (
            self.actions[indices].abs().sum((1, 2)) > 0)
        if continuing.any():
            continued_indices = indices[continuing]
            if self.execution_horizon < self.past_horizon:
                self.past[continued_indices] = torch.cat((
                    self.past[continued_indices, self.execution_horizon:],
                    self.actions[continued_indices]), dim=1)
                self.past_mask[continued_indices] = torch.cat((
                    self.past_mask[continued_indices, self.execution_horizon:],
                    torch.ones(len(continued_indices), self.execution_horizon,
                               device=self.device)), dim=1)
            else:
                self.past[continued_indices] = self.actions[
                    continued_indices, -self.past_horizon:]
                self.past_mask[continued_indices] = 1.0

        started = time.perf_counter()
        image = lidar_range_image(env)[indices]
        latent = self.vae.encode(image).latent_dist.mode() * self.scale
        if self.history_joint:
            valid = self.history_valid[indices]
            if valid.any():
                valid_indices = indices[valid]
                self.history_latents[valid_indices] = torch.cat((
                    self.history_latents[valid_indices, 1:],
                    latent[valid, None]), dim=1)
            if (~valid).any():
                invalid_indices = indices[~valid]
                self.history_latents[invalid_indices] = latent[~valid, None].expand(
                    -1, 3, -1, -1, -1)
            history = self.history_latents[indices]
            self.history_valid[indices] = True
        else:
            history = None
        state = env.drone.get_state(env_frame=False)[indices, 0, :13]
        target = env.target_pos[indices, 0]
        if self.condition_frame == CONDITION_FRAME:
            direction = env.target_dir[indices, 0]
            goal, proprio = goal_frame_causal_features(state, target, direction)
            self.yaw_error_abs.append(
                yaw_error_radians(state, direction).abs().detach().cpu())
        else:
            goal, proprio = causal_features(state, target)
        # Route-local seeds make fixed-environment evaluation invariant to the
        # number of routes evaluated in parallel.  The constants are distinct
        # large odd numbers; the modulo keeps seeds in torch's signed range.
        base_policy_seeds = (
            int(self.seed)
            + self.route_id[indices] * 1_000_003
            + self.route_replan_count[indices] * 10_007
        ).remainder(2**63 - 1)
        chosen = sample_actions(
            self.model, latent, goal, proprio, self.past[indices],
            self.past_mask[indices], self.stats, self.flow_steps,
            base_policy_seeds, history=history)
        self.actions[indices] = chosen[:, :self.execution_horizon]
        self.action_index[indices] = 0
        self.route_replan_count[indices] += 1
        self.plan_counter += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.inference_seconds.append(time.perf_counter() - started)

    @torch.no_grad()
    def act(self, tensordict, env):
        self._replan(env, self.action_index >= self.execution_horizon)
        rows = torch.arange(len(self.action_index), device=self.device)
        normalized = self.actions[rows, self.action_index]
        self.action_index += 1
        direction = tensordict["agents", "observation", "direction"].reshape(-1, 3)
        world_velocity = normalized_to_world_velocity(
            normalized, direction, self.action_limit)
        tensordict.set(("agents", "action_normalized"), normalized)
        tensordict.set(("agents", "action"), world_velocity)
        return tensordict

    def latency_summary(self) -> dict[str, Any]:
        if not self.inference_seconds:
            return {"plans": 0, "mean_s": None, "p95_s": None, "max_s": None}
        values = torch.tensor(self.inference_seconds)
        result = {
            "plans": len(self.inference_seconds),
            "mean_s": float(values.mean()),
            "p95_s": float(torch.quantile(values, 0.95)),
            "max_s": float(values.max()),
            "executed_head": self.executed_head,
        }
        if self.yaw_error_abs:
            yaw = torch.cat(self.yaw_error_abs)
            result.update({
                "sensor_goal_yaw_abs_mean_deg": float(yaw.mean()*180.0/torch.pi),
                "sensor_goal_yaw_abs_p95_deg": float(torch.quantile(yaw, 0.95)*180.0/torch.pi),
                "sensor_goal_yaw_abs_max_deg": float(yaw.max()*180.0/torch.pi),
            })
        return result


def save_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
