"""Fast-WAM-inspired action flow expert for the LaGen LiDAR UNet.

Fast-WAM uses a Mixture-of-Transformers because its world expert is a Video
DiT.  The NavRL world expert is a 2-D conditional UNet, so its transformer
blocks cannot be mixed layer-for-layer with ActionDiT.  This module preserves
the causal part of that design: an action flow expert reads clean current
observation tokens, while a shared observation encoder is co-trained by the
world and action objectives.  Future observation tokens are never exposed to
the action branch.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(timestep: torch.Tensor, width: int) -> torch.Tensor:
    """Standard diffusion timestep embedding."""
    half = width // 2
    exponent = -math.log(10000.0) * torch.arange(
        half, device=timestep.device, dtype=torch.float32
    ) / max(half - 1, 1)
    phase = timestep.float().unsqueeze(1) * exponent.exp().unsqueeze(0)
    value = torch.cat([phase.sin(), phase.cos()], dim=1)
    if width % 2:
        value = F.pad(value, (0, 1))
    return value.to(dtype=timestep.dtype)


class CircularConv2d(nn.Conv2d):
    """Circular padding on azimuth (H), zero padding on elevation (W)."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pad_h, pad_w = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        if pad_h:
            value = torch.cat([value[:, :, -pad_h:], value, value[:, :, :pad_h]], dim=2)
        if pad_w:
            value = F.pad(value, (pad_w, pad_w, 0, 0))
        return F.conv2d(value, self.weight, self.bias, self.stride,
                        padding=0, dilation=self.dilation, groups=self.groups)


def polar_position_encoding(height: int, width: int, channels: int) -> torch.Tensor:
    """Deterministic periodic azimuth/elevation features for LiDAR tokens.

    Cross-attention is otherwise permutation invariant and cannot distinguish
    an obstacle on the left from one on the right after the feature map is
    flattened.  The azimuth terms are periodic so the 360-degree seam remains
    continuous.
    """
    azimuth = torch.arange(height, dtype=torch.float32) * (2 * math.pi / height)
    elevation = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    azimuth, elevation = torch.meshgrid(azimuth, elevation, indexing="ij")
    base = torch.stack([
        azimuth.sin(), azimuth.cos(),
        (2 * azimuth).sin(), (2 * azimuth).cos(),
        elevation, (math.pi * elevation).sin(),
        (math.pi * elevation).cos(),
    ], dim=-1).reshape(1, height * width, 7)
    repeats = math.ceil(channels / base.shape[-1])
    return base.repeat(1, 1, repeats)[..., :channels]


class LiDARObservationEncoder(nn.Module):
    """Encode the clean current ``[4,27,5]`` latent into 135 causal tokens."""

    def __init__(self, width: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            CircularConv2d(4, 128, 3), nn.SiLU(),
            CircularConv2d(128, 256, 3), nn.SiLU(),
            CircularConv2d(256, width, 3),
        )
        self.norm = nn.LayerNorm(width)
        self.register_buffer(
            "polar_position", polar_position_encoding(27, 5, width),
            persistent=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or tuple(latent.shape[1:]) != (4, 27, 5):
            raise ValueError(f"Expected current latent [B,4,27,5], got {tuple(latent.shape)}")
        feature = self.net(latent)
        tokens = self.norm(feature.flatten(2).transpose(1, 2))
        return tokens + 0.10 * self.polar_position.to(dtype=tokens.dtype)


class ActionFlowBlock(nn.Module):
    """Action self-attention followed by causal current-observation attention."""

    def __init__(self, width: int, heads: int, ffn_width: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width), nn.GELU(approximate="tanh"),
            nn.Linear(ffn_width, width),
        )
        self.modulation = nn.Linear(width, 6 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, action: torch.Tensor, context: torch.Tensor,
                time_embedding: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.modulation(time_embedding).chunk(6, dim=-1)
        )
        value = self.norm1(action)
        value = value * (1 + scale1[:, None]) + shift1[:, None]
        value = self.self_attn(value, value, value, need_weights=False)[0]
        action = action + torch.tanh(gate1[:, None]) * value
        action = action + self.cross_attn(
            self.norm2(action), context, context, need_weights=False
        )[0]
        value = self.norm3(action)
        value = value * (1 + scale2[:, None]) + shift2[:, None]
        action = action + torch.tanh(gate2[:, None]) * self.ffn(value)
        return action


class ActionFlowExpert(nn.Module):
    """Generate a 30-step normalized PPO action chunk by rectified flow."""

    def __init__(self, action_dim: int = 3, horizon: int = 30,
                 width: int = 512, depth: int = 8, heads: int = 8,
                 ffn_width: int = 2048, goal_dim: int = 4,
                 proprio_dim: int = 10, past_horizon: int = 10):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.width = int(width)
        self.past_horizon = int(past_horizon)
        self.action_in = nn.Linear(action_dim, width)
        self.action_position = nn.Parameter(torch.randn(1, horizon, width) * 0.02)
        self.past_in = nn.Linear(action_dim + 1, width)
        self.past_position = nn.Parameter(torch.randn(1, past_horizon, width) * 0.02)
        self.goal_in = nn.Sequential(nn.Linear(goal_dim, width), nn.SiLU(),
                                     nn.Linear(width, width))
        self.proprio_in = nn.Sequential(nn.Linear(proprio_dim, width), nn.SiLU(),
                                        nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            ActionFlowBlock(width, heads, ffn_width) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, noisy_action: torch.Tensor, timestep: torch.Tensor,
                observation_tokens: torch.Tensor, goal: torch.Tensor,
                proprio: torch.Tensor, past_actions: torch.Tensor,
                past_mask: torch.Tensor) -> torch.Tensor:
        if tuple(noisy_action.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected noisy actions [B,{self.horizon},{self.action_dim}], "
                f"got {tuple(noisy_action.shape)}")
        if tuple(past_actions.shape[1:]) != (self.past_horizon, self.action_dim):
            raise ValueError(f"Unexpected past action shape {tuple(past_actions.shape)}")
        past = torch.cat([past_actions, past_mask.unsqueeze(-1)], dim=-1)
        context = torch.cat([
            observation_tokens,
            self.goal_in(goal).unsqueeze(1),
            self.proprio_in(proprio).unsqueeze(1),
            self.past_in(past) + self.past_position,
        ], dim=1)
        value = self.action_in(noisy_action) + self.action_position
        time = self.time(sinusoidal_embedding(timestep, self.width))
        value = value + time[:, None]
        for block in self.blocks:
            value = block(value, context, time)
        return self.out(self.norm(value))


class JointWorldActionModel(nn.Module):
    """Couple an existing LaGen UNet world model to the causal action expert."""

    def __init__(self, world_model: nn.Module, width: int = 512,
                 depth: int = 8, heads: int = 8, ffn_width: int = 2048):
        super().__init__()
        self.world = world_model
        self.observation = LiDARObservationEncoder(width)
        self.action_expert = ActionFlowExpert(
            width=width, depth=depth, heads=heads, ffn_width=ffn_width)
        # Zero initialization makes the first world forward exactly the loaded
        # checkpoint behavior. Joint training then learns how much shared
        # current-observation information the UNet conditioning should use.
        self.observation_to_world = nn.Linear(width, 768, bias=False)
        nn.init.zeros_(self.observation_to_world.weight)

    def encode_current(self, current_latent: torch.Tensor) -> torch.Tensor:
        return self.observation(current_latent)

    def predict_world(self, noisy_future: torch.Tensor,
                      current_latent: torch.Tensor, actions: torch.Tensor,
                      state: torch.Tensor, timestep: torch.Tensor,
                      observation_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if observation_tokens is None:
            observation_tokens = self.encode_current(current_latent)
        condition = self.world.condition(actions, state)
        residual = self.observation_to_world(observation_tokens.mean(dim=1))
        condition = condition.clone()
        condition[:, -1] = condition[:, -1] + residual
        return self.world.unet(
            torch.cat([noisy_future, current_latent], dim=1), timestep,
            encoder_hidden_states=condition,
        ).sample

    def predict_action_velocity(self, noisy_action: torch.Tensor,
                                action_timestep: torch.Tensor,
                                current_latent: torch.Tensor,
                                goal: torch.Tensor, proprio: torch.Tensor,
                                past_actions: torch.Tensor,
                                past_mask: torch.Tensor,
                                observation_tokens: torch.Tensor | None = None
                                ) -> torch.Tensor:
        if observation_tokens is None:
            observation_tokens = self.encode_current(current_latent)
        return self.action_expert(
            noisy_action, action_timestep, observation_tokens, goal, proprio,
            past_actions, past_mask)

    def forward(self, noisy_action: torch.Tensor, action_timestep: torch.Tensor,
                current_latent: torch.Tensor, goal: torch.Tensor,
                proprio: torch.Tensor, past_actions: torch.Tensor,
                past_mask: torch.Tensor, noisy_future: torch.Tensor | None = None,
                world_actions: torch.Tensor | None = None,
                world_state: torch.Tensor | None = None,
                world_timestep: torch.Tensor | None = None):
        """Single DDP-visible forward for action-only or joint optimization."""
        observation = self.encode_current(current_latent)
        # A non-persistent graph handle used only for the periodic diagnostic
        # that verifies world loss reaches the shared representation.  Taking
        # dL/dtokens avoids a second parameter backward through DDP.
        self._last_observation_tokens = observation
        action_velocity = self.predict_action_velocity(
            noisy_action, action_timestep, current_latent, goal, proprio,
            past_actions, past_mask, observation_tokens=observation)
        world_epsilon = None
        if noisy_future is not None:
            if world_actions is None or world_state is None or world_timestep is None:
                raise ValueError("joint forward requires all world-model conditions")
            world_epsilon = self.predict_world(
                noisy_future, current_latent, world_actions, world_state,
                world_timestep, observation_tokens=observation)
        return action_velocity, world_epsilon


class ActionOnlyModel(nn.Module):
    """Deployment-shaped Action Expert used before optional world coupling."""

    def __init__(self, width: int = 512, depth: int = 8, heads: int = 8,
                 ffn_width: int = 2048):
        super().__init__()
        self.observation = LiDARObservationEncoder(width)
        self.action_expert = ActionFlowExpert(
            width=width, depth=depth, heads=heads, ffn_width=ffn_width)

    def encode_current(self, current_latent: torch.Tensor) -> torch.Tensor:
        return self.observation(current_latent)

    def predict_action_velocity(self, noisy_action: torch.Tensor,
                                action_timestep: torch.Tensor,
                                current_latent: torch.Tensor,
                                goal: torch.Tensor, proprio: torch.Tensor,
                                past_actions: torch.Tensor,
                                past_mask: torch.Tensor,
                                observation_tokens: torch.Tensor | None = None
                                ) -> torch.Tensor:
        if observation_tokens is None:
            observation_tokens = self.encode_current(current_latent)
        return self.action_expert(
            noisy_action, action_timestep, observation_tokens, goal, proprio,
            past_actions, past_mask)

    def forward(self, noisy_action: torch.Tensor, action_timestep: torch.Tensor,
                current_latent: torch.Tensor, goal: torch.Tensor,
                proprio: torch.Tensor, past_actions: torch.Tensor,
                past_mask: torch.Tensor, **_unused):
        return (self.predict_action_velocity(
            noisy_action, action_timestep, current_latent, goal, proprio,
            past_actions, past_mask), None)

