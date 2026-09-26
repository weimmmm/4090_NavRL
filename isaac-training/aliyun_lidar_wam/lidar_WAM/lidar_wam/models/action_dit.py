"""Action DiT paired with the history-only LiDAR Video-DiT.

The action branch intentionally follows the same token/AdaLN/flow-matching
recipe as :mod:`lidar_video_dit`: noisy action tokens are appended after causal
history, goal/proprioception and past-action tokens.  Context tokens cannot
read the noisy action block, while action tokens can read the complete causal
context.  The history tokens are supplied by the first pretrained World-DiT
blocks (H6), so the action branch uses temporal World features rather than
only the raw input projection.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from lidar_wam.models.lidar_video_dit import VideoDiTBlock, timestep_embedding


class ActionDiT(nn.Module):
    """Predict a ten-step normalized velocity-action chunk with rectified flow."""

    def __init__(self, width: int = 512, depth: int = 8, heads: int = 8,
                 mlp_ratio: float = 4.0, action_horizon: int = 10,
                 action_dim: int = 3, history_tokens: int = 3 * 27 * 5,
                 proprio_dim: int = 10, past_horizon: int = 30):
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.history_tokens = int(history_tokens)
        self.past_horizon = int(past_horizon)
        self.action_in = nn.Linear(action_dim, width)
        self.goal_in = nn.Sequential(nn.Linear(4, width), nn.SiLU(),
                                     nn.Linear(width, width))
        self.proprio_in = nn.Sequential(nn.Linear(proprio_dim, width), nn.SiLU(),
                                        nn.Linear(width, width))
        self.past_in = nn.Linear(action_dim + 1, width)
        self.action_position = nn.Parameter(
            torch.randn(1, action_horizon, width) * 0.02)
        self.past_position = nn.Parameter(
            torch.randn(1, past_horizon, width) * 0.02)
        self.special_position = nn.Parameter(torch.randn(1, 2, width) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            VideoDiTBlock(width, heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(width, 2 * width))
        self.output_projection = nn.Linear(width, action_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer("causal_mask", self._make_attention_mask(),
                             persistent=False)

    @property
    def context_tokens(self) -> int:
        return self.history_tokens + 2 + self.past_horizon

    def _make_attention_mask(self) -> torch.Tensor:
        total = self.context_tokens + self.action_horizon
        mask = torch.zeros(total, total, dtype=torch.bool)
        # Causal context (history, goal, proprio and previous action) is not
        # allowed to read the noisy action being denoised.
        mask[:self.context_tokens, self.context_tokens:] = True
        return mask

    def forward(self, history_tokens: torch.Tensor, noisy_action: torch.Tensor,
                timestep: torch.Tensor, goal: torch.Tensor,
                proprio: torch.Tensor, past_actions: torch.Tensor,
                past_mask: torch.Tensor,
                action_mask: torch.Tensor | None = None) -> torch.Tensor:
        expected_history = (self.history_tokens, self.width)
        if history_tokens.ndim != 3 or tuple(history_tokens.shape[1:]) != expected_history:
            raise ValueError(
                f"Expected history tokens [B,{expected_history[0]},{expected_history[1]}], "
                f"got {tuple(history_tokens.shape)}")
        if tuple(noisy_action.shape[1:]) != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"Expected noisy actions [B,{self.action_horizon},{self.action_dim}], "
                f"got {tuple(noisy_action.shape)}")
        if tuple(past_actions.shape[1:]) != (self.past_horizon, self.action_dim):
            raise ValueError(f"Unexpected past action shape {tuple(past_actions.shape)}")
        past = torch.cat((past_actions, past_mask.unsqueeze(-1)), dim=-1)
        special = torch.cat((
            self.goal_in(goal).unsqueeze(1),
            self.proprio_in(proprio).unsqueeze(1)), dim=1)
        context = torch.cat((
            history_tokens,
            special + self.special_position,
            self.past_in(past) + self.past_position,
        ), dim=1)
        action = self.action_in(noisy_action) + self.action_position
        tokens = torch.cat((context, action), dim=1)
        key_padding_mask = None
        if action_mask is not None:
            if tuple(action_mask.shape[1:]) != (self.action_horizon,):
                raise ValueError(
                    f"Expected action mask [B,{self.action_horizon}], "
                    f"got {tuple(action_mask.shape)}")
            context_valid = torch.ones(
                len(tokens), self.context_tokens, device=tokens.device,
                dtype=torch.bool)
            key_padding_mask = ~torch.cat(
                (context_valid, action_mask.to(dtype=torch.bool)), dim=1)
        condition = self.time_mlp(timestep_embedding(timestep, self.width))
        for block in self.blocks:
            tokens = block(tokens, condition, self.causal_mask,
                           key_padding_mask=key_padding_mask)
        action = tokens[:, -self.action_horizon:]
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        action = self.final_norm(action) * (1 + scale[:, None]) + shift[:, None]
        return self.output_projection(action)


class JointHistoryWorldActionDiT(nn.Module):
    """World DiT + Action DiT with deep World-history feature sharing.

    ``encode_history`` is the World DiT tokenizer (H0).  The action branch
    consumes H6, obtained by running the observed history through the first
    ``shared_world_depth`` pretrained World blocks.  Because the block mask
    prevents history tokens from reading the noisy next-frame tokens, H6 is
    still strictly causal and can be computed without any future LiDAR.  The
    World prediction path remains the original full Video-DiT path, so its
    checkpoint is not silently changed at initialization.
    """

    format = "navrl-history-world-action-dit-v1"

    def __init__(self, world: nn.Module, width: int = 512, depth: int = 8,
                 heads: int = 8, mlp_ratio: float = 4.0,
                 past_horizon: int = 30, shared_world_depth: int = 6):
        super().__init__()
        self.world = world
        self.shared_world_depth = int(shared_world_depth)
        if not 0 <= self.shared_world_depth <= int(world.depth):
            raise ValueError("shared_world_depth must be between 0 and world.depth")
        self.action = ActionDiT(
            width=width, depth=depth, heads=heads, mlp_ratio=mlp_ratio,
            past_horizon=past_horizon)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.world.encode_history(history)

    def encode_history_features(self, history: torch.Tensor,
                                timestep: torch.Tensor | None = None
                                ) -> torch.Tensor:
        """Return causal H6 features from the pretrained World blocks.

        A zero flow time is used by default because Action DiT has its own
        independent flow time.  Passing a nonzero time is supported for
        diagnostics, but action deployment should keep this feature time at
        zero so it never depends on an unobserved future denoising state.
        """
        if timestep is None:
            timestep = torch.zeros(
                len(history), device=history.device, dtype=history.dtype)
        return self.world.encode_history_features(
            history, timestep, depth=self.shared_world_depth)

    def forward(self, history: torch.Tensor, noisy_future: torch.Tensor,
                future_timestep: torch.Tensor, noisy_action: torch.Tensor,
                action_timestep: torch.Tensor, goal: torch.Tensor,
                proprio: torch.Tensor, past_actions: torch.Tensor,
                past_mask: torch.Tensor,
                action_mask: torch.Tensor | None = None):
        # Keep the original H0 path for the World loss.  Action receives H6,
        # not just the raw tokenizer output, so the pretrained temporal
        # representation can influence action generation.
        shared = self.encode_history(history)
        action_features = self.encode_history_features(history)
        world_velocity = self.world(
            history, noisy_future, future_timestep, history_tokens=shared)
        action_velocity = self.action(
            action_features, noisy_action, action_timestep, goal, proprio,
            past_actions, past_mask, action_mask=action_mask)
        return world_velocity, action_velocity


def flow_sample_action(model, history_tokens, goal, proprio, past_actions,
                       past_mask, steps=20, generator=None):
    """Euler flow sampling in normalized action coordinates [0, 1]."""
    batch = len(history_tokens)
    value = torch.randn(
        batch, model.action_horizon, model.action_dim,
        device=history_tokens.device, dtype=history_tokens.dtype,
        generator=generator)
    schedule = torch.linspace(1, 0, steps + 1, device=value.device,
                              dtype=value.dtype)
    for current, following in zip(schedule[:-1], schedule[1:]):
        timestep = torch.full((batch,), current, device=value.device,
                              dtype=value.dtype)
        velocity = model(history_tokens, value, timestep, goal, proprio,
                         past_actions, past_mask)
        value = value + (following - current) * velocity
    # Rectified-flow integration already targets the normalized action
    # interval.  A sigmoid here would distort the learned endpoint and make
    # the deployed action distribution systematically too conservative.
    return value.clamp(0.0, 1.0)
