"""Standalone, noise-to-action flow policy; no deterministic Direct branch."""

from __future__ import annotations

import torch
from torch import nn

from lidar_wam.models.action_expert import (
    ActionFlowBlock, LiDARObservationEncoder, sinusoidal_embedding,
)
from lidar_wam.models.action_flow_sampling import sample_action_flow


# v3 changes the action domain from a unit L2 ball to a per-axis unit box.
# Do not silently reinterpret a legacy checkpoint's deployment contract.
FLOW_ACTION_EXPERT_FORMAT = "navrl-action-expert-flow-history1-v3"


class CurrentObservationActionFlow(nn.Module):
    """Action flow conditioned only on 135 LiDAR, one goal and one state token.

    No historical-action embedding, mask, placeholder token, or candidate
    head is constructed. The block implementation is shared with the legacy
    expert, but this network has its own history-free parameter layout.
    """

    def __init__(self, width=512, depth=8, heads=8, ffn_width=2048,
                 horizon=10, action_dim=3, proprio_dim=10):
        super().__init__()
        self.horizon, self.action_dim, self.width = horizon, action_dim, width
        self.action_in = nn.Linear(action_dim, width)
        self.action_position = nn.Parameter(torch.randn(1, horizon, width) * .02)
        self.goal_in = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        self.proprio_in = nn.Sequential(
            nn.Linear(proprio_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            ActionFlowBlock(width, heads, ffn_width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, noisy_action, timestep, observation_tokens, goal, proprio):
        if tuple(noisy_action.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError(f"unexpected noisy action shape {tuple(noisy_action.shape)}")
        context = torch.cat((observation_tokens, self.goal_in(goal)[:, None],
                             self.proprio_in(proprio)[:, None]), dim=1)
        time = self.time(sinusoidal_embedding(timestep, self.width))
        action = self.action_in(noisy_action) + self.action_position + time[:, None]
        for block in self.blocks:
            action = block(action, context, time)
        return self.out(self.norm(action))


class FlowActionExpertModel(nn.Module):
    """Encode current LiDAR and generate ten Lee velocity commands by flow.

    The frozen VAE remains outside this module. Only the observation encoder
    and history-free action flow are registered: there is no Direct head or World
    model. Forward predicts the interpolation field for a random training
    sigma; predict_action integrates from Gaussian noise in ten Euler steps.
    Final normalized outputs are bounded per axis, not by vector norm.
    With the training/deployment scale [2, 2, 2], each ego-FLU velocity axis
    is in [-2, 2] m/s and total speed can reach 2*sqrt(3) m/s.
    """

    format = FLOW_ACTION_EXPERT_FORMAT
    flow_only = True

    def __init__(self, width=512, depth=8, heads=8, ffn_width=2048,
                 horizon=10, action_dim=3, proprio_dim=10,
                 lidar_history_frames=1, sample_steps=10):
        super().__init__()
        if int(lidar_history_frames) != 1:
            raise ValueError("this Flow-only policy requires one current LiDAR frame")
        if isinstance(sample_steps, bool) or int(sample_steps) != sample_steps or sample_steps <= 0:
            raise ValueError("sample_steps must be a positive integer")
        self.lidar_history_frames = 1
        self.sample_steps = int(sample_steps)
        self.observation = LiDARObservationEncoder(width, history_frames=1)
        self.diffusion = CurrentObservationActionFlow(
            action_dim=action_dim, horizon=horizon, width=width, depth=depth,
            heads=heads, ffn_width=ffn_width, proprio_dim=proprio_dim)

    def encode_current(self, latent):
        return self.observation(latent)

    def forward(self, latent, noisy_action, timestep, goal, proprio):
        return self.diffusion(noisy_action, timestep, self.encode_current(latent),
                              goal, proprio)

    @torch.no_grad()
    def predict_action(self, latent, goal, proprio, *,
                       steps=None, generator=None, initial_noise=None,
                       noise_scale=1.0, project_output=True):
        return sample_action_flow(
            self.diffusion, self.encode_current(latent), goal, proprio,
            steps=self.sample_steps if steps is None else steps,
            generator=generator, initial_noise=initial_noise,
            noise_scale=noise_scale, project_output=project_output,
            output_constraint="unit_box")
