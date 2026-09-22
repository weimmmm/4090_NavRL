"""Runtime-safe direct-horizon LiDAR world-model definitions.

Only PyTorch/diffusers model code belongs here.  Offline training and Isaac
deployment both import this module without pulling in dataset/scientific I/O.
"""

from __future__ import annotations

import torch
from torch import nn
from diffusers import UNet2DConditionModel


WORLD_HORIZON = 3


class ActionCondition(nn.Module):
    """Embed one ten-command action segment and the causal ego state."""

    def __init__(self, width: int = 768, state_dim: int = 5):
        super().__init__()
        self.actions = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.state = nn.Sequential(
            nn.Linear(state_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.zeros(1, 10, width))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, actions: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            self.actions(actions) + self.position,
            self.state(state).unsqueeze(1),
        ), dim=1)


class WorldModel(nn.Module):
    """One-frame conditional UNet retained for checkpoint compatibility."""

    def __init__(self, state_dim: int = 5):
        super().__init__()
        self.condition = ActionCondition(state_dim=state_dim)
        self.unet = UNet2DConditionModel(
            sample_size=(27, 5), in_channels=8, out_channels=4,
            layers_per_block=4, block_out_channels=(256, 512, 512),
            down_block_types=(
                "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=(
                "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=768, attention_head_dim=8)

    def forward(self, noisy_next: torch.Tensor, previous: torch.Tensor,
                actions: torch.Tensor, state: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
        tokens = self.condition(actions, state)
        return self.unet(
            torch.cat((noisy_next, previous), dim=1), timestep,
            encoder_hidden_states=tokens).sample


class DirectHorizonActionCondition(ActionCondition):
    """Embed three ordered ten-command segments for direct t+3 prediction."""

    def __init__(self, width: int = 768, state_dim: int = 5):
        super().__init__(width=width, state_dim=state_dim)
        self.segment = nn.Parameter(torch.zeros(1, WORLD_HORIZON, 1, width))

    def forward(self, actions: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        expected = (WORLD_HORIZON, 10, 3)
        if actions.ndim != 4 or tuple(actions.shape[1:]) != expected:
            raise ValueError(
                f"Expected actions [B,{WORLD_HORIZON},10,3], got "
                f"{tuple(actions.shape)}")
        encoded = self.actions(actions)
        encoded = encoded + self.position[:, None] + self.segment
        return torch.cat((
            encoded.flatten(1, 2), self.state(state).unsqueeze(1)
        ), dim=1)


class DirectHorizonWorldModel(WorldModel):
    """Directly predict the LiDAR latent at t+3 (0.48 seconds)."""

    def __init__(self, state_dim: int = 5):
        super().__init__(state_dim=state_dim)
        self.condition = DirectHorizonActionCondition(state_dim=state_dim)
