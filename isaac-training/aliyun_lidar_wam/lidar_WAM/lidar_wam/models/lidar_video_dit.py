"""Lightweight block-causal Video DiT for next-frame LiDAR prediction.

The design follows the public MiniWorld/DiT recipe (latent video tokens,
block-causal attention, timestep-conditioned AdaLN and rectified flow), while
remaining dependency-free so the standalone lidar_WAM project can train in
its existing PyTorch environment.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timestep: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1))
    phase = timestep.float()[:, None] * frequency[None]
    value = torch.cat((phase.sin(), phase.cos()), dim=-1)
    if width % 2:
        value = F.pad(value, (0, 1))
    return value.to(timestep.dtype)


def polar_position(height: int, width: int, channels: int) -> torch.Tensor:
    """Periodic azimuth plus elevation encoding for flattened range tokens."""
    azimuth = torch.arange(height, dtype=torch.float32) * (2 * math.pi / height)
    elevation = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    azimuth, elevation = torch.meshgrid(azimuth, elevation, indexing="ij")
    base = torch.stack((
        azimuth.sin(), azimuth.cos(),
        (2 * azimuth).sin(), (2 * azimuth).cos(),
        elevation, (math.pi * elevation).sin(),
        (math.pi * elevation).cos()), dim=-1)
    base = base.reshape(1, 1, height * width, 7)
    return base.repeat(1, 1, 1, math.ceil(channels / 7))[..., :channels]


class VideoDiTBlock(nn.Module):
    """Pre-norm DiT block with AdaLN-Zero residual modulation."""

    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            width, heads, batch_first=True, dropout=0.0)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(approximate="tanh"),
            nn.Linear(hidden, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(value, shift, scale):
        return value * (1 + scale[:, None]) + shift[:, None]

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.modulation(condition).chunk(6, dim=-1))
        value = self._modulate(self.norm1(tokens), shift_a, scale_a)
        value = self.attention(
            value, value, value, attn_mask=attention_mask,
            need_weights=False)[0]
        tokens = tokens + gate_a[:, None] * value
        value = self._modulate(self.norm2(tokens), shift_m, scale_m)
        tokens = tokens + gate_m[:, None] * self.mlp(value)
        return tokens


class LiDARVideoDiT(nn.Module):
    """Predict one next latent from three clean historical LiDAR latents.

    Inputs:
      history: ``[B,3,4,27,5]`` clean frozen-VAE latents
      noisy_next: ``[B,4,27,5]`` rectified-flow state
      timestep: ``[B]`` in [0, 1]
    """

    history_frames = 3
    latent_channels = 4
    latent_height = 27
    latent_width = 5

    def __init__(self, width: int = 512, depth: int = 8, heads: int = 8,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.tokens_per_frame = self.latent_height * self.latent_width
        self.input_projection = nn.Linear(self.latent_channels, width)
        self.frame_position = nn.Parameter(torch.zeros(1, 4, 1, width))
        nn.init.normal_(self.frame_position, std=0.02)
        self.register_buffer(
            "spatial_position",
            polar_position(self.latent_height, self.latent_width, width),
            persistent=False)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            VideoDiTBlock(width, heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(
            width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(width, 2 * width))
        self.output_projection = nn.Linear(width, self.latent_channels)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer(
            "block_causal_mask", self._make_attention_mask(), persistent=False)

    def _make_attention_mask(self) -> torch.Tensor:
        total = 4 * self.tokens_per_frame
        history = 3 * self.tokens_per_frame
        # True means disallowed for torch MultiheadAttention.  Clean history
        # never reads the noisy prediction block; the prediction block reads
        # all three history frames and itself.
        mask = torch.zeros(total, total, dtype=torch.bool)
        mask[:history, history:] = True
        return mask

    def forward(self, history: torch.Tensor, noisy_next: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
        expected_history = (3, 4, 27, 5)
        if history.ndim != 5 or tuple(history.shape[1:]) != expected_history:
            raise ValueError(
                f"Expected history [B,3,4,27,5], got {tuple(history.shape)}")
        if noisy_next.ndim != 4 or tuple(noisy_next.shape[1:]) != (4, 27, 5):
            raise ValueError(
                f"Expected noisy next [B,4,27,5], got {tuple(noisy_next.shape)}")
        video = torch.cat((history, noisy_next[:, None]), dim=1)
        tokens = video.permute(0, 1, 3, 4, 2).reshape(
            len(video), 4, self.tokens_per_frame, 4)
        tokens = self.input_projection(tokens)
        tokens = tokens + self.frame_position + self.spatial_position.to(tokens.dtype)
        tokens = tokens.flatten(1, 2)
        condition = self.time_mlp(timestep_embedding(timestep, self.width))
        mask = self.block_causal_mask
        for block in self.blocks:
            tokens = block(tokens, condition, mask)
        future = tokens[:, -self.tokens_per_frame:]
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        future = self.final_norm(future) * (1 + scale[:, None]) + shift[:, None]
        future = self.output_projection(future)
        return future.reshape(
            len(video), self.latent_height, self.latent_width,
            self.latent_channels).permute(0, 3, 1, 2).contiguous()

