"""Epona MST temporal/spatial blocks adapted to NavRL LiDAR latent history.

Only the conditioning adapter is new. The circular VAE and DDPM UNet remain
unchanged, and the UNet still produces the final diffusion-generated frame.
"""

import torch
from torch import nn

from third_party.epona.stt import CausalTimeSpaceBlock, GPTConfig


class LidarHistoryMST(nn.Module):
    def __init__(self, width=256, heads=8, blocks=2, max_history=5):
        super().__init__()
        config = GPTConfig(
            block_size=max_history, n_embd=width, n_head=heads,
            attn_pdrop=0.0, resid_pdrop=0.0,
            patch_size=(27, 5), condition_frames=max_history,
            token_size_dict={"pose_tokens_size": 1, "yaw_token_size": 0,
                             "img_tokens_size": 135, "total_tokens_size": 136})
        self.image_projector = nn.Linear(4, width)
        self.action_projector = nn.Sequential(nn.Linear(30, width), nn.SiLU(),
                                               nn.Linear(width, width))
        self.time_embedding = nn.Parameter(torch.zeros(1, max_history, 1, width))
        nn.init.normal_(self.time_embedding, std=0.02)
        self.blocks = nn.ModuleList(CausalTimeSpaceBlock(config)
                                    for _ in range(blocks))
        self.output = nn.Linear(width, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("causal_mask", torch.triu(
            torch.full((max_history, max_history), float("-inf")), diagonal=1),
            persistent=False)

    def forward(self, history, past_actions):
        """Return an MST-conditioned previous latent for the frozen UNet.

        history: [B, T, 4, 27, 5] oldest to newest; past_actions: [B, T, 10, 3]
        contains only action chunks that led to those *observed* frames.
        """
        batch, count, channels, height, width = history.shape
        if channels != 4 or (height, width) != (27, 5):
            raise ValueError("Expected circular VAE latents [B,T,4,27,5]")
        if past_actions.shape != (batch, count, 10, 3):
            raise ValueError("Expected one past ten-action chunk per observation")
        if count > self.time_embedding.shape[1] or count < 1:
            raise ValueError("Invalid history length")
        images = history.permute(0, 1, 3, 4, 2).reshape(batch, count, 135, 4)
        image_tokens = self.image_projector(images)
        action_tokens = self.action_projector(
            past_actions.reshape(batch, count, 30)).unsqueeze(2)
        tokens = torch.cat((action_tokens, image_tokens), dim=2)
        tokens = tokens + self.time_embedding[:, -count:]
        mask = self.causal_mask[:count, :count]
        for block in self.blocks:
            tokens = block(tokens, mask)
        correction = self.output(tokens[:, -1, 1:]).reshape(
            batch, height, width, 4).permute(0, 3, 1, 2)
        return history[:, -1] + correction
