"""Small NavRL input adapter around the vendored NWM CDiT predictor.

The transformer blocks, initialization, patch embedder, final layer and
diffusion implementation remain the upstream NWM code. Only the rectangular
LiDAR latent, ten executed actions and previous state differ from NWM's RGB
navigation input.
"""

import torch
from torch import nn
from torch.nn import functional as F

from third_party.nwm.models import CDiT_models


LATENT_HEIGHT = 27  # circular horizontal angle axis
LATENT_WIDTH = 5    # vertical angle axis
PATCHED_HEIGHT = 28
PATCHED_WIDTH = 6


class NavRLCDiT(nn.Module):
    def __init__(self, size="CDiT-B/2"):
        super().__init__()
        self.backbone = CDiT_models[size](
            input_size=(PATCHED_HEIGHT, PATCHED_WIDTH),
            context_size=1, in_channels=4, learn_sigma=True,
        )
        width = self.backbone.x_embedder.proj.out_channels
        self.action_embed = nn.Linear(3, width)
        self.action_position = nn.Parameter(torch.zeros(10, width))
        self.state_embed = nn.Linear(11, width)
        nn.init.normal_(self.action_position, std=0.02)

    @staticmethod
    def pad_latent(z):
        if z.shape[-3:] != (4, LATENT_HEIGHT, LATENT_WIDTH):
            raise ValueError(f"Expected [B,4,27,5] circular VAE latent, got {tuple(z.shape)}")
        # The first spatial axis is the 360-degree LiDAR angle axis.
        return F.pad(torch.cat((z, z[..., :1, :]), dim=-2), (0, 1))

    @staticmethod
    def crop_latent(z):
        return z[..., :LATENT_HEIGHT, :LATENT_WIDTH]

    def forward(self, x, t, y, x_cond, rel_t):
        """NWM signature; y=(executed actions [B,10,3], previous state [B,11])."""
        actions, state = y
        if actions.shape[1:] != (10, 3) or state.shape[1:] != (11,):
            raise ValueError("Expected 10 executed 3D actions and 11 previous-state values")
        net = self.backbone
        x = net.x_embedder(x) + net.pos_embed[1]
        context = net.x_embedder(x_cond) + net.pos_embed[0]
        action_tokens = self.action_embed(actions) + self.action_position
        state_token = self.state_embed(state).unsqueeze(1)
        context = torch.cat((context, action_tokens, state_token), dim=1)
        condition = (net.t_embedder(t[:, None]) +
                     net.time_embedder(rel_t[:, None]) +
                     action_tokens.mean(dim=1) + state_token[:, 0])
        for block in net.blocks:
            x = block(x, condition, context)
        x = net.final_layer(x, condition)
        channels = net.out_channels
        patch = net.patch_size
        height, width = net.x_embedder.grid_size
        x = x.reshape(x.shape[0], height, width, patch, patch, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], channels, height * patch, width * patch)


def model_kwargs(previous, actions, state):
    return {"y": (actions, state), "x_cond": NavRLCDiT.pad_latent(previous),
            "rel_t": torch.full((len(previous),), 0.16, device=previous.device)}
