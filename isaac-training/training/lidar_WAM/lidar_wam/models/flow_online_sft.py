"""Differentiable deployment-equivalent Euler sampling for online Flow SFT.

PPO supplies only action zero. No synthetic ten-command expert trajectory is
created. Student/reference use identical initial noise for selective SFT.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DifferentiableFlowPolicy(nn.Module):
    def __init__(self, expert, project_output=True, checkpoint_steps=True):
        super().__init__()
        self.expert = expert
        self.project_output = bool(project_output)
        self.checkpoint_steps = bool(checkpoint_steps)

    def forward(self, latent, goal, proprio, initial_noise=None):
        tokens = self.expert.encode_current(latent)
        shape = (len(latent), self.expert.diffusion.horizon, 3)
        action = (torch.randn(shape, device=latent.device, dtype=torch.float32)
                  if initial_noise is None else initial_noise.float().clone())
        if tuple(action.shape) != shape or not bool(torch.isfinite(action).all()):
            raise ValueError("Expected finite initial noise [B,10,3]")
        for index in range(self.expert.sample_steps):
            time = action.new_full((len(action),), 1000. * (1. - index/self.expert.sample_steps))
            if self.checkpoint_steps and torch.is_grad_enabled():
                velocity = checkpoint(self.expert.diffusion, action, time, tokens,
                                      goal, proprio, use_reentrant=False)
            else:
                velocity = self.expert.diffusion(action, time, tokens, goal, proprio)
            action = action - velocity.float()/self.expert.sample_steps
        return action.clamp(-1., 1.) if self.project_output else action
