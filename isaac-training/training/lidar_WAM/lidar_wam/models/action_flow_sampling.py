"""Euler sampler for current-only flow and the legacy auxiliary-flow ablation.

Training interpolates a_sigma=(1-sigma)*a_clean+sigma*epsilon and targets
epsilon-a_clean. Sampling must therefore integrate from sigma=1 down to 0,
passing sigma*1000 to the exact same timestep embedding used in training.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def sample_action_flow(expert, observation_tokens, goal, proprio, *legacy_history,
                       steps=10, generator=None,
                       initial_noise=None, noise_scale=1.0, project_output=True,
                       output_constraint="unit_l2_ball"):
    """Generate normalized actions; never read clean expert actions.

    Only the final output is projected. The legacy default preserves the
    Direct head's unit-ball contract; the new Flow-only policy explicitly
    selects a unit box (each normalized axis in [-1, 1], not norm <= 1).
    Intermediate Euler states are not clipped or re-noised. Setting
    project_output=False returns the raw integral in either mode.
    The new standalone flow passes no history. Two legacy positional history
    tensors remain supported only for the original Direct+Flow ablation.
    """
    if isinstance(steps, bool) or int(steps) != steps or int(steps) <= 0:
        raise ValueError("steps must be a positive integer")
    steps = int(steps)
    if not math.isfinite(float(noise_scale)) or float(noise_scale) < 0:
        raise ValueError("noise_scale must be finite and non-negative")
    if len(legacy_history) not in (0, 2):
        raise ValueError("legacy experts require both history values and history mask")
    if output_constraint not in {"unit_l2_ball", "unit_box"}:
        raise ValueError("output_constraint must be unit_l2_ball or unit_box")
    shape = (len(observation_tokens), expert.horizon, expert.action_dim)
    if initial_noise is None:
        noise = torch.randn(shape, device=observation_tokens.device,
                            dtype=torch.float32, generator=generator)
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError(f"expected initial noise {shape}, got {initial_noise.shape}")
        noise = initial_noise.to(device=observation_tokens.device,
                                 dtype=torch.float32).clone()
    action = noise * float(noise_scale)
    for index in range(steps):
        timestep = torch.full((shape[0],), 1000.0 * (1.0 - index / steps),
                              device=action.device, dtype=torch.float32)
        velocity = expert(action, timestep, observation_tokens, goal, proprio,
                          *legacy_history)
        action = action - velocity.float() / steps
    if not bool(torch.isfinite(action).all()):
        raise FloatingPointError("non-finite Action Flow sample")
    if project_output:
        if output_constraint == "unit_box":
            action = action.clamp(-1.0, 1.0)
        else:
            action = action / action.norm(dim=-1, keepdim=True).clamp_min(1.0)
    return action
