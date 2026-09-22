"""Small navigation utilities copied from the NavRL training implementation.

Keeping these functions here prevents the standalone evaluator from importing
``isaac-training/training_legan`` or another source tree outside
``aliyun_lidar_wam``.
"""

from __future__ import annotations

import torch


def min_pool_ranges(distances: torch.Tensor, horizontal_sample: int,
                    vertical_sample: int) -> torch.Tensor:
    """Keep the nearest return in each non-overlapping angular block."""
    if horizontal_sample <= 0 or vertical_sample <= 0:
        raise ValueError("LiDAR sample factors must be positive")
    horizontal, vertical = distances.shape[-2:]
    if horizontal % horizontal_sample or vertical % vertical_sample:
        raise ValueError("Raw LiDAR dimensions must be divisible by sample factors")
    blocks = distances.reshape(
        *distances.shape[:-2],
        horizontal // horizontal_sample, horizontal_sample,
        vertical // vertical_sample, vertical_sample,
    )
    return blocks.amin(dim=(-3, -1))


def vec_to_new_frame(vec: torch.Tensor, goal_direction: torch.Tensor) -> torch.Tensor:
    """Express vectors in the horizontal frame whose x-axis points to the goal."""
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    batch = vec.shape[0]
    eps = torch.finfo(vec.dtype).eps
    goal_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    z = torch.tensor([0.0, 0.0, 1.0], device=vec.device, dtype=vec.dtype)
    goal_y = torch.cross(z.expand_as(goal_x), goal_x, dim=-1)
    goal_y /= goal_y.norm(dim=-1, keepdim=True).clamp_min(eps)
    goal_z = torch.cross(goal_x, goal_y, dim=-1)
    goal_z /= goal_z.norm(dim=-1, keepdim=True).clamp_min(eps)

    # Keep the training implementation's output contract: [B, K, 3] for a
    # vector sequence and [B, 1, 3] for one vector per environment.  Flattening
    # the one-vector goal basis avoids accidental four-dimensional broadcasting
    # when both ``vec`` and ``goal_direction`` contain a singleton agent axis.
    if vec.ndim == 3:
        vectors = vec.reshape(batch, vec.shape[1], 3)
    else:
        vectors = vec.reshape(batch, 1, 3)
    axes = (goal_x, goal_y, goal_z)
    components = [
        torch.bmm(vectors, axis.reshape(batch, 3, 1)) for axis in axes
    ]
    return torch.cat(components, dim=-1)


def vec_to_world(vec: torch.Tensor, goal_direction: torch.Tensor) -> torch.Tensor:
    """Match the PPO policy's goal-frame-to-world action conversion."""
    world_x = torch.tensor([1.0, 0.0, 0.0], device=vec.device,
                           dtype=vec.dtype).expand_as(goal_direction)
    world_in_goal = vec_to_new_frame(world_x, goal_direction)
    return vec_to_new_frame(vec, world_in_goal)


def construct_input(start: int, end: int) -> str:
    """Construct the regex fragment used for replicated obstacle prim paths."""
    return "(" + "|".join(str(index) for index in range(start, end)) + ")"
