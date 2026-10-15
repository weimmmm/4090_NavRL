"""Coordinate helpers retained by the Isaac navigation environment."""

import torch


def vec_to_new_frame(vec, goal_direction):
    if len(vec.size()) == 1:
        vec = vec.unsqueeze(0)

    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    z_direction = torch.tensor([0, 0, 1], device=vec.device, dtype=vec.dtype)
    goal_direction_y = torch.cross(
        z_direction.expand_as(goal_direction_x), goal_direction_x, dim=-1
    )
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y, dim=-1)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))
    return torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor(
        [1.0, 0.0, 0.0], device=vec.device, dtype=vec.dtype
    ).expand_as(goal_direction)
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)
    return vec_to_new_frame(vec, world_frame_new)


def construct_input(start, end):
    return "(" + "|".join(str(n) for n in range(start, end)) + ")"
