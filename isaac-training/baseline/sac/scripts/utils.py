"""Small geometry helpers used by NavigationEnv."""
import torch


def vec_to_new_frame(vec, goal_direction):
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    goal_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=vec.device).expand_as(goal_x)
    goal_y = torch.cross(z_axis, goal_x, dim=-1)
    goal_y = goal_y / goal_y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    goal_z = torch.cross(goal_x, goal_y, dim=-1)
    goal_z = goal_z / goal_z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if vec.ndim == 3:
        return torch.cat([
            torch.bmm(vec, goal_x.view(-1, 3, 1)),
            torch.bmm(vec, goal_y.view(-1, 3, 1)),
            torch.bmm(vec, goal_z.view(-1, 3, 1)),
        ], dim=-1)
    return torch.cat([
        torch.bmm(vec.view(-1, 1, 3), goal_x.view(-1, 3, 1)),
        torch.bmm(vec.view(-1, 1, 3), goal_y.view(-1, 3, 1)),
        torch.bmm(vec.view(-1, 1, 3), goal_z.view(-1, 3, 1)),
    ], dim=-1)


def vec_to_world(vec, goal_direction):
    world_x = torch.tensor([1.0, 0.0, 0.0], device=vec.device).expand_as(goal_direction)
    return vec_to_new_frame(vec, vec_to_new_frame(world_x, goal_direction))


def construct_input(start, end):
    return "(" + "|".join(str(index) for index in range(start, end)) + ")"