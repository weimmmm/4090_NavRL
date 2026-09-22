"""Network components and checkpoint buffers used by baseline inference."""
import torch
import torch.nn as nn

class ValueNorm(nn.Module):
    """Normalization buffers stored in baseline checkpoints."""

    def __init__(self, input_shape):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        return alpha, beta

class GAE(nn.Module):
    """Discount buffers stored in baseline checkpoints, not a GAE calculator."""

    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))

def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # goal direction y
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coodirnate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel
