import os
import time

import torch
import torch.nn as nn
import wandb
import numpy as np
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

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
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        return alpha, beta

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        done: torch.Tensor,
        value: torch.Tensor, 
        next_value: torch.Tensor,
        time_scale: torch.Tensor=None,
    ):
        num_steps = done.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - done.float()
        if time_scale is None:
            time_scale = torch.ones_like(reward)
        gae = 0
        for step in reversed(range(num_steps)):
            discount = self.gamma.pow(time_scale[:, step])
            trace_decay = self.lmbda.pow(time_scale[:, step])
            delta = (
                reward[:, step] 
                + discount * next_value[:, step] * not_done[:, step]
                - value[:, step]
            )
            advantages[:, step] = gae = (
                delta + discount * trace_decay * not_done[:, step] * gae
            )
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) 
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MEAN
):

    eval_cfg = cfg.get("eval", {})
    record_video = bool(eval_cfg.get("record_video", False))
    env.enable_render(record_video)
    env.eval()
    env.set_seed(seed)
    # Keep repeated evaluation runs reproducible while preserving random
    # timing during the rollout when timing.randomize_in_eval is enabled.
    if hasattr(env, "reset_timing_schedule"):
        env.reset_timing_schedule()

    render_callback = RenderCallback(interval=2) if record_video else None
    
    with set_exploration_type(exploration_type):
        rollout_kwargs = {
            "max_steps": int(eval_cfg.get("max_steps", env.max_episode_length)),
            "policy": policy,
            "auto_reset": True,
            "break_when_any_done": False,
            "return_contiguous": False,
        }
        if render_callback is not None:
            rollout_kwargs["callback"] = render_callback
        trajs = env.rollout(**rollout_kwargs)
    env.enable_render(not cfg.headless and record_video)
    env.reset()
    
    done = trajs.get(("next", "done")).bool()
    first_done = torch.argmax(done.long(), dim=1).cpu()
    has_done = done.any(dim=1).cpu()
    last_step = torch.full_like(first_done, done.shape[1] - 1)
    first_done = torch.where(has_done, first_done, last_step)

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    info = {
        "eval/stats." + k: torch.mean(v.float()).item() 
        for k, v in traj_stats.items()
    }

    # RenderCallback records every two policy transitions. Use the measured
    # duration from this rollout because inference delay is randomized.
    eval_transition_dt = float(
        traj_stats.get(
            "transition_dt",
            torch.tensor(float(cfg.sim.dt) * float(cfg.sim.substeps)),
        )
        .float()
        .mean()
        .item()
    )
    eval_transition_dt = max(eval_transition_dt, 1e-6)
    if render_callback is not None:
        frames = render_callback.get_video_array(axes="t h w c")
        video_dir = str(eval_cfg.get("video_dir", "videos"))
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(
            video_dir, f"delay_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        try:
            import imageio.v2 as imageio

            imageio.mimsave(video_path, frames, fps=0.5 / eval_transition_dt)
            print(f"[NavRL]: saved evaluation video to {os.path.abspath(video_path)}", flush=True)
        except Exception as exc:
            print(f"[NavRL]: warning: could not save evaluation video: {exc}", flush=True)
        info["recording"] = wandb.Video(
            np.transpose(frames, (0, 3, 1, 2)),
            fps=0.5 / eval_transition_dt,
            format="mp4"
        )
    env.train()
    # env.reset()

    return info


def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)
    # print("vec: ", vec.shape)

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


def construct_input(start, end):
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"
