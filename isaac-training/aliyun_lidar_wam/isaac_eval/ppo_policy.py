"""Inference-only PPO expert, matching the policy used to collect ``wam_data``."""

from __future__ import annotations

from typing import Iterable, Union

import torch
from torch import nn
from einops.layers.torch import Rearrange
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

from isaac_eval.nav_utils import vec_to_world
from isaac_eval.ppo_lidar_encoder import RangeImageEncoder


class ValueNorm(nn.Module):
    def __init__(self, input_shape: Union[int, Iterable], beta=0.995, epsilon=1e-5):
        super().__init__()
        self.input_shape = (torch.Size(input_shape) if isinstance(input_shape, Iterable)
                            else torch.Size((input_shape,)))
        self.epsilon = epsilon
        self.beta = beta
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))


def make_mlp(num_units):
    layers = []
    for width in num_units:
        layers.extend((nn.LazyLinear(width), nn.LeakyReLU(), nn.LayerNorm(width)))
    return nn.Sequential(*layers)


class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {
        "alpha": torch.distributions.constraints.positive,
        "beta": torch.distributions.constraints.positive,
    }

    def __init__(self, alpha, beta, validate_args=None):
        super().__init__(
            torch.distributions.Beta(alpha, beta), 1,
            validate_args=validate_args)


class BetaActor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()

    def forward(self, features):
        alpha = 1.0 + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1.0 + self.beta_softplus(self.beta_layer(features)) + 1e-6
        return alpha, beta


class GAE(nn.Module):
    """Included so the collector checkpoint loads strictly, though unused at inference."""

    def __init__(self, gamma=0.99, lmbda=0.95):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))


class PPOExpert(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device, lidar_range):
        super().__init__()
        self.cfg = cfg
        self.device = device
        feature_extractor_network = RangeImageEncoder(lidar_range).to(device)
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"), make_mlp([128, 64])).to(device)
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(
                feature_extractor_network,
                [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(
                dynamic_obstacle_network,
                [("agents", "observation", "dynamic_obstacle")],
                ["_dynamic_obstacle_feature"]),
            CatTensors(
                ["_cnn_feature", ("agents", "observation", "state"),
                 "_dynamic_obstacle_feature"],
                "_feature", del_keys=False),
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(device)
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(
                BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True,
        ).to(device)
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"]).to(device)
        self.value_norm = ValueNorm(1).to(device)
        self.gae = GAE().to(device)
        self(observation_spec.zero())

    def forward(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        normalized = tensordict["agents", "action_normalized"]
        actions = 2 * normalized * self.cfg.actor.action_limit - self.cfg.actor.action_limit
        direction = tensordict["agents", "observation", "direction"]
        tensordict["agents", "action"] = vec_to_world(actions, direction)
        return tensordict


def load_ppo_expert(checkpoint, cfg, observation_spec, action_spec, device,
                    lidar_range):
    policy = PPOExpert(
        cfg, observation_spec, action_spec, device, lidar_range).eval()
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(state, strict=True)
    return policy

