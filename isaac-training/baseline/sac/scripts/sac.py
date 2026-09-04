import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from einops.layers.torch import Rearrange
from copy import deepcopy


def _to_action_tensor(value, action_dim: int, default, *, dtype=torch.float32):
    if value is None:
        value = default
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone().to(dtype=dtype)
    elif isinstance(value, (int, float)):
        tensor = torch.full((action_dim,), float(value), dtype=dtype)
    else:
        values = [float(v) for v in value]
        if len(values) == 1:
            values = values * action_dim
        if len(values) != action_dim:
            raise ValueError(
                f"Expected scalar or {action_dim} per-axis values, got {len(values)}: {values}"
            )
        tensor = torch.tensor(values, dtype=dtype)
    if tensor.numel() == 1:
        tensor = tensor.expand(action_dim).clone()
    if tensor.shape != (action_dim,):
        tensor = tensor.reshape(-1)
    if tensor.numel() != action_dim:
        raise ValueError(
            f"Expected scalar or {action_dim} per-axis values, got shape {tuple(tensor.shape)}"
        )
    return tensor


class GaussianActor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        log_std_init: float = -1.0,
        log_std_min: float = -5.0,
        log_std_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "log_std_init",
            _to_action_tensor(log_std_init, action_dim, -1.0),
        )
        self.register_buffer(
            "log_std_min",
            _to_action_tensor(log_std_min, action_dim, -5.0),
        )
        self.register_buffer(
            "log_std_max",
            _to_action_tensor(log_std_max, action_dim, 0.5),
        )
        self.mean_layer = nn.LazyLinear(action_dim)
        self.log_std_layer = nn.LazyLinear(action_dim)

    def forward(self, features: torch.Tensor):
        loc = self.mean_layer(features)
        raw_log_std = self.log_std_layer(features)
        log_std = torch.max(torch.min(raw_log_std, self.log_std_max), self.log_std_min)
        return loc, log_std


class ObservationEncoder(nn.Module):
    # 修复 P2：把维度抽成命名常量。
    # FEATURE_DIM = LIDAR_FEATURE_DIM(128) + STATE_DIM(8) + DYN_OBS_FEATURE_DIM(64) = 200
    LIDAR_FEATURE_DIM = 128
    DYN_OBS_FEATURE_DIM = 64
    STATE_DIM = 8
    FEATURE_DIM = LIDAR_FEATURE_DIM + STATE_DIM + DYN_OBS_FEATURE_DIM

    def __init__(self):
        super().__init__()
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=[5, 3], padding=[2, 1]),
            nn.ELU(),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]),
            nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(self.LIDAR_FEATURE_DIM),
            nn.LayerNorm(self.LIDAR_FEATURE_DIM),
        )
        self.dynamic_obstacle_encoder = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.DYN_OBS_FEATURE_DIM),
            nn.LeakyReLU(),
            nn.LayerNorm(self.DYN_OBS_FEATURE_DIM),
        )
        self.output_norm = nn.LayerNorm(self.FEATURE_DIM)

    def forward(self, observation):
        if "state" not in observation.keys() and ("agents", "observation") in observation.keys(True, True):
            observation = observation["agents", "observation"]
        state = observation["state"]
        lidar = observation["lidar"]
        dynamic_obstacle = observation["dynamic_obstacle"]

        batch_shape = state.shape[:-1]
        state_flat = state.reshape(-1, state.shape[-1])
        lidar_flat = lidar.reshape(-1, *lidar.shape[-3:])
        dynamic_obstacle_flat = dynamic_obstacle.reshape(-1, *dynamic_obstacle.shape[-3:])

        lidar_feature = self.lidar_encoder(lidar_flat)
        dynamic_obstacle_feature = self.dynamic_obstacle_encoder(dynamic_obstacle_flat)
        feature = torch.cat([lidar_feature, state_flat, dynamic_obstacle_feature], dim=-1)
        feature = self.output_norm(feature)
        return feature.reshape(*batch_shape, feature.shape[-1])


def vec_to_new_frame(vec, goal_direction):
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x, dim=-1)
    goal_direction_y = goal_direction_y / goal_direction_y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y, dim=-1)
    goal_direction_z = goal_direction_z / goal_direction_z.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    vec_x_new = (vec * goal_direction_x).sum(dim=-1, keepdim=True)
    vec_y_new = (vec * goal_direction_y).sum(dim=-1, keepdim=True)
    vec_z_new = (vec * goal_direction_z).sum(dim=-1, keepdim=True)
    return torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)
    return vec_to_new_frame(vec, world_frame_new)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        log_std_init=-1.0,
        log_std_min=-5.0,
        log_std_max=0.5,
    ):
        super().__init__()
        self.feature_extractor = ObservationEncoder().to(device)
        self.actor_body = nn.Sequential(
            nn.Linear(ObservationEncoder.FEATURE_DIM, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
        ).to(device)
        self.actor_head = GaussianActor(
            action_dim,
            log_std_init=log_std_init,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(device)

    def _params(self, state):
        features = self.actor_body(self.feature_extractor(state))
        return self.actor_head(features)

    def forward(self, state, deterministic=False):
        loc, log_std = self._params(state)
        std = log_std.exp().clamp(min=1e-6)
        if deterministic:
            action = torch.tanh(loc)
        else:
            action = torch.tanh(torch.distributions.Normal(loc, std).rsample())
        return action, loc, log_std

    def sample_with_log_prob(self, state):
        loc, log_std = self._params(state)
        std = log_std.exp().clamp(min=1e-6)
        normal = torch.distributions.Normal(loc, std)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        # 修复 P0：用数值稳定的 tanh-Gaussian log_prob 公式
        # log(1 - tanh^2(x)) = 2 * (log(2) - x - softplus(-2x))
        # 旧实现 -log(1-a^2+1e-6) 在 |pre_tanh|>3 时严重失真。
        log_prob = normal.log_prob(pre_tanh)
        log_prob = log_prob - 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        return action, log_prob.sum(-1), loc, log_std

class CriticNetwork(nn.Module):
    def __init__(self,obs_dim ,action_dim, device):
        super().__init__()
        self.feature_extractor = ObservationEncoder().to(device)

        # Q网络
        self.qvalue = nn.Sequential(
            nn.Linear(ObservationEncoder.FEATURE_DIM + action_dim, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        ).to(device)

    def forward(self, s,a):
        features = self.feature_extractor(s)
        if a.ndim == features.ndim + 1 and a.shape[-2] == 1:
            a = a.squeeze(-2)
        critic_input = torch.cat([features, a], dim=-1)
        return self.qvalue(critic_input)
    



class SAC(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        # Initialize the SAC agent with configuration, observation and action specs, and device
        self.cfg = cfg
        self.obs_dim = observation_spec
        self.act_dim = action_spec
        self.device = torch.device(device)
        self.act_dim = self._resolve_action_dim(action_spec)
        # Initialize networks
        actor_cfg = getattr(cfg, "actor", {})
        log_std_init = getattr(actor_cfg, "log_std_init", -1.0)
        log_std_min = getattr(actor_cfg, "log_std_min", -5.0)
        log_std_max = getattr(actor_cfg, "log_std_max", 0.5)
        self.actor = ActorNetwork(
            self.obs_dim,
            self.act_dim,
            self.device,
            log_std_init=log_std_init,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(self.device)
        self.critic1 = CriticNetwork(self.obs_dim, self.act_dim, self.device).to(self.device)
        self.critic2 = CriticNetwork(self.obs_dim, self.act_dim, self.device).to(self.device)
        log_std_min_tensor = self.actor.actor_head.log_std_min
        log_std_max_tensor = self.actor.actor_head.log_std_max
        if torch.any(log_std_min_tensor > log_std_max_tensor):
            raise ValueError(
                "actor.log_std_min must be <= actor.log_std_max for every action axis, "
                f"got min={log_std_min_tensor.detach().cpu().tolist()}, "
                f"max={log_std_max_tensor.detach().cpu().tolist()}"
            )
        #Initialize Temperature parameter
        alpha_init = float(getattr(cfg, "alpha_init", 1.0))
        if alpha_init <= 0.0:
            raise ValueError(f"alpha_init must be > 0, got {alpha_init}")
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha_init, device=device)))
        self.alpha = self.log_alpha.exp().detach()
        self.min_alpha = float(getattr(cfg, "min_alpha", 0.0))
        self.max_alpha = float(getattr(cfg, "max_alpha", float("inf")))
        if self.min_alpha < 0.0:
            raise ValueError(f"min_alpha must be >= 0, got {self.min_alpha}")
        if math.isfinite(self.max_alpha) and self.max_alpha <= 0.0:
            raise ValueError(f"max_alpha must be > 0, got {self.max_alpha}")
        if math.isfinite(self.max_alpha) and self.min_alpha > self.max_alpha:
            raise ValueError(
                f"min_alpha must be <= max_alpha, got min_alpha={self.min_alpha}, "
                f"max_alpha={self.max_alpha}"
            )
        self.target_entropy = float(getattr(cfg, "target_entropy", self.act_dim))
        
        #Initialize Parameters
        self.gamma = getattr(cfg, 'gamma', 0.99)
        self.action_limit = getattr(cfg.actor, 'action_limit', 2.0)
        self.grad_clip_norm = float(getattr(cfg, "grad_clip_norm", 0.0))
        self.reward_scale = float(getattr(cfg, "reward_scale", 1.0))
        self.target_q_clip = float(getattr(cfg, "target_q_clip", 0.0))
        self.critic_loss_type = str(getattr(getattr(cfg, "critic", {}), "loss", "mse")).lower()
        self.actor_update_interval = max(1, int(getattr(cfg, "actor_update_interval", 1)))
        self.n_step = max(1, int(getattr(cfg, "n_step", 1)))
        self.bootstrap_gamma = self.gamma ** self.n_step
        self.learn_alpha = bool(getattr(cfg, "learn_alpha", True))
        self.update_step = 0
        # debug_log.log_grad_norm 由外部 set_debug_options() 注入，默认 false → 不算 grad norm。
        self._record_grad_norm = False

        self._materialize_lazy_modules(observation_spec)

        # v16 改回 v1 的初始化策略：所有 Linear/Conv2d 用 orthogonal_(weight, gain=0.01)。
        # 之前 v15 用 leaky_relu gain (~1.41) 让初始 actor 输出 fan-in 量级幅度，
        # 前几个 rollout out_of_bound=0.789 就是过激初始动作的证据。
        # v1 全 0.01 让初始 actor 输出接近 0（drone 几乎悬停），探索从安全状态起步，
        # 让 alpha=1.0 的高熵噪声主导早期探索，而不是被 deterministic 部分带飞。
        def init_(module):
            from torch.nn.parameter import UninitializedParameter

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                w = getattr(module, "weight", None)
                b = getattr(module, "bias", None)
                if w is None or isinstance(w, UninitializedParameter):
                    return
                nn.init.orthogonal_(module.weight, 0.01)
                if b is not None and not isinstance(b, UninitializedParameter):
                    nn.init.constant_(module.bias, 0.0)
        self.actor.apply(init_)
        self.critic1.apply(init_)
        self.critic2.apply(init_)
        self._init_actor_log_std_bias(log_std_init)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self._assert_parameters_initialized()

        # Improved optimizers with different learning rates
        # 注意：actor 优化器现在拆成两组——feature_extractor 单独一组，actor_body+actor_head 一组。
        # 目的是 AT 模式下可以单独把 feature_extractor 的 lr 缩到 feature_lr_scale × actor.lr，
        # 与 ppo.py 的 feature extractor lr 缩放语义对齐。AT 关闭时两组 lr 相同，行为与旧版完全一致。
        actor_feature_params = list(self.actor.feature_extractor.parameters())
        actor_head_params = list(self.actor.actor_body.parameters()) + list(self.actor.actor_head.parameters())
        self.actor_optim = torch.optim.Adam(
            [
                {"params": actor_feature_params, "lr": cfg.actor.learning_rate},
                {"params": actor_head_params, "lr": cfg.actor.learning_rate},
            ],
        )
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=cfg.critic.learning_rate)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=cfg.critic.learning_rate)
        if self.learn_alpha:
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_learning_rate)
        else:
            self.log_alpha.requires_grad_(False)
            self.alpha_optim = None


    def _init_actor_log_std_bias(self, log_std_init):
        # 修复 P2：必须先 materialize lazy modules 才能访问 bias，否则是 UninitializedParameter。
        from torch.nn.parameter import UninitializedParameter

        bias = self.actor.actor_head.log_std_layer.bias
        if isinstance(bias, UninitializedParameter):
            raise RuntimeError(
                "log_std_layer.bias is still UninitializedParameter; "
                "_init_actor_log_std_bias must be called after _materialize_lazy_modules."
            )
        with torch.no_grad():
            init = _to_action_tensor(log_std_init, self.act_dim, -1.0).to(
                device=bias.device,
                dtype=bias.dtype,
            )
            init = torch.max(torch.min(init, self.actor.actor_head.log_std_max), self.actor.actor_head.log_std_min)
            bias.copy_(init)

    @staticmethod
    def _resolve_action_dim(action_spec):
        if hasattr(action_spec, "keys") and ("agents", "action") in action_spec.keys(True, True):
            action_spec = action_spec[("agents", "action")]
        if hasattr(action_spec, "shape"):
            shape = tuple(action_spec.shape)
            if not shape:
                raise ValueError("action_spec has empty shape; cannot infer SAC action dimension")
            return int(shape[-1])
        return int(action_spec)

    def _materialize_lazy_modules(self, observation_spec):
        try:
            dummy_obs = observation_spec[("agents", "observation")].zero()
        except Exception:
            dummy_obs = observation_spec.zero()
            if hasattr(dummy_obs, "keys") and ("agents", "observation") in dummy_obs.keys(True, True):
                dummy_obs = dummy_obs["agents", "observation"]
        dummy_obs = dummy_obs.to(self.device)
        dummy_action = torch.zeros(
            (*tuple(dummy_obs.shape), self.act_dim),
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            self.actor(dummy_obs)
            self.critic1(dummy_obs, dummy_action)
            self.critic2(dummy_obs, dummy_action)

    def _assert_parameters_initialized(self):
        from torch.nn.parameter import UninitializedParameter

        for name, param in self.named_parameters():
            if isinstance(param, UninitializedParameter):
                raise RuntimeError(f"Uninitialized SAC parameter after dummy forward: {name}")

    def _ensure_batch_device(self, batch):
        batch_device = getattr(batch, "device", None)
        if batch_device != self.device:
            batch = batch.to(self.device, non_blocking=True)
        return batch

    @staticmethod
    def _squeeze_last_singleton(tensor):
        if tensor.ndim > 0 and tensor.shape[-1] == 1:
            return tensor.squeeze(-1)
        return tensor

    @staticmethod
    def _squeeze_agent_dim(tensor):
        if tensor.ndim >= 2 and tensor.shape[-2] == 1:
            return tensor.squeeze(-2)
        return tensor

    @staticmethod
    def _get_first_existing(tensordict, *keys):
        for key in keys:
            try:
                return tensordict[key]
            except KeyError:
                continue
        joined = ", ".join(str(key) for key in keys)
        raise KeyError(f"None of the required TensorDict keys exist: {joined}")

    def get_action(self, state, deterministic=True):
        if deterministic:
            with torch.no_grad():
                action,mu,log_std = self.actor(state, deterministic=True)
        else:
            action,mu,log_std = self.actor(state)
        return action
    def __call__(self, td):
        td = td.to(self.device)
        # 修复 P0：原写法 `action_n = torch.tanh(mu)` 在 deterministic 分支中
        # 等价于 tanh(tanh(mu))（因为 actor.forward 已经返回 tanh(loc)）。
        # 直接传 deterministic 标志给 actor，让它内部决定走 tanh(loc) 还是
        # tanh(rsample())，避免双重 tanh。
        deterministic = not self.training
        obs = td["agents", "observation"]
        action_n, _mu, _log_std = self.actor(
            obs,
            deterministic=deterministic,
        )
        # 修复 P0-2：同时把"对齐到 action 形状"的 action_n_aligned 写回 td。
        # 历史上 action_normalized 写的是 [num_envs, act_dim]，而 action 是 [num_envs, 1, act_dim]
        # （actions_to_world 内部为了 vec_to_world 把 action_n unsqueeze 了一份，但没回写）。
        # 后果：td 内部 action / action_normalized agent 维不一致；REPLAY_BUFFER_KEYS
        # 取的是 action_normalized，update() 里靠 _squeeze_agent_dim 兜底，但只要
        # 外部 transform / spec 改动稍变就会 silent broadcast。
        # 关键语义保持：action_normalized 仍是未乘 action_limit 的 tanh(.)，因为
        # update() 里 critic 训练就是在 [-1,1]^act_dim 域上学 Q(s,a)。
        actions_world = self.actions_to_world(action_n, td)
        action_n_aligned = action_n
        if action_n_aligned.ndim + 1 == actions_world.ndim and actions_world.shape[-2] == 1:
            action_n_aligned = action_n_aligned.unsqueeze(-2)
        td["agents", "action"] = actions_world
        td["agents", "action_normalized"] = action_n_aligned
        return td

    def _critic_loss(self, prediction, target):
        if self.critic_loss_type in {"huber", "smooth_l1", "smoothl1"}:
            return F.smooth_l1_loss(prediction, target)
        return F.mse_loss(prediction, target)

    def update(self, replay_buffer, batch_size, tau=0.005):
        """SAC training step with improved stability"""
        train_tds = []
        # Sample batch
        for _ in range(self.cfg.num_minibatches):
            batch = self._ensure_batch_device(replay_buffer.sample(batch_size))
            states = batch['agents','observation']
            actions = batch['agents','action_normalized']  # Normalize actions
            actions = self._squeeze_agent_dim(actions)
            rewards = self._squeeze_last_singleton(batch['next', 'agents','reward']).float() * self.reward_scale
            next_states = batch['next', 'agents','observation']
            done_tensor = self._get_first_existing(
                batch,
                ("next", "terminated"),
                ("next", "done"),
            )
            dones = self._squeeze_last_singleton(done_tensor).to(torch.bool).float()
            # ============ Update Critics ============
            with torch.no_grad():
                # Sample next actions using current policy
                next_actions, next_log_probs, _, _ = self.actor.sample_with_log_prob(next_states)

                # Target Q values
                next_q1 = self.critic1_target(next_states, next_actions).squeeze(-1)
                next_q2 = self.critic2_target(next_states, next_actions).squeeze(-1)
                next_q = torch.min(next_q1, next_q2)
                
                # Compute target with entropy regularization
                target_q = rewards + self.bootstrap_gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)
                if self.target_q_clip > 0.0:
                    target_q = target_q.clamp(-self.target_q_clip, self.target_q_clip)
            
            # Current Q values
            q1 = self.critic1(states, actions).squeeze(-1)
            q2 = self.critic2(states, actions).squeeze(-1)
            
            # Critic losses
            critic1_loss = self._critic_loss(q1, target_q)
            critic2_loss = self._critic_loss(q2, target_q)
            
            # Update critics
            self.critic1_optim.zero_grad(set_to_none=True)
            critic1_loss.backward()
            if self.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip_norm)
            grad_norm_c1 = self._grad_l2_norm(self.critic1.parameters()) if self._record_grad_norm else None
            self.critic1_optim.step()

            self.critic2_optim.zero_grad(set_to_none=True)
            critic2_loss.backward()
            if self.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip_norm)
            grad_norm_c2 = self._grad_l2_norm(self.critic2.parameters()) if self._record_grad_norm else None
            self.critic2_optim.step()
            
            # ============ Update Actor ============
            # 修复 P1：
            # - actor_update_interval > 1 时，跳过 actor 更新的步不做无意义的 critic forward。
            # - 把 alpha 更新解耦于 actor_update_interval：critic 每步都更新，actor 隔步更新，
            #   alpha 也每步更新（基于当前采样的 log_probs.detach()），这样 alpha 不会
            #   因为 actor 半频而落后于实际策略熵。
            do_actor_update = (self.update_step % self.actor_update_interval) == 0
            actions_new, log_probs, _, _ = self.actor.sample_with_log_prob(states)

            if do_actor_update:
                self._set_critic_requires_grad(False)
                try:
                    q1_new = self.critic1(states, actions_new).squeeze(-1)
                    q2_new = self.critic2(states, actions_new).squeeze(-1)
                    q_min = torch.min(q1_new, q2_new)
                    actor_loss = (self.alpha.detach() * log_probs - q_min).mean()

                    self.actor_optim.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    if self.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                    if self._record_grad_norm:
                        grad_norm_actor_total = self._grad_l2_norm(self.actor.parameters())
                        feat_mod = getattr(self.actor, "feature_extractor", None)
                        body_mod = getattr(self.actor, "actor_body", None)
                        head_mod = getattr(self.actor, "actor_head", None)
                        grad_norm_actor_feat = self._grad_l2_norm(feat_mod.parameters()) if feat_mod is not None else 0.0
                        body_head_params = []
                        if body_mod is not None:
                            body_head_params += list(body_mod.parameters())
                        if head_mod is not None:
                            body_head_params += list(head_mod.parameters())
                        grad_norm_actor_body_head = self._grad_l2_norm(body_head_params) if body_head_params else 0.0
                    else:
                        grad_norm_actor_total = None
                        grad_norm_actor_feat = None
                        grad_norm_actor_body_head = None
                    self.actor_optim.step()
                finally:
                    self._set_critic_requires_grad(True)
            else:
                # 不做 actor forward，节省一次 critic forward。
                # 用 critic 这步算出来的 q1/q2 做日志近似（而不是再跑一次 actor）。
                actor_loss = torch.zeros((), device=self.device)
                q1_new = q1.detach()
                q_min = torch.minimum(q1.detach(), q2.detach())
                grad_norm_actor_total = None
                grad_norm_actor_feat = None
                grad_norm_actor_body_head = None

            # ============ Update Temperature ============
            entropy = -log_probs.detach()
            if self.learn_alpha:
                alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
                self.alpha_optim.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optim.step()
                if self.min_alpha > 0.0 or math.isfinite(self.max_alpha):
                    min_log_alpha = math.log(max(self.min_alpha, 1e-8))
                    max_log_alpha = math.log(self.max_alpha) if math.isfinite(self.max_alpha) else float("inf")
                    self.log_alpha.data.clamp_(min=min_log_alpha, max=max_log_alpha)
            else:
                alpha_loss = torch.zeros((), device=self.device)
            self.alpha = self.log_alpha.exp().detach()
            
            # ============ Soft Update Target Networks ============
            self._soft_update(self.critic1_target, self.critic1, tau)
            self._soft_update(self.critic2_target, self.critic2, tau)
            

            train_td_dict = {
                "actor_loss": actor_loss.detach(),
                "q1_loss": critic1_loss.detach(),
                "q2_loss": critic2_loss.detach(),
                "alpha": self.alpha.detach(),
                "policy_entropy": entropy.mean(),
                "actor_update": torch.tensor(float(do_actor_update), device=self.device),
                "q1": q1.detach().mean(),
                "q_min": q_min.detach().mean(),
                "q1_new": q1_new.detach().mean(),
                "td_error": (q1.detach() - target_q).mean(),
                "td_error_abs": (q1.detach() - target_q).abs().mean(),
            }
            # grad norm（仅 cfg.debug_log.log_grad_norm=true 时计算）。
            # 占位 0.0 保证 stack 时 key 一致，避免某些 minibatch 缺 key 导致 KeyError。
            if self._record_grad_norm:
                _g = lambda v: torch.tensor(float(v) if v is not None else 0.0, device=self.device)
                train_td_dict.update({
                    "grad_norm/critic1":         _g(grad_norm_c1),
                    "grad_norm/critic2":         _g(grad_norm_c2),
                    "grad_norm/actor_total":     _g(grad_norm_actor_total),
                    "grad_norm/actor_feature":   _g(grad_norm_actor_feat),
                    "grad_norm/actor_body_head": _g(grad_norm_actor_body_head),
                })
            train_td = TensorDict(train_td_dict, [])
            train_tds.append(train_td)
            self.update_step += 1
        loss_infos = torch.stack(train_tds).to_tensordict()
        loss_infos = loss_infos.apply(torch.mean, batch_size=[])
        return {k: v.mean().item() for k, v in loss_infos.items()}
    def actions_to_world(self, actions, tensordict):
        """将归一化动作 (in [-1,1]^act_dim) 转到世界系，乘上 action_limit 后变速度向量。"""
        actions = actions * self.cfg.actor.action_limit
        direction = tensordict["agents", "observation", "direction"]
        if direction.ndim == actions.ndim + 1 and direction.shape[-2] == 1:
            actions = actions.unsqueeze(-2)
        actions_world = vec_to_world(actions, direction)
        return actions_world
    def _soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    def _set_critic_requires_grad(self, requires_grad):
        for critic in (self.critic1, self.critic2):
            for param in critic.parameters():
                param.requires_grad_(requires_grad)

    def set_debug_options(self, *, log_grad_norm: bool = False) -> None:
        """由 train_sac.py 在初始化完 policy 后调用一次，把 debug 开关注入。"""
        self._record_grad_norm = bool(log_grad_norm)

    @staticmethod
    def _grad_l2_norm(params) -> float:
        total = 0.0
        any_grad = False
        for p in params:
            if p is None or p.grad is None:
                continue
            any_grad = True
            total += float(p.grad.detach().pow(2).sum().item())
        return math.sqrt(total) if any_grad else 0.0