"""Compare policies under the same two-stage random timing schedule."""

import importlib.util
import os
import sys
from datetime import datetime

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from torchrl.envs.utils import ExplorationType, set_exploration_type


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
PROJECT_PATH = os.path.dirname(FILE_PATH)


def _resolve_project_path(path):
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_PATH, path))


def _load_baseline_ppo():
    """Load the bundled original 8-D PPO without shadowing delay PPO."""
    baseline_scripts = os.path.join(PROJECT_PATH, "scripts")
    module_path = os.path.join(baseline_scripts, "baseline_ppo.py")
    spec = importlib.util.spec_from_file_location("baseline_ppo_for_eval", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline PPO from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, baseline_scripts)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.PPO


class _BaselineObservationSpec:
    """Give baseline PPO a zero observation with its original 8-D state."""

    def __init__(self, spec):
        self.spec = spec

    def zero(self):
        tensordict = self.spec.zero()
        state_key = ("agents", "observation", "state")
        tensordict.set(state_key, tensordict.get(state_key)[..., :8].contiguous())
        return tensordict


class _BaselinePolicy:
    """Remove timing-aware state features before calling the old policy."""

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, tensordict):
        policy_input = tensordict.clone()
        state_key = ("agents", "observation", "state")
        policy_input.set(state_key, policy_input.get(state_key)[..., :8].contiguous())
        self.policy(policy_input)
        for key in (("agents", "action_normalized"), ("agents", "action")):
            tensordict.set(key, policy_input.get(key))
        return tensordict


class _TimedRenderCallback:
    """Render frames and keep each outer policy transition duration."""

    def __init__(self, base_env, interval=2):
        from omni_drones.utils.torchrl import RenderCallback

        self.base_env = base_env
        self.interval = int(interval)
        self._callback = RenderCallback(interval=self.interval)
        self.transition_dts = []

    def __call__(self, *args, **kwargs):
        result = self._callback(*args, **kwargs)
        # RenderCallback is invoked after the environment step, so this is
        # the interval represented by the newly rendered state.
        self.transition_dts.append(float(self.base_env.transition_dt.mean().item()))
        return result

    def get_video_array(self, *args, **kwargs):
        return self._callback.get_video_array(*args, **kwargs)


def _load_policy(policy_cls, cfg, observation_spec, action_spec, checkpoint, device, timing_reference):
    policy = policy_cls(
        cfg.algo,
        observation_spec,
        action_spec,
        device,
        timing_reference,
    ) if timing_reference is not None else policy_cls(
        cfg.algo,
        observation_spec,
        action_spec,
        device,
    )
    state_dict = torch.load(to_absolute_path(checkpoint), map_location=device)
    policy.load_state_dict(state_dict)
    return policy


def _save_results(cfg, metrics):
    result_dir = str(cfg.eval.result_dir)
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(PROJECT_PATH, result_dir)
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(result_dir, f"evaluation_{timestamp}.yaml")
    payload = {
        "implementation": "two_stage_async_fifo",
        "environment": "training_delay.NavigationEnv",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_checkpoint": to_absolute_path(str(cfg.baseline_checkpoint)),
        "delay_checkpoint": (
            to_absolute_path(str(cfg.delay_checkpoint))
            if cfg.delay_checkpoint is not None
            else None
        ),
        "dataset_path": _resolve_project_path(cfg.eval.dataset_path),
        "seed": int(cfg.seed),
        "timing": OmegaConf.to_container(cfg.timing, resolve=True),
        "metrics": metrics,
    }
    OmegaConf.save(OmegaConf.create(payload), result_path)
    return result_path


def _first_episode_stats(trajs):
    done = trajs.get(("next", "done"))
    first_done = torch.argmax(done.long(), dim=1).cpu()
    has_done = done.any(dim=1).cpu()
    last_step = torch.full_like(first_done, done.shape[1] - 1)
    first_done = torch.where(has_done, first_done, last_step)

    def take_first_episode(tensor):
        tensor = tensor.cpu()
        indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    return {
        key: take_first_episode(value)
        for key, value in trajs[("next", "stats")].items()
    }


@torch.no_grad()
def _evaluate_one(env, base_env, policy, name, cfg):
    seed = int(cfg.seed)
    env.eval()
    env.set_seed(seed, static_seed=True)
    # A dedicated timing RNG makes both policies see the same inference and
    # command delay sequence, independent of policy-dependent episode resets.
    base_env.reset_timing_schedule(seed)
    env.reset()

    record_video = bool(cfg.eval.record_video)
    env.enable_render(record_video)
    callback = None
    if record_video:
        callback = _TimedRenderCallback(base_env, interval=2)

    rollout_kwargs = {
        "max_steps": int(cfg.eval.max_steps),
        "policy": policy,
        "auto_reset": True,
        "break_when_any_done": False,
        "return_contiguous": False,
    }
    if callback is not None:
        rollout_kwargs["callback"] = callback
    with set_exploration_type(ExplorationType.MEAN):
        trajs = env.rollout(**rollout_kwargs)

    stats = _first_episode_stats(trajs)
    metrics = {
        f"{name}/{key}": float(value.float().mean())
        for key, value in stats.items()
    }

    # episode_len is already reported in equivalent nominal control steps by
    # the delay environment. Expose the physical-time equivalent separately;
    # decision_count remains the raw number of policy commands.
    if "episode_len" in stats and "episode_time" in stats:
        nominal_dt = float(cfg.sim.dt) * float(cfg.sim.substeps)
        metrics[f"{name}/equivalent_episode_len"] = (
            float(stats["episode_time"].float().mean()) / nominal_dt
        )

    if callback is not None:
        import imageio_ffmpeg

        video_dir = cfg.eval.video_dir
        if not os.path.isabs(video_dir):
            video_dir = os.path.join(PROJECT_PATH, video_dir)
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"{name}.mp4")
        video_array = callback.get_video_array(axes="t h w c")[..., :3]
        if video_array.dtype != np.uint8:
            if video_array.max() <= 1.0:
                video_array = video_array * 255.0
            video_array = np.clip(video_array, 0.0, 255.0).astype(np.uint8)
        video_array = np.ascontiguousarray(video_array)
        height, width = video_array.shape[1:3]
        # Frames are captured every two policy decisions. Use the measured
        # transition duration so playback follows simulated physical time.
        if callback.transition_dts:
            mean_transition_dt = float(np.mean(callback.transition_dts))
        else:
            mean_transition_dt = float(cfg.timing.reference_dt)
        fps = max(1, int(round(1.0 / (mean_transition_dt * callback.interval))))
        metrics[f"{name}/mean_transition_dt"] = mean_transition_dt
        metrics[f"{name}/video_fps"] = fps
        writer = imageio_ffmpeg.write_frames(
            video_path,
            size=(width, height),
            fps=fps,
            codec="libx264",
            quality=8,
        )
        writer.send(None)
        try:
            for frame in video_array:
                writer.send(frame)
        finally:
            writer.close()
        metrics[f"{name}/video_path"] = video_path

    env.enable_render(False)
    env.reset()
    return metrics


@hydra.main(config_path=FILE_PATH, config_name="eval_random", version_base=None)
def main(cfg: DictConfig):
    cfg.device = f"cuda:{cfg.gpu_id}"
    cfg.sim.device = cfg.device
    dataset_path = _resolve_project_path(cfg.eval.dataset_path)
    dataset = torch.load(dataset_path, map_location="cpu")
    cfg.env.num_envs = int(dataset["num_envs"])

    if cfg.baseline_checkpoint is None:
        raise ValueError("Set baseline_checkpoint=/path/to/baseline.pt")

    sim_app = SimulationApp(
        {
            "headless": cfg.headless,
            "anti_aliasing": 1,
            "active_gpu": cfg.gpu_id,
            "physics_gpu": cfg.gpu_id,
            "multi_gpu": False,
        }
    )

    try:
        from eval_env import TwoStageDelayEvalEnv
        from ppo import PPO as DelayPPO

        env = TwoStageDelayEvalEnv(cfg)
        transformed_env = env.eval()
        transformed_env.set_seed(cfg.seed, static_seed=True)

        baseline_ppo = _load_baseline_ppo()
        baseline_action_spec = type(
            "ActionSpecView", (), {"shape": (1, 3)}
        )()
        baseline_policy = _load_policy(
            baseline_ppo,
            cfg,
            _BaselineObservationSpec(transformed_env.observation_spec),
            baseline_action_spec,
            cfg.baseline_checkpoint,
            cfg.device,
            None,
        )
        baseline_policy = _BaselinePolicy(baseline_policy)

        metrics = {}
        if cfg.delay_checkpoint is None:
            metrics.update(_evaluate_one(transformed_env, env, baseline_policy, "random_timing", cfg))
        else:
            delay_policy = _load_policy(
                DelayPPO,
                cfg,
                transformed_env.observation_spec,
                transformed_env.action_spec,
                cfg.delay_checkpoint,
                cfg.device,
                float(cfg.timing.reference_dt),
            )
            metrics.update(_evaluate_one(transformed_env, env, baseline_policy, "baseline", cfg))
            metrics.update(_evaluate_one(transformed_env, env, delay_policy, "delay", cfg))
        printable = {key: value for key, value in metrics.items()}
        print("[NavRL]: random-delay comparison results")
        print(OmegaConf.to_yaml(OmegaConf.create(printable), sort_keys=True))
        result_path = _save_results(cfg, printable)
        print(f"[NavRL]: saved evaluation results to {result_path}")
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
