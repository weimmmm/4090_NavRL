"""Compare baseline and delay-aware policies under random two-stage timing."""

import importlib.util
import os
import sys

OMNIDRONES_SOURCE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "OmniDrones")
)
if OMNIDRONES_SOURCE not in sys.path:
    sys.path.insert(0, OMNIDRONES_SOURCE)

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from omni.isaac.kit import SimulationApp
from torchrl.envs.utils import ExplorationType


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
PROJECT_PATH = os.path.dirname(FILE_PATH)


def _resolve_checkpoint(path, name):
    if path is None:
        return None
    checkpoint = to_absolute_path(str(path))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"{name} does not exist: {checkpoint}")
    return checkpoint


def _load_baseline_ppo():
    """Load the original 8-D PPO without shadowing the delay PPO module."""
    baseline_path = os.path.join(PROJECT_PATH, "..", "training", "scripts", "ppo.py")
    baseline_path = os.path.abspath(baseline_path)
    spec = importlib.util.spec_from_file_location("baseline_ppo_for_eval", baseline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline PPO from {baseline_path}")
    module = importlib.util.module_from_spec(spec)
    baseline_scripts = os.path.dirname(baseline_path)
    sys.path.insert(0, baseline_scripts)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.PPO


class _BaselineObservationSpec:
    """Present an 8-D zero state to the original PPO lazy modules."""

    def __init__(self, spec):
        self.spec = spec

    def zero(self):
        tensordict = self.spec.zero()
        state_key = ("agents", "observation", "state")
        state = tensordict.get(state_key)
        tensordict.set(state_key, state[..., :8].contiguous())
        return tensordict


class _BaselinePolicy:
    """Strip the two timing features before calling the original policy."""

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, tensordict):
        policy_input = tensordict.clone()
        state_key = ("agents", "observation", "state")
        state = policy_input.get(state_key)
        policy_input.set(state_key, state[..., :8].contiguous())
        self.policy(policy_input)
        for key in (("agents", "action_normalized"), ("agents", "action")):
            tensordict.set(key, policy_input.get(key))
        return tensordict


def _load_policy(policy_cls, cfg, observation_spec, action_spec, checkpoint, timing_ref):
    if timing_ref is None:
        policy = policy_cls(cfg.algo, observation_spec, action_spec, cfg.device)
    else:
        policy = policy_cls(
            cfg.algo,
            observation_spec,
            action_spec,
            cfg.device,
            timing_ref,
        )
    policy.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
    return policy


@hydra.main(config_path=FILE_PATH, config_name="eval_random", version_base=None)
def main(cfg: DictConfig):
    from env import NavigationEnv
    from ppo import PPO as DelayPPO
    from utils import evaluate

    baseline_checkpoint = _resolve_checkpoint(
        cfg.get("baseline_checkpoint"), "baseline_checkpoint"
    )
    delay_checkpoint = _resolve_checkpoint(
        cfg.get("delay_checkpoint"), "delay_checkpoint"
    )
    if baseline_checkpoint is None and delay_checkpoint is None:
        raise ValueError(
            "Set baseline_checkpoint and/or delay_checkpoint to a .pt file"
        )

    sim_app = SimulationApp(
        {
            "headless": cfg.headless,
            "anti_aliasing": 1,
            "active_gpu": int(str(cfg.device).split(":")[-1]),
            "physics_gpu": int(str(cfg.device).split(":")[-1]),
            "multi_gpu": False,
        }
    )
    try:
        env = NavigationEnv(cfg).eval()
        delay_policy = None
        baseline_policy = None

        if baseline_checkpoint is not None:
            baseline_ppo = _load_baseline_ppo()
            baseline_action_spec = type("ActionSpecView", (), {"shape": (1, 3)})()
            baseline_policy = _BaselinePolicy(
                _load_policy(
                    baseline_ppo,
                    cfg,
                    _BaselineObservationSpec(env.observation_spec),
                    baseline_action_spec,
                    baseline_checkpoint,
                    None,
                )
            )
            print(f"[NavRL]: loaded baseline checkpoint: {baseline_checkpoint}")

        if delay_checkpoint is not None:
            delay_policy = _load_policy(
                DelayPPO,
                cfg,
                env.observation_spec,
                env.action_spec,
                delay_checkpoint,
                float(cfg.timing.reference_dt),
            )
            print(f"[NavRL]: loaded delay-aware checkpoint: {delay_checkpoint}")

        # Each rollout starts from the same environment seed and timing state.
        # Once trajectories terminate at different times, their later publisher
        # events may consume different command-delay samples.
        for name, policy in (("baseline", baseline_policy), ("delay", delay_policy)):
            if policy is None:
                continue
            metrics = evaluate(
                env=env,
                policy=policy,
                seed=int(cfg.seed),
                cfg=cfg,
                exploration_type=ExplorationType.MEAN,
            )
            print(f"[NavRL]: {name} evaluation")
            for key, value in metrics.items():
                if key != "recording":
                    print(f"{name}/{key}: {value}")
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
