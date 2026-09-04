"""Evaluate checkpoints with the live training_delay environment and timing model."""

import glob
import importlib.util
import os
import re
import sys
from datetime import datetime

OMNIDRONES_SOURCE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "OmniDrones")
)
TRAINING_DELAY_SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "training_delay", "scripts")
)
for source_path in (TRAINING_DELAY_SCRIPTS, OMNIDRONES_SOURCE):
    if source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from omni.isaac.kit import SimulationApp
from torchrl.envs.utils import ExplorationType


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
PROJECT_PATH = os.path.dirname(FILE_PATH)
TRAINING_DELAY_PATH = os.path.abspath(os.path.join(PROJECT_PATH, "..", "training_delay"))


def _resolve_project_path(path):
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_PATH, path))


def _latest_delay_checkpoint():
    pattern = os.path.join(TRAINING_DELAY_PATH, "wandb", "**", "files", "checkpoint_*.pt")
    candidates = [path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path)]
    if not candidates:
        raise FileNotFoundError(
            f"No training_delay checkpoint matched {pattern}. "
            "Set delay_checkpoint=/absolute/path/checkpoint_N.pt explicitly."
        )
    return max(candidates, key=lambda path: (os.path.getmtime(path), path))


def _resolve_checkpoint(path, name):
    if path is None:
        return None
    if name == "delay_checkpoint" and str(path).strip().lower() in {"latest", "auto"}:
        checkpoint = _latest_delay_checkpoint()
    else:
        checkpoint = to_absolute_path(str(path))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"{name} does not exist: {checkpoint}")
    return os.path.abspath(checkpoint)


def _checkpoint_step(path):
    match = re.search(r"checkpoint_(\d+)\.pt$", path)
    return int(match.group(1)) if match else None


def _load_baseline_ppo():
    """Load the original 8-D PPO without shadowing the live delay PPO."""
    module_path = os.path.abspath(
        os.path.join(PROJECT_PATH, "..", "training", "scripts", "ppo.py")
    )
    spec = importlib.util.spec_from_file_location("baseline_ppo_for_eval", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline PPO from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(module_path))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.PPO


class _BaselineObservationSpec:
    def __init__(self, spec):
        self.spec = spec

    def zero(self):
        tensordict = self.spec.zero()
        state_key = ("agents", "observation", "state")
        tensordict.set(state_key, tensordict.get(state_key)[..., :8].contiguous())
        return tensordict


class _BaselinePolicy:
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


def _load_policy(policy_cls, cfg, observation_spec, action_spec, checkpoint, timing_ref):
    args = (cfg.algo, observation_spec, action_spec, cfg.device)
    policy = policy_cls(*args) if timing_ref is None else policy_cls(*args, timing_ref)
    policy.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
    return policy


def _save_results(cfg, checkpoints, metrics, dataset):
    result_dir = _resolve_project_path(cfg.eval.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(result_dir, f"evaluation_{timestamp}.yaml")
    payload = {
        "implementation": "training_delay_live_50hz_overlapping_transport",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training_source": TRAINING_DELAY_PATH,
        "dataset_path": str(cfg.eval.dataset_path),
        "seed": int(cfg.seed),
        "environment": {
            "num_envs": int(cfg.env.num_envs),
            "num_static_obstacles": int(cfg.env.num_obstacles),
            "num_dynamic_obstacles": int(cfg.env_dyn.num_obstacles),
            "max_steps": int(cfg.eval.max_steps),
        },
        "scenarios": {
            "format_version": int(dataset["format_version"]),
            "sampling": str(dataset.get("sampling", "unspecified")),
            "seed": int(dataset.get("seed", cfg.seed)),
            "terrain_seed": int(dataset["terrain_seed"]),
            "mean_horizontal_distance": float(
                (
                    dataset["target_pos"][:, 0, :2]
                    - dataset["start_pos"][:, 0, :2]
                )
                .norm(dim=-1)
                .mean()
            ),
        },
        "timing": OmegaConf.to_container(cfg.timing, resolve=True),
        "checkpoints": {
            name: {"path": path, "step": _checkpoint_step(path)}
            for name, path in checkpoints.items()
            if path is not None
        },
        "metrics": metrics,
    }
    OmegaConf.save(OmegaConf.create(payload), result_path)
    return result_path


def _prefix_metrics(name, metrics):
    result = {}
    for key, value in metrics.items():
        if key == "recording":
            continue
        short_key = key.removeprefix("eval/")
        result[f"{name}/{short_key}"] = float(value)
    return result


@hydra.main(config_path=FILE_PATH, config_name="eval_random", version_base=None)
def main(cfg: DictConfig):
    with open_dict(cfg):
        cfg.device = f"cuda:{int(cfg.gpu_id)}"
        cfg.sim.device = cfg.device
        cfg.eval.dataset_path = _resolve_project_path(cfg.eval.dataset_path)
        cfg.eval.video_dir = _resolve_project_path(cfg.eval.video_dir)

    dataset = torch.load(cfg.eval.dataset_path, map_location="cpu")
    expected = {
        "num_envs": int(cfg.env.num_envs),
        "num_obstacles": int(cfg.env.num_obstacles),
        "num_dynamic_obstacles": int(cfg.env_dyn.num_obstacles),
    }
    for key, value in expected.items():
        if int(dataset[key]) != value:
            raise ValueError(
                f"Evaluation dataset {key}={dataset[key]}, but config requires {value}. "
                "Regenerate it with scripts/create_eval_env.py."
            )

    checkpoints = {
        "baseline": _resolve_checkpoint(cfg.get("baseline_checkpoint"), "baseline_checkpoint"),
        "delay": _resolve_checkpoint(cfg.get("delay_checkpoint"), "delay_checkpoint"),
    }
    if all(path is None for path in checkpoints.values()):
        raise ValueError("Set baseline_checkpoint and/or delay_checkpoint")

    sim_app = SimulationApp(
        {
            "headless": bool(cfg.headless),
            "anti_aliasing": 1,
            "active_gpu": int(cfg.gpu_id),
            "physics_gpu": int(cfg.gpu_id),
            "multi_gpu": False,
        }
    )
    evaluation_succeeded = False
    try:
        from env import NavigationEnv
        from ppo import PPO as DelayPPO
        from utils import evaluate

        # Dynamic obstacle placement happens during NavigationEnv construction
        # and uses NumPy. Seed both RNGs before construction so the full scene,
        # not only its fixed start/target tensors, is reproducible.
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))
        torch.cuda.manual_seed_all(int(cfg.seed))
        env = NavigationEnv(cfg).eval()
        policies = {}

        if checkpoints["baseline"] is not None:
            baseline_ppo = _load_baseline_ppo()
            baseline_action_spec = type("ActionSpecView", (), {"shape": (1, 3)})()
            policies["baseline"] = _BaselinePolicy(
                _load_policy(
                    baseline_ppo,
                    cfg,
                    _BaselineObservationSpec(env.observation_spec),
                    baseline_action_spec,
                    checkpoints["baseline"],
                    None,
                )
            )

        if checkpoints["delay"] is not None:
            policies["delay"] = _load_policy(
                DelayPPO,
                cfg,
                env.observation_spec,
                env.action_spec,
                checkpoints["delay"],
                float(cfg.timing.reference_dt),
            )

        metrics = {}
        for name, policy in policies.items():
            print(f"[NavRL]: evaluating {name}: {checkpoints[name]}", flush=True)
            policy_metrics = evaluate(
                env=env,
                policy=policy,
                seed=int(cfg.seed),
                cfg=cfg,
                exploration_type=ExplorationType.MEAN,
            )
            metrics.update(_prefix_metrics(name, policy_metrics))

        print("[NavRL]: evaluation results", flush=True)
        print(OmegaConf.to_yaml(OmegaConf.create(metrics), sort_keys=True))
        result_path = _save_results(cfg, checkpoints, metrics, dataset)
        print(f"[NavRL]: saved evaluation results to {result_path}", flush=True)
        evaluation_succeeded = True
    finally:
        # Isaac Sim 2023 can segfault in close() while unwinding a Python
        # exception, which hides the actionable traceback. The OS releases
        # simulator resources when the failed process exits.
        if evaluation_succeeded:
            sim_app.close()


if __name__ == "__main__":
    main()
