"""Evaluate a delay-aware policy in the current two-stage delay environment."""

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


def _checkpoint_path(cfg: DictConfig) -> str:
    checkpoint = cfg.get("delay_checkpoint")
    if checkpoint is None:
        checkpoint = cfg.get("checkpoint_path")
    if checkpoint is None:
        raise ValueError(
            "Set delay_checkpoint=/path/to/checkpoint.pt (or checkpoint_path=...)"
        )
    checkpoint = to_absolute_path(str(checkpoint))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Policy checkpoint does not exist: {checkpoint}")
    return checkpoint


@hydra.main(config_path=FILE_PATH, config_name="eval_random", version_base=None)
def main(cfg: DictConfig):
    # The environment owns the velocity controller and applies cmd_vel after
    # the simulated two-stage delay. Do not wrap it in another VelController.
    from env import NavigationEnv
    from ppo import PPO
    from utils import evaluate

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
        env.set_seed(cfg.seed)

        policy = PPO(
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            cfg.device,
            cfg.timing.reference_dt,
        )
        checkpoint = _checkpoint_path(cfg)
        try:
            policy.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
        except RuntimeError as exc:
            raise RuntimeError(
                "The checkpoint is incompatible with the delay-aware policy. "
                "A delay-aware checkpoint must contain the 10-D state input."
            ) from exc
        print(f"[NavRL]: loaded delay-aware checkpoint: {checkpoint}")

        eval_info = evaluate(
            env=env,
            policy=policy,
            seed=cfg.seed,
            cfg=cfg,
            exploration_type=ExplorationType.MEAN,
        )
        for key, value in eval_info.items():
            if key != "recording":
                print(f"{key}: {value}")
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
