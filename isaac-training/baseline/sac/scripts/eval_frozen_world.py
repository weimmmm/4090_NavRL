"""Evaluate one SAC checkpoint on eval_3407_2048.pt and print metrics."""
import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMNI_KIT_RENDERER", "Vulkan")
os.environ.setdefault("OMNI_KIT_NO_OPENGL_RENDERING", "1")

import numpy as np
import torch
from omni.isaac.kit import SimulationApp


ROOT = Path(__file__).resolve().parents[1]


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--world", required=True, type=Path)
    return parser.parse_args()


def metrics(trajs):
    done = trajs["next", "done"].to(torch.bool).squeeze(-1)
    first = torch.argmax(done.long(), dim=1)
    first = torch.where(done.any(dim=1), first, torch.full_like(first, done.shape[1] - 1))
    def terminal(value):
        # input is [num_envs, time, ...], so index must have identical rank.
        index = first.reshape(first.shape + (1,) * (value.ndim - 1))
        return torch.take_along_dim(value, index, dim=1).reshape(-1)
    result = {"episode_length": (first.float() + 1).mean().item()}
    for key, value in trajs["next", "stats"].items():
        result[f"stats.{key}"] = terminal(value.float()).mean().item()
    state = terminal(trajs["next", "agents", "observation", "state"].float()).squeeze(-2)
    result["goal_distance_mean"] = torch.sqrt(state[..., 3].square() + state[..., 4].square()).mean().item()
    return result


def main():
    args = parse_args()
    checkpoint, world = args.checkpoint.resolve(), args.world.resolve()
    if not checkpoint.is_file() or not world.is_file():
        raise FileNotFoundError(f"checkpoint={checkpoint}, world={world}")
    snap = torch.load(world, map_location="cpu")
    meta, cfg_data = snap["meta"], snap["meta"]["resolved_config"]
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg_data)
    cfg.env.num_envs = int(meta["num_envs"])
    cfg.env.max_episode_length = int(meta["max_episode_length"])
    cfg.env.num_obstacles = int(meta["num_obstacles_static"])
    cfg.env_dyn.num_obstacles = int(meta["num_obstacles_dynamic"])
    cfg.seed = int(meta["world_seed"])
    configure_seed(int(cfg.seed))

    sim_app = SimulationApp({"headless": True, "anti_aliasing": 1, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0})
    try:
        from evalenv import EvalEnv
        from sac import SAC
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        from torchrl.envs.transforms import Compose, TransformedEnv
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        print("[EVAL] creating frozen environment...", flush=True)
        env = EvalEnv(cfg, str(world))
        print("[EVAL] creating controller and policy...", flush=True)
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        env = TransformedEnv(env, Compose(VelController(controller, yaw_control=False))).eval()
        env.set_seed(int(cfg.seed))
        policy = SAC(cfg.algo, env.observation_spec, env.action_spec, cfg.device)
        policy.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
        policy.eval()
        print("[EVAL] rollout started...", flush=True)
        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            trajs = env.rollout(max_steps=env.max_episode_length, policy=policy, auto_reset=True, break_when_any_done=False, return_contiguous=False)
        print(f"[EVAL] checkpoint: {checkpoint}", flush=True)
        print(f"[EVAL] world: {world}", flush=True)
        for key, value in sorted(metrics(trajs).items()):
            print(f"  {key}: {value:.6f}", flush=True)
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()