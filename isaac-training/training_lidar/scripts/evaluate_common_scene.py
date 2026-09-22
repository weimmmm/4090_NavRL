"""Evaluate the baseline and range-image policies on the same scene layout."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omni.isaac.kit import SimulationApp


ROOT = Path(__file__).resolve().parents[2]
OMNIDRONES_SOURCE = ROOT / "third_party" / "OmniDrones"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "range_image"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--static-obstacles", type=int, default=350)
    parser.add_argument("--dynamic-obstacles", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_config(args):
    project = ROOT / ("training" if args.variant == "baseline" else "training_lidar")
    with initialize_config_dir(config_dir=str(project / "cfg"), version_base=None):
        cfg = compose(config_name="train")
    cfg.device = args.device
    cfg.headless = True
    cfg.seed = args.seed
    cfg.env.num_envs = args.num_envs
    cfg.env.num_obstacles = args.static_obstacles
    cfg.env_dyn.num_obstacles = args.dynamic_obstacles
    cfg.enable_eval = False
    cfg.record_eval_video = False
    if args.variant == "baseline":
        cfg.sensor.mesh_prim_path = "/World/ground"
    return cfg, project


def masked_mean(values, mask):
    selected = values[mask]
    return selected.float().mean().item() if selected.numel() else None


def main():
    args = parse_args()
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.static_obstacles < 0 or args.dynamic_obstacles < 0:
        raise ValueError("obstacle counts must be non-negative")
    cfg, project = load_config(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gpu_id = int(args.device.split(":")[-1])
    app = SimulationApp({
        "headless": True,
        "anti_aliasing": 1,
        "active_gpu": gpu_id,
        "physics_gpu": gpu_id,
        "multi_gpu": False,
    })
    env = None
    started = time.perf_counter()
    try:
        sys.path.insert(0, str(OMNIDRONES_SOURCE))
        sys.path.insert(0, str(project / "scripts"))
        from env import NavigationEnv
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        from ppo import PPO
        from torchrl.envs.transforms import Compose, TransformedEnv
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        env = NavigationEnv(cfg)
        env.eval()
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        transformed_env = TransformedEnv(
            env, Compose(VelController(controller, yaw_control=False))
        ).eval()
        transformed_env.set_seed(args.seed)
        if args.variant == "baseline":
            policy = PPO(cfg.algo, transformed_env.observation_spec,
                         transformed_env.action_spec, cfg.device)
        else:
            policy = PPO(cfg.algo, transformed_env.observation_spec,
                         transformed_env.action_spec, cfg.device,
                         lidar_range=cfg.sensor.lidar_range)
        policy.load_state_dict(torch.load(args.checkpoint, map_location=cfg.device))

        td = transformed_env.reset()
        previous_position = env.drone.pos.squeeze(1).clone()
        finished = torch.zeros(args.num_envs, dtype=torch.bool, device=cfg.device)
        success = torch.zeros_like(finished)
        collision = torch.zeros_like(finished)
        out_of_bounds = torch.zeros_like(finished)
        finish_step = torch.full((args.num_envs,), cfg.env.max_episode_length,
                                 dtype=torch.long, device=cfg.device)
        path_length = torch.zeros(args.num_envs, device=cfg.device)
        episode_return = torch.zeros(args.num_envs, device=cfg.device)

        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            for step in range(1, cfg.env.max_episode_length + 1):
                policy(td)
                transition = transformed_env.step(td)
                next_td = transition["next"]
                active = ~finished
                position = env.drone.pos.squeeze(1)
                path_length[active] += (position - previous_position)[active].norm(dim=-1)
                episode_return[active] += next_td["agents", "reward"].reshape(-1)[active]

                near_goal = (env.target_pos.squeeze(1) - position).norm(dim=-1) < 0.5
                hit = next_td["stats", "collision"].reshape(-1).bool()
                bounds = (position[:, 2] < 0.2) | (position[:, 2] > 4.0)
                newly_collided = active & hit
                newly_out = active & ~hit & bounds
                newly_successful = active & ~hit & ~bounds & near_goal
                newly_finished = newly_collided | newly_out | newly_successful
                collision |= newly_collided
                out_of_bounds |= newly_out
                success |= newly_successful
                finish_step[newly_finished] = step
                finished |= newly_finished
                previous_position = position.clone()
                td = next_td
                if finished.all():
                    break

        timeout = ~finished
        result = {
            "variant": args.variant,
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "max_episode_length": int(cfg.env.max_episode_length),
            "scene": {
                "terrain_seed": 0,
                "static_obstacles": int(cfg.env.num_obstacles),
                "dynamic_obstacles": int(cfg.env_dyn.num_obstacles),
                "mesh_prim_path": env.lidar.cfg.mesh_prim_paths[0],
            },
            "sensor": {
                "range": float(cfg.sensor.lidar_range),
                "vertical_beams": int(cfg.sensor.lidar_vbeams),
                "horizontal_resolution": float(cfg.sensor.lidar_hres),
            },
            "success_rate": success.float().mean().item(),
            "collision_rate": collision.float().mean().item(),
            "out_of_bounds_rate": out_of_bounds.float().mean().item(),
            "timeout_rate": timeout.float().mean().item(),
            "mean_success_steps": masked_mean(finish_step, success),
            "mean_success_path_length": masked_mean(path_length, success),
            "mean_return": episode_return.mean().item(),
            "completed_steps": step,
            "wall_time_seconds": time.perf_counter() - started,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print("COMMON_EVAL_RESULT=" + json.dumps(result), flush=True)
    finally:
        if env is not None:
            env.sim.stop()
        app.close()


if __name__ == "__main__":
    main()
