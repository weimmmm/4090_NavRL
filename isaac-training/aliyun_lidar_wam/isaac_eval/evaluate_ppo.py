"""Evaluate the deterministic PPO expert that collected ``wam_data``."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_eval.bootstrap import activate_vendored_sources

activate_vendored_sources()


def parse_args():
    default_checkpoint = (
        ROOT / "isaac_eval" / "checkpoints" / "ppo_dataset_collector.pt")
    default_environment = (
        ROOT / "isaac_eval" / "environments" / "static350_seed18_n256.pt")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--environment", type=Path, default=default_environment)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "isaac_eval" / "results" / "ppo_fixed_env")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int,
                        help="Defaults to one episode per fixed environment.")
    parser.add_argument("--route-start", type=int, default=0)
    parser.add_argument("--route-limit", type=int)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    for name in ("checkpoint", "environment"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_file():
            parser.error(f"{name} does not exist: {value}")
        setattr(args, name, value)
    args.output = args.output.expanduser().resolve()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.route_start < 0 or (args.route_limit is not None and args.route_limit <= 0):
        parser.error("route start/limit is invalid")
    return args


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    args = parse_args()
    from isaac_eval.environment_file import load_environment, slice_environment

    fixed = load_environment(args.environment)
    source_count = int(fixed["num_envs"])
    if args.route_start or args.route_limit is not None:
        route_limit = (source_count-args.route_start
                       if args.route_limit is None else args.route_limit)
        if args.route_start+route_limit > source_count:
            raise ValueError("requested route slice exceeds environment route count")
        fixed = slice_environment(
            fixed, range(args.route_start, args.route_start+route_limit))
    num_envs = int(fixed["num_envs"])
    episodes = args.episodes or num_envs
    terrain_seed = int(fixed["terrain_seed"])

    from omni.isaac.kit import SimulationApp

    gpu_id = int(str(args.device).split(":")[-1])
    app = SimulationApp({
        "headless": not args.render,
        "anti_aliasing": 1,
        "active_gpu": gpu_id,
        "physics_gpu": gpu_id,
        "multi_gpu": False,
    })
    print("[ppo_eval] SimulationApp ready", flush=True)
    env = None
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from tensordict import TensorDict
        from torchrl.envs.transforms import Compose, TransformedEnv
        from torchrl.envs.utils import ExplorationType, set_exploration_type
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController

        from isaac_eval.navigation_env import NavigationEnv
        from isaac_eval.ppo_policy import load_ppo_expert

        with initialize_config_dir(
                config_dir=str(ROOT / "isaac_eval" / "cfg"), version_base=None):
            cfg = compose(config_name="train")
        cfg.device = args.device
        cfg.headless = not args.render
        cfg.seed = terrain_seed
        cfg.env.num_envs = num_envs
        cfg.env.max_episode_length = int(fixed["max_steps"])
        cfg.env.num_obstacles = int(fixed["static_obstacles"])
        cfg.env.randomize_routes = False
        cfg.env_dyn.num_obstacles = 0
        cfg.enable_eval = False
        cfg.record_eval_video = False

        np.random.seed(terrain_seed)
        torch.manual_seed(terrain_seed)
        torch.cuda.manual_seed_all(terrain_seed)
        env = NavigationEnv(cfg, eval_environment=fixed)
        env.eval()
        env.enable_render(args.render)
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        transformed_env = TransformedEnv(
            env, Compose(VelController(controller, yaw_control=False))).eval()
        transformed_env.set_seed(terrain_seed)
        td = transformed_env.reset()
        policy = load_ppo_expert(
            args.checkpoint, cfg.algo, transformed_env.observation_spec,
            transformed_env.action_spec, cfg.device, cfg.sensor.lidar_range)
        print("[ppo_eval] fixed environment and collector PPO loaded", flush=True)

        device = torch.device(args.device)
        path_length = torch.zeros(num_envs, device=device)
        episode_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        min_clearance = torch.full(
            (num_envs,), float(env.lidar_range), device=device)
        previous_position = env.drone.pos[:, 0].clone()
        episode_number = torch.zeros(num_envs, device=device, dtype=torch.long)
        per_env, remainder = divmod(episodes, num_envs)
        quota = torch.full((num_envs,), per_env, device=device, dtype=torch.long)
        quota[:remainder] += 1
        collected = torch.zeros_like(quota)
        inference_seconds = []
        rows = []

        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            while len(rows) < episodes:
                tick = time.perf_counter()
                policy(td)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds.append(time.perf_counter() - tick)
                td = transformed_env.step(td)["next"]

                position = env.drone.pos[:, 0]
                path_length += (position - previous_position).norm(dim=-1)
                previous_position = position.clone()
                episode_steps += 1
                clearance = td[
                    "agents", "observation", "lidar"].reshape(num_envs, -1).amin(-1)
                min_clearance = torch.minimum(min_clearance, clearance)
                collision = td["stats", "collision"].reshape(-1).bool()
                reach_goal = td["stats", "reach_goal"].reshape(-1).bool()
                truncated = td["truncated"].reshape(-1).bool()
                out_of_bounds = (position[:, 2] < 0.2) | (position[:, 2] > 4.0)
                done = collision | reach_goal | truncated | out_of_bounds

                for index in done.nonzero().flatten().tolist():
                    if collected[index] >= quota[index]:
                        continue
                    if collision[index]:
                        reason = "collision"
                    elif out_of_bounds[index]:
                        reason = "out_of_bounds"
                    elif reach_goal[index]:
                        reason = "reach_goal"
                    else:
                        reason = "timeout"
                    rows.append({
                        "episode": len(rows),
                        "env_id": index,
                        "route_id": (int(fixed["source_env_ids"][index])
                                     if "source_env_ids" in fixed else index),
                        "env_episode": int(episode_number[index]),
                        "termination_reason": reason,
                        "steps": int(episode_steps[index]),
                        "duration_s": float(episode_steps[index]) * float(cfg.sim.dt),
                        "path_length_m": float(path_length[index]),
                        "min_clearance_m": float(min_clearance[index]),
                        "start_position": fixed["start_positions"][index, 0].tolist(),
                        "target_position": fixed["target_positions"][index, 0].tolist(),
                        "start_side": int(fixed["start_sides"][index]),
                        "target_side": int(fixed["target_sides"][index]),
                    })
                    collected[index] += 1
                if len(rows) >= episodes:
                    break
                if done.any():
                    reset_request = TensorDict(
                        {"_reset": done.unsqueeze(-1)}, batch_size=[num_envs],
                        device=device)
                    td = transformed_env.reset(reset_request)
                    episode_number[done] += 1
                    path_length[done] = 0
                    episode_steps[done] = 0
                    min_clearance[done] = float(env.lidar_range)
                    previous_position[done] = env.drone.pos[done, 0]

        successful = [row for row in rows if row["termination_reason"] == "reach_goal"]
        latency = torch.tensor(inference_seconds)
        summary = {
            "policy": "ppo_dataset_collector",
            "exploration_type": "mean",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "environment_file": str(args.environment),
            "environment_sha256": hashlib.sha256(args.environment.read_bytes()).hexdigest(),
            "terrain_seed": terrain_seed,
            "num_envs": num_envs,
            "route_start": args.route_start,
            "route_limit": args.route_limit,
            "source_env_ids": [int(v) for v in fixed.get(
                "source_env_ids", torch.arange(num_envs)).tolist()],
            "episodes": len(rows),
            "static_obstacles": int(fixed["static_obstacles"]),
            "dynamic_obstacles": 0,
            "max_steps": int(fixed["max_steps"]),
            "success_rate": len(successful) / max(len(rows), 1),
            "collision_rate": sum(r["termination_reason"] == "collision" for r in rows) / max(len(rows), 1),
            "out_of_bounds_rate": sum(r["termination_reason"] == "out_of_bounds" for r in rows) / max(len(rows), 1),
            "timeout_rate": sum(r["termination_reason"] == "timeout" for r in rows) / max(len(rows), 1),
            "mean_episode_steps": mean(rows, "steps"),
            "mean_path_length_m": mean(rows, "path_length_m"),
            "mean_min_clearance_m": mean(rows, "min_clearance_m"),
            "wall_time_s": time.perf_counter() - started,
            "inference_latency": {
                "calls": len(inference_seconds),
                "mean_s": float(latency.mean()),
                "p95_s": float(torch.quantile(latency, 0.95)),
                "max_s": float(latency.max()),
            },
        }
        output = args.output / (
            f"ppo_seed{terrain_seed}_n{episodes}_routes{args.route_start:03d}.json")
        save_json(output, {"summary": summary, "episode_results": rows})
        print(json.dumps({"output": str(output), **summary}, indent=2), flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
