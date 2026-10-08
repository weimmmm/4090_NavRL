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
    parser.add_argument(
        "--max-steps", type=int, default=2500,
        help="Evaluation horizon. Overrides the value stored in the environment file.")
    parser.add_argument(
        "--goal-radius", type=float, default=0.5,
        help="Entering this radius starts zero-velocity braking.")
    parser.add_argument(
        "--settle-speed", type=float, default=0.1,
        help="Speed in m/s below which the drone is considered nearly stopped.")
    parser.add_argument(
        "--settle-steps", type=int, default=10,
        help="Consecutive low-speed simulation steps required for reach_goal.")
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
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.goal_radius <= 0:
        parser.error("--goal-radius must be positive")
    if args.settle_speed <= 0:
        parser.error("--settle-speed must be positive")
    if args.settle_steps <= 0:
        parser.error("--settle-steps must be positive")
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
    source_max_steps = int(fixed["max_steps"])
    source_count = int(fixed["num_envs"])
    if args.route_start or args.route_limit is not None:
        route_limit = (source_count-args.route_start
                       if args.route_limit is None else args.route_limit)
        if args.route_start+route_limit > source_count:
            raise ValueError("requested route slice exceeds environment route count")
        fixed = slice_environment(
            fixed, range(args.route_start, args.route_start+route_limit))
    # Keep the fixed geometry/routes, but allow evaluation to use a longer
    # horizon than the value embedded when the environment was created.
    fixed = dict(fixed)
    fixed["max_steps"] = int(args.max_steps)
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
        braking = torch.zeros(num_envs, device=device, dtype=torch.bool)
        settle_streak = torch.zeros(num_envs, device=device, dtype=torch.long)
        brake_attempts = torch.zeros(num_envs, device=device, dtype=torch.long)
        brake_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        brake_entry_position = torch.zeros(
            num_envs, 3, device=device, dtype=previous_position.dtype)
        brake_max_drift = torch.zeros(
            num_envs, device=device, dtype=previous_position.dtype)
        inference_seconds = []
        rows = []

        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            while len(rows) < episodes:
                tick = time.perf_counter()
                policy(td)
                # Enter a latched braking state inside the goal sphere.  PPO
                # remains disabled until the drone has actually settled.  If
                # it settles outside the sphere, release the latch and let PPO
                # approach again; merely passing through the goal is not a
                # successful episode.
                distance_before_step = (
                    env.target_pos[:, 0] - env.drone.pos[:, 0]).norm(dim=-1)
                start_braking = (~braking) & (
                    distance_before_step < args.goal_radius)
                if start_braking.any():
                    braking[start_braking] = True
                    brake_attempts[start_braking] += 1
                    brake_entry_position[start_braking] = env.drone.pos[
                        start_braking, 0]
                if braking.any():
                    td["agents", "action"][braking] = 0.0
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
                truncated = td["truncated"].reshape(-1).bool()
                out_of_bounds = (position[:, 2] < 0.2) | (position[:, 2] > 4.0)
                speed = env.drone.vel_w[:, 0, :3].norm(dim=-1)
                goal_distance = (env.target_pos[:, 0] - position).norm(dim=-1)
                if braking.any():
                    brake_steps[braking] += 1
                    drift = (position - brake_entry_position).norm(dim=-1)
                    brake_max_drift[braking] = torch.maximum(
                        brake_max_drift[braking], drift[braking])
                low_speed = braking & (speed < args.settle_speed)
                settle_streak = torch.where(
                    low_speed, settle_streak + 1,
                    torch.zeros_like(settle_streak))
                settled = braking & (settle_streak >= args.settle_steps)
                reach_goal = settled & (goal_distance < args.goal_radius)
                settled_outside = settled & ~reach_goal
                if settled_outside.any():
                    braking[settled_outside] = False
                    settle_streak[settled_outside] = 0
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
                        "final_goal_distance_m": float(goal_distance[index]),
                        "final_speed_mps": float(speed[index]),
                        "goal_stop_attempts": int(brake_attempts[index]),
                        "goal_stop_steps": int(brake_steps[index]),
                        "goal_stop_max_drift_m": float(brake_max_drift[index]),
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
                    braking[done] = False
                    settle_streak[done] = 0
                    brake_attempts[done] = 0
                    brake_steps[done] = 0
                    brake_entry_position[done] = 0
                    brake_max_drift[done] = 0

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
            "source_environment_max_steps": source_max_steps,
            "goal_radius_m": float(args.goal_radius),
            "goal_terminal_control": "zero_velocity_until_settled_inside_goal",
            "settle_speed_mps": float(args.settle_speed),
            "settle_consecutive_steps": int(args.settle_steps),
            "success_rate": len(successful) / max(len(rows), 1),
            "collision_rate": sum(r["termination_reason"] == "collision" for r in rows) / max(len(rows), 1),
            "out_of_bounds_rate": sum(r["termination_reason"] == "out_of_bounds" for r in rows) / max(len(rows), 1),
            "timeout_rate": sum(r["termination_reason"] == "timeout" for r in rows) / max(len(rows), 1),
            "mean_episode_steps": mean(rows, "steps"),
            "mean_path_length_m": mean(rows, "path_length_m"),
            "mean_min_clearance_m": mean(rows, "min_clearance_m"),
            "successful_mean_stop_drift_m": mean(
                successful, "goal_stop_max_drift_m"),
            "successful_max_stop_drift_m": max(
                (float(row["goal_stop_max_drift_m"]) for row in successful),
                default=0.0),
            "successful_mean_final_speed_mps": mean(
                successful, "final_speed_mps"),
            "successful_mean_final_goal_distance_m": mean(
                successful, "final_goal_distance_m"),
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
