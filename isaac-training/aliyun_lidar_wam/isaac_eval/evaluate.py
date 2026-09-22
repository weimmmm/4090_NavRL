"""Run closed-loop Action Expert navigation evaluation in Isaac Sim 2023."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    # Load the original joint training checkpoint by default.  The evaluator
    # selects the Action Expert tensors in memory and does not require a prior
    # compact-checkpoint export step.
    default_checkpoint = ROOT / "lidar_WAM" / "outputs" / "action_expert_joint" / "best.pt"
    default_environment = (
        ROOT / "isaac_eval" / "environments" / "static350_seed18_n256.pt")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=default_checkpoint)
    parser.add_argument("--output", type=Path, default=ROOT / "isaac_eval" / "results")
    parser.add_argument(
        "--environment", type=Path,
        default=default_environment if default_environment.exists() else None,
        help="Fixed .pt evaluation environment (routes and terrain recipe).")
    parser.add_argument(
        "--random-environment", action="store_true",
        help="Ignore the default .pt environment and sample routes at reset.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int,
                        help="Terrain seed; must match --environment when one is used.")
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int,
                        help="Parallel environments; must match the .pt environment.")
    parser.add_argument("--episodes", type=int,
                        help="Defaults to one episode per environment.")
    parser.add_argument("--max-steps", type=int,
                        help="Must match --environment when one is used.")
    parser.add_argument("--static-obstacles", type=int,
                        help="Must match --environment when one is used.")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument(
        "--allow-legacy-body", action="store_true",
        help="Allow a legacy checkpoint with implicit body-frame conditions.")
    parser.add_argument(
        "--route-limit", type=int,
        help="Use N routes from a fixed .pt without creating a new file.")
    parser.add_argument(
        "--route-start", type=int, default=0,
        help="First source route used with --route-limit (default: 0).")
    parser.add_argument("--render", action="store_true",
                        help="Open the Isaac viewer instead of running headless.")
    args = parser.parse_args()
    if args.random_environment:
        args.environment = None
    if args.flow_steps <= 0:
        parser.error("--flow-steps must be positive")
    if args.route_limit is not None and args.route_limit <= 0:
        parser.error("--route-limit must be positive")
    if args.route_start < 0:
        parser.error("--route-start must be nonnegative")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.environment is not None:
        args.environment = args.environment.expanduser().resolve()
        if not args.environment.is_file():
            parser.error(f"environment does not exist: {args.environment}")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return parser, args


def resolve_environment_args(parser, args, environment):
    defaults = {
        "seed": 18,
        "num_envs": 16,
        "max_steps": 2200,
        "static_obstacles": 350,
    }
    environment_fields = {
        "seed": "terrain_seed",
        "num_envs": "num_envs",
        "max_steps": "max_steps",
        "static_obstacles": "static_obstacles",
    }
    for argument, field in environment_fields.items():
        requested = getattr(args, argument)
        expected = int(environment[field]) if environment is not None else defaults[argument]
        if environment is not None and requested is not None and requested != expected:
            parser.error(
                f"--{argument.replace('_', '-')}={requested} conflicts with "
                f"{args.environment.name}, which requires {expected}")
        setattr(args, argument, expected if requested is None else requested)
    if args.episodes is None:
        args.episodes = args.num_envs
    if args.num_envs <= 0 or args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--num-envs, --episodes, and --max-steps must be positive")


def load_config(args):
    from hydra import compose, initialize_config_dir

    config_dir = ROOT / "isaac_eval" / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train")
    cfg.device = args.device
    cfg.headless = not args.render
    cfg.seed = args.seed
    cfg.env.num_envs = args.num_envs
    cfg.env.max_episode_length = args.max_steps
    cfg.env.num_obstacles = args.static_obstacles
    cfg.env.randomize_routes = args.environment is None
    cfg.env_dyn.num_obstacles = 0
    cfg.enable_eval = False
    cfg.record_eval_video = False
    return cfg


def summarize(episodes, latency, args, checkpoint_step, elapsed):
    count = len(episodes)
    successes = sum(row["termination_reason"] == "reach_goal" for row in episodes)
    collisions = sum(row["termination_reason"] == "collision" for row in episodes)
    out_of_bounds = sum(row["termination_reason"] == "out_of_bounds" for row in episodes)
    timeouts = sum(row["termination_reason"] == "timeout" for row in episodes)

    def mean(key):
        return sum(float(row[key]) for row in episodes) / max(count, 1)

    successful = [row for row in episodes if row["termination_reason"] == "reach_goal"]
    return {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "environment_file": str(args.environment) if args.environment else None,
        "environment_sha256": args.environment_sha256,
        "terrain_seed": args.seed,
        "policy_seed": args.policy_seed,
        "num_envs": args.num_envs,
        "episodes": count,
        "static_obstacles": args.static_obstacles,
        "dynamic_obstacles": 0,
        "max_steps": args.max_steps,
        "replan_interval_steps": 10,
        "generated_horizon_steps": 30,
        "executed_head": getattr(args, "executed_head", "flow30"),
        "flow_steps": args.flow_steps,
        "condition_frame": args.condition_frame,
        "checkpoint_semantics": args.checkpoint_semantics,
        "route_limit": args.route_limit,
        "route_start": args.route_start,
        "source_env_ids": args.source_env_ids,
        "success_rate": successes / max(count, 1),
        "collision_rate": collisions / max(count, 1),
        "out_of_bounds_rate": out_of_bounds / max(count, 1),
        "timeout_rate": timeouts / max(count, 1),
        "mean_episode_steps": mean("steps"),
        "mean_path_length_m": mean("path_length_m"),
        "mean_min_clearance_m": mean("min_clearance_m"),
        "successful_mean_steps": (
            sum(row["steps"] for row in successful) / len(successful)
            if successful else None),
        "successful_mean_path_length_m": (
            sum(row["path_length_m"] for row in successful) / len(successful)
            if successful else None),
        "wall_time_s": elapsed,
        "inference_latency": latency,
    }


def main():
    parser, args = parse_args()
    from isaac_eval.environment_file import load_environment, slice_environment

    environment = load_environment(args.environment) if args.environment else None
    args.environment_sha256 = (
        hashlib.sha256(args.environment.read_bytes()).hexdigest()
        if args.environment else None)
    source_count = int(environment["num_envs"]) if environment is not None else None
    if args.route_limit is not None or args.route_start:
        if environment is None:
            parser.error("route slicing requires --environment")
        route_limit = (source_count-args.route_start
                       if args.route_limit is None else args.route_limit)
        if args.route_start+route_limit > source_count:
            parser.error("requested route slice exceeds the environment route count")
        environment = slice_environment(
            environment, range(args.route_start, args.route_start+route_limit))
    args.source_env_ids = (
        [int(v) for v in (environment["source_env_ids"].tolist()
                          if "source_env_ids" in environment
                          else range(int(environment["num_envs"])))]
        if environment is not None else None)
    resolve_environment_args(parser, args, environment)
    cfg = load_config(args)

    from omni.isaac.kit import SimulationApp

    gpu_id = int(str(args.device).split(":")[-1])
    simulation_app = SimulationApp({
        "headless": not args.render,
        "anti_aliasing": 1,
        "active_gpu": gpu_id,
        "physics_gpu": gpu_id,
        "multi_gpu": False,
    })
    print("[isaac_eval] SimulationApp ready", flush=True)
    env = None
    completed = False
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        print("[isaac_eval] NumPy/PyTorch imports ready", flush=True)
        from tensordict import TensorDict
        from torchrl.envs.transforms import Compose, TransformedEnv
        print("[isaac_eval] TensorDict/TorchRL imports ready", flush=True)
        from omni_drones.controllers import LeePositionController
        print("[isaac_eval] OmniDrones controller import ready", flush=True)
        from omni_drones.utils.torchrl.transforms import VelController
        print("[isaac_eval] Isaac/OmniDrones imports ready", flush=True)

        from isaac_eval.navigation_env import NavigationEnv
        from isaac_eval.policy import RecedingHorizonPolicy, lidar_range_image, save_json
        print("[isaac_eval] local evaluator imports ready", flush=True)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        env = NavigationEnv(cfg, eval_environment=environment)
        print("[isaac_eval] NavigationEnv constructed", flush=True)
        env.eval()
        env.enable_render(args.render)
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        transformed_env = TransformedEnv(
            env, Compose(VelController(controller, yaw_control=False))).eval()
        transformed_env.set_seed(args.seed)
        td = transformed_env.reset()

        device = torch.device(args.device)
        policy = RecedingHorizonPolicy(
            args.checkpoint, args.num_envs, device,
            flow_steps=args.flow_steps, seed=args.policy_seed,
            action_limit=float(cfg.algo.actor.action_limit),
            allow_legacy_body=args.allow_legacy_body,
            route_ids=(environment.get("source_env_ids")
                       if environment is not None else None))
        args.condition_frame = policy.condition_frame
        args.executed_head = policy.executed_head
        args.checkpoint_semantics = policy.semantics
        path_length = torch.zeros(args.num_envs, device=device)
        episode_steps = torch.zeros(args.num_envs, device=device, dtype=torch.long)
        min_clearance = torch.full((args.num_envs,), float(env.lidar_range), device=device)
        previous_position = env.drone.pos[:, 0].clone()
        episode_number = torch.zeros(args.num_envs, device=device, dtype=torch.long)
        # Split the requested episodes across environments.  Without a per-env
        # quota, environments that collide early can reset and be counted more
        # than once while slower environments are still on their first episode,
        # which biases aggregate results toward short failures.
        episodes_per_env, remainder = divmod(args.episodes, args.num_envs)
        episode_quota = torch.full(
            (args.num_envs,), episodes_per_env, device=device, dtype=torch.long)
        episode_quota[:remainder] += 1
        collected_per_env = torch.zeros_like(episode_quota)
        rows = []

        with torch.no_grad():
            while len(rows) < args.episodes:
                policy.act(td, env)
                transition = transformed_env.step(td)
                td = transition["next"]

                position = env.drone.pos[:, 0]
                path_length += (position - previous_position).norm(dim=-1)
                previous_position = position.clone()
                episode_steps += 1
                image = lidar_range_image(env)
                live_range = (image[:, 0, :, :18] + 1.0) * (float(env.lidar_range) / 2.0)
                valid = image[:, 1, :, :18] > 0
                clearance = torch.where(
                    valid, live_range, torch.full_like(live_range, env.lidar_range)
                ).amin(dim=(1, 2))
                min_clearance = torch.minimum(min_clearance, clearance)

                collision = td["stats", "collision"].reshape(-1).bool()
                reach_goal = td["stats", "reach_goal"].reshape(-1).bool()
                truncated = td["truncated"].reshape(-1).bool()
                out_of_bounds = (position[:, 2] < 0.2) | (position[:, 2] > 4.0)
                done = collision | reach_goal | truncated | out_of_bounds
                done_indices = done.nonzero().flatten().tolist()
                for index in done_indices:
                    if len(rows) >= args.episodes:
                        break
                    if collected_per_env[index] >= episode_quota[index]:
                        continue
                    if collision[index]:
                        reason = "collision"
                    elif out_of_bounds[index]:
                        reason = "out_of_bounds"
                    elif reach_goal[index]:
                        reason = "reach_goal"
                    else:
                        reason = "timeout"
                    row = {
                        "episode": len(rows),
                        "env_id": index,
                        "route_id": (int(environment["source_env_ids"][index])
                                     if environment is not None
                                     and "source_env_ids" in environment else index),
                        "env_episode": int(episode_number[index]),
                        "termination_reason": reason,
                        "steps": int(episode_steps[index]),
                        "duration_s": float(episode_steps[index]) * float(cfg.sim.dt),
                        "path_length_m": float(path_length[index]),
                        "min_clearance_m": float(min_clearance[index]),
                    }
                    if environment is not None:
                        row["start_position"] = environment[
                            "start_positions"][index, 0].tolist()
                        row["target_position"] = environment[
                            "target_positions"][index, 0].tolist()
                        row["start_side"] = int(environment["start_sides"][index])
                        row["target_side"] = int(environment["target_sides"][index])
                    rows.append(row)
                    collected_per_env[index] += 1
                if len(rows) >= args.episodes:
                    break

                if done_indices:
                    reset_request = TensorDict(
                        {"_reset": done.unsqueeze(-1)}, batch_size=[args.num_envs],
                        device=device)
                    td = transformed_env.reset(reset_request)
                    policy.reset(done)
                    episode_number[done] += 1
                    path_length[done] = 0
                    episode_steps[done] = 0
                    min_clearance[done] = float(env.lidar_range)
                    previous_position[done] = env.drone.pos[done, 0]

                if len(rows) and len(rows) % max(args.num_envs, 10) == 0:
                    print(json.dumps({"completed_episodes": len(rows),
                                      "target_episodes": args.episodes}), flush=True)

        elapsed = time.perf_counter() - started
        latency = policy.latency_summary()
        summary = summarize(rows, latency, args, policy.checkpoint_step, elapsed)
        result = {"summary": summary, "episode_results": rows}
        checkpoint_name = args.checkpoint.stem.replace(" ", "_")
        output = args.output / (
            f"action_expert_{checkpoint_name}_seed{args.seed}_n{args.episodes}"
            f"_routes{args.route_start:03d}"
            f"_policy{args.policy_seed}.json")
        save_json(output, result)
        print(json.dumps({"output": str(output), **summary}, indent=2), flush=True)
        completed = True
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        if completed:
            # Isaac Sim 2023 can hang or segfault while unloading plugins
            # after a successful headless evaluation.  The JSON is already
            # atomically written and the environment has released its state.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        simulation_app.close()


if __name__ == "__main__":
    main()
