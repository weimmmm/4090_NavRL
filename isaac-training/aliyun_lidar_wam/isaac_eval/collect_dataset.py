"""Collect full-horizon PPO episodes into an accepted-only HDF5 shard.

Entering the goal radius starts a zero-command braking phase.  Reach-goal is
recorded only after the vehicle settles inside the radius. Collision and
out-of-bounds episodes are removed in their entirety before publication.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_eval.bootstrap import activate_vendored_sources

activate_vendored_sources()


def split_for_seed(seed: int) -> str:
    if 0 <= seed <= 7:
        return "train"
    if seed == 8:
        return "val"
    if seed == 9:
        return "test"
    return "smoke"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--route-seed", type=int)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--static-obstacles", type=int, default=350)
    parser.add_argument("--opposite-fraction", type=float, default=0.5)
    parser.add_argument("--goal-radius", type=float, default=0.5)
    parser.add_argument("--settle-speed", type=float, default=0.1)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--keep-environment", action="store_true",
                        help="Keep the accepted .pt snapshot (normally only shard 0 per condition).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("auto", "train", "val", "test", "smoke"), default="auto")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "datasets" / "wam_40x40_obs200_350_settled_4m")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "isaac_eval" / "checkpoints" / "ppo_dataset_collector.pt")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--compression", choices=("lzf", "gzip"), default="lzf")
    args = parser.parse_args()
    args.route_seed = args.seed if args.route_seed is None else args.route_seed
    args.split = split_for_seed(args.seed) if args.split == "auto" else args.split
    args.output_root = args.output_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.num_envs <= 0 or args.sample_interval <= 0 or args.sample_interval != 10:
        parser.error("--num-envs must be positive and --sample-interval must be 10")
    if args.max_steps <= 0 or args.static_obstacles not in (200, 350):
        parser.error("this protocol requires positive max steps and 200 or 350 obstacles")
    if args.goal_radius <= 0 or args.settle_speed <= 0 or args.settle_steps <= 0:
        parser.error("goal/braking thresholds must be positive")
    if not 0.0 <= args.opposite_fraction <= 1.0:
        parser.error("--opposite-fraction must be in [0,1]")
    if args.shard_id is not None and args.shard_id < 0:
        parser.error("--shard-id must be non-negative")
    return args


def _save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _stage_mesh():
    """Read back the actual USD points/faces used by physics and RayCaster."""
    import numpy as np
    import omni.usd
    from pxr import UsdGeom

    path = "/World/ground/terrain/mesh"
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"terrain mesh prim does not exist: {path}")
    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    if not len(counts) or not np.all(counts == 3):
        raise RuntimeError("terrain mesh is not triangulated")
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    return vertices, faces


def _lidar_config(cfg):
    horizontal = int(float(cfg.sensor.lidar_hfov) / float(cfg.sensor.lidar_hres))
    return {
        "range_m": float(cfg.sensor.lidar_range),
        "horizontal_fov_deg": float(cfg.sensor.lidar_hfov),
        "vertical_fov_deg": [float(x) for x in cfg.sensor.lidar_vfov],
        "policy_vertical_beams": int(cfg.sensor.lidar_vbeams),
        "policy_horizontal_resolution_deg": float(cfg.sensor.lidar_hres),
        "horizontal_sample_factor": int(cfg.sensor.lidar_h_sample),
        "vertical_sample_factor": int(cfg.sensor.lidar_v_sample),
        "raw_resolution": [horizontal * int(cfg.sensor.lidar_h_sample),
                           int(cfg.sensor.lidar_vbeams) * int(cfg.sensor.lidar_v_sample)],
        "policy_resolution": [int(cfg.sensor.lidar_vbeams), horizontal],
        "attach_yaw_only": True,
    }


def _frame_rows(*, indices, seed, sim_step, frame_index, td, env,
                range_values, target_dir, norm_actions, world_actions,
                action_masks, step_deltas, terminal_reasons):
    import numpy as np
    import torch

    ids = indices.detach().cpu().tolist()
    drone = td["info", "drone_state"].reshape(env.num_envs, 13)[indices]
    state = td["agents", "observation", "state"].reshape(env.num_envs, 8)[indices]
    policy_ranges = td["agents", "observation", "lidar"][indices]
    targets = env.target_pos[indices, 0]
    distance = (targets - drone[:, :3]).norm(dim=-1)
    clearance = policy_ranges.reshape(len(ids), -1).amin(dim=-1)
    rows = []
    for local, env_id in enumerate(ids):
        reason = terminal_reasons.get(env_id, "")
        token = f"s{seed:04d}-e{env_id:04d}-f{int(frame_index[env_id]):06d}"
        flags = {name: int(reason == name) for name in
                 ("collision", "reach_goal", "out_of_bounds", "timeout")}
        rows.append({
            "range_values": range_values[indices[local]].detach().cpu().numpy(),
            "policy_ranges": policy_ranges[local].detach().cpu().numpy(),
            "drone_state": drone[local].detach().cpu().numpy(),
            "policy_state": state[local].detach().cpu().numpy(),
            "target_position": targets[local].detach().cpu().numpy(),
            "target_dir_2d": target_dir[indices[local]].detach().cpu().numpy(),
            "normalized_action_sequence": norm_actions[indices[local]].detach().cpu().numpy(),
            "world_action_sequence": world_actions[indices[local]].detach().cpu().numpy(),
            "action_mask": action_masks[indices[local]].detach().cpu().numpy().astype(np.uint8),
            "scene_id": env_id, "env_id": env_id,
            "frame_index": int(frame_index[env_id]), "sim_step": int(sim_step[env_id]),
            "timestamp_us": int(round(float(sim_step[env_id]) * float(env.dt) * 1_000_000)),
            "step_delta": int(step_deltas[indices[local]]), **flags,
            "min_lidar_clearance": float(clearance[local]),
            "distance_to_goal": float(distance[local]),
            "token": token, "scene_token": f"s{seed:04d}-e{env_id:04d}",
            "termination_reason": reason,
        })
    return rows


def main():
    args = parse_args()
    # Restrict each Isaac process before CUDA is initialized. SimulationApp also
    # receives the logical active GPU below.
    physical_gpu = int(args.device.split(":")[-1])
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(physical_gpu))

    from omni.isaac.kit import SimulationApp
    app = SimulationApp({
        "headless": not args.render, "anti_aliasing": 1, "active_gpu": 0,
        "physics_gpu": 0, "multi_gpu": False,
    })
    env = None
    writer = None
    completed = False
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from torchrl.envs.transforms import Compose, TransformedEnv
        from torchrl.envs.utils import ExplorationType, set_exploration_type
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torch import euler_to_quaternion
        from omni_drones.utils.torchrl.transforms import VelController
        from omni.isaac.orbit.terrains import (
            HfDiscreteObstaclesTerrainCfg, TerrainGenerator, TerrainGeneratorCfg)

        from isaac_eval.dataset_io import TrajectoryWriter, validate_dataset
        from isaac_eval.environment_file import (
            file_sha256, generate_environment, load_environment,
            make_environment_v2, mesh_sha256, save_environment,
            slice_environment)
        from isaac_eval.navigation_env import NavigationEnv
        from isaac_eval.policy import lidar_range_image
        from isaac_eval.ppo_policy import load_ppo_expert

        shard_id = args.seed if args.shard_id is None else args.shard_id
        legacy_layout = args.shard_id is None
        keep_environment = args.keep_environment or legacy_layout
        if legacy_layout:
            # Preserve compatibility with the original seed00-09 collector.
            seed_dir = args.output_root / args.split / f"seed_{args.seed:04d}"
            environment_path = (args.output_root / "environments" /
                                f"static{args.static_obstacles}_seed{args.seed:02d}_"
                                f"n{args.num_envs}.pt")
            retained_environment_path = environment_path
        else:
            seed_dir = (args.output_root / args.split /
                        f"obs_{args.static_obstacles:04d}" / f"shard_{shard_id:05d}")
            # A temporary exact snapshot is required for simulation and
            # validation. Production keeps only shard 0 per condition.
            environment_path = seed_dir / "environment.partial.pt"
            retained_environment_path = (args.output_root / "environments" /
                                         f"map40_obs{args.static_obstacles:04d}.pt")
        partial_path = seed_dir / "trajectories.partial.h5"
        dataset_path = seed_dir / "trajectories.h5"
        summary_path = seed_dir / "summary.json"
        checked_paths = [environment_path, partial_path, dataset_path, summary_path]
        if keep_environment:
            checked_paths.append(retained_environment_path)
        conflicts = [path for path in checked_paths
                     if path.exists()]
        if conflicts:
            raise FileExistsError("refusing to overwrite: " + ", ".join(map(str, conflicts)))
        seed_dir.mkdir(parents=True, exist_ok=True)

        with initialize_config_dir(config_dir=str(ROOT / "isaac_eval" / "cfg"), version_base=None):
            cfg = compose(config_name="train")
        cfg.device = "cuda:0"
        cfg.headless = not args.render
        cfg.seed = args.seed
        cfg.env.num_envs = args.num_envs
        cfg.env.max_episode_length = args.max_steps
        cfg.env.num_obstacles = args.static_obstacles
        cfg.env.randomize_routes = False
        cfg.env_dyn.num_obstacles = 0
        cfg.enable_eval = False
        cfg.record_eval_video = False

        # Generate the complete snapshot first.  The live environment below is
        # then constructed from this exact mesh and route set, never from a
        # second random terrain invocation.
        routes = generate_environment(
            num_envs=args.num_envs, terrain_seed=args.seed,
            route_seed=args.route_seed, static_obstacles=args.static_obstacles,
            max_steps=args.max_steps,
            opposite_fraction=args.opposite_fraction)
        terrain_generator = TerrainGenerator(TerrainGeneratorCfg(
            seed=args.seed, size=(40.0, 40.0), border_width=5.0,
            num_rows=1, num_cols=1, horizontal_scale=0.1,
            vertical_scale=0.1, slope_threshold=0.75, use_cache=False,
            color_scheme="none",
            sub_terrains={"obstacles": HfDiscreteObstaclesTerrainCfg(
                horizontal_scale=0.1, vertical_scale=0.1, border_width=0.0,
                num_obstacles=args.static_obstacles, obstacle_height_mode="range",
                obstacle_width_range=(0.4, 1.1),
                obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                platform_width=0.0)},
        ), device="cpu")
        generated_mesh = terrain_generator.terrain_mesh
        target_dir_2d = routes["target_positions"] - routes["start_positions"]
        target_dir_2d[..., 2] = 0
        rpy = torch.zeros(args.num_envs, 1, 3)
        route_delta = routes["target_positions"] - routes["start_positions"]
        rpy[..., 2] = torch.atan2(route_delta[..., 1], route_delta[..., 0])
        snapshot = make_environment_v2(
            vertices=np.asarray(generated_mesh.vertices, dtype=np.float32),
            faces=np.asarray(generated_mesh.faces, dtype=np.int32),
            start_positions=routes["start_positions"],
            target_positions=routes["target_positions"],
            start_sides=routes["start_sides"], target_sides=routes["target_sides"],
            initial_quaternions=euler_to_quaternion(rpy),
            initial_velocities=torch.zeros(args.num_envs, 1, 6),
            target_dir_2d=target_dir_2d, terrain_seed=args.seed,
            route_seed=args.route_seed, static_obstacles=args.static_obstacles,
            max_steps=args.max_steps, sim_dt=float(cfg.sim.dt),
            lidar_config=_lidar_config(cfg), checkpoint_path=args.checkpoint)
        save_environment(environment_path, snapshot)
        loaded_a, loaded_b = load_environment(environment_path), load_environment(environment_path)
        for key in ("start_positions", "target_positions", "initial_quaternions",
                    "initial_velocities", "target_dir_2d"):
            if not torch.equal(loaded_a[key], loaded_b[key]):
                raise RuntimeError(f"environment reload differs for {key}")

        np.random.seed(args.route_seed)
        torch.manual_seed(args.route_seed)
        torch.cuda.manual_seed_all(args.route_seed)
        env = NavigationEnv(cfg, eval_environment=loaded_a)
        env.eval()
        env.enable_render(args.render)
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        transformed = TransformedEnv(
            env, Compose(VelController(controller, yaw_control=False))).eval()
        transformed.set_seed(args.route_seed)
        td = transformed.reset()

        vertices, faces = _stage_mesh()
        live_hash = mesh_sha256(vertices, faces)
        initial_state = td["info", "drone_state"].detach().cpu()
        if snapshot["terrain_mesh"]["sha256"] != live_hash:
            raise RuntimeError("snapshot mesh hash differs from the live Isaac Stage")
        if not torch.equal(initial_state[..., :3], snapshot["start_positions"]):
            raise RuntimeError("live starts differ from the saved environment")
        if not torch.equal(env.target_pos.detach().cpu(), snapshot["target_positions"]):
            raise RuntimeError("live targets differ from the saved environment")

        policy = load_ppo_expert(
            args.checkpoint, cfg.algo, transformed.observation_spec,
            transformed.action_spec, cfg.device, cfg.sensor.lidar_range)
        num_envs, horizon = args.num_envs, args.sample_interval
        device = torch.device(cfg.device)
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        sim_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        since_frame = torch.zeros_like(sim_steps)
        frame_index = torch.zeros_like(sim_steps)
        norm_buffer = torch.zeros(num_envs, horizon, 3, device=device)
        world_buffer = torch.zeros_like(norm_buffer)
        action_mask = torch.zeros(num_envs, horizon, dtype=torch.uint8, device=device)
        path_length = torch.zeros(num_envs, device=device)
        min_clearance = torch.full((num_envs,), float(env.lidar_range), device=device)
        previous_position = initial_state[:, 0, :3].to(device)
        terminal_reasons: dict[int, str] = {}
        braking = torch.zeros(num_envs, dtype=torch.bool, device=device)
        settle_streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        brake_attempts = torch.zeros(num_envs, dtype=torch.long, device=device)

        metadata = {
            "format": "navrl-isaac-trajectory-hdf5-v2", "terrain_seed": args.seed,
            "route_seed": args.route_seed, "num_envs": num_envs,
            "sample_interval": horizon, "max_steps": args.max_steps,
            "sim_dt": float(cfg.sim.dt), "static_obstacles": args.static_obstacles,
            "map_size_m": [40.0, 40.0], "shard_id": shard_id,
            "goal_radius_m": args.goal_radius,
            "settle_speed_mps": args.settle_speed,
            "settle_steps": args.settle_steps,
            "opposite_fraction": args.opposite_fraction,
            "route_sampling": routes["route_sampling"],
            "discarded_terminations": ["collision", "out_of_bounds"],
            "dynamic_obstacles": 0, "policy": "ppo_dataset_collector",
            "exploration_type": "mean", "checkpoint_sha256": file_sha256(args.checkpoint),
            "environment_file": str(environment_path.relative_to(args.output_root)),
            "environment_sha256": file_sha256(environment_path),
            "terrain_mesh_sha256": live_hash,
        }
        writer = TrajectoryWriter(partial_path, metadata, compression=args.compression)

        # Initial physical frame: no preceding action interval.
        initial_ids = torch.arange(num_envs, device=device)
        raw = lidar_range_image(env)
        initial_rows = _frame_rows(
            indices=initial_ids, seed=args.seed, sim_step=sim_steps,
            frame_index=frame_index, td=td, env=env, range_values=raw,
            target_dir=snapshot["target_dir_2d"][:, 0].to(device), norm_actions=norm_buffer,
            world_actions=world_buffer, action_masks=action_mask,
            step_deltas=since_frame, terminal_reasons={})
        writer.append(initial_rows)
        frame_index += 1

        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            while active.any():
                policy(td)
                normalized = td["agents", "action_normalized"].reshape(num_envs, 3)
                world = td["agents", "action"].reshape(num_envs, 3)
                position_before = env.drone.pos[:, 0]
                distance_before = (env.target_pos[:, 0] - position_before).norm(dim=-1)
                start_braking = active & (~braking) & (distance_before < args.goal_radius)
                braking[start_braking] = True
                brake_attempts[start_braking] += 1
                # Zero world velocity is the actual command sent to the
                # controller. In normalized PPO coordinates zero is 0.5.
                if braking.any():
                    normalized[braking] = 0.5
                    world[braking] = 0.0
                    td["agents", "action_normalized"].reshape(num_envs, 3)[braking] = 0.5
                    td["agents", "action"].reshape(num_envs, 3)[braking] = 0.0
                slots = since_frame.clamp_max(horizon - 1)
                active_ids = active.nonzero().flatten()
                norm_buffer[active_ids, slots[active_ids]] = normalized[active_ids]
                world_buffer[active_ids, slots[active_ids]] = world[active_ids]
                action_mask[active_ids, slots[active_ids]] = 1
                # Finished drones remain inert and are never reset or recorded again.
                if (~active).any():
                    td["agents", "action_normalized"][~active] = 0.5
                    td["agents", "action"][~active] = 0.0
                td = transformed.step(td)["next"]

                position = env.drone.pos[:, 0]
                path_length[active] += (position[active] - previous_position[active]).norm(dim=-1)
                previous_position[active] = position[active]
                sim_steps[active] += 1
                since_frame[active] += 1
                clearance = td["agents", "observation", "lidar"].reshape(num_envs, -1).amin(-1)
                min_clearance[active] = torch.minimum(min_clearance[active], clearance[active])
                collision = td["stats", "collision"].reshape(-1).bool() & active
                out = (((position[:, 2] < .2) | (position[:, 2] > 4.0)) & active)
                goal_distance = (env.target_pos[:, 0] - position).norm(dim=-1)
                speed = env.drone.vel_w[:, 0, :3].norm(dim=-1)
                low_speed = braking & active & (speed < args.settle_speed)
                settle_streak = torch.where(
                    low_speed, settle_streak + 1, torch.zeros_like(settle_streak))
                settled = braking & active & (settle_streak >= args.settle_steps)
                reach = settled & (goal_distance < args.goal_radius)
                # A vehicle can drift outside the radius while braking. It is
                # not declared successful; release it so PPO can reacquire.
                settled_outside = settled & (~reach)
                braking[settled_outside] = False
                settle_streak[settled_outside] = 0
                timeout = ((td["truncated"].reshape(-1).bool() |
                            (sim_steps >= args.max_steps)) & active)
                done = collision | reach | out | timeout
                for env_id in done.nonzero().flatten().cpu().tolist():
                    if collision[env_id]:
                        terminal_reasons[env_id] = "collision"
                    elif out[env_id]:
                        terminal_reasons[env_id] = "out_of_bounds"
                    elif reach[env_id]:
                        terminal_reasons[env_id] = "reach_goal"
                    else:
                        terminal_reasons[env_id] = "timeout"
                capture = active & ((since_frame >= horizon) | done)
                if capture.any():
                    capture_ids = capture.nonzero().flatten()
                    raw = lidar_range_image(env)
                    rows = _frame_rows(
                        indices=capture_ids, seed=args.seed, sim_step=sim_steps,
                        frame_index=frame_index, td=td, env=env, range_values=raw,
                        target_dir=snapshot["target_dir_2d"][:, 0].to(device),
                        norm_actions=norm_buffer, world_actions=world_buffer,
                        action_masks=action_mask, step_deltas=since_frame,
                        terminal_reasons=terminal_reasons)
                    writer.append(rows)
                    frame_index[capture_ids] += 1
                    since_frame[capture_ids] = 0
                    norm_buffer[capture_ids] = 0
                    world_buffer[capture_ids] = 0
                    action_mask[capture_ids] = 0
                active[done] = False
                if int(sim_steps.max()) % 100 == 0:
                    print(f"[collect] seed={args.seed} step={int(sim_steps.max())} "
                          f"active={int(active.sum())} frames={writer.count}", flush=True)

        raw_reason_counts = Counter(terminal_reasons.values())
        if len(terminal_reasons) != num_envs:
            raise RuntimeError(f"only {len(terminal_reasons)}/{num_envs} scenes terminated")
        all_episodes = []
        for env_id in range(num_envs):
            all_episodes.append({
                "scene_id": env_id, "env_id": env_id,
                "termination_reason": terminal_reasons[env_id],
                "steps": int(sim_steps[env_id]), "frames": int(frame_index[env_id]),
                "path_length_m": float(path_length[env_id]),
                "min_clearance_m": float(min_clearance[env_id]),
                "start_position": snapshot["start_positions"][env_id, 0].tolist(),
                "target_position": snapshot["target_positions"][env_id, 0].tolist(),
                "start_side": int(snapshot["start_sides"][env_id]),
                "target_side": int(snapshot["target_sides"][env_id]),
                "brake_attempts": int(brake_attempts[env_id]),
            })
        # A collision (or leaving the valid flight volume) invalidates the
        # complete trajectory, including all frames written before failure.
        accepted_source_ids = [env_id for env_id in range(num_envs)
                               if terminal_reasons[env_id] not in
                               ("collision", "out_of_bounds")]
        accepted_snapshot = slice_environment(snapshot, accepted_source_ids)
        save_environment(environment_path, accepted_snapshot)
        kept_environment = (str(retained_environment_path.relative_to(args.output_root))
                            if keep_environment else None)
        mapping = writer.retain_scenes(
            accepted_source_ids,
            token_prefix=(f"o{args.static_obstacles:04d}-h{shard_id:05d}"
                          f"-s{args.seed:08d}"),
            metadata_updates={
                "environment_file": kept_environment,
                "environment_sha256": file_sha256(environment_path),
                "accepted_scenes": len(accepted_source_ids),
                "discarded_scenes": num_envs - len(accepted_source_ids),
            })
        episodes = []
        for episode in all_episodes:
            source_id = int(episode["scene_id"])
            if source_id not in mapping:
                continue
            item = dict(episode)
            item["source_env_id"] = source_id
            item["scene_id"] = mapping[source_id]
            item["env_id"] = mapping[source_id]
            episodes.append(item)
        accepted_reason_counts = Counter(x["termination_reason"] for x in episodes)
        summary = {
            **metadata, "frames": writer.count, "scenes": len(episodes),
            "num_envs": len(episodes),
            "source_scenes": num_envs,
            "discarded_scenes": num_envs - len(episodes),
            "raw_termination_counts": dict(raw_reason_counts),
            "termination_counts": dict(accepted_reason_counts),
            "mean_episode_steps": float(np.mean([x["steps"] for x in episodes])),
            "mean_frames_per_scene": float(np.mean([x["frames"] for x in episodes])),
            "wall_time_s": time.perf_counter() - started,
            "environment_path": (str(retained_environment_path)
                                 if keep_environment else None),
            "environment_sha256": file_sha256(environment_path),
            "dataset_path": str(dataset_path),
        }
        writer.close(summary)
        writer = None
        validation = validate_dataset(
            partial_path, accepted_snapshot, len(accepted_source_ids), horizon)
        partial_path.replace(dataset_path)
        if keep_environment and environment_path != retained_environment_path:
            retained_environment_path.parent.mkdir(parents=True, exist_ok=True)
            environment_path.replace(retained_environment_path)
        elif not keep_environment:
            environment_path.unlink()
        validation["dataset"] = str(dataset_path)
        summary["dataset_sha256"] = file_sha256(dataset_path)
        summary["dataset_size_bytes"] = dataset_path.stat().st_size
        summary["validation"] = validation
        _save_json(summary_path, {
            "summary": summary, "episodes": episodes,
            "discarded_episodes": [x for x in all_episodes
                                   if x["termination_reason"] in
                                   ("collision", "out_of_bounds")],
        })
        print(json.dumps(summary, indent=2), flush=True)
        completed = True
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if completed:
            # Isaac Sim 2023.1 can segfault while unloading plugins after a
            # successful headless run.  All files are closed and validated at
            # this point, so process exit gives CUDA/driver resources back to
            # the OS without turning a good seed into a failed job.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
