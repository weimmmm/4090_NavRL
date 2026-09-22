"""Replay saved dataset actions in their exact fixed Isaac environment.

The HDF5 convention is intentionally explicit: row zero is the initial frame,
and every later row stores the actions that moved the drone from the preceding
frame to that row.  This tool therefore executes row ``i + 1`` before comparing
the live state with row ``i + 1``.  It never feeds a row's incoming actions
back into that same row's observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_eval.bootstrap import activate_vendored_sources

activate_vendored_sources()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-start", type=int, default=0)
    parser.add_argument("--route-limit", type=int)
    parser.add_argument("--action-source", choices=("world", "recomputed"),
                        default="world")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--initial-lidar-tolerance", type=float, default=1e-5)
    parser.add_argument("--divergence-position-m", type=float, default=0.05)
    args = parser.parse_args()
    args.environment = args.environment.expanduser().resolve()
    args.dataset = args.dataset.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.environment.is_file() or not args.dataset.is_file():
        parser.error("--environment and --dataset must exist")
    if args.route_start < 0:
        parser.error("--route-start must be nonnegative")
    if args.route_limit is not None and args.route_limit <= 0:
        parser.error("--route-limit must be positive")
    return parser, args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(value) -> str:
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def goal_actions_to_world(normalized: np.ndarray,
                          direction: np.ndarray) -> np.ndarray:
    """Convert normalized goal-frame velocities to world coordinates."""
    direction = np.asarray(direction, np.float32).copy()
    direction[..., 2] = 0.0
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float32).eps):
        raise ValueError("zero target_dir_2d in dataset")
    goal_x = direction / norm
    goal_y = np.stack((-goal_x[..., 1], goal_x[..., 0],
                       np.zeros_like(goal_x[..., 0])), axis=-1)
    goal_z = np.broadcast_to(
        np.asarray([0.0, 0.0, 1.0], np.float32), goal_x.shape)
    velocity = 4.0 * np.asarray(normalized, np.float32) - 2.0
    return (velocity[..., 0:1] * goal_x[..., None, :]
            + velocity[..., 1:2] * goal_y[..., None, :]
            + velocity[..., 2:3] * goal_z[..., None, :])


def load_replay_data(dataset_path: Path, source_route_ids: list[int],
                     max_steps: int):
    import h5py

    handle = h5py.File(dataset_path, "r")
    frames = handle["frames"] if "frames" in handle else handle
    scene = np.asarray(frames["scene_id"][:], np.int64)
    frame_index = np.asarray(frames["frame_index"][:], np.int64)
    sim_step = np.asarray(frames["sim_step"][:], np.int64)
    step_delta = np.asarray(frames["step_delta"][:], np.int64)
    mask = np.asarray(frames["action_mask"][:], bool)
    world = np.asarray(frames["world_action_sequence"][:], np.float32)
    normalized = np.asarray(frames["normalized_action_sequence"][:], np.float32)
    direction = np.asarray(frames["target_dir_2d"][:], np.float32)

    route_rows: list[np.ndarray] = []
    actions = np.zeros((len(source_route_ids), max_steps, 3), np.float32)
    recomputed_actions = np.zeros_like(actions)
    capture_by_step: dict[int, list[tuple[int, int]]] = defaultdict(list)
    reference_reasons, reference_steps = [], []
    conversion_max = 0.0

    for local_id, source_id in enumerate(source_route_ids):
        rows = np.flatnonzero(scene == source_id)
        if not len(rows):
            raise ValueError(f"dataset has no scene_id={source_id}")
        rows = rows[np.argsort(frame_index[rows])]
        if not np.array_equal(frame_index[rows], np.arange(len(rows))):
            raise ValueError(f"scene {source_id}: non-contiguous frame indices")
        if sim_step[rows[0]] != 0 or step_delta[rows[0]] != 0 or mask[rows[0]].any():
            raise ValueError(f"scene {source_id}: invalid initial frame")
        previous_step = 0
        for row in rows[1:]:
            end = int(sim_step[row])
            delta = int(step_delta[row])
            if end - previous_step != delta or not 1 <= delta <= 10:
                raise ValueError(f"scene {source_id}: broken step interval at row {row}")
            if int(mask[row].sum()) != delta or not mask[row, :delta].all():
                raise ValueError(f"scene {source_id}: invalid action mask at row {row}")
            if end > max_steps:
                raise ValueError(f"scene {source_id}: sim_step {end} exceeds {max_steps}")
            stored = world[row, :delta]
            converted = goal_actions_to_world(
                normalized[row:row+1, :delta], direction[row:row+1])[0]
            conversion_max = max(
                conversion_max, float(np.max(np.abs(stored-converted))))
            actions[local_id, previous_step:end] = stored
            recomputed_actions[local_id, previous_step:end] = converted
            capture_by_step[end].append((local_id, int(row)))
            previous_step = end
        reason = _decode(frames["termination_reason"][rows[-1]])
        if reason not in ("reach_goal", "collision", "out_of_bounds", "timeout"):
            raise ValueError(f"scene {source_id}: invalid final reason {reason!r}")
        reference_reasons.append(reason)
        reference_steps.append(int(sim_step[rows[-1]]))
        route_rows.append(rows)

    metadata = json.loads(handle.attrs.get("metadata_json", "{}"))
    return {
        "handle": handle,
        "frames": frames,
        "route_rows": route_rows,
        "actions": actions,
        "recomputed_actions": recomputed_actions,
        "capture_by_step": capture_by_step,
        "reference_reasons": reference_reasons,
        "reference_steps": np.asarray(reference_steps, np.int64),
        "conversion_max_abs_error": conversion_max,
        "metadata": metadata,
    }


def _statistics(values: list[float]):
    if not values:
        return {"count": 0, "mean": None, "median": None,
                "p95": None, "p99": None, "max": None}
    value = np.asarray(values, np.float64)
    return {
        "count": int(len(value)), "mean": float(value.mean()),
        "median": float(np.median(value)),
        "p95": float(np.quantile(value, 0.95)),
        "p99": float(np.quantile(value, 0.99)), "max": float(value.max()),
    }


def _quaternion_error(reference: np.ndarray, live: np.ndarray) -> np.ndarray:
    reference = reference / np.maximum(np.linalg.norm(reference, axis=-1, keepdims=True), 1e-12)
    live = live / np.maximum(np.linalg.norm(live, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(reference*live, axis=-1)).clip(0.0, 1.0)
    return 2.0*np.arccos(dot)


def main():
    parser, args = parse_args()
    from isaac_eval.environment_file import load_environment, mesh_sha256, slice_environment

    complete = load_environment(args.environment)
    source_count = int(complete["num_envs"])
    route_limit = source_count-args.route_start if args.route_limit is None else args.route_limit
    if args.route_start+route_limit > source_count:
        parser.error("requested route slice exceeds the environment")
    source_route_ids = list(range(args.route_start, args.route_start+route_limit))
    fixed = slice_environment(complete, source_route_ids)
    replay = load_replay_data(args.dataset, source_route_ids, int(fixed["max_steps"]))
    if replay["conversion_max_abs_error"] > 1e-5:
        replay["handle"].close()
        raise RuntimeError(
            "normalized-to-world action reconstruction exceeds 1e-5: "
            f"{replay['conversion_max_abs_error']}")

    physical_gpu = int(args.device.split(":")[-1])
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(physical_gpu))
    from omni.isaac.kit import SimulationApp
    app = SimulationApp({
        "headless": not args.render, "anti_aliasing": 1,
        "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False,
    })
    env = None
    completed = False
    started = time.perf_counter()
    try:
        import torch
        from hydra import compose, initialize_config_dir
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        from torchrl.envs.transforms import Compose, TransformedEnv

        from isaac_eval.collect_dataset import _stage_mesh
        from isaac_eval.navigation_env import NavigationEnv
        from isaac_eval.policy import lidar_range_image

        with initialize_config_dir(
                config_dir=str(ROOT / "isaac_eval" / "cfg"), version_base=None):
            cfg = compose(config_name="train")
        cfg.device = "cuda:0"
        cfg.headless = not args.render
        cfg.seed = int(fixed["terrain_seed"])
        cfg.env.num_envs = route_limit
        cfg.env.max_episode_length = int(fixed["max_steps"])
        cfg.env.num_obstacles = int(fixed["static_obstacles"])
        cfg.env.randomize_routes = False
        cfg.env_dyn.num_obstacles = 0
        cfg.enable_eval = False
        cfg.record_eval_video = False

        np.random.seed(int(fixed["route_seed"]))
        torch.manual_seed(int(fixed["route_seed"]))
        torch.cuda.manual_seed_all(int(fixed["route_seed"]))
        env = NavigationEnv(cfg, eval_environment=fixed)
        env.eval()
        env.enable_render(args.render)
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        transformed = TransformedEnv(
            env, Compose(VelController(controller, yaw_control=False))).eval()
        transformed.set_seed(int(fixed["route_seed"]))
        td = transformed.reset()
        device = torch.device(cfg.device)

        vertices, faces = _stage_mesh()
        live_mesh_hash = mesh_sha256(vertices, faces)
        expected_mesh_hash = fixed["terrain_mesh"]["sha256"]
        initial_state = td["info", "drone_state"].reshape(route_limit, 13)
        starts_match = torch.equal(
            initial_state[:, :3].cpu(), fixed["start_positions"][:, 0])
        targets_match = torch.equal(
            env.target_pos.cpu(), fixed["target_positions"])
        initial_image = lidar_range_image(env).detach().cpu().numpy()
        initial_rows = [int(rows[0]) for rows in replay["route_rows"]]
        reference_initial = replay["frames"]["range_values"][initial_rows]
        initial_lidar_error = np.abs(initial_image-reference_initial)
        initial_lidar_max = float(initial_lidar_error.max())
        if (live_mesh_hash != expected_mesh_hash or not starts_match
                or not targets_match
                or initial_lidar_max > args.initial_lidar_tolerance):
            raise RuntimeError("initial fixed-environment replay check failed: " + json.dumps({
                "mesh_matches": live_mesh_hash == expected_mesh_hash,
                "starts_match": starts_match, "targets_match": targets_match,
                "initial_lidar_max_abs_error": initial_lidar_max,
            }))

        selected_actions = (replay["actions"] if args.action_source == "world"
                            else replay["recomputed_actions"])
        selected_actions = torch.from_numpy(selected_actions).to(device)
        active = torch.ones(route_limit, dtype=torch.bool, device=device)
        replay_reason = [None]*route_limit
        replay_step = [None]*route_limit
        first_divergence = [None]*route_limit
        position_errors: list[float] = []
        velocity_errors: list[float] = []
        angular_velocity_errors: list[float] = []
        quaternion_errors: list[float] = []
        lidar_maes: list[float] = []
        lidar_maxes: list[float] = []
        frame_comparisons = 0
        max_reference_step = int(replay["reference_steps"].max())

        with torch.no_grad():
            for step in range(1, max_reference_step+1):
                # Keep the pre-step mask so the terminal frame is compared,
                # while routes that terminated on an earlier step never
                # contaminate later trajectory-error statistics.
                active_before_step = active.clone()
                command = selected_actions[:, step-1].clone()
                command[~active] = 0.0
                td.set(("agents", "action"), command[:, None])
                td = transformed.step(td)["next"]
                position = env.drone.pos[:, 0]
                collision = td["stats", "collision"].reshape(-1).bool() & active
                reach = td["stats", "reach_goal"].reshape(-1).bool() & active
                out = (((position[:, 2] < 0.2) | (position[:, 2] > 4.0)) & active)
                timeout = ((td["truncated"].reshape(-1).bool()
                            | (step >= int(fixed["max_steps"]))) & active)
                done = collision | reach | out | timeout
                for local_id in done.nonzero().flatten().cpu().tolist():
                    if collision[local_id]:
                        reason = "collision"
                    elif out[local_id]:
                        reason = "out_of_bounds"
                    elif reach[local_id]:
                        reason = "reach_goal"
                    else:
                        reason = "timeout"
                    replay_reason[local_id] = reason
                    replay_step[local_id] = step
                active[done] = False

                comparisons = replay["capture_by_step"].get(step, [])
                if comparisons:
                    comparisons = [
                        item for item in comparisons
                        if bool(active_before_step[item[0]])
                    ]
                if comparisons:
                    local_ids = np.asarray(
                        [item[0] for item in comparisons], np.int64)
                    rows = np.asarray(
                        [item[1] for item in comparisons], np.int64)
                    order = np.argsort(rows)
                    sorted_rows = rows[order]
                    reference_state = np.asarray(
                        replay["frames"]["drone_state"][sorted_rows], np.float32)
                    reference_range = np.asarray(
                        replay["frames"]["range_values"][sorted_rows], np.float32)
                    inverse = np.argsort(order)
                    reference_state = reference_state[inverse]
                    reference_range = reference_range[inverse]
                    live_state = td["info", "drone_state"].reshape(
                        route_limit, 13)[torch.as_tensor(local_ids, device=device)]
                    live_state = live_state.detach().cpu().numpy()
                    live_range = lidar_range_image(env)[
                        torch.as_tensor(local_ids, device=device)].detach().cpu().numpy()
                    position_error = np.linalg.norm(
                        live_state[:, :3]-reference_state[:, :3], axis=-1)
                    velocity_error = np.linalg.norm(
                        live_state[:, 7:10]-reference_state[:, 7:10], axis=-1)
                    angular_error = np.linalg.norm(
                        live_state[:, 10:13]-reference_state[:, 10:13], axis=-1)
                    quaternion_error = _quaternion_error(
                        reference_state[:, 3:7], live_state[:, 3:7])
                    range_error = np.abs(live_range-reference_range)
                    position_errors.extend(position_error.tolist())
                    velocity_errors.extend(velocity_error.tolist())
                    angular_velocity_errors.extend(angular_error.tolist())
                    quaternion_errors.extend(quaternion_error.tolist())
                    lidar_maes.extend(range_error.reshape(len(rows), -1).mean(-1).tolist())
                    lidar_maxes.extend(range_error.reshape(len(rows), -1).max(-1).tolist())
                    frame_comparisons += len(rows)
                    for index, error in zip(local_ids.tolist(), position_error.tolist()):
                        if (first_divergence[index] is None
                                and error > args.divergence_position_m):
                            first_divergence[index] = step

                reference_end = torch.from_numpy(
                    replay["reference_steps"] == step).to(device)
                missing = active & reference_end
                for local_id in missing.nonzero().flatten().cpu().tolist():
                    replay_reason[local_id] = "no_terminal_at_reference_end"
                    replay_step[local_id] = step
                active[missing] = False
                if step % 100 == 0 or not active.any():
                    counts = Counter(reason for reason in replay_reason if reason)
                    print(json.dumps({
                        "step": step, "active": int(active.sum()),
                        "termination_counts": dict(counts),
                    }), flush=True)
                if not active.any():
                    break

        episodes = []
        matrix: dict[str, Counter] = defaultdict(Counter)
        for local_id, source_id in enumerate(source_route_ids):
            expected = replay["reference_reasons"][local_id]
            actual = replay_reason[local_id] or "unfinished"
            matrix[expected][actual] += 1
            episodes.append({
                "route_id": source_id,
                "reference_termination_reason": expected,
                "reference_step": int(replay["reference_steps"][local_id]),
                "replay_termination_reason": actual,
                "replay_step": replay_step[local_id],
                "termination_reason_matches": expected == actual,
                "first_position_divergence_step": first_divergence[local_id],
            })
        replay_counts = Counter(row["replay_termination_reason"] for row in episodes)
        reference_counts = Counter(row["reference_termination_reason"] for row in episodes)
        reason_match_rate = sum(
            row["termination_reason_matches"] for row in episodes)/len(episodes)
        success_rate = replay_counts["reach_goal"]/len(episodes)
        reference_success_rate = reference_counts["reach_goal"]/len(episodes)
        if success_rate >= 0.80 and reason_match_rate >= 0.95:
            verdict = "green"
        elif success_rate >= 0.70 and reason_match_rate >= 0.80:
            verdict = "yellow"
        else:
            verdict = "red"
        result = {
            "format": "navrl-dataset-action-replay-v1",
            "summary": {
                "environment": str(args.environment),
                "environment_sha256": file_sha256(args.environment),
                "dataset": str(args.dataset),
                "dataset_sha256": file_sha256(args.dataset),
                "route_start": args.route_start, "route_limit": route_limit,
                "source_route_ids": source_route_ids,
                "action_source": args.action_source,
                "normalized_to_world_max_abs_error": replay[
                    "conversion_max_abs_error"],
                "mesh_sha256": live_mesh_hash,
                "mesh_hash_matches": live_mesh_hash == expected_mesh_hash,
                "starts_match": starts_match, "targets_match": targets_match,
                "initial_lidar_max_abs_error": initial_lidar_max,
                "initial_lidar_mean_abs_error": float(initial_lidar_error.mean()),
                "reference_termination_counts": dict(reference_counts),
                "replay_termination_counts": dict(replay_counts),
                "reference_success_rate": reference_success_rate,
                "replay_success_rate": success_rate,
                "termination_reason_match_rate": reason_match_rate,
                "termination_confusion": {
                    expected: dict(actual) for expected, actual in matrix.items()},
                "frame_comparisons": frame_comparisons,
                "position_error_m": _statistics(position_errors),
                "linear_velocity_error_mps": _statistics(velocity_errors),
                "angular_velocity_error_radps": _statistics(angular_velocity_errors),
                "quaternion_angle_error_rad": _statistics(quaternion_errors),
                "lidar_mae_normalized": _statistics(lidar_maes),
                "lidar_max_abs_error_normalized": _statistics(lidar_maxes),
                "verdict": verdict,
                "wall_time_s": time.perf_counter()-started,
            },
            "episode_results": episodes,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix+".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        os.replace(temporary, args.output)
        print(json.dumps({"output": str(args.output), **result["summary"]},
                         indent=2), flush=True)
        completed = True
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        replay["handle"].close()
        if env is not None:
            env.close()
        if completed:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
