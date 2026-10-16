"""Action-only windows from the L20 PPO success-only trajectory dataset.

The collector records one LiDAR frame every ten 16-ms physics steps.  The
action sequence stored on a frame is the command sequence that was executed
*before* that frame was captured.  Consequently an action-policy item uses
only the current LiDAR frame ``t`` and reads its ten future Lee commands from
frame ``t+1``.  No future LiDAR value is returned to the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


ACTION_HORIZON = 10
LIDAR_HISTORY_FRAMES = 1
PAST_HORIZON = 30
GOAL_DIM = 4
PROPRIO_DIM = 10
DATASET_FORMAT = "navrl-isaac-trajectory-hdf5-v2"
ACTION_LABEL = "actual_world_velocity_input_to_original_lee"

REQUIRED_FRAME_FIELDS = {
    "action_mask",
    "drone_state",
    "env_id",
    "frame_index",
    "range_values",
    "sim_step",
    "target_position",
    "world_action_sequence",
}
REQUIRED_STEP_FIELDS = {
    "action_valid",
    "ego_acceleration",
    "ego_velocity",
    "episode_index",
    "lee_target_velocity_world",
    "world_velocity",
}
REQUIRED_EPISODE_FIELDS = {"step_length", "step_offset"}


def discover_shard_names(data_root: Path) -> tuple[str, ...]:
    """Return unique root-relative directories containing trajectory shards."""
    root = Path(data_root).expanduser().resolve()
    return tuple(
        path.parent.relative_to(root).as_posix()
        for path in sorted(root.rglob("trajectories.h5")))


def _world_to_ego_yaw(vector: np.ndarray,
                      quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world ENU vectors into the current yaw-only ego FLU frame."""
    value = np.asarray(vector, dtype=np.float32)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-6):
        raise ValueError("zero quaternion in L20 trajectory")
    quaternion = quaternion / norm
    w, x, y, z = (quaternion[..., index] for index in range(4))
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    while yaw.ndim < value.ndim - 1:
        yaw = np.expand_dims(yaw, axis=-1)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    result = np.empty_like(value, dtype=np.float32)
    result[..., 0] = cosine * value[..., 0] + sine * value[..., 1]
    result[..., 1] = -sine * value[..., 0] + cosine * value[..., 1]
    result[..., 2] = value[..., 2]
    return result


def _goal_features(position_world: np.ndarray, goal_world: np.ndarray,
                   quaternion_wxyz: np.ndarray) -> np.ndarray:
    goal = _world_to_ego_yaw(
        np.asarray(goal_world, np.float32) - np.asarray(position_world, np.float32),
        quaternion_wxyz,
    )
    return np.concatenate((goal, [np.linalg.norm(goal)])).astype(np.float32)


def _proprio_features(velocity_ego: np.ndarray,
                       acceleration_ego: np.ndarray,
                       quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float32)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-6:
        raise ValueError("zero quaternion in L20 trajectory")
    value = np.concatenate((
        np.asarray(velocity_ego, dtype=np.float32),
        np.asarray(acceleration_ego, dtype=np.float32),
        quaternion / norm,
    )).astype(np.float32)
    if value.shape != (PROPRIO_DIM,):
        raise ValueError(f"unexpected proprio shape {value.shape}")
    return value


class SuperLeeActionDataset(Dataset):
    """Lazy single-frame loader for exact commands sent to the original Lee."""

    action_horizon = ACTION_HORIZON
    lidar_history_frames = LIDAR_HISTORY_FRAMES
    past_horizon = PAST_HORIZON
    goal_dim = GOAL_DIM
    proprio_dim = PROPRIO_DIM
    action_units = "m/s"
    action_frame = "current_observation_yaw_only_ego_flu"
    action_semantics = "actual_velocity_input_to_original_lee"
    past_input_semantics = (
        "actual_vehicle_velocity_in_current_observation_ego_frame")

    def __init__(self, data_root: Path,
                 shard_names: Iterable[str] | None = None,
                 max_windows: int | None = None,
                 random_seed: int = 42, include_history: bool = True):
        self.data_root = Path(data_root).expanduser().resolve()
        self.include_history = bool(include_history)
        available = {
            path.parent.relative_to(self.data_root).as_posix(): path
            for path in sorted(self.data_root.rglob("trajectories.h5"))
        }
        if not available:
            raise FileNotFoundError(
                f"no trajectories.h5 below {self.data_root}")
        if shard_names is None:
            selected_names = tuple(sorted(available))
        else:
            selected_names = tuple(dict.fromkeys(str(name) for name in shard_names))
            missing = sorted(set(selected_names) - set(available))
            if missing:
                raise FileNotFoundError(
                    "requested trajectory shards are missing: "
                    + ", ".join(missing))
        if not selected_names:
            raise ValueError("at least one trajectory shard must be selected")

        self.shard_names = selected_names
        self.scene_paths = tuple(available[name] for name in selected_names)
        self._handles: dict[int, h5py.File] = {}
        window_chunks: list[np.ndarray] = []
        self.window_count_by_shard: dict[str, int] = {}

        for shard_index, (name, path) in enumerate(
                zip(self.shard_names, self.scene_paths)):
            with h5py.File(path, "r") as handle:
                self._validate_handle(handle, path, require_history=self.include_history)
                frames = handle["frames"]
                env_id = np.asarray(frames["env_id"][:], dtype=np.int32)
                frame_index = np.asarray(
                    frames["frame_index"][:], dtype=np.int32)
                sim_step = np.asarray(frames["sim_step"][:], dtype=np.int32)
                order = np.lexsort((frame_index, env_id)).astype(np.int32)
                if len(order) < LIDAR_HISTORY_FRAMES + 1:
                    rows = np.empty((0, 5), dtype=np.int32)
                else:
                    candidate = np.stack((
                        order[:-1], order[1:]),
                        axis=1,
                    )
                    candidate_env = env_id[candidate]
                    candidate_frame = frame_index[candidate]
                    candidate_step = sim_step[candidate]
                    valid = np.all(
                        candidate_env == candidate_env[:, :1], axis=1)
                    valid &= np.all(
                        np.diff(candidate_frame, axis=1) == 1, axis=1)
                    valid &= np.all(
                        np.diff(candidate_step, axis=1) == ACTION_HORIZON,
                        axis=1)

                    # Frame t+1 contains exactly the commands executed during
                    # [t,t+1].  Initial and shortened terminal frames are not
                    # valid full-horizon supervision.
                    all_masks = np.asarray(
                        frames["action_mask"][:], dtype=np.bool_)
                    valid &= np.all(all_masks[candidate[:, 1]], axis=1)

                    episode = candidate_env[:, 0].astype(np.int64)
                    lengths = np.asarray(
                        handle["episodes/step_length"][:], dtype=np.int64)
                    offsets = np.asarray(
                        handle["episodes/step_offset"][:], dtype=np.int64)
                    if (episode.size and
                            (episode.min() < 0 or episode.max() >= len(lengths))):
                        raise ValueError(f"frame env_id is not an episode index in {path}")
                    current_step = candidate_step[:, 0].astype(np.int64)
                    possible = current_step + ACTION_HORIZON <= lengths[episode] - 1
                    valid &= possible

                    action_valid = np.asarray(
                        handle["steps/action_valid"][:], dtype=np.bool_)
                    invalid_prefix = np.concatenate((
                        np.zeros(1, dtype=np.int64),
                        np.cumsum(~action_valid, dtype=np.int64),
                    ))
                    starts = offsets[episode] + current_step
                    stops = starts + ACTION_HORIZON
                    # Some terminal candidates are already marked impossible
                    # above; clip only to make this vectorized lookup safe for
                    # the final episode in a shard.  ``possible`` keeps those
                    # clipped candidates out of the resulting windows.
                    safe_stops = np.minimum(stops, len(action_valid))
                    valid &= (
                        invalid_prefix[safe_stops]
                        - invalid_prefix[starts]
                    ) == 0
                    kept = candidate[valid]
                    kept_episode = episode[valid].astype(np.int32)
                    kept_step = current_step[valid].astype(np.int32)
                    rows = np.column_stack((
                        np.full(len(kept), shard_index, dtype=np.int32),
                        kept_episode,
                        kept,
                        kept_step,
                    )).astype(np.int32, copy=False)

            self.window_count_by_shard[name] = int(len(rows))
            if len(rows):
                window_chunks.append(rows)

        if not window_chunks:
            raise RuntimeError("no valid single-frame action windows in L20 dataset")
        self._windows = np.concatenate(window_chunks, axis=0)
        if max_windows is not None and len(self._windows) > int(max_windows):
            generator = np.random.default_rng(int(random_seed))
            chosen = np.sort(generator.choice(
                len(self._windows), int(max_windows), replace=False))
            self._windows = self._windows[chosen]

    @staticmethod
    def _validate_handle(handle: h5py.File, path: Path, require_history=True) -> None:
        if str(handle.attrs.get("format", "")) != DATASET_FORMAT:
            raise ValueError(f"unsupported L20 dataset format in {path}")
        try:
            metadata = json.loads(str(handle.attrs["metadata_json"]))
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid metadata_json in {path}") from error
        if str(metadata.get("action_label", "")) != ACTION_LABEL:
            raise ValueError(f"unsupported action label in {path}")
        if not np.isclose(float(metadata.get("sim_dt", -1.0)), 0.016):
            raise ValueError(f"physics dt changed in {path}")
        if int(metadata.get("sample_interval", -1)) != ACTION_HORIZON:
            raise ValueError(f"LiDAR interval changed in {path}")
        if not bool(metadata.get("success_only", False)):
            raise ValueError(f"dataset is not success-only: {path}")
        recording = metadata.get("recording_lidar", {})
        if not np.isclose(float(recording.get("range_m", -1.0)), 5.0):
            raise ValueError(f"recording LiDAR is not 5 m in {path}")
        missing_frames = REQUIRED_FRAME_FIELDS - set(handle["frames"].keys())
        required_steps = REQUIRED_STEP_FIELDS if require_history else (
            REQUIRED_STEP_FIELDS - {"world_velocity"})
        missing_steps = required_steps - set(handle["steps"].keys())
        missing_episodes = REQUIRED_EPISODE_FIELDS - set(
            handle["episodes"].keys())
        if missing_frames or missing_steps or missing_episodes:
            raise ValueError(
                f"missing required fields in {path}: frames={sorted(missing_frames)}, "
                f"steps={sorted(missing_steps)}, "
                f"episodes={sorted(missing_episodes)}")
        if tuple(handle["frames/range_values"].shape[1:]) != (2, 108, 20):
            raise ValueError(f"unexpected LiDAR image shape in {path}")
        if int(handle.attrs.get("dense_state_interval_steps", -1)) != 1:
            raise ValueError(f"dense state interval changed in {path}")

    def __len__(self) -> int:
        return int(len(self._windows))

    def _handle(self, shard_index: int) -> h5py.File:
        handle = self._handles.get(shard_index)
        if handle is None:
            handle = h5py.File(self.scene_paths[shard_index], "r")
            self._handles[shard_index] = handle
        return handle

    def __getstate__(self):
        value = dict(self.__dict__)
        value["_handles"] = {}
        return value

    def _location(self, item: int):
        values = (int(value) for value in self._windows[int(item)])
        shard, episode, current, successor, current_step = values
        handle = self._handle(shard)
        step_offset = int(handle["episodes/step_offset"][episode])
        return (shard, episode, current, successor,
                current_step, step_offset, handle)

    def conditions(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        """Read current goal and ego state without loading LiDAR arrays."""
        (_, _, current, _, current_step, step_offset,
         handle) = self._location(item)
        frame_state = np.asarray(
            handle["frames/drone_state"][current], dtype=np.float32)
        goal_world = np.asarray(
            handle["frames/target_position"][current], dtype=np.float32)
        dense_row = step_offset + current_step
        return (
            _goal_features(frame_state[:3], goal_world, frame_state[3:7]),
            _proprio_features(
                handle["steps/ego_velocity"][dense_row],
                handle["steps/ego_acceleration"][dense_row],
                frame_state[3:7],
            ),
        )

    def conditions_many(self, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch condition reads by shard for fast normalization statistics."""
        items = np.asarray(items, dtype=np.int64)
        goals = np.empty((len(items), GOAL_DIM), dtype=np.float32)
        proprios = np.empty((len(items), PROPRIO_DIM), dtype=np.float32)
        selected = self._windows[items]
        shards = selected[:, 0]
        for shard_index in np.unique(shards):
            output_rows = np.flatnonzero(shards == shard_index)
            values = selected[output_rows]
            handle = self._handle(int(shard_index))
            episodes = values[:, 1].astype(np.int64, copy=False)
            current_rows = values[:, 2].astype(np.int64, copy=False)
            current_steps = values[:, 4].astype(np.int64, copy=False)
            offsets = np.asarray(
                handle["episodes/step_offset"][:], dtype=np.int64)
            dense_rows = offsets[episodes] + current_steps

            def read_unique(dataset, rows):
                unique, inverse = np.unique(rows, return_inverse=True)
                return np.asarray(dataset[unique], dtype=np.float32)[inverse]

            state = read_unique(handle["frames/drone_state"], current_rows)
            target = read_unique(handle["frames/target_position"], current_rows)
            velocity = read_unique(handle["steps/ego_velocity"], dense_rows)
            acceleration = read_unique(
                handle["steps/ego_acceleration"], dense_rows)
            quaternion = state[:, 3:7]
            quaternion /= np.maximum(
                np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-6)
            goal_xyz = _world_to_ego_yaw(
                target - state[:, :3], quaternion)
            goals[output_rows, :3] = goal_xyz
            goals[output_rows, 3] = np.linalg.norm(goal_xyz, axis=-1)
            proprios[output_rows] = np.concatenate((
                velocity, acceleration, quaternion), axis=-1)
        if not np.isfinite(goals).all() or not np.isfinite(proprios).all():
            raise ValueError("non-finite batched L20 conditions")
        return goals, proprios

    def __getitem__(self, item: int, load_lidar: bool = True):
        (shard, episode, current, successor, current_step,
         step_offset, handle) = self._location(item)
        frames = handle["frames"]
        image = (np.asarray(frames["range_values"][current], dtype=np.float32)
                 if load_lidar else np.zeros((2, 108, 20), dtype=np.float32))
        if image.shape != (2, 108, 20):
            raise ValueError(f"unexpected LiDAR frame shape {image.shape}")

        current_state = np.asarray(
            frames["drone_state"][current], dtype=np.float32)
        world_actions = np.asarray(
            frames["world_action_sequence"][successor], dtype=np.float32)
        actions = _world_to_ego_yaw(world_actions, current_state[3:7])
        if actions.shape != (ACTION_HORIZON, 3):
            raise ValueError(f"unexpected future action shape {actions.shape}")

        include_history = getattr(self, "include_history", True)
        if include_history:
            past = np.zeros((PAST_HORIZON, 3), dtype=np.float32)
            past_mask = np.zeros(PAST_HORIZON, dtype=np.float32)
            past_count = min(PAST_HORIZON, current_step)
            if past_count:
                start = step_offset + current_step - past_count
                stop = step_offset + current_step
                world_past = np.asarray(
                    handle["steps/world_velocity"][start:stop], dtype=np.float32)
                # Rotate all historical rows into the current observation's
                # yaw frame; legacy checkpoints retain this exact input.
                past[-past_count:] = _world_to_ego_yaw(
                    world_past, current_state[3:7])
                past_mask[-past_count:] = 1.0

        goal, proprio = self.conditions(item)
        source = np.asarray((
            shard, episode, int(frames["frame_index"][current]),
            current_step,
        ), dtype=np.int64)
        values = (image, actions, goal, proprio)
        if include_history:
            values += (past, past_mask)
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError(
                f"non-finite action window {self.scene_paths[shard]}:"
                f"{episode}:{current_step}")
        return tuple(torch.from_numpy(value) for value in values) + (
            torch.from_numpy(source),)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def fixed_eval_indices(length: int, count: int, seed: int) -> np.ndarray:
    size = min(int(length), int(count))
    generator = np.random.default_rng(int(seed))
    return np.sort(generator.choice(
        int(length), size, replace=False)).astype(np.int64)
