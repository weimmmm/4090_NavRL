"""Strictly causal SUPER windows for the World-DiT + Action-DiT pipeline.

Each item uses three observed LiDAR frames ``[t-2,t-1,t]``.  World-DiT
predicts the single LiDAR latent at ``t+3`` (0.48 s), while Action-DiT predicts
the thirty ego-frame velocity commands executed between ``t`` and ``t+3``.
Only successful trajectories are present in the collector shards.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from lidar_wam.runner.super_future_data import (
    ACTION_SCALE_MPS,
    ACTION_STEPS_PER_FRAME,
    _range_image,
)


HISTORY_FRAMES = 3
FUTURE_FRAMES = 3
ACTION_HORIZON = FUTURE_FRAMES * ACTION_STEPS_PER_FRAME
PROPRIO_DIM = 13
GOAL_DIM = 4


def _normalized_goal(goal_ego: np.ndarray) -> np.ndarray:
    goal = np.asarray(goal_ego, dtype=np.float32)
    return np.concatenate((goal / 25.0, [np.linalg.norm(goal) / 25.0])).astype(
        np.float32
    )


def _normalized_proprio(
    position: np.ndarray,
    velocity_ego: np.ndarray,
    acceleration_ego: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float32).copy()
    # The maps span approximately 40 x 40 m and flight altitude is near 2 m.
    position[:2] /= 20.0
    position[2] /= 5.0
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float32)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-6:
        raise ValueError("zero quaternion in SUPER trajectory")
    quaternion = quaternion / norm
    value = np.concatenate(
        (
            position,
            np.asarray(velocity_ego, dtype=np.float32) / ACTION_SCALE_MPS,
            np.asarray(acceleration_ego, dtype=np.float32) / 4.0,
            quaternion,
        )
    ).astype(np.float32)
    if value.shape != (PROPRIO_DIM,):
        raise ValueError(f"unexpected proprio shape {value.shape}")
    return value


class SuperDualDiTDataset(Dataset):
    """Lazy per-scene HDF5 loader for direct-``t+3`` dual-DiT training."""

    scale = 0.528144
    action_horizon = ACTION_HORIZON
    goal_dim = GOAL_DIM
    proprio_dim = PROPRIO_DIM

    def __init__(self, data_root: Path, max_windows: int | None = None,
                 random_seed: int = 42):
        self.data_root = Path(data_root).expanduser().resolve()
        self.scene_paths = sorted(self.data_root.glob("scene_*/trajectories.h5"))
        if not self.scene_paths:
            raise FileNotFoundError(
                f"no scene_*/trajectories.h5 under {self.data_root}")
        self._handles: dict[int, h5py.File] = {}
        self._windows: list[tuple[int, int, int]] = []
        for scene_index, path in enumerate(self.scene_paths):
            with h5py.File(path, "r") as handle:
                lidar_steps = np.asarray(handle["lidar/step"][:], dtype=np.int32)
                lidar_episode = np.asarray(
                    handle["lidar/episode_index"][:], dtype=np.int32)
                state_lengths = np.asarray(
                    handle["episodes/state_length"][:], dtype=np.int32)
                lidar_offsets = np.asarray(
                    handle["episodes/lidar_offset"][:], dtype=np.int64)
                lidar_lengths = np.asarray(
                    handle["episodes/lidar_length"][:], dtype=np.int32)
                for episode_index in range(len(state_lengths)):
                    lidar_offset = int(lidar_offsets[episode_index])
                    lidar_length = int(lidar_lengths[episode_index])
                    # current local frame k needs k-2,k-1,k and target k+3.
                    for current_local in range(2, lidar_length - FUTURE_FRAMES):
                        rows = [
                            lidar_offset + current_local - 2,
                            lidar_offset + current_local - 1,
                            lidar_offset + current_local,
                            lidar_offset + current_local + FUTURE_FRAMES,
                        ]
                        if not np.all(lidar_episode[rows] == episode_index):
                            raise ValueError(f"non-contiguous episode in {path}")
                        steps = lidar_steps[rows]
                        if not np.array_equal(
                            np.diff(steps),
                            np.asarray([10, 10, 30], dtype=np.int32),
                        ):
                            continue
                        start_step = int(steps[2])
                        if start_step + ACTION_HORIZON >= int(
                            state_lengths[episode_index]
                        ):
                            continue
                        self._windows.append(
                            (scene_index, episode_index, current_local))

        if not self._windows:
            raise RuntimeError("no valid history3/t+3 windows in SUPER dataset")
        if max_windows is not None and len(self._windows) > int(max_windows):
            generator = np.random.default_rng(int(random_seed))
            chosen = np.sort(
                generator.choice(len(self._windows), int(max_windows), replace=False)
            )
            self._windows = [self._windows[int(index)] for index in chosen]

    def __len__(self) -> int:
        return len(self._windows)

    def _handle(self, scene_index: int) -> h5py.File:
        handle = self._handles.get(scene_index)
        if handle is None:
            handle = h5py.File(self.scene_paths[scene_index], "r")
            self._handles[scene_index] = handle
        return handle

    def __getstate__(self):
        value = dict(self.__dict__)
        value["_handles"] = {}
        return value

    def __getitem__(self, item: int):
        scene_index, episode_index, current_local = self._windows[int(item)]
        handle = self._handle(scene_index)
        lidar_offset = int(handle["episodes/lidar_offset"][episode_index])
        state_offset = int(handle["episodes/state_offset"][episode_index])
        lidar_rows = [
            lidar_offset + current_local - 2,
            lidar_offset + current_local - 1,
            lidar_offset + current_local,
            lidar_offset + current_local + FUTURE_FRAMES,
        ]
        lidar_steps = np.asarray(handle["lidar/step"][lidar_rows], np.int32)
        history = np.stack(
            [_range_image(handle["lidar/range_m"][row]) for row in lidar_rows[:3]]
        )
        target = _range_image(handle["lidar/range_m"][lidar_rows[3]])
        current_step = int(lidar_steps[2])
        action_slice = slice(
            state_offset + current_step,
            state_offset + current_step + ACTION_HORIZON,
        )
        actions = np.asarray(
            handle["steps/super_velocity_ego"][action_slice], dtype=np.float32
        )
        if actions.shape != (ACTION_HORIZON, 3):
            raise ValueError(f"unexpected future action shape {actions.shape}")
        actions = np.clip(actions / ACTION_SCALE_MPS, -1.0, 1.0)

        past = np.zeros((ACTION_STEPS_PER_FRAME, 3), dtype=np.float32)
        past_mask = np.zeros(ACTION_STEPS_PER_FRAME, dtype=np.float32)
        past_count = min(ACTION_STEPS_PER_FRAME, current_step)
        if past_count:
            past_slice = slice(
                state_offset + current_step - past_count,
                state_offset + current_step,
            )
            past[-past_count:] = np.clip(
                np.asarray(
                    handle["steps/super_velocity_ego"][past_slice],
                    dtype=np.float32,
                ) / ACTION_SCALE_MPS,
                -1.0,
                1.0,
            )
            past_mask[-past_count:] = 1.0

        row = state_offset + current_step
        goal = _normalized_goal(handle["steps/goal_ego"][row])
        proprio = _normalized_proprio(
            handle["steps/world_position"][row],
            handle["steps/ego_velocity"][row],
            handle["steps/ego_acceleration"][row],
            handle["steps/quaternion_wxyz"][row],
        )
        source = np.asarray(
            [scene_index, episode_index, current_local], dtype=np.int64)
        values = (history, target, actions, goal, proprio, past, past_mask)
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError(
                f"non-finite dual-DiT window {self.scene_paths[scene_index]}:"
                f"{episode_index}:{current_local}")
        return tuple(torch.from_numpy(value) for value in values) + (
            torch.from_numpy(source),
        )

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
    return np.sort(generator.choice(int(length), size, replace=False)).astype(
        np.int64
    )
