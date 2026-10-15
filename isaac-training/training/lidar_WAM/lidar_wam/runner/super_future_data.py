"""SUPER-trajectory windows for the frozen-VAE Future Model.

The collector stores one LiDAR frame every ten 0.016 s physics steps.  This
adapter keeps the dataset in its native per-scene HDF5 layout and exposes a
direct t+3 sample without making a PPO-style intermediate dataset.
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


LIDAR_RANGE_M = 5.0
ACTION_SCALE_MPS = 2.0
ACTION_HORIZON = 3
ACTION_STEPS_PER_FRAME = 10
STATE_DIM = 9


def _range_image(proximity: np.ndarray) -> np.ndarray:
    """Convert collector proximity (0=no hit, 5=near contact) to WAM image."""
    proximity = np.asarray(proximity, dtype=np.float32)
    if proximity.shape != (108, 18):
        raise ValueError(f"expected [108,18] proximity, got {proximity.shape}")
    distance = np.clip(LIDAR_RANGE_M - proximity, 0.0, LIDAR_RANGE_M)
    valid = proximity > 1e-4
    image = np.ones((2, 108, 20), dtype=np.float32)
    image[1].fill(-1.0)
    image[0, :, :18] = 2.0 * distance / LIDAR_RANGE_M - 1.0
    image[1, :, :18] = valid.astype(np.float32) * 2.0 - 1.0
    return image


class SuperFutureDataset(Dataset):
    """Random-access t+3 windows over all successful SUPER shards."""

    scale = 0.528144
    state_dim = STATE_DIM
    horizon = ACTION_HORIZON

    def __init__(
        self,
        data_root: Path,
        max_windows: int | None = None,
        random_seed: int = 42,
        indices: np.ndarray | None = None,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.scene_paths = sorted(self.data_root.glob("scene_*/trajectories.h5"))
        if not self.scene_paths:
            raise FileNotFoundError(f"no scene_*/trajectories.h5 under {self.data_root}")
        self._handles: dict[int, h5py.File] = {}
        self._scene_offsets: list[int] = [0]
        self._windows: list[tuple[int, int, int]] = []
        for scene_index, path in enumerate(self.scene_paths):
            with h5py.File(path, "r") as handle:
                lidar_steps = np.asarray(handle["lidar/step"][:], dtype=np.int32)
                lidar_episode = np.asarray(
                    handle["lidar/episode_index"][:], dtype=np.int32
                )
                state_offsets = np.asarray(
                    handle["episodes/state_offset"][:], dtype=np.int64
                )
                state_lengths = np.asarray(
                    handle["episodes/state_length"][:], dtype=np.int32
                )
                lidar_offsets = np.asarray(
                    handle["episodes/lidar_offset"][:], dtype=np.int64
                )
                lidar_lengths = np.asarray(
                    handle["episodes/lidar_length"][:], dtype=np.int32
                )
                for episode_index in range(len(state_offsets)):
                    lo = int(lidar_offsets[episode_index])
                    hi = lo + int(lidar_lengths[episode_index])
                    if hi - lo < ACTION_HORIZON + 1:
                        continue
                    for local_frame in range(hi - lo - ACTION_HORIZON):
                        lidar_row = lo + local_frame
                        if int(lidar_episode[lidar_row]) != episode_index:
                            raise ValueError(f"non-contiguous lidar episode in {path}")
                        start_step = int(lidar_steps[lidar_row])
                        target_row = lo + local_frame + ACTION_HORIZON
                        target_step = int(lidar_steps[target_row])
                        state_length = int(state_lengths[episode_index])
                        if start_step + ACTION_HORIZON * ACTION_STEPS_PER_FRAME >= state_length:
                            continue
                        if target_step - start_step != ACTION_HORIZON * ACTION_STEPS_PER_FRAME:
                            continue
                        self._windows.append((scene_index, episode_index, local_frame))
            self._scene_offsets.append(len(self._windows))

        if not self._windows:
            raise RuntimeError("no valid t+3 windows in SUPER dataset")
        if max_windows is not None and len(self._windows) > int(max_windows):
            rng = np.random.default_rng(int(random_seed))
            chosen = np.sort(
                rng.choice(len(self._windows), int(max_windows), replace=False)
            )
            self._windows = [self._windows[int(index)] for index in chosen]
        self.indices = (
            np.arange(len(self._windows), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _handle(self, scene_index: int) -> h5py.File:
        handle = self._handles.get(scene_index)
        if handle is None:
            handle = h5py.File(self.scene_paths[scene_index], "r")
            self._handles[scene_index] = handle
        return handle

    def __getitem__(self, item: int):
        scene_index, episode_index, local_frame = self._windows[int(self.indices[item])]
        handle = self._handle(scene_index)
        lidar_offset = int(handle["episodes/lidar_offset"][episode_index])
        state_offset = int(handle["episodes/state_offset"][episode_index])
        lidar_row = lidar_offset + local_frame
        lidar_rows = [lidar_row + offset for offset in range(ACTION_HORIZON + 1)]
        lidar_steps = np.asarray(handle["lidar/step"][lidar_rows], dtype=np.int32)
        previous = _range_image(handle["lidar/range_m"][lidar_rows[0]])
        target = _range_image(handle["lidar/range_m"][lidar_rows[-1]])

        start_step = int(lidar_steps[0])
        action_rows = slice(
            state_offset + start_step,
            state_offset + start_step + ACTION_HORIZON * ACTION_STEPS_PER_FRAME,
        )
        velocity = np.asarray(handle["steps/super_velocity_ego"][action_rows], dtype=np.float32)
        if velocity.shape != (ACTION_HORIZON * ACTION_STEPS_PER_FRAME, 3):
            raise ValueError(f"unexpected action shape {velocity.shape}")
        actions = np.clip(
            velocity.reshape(ACTION_HORIZON, ACTION_STEPS_PER_FRAME, 3)
            / ACTION_SCALE_MPS,
            -1.0,
            1.0,
        )

        state_row = state_offset + start_step
        ego_velocity = np.asarray(handle["steps/ego_velocity"][state_row], dtype=np.float32)
        ego_acceleration = np.asarray(
            handle["steps/ego_acceleration"][state_row], dtype=np.float32
        )
        goal_ego = np.asarray(handle["steps/goal_ego"][state_row], dtype=np.float32)
        state = np.concatenate(
            (
                ego_velocity / ACTION_SCALE_MPS,
                ego_acceleration / 4.0,
                goal_ego / 25.0,
            ),
            axis=0,
        ).astype(np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"invalid state at {self.scene_paths[scene_index]}:{state_row}")

        return (
            torch.from_numpy(previous),
            torch.from_numpy(target),
            torch.from_numpy(actions),
            torch.from_numpy(state),
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


def make_train_eval_indices(length: int, eval_windows: int, seed: int) -> np.ndarray:
    count = min(int(eval_windows), int(length))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(length), count, replace=False)).astype(np.int64)
