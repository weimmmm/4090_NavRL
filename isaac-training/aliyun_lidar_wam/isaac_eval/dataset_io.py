"""Streaming HDF5 storage and validation for complete navigation episodes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DATASET_FORMAT = "navrl-isaac-trajectory-hdf5-v2"
TERMINATION_REASONS = ("reach_goal", "collision", "out_of_bounds", "timeout")

ARRAY_FIELDS = {
    "range_values": (np.float32, (2, 108, 20)),
    "policy_ranges": (np.float32, (1, 6, 36)),
    "drone_state": (np.float32, (13,)),
    "policy_state": (np.float32, (8,)),
    "target_position": (np.float32, (3,)),
    "target_dir_2d": (np.float32, (3,)),
    "normalized_action_sequence": (np.float32, (10, 3)),
    "world_action_sequence": (np.float32, (10, 3)),
    "action_mask": (np.uint8, (10,)),
}
SCALAR_FIELDS = {
    "scene_id": np.int32, "env_id": np.int32, "frame_index": np.int32,
    "sim_step": np.int32, "timestamp_us": np.int64, "step_delta": np.int16,
    "collision": np.uint8, "reach_goal": np.uint8,
    "out_of_bounds": np.uint8, "timeout": np.uint8,
    "min_lidar_clearance": np.float32, "distance_to_goal": np.float32,
}
STRING_FIELDS = {"token": 64, "scene_token": 32, "prev_token": 64,
                 "next_token": 64, "termination_reason": 20}


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required inside the Isaac collection environment") from exc
    return h5py


class TrajectoryWriter:
    """Append frame batches without retaining the full dataset in memory."""

    def __init__(self, path: Path, metadata: dict[str, Any], compression: str = "lzf"):
        h5py = _h5py()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite existing dataset: {self.path}")
        self.file = h5py.File(self.path, "w")
        self.file.attrs["format"] = DATASET_FORMAT
        self.file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        self.frames = self.file.create_group("frames")
        for name, (dtype, tail) in ARRAY_FIELDS.items():
            self.frames.create_dataset(
                name, shape=(0, *tail), maxshape=(None, *tail), dtype=dtype,
                chunks=(min(128, max(1, 1048576 // (np.dtype(dtype).itemsize * int(np.prod(tail))))), *tail),
                compression=compression)
        for name, dtype in SCALAR_FIELDS.items():
            self.frames.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                       chunks=(4096,), compression=compression)
        for name, width in STRING_FIELDS.items():
            self.frames.create_dataset(name, shape=(0,), maxshape=(None,), dtype=f"S{width}",
                                       chunks=(4096,), compression=compression)
        # Compatibility aliases do not duplicate data.  New code can use the
        # explicit world-action name; existing WAM tooling can use action_sequence
        # and root-level dataset names.
        self.frames["action_sequence"] = self.frames["world_action_sequence"]
        for name, dataset in self.frames.items():
            self.file[name] = dataset
        self.count = 0
        self.last_row: dict[int, int] = {}
        self.last_token: dict[int, str] = {}

    def append(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        begin, end = self.count, self.count + len(rows)
        for dataset in self.frames.values():
            dataset.resize(end, axis=0)
        for name in ARRAY_FIELDS:
            self.frames[name][begin:end] = np.stack([row[name] for row in rows])
        for name in SCALAR_FIELDS:
            self.frames[name][begin:end] = np.asarray([row[name] for row in rows])

        tokens, previous = [], []
        for offset, row in enumerate(rows):
            scene_id = int(row["scene_id"])
            token = str(row["token"])
            previous_token = self.last_token.get(scene_id, "")
            tokens.append(token)
            previous.append(previous_token)
            if scene_id in self.last_row:
                self.frames["next_token"][self.last_row[scene_id]] = token.encode()
            self.last_row[scene_id] = begin + offset
            self.last_token[scene_id] = token
        strings = {
            "token": tokens,
            "scene_token": [row["scene_token"] for row in rows],
            "prev_token": previous,
            "next_token": ["" for _ in rows],
            "termination_reason": [row.get("termination_reason", "") for row in rows],
        }
        for name, values in strings.items():
            self.frames[name][begin:end] = np.asarray(values, dtype=self.frames[name].dtype)
        self.count = end

    def close(self, summary: dict[str, Any] | None = None) -> None:
        if self.file is None:
            return
        if summary is not None:
            self.file.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        self.file.flush()
        self.file.close()
        self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def validate_dataset(dataset_path: Path, environment: dict[str, Any],
                     expected_scenes: int | None = None,
                     sample_interval: int = 10) -> dict[str, Any]:
    """Perform structural, linkage, route, and terminal-completeness checks."""
    h5py = _h5py()
    dataset_path = Path(dataset_path)
    with h5py.File(dataset_path, "r") as data:
        if data.attrs.get("format") != DATASET_FORMAT:
            raise ValueError(f"unexpected dataset format in {dataset_path}")
        frames = data["frames"]
        required = set(ARRAY_FIELDS) | set(SCALAR_FIELDS) | set(STRING_FIELDS)
        missing = sorted(required - set(frames.keys()))
        if missing:
            raise ValueError(f"missing frame datasets: {missing}")
        counts = {name: len(frames[name]) for name in required}
        if len(set(counts.values())) != 1:
            raise ValueError(f"inconsistent HDF5 lengths: {counts}")
        total = next(iter(counts.values()))
        scene_ids = frames["scene_id"][:].astype(np.int64)
        unique = np.unique(scene_ids)
        expected = int(environment["num_envs"] if expected_scenes is None else expected_scenes)
        if len(unique) != expected or not np.array_equal(unique, np.arange(expected)):
            raise ValueError(f"expected scene ids 0..{expected - 1}, found {unique.tolist()}")
        reasons = Counter()
        episode_steps, frame_counts = [], []
        starts = environment["start_positions"].detach().cpu().numpy()[:, 0]
        targets = environment["target_positions"].detach().cpu().numpy()[:, 0]
        for scene_id in unique:
            indices = np.flatnonzero(scene_ids == scene_id)
            frame_index = frames["frame_index"][indices]
            sim_steps = frames["sim_step"][indices]
            deltas = frames["step_delta"][indices]
            masks = frames["action_mask"][indices]
            if not np.array_equal(frame_index, np.arange(len(indices))):
                raise ValueError(f"scene {scene_id}: non-contiguous frame indices")
            if sim_steps[0] != 0 or deltas[0] != 0 or masks[0].any():
                raise ValueError(f"scene {scene_id}: invalid initial frame")
            if len(indices) > 2 and not np.all(deltas[1:-1] == sample_interval):
                raise ValueError(f"scene {scene_id}: non-terminal step_delta is not {sample_interval}")
            if not 1 <= int(deltas[-1]) <= sample_interval:
                raise ValueError(f"scene {scene_id}: invalid terminal step_delta {deltas[-1]}")
            if not np.array_equal(masks.sum(axis=1), deltas):
                raise ValueError(f"scene {scene_id}: action_mask does not match step_delta")
            if not np.all(np.diff(sim_steps) == deltas[1:]):
                raise ValueError(f"scene {scene_id}: sim-step deltas are inconsistent")
            reason_values = [x.decode() for x in frames["termination_reason"][indices]]
            if any(reason_values[:-1]) or reason_values[-1] not in TERMINATION_REASONS:
                raise ValueError(f"scene {scene_id}: incomplete or multiple termination records")
            reason = reason_values[-1]
            flags = {name: bool(frames[name][indices[-1]]) for name in TERMINATION_REASONS}
            if sum(flags.values()) != 1 or not flags[reason]:
                raise ValueError(f"scene {scene_id}: terminal flag/reason mismatch")
            tokens = [x.decode() for x in frames["token"][indices]]
            prev = [x.decode() for x in frames["prev_token"][indices]]
            following = [x.decode() for x in frames["next_token"][indices]]
            if prev != [""] + tokens[:-1] or following != tokens[1:] + [""]:
                raise ValueError(f"scene {scene_id}: broken or cross-scene token linkage")
            first_state = frames["drone_state"][indices[0]]
            first_target = frames["target_position"][indices[0]]
            if not np.array_equal(first_state[:3], starts[scene_id]):
                raise ValueError(f"scene {scene_id}: start differs from environment snapshot")
            if not np.array_equal(first_target, targets[scene_id]):
                raise ValueError(f"scene {scene_id}: target differs from environment snapshot")
            reasons[reason] += 1
            episode_steps.append(int(sim_steps[-1]))
            frame_counts.append(len(indices))
        return {
            "dataset": str(dataset_path), "frames": int(total),
            "scenes": int(len(unique)), "termination_counts": dict(reasons),
            "mean_episode_steps": float(np.mean(episode_steps)),
            "min_episode_steps": int(np.min(episode_steps)),
            "max_episode_steps": int(np.max(episode_steps)),
            "mean_frames_per_scene": float(np.mean(frame_counts)),
            "size_bytes": dataset_path.stat().st_size,
        }
