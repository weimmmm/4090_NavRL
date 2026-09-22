"""Shard-native loader for ``navrl-isaac-trajectory-hdf5-v2`` datasets.

Rows store the actions that arrived at that row.  Consequently a policy
observation at row ``i`` uses row ``i`` as its past action and the three token
successors as its 30-step supervision.  This module makes that convention
explicit and rejects ambiguous or cross-scene chains.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from lidar_wam.coordinates import numpy_goal_frame_causal_features


FORMAT = "navrl-wam-window-index-v3"
EXPECTED_WINDOWS = {"train": 480792, "val": 62792, "test": 61327}
SPLITS = ("train", "val", "test")


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for the v2 trajectory dataset") from exc
    return h5py


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(values) -> list[str]:
    return [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values]


def read_manifest(dataset_root: Path) -> dict[str, Any]:
    path = Path(dataset_root) / "manifest.json"
    value = json.loads(path.read_text())
    if value.get("format") != "navrl-isaac-dataset-manifest-v2":
        raise ValueError(f"unsupported dataset manifest: {value.get('format')!r}")
    return value


def split_entries(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return [row for row in read_manifest(dataset_root)["entries"]
            if row["split"] == split]


def index_path(index_root: Path, split: str) -> Path:
    return Path(index_root) / f"{split}_windows.npz"


def _turn_score(actions: np.ndarray, past: np.ndarray | None) -> float:
    flat = actions.reshape(-1, 3)
    differences = np.diff(flat, axis=0)
    score = float(np.linalg.norm(differences, axis=-1).max(initial=0.0))
    if past is not None:
        score = max(score, float(np.linalg.norm(flat[0] - past[-1])))
    return score


def build_split_index(dataset_root: Path, split: str, index_root: Path,
                      strict_expected: bool = True) -> dict[str, Any]:
    """Scan token linkage and save compact, deterministic t+3 window metadata."""
    h5py = _h5py()
    dataset_root, index_root = Path(dataset_root), Path(index_root)
    entries = split_entries(dataset_root, split)
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    rejection: Counter[str] = Counter()
    shard_rows = []

    for shard_id, entry in enumerate(entries):
        path = dataset_root / entry["dataset"]
        with h5py.File(path, "r") as h5:
            frames = h5["frames"] if "frames" in h5 else h5
            tokens = _decode(frames["token"][:])
            previous = _decode(frames["prev_token"][:])
            following = _decode(frames["next_token"][:])
            scenes = _decode(frames["scene_token"][:])
            frame_index = np.asarray(frames["frame_index"][:], np.int64)
            scene_id = np.asarray(frames["scene_id"][:], np.int32)
            step_delta = np.asarray(frames["step_delta"][:], np.int16)
            action_mask = np.asarray(frames["action_mask"][:], bool)
            collision = np.asarray(frames["collision"][:], bool)
            oob = np.asarray(frames["out_of_bounds"][:], bool)
            action = np.asarray(frames["normalized_action_sequence"][:], np.float32)
            state = np.asarray(frames["drone_state"][:], np.float32)
            target = np.asarray(frames["target_position"][:], np.float32)
            direction = np.asarray(frames["target_dir_2d"][:], np.float32)
            clearance = np.asarray(frames["min_lidar_clearance"][:], np.float32)
        all_goal, all_proprio = numpy_goal_frame_causal_features(
            state, target, direction)

        if len(tokens) != len(set(tokens)):
            raise ValueError(f"duplicate token in {path}")
        token_to_row = {token: row for row, token in enumerate(tokens)}
        if any(previous[row] and token_to_row.get(previous[row]) is None
               for row in range(len(tokens))):
            raise ValueError(f"dangling prev_token in {path}")

        rows, goals, proprios, world_states = [], [], [], []
        past_valid, clearances, turns, scene_ids = [], [], [], []
        for current in range(len(tokens)):
            chain = [current]
            for _ in range(3):
                token = following[chain[-1]]
                if not token or token not in token_to_row:
                    break
                chain.append(token_to_row[token])
            if len(chain) != 4:
                rejection["short_chain"] += 1
                continue
            future = chain[1:]
            if any(scenes[row] != scenes[current] for row in chain):
                rejection["scene_boundary"] += 1
                continue
            if any(frame_index[row] != frame_index[current] + offset
                   for offset, row in enumerate(chain)):
                rejection["nonconsecutive_frame"] += 1
                continue
            if any(step_delta[row] != 10 for row in future):
                rejection["wrong_step_delta"] += 1
                continue
            if any(not action_mask[row].all() for row in future):
                rejection["invalid_action_mask"] += 1
                continue
            future_action = action[future]
            if not np.isfinite(future_action).all():
                rejection["nonfinite_action"] += 1
                continue
            if collision[future].any() or oob[future].any():
                rejection["collision_or_oob_in_target"] += 1
                continue
            if not (np.isfinite(state[current]).all()
                    and np.isfinite(target[current]).all()
                    and np.isfinite(direction[current]).all()):
                rejection["nonfinite_condition"] += 1
                continue

            previous_is_valid = bool(
                step_delta[current] == 10 and action_mask[current].all()
                and np.isfinite(action[current]).all() and previous[current])
            previous_action = action[current] if previous_is_valid else None
            goal_velocity = all_proprio[current, 1:4]
            angular_velocity = all_proprio[current, 4:7]
            rows.append(chain)
            goals.append(all_goal[current])
            proprios.append(all_proprio[current])
            # Preserve the old five-dimensional world-condition interface while
            # removing its future-derived acceleration leakage.
            world_states.append([
                goal_velocity[0], goal_velocity[1], 0.0, 0.0,
                angular_velocity[2]])
            past_valid.append(previous_is_valid)
            clearances.append(float(clearance[current]))
            turns.append(_turn_score(future_action, previous_action))
            scene_ids.append(int(scene_id[current]))

        count = len(rows)
        arrays["shard"].append(np.full(count, shard_id, np.int16))
        arrays["rows"].append(np.asarray(rows, np.int64))
        arrays["goal"].append(np.asarray(goals, np.float32))
        arrays["proprio"].append(np.asarray(proprios, np.float32))
        arrays["world_state"].append(np.asarray(world_states, np.float32))
        arrays["past_valid"].append(np.asarray(past_valid, bool))
        arrays["clearance"].append(np.asarray(clearances, np.float32))
        arrays["turn_score"].append(np.asarray(turns, np.float32))
        arrays["scene_id"].append(np.asarray(scene_ids, np.int32))
        shard_rows.append({"shard_id": shard_id, "seed": int(entry["seed"]),
                           "dataset": entry["dataset"], "windows": count})

    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    expected = EXPECTED_WINDOWS.get(split)
    # The published counts include all geometrically valid chains.  Excluding
    # collision/OOB target chunks is a deliberate training filter, so report
    # both counts and only enforce the unfiltered invariant in ``inspect``.
    filtered_count = len(merged["rows"])
    unfiltered_count = filtered_count + rejection["collision_or_oob_in_target"]
    if strict_expected and expected is not None and unfiltered_count != expected:
        raise ValueError(
            f"{split}: expected {expected} linked windows before failure filtering, "
            f"found {unfiltered_count}")

    if split == "train":
        clearance_q25 = float(np.quantile(merged["clearance"], 0.25))
        turn_q75 = float(np.quantile(merged["turn_score"], 0.75))
    else:
        clearance_q25 = turn_q75 = None
    metadata = {
        "format": FORMAT, "split": split,
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "expected_unfiltered_windows": expected,
        "unfiltered_windows": unfiltered_count,
        "windows": filtered_count,
        "failure_target_windows_excluded": rejection["collision_or_oob_in_target"],
        "rejection": dict(rejection), "shards": shard_rows,
        "clearance_q25": clearance_q25, "turn_score_q75": turn_q75,
        "time_alignment": {
            "observation": "rows[:,0]", "past_actions": "rows[:,0]",
            "future_actions": "rows[:,1:4]", "world_target": "rows[:,3]",
        },
    }
    index_root.mkdir(parents=True, exist_ok=True)
    destination = index_path(index_root, split)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **merged,
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    os.replace(temporary, destination)
    return metadata


def build_all_indices(dataset_root: Path, index_root: Path,
                      strict_expected: bool = True) -> dict[str, Any]:
    return {split: build_split_index(dataset_root, split, index_root, strict_expected)
            for split in SPLITS}


class V2WindowDataset(Dataset):
    """Lazy HDF5/memmap access to aligned Action Expert/world-model windows."""

    def __init__(self, split: str, dataset_root: Path, latent_root: Path,
                 index_root: Path, overfit: bool = False,
                 samples_per_seed: int | None = None, random_seed: int = 42,
                 limit: int | None = None, scene_min: int | None = None,
                 scene_max: int | None = None):
        self.split = split
        self.dataset_root = Path(dataset_root)
        self.latent_root = Path(latent_root)
        self.index_root = Path(index_root)
        archive = np.load(index_path(self.index_root, split), allow_pickle=False)
        self.metadata = json.loads(str(archive["metadata_json"]))
        if self.metadata.get("format") != FORMAT:
            raise ValueError("incompatible v2 window index")
        if self.metadata.get("dataset_manifest_sha256") != file_sha256(
                self.dataset_root / "manifest.json"):
            raise ValueError("window index belongs to another dataset manifest")
        self.entries = split_entries(self.dataset_root, split)
        selection = np.arange(len(archive["rows"]), dtype=np.int64)
        if overfit:
            shard = np.asarray(archive["shard"])
            scene = np.asarray(archive["scene_id"])
            selection = selection[(shard == 0) & (scene < 128)]
        if scene_min is not None:
            scene = np.asarray(archive["scene_id"])
            selection = selection[scene[selection] >= int(scene_min)]
        if scene_max is not None:
            scene = np.asarray(archive["scene_id"])
            selection = selection[scene[selection] < int(scene_max)]
        elif samples_per_seed is not None:
            rng = np.random.default_rng(random_seed)
            shard = np.asarray(archive["shard"])
            pieces = []
            for shard_id in np.unique(shard):
                candidates = selection[shard == shard_id]
                count = min(int(samples_per_seed), len(candidates))
                pieces.append(np.sort(rng.choice(candidates, count, replace=False)))
            selection = np.concatenate(pieces)
        if limit is not None and len(selection) > int(limit):
            rng = np.random.default_rng(random_seed)
            selection = np.sort(rng.choice(selection, int(limit), replace=False))
        self.shard = np.asarray(archive["shard"])[selection]
        self.rows = np.asarray(archive["rows"])[selection]
        self.goal = torch.from_numpy(np.asarray(archive["goal"])[selection].copy())
        self.proprio = torch.from_numpy(np.asarray(archive["proprio"])[selection].copy())
        self.world_state = torch.from_numpy(
            np.asarray(archive["world_state"])[selection].copy())
        self.past_valid = np.asarray(archive["past_valid"])[selection]
        self.clearance = np.asarray(archive["clearance"])[selection]
        self.turn_score = np.asarray(archive["turn_score"])[selection]
        self.scene_id = np.asarray(archive["scene_id"])[selection]
        self.seeds = np.asarray([
            int(self.entries[int(shard_id)]["seed"]) for shard_id in self.shard
        ], dtype=np.int16)
        self.source_indices = np.concatenate(
            (self.shard[:, None].astype(np.int64), self.rows), axis=1)
        self._h5: dict[int, Any] = {}
        self._latents: dict[int, np.ndarray] = {}
        latent_meta = json.loads((self.latent_root / "metadata.json").read_text())
        if latent_meta.get("dataset_manifest_sha256") != self.metadata[
                "dataset_manifest_sha256"]:
            raise ValueError("latent cache belongs to another dataset manifest")
        self.latent_metadata = latent_meta
        # Cache values are already scaled.  ``scale`` remains the VAE scaling
        # factor needed to decode them via ``vae.decode(latent / scale)``.
        self.scale = float(latent_meta["scaling_factor"])

    def __getstate__(self):
        value = dict(self.__dict__)
        value["_h5"] = {}
        value["_latents"] = {}
        return value

    def _handles(self, shard_id: int):
        if shard_id not in self._h5:
            h5py = _h5py()
            path = self.dataset_root / self.entries[shard_id]["dataset"]
            handle = h5py.File(path, "r")
            self._h5[shard_id] = handle["frames"] if "frames" in handle else handle
            latent_path = self.latent_root / f"seed_{int(self.entries[shard_id]['seed']):04d}.npy"
            self._latents[shard_id] = np.load(latent_path, mmap_mode="r")
        return self._h5[shard_id], self._latents[shard_id]

    def __len__(self):
        return len(self.rows)

    def current_range(self, index: int) -> torch.Tensor:
        """Return the physical two-channel LiDAR image at the causal input row."""
        shard_id = int(self.shard[index])
        current = int(self.rows[index, 0])
        h5, _ = self._handles(shard_id)
        return torch.from_numpy(np.asarray(
            h5["range_values"][current], dtype=np.float32))

    def __getitem__(self, index):
        shard_id = int(self.shard[index])
        chain = [int(v) for v in self.rows[index]]
        current, future_rows = chain[0], chain[1:]
        h5, latent = self._handles(shard_id)
        future = np.asarray(h5["normalized_action_sequence"][future_rows],
                            dtype=np.float32)
        if self.past_valid[index]:
            past = np.asarray(h5["normalized_action_sequence"][current], np.float32)
            past_mask = np.ones(10, np.float32)
        else:
            past = np.zeros((10, 3), np.float32)
            past_mask = np.zeros(10, np.float32)
        return (
            torch.from_numpy(np.asarray(latent[current], np.float32)),
            torch.from_numpy(np.asarray(latent[future_rows[-1]], np.float32)),
            torch.from_numpy(future), self.world_state[index],
            torch.from_numpy(np.asarray(
                h5["range_values"][future_rows[-1]], np.float32)),
            torch.from_numpy(self.source_indices[index]), self.goal[index],
            self.proprio[index], torch.from_numpy(past),
            torch.from_numpy(past_mask),
        )

    def close(self):
        for frames in self._h5.values():
            frames.file.close()
        self._h5.clear()
        self._latents.clear()


class StratifiedSampler(Sampler[int]):
    """Draw 50% uniform, 25% low-clearance and 25% high-turn windows."""

    def __init__(self, dataset: V2WindowDataset, num_samples: int,
                 seed: int = 42, rank: int = 0):
        self.num_samples = int(num_samples)
        self.seed = int(seed) + int(rank) * 100003
        clearance_q25 = float(dataset.metadata["clearance_q25"])
        turn_q75 = float(dataset.metadata["turn_score_q75"])
        self.all = torch.arange(len(dataset), dtype=torch.long)
        self.low = torch.from_numpy(np.flatnonzero(dataset.clearance <= clearance_q25))
        self.turn = torch.from_numpy(np.flatnonzero(dataset.turn_score >= turn_q75))
        if not len(self.low) or not len(self.turn):
            raise ValueError("empty hard-example stratum")
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        counts = (self.num_samples // 2, self.num_samples // 4)
        pieces = [
            self.all[torch.randint(len(self.all), (counts[0],), generator=generator)],
            self.low[torch.randint(len(self.low), (counts[1],), generator=generator)],
        ]
        final = self.num_samples - sum(len(piece) for piece in pieces)
        pieces.append(self.turn[torch.randint(len(self.turn), (final,), generator=generator)])
        order = torch.randperm(self.num_samples, generator=generator)
        self.epoch += 1
        return iter(torch.cat(pieces)[order].tolist())


def compute_condition_stats(dataset: V2WindowDataset) -> dict[str, Any]:
    goal = dataset.goal.numpy()
    proprio = dataset.proprio.numpy()
    # Stream one shard at a time so this does not materialize ~200 MB twice or
    # issue hundreds of thousands of tiny HDF5 reads.
    action_sum = np.zeros(3, np.float64)
    action_sq = np.zeros(3, np.float64)
    logit_sum = np.zeros(3, np.float64)
    logit_sq = np.zeros(3, np.float64)
    count = 0
    for shard_id in np.unique(dataset.shard):
        h5, _ = dataset._handles(int(shard_id))
        all_actions = np.asarray(h5["normalized_action_sequence"][:], np.float32)
        rows = dataset.rows[dataset.shard == shard_id, 1:]
        value = all_actions[rows].reshape(-1, 3)
        clipped = np.clip(value, 1e-4, 1 - 1e-4)
        logit = np.log(clipped) - np.log1p(-clipped)
        action_sum += value.sum(0); action_sq += np.square(value).sum(0)
        logit_sum += logit.sum(0); logit_sq += np.square(logit).sum(0)
        count += len(value)
    action_mean = action_sum / count
    logit_mean = logit_sum / count
    logit_std = np.sqrt(np.maximum(logit_sq / count - np.square(logit_mean), 1e-8))
    return {
        "action_logit_mean": logit_mean.tolist(),
        "action_logit_std": np.maximum(logit_std, 1e-4).tolist(),
        "action_raw_mean": action_mean.tolist(),
        "goal_mean": goal.mean(0).tolist(),
        "goal_std": np.maximum(goal.std(0), 1e-4).tolist(),
        "proprio_mean": proprio.mean(0).tolist(),
        "proprio_std": np.maximum(proprio.std(0), 1e-4).tolist(),
        "logit_clip_epsilon": 1e-4,
        "clearance_q25": float(dataset.metadata["clearance_q25"]),
        "turn_score_q75": float(dataset.metadata["turn_score_q75"]),
    }
