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
SPLITS = ("train", "val", "test", "unseen")


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


def _entry_identity(entry: dict[str, Any]) -> str:
    """Stable shard identity used by logical split protocols."""
    return str(entry["dataset"])


def validation_trajectory_keys(
        trajectories_by_seed: dict[int, list[tuple[str, str]]],
        fraction: float = 0.05, random_seed: int = 42,
) -> set[tuple[str, str]]:
    """Select an exact deterministic fraction of whole trajectories per seed.

    A key is ``(dataset relative path, scene token)`` so scene identifiers that
    restart in separate HDF5 shards cannot collide.  Selection is performed on
    trajectories, never windows, preventing overlapping windows from leaking
    between training and validation.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must be between zero and one")
    selected: set[tuple[str, str]] = set()
    for seed in sorted(trajectories_by_seed):
        keys = sorted(set(trajectories_by_seed[seed]))
        if not keys:
            raise ValueError(f"seed {seed} has no trajectories")
        count = max(1, int(round(len(keys) * fraction)))
        generator = np.random.default_rng(int(random_seed) + int(seed) * 100003)
        chosen = generator.choice(len(keys), size=count, replace=False)
        selected.update(keys[int(index)] for index in np.sort(chosen))
    return selected


def _trajectory_inventory(dataset_root: Path,
                          entries: list[dict[str, Any]]) -> dict[int, list[tuple[str, str]]]:
    h5py = _h5py()
    result: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for entry in entries:
        with h5py.File(Path(dataset_root) / entry["dataset"], "r") as handle:
            frames = handle["frames"] if "frames" in handle else handle
            scenes = sorted(set(_decode(frames["scene_token"][:])))
        identity = _entry_identity(entry)
        result[int(entry["seed"])].extend((identity, scene) for scene in scenes)
    return result


def entry_latent_filename(entry: dict[str, Any]) -> str:
    """Return the per-shard latent filename without breaking old manifests."""
    name = str(entry.get("latent_cache", f"seed_{int(entry['seed']):04d}.npy"))
    path = Path(name)
    if path.name != name or path.suffix != ".npy":
        raise ValueError(f"invalid latent_cache filename: {name!r}")
    return name


def index_path(index_root: Path, split: str) -> Path:
    return Path(index_root) / f"{split}_windows.npz"


def random_window_subset(selection: np.ndarray, limit: int | None,
                         random_seed: int) -> np.ndarray:
    """Choose at most ``limit`` windows globally and reproducibly."""
    selection = np.asarray(selection, dtype=np.int64)
    if limit is None or len(selection) <= int(limit):
        return selection
    if int(limit) <= 0:
        raise ValueError("window sample limit must be positive")
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(selection, int(limit), replace=False))


def _turn_score(actions: np.ndarray, past: np.ndarray | None) -> float:
    flat = actions.reshape(-1, 3)
    differences = np.diff(flat, axis=0)
    score = float(np.linalg.norm(differences, axis=-1).max(initial=0.0))
    if past is not None:
        score = max(score, float(np.linalg.norm(flat[0] - past[-1])))
    return score


def build_split_index(dataset_root: Path, split: str, index_root: Path,
                      strict_expected: bool = True,
                      entries_override: list[dict[str, Any]] | None = None,
                      include_trajectories: set[tuple[str, str]] | None = None,
                      exclude_trajectories: set[tuple[str, str]] | None = None,
                      protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Scan token linkage and save compact, deterministic t+3 window metadata."""
    h5py = _h5py()
    dataset_root, index_root = Path(dataset_root), Path(index_root)
    entries = (split_entries(dataset_root, split) if entries_override is None
               else [dict(row) for row in entries_override])
    if not entries:
        raise ValueError(f"no dataset entries for split {split!r}")
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
            trajectory_key = (_entry_identity(entry), scenes[current])
            if (include_trajectories is not None
                    and trajectory_key not in include_trajectories):
                continue
            if (exclude_trajectories is not None
                    and trajectory_key in exclude_trajectories):
                continue
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
    manifest = read_manifest(dataset_root)
    expected_map = manifest.get("expected_windows", EXPECTED_WINDOWS)
    expected = expected_map.get(split) if isinstance(expected_map, dict) else None
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
        # Logical protocols can reuse one physical shard in train and val.  The
        # loader must therefore use the exact entry table embedded in the index
        # instead of deriving it from the manifest's original split labels.
        "entries": entries,
        "protocol": protocol,
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
    available = {row["split"] for row in read_manifest(dataset_root)["entries"]}
    return {split: build_split_index(dataset_root, split, index_root, strict_expected)
            for split in SPLITS if split in available}


def build_seed_holdout_indices(
        dataset_root: Path, index_root: Path, validation_fraction: float = 0.05,
        random_seed: int = 42, train_seeds=range(8), unseen_seeds=(8, 9),
) -> dict[str, Any]:
    """Build train/val/unseen indices without copying HDF5 or latent shards."""
    manifest = read_manifest(dataset_root)
    entries = [dict(row) for row in manifest["entries"]]
    train_seed_set = {int(value) for value in train_seeds}
    unseen_seed_set = {int(value) for value in unseen_seeds}
    overlap = train_seed_set & unseen_seed_set
    if overlap:
        raise ValueError(f"train and unseen seeds overlap: {sorted(overlap)}")
    train_entries = [row for row in entries if int(row["seed"]) in train_seed_set]
    unseen_entries = [row for row in entries if int(row["seed"]) in unseen_seed_set]
    present_train = {int(row["seed"]) for row in train_entries}
    present_unseen = {int(row["seed"]) for row in unseen_entries}
    if present_train != train_seed_set:
        raise ValueError(f"missing train seeds: {sorted(train_seed_set-present_train)}")
    if present_unseen != unseen_seed_set:
        raise ValueError(f"missing unseen seeds: {sorted(unseen_seed_set-present_unseen)}")

    inventory = _trajectory_inventory(dataset_root, train_entries)
    held_out = validation_trajectory_keys(
        inventory, fraction=validation_fraction, random_seed=random_seed)
    trajectory_counts = {str(seed): len(set(keys))
                         for seed, keys in sorted(inventory.items())}
    validation_counts = {
        str(seed): sum(key in held_out for key in set(keys))
        for seed, keys in sorted(inventory.items())
    }
    protocol = {
        "name": "seed-0-7-trajectory-holdout-v1",
        "train_seeds": sorted(train_seed_set),
        "unseen_seeds": sorted(unseen_seed_set),
        "validation_fraction": float(validation_fraction),
        "selection_random_seed": int(random_seed),
        "trajectory_counts_by_seed": trajectory_counts,
        "validation_trajectory_counts_by_seed": validation_counts,
        "validation_unit": "whole scene_token trajectory",
    }
    results = {
        "train": build_split_index(
            dataset_root, "train", index_root, strict_expected=False,
            entries_override=train_entries, exclude_trajectories=held_out,
            protocol={**protocol, "role": "train"}),
        "val": build_split_index(
            dataset_root, "val", index_root, strict_expected=False,
            entries_override=train_entries, include_trajectories=held_out,
            protocol={**protocol, "role": "in_distribution_validation"}),
        "unseen": build_split_index(
            dataset_root, "unseen", index_root, strict_expected=False,
            entries_override=unseen_entries,
            protocol={**protocol, "role": "held_out_seed_generalization"}),
    }
    return results


class V2WindowDataset(Dataset):
    """Lazy HDF5/memmap access to aligned Action Expert/world-model windows."""

    def __init__(self, split: str, dataset_root: Path, latent_root: Path,
                 index_root: Path, overfit: bool = False,
                 samples_per_seed: int | None = None, random_seed: int = 42,
                 limit: int | None = None, scene_min: int | None = None,
                 scene_max: int | None = None,
                 all_future_targets: bool = False):
        self.split = split
        self.dataset_root = Path(dataset_root)
        self.latent_root = Path(latent_root)
        self.index_root = Path(index_root)
        self.all_future_targets = bool(all_future_targets)
        archive = np.load(index_path(self.index_root, split), allow_pickle=False)
        self.metadata = json.loads(str(archive["metadata_json"]))
        if self.metadata.get("format") != FORMAT:
            raise ValueError("incompatible v2 window index")
        if self.metadata.get("dataset_manifest_sha256") != file_sha256(
                self.dataset_root / "manifest.json"):
            raise ValueError("window index belongs to another dataset manifest")
        indexed_entries = self.metadata.get("entries")
        self.entries = [dict(row) for row in (
            indexed_entries if indexed_entries is not None
            else split_entries(self.dataset_root, split))]
        if not self.entries:
            raise ValueError(f"index {split!r} has no embedded or manifest entries")
        all_shards = np.asarray(archive["shard"])
        all_seeds = np.asarray([
            int(self.entries[int(shard_id)]["seed"]) for shard_id in all_shards
        ], dtype=np.int16)
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
            pieces = []
            for seed in np.unique(all_seeds[selection]):
                candidates = selection[all_seeds[selection] == seed]
                count = min(int(samples_per_seed), len(candidates))
                pieces.append(np.sort(rng.choice(candidates, count, replace=False)))
            selection = np.concatenate(pieces)
        selection = random_window_subset(selection, limit, random_seed)
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
        self.seeds = all_seeds[selection]
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
            latent_path = self.latent_root / entry_latent_filename(
                self.entries[shard_id])
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
        target_rows = future_rows if self.all_future_targets else future_rows[-1:]
        target_latent = np.asarray(latent[target_rows], np.float32)
        target_image = np.asarray(h5["range_values"][target_rows], np.float32)
        if not self.all_future_targets:
            target_latent = target_latent[0]
            target_image = target_image[0]
        return (
            torch.from_numpy(np.asarray(latent[current], np.float32)),
            torch.from_numpy(target_latent),
            torch.from_numpy(future), self.world_state[index],
            torch.from_numpy(target_image),
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
