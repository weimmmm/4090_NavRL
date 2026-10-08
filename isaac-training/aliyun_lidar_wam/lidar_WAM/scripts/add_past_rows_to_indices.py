"""Add three-row action history metadata to existing v2 window indices.

This is useful when a large dataset already has validated train/val indices:
the future-window linkage does not need to be rescanned just to expose the
two preceding ten-action chunks to the standalone Action Expert.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


def _decode(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def augment(old_path: Path, new_path: Path, dataset_root: Path) -> None:
    archive = np.load(old_path, allow_pickle=False)
    if "past_rows" in archive.files:
        print(f"already contains past_rows: {old_path}", flush=True)
        return
    metadata = json.loads(str(archive["metadata_json"]))
    entries = metadata.get("entries")
    if not entries:
        raise ValueError(f"index has no embedded entries: {old_path}")
    shards = np.asarray(archive["shard"], dtype=np.int64)
    rows = np.asarray(archive["rows"], dtype=np.int64)
    history = np.full((len(rows), 3), -1, dtype=np.int64)
    for shard_id in np.unique(shards):
        positions = np.flatnonzero(shards == shard_id)
        entry = entries[int(shard_id)]
        with h5py.File(dataset_root / entry["dataset"], "r") as handle:
            frames = handle["frames"] if "frames" in handle else handle
            tokens = [_decode(value) for value in frames["token"][:]]
            previous = [_decode(value) for value in frames["prev_token"][:]]
        token_to_row = {token: index for index, token in enumerate(tokens)}
        previous_rows = np.full(len(tokens), -1, dtype=np.int64)
        for index, token in enumerate(previous):
            if token:
                previous_rows[index] = token_to_row.get(token, -1)
        current = rows[positions, 0]
        one_back = previous_rows[current]
        safe_one_back = np.maximum(one_back, 0)
        two_back = np.where(one_back >= 0,
                            previous_rows[safe_one_back], -1)
        history[positions] = np.stack((two_back, one_back, current), axis=1)

    metadata["time_alignment"] = dict(metadata.get("time_alignment", {}))
    metadata["time_alignment"]["past_action_history"] = (
        "past_rows[:,0:3] oldest_to_current")
    values = {key: archive[key] for key in archive.files
              if key != "metadata_json"}
    values["past_rows"] = history
    values["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    new_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = new_path.with_suffix(new_path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, new_path)
    print(f"wrote {new_path} windows={len(rows)} history_shape={history.shape}",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--old-index", type=Path, required=True)
    parser.add_argument("--new-index", type=Path, required=True)
    args = parser.parse_args()
    augment(args.old_index, args.new_index, args.dataset_root)


if __name__ == "__main__":
    main()
