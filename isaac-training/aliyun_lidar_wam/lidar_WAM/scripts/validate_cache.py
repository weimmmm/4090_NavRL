"""Validate converted NavRL LaGen HDF5 caches."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


IMAGE_KEYS = (
    "range_values",
    "prev_range_values",
    "estimate_range_in",
    "estimate_range_out",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_split(cache_root, cache_info, source_manifest, batch_size):
    path = cache_root / Path(cache_info["path"]).name
    if not path.is_file():
        path = Path(cache_info["path"])
    require(path.is_file(), f"Missing cache: {path}")

    split = cache_info["split"]
    expected = source_manifest["splits"][split]
    expected_seeds = np.asarray(expected["seeds"], dtype=np.int16)
    expected_count = int(expected["num_transitions"])
    with h5py.File(path, "r") as cache:
        require(cache.attrs["format"] == "navrl-lagen-h5-v1",
                f"Unexpected format in {path}")
        require(cache.attrs["split"] == split, f"Split mismatch in {path}")
        require(len(cache["token"]) == expected_count,
                f"Transition count mismatch in {path}")
        require(set(IMAGE_KEYS).issubset(cache), f"Missing image keys in {path}")
        for key in IMAGE_KEYS:
            require(cache[key].shape == (expected_count, 2, 108, 20),
                    f"Unexpected {key} shape in {path}")

        seeds = cache["terrain_seed"][:]
        require(np.array_equal(np.unique(seeds), expected_seeds),
                f"Terrain seeds differ in {path}")
        tokens = cache["token"].asstr()[:]
        previous_tokens = cache["prev_token"].asstr()[:]
        scene_tokens = cache["scene_token"].asstr()[:]
        require(len(set(tokens)) == expected_count, f"Duplicate tokens in {path}")
        require(np.all(tokens != previous_tokens), f"Self-linked tokens in {path}")
        require(np.all(scene_tokens != ""), f"Empty scene tokens in {path}")

        action_mask = cache["action_mask"][:]
        action_counts = action_mask.sum(axis=1)
        step_delta = cache["step_delta"][:]
        require(np.array_equal(action_counts, step_delta),
                f"Action lengths and step deltas differ in {path}")
        expected_mask = np.arange(action_mask.shape[1])[None, :] < action_counts[:, None]
        require(np.array_equal(action_mask.astype(bool), expected_mask),
                f"Non-contiguous action masks in {path}")
        require(np.isfinite(cache["prev_trans_mat"][:]).all(),
                f"Non-finite transforms in {path}")
        require(np.allclose(cache["prev_trans_mat"][:, 3], [0, 0, 0, 1], atol=1e-5),
                f"Invalid homogeneous transforms in {path}")
        require(np.isfinite(cache["prev_ego_feats"][:]).all(),
                f"Non-finite ego features in {path}")
        reasons = cache["termination_reason"].asstr()[:]
        require(np.array_equal(cache["done"][:].astype(bool), reasons != ""),
                f"Done flags and termination reasons differ in {path}")

        valid_pixels = 0
        warped_valid_pixels = 0
        pixel_count = expected_count * 108 * 18
        padding = np.asarray([1.0, -1.0], dtype=np.float32)[None, :, None, None]
        for start in range(0, expected_count, batch_size):
            stop = min(start + batch_size, expected_count)
            for key in IMAGE_KEYS:
                images = cache[key][start:stop]
                require(np.isfinite(images).all(), f"Non-finite {key} in {path}")
                require(np.logical_and(images >= -1.0001, images <= 1.0001).all(),
                        f"Out-of-range {key} in {path}")
                require(np.array_equal(images[:, :, :, 18:],
                                       np.broadcast_to(padding, images[:, :, :, 18:].shape)),
                        f"Invalid elevation padding for {key} in {path}")
            current = cache["range_values"][start:stop]
            warped = cache["estimate_range_out"][start:stop]
            empty = cache["estimate_range_in"][start:stop]
            require(np.all(empty[:, 0] == 1) and np.all(empty[:, 1] == -1),
                    f"Static estimate_range_in is not empty in {path}")
            valid_pixels += int((current[:, 1, :, :18] > 0).sum())
            warped_valid_pixels += int((warped[:, 1, :, :18] > 0).sum())

    return {
        "split": split,
        "num_transitions": expected_count,
        "terrain_seeds": expected_seeds.tolist(),
        "size_bytes": path.stat().st_size,
        "valid_fraction": valid_pixels / pixel_count,
        "warped_valid_fraction": warped_valid_pixels / pixel_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--source-dataset", type=Path,
                        help="Override the source_dataset path saved in the cache manifest")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    cache_manifest = json.loads((args.cache / "dataset.json").read_text())
    source = args.source_dataset or Path(cache_manifest["source_dataset"])
    source_manifest = json.loads((source / "dataset.json").read_text())
    results = [
        validate_split(args.cache, cache_info, source_manifest, args.batch_size)
        for cache_info in cache_manifest["splits"]
    ]
    result = {
        "format": cache_manifest["format"],
        "num_transitions": sum(item["num_transitions"] for item in results),
        "size_bytes": sum(item["size_bytes"] for item in results),
        "splits": results,
    }
    print("LAGEN_CACHE_VALIDATION=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
