"""Convert NavRL static LiDAR shards into compact LaGen training caches."""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


IMAGE_KEYS = (
    "range_values",
    "prev_range_values",
    "estimate_range_in",
    "estimate_range_out",
)
MAX_ACTION_STEPS = 10
ACTION_DIMS = 3
PADDED_ELEVATION = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_manifest(dataset):
    manifest = json.loads((dataset / "dataset.json").read_text())
    require(manifest["format"] == "navrl-static-multiseed-v1", "Unsupported dataset format")
    require(manifest["static_obstacles"] == 350, "Expected 350 static obstacles")
    require(manifest["dynamic_obstacles"] == 0, "Expected a static-only dataset")
    return manifest


def create_cache(path, count, compression):
    cache = h5py.File(path, "w")
    image_options = {
        "chunks": (1, 2, 108, PADDED_ELEVATION),
        "shuffle": True,
    }
    if compression != "none":
        image_options["compression"] = compression
    for key in IMAGE_KEYS:
        cache.create_dataset(
            key, (count, 2, 108, PADDED_ELEVATION), dtype=np.float32,
            **image_options,
        )

    medium_chunk = min(count, 64)
    large_chunk = min(count, 256)
    cache.create_dataset("action_sequence", (count, MAX_ACTION_STEPS, ACTION_DIMS),
                         dtype=np.float32,
                         chunks=(medium_chunk, MAX_ACTION_STEPS, ACTION_DIMS))
    cache.create_dataset("normalized_action_sequence",
                         (count, MAX_ACTION_STEPS, ACTION_DIMS), dtype=np.float32,
                         chunks=(medium_chunk, MAX_ACTION_STEPS, ACTION_DIMS))
    cache.create_dataset("action_mask", (count, MAX_ACTION_STEPS), dtype=np.uint8,
                         chunks=(large_chunk, MAX_ACTION_STEPS))
    cache.create_dataset("prev_trans_mat", (count, 4, 4), dtype=np.float32,
                         chunks=(large_chunk, 4, 4))
    cache.create_dataset("prev_ego_feats", (count, 5), dtype=np.float32,
                         chunks=(large_chunk, 5))
    cache.create_dataset("prev_drone_state", (count, 13), dtype=np.float32,
                         chunks=(large_chunk, 13))
    cache.create_dataset("drone_state", (count, 13), dtype=np.float32,
                         chunks=(large_chunk, 13))
    cache.create_dataset("terrain_seed", (count,), dtype=np.int16)
    cache.create_dataset("frame_idx", (count,), dtype=np.int32)
    cache.create_dataset("sim_step", (count,), dtype=np.int32)
    cache.create_dataset("step_delta", (count,), dtype=np.int16)
    cache.create_dataset("done", (count,), dtype=np.uint8)
    strings = h5py.string_dtype(encoding="utf-8")
    for key in ("token", "prev_token", "scene_token", "termination_reason"):
        cache.create_dataset(key, (count,), dtype=strings)
    return cache


def encode_range_image(ranges, valid_mask, max_range):
    require(ranges.shape == (18, 108), f"Unexpected range shape: {ranges.shape}")
    require(valid_mask.shape == ranges.shape, "Range and validity shapes differ")
    require(np.isfinite(ranges).all(), "Range image contains non-finite values")
    require((ranges >= 0).all() and (ranges <= max_range + 1e-4).all(),
            "Range image is outside the configured sensor range")
    image = np.empty((2, 108, PADDED_ELEVATION), dtype=np.float32)
    image[0].fill(1.0)
    image[1].fill(-1.0)
    image[0, :, :18] = (ranges.T / max_range) * 2.0 - 1.0
    image[1, :, :18] = valid_mask.T.astype(np.float32) * 2.0 - 1.0
    return image


def project_points(points, horizontal_angles, vertical_angles, max_range):
    ranges = np.full((len(vertical_angles), len(horizontal_angles)), max_range,
                     dtype=np.float32)
    valid_mask = np.zeros_like(ranges, dtype=bool)
    if not len(points):
        return ranges, valid_mask

    point_ranges = np.linalg.norm(points, axis=1)
    horizontal_norm = np.linalg.norm(points[:, :2], axis=1)
    azimuth = np.rad2deg(np.arctan2(points[:, 1], points[:, 0]))
    elevation = np.rad2deg(np.arctan2(points[:, 2], horizontal_norm))
    horizontal_step = float(np.median(np.diff(horizontal_angles)))
    vertical_step = float(np.median(np.diff(vertical_angles)))
    in_view = (
        np.isfinite(points).all(axis=1)
        & (point_ranges > 1e-6)
        & (point_ranges <= max_range)
        & (elevation >= vertical_angles[0] - vertical_step / 2)
        & (elevation <= vertical_angles[-1] + vertical_step / 2)
    )
    if not in_view.any():
        return ranges, valid_mask

    point_ranges = point_ranges[in_view].astype(np.float32)
    azimuth = azimuth[in_view]
    elevation = elevation[in_view]
    columns = np.rint((azimuth - horizontal_angles[0]) / horizontal_step).astype(np.int64)
    columns %= len(horizontal_angles)
    rows = np.abs(elevation[:, None] - vertical_angles[None, :]).argmin(axis=1)
    flat_indices = rows * len(horizontal_angles) + columns
    flat_ranges = ranges.reshape(-1)
    np.minimum.at(flat_ranges, flat_indices, point_ranges)
    valid_mask = ranges < max_range
    return ranges, valid_mask


def warp_previous_range(previous, transform, directions, horizontal_angles,
                        vertical_angles, max_range):
    valid = previous["valid_mask"]
    points = directions[valid] * previous["ranges"][valid, None]
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    return project_points(
        transformed, horizontal_angles, vertical_angles, max_range
    )


def load_frame_data(shard, record, max_range):
    with np.load(shard / record["range_path"]) as archive:
        ranges = archive["ranges"].astype(np.float32)
        valid_mask = archive["valid_mask"].astype(bool)
        action_sequence = archive["velocity_commands"].astype(np.float32)
        normalized_actions = archive["normalized_actions"].astype(np.float32)
    require(len(action_sequence) <= MAX_ACTION_STEPS, "Action sequence exceeds cache width")
    require(action_sequence.shape == normalized_actions.shape,
            "Command and normalized-action shapes differ")
    require(action_sequence.shape[1:] == (ACTION_DIMS,), "Unexpected action dimensions")
    return {
        "ranges": ranges,
        "valid_mask": valid_mask,
        "image": encode_range_image(ranges, valid_mask, max_range),
        "action_sequence": action_sequence,
        "normalized_actions": normalized_actions,
    }


def write_transition(cache, index, record, previous_record, current, previous,
                     warped_image, empty_image, simulation_dt):
    cache["range_values"][index] = current["image"]
    cache["prev_range_values"][index] = previous["image"]
    cache["estimate_range_in"][index] = empty_image
    cache["estimate_range_out"][index] = warped_image

    action_count = len(current["action_sequence"])
    cache["action_sequence"][index] = 0
    cache["normalized_action_sequence"][index] = 0
    cache["action_mask"][index] = 0
    if action_count:
        cache["action_sequence"][index, :action_count] = current["action_sequence"]
        cache["normalized_action_sequence"][index, :action_count] = current[
            "normalized_actions"
        ]
        cache["action_mask"][index, :action_count] = 1

    cache["prev_trans_mat"][index] = np.asarray(
        record["transform_current_from_previous"], dtype=np.float32
    )
    previous_state = np.asarray(previous_record["drone_state"], dtype=np.float32)
    current_state = np.asarray(record["drone_state"], dtype=np.float32)
    step_delta = record["sim_step"] - previous_record["sim_step"]
    elapsed = step_delta * simulation_dt
    ego_features = np.zeros(5, dtype=np.float32)
    ego_features[:2] = previous_state[7:9]
    ego_features[2:4] = (current_state[7:9] - previous_state[7:9]) / elapsed
    ego_features[4] = previous_state[12]
    cache["prev_ego_feats"][index] = ego_features
    cache["prev_drone_state"][index] = previous_state
    cache["drone_state"][index] = current_state
    cache["terrain_seed"][index] = record["terrain_seed"]
    cache["frame_idx"][index] = record["frame_idx"]
    cache["sim_step"][index] = record["sim_step"]
    cache["step_delta"][index] = step_delta
    cache["done"][index] = int(record["done"])
    cache["token"][index] = record["token"]
    cache["prev_token"][index] = record["prev_token"]
    cache["scene_token"][index] = record["scene_token"]
    cache["termination_reason"][index] = record["termination_reason"] or ""


def convert_split(dataset, manifest, split, output, args):
    shards = sorted(
        (shard for shard in manifest["shards"] if shard["split"] == split),
        key=lambda shard: shard["seed"],
    )
    expected_count = sum(shard["num_transitions"] for shard in shards)
    if args.limit is not None:
        require(args.limit > 0, "--limit must be positive")
        expected_count = min(expected_count, args.limit)
    require(expected_count > 0, f"No transitions found for {split}")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it")
    if partial.exists():
        partial.unlink()

    compression = args.compression if args.compression != "none" else "none"
    cache = create_cache(partial, expected_count, compression)
    lidar = manifest["lidar"]
    max_range = float(lidar["max_range_m"])
    first_shard = dataset / shards[0]["path"]
    first_summary = json.loads((first_shard / "dataset.json").read_text())
    simulation_dt = (
        float(first_summary["sample_period_seconds"])
        / int(first_summary["sample_interval_steps"])
    )
    empty_ranges = np.full((18, 108), max_range, dtype=np.float32)
    empty_image = encode_range_image(empty_ranges, np.zeros((18, 108), bool), max_range)
    written = 0

    try:
        for shard_info in shards:
            shard = dataset / shard_info["path"]
            shard_summary = json.loads((shard / "dataset.json").read_text())
            shard_dt = (
                float(shard_summary["sample_period_seconds"])
                / int(shard_summary["sample_interval_steps"])
            )
            require(np.isclose(shard_dt, simulation_dt),
                    f"Simulation timestep differs in {shard}")
            calibration = json.loads((shard / "calibration" / "lidar.json").read_text())
            directions = np.load(shard / calibration["ray_directions_path"]).astype(np.float32)
            require(directions.shape == (108 * 18, 3),
                    f"Unexpected ray directions in {shard}")
            directions = directions.reshape(108, 18, 3).transpose(1, 0, 2)
            directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
            # The JSON labels horizontal column zero as -180 degrees, but the
            # saved ray at that column points at 0 degrees. The rays define the
            # actual range-image geometry; using the labels flips a warp by 180.
            horizontal_angles = np.rad2deg(np.unwrap(np.arctan2(
                directions[0, :, 1], directions[0, :, 0]))).astype(np.float32)
            vertical_angles = np.rad2deg(np.arctan2(
                directions[:, 0, 2], np.linalg.norm(directions[:, 0, :2], axis=1)
            )).astype(np.float32)
            require(np.all(np.diff(horizontal_angles) > 0),
                    f"Unexpected horizontal ray order in {shard}")
            require(np.all(np.diff(vertical_angles) > 0),
                    f"Unexpected vertical ray order in {shard}")

            records = [
                json.loads(line)
                for line in (shard / "metadata" / "frames.jsonl").read_text().splitlines()
            ]
            records_by_token = {record["token"]: record for record in records}
            frame_cache = {}
            for record in records:
                if written >= expected_count:
                    break
                current = load_frame_data(shard, record, max_range)
                previous_token = record["prev_token"]
                if previous_token is not None:
                    require(previous_token in frame_cache,
                            f"Previous frame was not seen before {record['token']}")
                    previous = frame_cache.pop(previous_token)
                    previous_record = records_by_token[previous_token]
                    require(previous_record["scene_token"] == record["scene_token"],
                            f"Cross-scene transition at {record['token']}")
                    require(len(current["action_sequence"]) == record["action_sequence_length"],
                            f"Action count mismatch at {record['token']}")
                    transform = np.asarray(
                        record["transform_current_from_previous"], dtype=np.float32
                    )
                    warped_ranges, warped_valid = warp_previous_range(
                        previous, transform, directions, horizontal_angles,
                        vertical_angles, max_range,
                    )
                    warped_image = encode_range_image(
                        warped_ranges, warped_valid, max_range
                    )
                    write_transition(
                        cache, written, record, previous_record, current, previous,
                        warped_image, empty_image, simulation_dt,
                    )
                    written += 1
                    if written % 1000 == 0 or written == expected_count:
                        print(f"[converter] split={split} {written}/{expected_count}", flush=True)
                frame_cache[record["token"]] = current
            if written >= expected_count:
                break

        require(written == expected_count,
                f"Expected {expected_count} transitions, wrote {written}")
        cache.attrs.update({
            "format": "navrl-lagen-h5-v1",
            "split": split,
            "num_transitions": written,
            "source_dataset": str(dataset),
            "layout": "channel_azimuth_elevation",
            "original_shape_elevation_azimuth": json.dumps([18, 108]),
            "cache_shape_azimuth_elevation": json.dumps([108, PADDED_ELEVATION]),
            "valid_elevation_columns": 18,
            "max_range_m": max_range,
            "range_normalization": "2 * range_m / max_range_m - 1",
            "mask_encoding": "valid=1, invalid=-1",
            "estimate_range_in": "empty because the scene is static",
            "estimate_range_out": "previous valid points transformed into current LiDAR frame",
            "prev_ego_feats": "previous vx, vy; finite-difference ax, ay; previous yaw rate",
            "simulation_dt_s": simulation_dt,
        })
    finally:
        cache.close()

    with h5py.File(partial, "r") as check:
        require(len(check["token"]) == expected_count, "Cache length mismatch")
        sample_indices = sorted({0, expected_count // 2, expected_count - 1})
        for index in sample_indices:
            for key in IMAGE_KEYS:
                image = check[key][index]
                require(image.shape == (2, 108, PADDED_ELEVATION),
                        f"Invalid {key} shape")
                require(np.isfinite(image).all(), f"Non-finite values in {key}")
                require((image >= -1.0001).all() and (image <= 1.0001).all(),
                        f"Values outside [-1, 1] in {key}")
            require(np.allclose(check["prev_trans_mat"][index, 3], [0, 0, 0, 1],
                                atol=1e-5), "Invalid homogeneous transform")

    if output.exists():
        output.unlink()
    os.replace(partial, output)
    result = {
        "format": "navrl-lagen-h5-v1",
        "split": split,
        "num_transitions": written,
        "path": output.name,
        "size_bytes": output.stat().st_size,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("LAGEN_CACHE_RESULT=" + json.dumps(result), flush=True)


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    manifest = load_manifest(dataset)
    output_dir = (args.output or dataset / "lagen_cache").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for split in args.splits:
        output = output_dir / f"navrl_static_{split}.h5"
        convert_split(
            dataset, manifest, split,
            output, args,
        )
        converted.append(json.loads(output.with_suffix(".json").read_text()))
    conversion_manifest = {
        "format": "navrl-lagen-cache-set-v1",
        "source_dataset": str(dataset),
        "source_format": manifest["format"],
        "image_layout": "channel_azimuth_elevation",
        "image_shape": [2, 108, PADDED_ELEVATION],
        "valid_elevation_columns": 18,
        "splits": converted,
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(conversion_manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
