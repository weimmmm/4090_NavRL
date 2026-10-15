"""Generate deterministic 40 m x 40 m SUPER data-collection scenes.

Each environment file is self-contained: it stores the terrain height field,
explicit obstacle boxes, and 1024 fixed start/goal pairs.  A map-only copy and
an SVG top-down preview are written below ``data/map``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


FORMAT_VERSION = 1
NUM_DRONES = 1024
FIELD_SIZE_M = 40.0
FIELD_HALF_EXTENT_M = FIELD_SIZE_M / 2.0
WORLD_HALF_EXTENT_M = 25.0
ROUTE_EDGE_M = 24.0
FLIGHT_HEIGHT_M = 2.0
HORIZONTAL_SCALE_M = 0.1
VERTICAL_SCALE_M = 0.1
GRID_SIZE = int(round(FIELD_SIZE_M / HORIZONTAL_SCALE_M))
ROUTE_NAMES = (
    "top_to_bottom",
    "bottom_to_top",
    "left_to_right",
    "right_to_left",
)

SPLITS_BY_CONDITION = {
    350: ("train",) * 5,
    150: ("train",) * 3,
    0: ("train",) * 2,
}

# Match the active HfDiscreteObstaclesTerrainCfg in training/scripts/env.py.
OBSTACLE_WIDTH_CHOICES_CELLS = np.arange(
    int(0.4 / HORIZONTAL_SCALE_M), int(1.1 / HORIZONTAL_SCALE_M), 4
)
OBSTACLE_HEIGHT_BOUNDS_M = np.asarray([1.0, 1.5, 2.0, 4.0, 6.0])
OBSTACLE_HEIGHT_PROBABILITY = np.asarray([0.10, 0.15, 0.20, 0.55])
OBSTACLE_POSITION_CHOICES = np.arange(0, GRID_SIZE, 4)


def _good_distance(
    x: int,
    y: int,
    width: int,
    length: int,
    previous: list[tuple[int, int, int, int]],
) -> bool:
    """Mirror the spacing predicate in Orbit's locally modified terrain code."""
    for xp, yp, _wp, _lp in previous:
        dx = abs(xp - x) - width
        dy = abs(yp - y) - length
        if dx < 0 and dy < 0:
            continue
        distance = math.sqrt(dx * dx + dy * dy)
        if 2 <= distance <= 10:
            return False
    return True


def generate_map(obstacle_count: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    height_field = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    grid_bounds = np.zeros((obstacle_count, 4), dtype=np.int32)
    centers = np.zeros((obstacle_count, 3), dtype=np.float32)
    sizes = np.zeros((obstacle_count, 3), dtype=np.float32)
    previous: list[tuple[int, int, int, int]] = []

    for obstacle_id in range(obstacle_count):
        height_bin = int(rng.choice(4, p=OBSTACLE_HEIGHT_PROBABILITY))
        height = float(
            rng.uniform(
                OBSTACLE_HEIGHT_BOUNDS_M[height_bin],
                OBSTACLE_HEIGHT_BOUNDS_M[height_bin + 1],
            )
        )

        for _ in range(100_000):
            width = int(rng.choice(OBSTACLE_WIDTH_CHOICES_CELLS))
            length = int(rng.choice(OBSTACLE_WIDTH_CHOICES_CELLS))
            x_start = int(rng.choice(OBSTACLE_POSITION_CHOICES))
            y_start = int(rng.choice(OBSTACLE_POSITION_CHOICES))
            x_start = min(x_start, GRID_SIZE - width)
            y_start = min(y_start, GRID_SIZE - length)
            if not previous or _good_distance(
                x_start, y_start, width, length, previous
            ):
                break
        else:
            raise RuntimeError(f"Could not place obstacle {obstacle_id}")

        previous.append((x_start, y_start, width, length))
        height_field[
            x_start : x_start + width, y_start : y_start + length
        ] = height
        width_m = width * HORIZONTAL_SCALE_M
        length_m = length * HORIZONTAL_SCALE_M
        centers[obstacle_id] = (
            -FIELD_HALF_EXTENT_M
            + (x_start + width / 2.0) * HORIZONTAL_SCALE_M,
            -FIELD_HALF_EXTENT_M
            + (y_start + length / 2.0) * HORIZONTAL_SCALE_M,
            height / 2.0,
        )
        sizes[obstacle_id] = (width_m, length_m, height)
        grid_bounds[obstacle_id] = (
            x_start,
            y_start,
            x_start + width,
            y_start + length,
        )

    return {
        "height_field_m": height_field,
        "obstacle_center_xyz": centers,
        "obstacle_size_xyz": sizes,
        "obstacle_grid_bounds_xyxy": grid_bounds,
    }


def generate_routes(seed: int, num_drones: int = NUM_DRONES) -> dict[str, np.ndarray]:
    if num_drones < 4 or num_drones % 4:
        raise ValueError("Balanced opposite-edge routes require a positive multiple of four")
    rng = np.random.RandomState(seed)
    route_id = np.arange(num_drones, dtype=np.int64) % 4
    start_lateral = rng.uniform(-ROUTE_EDGE_M, ROUTE_EDGE_M, num_drones)
    goal_lateral = rng.uniform(-ROUTE_EDGE_M, ROUTE_EDGE_M, num_drones)
    start = np.zeros((num_drones, 3), dtype=np.float32)
    goal = np.zeros((num_drones, 3), dtype=np.float32)
    start[:, 2] = FLIGHT_HEIGHT_M
    goal[:, 2] = FLIGHT_HEIGHT_M

    mask = route_id == 0
    start[mask, 0], start[mask, 1] = start_lateral[mask], ROUTE_EDGE_M
    goal[mask, 0], goal[mask, 1] = goal_lateral[mask], -ROUTE_EDGE_M
    mask = route_id == 1
    start[mask, 0], start[mask, 1] = start_lateral[mask], -ROUTE_EDGE_M
    goal[mask, 0], goal[mask, 1] = goal_lateral[mask], ROUTE_EDGE_M
    mask = route_id == 2
    start[mask, 0], start[mask, 1] = -ROUTE_EDGE_M, start_lateral[mask]
    goal[mask, 0], goal[mask, 1] = ROUTE_EDGE_M, goal_lateral[mask]
    mask = route_id == 3
    start[mask, 0], start[mask, 1] = ROUTE_EDGE_M, start_lateral[mask]
    goal[mask, 0], goal[mask, 1] = -ROUTE_EDGE_M, goal_lateral[mask]

    yaw = np.arctan2(goal[:, 1] - start[:, 1], goal[:, 0] - start[:, 0])
    orientation = np.zeros((num_drones, 4), dtype=np.float32)
    orientation[:, 0] = np.cos(yaw / 2.0)
    orientation[:, 3] = np.sin(yaw / 2.0)
    return {
        "start_position": start,
        "goal_position": goal,
        "initial_yaw": yaw.astype(np.float32),
        "initial_orientation_wxyz": orientation,
        "route_id": route_id,
    }


def map_digest(map_arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(map_arrays):
        value = np.ascontiguousarray(map_arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def tensor_dict(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in arrays.items()}


def write_svg(path: Path, scene_id: str, map_arrays: dict[str, np.ndarray]) -> None:
    canvas = 900
    margin = 54
    usable = canvas - 2 * margin

    def sx(x: float) -> float:
        return margin + (x + WORLD_HALF_EXTENT_M) / (2 * WORLD_HALF_EXTENT_M) * usable

    def sy(y: float) -> float:
        return canvas - margin - (y + WORLD_HALF_EXTENT_M) / (2 * WORLD_HALF_EXTENT_M) * usable

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas}" height="{canvas}" viewBox="0 0 {canvas} {canvas}">',
        '<rect width="100%" height="100%" fill="#f7f7f7"/>',
        f'<rect x="{sx(-25):.2f}" y="{sy(25):.2f}" width="{sx(25)-sx(-25):.2f}" height="{sy(-25)-sy(25):.2f}" fill="#e8e8e8" stroke="#333"/>',
        f'<rect x="{sx(-20):.2f}" y="{sy(20):.2f}" width="{sx(20)-sx(-20):.2f}" height="{sy(-20)-sy(20):.2f}" fill="#fff" stroke="#777" stroke-dasharray="6 4"/>',
    ]
    centers = map_arrays["obstacle_center_xyz"]
    sizes = map_arrays["obstacle_size_xyz"]
    for center, size in zip(centers, sizes):
        x0 = center[0] - size[0] / 2
        y1 = center[1] + size[1] / 2
        shade = int(210 - min(float(size[2]), 6.0) / 6.0 * 150)
        parts.append(
            f'<rect x="{sx(float(x0)):.2f}" y="{sy(float(y1)):.2f}" '
            f'width="{size[0] / 50.0 * usable:.2f}" height="{size[1] / 50.0 * usable:.2f}" '
            f'fill="rgb({shade},{shade},{shade})" stroke="#111" stroke-width="0.35"/>'
        )
    parts.extend(
        [
            f'<text x="{canvas/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{scene_id}: {len(centers)} obstacles</text>',
            f'<text x="{canvas/2}" y="{canvas-12}" text-anchor="middle" font-family="sans-serif" font-size="14">40 m obstacle field; 50 m valid area; height encoded by darkness</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def validate_scene(scene: dict) -> None:
    count = int(scene["num_drones"])
    assert count >= 4 and count % 4 == 0
    assert scene["start_position"].shape == (count, 3)
    assert scene["goal_position"].shape == (count, 3)
    assert scene["initial_orientation_wxyz"].shape == (count, 4)
    assert scene["obstacle_center_xyz"].shape[0] == scene["obstacle_count"]
    assert scene["height_field_m"].shape == (GRID_SIZE, GRID_SIZE)
    assert torch.all(scene["start_position"][:, 2] == FLIGHT_HEIGHT_M)
    assert torch.all(scene["goal_position"][:, 2] == FLIGHT_HEIGHT_M)
    assert torch.equal(torch.bincount(scene["route_id"], minlength=4), torch.full((4,), count//4))
    # Starts/goals are on +/-24 m, outside the +/-20 m obstacle field.
    assert torch.all(torch.amax(torch.abs(scene["start_position"][:, :2]), dim=1) == ROUTE_EDGE_M)
    assert torch.all(torch.amax(torch.abs(scene["goal_position"][:, :2]), dim=1) == ROUTE_EDGE_M)


def build_scene_payload(
    scene_id: str,
    obstacle_count: int,
    map_seed: int,
    route_seed: int,
    num_drones: int = NUM_DRONES,
) -> dict:
    """Build a deterministic scene in memory without writing map artifacts."""
    map_arrays = generate_map(obstacle_count, map_seed)
    route_arrays = generate_routes(route_seed, num_drones)
    scene_payload = {
        "format_version": FORMAT_VERSION,
        "scene_id": scene_id,
        "obstacle_count": obstacle_count,
        "map_seed": map_seed,
        "route_seed": route_seed,
        "map_sha256": map_digest(map_arrays),
        "route_sha256": map_digest(route_arrays),
        "field_size_m": FIELD_SIZE_M,
        "world_half_extent_m": WORLD_HALF_EXTENT_M,
        "route_edge_m": ROUTE_EDGE_M,
        "flight_height_m": FLIGHT_HEIGHT_M,
        "horizontal_scale_m": HORIZONTAL_SCALE_M,
        "vertical_scale_m": VERTICAL_SCALE_M,
        "num_drones": num_drones,
        "route_names": ROUTE_NAMES,
        **tensor_dict(map_arrays),
        **tensor_dict(route_arrays),
    }
    validate_scene(scene_payload)
    return scene_payload


def create_scene(output_dir: Path, obstacle_count: int, index: int) -> dict:
    scene_id = f"scene_{obstacle_count:03d}_{index:02d}"
    map_seed = obstacle_count * 10_000 + index * 101 + 17
    route_seed = obstacle_count * 10_000 + index * 101 + 53
    scene_payload = build_scene_payload(
        scene_id, obstacle_count, map_seed, route_seed
    )
    digest = scene_payload["map_sha256"]
    map_keys = (
        "height_field_m",
        "obstacle_center_xyz",
        "obstacle_size_xyz",
        "obstacle_grid_bounds_xyxy",
    )
    map_arrays = {key: scene_payload[key].numpy() for key in map_keys}
    map_payload = {
        key: value
        for key, value in scene_payload.items()
        if key not in {
            "num_drones",
            "route_names",
            "start_position",
            "goal_position",
            "initial_yaw",
            "initial_orientation_wxyz",
            "route_id",
        }
    }

    split = SPLITS_BY_CONDITION[obstacle_count][index]
    environment_dir = output_dir / "environments"
    map_dir = output_dir / "maps"
    environment_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    scene_path = environment_dir / f"{scene_id}.pt"
    map_path = map_dir / f"{scene_id}_map.pt"
    preview_path = map_dir / f"{scene_id}_map.svg"
    torch.save(scene_payload, scene_path)
    torch.save(map_payload, map_path)
    write_svg(preview_path, scene_id, map_arrays)

    # Verify the serialized files rather than only the in-memory payload.
    loaded_scene = torch.load(scene_path, map_location="cpu", weights_only=True)
    loaded_map = torch.load(map_path, map_location="cpu", weights_only=True)
    validate_scene(loaded_scene)
    assert loaded_scene["map_sha256"] == loaded_map["map_sha256"] == digest
    assert torch.equal(loaded_scene["height_field_m"], loaded_map["height_field_m"])
    return {
        "scene_id": scene_id,
        "split": split,
        "obstacle_count": obstacle_count,
        "num_drones": NUM_DRONES,
        "map_seed": map_seed,
        "route_seed": route_seed,
        "map_sha256": digest,
        "route_sha256": scene_payload["route_sha256"],
        "environment_file": str(scene_path.relative_to(output_dir)),
        "map_file": str(map_path.relative_to(output_dir)),
        "preview_file": str(preview_path.relative_to(output_dir)),
        "dataset": f"{split}/{scene_id}/trajectories.h5",
        "summary": f"{split}/{scene_id}/summary.json",
        "scene_bytes": scene_path.stat().st_size,
        "map_bytes": map_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in (
        "environments",
        "maps",
        "condition_manifests",
        "logs",
        "parallel_staging",
        "_interrupted",
        "train",
        "val",
        "test",
    ):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    scene_plan = ((350, 5), (150, 3), (0, 2))
    manifest = []
    for obstacle_count, count in scene_plan:
        for index in range(count):
            record = create_scene(args.output_dir, obstacle_count, index)
            manifest.append(record)
            shard_dir = args.output_dir / record["split"] / record["scene_id"]
            shard_dir.mkdir(parents=True, exist_ok=True)
            (shard_dir / "scene.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            print(
                f"[fixed-env] {record['scene_id']}: obstacles={obstacle_count}, "
                f"drones={NUM_DRONES}, sha256={record['map_sha256'][:12]}",
                flush=True,
            )

    conditions = []
    for obstacle_count, count in scene_plan:
        label = f"map40_obs{obstacle_count:03d}"
        entries = [item for item in manifest if item["obstacle_count"] == obstacle_count]
        condition = {
            "format": "navrl-super-fixed-condition-v1",
            "label": label,
            "map_size_m": FIELD_SIZE_M,
            "static_obstacles": obstacle_count,
            "num_fixed_scenes": count,
            "num_drones_per_scene": NUM_DRONES,
            "entries": entries,
        }
        conditions.append(condition)
        (args.output_dir / "condition_manifests" / f"{label}.json").write_text(
            json.dumps(condition, indent=2), encoding="utf-8"
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "navrl-super-wam-dataset-v1",
                "dataset_name": "super_wam_velocity_40m_v1",
                "format_version": FORMAT_VERSION,
                "num_scenes": len(manifest),
                "num_drones_per_scene": NUM_DRONES,
                "split_scene_counts": {
                    split: sum(item["split"] == split for item in manifest)
                    for split in ("train", "val", "test")
                },
                "conditions": [
                    {
                        "label": condition["label"],
                        "map_size_m": condition["map_size_m"],
                        "static_obstacles": condition["static_obstacles"],
                        "num_fixed_scenes": condition["num_fixed_scenes"],
                    }
                    for condition in conditions
                ],
                "scenes": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[fixed-env] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
