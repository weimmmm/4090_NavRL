"""Self-contained, reproducible Isaac navigation environment snapshots."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ENVIRONMENT_FORMAT_V1 = "navrl-isaac-eval-environment-v1"
ENVIRONMENT_FORMAT_V2 = "navrl-isaac-environment-v2"
# Backwards compatibility for the historical recipe-only creator.
ENVIRONMENT_FORMAT = ENVIRONMENT_FORMAT_V1

DEFAULT_TERRAIN_CONFIG = {
    "map_range": [20.0, 20.0, 4.5], "size": [40.0, 40.0],
    "border_width": 5.0, "horizontal_scale": 0.1, "vertical_scale": 0.1,
    "slope_threshold": 0.75, "obstacle_width_range": [0.4, 1.1],
    "obstacle_height_range": [1.0, 1.5, 2.0, 4.0, 6.0],
    "obstacle_height_probability": [0.1, 0.15, 0.20, 0.55],
}


def _cpu(value: Any, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device="cpu").contiguous()


def mesh_sha256(vertices: Any, faces: Any) -> str:
    """Hash canonical little-endian arrays, including their shapes."""
    vertices_base = (vertices.detach().cpu().numpy()
                     if isinstance(vertices, torch.Tensor) else np.asarray(vertices))
    faces_base = (faces.detach().cpu().numpy()
                  if isinstance(faces, torch.Tensor) else np.asarray(faces))
    vertices_np = np.ascontiguousarray(vertices_base, dtype="<f4")
    faces_np = np.ascontiguousarray(faces_base, dtype="<i4")
    digest = hashlib.sha256(b"navrl-triangle-mesh-v1\0")
    digest.update(np.asarray(vertices_np.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(faces_np.shape, dtype="<i8").tobytes())
    digest.update(vertices_np.tobytes())
    digest.update(faces_np.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_boundary_sides(positions: Any) -> torch.Tensor:
    points = torch.as_tensor(positions, dtype=torch.float32).reshape(-1, 3)
    distance = torch.stack((
        (points[:, 1] - 24.0).abs(), (points[:, 1] + 24.0).abs(),
        (points[:, 0] - 24.0).abs(), (points[:, 0] + 24.0).abs()), dim=-1)
    sides = distance.argmin(dim=-1).to(torch.int64)
    if not torch.all(distance.gather(1, sides[:, None]).squeeze(1) < 1e-4):
        raise ValueError("route position is not on a map boundary")
    return sides


def _sample_boundary_points(num_envs: int, generator: torch.Generator):
    masks = torch.tensor([[1., 0., 1.], [1., 0., 1.],
                          [0., 1., 1.], [0., 1., 1.]])
    shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.],
                           [24., 0., 0.], [-24., 0., 0.]])
    sides = torch.randint(0, 4, (num_envs,), generator=generator)
    points = 48. * torch.rand((num_envs, 1, 3), generator=generator) - 24.
    points[:, 0, 2] = .5 + 2. * torch.rand(num_envs, generator=generator)
    return (points * masks[sides].unsqueeze(1) + shifts[sides].unsqueeze(1)), sides


def generate_environment(num_envs: int = 256, terrain_seed: int = 18,
                         route_seed: int = 18, static_obstacles: int = 350,
                         max_steps: int = 2200) -> dict[str, Any]:
    """Create a legacy v1 route/recipe environment."""
    if num_envs <= 0 or static_obstacles < 0 or max_steps <= 0:
        raise ValueError("invalid evaluation environment dimensions")
    generator = torch.Generator(device="cpu").manual_seed(route_seed)
    targets, target_sides = _sample_boundary_points(num_envs, generator)
    starts, start_sides = _sample_boundary_points(num_envs, generator)
    return {
        "format": ENVIRONMENT_FORMAT_V1, "num_envs": int(num_envs),
        "terrain_seed": int(terrain_seed), "route_seed": int(route_seed),
        "static_obstacles": int(static_obstacles), "dynamic_obstacles": 0,
        "max_steps": int(max_steps), "terrain": dict(DEFAULT_TERRAIN_CONFIG),
        "route_sampling": "independent_uniform_four_edges",
        "start_positions": starts.float(), "target_positions": targets.float(),
        "start_sides": start_sides, "target_sides": target_sides,
    }


def make_environment_v2(*, vertices: Any, faces: Any,
                        start_positions: Any, target_positions: Any,
                        initial_quaternions: Any, initial_velocities: Any,
                        target_dir_2d: Any, terrain_seed: int, route_seed: int,
                        static_obstacles: int, max_steps: int, sim_dt: float,
                        lidar_config: dict[str, Any], checkpoint_path: Path,
                        terrain_config: dict[str, Any] | None = None,
                        start_sides: Any | None = None,
                        target_sides: Any | None = None) -> dict[str, Any]:
    starts = _cpu(start_positions, torch.float32).reshape(-1, 1, 3)
    targets = _cpu(target_positions, torch.float32).reshape(-1, 1, 3)
    num_envs = len(starts)
    if len(targets) != num_envs:
        raise ValueError("start and target counts differ")
    mesh_vertices = _cpu(vertices, torch.float32).reshape(-1, 3)
    mesh_faces = _cpu(faces, torch.int32).reshape(-1, 3)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    value = {
        "format": ENVIRONMENT_FORMAT_V2, "format_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_envs": num_envs, "terrain_seed": int(terrain_seed),
        "route_seed": int(route_seed),
        "route_sampling": "independent_uniform_four_edges",
        "static_obstacles": int(static_obstacles), "dynamic_obstacles": 0,
        "max_steps": int(max_steps), "sim_dt": float(sim_dt),
        "terrain": dict(DEFAULT_TERRAIN_CONFIG if terrain_config is None else terrain_config),
        "terrain_mesh": {
            "vertices": mesh_vertices, "faces": mesh_faces,
            "sha256": mesh_sha256(mesh_vertices, mesh_faces),
            "stage_prim_path": "/World/ground/terrain/mesh",
        },
        "start_positions": starts, "target_positions": targets,
        "start_sides": infer_boundary_sides(starts) if start_sides is None else _cpu(start_sides, torch.int64).reshape(-1),
        "target_sides": infer_boundary_sides(targets) if target_sides is None else _cpu(target_sides, torch.int64).reshape(-1),
        "initial_quaternions": _cpu(initial_quaternions, torch.float32).reshape(num_envs, 1, 4),
        "initial_velocities": _cpu(initial_velocities, torch.float32).reshape(num_envs, 1, 6),
        "target_dir_2d": _cpu(target_dir_2d, torch.float32).reshape(num_envs, 1, 3),
        "lidar": dict(lidar_config),
        "ppo": {"checkpoint_filename": checkpoint_path.name,
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "exploration_type": "mean"},
    }
    return validate_environment(value)


def _validate_routes(value: dict[str, Any], count: int) -> None:
    for key in ("start_positions", "target_positions"):
        tensor = value.get(key)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (count, 1, 3):
            raise ValueError(f"{key} must have shape [{count}, 1, 3]")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key} contains non-finite values")
    for key in ("start_sides", "target_sides"):
        tensor = value.get(key)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (count,):
            raise ValueError(f"{key} must have shape [{count}]")
        if ((tensor < 0) | (tensor > 3)).any():
            raise ValueError(f"{key} values must be in [0, 3]")


def validate_environment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("environment must be a dictionary")
    file_format = value.get("format")
    if file_format not in (ENVIRONMENT_FORMAT_V1, ENVIRONMENT_FORMAT_V2):
        raise ValueError(f"unsupported environment format {file_format!r}")
    count = int(value["num_envs"])
    if count <= 0 or int(value["max_steps"]) <= 0:
        raise ValueError("invalid environment dimensions")
    _validate_routes(value, count)
    if int(value.get("dynamic_obstacles", -1)) != 0:
        raise ValueError("only static obstacles are supported")
    if file_format == ENVIRONMENT_FORMAT_V1:
        return value
    mesh = value.get("terrain_mesh", {})
    vertices, faces = mesh.get("vertices"), mesh.get("faces")
    if not isinstance(vertices, torch.Tensor) or tuple(vertices.shape[1:]) != (3,):
        raise ValueError("terrain mesh vertices must have shape [V, 3]")
    if not isinstance(faces, torch.Tensor) or tuple(faces.shape[1:]) != (3,):
        raise ValueError("terrain mesh faces must have shape [F, 3]")
    if not torch.isfinite(vertices).all():
        raise ValueError("terrain mesh contains non-finite vertices")
    if faces.numel() and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("terrain mesh contains invalid face indices")
    actual_hash = mesh_sha256(vertices, faces)
    if mesh.get("sha256") != actual_hash:
        raise ValueError(f"terrain mesh SHA-256 mismatch: {mesh.get('sha256')} != {actual_hash}")
    for key, shape in {
        "initial_quaternions": (count, 1, 4),
        "initial_velocities": (count, 1, 6),
        "target_dir_2d": (count, 1, 3),
    }.items():
        tensor = value.get(key)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"{key} must have shape {list(shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key} contains non-finite values")
    if value["target_dir_2d"][..., 2].abs().max() > 1e-6:
        raise ValueError("target_dir_2d must have zero vertical component")
    if float(value["sim_dt"]) <= 0 or not isinstance(value.get("lidar"), dict):
        raise ValueError("invalid simulation or LiDAR metadata")
    if not isinstance(value.get("ppo"), dict):
        raise ValueError("v2 environment requires PPO metadata")
    return value


def slice_environment(value: dict[str, Any], env_ids: Iterable[int]) -> dict[str, Any]:
    validate_environment(value)
    indices = torch.as_tensor(list(env_ids), dtype=torch.long)
    if not len(indices) or indices.min() < 0 or indices.max() >= int(value["num_envs"]):
        raise ValueError("invalid environment slice")
    result = dict(value)
    result["num_envs"] = len(indices)
    for key in ("start_positions", "target_positions", "start_sides", "target_sides",
                "initial_quaternions", "initial_velocities", "target_dir_2d"):
        if key in value:
            result[key] = value[key][indices].clone()
    result["source_num_envs"] = int(value["num_envs"])
    result["source_env_ids"] = indices.clone()
    return validate_environment(result)


def load_environment(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    return validate_environment(value)


def save_environment(path: Path, value: dict[str, Any]) -> None:
    validate_environment(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
