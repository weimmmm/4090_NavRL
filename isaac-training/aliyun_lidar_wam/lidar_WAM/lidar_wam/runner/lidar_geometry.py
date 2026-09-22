"""Ray-calibrated LiDAR reprojection for read-only NavRL diagnostics.

The original calibration labels horizontal column zero as -180 degrees, while
its saved ray direction points along +x (zero degrees). Projection therefore
uses the saved directions, not the inconsistent horizontal-angle labels.
"""

from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation


def load_rays(dataset_root, split, seed):
    shard = Path(dataset_root) / split / f"seed_{seed:04d}"
    calibration = json.loads((shard / "calibration" / "lidar.json").read_text())
    rays = np.load(shard / calibration["ray_directions_path"])
    if rays.shape != (108 * 18, 3):
        raise ValueError(f"Unexpected ray directions: {rays.shape}")
    rays = rays.reshape(108, 18, 3).astype(np.float64)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    azimuth = np.unwrap(np.arctan2(rays[:, 0, 1], rays[:, 0, 0]))
    elevation = np.arctan2(rays[0, :, 2], np.linalg.norm(rays[0, :, :2], axis=1))
    if not np.all(np.diff(azimuth) > 0) or not np.all(np.diff(elevation) > 0):
        raise ValueError("Ray grid is not ordered as expected")
    return rays, azimuth, elevation


def frame_points(frame, rays, threshold=0.0):
    valid = frame[1, :, :18] > threshold
    distance = np.clip((frame[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
    return (rays[valid] * distance[valid, None]).astype(np.float32)


def project_points(points, azimuth, elevation):
    """Project onto nearest calibrated rays, keeping the nearest return."""
    ranges = np.full((108, 18), 10.0, dtype=np.float32)
    if len(points):
        distance = np.linalg.norm(points, axis=1)
        keep = np.isfinite(points).all(axis=1) & (distance > 1e-6) & (distance < 10.0)
        points, distance = points[keep], distance[keep]
        if len(points):
            angle = np.arctan2(points[:, 1], points[:, 0])
            horizontal_step = float(np.median(np.diff(azimuth)))
            col = np.rint((angle - azimuth[0]) / horizontal_step).astype(np.int64) % 108
            vertical = np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1))
            row = np.abs(vertical[:, None] - elevation[None, :]).argmin(axis=1)
            vertical_step = float(np.median(np.diff(elevation)))
            in_view = ((vertical >= elevation[0] - vertical_step / 2) &
                       (vertical <= elevation[-1] + vertical_step / 2))
            np.minimum.at(ranges.reshape(-1), col[in_view] * 18 + row[in_view],
                          distance[in_view].astype(np.float32))
    frame = np.empty((2, 108, 20), dtype=np.float32)
    frame[0].fill(1.0)
    frame[1].fill(-1.0)
    frame[0, :, :18] = ranges / 5.0 - 1.0
    frame[1, :, :18] = np.where(ranges < 10.0, 1.0, -1.0)
    return frame


def warp_frame(previous, transform, rays, azimuth, elevation):
    points = frame_points(previous, rays)
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    return project_points(transformed, azimuth, elevation)


def predict_transform(previous_state, dt=0.16):
    """Constant measured velocity and yaw-rate baseline; no future state."""
    quaternion = previous_state[3:7]  # w, x, y, z
    yaw = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]]).as_euler("xyz")[2]
    previous_rotation = Rotation.from_euler("z", yaw).as_matrix()
    future_rotation = Rotation.from_euler("z", yaw + previous_state[12] * dt).as_matrix()
    predicted_position = previous_state[:3] + previous_state[7:10] * dt
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = future_rotation.T @ previous_rotation
    transform[:3, 3] = future_rotation.T @ (previous_state[:3] - predicted_position)
    return transform
