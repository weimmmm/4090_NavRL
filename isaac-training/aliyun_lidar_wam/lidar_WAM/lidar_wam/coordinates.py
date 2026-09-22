"""Canonical coordinate and action semantics for NavRL WAM v2.

The PPO actor predicts normalized velocity commands in a fixed goal frame.  Its
horizontal x axis is the start-to-target direction stored at reset, y points to
the left, and z is world up.  Both offline training and online deployment must
use these helpers; duplicating this conversion was the source of the legacy
body/goal mismatch.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


ACTION_LIMIT_MPS = 2.0
ACTION_HORIZON = 30
EXECUTION_HORIZON = 10
CONDITION_FRAME = "fixed_start_to_target_goal_frame"
ACTION_FRAME = CONDITION_FRAME
LIDAR_FRAME = "sensor_yaw"


def _torch_goal_basis(direction: torch.Tensor) -> torch.Tensor:
    direction = direction.reshape(-1, 3)
    eps = torch.finfo(direction.dtype).eps
    x = direction.clone()
    x[:, 2] = 0
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    up = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=x.dtype).expand_as(x)
    y = torch.cross(up, x, dim=-1)
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def _numpy_goal_basis(direction: Any) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float32).reshape(-1, 3)
    x = direction.copy()
    x[:, 2] = 0
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("target_dir_2d contains a zero horizontal vector")
    x /= norm
    up = np.broadcast_to(np.asarray([0.0, 0.0, 1.0], np.float32), x.shape)
    y = np.cross(up, x)
    y /= np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-8)
    z = np.cross(x, y)
    return np.stack((x, y, z), axis=-1)


def world_to_goal(vector: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Express world vectors in the fixed goal frame."""
    original = vector.shape
    vector = vector.reshape(len(direction), -1, 3)
    result = torch.matmul(vector, _torch_goal_basis(direction))
    return result.reshape(original)


def goal_to_world(vector: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Express fixed-goal-frame vectors in world coordinates exactly once."""
    original = vector.shape
    vector = vector.reshape(len(direction), -1, 3)
    result = torch.matmul(vector, _torch_goal_basis(direction).transpose(-1, -2))
    return result.reshape(original)


def numpy_world_to_goal(vector: Any, direction: Any) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    original = vector.shape
    direction = np.asarray(direction, dtype=np.float32).reshape(-1, 3)
    vector = vector.reshape(len(direction), -1, 3)
    return np.matmul(vector, _numpy_goal_basis(direction)).reshape(original)


def normalized_to_goal_velocity(action: torch.Tensor,
                                action_limit: float = ACTION_LIMIT_MPS) -> torch.Tensor:
    return 2.0 * action * float(action_limit) - float(action_limit)


def normalized_to_world_velocity(action: torch.Tensor, direction: torch.Tensor,
                                 action_limit: float = ACTION_LIMIT_MPS) -> torch.Tensor:
    return goal_to_world(normalized_to_goal_velocity(action, action_limit), direction)


def numpy_normalized_to_world_velocity(action: Any, direction: Any,
                                       action_limit: float = ACTION_LIMIT_MPS) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    local = 2.0 * action * float(action_limit) - float(action_limit)
    direction = np.asarray(direction, dtype=np.float32).reshape(-1, 3)
    original = local.shape
    local = local.reshape(len(direction), -1, 3)
    return np.matmul(local, _numpy_goal_basis(direction).transpose(0, 2, 1)).reshape(original)


def goal_frame_causal_features(drone_state: torch.Tensor,
                               target_world: torch.Tensor,
                               goal_direction: torch.Tensor):
    """Return goal [xyz,distance] and 10-D proprioception in PPO's frame."""
    state = drone_state.reshape(-1, 13)
    target = target_world.reshape(-1, 3)
    direction = goal_direction.reshape(-1, 3)
    relative = world_to_goal(target - state[:, :3], direction)
    goal = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), dim=-1)
    velocity = world_to_goal(state[:, 7:10], direction)
    angular = world_to_goal(state[:, 10:13], direction)
    gravity_world = torch.tensor(
        [0.0, 0.0, -1.0], device=state.device, dtype=state.dtype
    ).expand(len(state), -1)
    gravity = world_to_goal(gravity_world, direction)
    proprio = torch.cat((state[:, 2:3], velocity, angular, gravity), dim=-1)
    return goal, proprio


def numpy_goal_frame_causal_features(drone_state: Any, target_world: Any,
                                     goal_direction: Any):
    # Use the exact deployment implementation even while building NumPy-backed
    # dataset indices.  Independent NumPy norm/sqrt kernels differed by up to
    # 1.5e-5 for long routes, which defeats the strict offline/online parity
    # guarantee even though the geometric frame was the same.
    state = torch.from_numpy(
        np.asarray(drone_state, dtype=np.float32).reshape(-1, 13))
    target = torch.from_numpy(
        np.asarray(target_world, dtype=np.float32).reshape(-1, 3))
    direction = torch.from_numpy(
        np.asarray(goal_direction, dtype=np.float32).reshape(-1, 3))
    with torch.no_grad():
        goal, proprio = goal_frame_causal_features(state, target, direction)
    return goal.numpy(), proprio.numpy()


def yaw_error_radians(drone_state: torch.Tensor,
                      goal_direction: torch.Tensor) -> torch.Tensor:
    """Signed sensor-yaw minus fixed-goal-yaw, wrapped to [-pi, pi]."""
    quaternion = drone_state.reshape(-1, 13)[:, 3:7]
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    direction = goal_direction.reshape(-1, 3)
    goal_yaw = torch.atan2(direction[:, 1], direction[:, 0])
    return torch.remainder(yaw - goal_yaw + math.pi, 2 * math.pi) - math.pi


def semantics() -> dict[str, Any]:
    return {
        "condition_frame": CONDITION_FRAME,
        "action_frame": ACTION_FRAME,
        "lidar_frame": LIDAR_FRAME,
        "normalized_action_range": [0.0, 1.0],
        "action_limit_mps": ACTION_LIMIT_MPS,
        "action_horizon": ACTION_HORIZON,
        "execution_horizon": EXECUTION_HORIZON,
        "world_conversion": "v_goal=4*action-2; v_world=goal_basis@v_goal exactly once",
    }
