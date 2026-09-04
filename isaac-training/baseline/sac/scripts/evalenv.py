"""Frozen-world replay environment for SAC evaluation."""
import numpy as np
import torch

from env import NavigationEnv


class EvalEnv(NavigationEnv):
    SNAPSHOT_SCHEMA_VERSION = 1

    def __init__(self, cfg, world_snapshot_path=None):
        super().__init__(cfg)
        self._world_snapshot = None
        self._world_step = 0
        if world_snapshot_path:
            self.load_world(world_snapshot_path)

    def load_world(self, path):
        snap = torch.load(path, map_location="cpu")
        meta = snap.get("meta", {})
        expected = {
            "schema_version": self.SNAPSHOT_SCHEMA_VERSION,
            "num_envs": int(self.num_envs),
            "max_episode_length": int(self.max_episode_length),
            "num_obstacles_static": int(self.cfg.env.num_obstacles),
            "num_obstacles_dynamic": int(self.cfg.env_dyn.num_obstacles),
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(f"[EvalEnv]: snapshot mismatch for {key}: {meta.get(key)!r} != {value!r}")
        required = (
            "drone_pos_init", "drone_rot_init", "target_pos_init", "target_dir_init",
            "dyn_obs_origin", "dyn_obs_state_init", "dyn_obs_goal_init", "dyn_obs_vel_init",
            "dyn_obs_traj_state", "dyn_obs_traj_vel",
        )
        missing = [key for key in required if key not in snap]
        if missing:
            raise RuntimeError(f"[EvalEnv]: snapshot missing keys: {missing}")
        self._world_snapshot = {
            key: snap[key].to(self.device, dtype=torch.float32) for key in required
        }
        if "height_range_init" in snap:
            self._world_snapshot["height_range_init"] = snap["height_range_init"].to(self.device, dtype=torch.float32)
        self._world_step = 0
        self.dyn_obs_initialized = False
        print(f"[EvalEnv]: loaded frozen world {path} (seed={meta.get('world_seed')}, envs={meta.get('num_envs')})")

    def unload_world(self):
        """Return this same simulator environment to normal training sampling."""
        self._world_snapshot = None
        self._world_step = 0
        self.dyn_obs_initialized = False

    def _initialize_dynamic_obstacles(self):
        if self._world_snapshot is None:
            return super()._initialize_dynamic_obstacles()
        if self.cfg.env_dyn.num_obstacles == 0 or self.dyn_obs_initialized:
            return
        snap = self._world_snapshot
        self.dyn_obs_origin.copy_(snap["dyn_obs_origin"])
        self.dyn_obs_state.copy_(snap["dyn_obs_state_init"])
        self.dyn_obs_goal.copy_(snap["dyn_obs_goal_init"])
        self.dyn_obs_vel.copy_(snap["dyn_obs_vel_init"])
        self.dyn_obs_step_count = self._world_step = 0
        for i, obstacle in enumerate(self.dyn_obs_list):
            start, end = i * self.dyn_obs_num_of_each_category, (i + 1) * self.dyn_obs_num_of_each_category
            obstacle.write_root_state_to_sim(self.dyn_obs_state[start:end])
            obstacle.write_data_to_sim()
        self.dyn_obs_initialized = True

    def move_dynamic_obstacle(self):
        if self._world_snapshot is None:
            return super().move_dynamic_obstacle()
        if self.cfg.env_dyn.num_obstacles == 0:
            return
        snap = self._world_snapshot
        step = min(self._world_step, int(snap["dyn_obs_traj_state"].shape[0]) - 1)
        self.dyn_obs_state.copy_(snap["dyn_obs_traj_state"][step])
        self.dyn_obs_vel.copy_(snap["dyn_obs_traj_vel"][step])
        for i, obstacle in enumerate(self.dyn_obs_list):
            start, end = i * self.dyn_obs_num_of_each_category, (i + 1) * self.dyn_obs_num_of_each_category
            obstacle.write_root_state_to_sim(self.dyn_obs_state[start:end])
            obstacle.write_data_to_sim()
            obstacle.update(self.cfg.sim.dt)
        self.dyn_obs_step_count += 1
        self._world_step += 1

    def _reset_idx(self, env_ids):
        if self._world_snapshot is None:
            return super()._reset_idx(env_ids)
        if not self.dyn_obs_initialized:
            self._initialize_dynamic_obstacles()
        self.drone._reset_idx(env_ids, self.training)
        snap = self._world_snapshot
        self.target_pos[env_ids] = snap["target_pos_init"][env_ids]
        self.target_dir[env_ids] = snap["target_dir_init"][env_ids]
        pos, rot = snap["drone_pos_init"][env_ids], snap["drone_rot_init"][env_ids]
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.0
        if "height_range_init" in snap:
            self.height_range[env_ids] = snap["height_range_init"][env_ids]
        self.stats[env_ids] = 0.0

    def set_seed(self, seed, static_seed=False):
        if self._world_snapshot is None:
            return super().set_seed(seed, static_seed=static_seed)
        self.seed = self.dyn_obs_seed = int(seed)
        self._world_step = 0
        self.dyn_obs_initialized = False
        return int(seed)