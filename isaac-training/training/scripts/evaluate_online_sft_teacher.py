"""Audit the EXACT online SFT PPO under the existing fixed-route evaluator.

Student training/deployment code and Lee controller remain unchanged. This
standalone process replaces ONLY the policy factory in its own Python module.
The imported student checkpoint is an adapter/schema scaffold, never queried.
PPO uses its dataset-matched 4m/36x4 sensor and privileged cuboid features.
We separately test native per-axis commands, feasible commands, and feasible
K=10 constant-command feedback. All retain the ten-physics-step bootstrap,
strict 0.5m braking, 0.2m/s/10-step success and the same 128 fixed routes.
"""
import hashlib
from types import SimpleNamespace

import wam_closed_loop_eval as evaluator

BasePolicy = evaluator.WamLeePolicy


class TeacherBenchmarkPolicy(BasePolicy):
    def __init__(self, cfg, env, lee, checkpoint, world_checkpoint):
        super().__init__(cfg, env, lee, checkpoint, world_checkpoint)
        import torch
        from omegaconf import OmegaConf
        from torchrl.data import UnboundedContinuousTensorSpec
        from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
        from privileged_ppo import PPO
        from collect_action_dagger import static_observation
        from teacher_benchmark_commands import teacher_ego_chunk

        if (not self.is_standalone_action_expert or getattr(self, "use_past_actions", False)
                or self.action_semantics != "actual_velocity_input_to_original_lee"
                or not self.goal_stop_enabled or self.safety_enabled):
            raise ValueError("Keep the original single-frame student adapter/strict braking")
        self.mode = str(cfg.wam_eval.teacher_velocity_domain)
        self.interval = int(cfg.wam_eval.teacher_query_interval_physics_steps)
        if self.mode not in ("native", "student-norm", "student-box") or self.interval not in (1, 10):
            raise ValueError("Teacher benchmark supports native/feasible commands and K=1/10")
        self.bootstrap_physics_steps = 10
        self.teacher_path = evaluator._resolve(
            evaluator.Path(__file__).resolve().parents[2], cfg.wam_eval.teacher_checkpoint)
        digest = hashlib.sha256(self.teacher_path.read_bytes()).hexdigest()
        if digest != "1fb660d0f7ce3f136daa6967095149b8ad1a34a32eb3e67216bef22d856a323e":
            raise ValueError("Teacher differs from the original dataset/online-SFT PPO")
        n = env.num_envs
        spec = env.observation_spec.clone()
        for name, shape in (("lidar", (n, 1, 36, 4)),
                            ("dynamic_obstacle", (n, 1, 5, 10)),
                            ("static_obstacle", (n, 1, 5, 10))):
            spec["agents", "observation", name] = UnboundedContinuousTensorSpec(shape, device=cfg.device)
        ppo_cfg = OmegaConf.create({"feature_extractor": {"learning_rate": .0005},
                                   "actor": {"learning_rate": .0005, "action_limit": 2.},
                                   "critic": {"learning_rate": .0005}})
        self.teacher = PPO(ppo_cfg, spec, SimpleNamespace(shape=(1, 3)), cfg.device)
        self.teacher.load_state_dict(torch.load(self.teacher_path, map_location=cfg.device), strict=True)
        self.teacher.eval().requires_grad_(False)
        sensor_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 0.)), attach_yaw_only=True,
            pattern_cfg=patterns.BpearlPatternCfg(horizontal_fov=360., horizontal_res=10.,
                                                  vertical_ray_angles=torch.linspace(-10., 20., 4)),
            max_distance=4., mesh_prim_paths=[cfg.sensor.mesh_prim_path], debug_vis=False)
        self.teacher_sensor = RayCaster(sensor_cfg)
        self.teacher_sensor._initialize_impl()
        self.centers = env.fixed_scene["obstacle_center_xyz"].to(cfg.device)
        self.sizes = env.fixed_scene["obstacle_size_xyz"].to(cfg.device)
        self.static_features, self.make_chunk = static_observation, teacher_ego_chunk
        self.checkpoint_step = 27000
        self.benchmark_metadata = {
            "policy": "privileged_ppo_online_sft_teacher",
            "action_generation": "ppo_teacher_diagnostic", "sample_steps": 0,
            "training_action_diffusion": False, "deployment_action_diffusion": False,
            "checkpoint": str(self.teacher_path), "teacher_sha256": digest,
            "student_scaffold_checkpoint_not_queried": str(checkpoint),
            "teacher_query_interval_physics_steps": self.interval,
            "teacher_velocity_domain": self.mode, "bootstrap_physical_steps": 10,
            "teacher_lidar": {"range_m": 4., "shape": [36, 4]},
            "teacher_privileged_static_features": {"count": 5, "radius_m": 8.},
            "teacher_k10_semantics": "hold current ego command under unchanged current-yaw adapter",
        }

    def __call__(self, td):
        self.teacher_td = td
        from torchrl.envs.utils import ExplorationType, set_exploration_type
        with set_exploration_type(ExplorationType.MEAN):
            return super().__call__(td)

    def _sample_chunk(self, state, acceleration_world):
        torch = self.torch
        self.teacher_sensor.update(self.dt, force_recompute=True)
        td = self.teacher_td.clone()
        distance = (self.teacher_sensor.data.ray_hits_w
                    - self.teacher_sensor.data.pos_w[:, None]).norm(dim=-1)
        distance = torch.nan_to_num(distance, nan=4., posinf=4.).clamp(0., 4.)
        td["agents", "observation", "lidar"] = 4.-distance.reshape(self.num_envs, 1, 36, 4)
        td["agents", "observation", "dynamic_obstacle"] = torch.zeros(
            self.num_envs, 1, 5, 10, device=self.device)
        td["agents", "observation", "static_obstacle"] = self.static_features(
            state[:, :3], td["agents", "observation", "direction"], self.centers, self.sizes)
        self.teacher(td)
        world = td["agents", "action"].reshape(self.num_envs, 3)
        self.chunk.copy_(self.make_chunk(world, state[:, 3:7], self.mode, self.step_index,
                                        self.bootstrap_physics_steps, self.bootstrap_forward_speed))
        self.inference_calls += 1


if __name__ == "__main__":
    # This assignment is confined to this diagnostic process. Neither the
    # student's evaluator nor the active training actor imports this module.
    evaluator.WamLeePolicy = TeacherBenchmarkPolicy
    evaluator.main()
