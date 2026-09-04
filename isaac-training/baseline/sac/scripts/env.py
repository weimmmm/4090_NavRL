import torch
import einops
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg

# =====================================================================
# 评估量尺 v1 · 集中常量（改动需升协议版本 cfg/eval_protocol_v1.json 并全员重测）
# ---------------------------------------------------------------------
# 这些是「reach / collision / out_of_bound」指标定义所依赖的几何阈值，属于评估量尺，
# 不是训练旋钮。eval_protocol.assert_eval_protocol() 会把 env 运行时值与金标准比对，
# 任一被改动且未升协议版本，评估会当场 raise。
# OBS_ENCODING_VERSION：动态障碍 height/width 三态编码版本（P0-1，v19 起）。
#   旧编码(PPO/SDAC/SAC≤v18)的 ckpt 载入当前 env 复评属 OOD，会被 check_ckpt_encoding 点名。
OBS_ENCODING_VERSION = "v19_3state"
COLLISION_RADIUS = 0.3      # 无人机碰撞半径(m)：静障 lidar_range-0.3、动障 size/2+0.3
REACH_THRESHOLD = 0.5       # 到达判定距离(m)
Z_LOW = 0.2                 # 出界下界(m)
Z_HIGH = 4.0                # 出界上界(m)
# =====================================================================


class NavigationEnv(IsaacEnv):
    # 评估量尺常量（类属性，保证未经 __init__ 也可读；assert_eval_protocol 读实例属性）
    OBS_ENCODING_VERSION = OBS_ENCODING_VERSION

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        self.seed = cfg.seed
        self.dyn_obs_seed = cfg.seed
        self.dyn_obs_initialized = False
        # _design_scene() is called by IsaacEnv.__init__ and samples dynamic
        # obstacle origins, so the NumPy-side local RNG must exist before super().
        self.dyn_obs_np_rng = np.random.default_rng(int(self.dyn_obs_seed))
        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)

        # 评估量尺实例属性（供 eval_protocol.assert_eval_protocol 读运行时值比对金标准）
        self.obs_encoding_version = OBS_ENCODING_VERSION
        self.collision_radius = COLLISION_RADIUS
        self.reach_threshold = REACH_THRESHOLD
        self.z_low = Z_LOW
        self.z_high = Z_HIGH

        # 训练侧 reward 系数：从 cfg.reward.* 读取，缺省回退到历史硬编码值（无 reward 段时行为 bit 级不变）。
        # 好处：调 reward 不用动 env.py（改 yaml / CLI 即可），且每个 run 的 reward 配方随 cfg 自动进 wandb config。
        # 注意：这些是训练旋钮，不是评估量尺；改它们不破坏鲁棒性对比公平性。
        _rwd = getattr(cfg, "reward", None)
        def _rw(name, default):
            return float(getattr(_rwd, name, default)) if _rwd is not None else float(default)
        self.reward_coef = {
            "vel": _rw("vel_coef", 0.3),
            "step_bias": _rw("step_bias", 0.1),
            "safety_static": _rw("safety_static_coef", 1.0),
            "safety_dynamic": _rw("safety_dynamic_coef", 1.0),
            "smooth": _rw("smooth_coef", 0.1),
            "height": _rw("height_coef", 2.0),
            "safety_clamp_min": _rw("safety_clamp_min", -5.0),
            "terminal_reach": _rw("terminal_reach", 100.0),
            "terminal_collision": _rw("terminal_collision", 200.0),
            "terminal_out_of_bound": _rw("terminal_out_of_bound", 100.0),
        }

        super().__init__(cfg, cfg.headless)
        self._init_local_rngs(int(self.seed))
        
        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())


        # LiDAR Intialization
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # attach_yaw_only=False,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres, # horizontal default is set to 10
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            # mesh_prim_paths=["/World"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams) 
        
        # start and target 
        with torch.device(self.device):
            # self.start_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # Coordinate change: add target direction variable
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)
            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.     

    def _init_local_rngs(self, seed: int):
        # Keep environment randomness isolated from the global torch/np RNGs used
        # by policy sampling, replay sampling, and optimizer-side randomness.
        self._rng_device = torch.device(self.device)
        self.layout_generator = torch.Generator(device=self._rng_device)
        self.layout_generator.manual_seed(seed)
        self.dyn_obs_torch_generator = torch.Generator(device=self._rng_device)
        self.dyn_obs_torch_generator.manual_seed(seed)
        self.dyn_obs_np_rng = np.random.default_rng(seed)

    def snapshot_random_state(self):
        return {
            "seed": int(self.seed),
            "dyn_obs_seed": int(self.dyn_obs_seed),
            "layout_torch": self.layout_generator.get_state().clone().cpu(),
            "dyn_obs_torch": self.dyn_obs_torch_generator.get_state().clone().cpu(),
            "dyn_obs_np": self.dyn_obs_np_rng.bit_generator.state,
        }

    def restore_random_state(self, snapshot):
        if not snapshot:
            return
        self.seed = int(snapshot["seed"])
        self.dyn_obs_seed = int(snapshot["dyn_obs_seed"])
        self.layout_generator.set_state(snapshot["layout_torch"].cpu())
        self.dyn_obs_torch_generator.set_state(snapshot["dyn_obs_torch"].cpu())
        self.dyn_obs_np_rng = np.random.default_rng()
        self.dyn_obs_np_rng.bit_generator.state = snapshot["dyn_obs_np"]

    def set_seed(self, seed: int, static_seed: bool = False):
        # 修复 P1：把 static_seed 透传给父类，否则评估时父类内部随机性可能被错误地 increment。
        super().set_seed(seed, static_seed=static_seed)
        self.seed = seed
        # 修复 P0-3：deterministic eval 在 finally 阶段会以 (cfg.seed + step + 1)
        # 调 set_seed，过去这里会无条件重设全局 torch/np RNG 并清掉 dyn_obs_initialized。
        # 后果：
        #   1) 训练 RNG 在每个 eval 间隔后被强制 rebase 成可预测序列，actor 噪声丧失独立性；
        #   2) 每次 eval 后 _reset_idx 触发 _initialize_dynamic_obstacles，重新采样动态障碍 origin，
        #      训练 layout 漂移，且额外消耗 CPU/GPU。
        # 修复策略：static_seed=True（评估返回时使用）走"只更新成员变量"路径，不动全局 RNG，
        # 也不清 dyn_obs_initialized；只有 static_seed=False（首次种子或显式重置）才重设。
        if not static_seed:
            self.dyn_obs_seed = seed
            self.dyn_obs_initialized = False
            self.layout_generator.manual_seed(int(seed))
            self.dyn_obs_torch_generator.manual_seed(int(seed))
            self.dyn_obs_np_rng = np.random.default_rng(int(seed))

    def _is_dyn_obs_pos_valid(self, prev_pos_list, curr_pos, min_dist):
        for prev_pos in prev_pos_list:
            if np.linalg.norm(curr_pos - prev_pos) <= min_dist:
                return False
        return True

    def _sample_dyn_obs_origin(self, prev_pos_list, obs_dist, is_3d_obstacle):
        # Use attempt-count relaxation instead of wall clock so origin sampling
        # stays reproducible under a fixed seed.
        max_attempts = max(1, int(getattr(self.cfg.env_dyn, "origin_sample_max_attempts", 4096)))
        relax_interval = max(1, int(getattr(self.cfg.env_dyn, "origin_relax_interval", 256)))
        relax_factor = float(getattr(self.cfg.env_dyn, "origin_relax_factor", 0.8))
        relax_factor = min(max(relax_factor, 0.1), 1.0)

        chosen = None
        last_sample = None
        for attempt in range(max_attempts):
            ox = self.dyn_obs_np_rng.uniform(low=-self.map_range[0], high=self.map_range[0])
            oy = self.dyn_obs_np_rng.uniform(low=-self.map_range[1], high=self.map_range[1])
            if is_3d_obstacle:
                oz = self.dyn_obs_np_rng.uniform(low=0.0, high=self.map_range[2])
            else:
                oz = self.max_obs_2d_height / 2.0
            curr_pos = np.array([ox, oy])
            last_sample = [ox, oy, oz, curr_pos]

            relax_level = attempt // relax_interval
            adjusted_obs_dist = obs_dist * (relax_factor ** relax_level)
            if self._is_dyn_obs_pos_valid(prev_pos_list, curr_pos, adjusted_obs_dist):
                chosen = last_sample
                break

        if chosen is None:
            chosen = last_sample
        ox, oy, oz, curr_pos = chosen
        prev_pos_list.append(curr_pos)
        return [ox, oy, oz]

    def snapshot_dyn_obs(self):
        if self.cfg.env_dyn.num_obstacles == 0:
            return None
        return {
            "origin": self.dyn_obs_origin.clone(),
            "state": self.dyn_obs_state.clone(),
            "goal": self.dyn_obs_goal.clone(),
            "vel": self.dyn_obs_vel.clone(),
            "step_count": int(self.dyn_obs_step_count),
        }

    def restore_dyn_obs(self, snapshot):
        if snapshot is None or self.cfg.env_dyn.num_obstacles == 0:
            return
        self.dyn_obs_origin.copy_(snapshot["origin"])
        self.dyn_obs_state.copy_(snapshot["state"])
        self.dyn_obs_goal.copy_(snapshot["goal"])
        self.dyn_obs_vel.copy_(snapshot["vel"])
        self.dyn_obs_step_count = int(snapshot.get("step_count", 0))
        self.dyn_obs_initialized = True

        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            start_idx = category_idx * self.dyn_obs_num_of_each_category
            end_idx = (category_idx + 1) * self.dyn_obs_num_of_each_category
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[start_idx:end_idx])
            dynamic_obstacle.write_data_to_sim()


    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # This visual grid plane is separate from the generated terrain at
        # ``/World/ground`` below.  Its default USD lives on Nucleus, so the
        # frozen-world evaluator can skip it and remain fully offline.  Normal
        # training keeps the historical behaviour.
        if not bool(getattr(self.cfg, "skip_default_ground_plane", False)):
            cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
            cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2), 
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            # Terrain-origin markers are purely visual and reference Nucleus
            # assets.  Keep them for training/debugging, but disable them for
            # frozen-world evaluation so it runs without a Nucleus server.
            debug_vis=not bool(getattr(self.cfg, "skip_default_ground_plane", False)),
        )
        terrain_importer = TerrainImporter(terrain_cfg)

        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        # Dynamic Obstacles
        # NOTE: we use cuboid to represent 3D dynamic obstacles which can float in the air 
        # and the long cylinder to represent 2D dynamic obstacles for which the drone can only pass in 2D 
        # The width of the dynamic obstacles is divided into N_w=4 bins
        # [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # The height of the dynamic obstacles is divided into N_h=2 bins
        # [[0, 0.5], [0.5, inf]] we want to distinguish 3D obstacles and 2d obstacles
        N_w = 4 # number of width intervals between [0, 1]
        N_h = 2 # number of height: current only support binary
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        # 修复 P1：每类至少 1 个，否则后面 obs_dist=2*sqrt(...//0) 会除零，
        # 且 LazyTensor 全空导致 spawn 失败。
        if self.dyn_obs_num_of_each_category == 0:
            raise ValueError(
                f"env_dyn.num_obstacles ({self.cfg.env_dyn.num_obstacles}) must be >= "
                f"{dyn_obs_category_num} (one per cuboid/cylinder size class), "
                "otherwise dynamic obstacle spawn becomes empty."
            )
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # in case of the roundup error


        # Dynamic obstacle info
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 13 is based on the states from sim, we only care the first three which is position
        self.dyn_obs_state[:, 3] = 1. # Quaternion
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # size of dynamic obstacles


        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                origin = self._sample_dyn_obs_origin(
                    prev_pos_list=prev_pos_list,
                    obs_dist=obs_dist,
                    is_3d_obstacle=(category_idx < cuboid_category_num),
                )
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # Spawn various sizes of dynamic obstacles 
            if (category_idx < cuboid_category_num):
                # spawn for 3D dynamic obstacles
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # spawn for 2D dynamic obstacles
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height, 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)



    def _initialize_dynamic_obstacles(self):
        if self.cfg.env_dyn.num_obstacles == 0 or self.dyn_obs_initialized:
            return

        self.dyn_obs_np_rng = np.random.default_rng(int(self.dyn_obs_seed))
        self.dyn_obs_torch_generator.manual_seed(int(self.dyn_obs_seed))

        self.dyn_obs_origin.zero_()
        self.dyn_obs_state.zero_()
        self.dyn_obs_state[:, 3] = 1.
        self.dyn_obs_goal.zero_()
        self.dyn_obs_vel.zero_()
        self.dyn_obs_step_count = 0

        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles)
        prev_pos_list = []
        N_w = 4
        N_h = 2
        dyn_obs_category_num = N_w * N_h
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num / N_h)

        for category_idx in range(cuboid_category_num + cylinder_category_num):
            start_idx = category_idx * self.dyn_obs_num_of_each_category
            end_idx = (category_idx + 1) * self.dyn_obs_num_of_each_category
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                origin = self._sample_dyn_obs_origin(
                    prev_pos_list=prev_pos_list,
                    obs_dist=obs_dist,
                    is_3d_obstacle=(category_idx < cuboid_category_num),
                )
                obs_global_idx = start_idx + origin_idx
                origin_tensor = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)
                self.dyn_obs_origin[obs_global_idx] = origin_tensor
                self.dyn_obs_state[obs_global_idx, :3] = origin_tensor

            dynamic_obstacle = self.dyn_obs_list[category_idx]
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[start_idx:end_idx])
            dynamic_obstacle.write_data_to_sim()

        self.dyn_obs_initialized = True


    def move_dynamic_obstacle(self):
        step_seed = int(self.dyn_obs_seed + self.dyn_obs_step_count)
        self.dyn_obs_np_rng = np.random.default_rng(step_seed)
        self.dyn_obs_torch_generator.manual_seed(step_seed)

        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold
        
        # sample new goals in local range
        num_new_goal = int(torch.sum(dyn_obs_new_goal_mask).item())
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(
            num_new_goal, 1, dtype=torch.float, device=self.cfg.device, generator=self.dyn_obs_torch_generator
        )
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(
            num_new_goal, 1, dtype=torch.float, device=self.cfg.device, generator=self.dyn_obs_torch_generator
        )
        # 修复 P0：z 轴下界必须用 local_range[2]，原写法用了 local_range[1]，
        # 当 yaml 里 local_range=[5,5,4.5] 时 z 区间会变成非对称 [-5, +4]。
        sample_z_local = -self.cfg.env_dyn.local_range[2] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(
            num_new_goal, 1, dtype=torch.float, device=self.cfg.device, generator=self.dyn_obs_torch_generator
        )
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # apply local goal to the global range
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # clamp the range if out of the static env range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles


        # Step 2: Random sample velocity for roughly every 2 seconds
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(
                  self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device,
                  generator=self.dyn_obs_torch_generator,
              )
            # 修复 P0：障碍物刚到达 goal 时 (goal - state) 接近 0，需要 clamp_min 防止除零→NaN/Inf 扩散到 lidar。
            direction = self.dyn_obs_goal - self.dyn_obs_state[:, :3]
            direction_norm = torch.norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * direction / direction_norm

        # Step 3: Calculate new position update for current timestep
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1


    def _set_specs(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10

        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # CSAC（约束 SAC）需要 env 额外输出每步 cost；sac/ppo 不读这个键。
        # 默认关闭：emit_cost 缺省为 False 时，reward_spec/stats/输出 td 与历史完全一致，
        # train_sac.py / train_ppo.py 行为字节级不变。train_csac.yaml 里设 env.emit_cost=true 打开。
        self.emit_cost = bool(getattr(self.cfg.env, "emit_cost", False))
        if self.emit_cost:
            # 占位初值，保证 _compute_reward_and_done 在首个 step 前被引用也不报错。
            self.cost = torch.zeros(self.num_envs, 1, device=self.device)

        # Reward Spec
        reward_agents_spec = {"reward": UnboundedContinuousTensorSpec((1,))}
        if self.emit_cost:
            reward_agents_spec["cost"] = UnboundedContinuousTensorSpec((1,))
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec(reward_agents_spec)
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_fields = {
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "collision_static": UnboundedContinuousTensorSpec(1),
            "collision_dynamic": UnboundedContinuousTensorSpec(1),
            "out_of_bound": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
        }
        if self.emit_cost:
            stats_fields["cost"] = UnboundedContinuousTensorSpec(1)
        stats_spec = CompositeSpec(stats_fields).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def _layout_mode(self) -> str:
        if self.training:
            mode = str(getattr(self.cfg.env, "train_layout", "fixed_y_random"))
        else:
            if hasattr(self.cfg.env, "eval_layout"):
                mode = str(getattr(self.cfg.env, "eval_layout"))
            else:
                # Backward compatibility for older eval configs.
                mode = "fixed_y_random" if bool(getattr(self.cfg.env, "use_train_dist_at_eval", True)) else "fixed_y_grid"

        aliases = {
            "full": "fixed_y_random",
            "full_random": "fixed_y_random",
            "train_dist": "fixed_y_random",
            "random": "fixed_y_random",
            "grid": "fixed_y_grid",
            "linspace": "fixed_y_grid",
            "fixed": "fixed_y_grid",
            "four_side": "four_side_random",
            "four_side_train": "four_side_random",
            "sac_original": "four_side_random",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"fixed_y_random", "fixed_y_grid", "four_side_random"}:
            raise ValueError(
                f"Unsupported env layout '{mode}'. Expected one of "
                "fixed_y_random, fixed_y_grid, four_side_random."
            )
        return mode

    def _sample_fixed_y_random_pos(self, env_ids: torch.Tensor, y_value: float) -> torch.Tensor:
        pos = torch.zeros(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device)
        pos[:, 0, 0] = (
            torch.rand(
                env_ids.size(0), dtype=torch.float, device=self.device,
                generator=self.layout_generator,
            ) - 0.5
        ) * 32.
        pos[:, 0, 1] = y_value
        pos[:, 0, 2] = 0.5 + torch.rand(
            env_ids.size(0), dtype=torch.float, device=self.device,
            generator=self.layout_generator,
        ) * (2.5 - 0.5)
        return pos

    def _sample_fixed_y_grid_pos(self, env_ids: torch.Tensor, y_value: float) -> torch.Tensor:
        env_ids_long = env_ids.to(torch.long)
        pos = torch.zeros(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device)
        grid_x = torch.linspace(-0.5, 0.5, self.num_envs, dtype=torch.float, device=self.device) * 32.
        pos[:, 0, 0] = grid_x[env_ids_long]
        pos[:, 0, 1] = y_value
        pos[:, 0, 2] = 2.
        return pos

    def _sample_four_side_random_pos(self, env_ids: torch.Tensor) -> torch.Tensor:
        masks = torch.tensor(
            [[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]],
            dtype=torch.float,
            device=self.device,
        )
        shifts = torch.tensor(
            [[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]],
            dtype=torch.float,
            device=self.device,
        )
        side_ids = torch.randint(
            0, masks.size(0), (env_ids.size(0),), device=self.device,
            generator=self.layout_generator,
        )
        base = 48. * torch.rand(
            env_ids.size(0), 1, 3, dtype=torch.float, device=self.device,
            generator=self.layout_generator,
        ) + (-24.)
        base[:, 0, 2] = 0.5 + torch.rand(
            env_ids.size(0), dtype=torch.float, device=self.device,
            generator=self.layout_generator,
        ) * (2.5 - 0.5)
        return base * masks[side_ids].unsqueeze(1) + shifts[side_ids].unsqueeze(1)

    def reset_target(self, env_ids: torch.Tensor):
        layout = self._layout_mode()
        if layout == "four_side_random":
            target_pos = self._sample_four_side_random_pos(env_ids)
        elif layout == "fixed_y_grid":
            target_pos = self._sample_fixed_y_grid_pos(env_ids, -24.)
        else:
            target_pos = self._sample_fixed_y_random_pos(env_ids, -24.)
        self.target_pos[env_ids] = target_pos


    def _reset_idx(self, env_ids: torch.Tensor):
        if not self.dyn_obs_initialized:
            self._initialize_dynamic_obstacles()

        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        layout = self._layout_mode()
        if layout == "four_side_random":
            pos = self._sample_four_side_random_pos(env_ids)
        elif layout == "fixed_y_grid":
            pos = self._sample_fixed_y_grid_pos(env_ids, 24.)
        else:
            pos = self._sample_fixed_y_random_pos(env_ids, 24.)
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        # height_range 用固定宽松区间（物理边界 [0.2, 4.0] 的内 80%），给避障留充分垂直空间。
        # 旧设计按起点/终点 z 动态收窄会变成 ~1m 窄带，导致 agent 在避障时被过度惩罚，反而宁可贴地"白嫖"出界。
        self.height_range[env_ids, 0, 0] = 0.6
        self.height_range[env_ids, 0, 1] = 3.6

        self.stats[env_ids] = 0.  
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")] 
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)
    
    # get current states/observation
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False) # (world_pos, orientation (quat), world_vel_and_angular, heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller

        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        ) # lidar scan store the data that is range - distance and it is in lidar's local frame

        # Optional render for LiDAR
        if self._should_render(0):
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            # set_camera_view(
            #     eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
            #     target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)                        
            # )
            v = (self.lidar.data.ray_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            # self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
            # self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])
            self.debug_draw.vector(x.expand_as(v[:, 0])[0], v[0, 0])

        # ---------Network Input II: Drone's internal states---------
        # a. distance info in horizontal and vertical plane
        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True) # start to goal distance
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)
        
        
        # b. unit direction vector to goal
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0

        rpos_clipped = rpos / distance.clamp(1e-6) # unit vector: start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d) # express in the goal coodinate
        
        # c. velocity in the goal frame
        vel_w = self.root_state[..., 7:10] # world vel
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)   # coordinate change for velocity

        # final drone's internal states
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

        if (self.cfg.env_dyn.num_obstacles != 0):
            # ---------Network Input III: Dynamic obstacle states--------
            # ------------------------------------------------------------
            # a. Closest N obstacles relative position in the goal frame 
            # Find the N closest and within range obstacles for each drone
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3] 
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0)/2):, 2] = 0.
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos_expanded[..., :2], dim=2)  # Shape: (1000, 40). calculate 2d distance to each obstacle for all drones
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance_2d, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # pick top N closest obstacle index
            dyn_obs_range_mask = dyn_obs_distance_2d.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # relative distance of obstacles in the goal frame
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_2d) 
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0. # exclude out of range obstacles
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            # b. Velocity in the goal frame for the dynamic obstacles
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_2d) 

            # c. Size of dynamic obstacles in category
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx] # the acutal size

            # 修复 P0-1：原编码下 in-range 2D 障碍 (cylinder, height=5.0) 的 height_category 被
            # torch.where 映射成 0，而 out-of-range 障碍也被强制为 0，actor 完全无法
            # 区分"近处的 2D 高墙"和"远处占位空槽"。同样地，width_category 的 out-of-range
            # 强制 0 与 in-range 最小宽度类 (category=0) 撞值。
            #
            # 新编码（保持向量维度 = 1，和旧版兼容）：
            #   width_category:
            #     - in-range:    width / width_res - 1, 取值 ∈ {0, 1, 2, 3}
            #     - out-of-range: -1（占位标志，与任何 in-range 类别都不重合）
            #   height_category:
            #     - in-range 3D (cuboid, height ≤ max_obs_3d_height): 0
            #     - in-range 2D (cylinder, height >  max_obs_3d_height): 1
            #     - out-of-range:                                       2
            # 这样三种状态彼此可分。critic/actor 看到 height_category=2 即可学到
            # "这是占位"的特殊语义，避免把远处占位误读为某种障碍尺寸。
            #
            # ⚠️ 兼容性提示：此改动改变了 dynamic_obstacle 输入的数值分布（shape 不变）。
            # 加载在旧编码下训练的 checkpoint 后，前若干 rollout 行为会偏移；
            # 若需对照实验请从头训练一次。
            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1.
            closest_dyn_obs_width_category[dyn_obs_range_mask] = -1.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(
                closest_dyn_obs_height > self.max_obs_3d_height,
                torch.ones_like(closest_dyn_obs_height),
                torch.zeros_like(closest_dyn_obs_height),
            )
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 2.

            # concatenate all for dynamic obstacles
            # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)

            # check dynamic obstacle collision for later reward
            closest_dyn_obs_distance_2d_collsion = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collsion[dyn_obs_range_mask] = float('inf')
            # 修复 P0：z 距离用 abs() 表示，去掉冗余的 .norm()。
            # 注：2D 障碍（数组后半）在 619 行已经把 rpos_z 清零，所以 |z|=0
            # 永远满足 z 碰撞条件——这是有意把 2D 障碍当作"无限高墙"。
            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].abs().unsqueeze(-1)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float('inf')
            dynamic_collision_2d = closest_dyn_obs_distance_2d_collsion <= (closest_dyn_obs_width/2. + COLLISION_RADIUS)
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= (closest_dyn_obs_height/2. + COLLISION_RADIUS)
            dynamic_collision_each = dynamic_collision_2d & dynamic_collision_z
            dynamic_collision = torch.any(dynamic_collision_each, dim=1)

            # distance to dynamic obstacle for reward calculation (not 100% correct in math but should be good enough for approximation)
            closest_dyn_obs_distance_reward = closest_dyn_obs_rpos.norm(dim=-1) - closest_dyn_obs_size[..., 0]/2. # for those 2D obstacle, z distance will not be considered
            closest_dyn_obs_distance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range
            
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10, device=self.cfg.device)
            dynamic_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)
            
        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_2d,
            "dynamic_obstacle": dyn_obs_states
        }


        # ================================================================
        # 评估量尺 v1 · 碰撞 / 到达 / 出界 / 终止（禁止在此调参；改动需升协议版本 + 全员重测）
        #   reach_goal / collision / out_of_bound / terminated 的定义 = 评估指标定义本身。
        #   几何阈值集中在文件顶部 COLLISION_RADIUS / REACH_THRESHOLD / Z_LOW / Z_HIGH；
        #   eval_protocol.assert_eval_protocol 会在评估启动时把运行时值与金标准比对。
        # ================================================================
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - COLLISION_RADIUS) # 碰撞半径见 COLLISION_RADIUS（评估量尺）
        collision = static_collision | dynamic_collision

        reach_goal = (distance.squeeze(-1) < REACH_THRESHOLD)
        reach_goal_clean = reach_goal & (~collision)
        below_bound = self.drone.pos[..., 2] < Z_LOW
        above_bound = self.drone.pos[..., 2] > Z_HIGH
        out_of_bound = below_bound | above_bound
        self.terminated = reach_goal | out_of_bound | collision
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # progress buf is to track the step number

        # ================================================================
        # 训练侧 · reward / cost（可自由调，不影响上面的评估量尺）
        # ================================================================
        # CSAC cost：每步代价 = 是否碰撞。仅 emit_cost=true 时输出。
        if self.emit_cost:
            self.cost = collision.float()

        # reward 全部搬到 _compute_reward()（纯训练侧）：消费上面已算好的几何量，不改动它们。
        # 调 reward 系数 / 终点奖励请改 _compute_reward，不要碰本函数的「评估量尺」段。
        self.reward = self._compute_reward(
            rpos, distance,
            closest_dyn_obs_distance_reward if self.cfg.env_dyn.num_obstacles != 0 else None,
            reach_goal_clean, collision, out_of_bound,
        )

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal_clean.float()
        self.stats["collision"] = collision.float()
        # 碰撞来源分解（诊断用，不影响评估量尺）：static/dynamic 各自原始掩码，
        # 同一步可能同时为真，故 static+dynamic 可能略大于 collision（重叠部分）。
        self.stats["collision_static"] = static_collision.float()
        self.stats["collision_dynamic"] = dynamic_collision.float()
        self.stats["out_of_bound"] = (out_of_bound & (~collision)).float()
        self.stats["truncated"] = self.truncated.float()
        if self.emit_cost:
            self.stats["cost"] += self.cost

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward(self, rpos, distance, closest_dyn_obs_distance_reward,
                        reach_goal_clean, collision, out_of_bound):
        """训练侧 reward（与评估量尺解耦）。

        ⚠ 本方法只产出并返回 reward；reach_goal / collision / out_of_bound / terminated
        等评估指标定义在 _compute_state_and_obs 的「评估量尺」段，调这里的系数不会影响量尺。
        消费 _compute_state_and_obs 已算好的几何量（rpos / distance / 碰撞掩码等），不改动它们。
        以下所有系数都是训练旋钮，可自由调整 / 做消融。

        与重构前严格等价：各 reward 分项与碰撞/终止互不依赖，仅是把"先算分项再算量尺"
        调整为"先算量尺再算分项"，逐行算术与求值顺序不变 → 数值 bit 级一致。
        系数取自 self.reward_coef（来自 cfg.reward.*，缺省=历史值）。
        """
        rc = self.reward_coef
        # a. safety reward for static obstacles（减 log(lidar_range) baseline：远离≈0，贴近为负，clamp 防贴边像素主导）
        safety_log_baseline = np.log(self.lidar_range)
        reward_safety_static = (torch.log((self.lidar_range-self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=(2, 3)) - safety_log_baseline).clamp(min=rc["safety_clamp_min"])

        # b. safety reward for dynamic obstacles
        reward_safety_dynamic = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        if (self.cfg.env_dyn.num_obstacles != 0):
            reward_safety_dynamic = (torch.log((closest_dyn_obs_distance_reward).clamp(min=1e-6, max=self.lidar_range)).mean(dim=-1, keepdim=True) - safety_log_baseline).clamp(min=rc["safety_clamp_min"])

        # c. velocity reward for goal direction
        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)#.clip(max=2.0)

        # d. smoothness reward for action smoothness
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)

        # e. height penalty reward for flying unnessarily high or low
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_height[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)] = ( (self.drone.pos[..., 2] - self.height_range[..., 1] - 0.2)**2 )[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)]
        penalty_height[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)] = ( (self.height_range[..., 0] - 0.2 - self.drone.pos[..., 2])**2 )[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)]

        # f. Final reward calculation
        # reward_vel 系数从 1.0 降到 0.3，避免 agent 朝目标直冲撞墙（冲撞短期回报高于绕障）
        # penalty_height 系数从 8.0 降到 2.0，配合拉宽的 height_range，允许避障时临时变高/低
        if (self.cfg.env_dyn.num_obstacles != 0):
            reward = reward_vel * rc["vel"] + rc["step_bias"] + reward_safety_static * rc["safety_static"] + reward_safety_dynamic * rc["safety_dynamic"] - penalty_smooth * rc["smooth"] - penalty_height * rc["height"]
        else:
            reward = reward_vel * rc["vel"] + rc["step_bias"] + reward_safety_static * rc["safety_static"] - penalty_smooth * rc["smooth"] - penalty_height * rc["height"]

        # g. Terminal rewards
        # - reach_goal_clean +100: 到目标且没撞
        # - collision -200: 撞障碍，最严重
        # - out_of_bound -100: 飞到 z<0.2 或 z>4.0 也给显式惩罚，堵死"贴地/顶天白嫖结束 episode"的捷径
        reward[reach_goal_clean] += rc["terminal_reach"]
        reward[collision] -= rc["terminal_collision"]
        reward[out_of_bound & (~collision)] -= rc["terminal_out_of_bound"]

        # update previous velocity for smoothness calculation in the next iteration
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()
        return reward

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        agents_out = {"reward": reward}
        if self.emit_cost:
            agents_out["cost"] = self.cost
        return TensorDict(
            {
                "agents": agents_out,
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )