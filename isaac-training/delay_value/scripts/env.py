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
from omni_drones.controllers import LeePositionController
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time
import os
from timing import TwoStageDelaySchedule

class NavigationEnv(IsaacEnv):

    # Actor inference and command transport are separate stages. Transport is
    # queued and overlaps the following inference, while the unchanged
    # velocity controller keeps running on the most recently applied cmd_vel.

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        self.eval_scenarios = None
        self.terrain_seed = 0
        eval_cfg = cfg.get("eval", {})
        if eval_cfg.get("enabled", False):
            dataset_path = eval_cfg.get("dataset_path")
            if not dataset_path:
                raise ValueError("eval.dataset_path is required when eval.enabled=true")
            if not os.path.isabs(dataset_path):
                project_dir = os.path.dirname(os.path.dirname(__file__))
                dataset_path = os.path.join(project_dir, dataset_path)
            self.eval_scenarios = torch.load(dataset_path, map_location="cpu")
            self._validate_eval_scenarios(self.eval_scenarios, cfg)
            self.terrain_seed = int(self.eval_scenarios["terrain_seed"])
            print(f"[Navigation Environment]: Loaded fixed evaluation set: {dataset_path}")

        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)

        super().__init__(cfg, cfg.headless)
        
        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.velocity_controller = LeePositionController(
            9.81, self.drone.params
        ).to(self.device)
        if self.eval_scenarios is not None:
            self.eval_start_pos = self.eval_scenarios["start_pos"].to(self.device)
            self.eval_target_pos = self.eval_scenarios["target_pos"].to(self.device)

        self.reference_dt = float(cfg.timing.reference_dt)
        if self.reference_dt <= 0.0:
            raise ValueError("timing.reference_dt must be positive")
        self.timing_schedule = TwoStageDelaySchedule(
            cfg.timing,
            physics_dt=self.dt,
            nominal_steps=self.substeps,
        )
        self.max_command_delay_steps = self.timing_schedule.command_range[1]

        # The policy selects world-frame velocity commands. active_cmd_vel is
        # visible to the low-level controller. Actor outputs enter a persistent
        # FIFO after inference and become active when transport finishes.
        self.active_cmd_vel = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.pending_cmd_vel = torch.zeros_like(self.active_cmd_vel)
        self.next_pending_cmd_vel = torch.zeros_like(self.active_cmd_vel)
        self.inference_delay = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.command_delay = torch.zeros_like(self.inference_delay)
        self.total_delay = torch.zeros_like(self.inference_delay)
        self.transition_dt = torch.full_like(self.inference_delay, self.reference_dt)
        self.command_age_at_update = torch.zeros_like(self.inference_delay)
        self.active_command_age = torch.zeros_like(self.inference_delay)
        self.pending_command_age = torch.zeros_like(self.inference_delay)
        self.command_queue_depth = torch.zeros_like(self.inference_delay)
        self._command_queue = []
        self.reward_gamma = float(cfg.algo.get("gamma", 0.99))
        if not 0.0 < self.reward_gamma <= 1.0:
            raise ValueError("algo.gamma must be in (0, 1]")
        self._record_physics_step = False
        self._stats_mask = torch.ones_like(self.inference_delay)

        # Keep the physical horizon equal to the no-delay environment.
        self.max_episode_time = float(self.max_episode_length * self.dt * self.substeps)
        self.nominal_command_dt = float(self.dt * self.substeps)
        self.episode_time = torch.zeros_like(self.inference_delay)


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


    @staticmethod
    def _validate_eval_scenarios(scenarios, cfg):
        required_keys = {
            "format_version",
            "terrain_seed",
            "num_envs",
            "num_obstacles",
            "start_pos",
            "target_pos",
        }
        missing = required_keys - scenarios.keys()
        if missing:
            raise ValueError(f"Evaluation dataset is missing keys: {sorted(missing)}")
        if scenarios["format_version"] != 1:
            raise ValueError(
                f"Unsupported evaluation dataset version: {scenarios['format_version']}"
            )
        if scenarios["num_envs"] != cfg.env.num_envs:
            raise ValueError(
                f"Evaluation dataset has {scenarios['num_envs']} environments, "
                f"but cfg.env.num_envs is {cfg.env.num_envs}"
            )
        if scenarios["num_obstacles"] != cfg.env.num_obstacles:
            raise ValueError(
                f"Evaluation dataset expects {scenarios['num_obstacles']} static obstacles, "
                f"but cfg.env.num_obstacles is {cfg.env.num_obstacles}"
            )
        expected_shape = (cfg.env.num_envs, 1, 3)
        for key in ("start_pos", "target_pos"):
            value = scenarios[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                raise ValueError(f"{key} must be a tensor with shape {expected_shape}")

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
        
        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=self.terrain_seed,
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
            debug_vis=True,
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


        # helper function to check pos validity for even distribution condition
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        curr_obs_dist = obs_dist
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # random sample an origin until satisfy the evenly distributed condition
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2]) 
                    else:
                        oz = self.max_obs_2d_height/2. # half of the height
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
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



    def move_dynamic_obstacle(self):
        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold
        
        # sample new goals in local range
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
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
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # Step 3: Calculate new position update for current timestep
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1


    def _set_specs(self):
        # Original navigation state (8), applied cmd_vel (3), next queued
        # cmd_vel (3), last measured delays (3), pending age (1), queue depth
        # (1). Commands are expressed in the goal frame; no future sampled
        # delay is exposed to the policy.
        observation_dim = 19
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
                "action": UnboundedContinuousTensorSpec(
                    (self.drone.n, 3), device=self.device
                ),
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "decision_count": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "inference_delay": UnboundedContinuousTensorSpec(1),
            "command_delay": UnboundedContinuousTensorSpec(1),
            "total_delay": UnboundedContinuousTensorSpec(1),
            "sampled_inference_delay": UnboundedContinuousTensorSpec(1),
            "sampled_command_delay": UnboundedContinuousTensorSpec(1),
            "sampled_total_delay": UnboundedContinuousTensorSpec(1),
            "transition_dt": UnboundedContinuousTensorSpec(1),
            "command_age_at_update": UnboundedContinuousTensorSpec(1),
            "pending_command_age": UnboundedContinuousTensorSpec(1),
            "command_queue_depth": UnboundedContinuousTensorSpec(1),
            "command_update_count": UnboundedContinuousTensorSpec(1),
            "controller_update_count": UnboundedContinuousTensorSpec(1),
            "episode_time": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            # decide which side
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # generate random positions
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights# height
            target_pos = target_pos * selected_masks + selected_shifts
            
            # apply target pos
            self.target_pos[env_ids] = target_pos

            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.    
        else:
            if self.eval_scenarios is not None:
                self.target_pos[env_ids] = self.eval_target_pos[env_ids]
            else:
                target_x = torch.linspace(
                    -0.5, 0.5, self.num_envs, device=self.device
                ) * 32.
                self.target_pos[env_ids, 0, 0] = target_x[env_ids]
                self.target_pos[env_ids, 0, 1] = -24.
                self.target_pos[env_ids, 0, 2] = 2.


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        if (self.training):
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # generate random positions
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights# height
            pos = pos * selected_masks + selected_shifts
            
            # pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            # pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            # pos[:, 0, 1] = -24.
            # pos[:, 0, 2] = 2.
        else:
            if self.eval_scenarios is not None:
                pos = self.eval_start_pos[env_ids].clone()
            else:
                pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
                pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
                pos[:, 0, 1] = 24.
                pos[:, 0, 2] = 2.
        
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
        self.active_cmd_vel[env_ids] = 0.
        self.pending_cmd_vel[env_ids] = 0.
        self.next_pending_cmd_vel[env_ids] = 0.
        self.inference_delay[env_ids] = 0.
        self.command_delay[env_ids] = 0.
        self.total_delay[env_ids] = 0.
        self.transition_dt[env_ids] = self.nominal_command_dt
        self.command_age_at_update[env_ids] = 0.
        self.active_command_age[env_ids] = 0.
        self.pending_command_age[env_ids] = 0.
        self.command_queue_depth[env_ids] = 0.
        self.episode_time[env_ids] = 0.
        if env_ids.numel() == self.num_envs:
            self._command_queue.clear()
        else:
            for entry in self._command_queue:
                entry["valid"][env_ids] = False
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        self.stats[env_ids] = 0.  

    def reset_timing_schedule(self, seed=None):
        """Reset both delay stages before a reproducible rollout."""
        self.timing_schedule.reset(seed=seed)

    def reset_delay_schedule(self, seed=None):
        # Backward-compatible name used by older evaluation scripts.
        self.reset_timing_schedule(seed=seed)

    def _read_actor_command(self, tensordict: TensorDictBase):
        command = tensordict[("agents", "action")]
        if command.ndim == 3 and command.shape[-2] == 1:
            command = command.squeeze(-2)
        expected_shape = (self.num_envs, 3)
        if tuple(command.shape) != expected_shape:
            raise ValueError(
                f"Expected actor cmd_vel with shape {expected_shape}, got {tuple(command.shape)}"
            )
        return command

    def _apply_velocity_control(self):
        root_state = self.drone.get_state(env_frame=False)
        self.info["drone_state"][:] = root_state[..., :13]
        motor_action = self.velocity_controller(
            root_state[..., :13],
            target_vel=self.active_cmd_vel.unsqueeze(1),
            target_yaw=None,
        )
        torch.nan_to_num_(motor_action, 0.)
        self.drone.apply_action(motor_action)

    def _apply_command(
        self,
        command: torch.Tensor,
        mask: torch.Tensor,
        command_age_steps: torch.Tensor,
        inference_delay: float,
    ):
        mask = mask.reshape(self.num_envs)
        self.command_age_at_update[mask] = self.active_command_age[mask]
        self.active_cmd_vel[mask] = command[mask]
        self.active_command_age[mask] = 0.
        actual_command_delay = command_age_steps[mask].unsqueeze(-1) * self.dt
        self.command_delay[mask] = actual_command_delay
        self.total_delay[mask] = inference_delay + actual_command_delay
        self.stats["command_update_count"][mask] += 1.

    def _enqueue_command(
        self,
        command: torch.Tensor,
        delay_steps: int,
        valid_mask: torch.Tensor,
        inference_delay: float,
    ):
        valid_mask = valid_mask.reshape(self.num_envs).clone()
        if delay_steps == 0 and not self._command_queue:
            self._apply_command(
                command,
                valid_mask,
                command_age_steps=torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device
                ),
                inference_delay=inference_delay,
            )
            return

        scheduled_steps = torch.full(
            (self.num_envs,),
            int(delay_steps),
            dtype=torch.long,
            device=self.device,
        )
        tail_remaining = torch.zeros_like(scheduled_steps)
        for entry in self._command_queue:
            tail_remaining = torch.where(
                entry["valid"], entry["remaining_steps"], tail_remaining
            )
        # ROS transports preserve publication order per environment. A newer
        # command may share an arrival boundary with the FIFO tail, but cannot
        # overtake it.
        scheduled_steps = torch.maximum(scheduled_steps, tail_remaining)
        immediate = valid_mask & scheduled_steps.eq(0)
        self._apply_command(
            command,
            immediate,
            command_age_steps=torch.zeros_like(scheduled_steps),
            inference_delay=inference_delay,
        )
        queued = valid_mask & scheduled_steps.gt(0)
        self._command_queue.append(
            {
                "command": command.clone(),
                "remaining_steps": scheduled_steps,
                "age_steps": torch.zeros_like(scheduled_steps),
                "inference_delay": float(inference_delay),
                "valid": queued,
                "retention_steps": max(1, self.max_command_delay_steps),
            }
        )

    def _advance_command_transport(self, alive_mask: torch.Tensor):
        alive_mask = alive_mask.reshape(self.num_envs)
        remaining_entries = []
        for entry in self._command_queue:
            valid = entry["valid"]
            entry["remaining_steps"][valid] -= 1
            entry["age_steps"][valid] += 1
            due = valid & entry["remaining_steps"].le(0)
            self._apply_command(
                entry["command"],
                due & alive_mask,
                command_age_steps=entry["age_steps"],
                inference_delay=entry["inference_delay"],
            )
            entry["valid"][due] = False
            entry["retention_steps"] -= 1
            if entry["retention_steps"] > 0:
                remaining_entries.append(entry)
        self._command_queue = remaining_entries
        self._refresh_command_queue_state()

    def _refresh_command_queue_state(self):
        self.next_pending_cmd_vel.zero_()
        self.pending_command_age.zero_()
        self.command_queue_depth.zero_()
        waiting_for_head = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for entry in self._command_queue:
            valid = entry["valid"]
            self.command_queue_depth[valid] += 1.
            take = valid & waiting_for_head
            self.next_pending_cmd_vel[take] = entry["command"][take]
            self.pending_command_age[take] = (
                entry["age_steps"][take].unsqueeze(-1) * self.dt
            )
            waiting_for_head[take] = False

    def _advance_physics_tick(self, render_step: int, alive_mask: torch.Tensor):
        self._apply_velocity_control()
        self.sim.step(self._should_render(render_step))
        if self.cfg.env_dyn.num_obstacles != 0:
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)
        alive = alive_mask.float()
        self.episode_time += self.dt * alive
        self.active_command_age += self.dt * alive
        self.stats["controller_update_count"] += alive
        self._advance_command_transport(alive_mask)
        self._stats_mask = alive
        self._record_physics_step = True
        try:
            return self._compute_state_and_obs()
        finally:
            self._record_physics_step = False

    def _update_observation_timing(self, tensordict: TensorDictBase):
        state_key = ("agents", "observation", "state")
        direction_key = ("agents", "observation", "direction")
        state = tensordict[state_key]
        direction = tensordict[direction_key]
        active_cmd_goal = vec_to_new_frame(
            self.active_cmd_vel.unsqueeze(1), direction
        ).squeeze(1)
        pending_cmd_goal = vec_to_new_frame(
            self.next_pending_cmd_vel.unsqueeze(1), direction
        ).squeeze(1)
        state[..., 8:11] = active_cmd_goal
        state[..., 11:14] = pending_cmd_goal
        state[..., 14:15] = self.inference_delay / self.reference_dt
        state[..., 15:16] = self.command_delay / self.reference_dt
        state[..., 16:17] = self.total_delay / self.reference_dt
        state[..., 17:18] = self.pending_command_age / self.reference_dt
        state[..., 18:19] = self.command_queue_depth

    def _step(self, tensordict: TensorDictBase):
        timing = self.timing_schedule.sample(self.training)
        self.pending_cmd_vel.copy_(self._read_actor_command(tensordict))

        sampled_inference_delay = timing.inference_steps * self.dt
        sampled_command_delay = timing.command_steps * self.dt
        sampled_total_delay = sampled_inference_delay + sampled_command_delay
        self.stats["sampled_inference_delay"].fill_(sampled_inference_delay)
        self.stats["sampled_command_delay"].fill_(sampled_command_delay)
        self.stats["sampled_total_delay"].fill_(sampled_total_delay)
        self.transition_dt.zero_()

        self.progress_buf += 1

        transition_reward = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        terminated = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device
        )
        truncated = torch.zeros_like(terminated)
        step_tensordict = None
        render_step = 0
        command_enqueued = False

        # Zero inference latency makes the actor output available at the start
        # of this nominal control interval.
        if timing.inference_steps == 0:
            valid = ~(terminated | truncated)
            self.inference_delay[valid] = sampled_inference_delay
            self._enqueue_command(
                self.pending_cmd_vel,
                timing.command_steps,
                valid,
                sampled_inference_delay,
            )
            self._refresh_command_queue_state()
            command_enqueued = True

        while render_step < timing.elapsed_steps:
            alive = ~(terminated | truncated)
            self.transition_dt += self.dt * alive.float()
            step_tensordict = self._advance_physics_tick(render_step, alive)
            transition_reward += (
                (self.reward_gamma ** render_step)
                * self.reward
                * alive.float()
            )
            terminated |= self.terminated
            truncated |= self.truncated
            render_step += 1

            # The actor output appears after inference_steps. Its transport
            # countdown starts from this boundary and overlaps future inference.
            if not command_enqueued and render_step == timing.inference_steps:
                valid = ~(terminated | truncated)
                self.inference_delay[valid] = sampled_inference_delay
                self._enqueue_command(
                    self.pending_cmd_vel,
                    timing.command_steps,
                    valid,
                    sampled_inference_delay,
                )
                self._refresh_command_queue_state()
                command_enqueued = True

        if not command_enqueued:
            raise RuntimeError("Actor command was not enqueued after inference")

        self.reward = transition_reward
        self.terminated = terminated
        self.truncated = truncated
        self.stats["inference_delay"] = self.inference_delay
        self.stats["command_delay"] = self.command_delay
        self.stats["total_delay"] = self.total_delay
        self.stats["transition_dt"] = self.transition_dt
        self.stats["command_age_at_update"] = self.command_age_at_update
        self.stats["pending_command_age"] = self.pending_command_age
        self.stats["command_queue_depth"] = self.command_queue_depth

        self._update_observation_timing(step_tensordict)
        step_tensordict.set("stats", self.stats.clone())
        next_tensordict = TensorDict({}, self.batch_size, device=self.device)
        next_tensordict.update(step_tensordict)
        next_tensordict.update(self._compute_reward_and_done())
        return next_tensordict
    
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

        # The first eight values preserve the baseline state layout. Timing-aware
        # checkpoints additionally observe applied/queued commands and only
        # timing values that are causally available at this boundary.
        drone_state = torch.cat(
            [rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1
        ).squeeze(1)
        active_cmd_goal = vec_to_new_frame(
            self.active_cmd_vel.unsqueeze(1), target_dir_2d
        ).squeeze(1)
        pending_cmd_goal = vec_to_new_frame(
            self.next_pending_cmd_vel.unsqueeze(1), target_dir_2d
        ).squeeze(1)
        timing_state = torch.cat(
            [
                self.inference_delay / self.reference_dt,
                self.command_delay / self.reference_dt,
                self.total_delay / self.reference_dt,
                self.pending_command_age / self.reference_dt,
                self.command_queue_depth,
            ],
            dim=-1,
        )
        drone_state = torch.cat(
            [drone_state, active_cmd_goal, pending_cmd_goal, timing_state], dim=-1
        )

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

            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1. # convert to category: [0, 1, 2, 3]
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            # concatenate all for dynamic obstacles
            # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)

            # check dynamic obstacle collision for later reward
            closest_dyn_obs_distance_2d_collsion = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collsion[dyn_obs_range_mask] = float('inf')
            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].unsqueeze(-1).norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float('inf')
            dynamic_collision_2d = closest_dyn_obs_distance_2d_collsion <= (closest_dyn_obs_width/2. + 0.3)
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= (closest_dyn_obs_height/2. + 0.3)
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


        # -----------------Reward Calculation-----------------
        # a. safety reward for static obstacles
        reward_safety_static = torch.log((self.lidar_range-self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=(2, 3))
        

        # b. safety reward for dynamic obstacles
        if (self.cfg.env_dyn.num_obstacles != 0):
            reward_safety_dynamic = torch.log((closest_dyn_obs_distance_reward).clamp(min=1e-6, max=self.lidar_range)).mean(dim=-1, keepdim=True)

        # c. velocity reward for goal direction
        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)#.clip(max=2.0)
        
        # d. smoothness reward for action smoothness
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)
        
        # e. height penalty reward for flying unnessarily high or low
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_height[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)] = ( (self.drone.pos[..., 2] - self.height_range[..., 1] - 0.2)**2 )[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)]
        penalty_height[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)] = ( (self.height_range[..., 0] - 0.2 - self.drone.pos[..., 2])**2 )[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)]


        # f. Collision condition with its penalty
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.3) # 0.3 collision radius
        collision = static_collision | dynamic_collision
        
        # Reward is evaluated once per fixed physics tick. _step aggregates it
        # with the same per-tick gamma used by PPO.
        if (self.cfg.env_dyn.num_obstacles != 0):
            reward_rate = reward_vel + 1. + reward_safety_static + reward_safety_dynamic - penalty_height * 8.0
        else:
            reward_rate = reward_vel + 1. + reward_safety_static - penalty_height * 8.0
        self.reward = reward_rate - penalty_smooth * 0.1

        # Terminal reward
        # self.reward[collision] -= 50. # collision

        # Terminate Conditions
        reach_goal = (distance.squeeze(-1) < 0.5)
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.
        self.terminated = below_bound | above_bound | collision
        # Keep the physical episode horizon equal to the nominal environment.
        # A delayed command may cover several physics steps, but it must not
        # give the drone extra simulated time to reach the goal.
        self.truncated = (self.episode_time >= self.max_episode_time)

        if self._record_physics_step:
            stats_mask = self._stats_mask
            velocity_mask = stats_mask.unsqueeze(-1).bool()
            self.prev_drone_vel_w = torch.where(
                velocity_mask,
                self.drone.vel_w[..., :3],
                self.prev_drone_vel_w,
            )
            self.stats["return"] += self.reward * stats_mask
            self.stats["episode_len"][:] = (
                self.episode_time / self.nominal_command_dt
            ).clamp(max=float(self.max_episode_length))
            self.stats["decision_count"][:] = self.progress_buf.unsqueeze(1)
            # Match baseline semantics: keep the goal condition from the
            # latest valid physics step, including the terminal step.
            self.stats["reach_goal"] = torch.where(
                stats_mask.bool(), reach_goal.float(), self.stats["reach_goal"]
            )
            self.stats["collision"] = torch.maximum(
                self.stats["collision"], collision.float() * stats_mask
            )
            self.stats["truncated"] = torch.maximum(
                self.stats["truncated"], self.truncated.float() * stats_mask
            )
            self.stats["inference_delay"] = self.inference_delay
            self.stats["command_delay"] = self.command_delay
            self.stats["total_delay"] = self.total_delay
            self.stats["transition_dt"] = self.transition_dt
            self.stats["command_age_at_update"] = self.command_age_at_update
            self.stats["pending_command_age"] = self.pending_command_age
            self.stats["command_queue_depth"] = self.command_queue_depth
            self.stats["episode_time"] = self.episode_time

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

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
