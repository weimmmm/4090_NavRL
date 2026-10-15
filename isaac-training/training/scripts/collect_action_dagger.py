"""Student-controlled K=10 rollout; PPO labels only each chunk's first action.

The teacher has its ORIGINAL 4 m / 36x4 sensor and static privileged features.
The student continues to use the existing 5 m sensor and unchanged Lee policy.
No teacher observation is passed to the student. Bootstrap/braking overrides
are excluded from the supervised buffer. Outputs are normalized cached inputs.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
from types import SimpleNamespace

from wam_closed_loop_eval import WamLeePolicy, CONFIG_DIR, _resolve
import hydra
from omni.isaac.kit import SimulationApp


def static_observation(position, direction, centers, sizes, count=5, radius=8.0):
    """Exact cuboid feature convention used by the migrated privileged PPO."""
    import torch
    from privileged_ppo_utils import vec_to_new_frame
    output = position.new_zeros(len(position), 1, count, 10)
    relative = centers[None] - position[:, None]
    surface = (relative.abs() - sizes[None] * 0.5).clamp_min(0).norm(dim=-1)
    distance, indices = surface.topk(min(count, len(centers)), largest=False)
    indices3 = indices[..., None].expand(-1, -1, 3)
    selected = relative.gather(1, indices3)
    size = sizes[None].expand(len(position), -1, -1).gather(1, indices3)
    local = vec_to_new_frame(selected, direction)
    yaw = -torch.atan2(direction[..., 1], direction[..., 0])[..., None]
    yaw = torch.atan2(yaw.sin(), yaw.cos()) / torch.pi
    yaw = yaw.expand(-1, indices.shape[1], -1)
    features = torch.cat((local / selected.norm(dim=-1, keepdim=True).clamp_min(1e-6),
                          local[..., :2].norm(dim=-1, keepdim=True), local[..., 2:3],
                          size, torch.zeros_like(yaw), yaw), dim=-1)
    features[distance > radius] = 0
    output[:, 0, :indices.shape[1]] = features
    return output


@hydra.main(config_path=CONFIG_DIR, config_name="train", version_base=None)
def main(cfg):
    import torch
    root = Path(__file__).resolve().parents[2]
    checkpoint = _resolve(root, cfg.wam_eval.checkpoint)
    teacher_path = _resolve(root, cfg.dagger.teacher_checkpoint)
    output = _resolve(root, cfg.dagger.output)
    if output.exists():
        raise FileExistsError(output)
    expected_sha = "1fb660d0f7ce3f136daa6967095149b8ad1a34a32eb3e67216bef22d856a323e"
    teacher_sha = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    if teacher_sha != expected_sha:
        raise ValueError("teacher does not match the 5 m dataset's privileged PPO")
    if float(cfg.sim.dt) != .016 or int(cfg.wam_eval.replan_interval_steps) != 10:
        raise ValueError("Keep the baseline dt=.016 and K=10")
    if int(cfg.env_dyn.num_obstacles) != 0:
        raise ValueError("This pilot supports static maps only")
    app = SimulationApp({"headless": True, "anti_aliasing": 1,
                         "active_gpu": int(os.environ.get("ISAAC_ACTIVE_GPU", "0")),
                         "physics_gpu": int(str(cfg.device).split(":")[-1]),
                         "multi_gpu": False})
    # Explicit known asset root avoids Nucleus auto-discovery hanging on
    # unreachable mounted servers. It affects only this simulation process.
    import omni.isaac.core.utils.nucleus as nucleus
    if os.environ.get("NAVRL_ASSETS_ROOT"):
        nucleus.get_assets_root_path = lambda: os.environ["NAVRL_ASSETS_ROOT"]
    from env import NavigationEnv
    from privileged_ppo import PPO
    from omni_drones.controllers import LeePositionController
    from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
    from torchrl.data import UnboundedContinuousTensorSpec
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from omegaconf import OmegaConf

    env = NavigationEnv(cfg)
    env.eval()
    env.set_seed(int(cfg.seed))
    lee = LeePositionController(9.81, env.drone.params).to(cfg.device)
    student = WamLeePolicy(cfg, env, lee, checkpoint, checkpoint)
    if student.use_past_actions or student.lidar_history_frames != 1:
        raise ValueError("Pilot must start from the validated single-frame/no-past baseline")
    sensor_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 0.)), attach_yaw_only=True,
        pattern_cfg=patterns.BpearlPatternCfg(horizontal_fov=360., horizontal_res=10.,
                                              vertical_ray_angles=torch.linspace(-10., 20., 4)),
        max_distance=4., mesh_prim_paths=[cfg.sensor.mesh_prim_path], debug_vis=False)
    teacher_sensor = RayCaster(sensor_cfg)
    teacher_sensor._initialize_impl()
    n = int(cfg.env.num_envs)
    spec = env.observation_spec.clone()
    for name, shape in (("lidar", (n, 1, 36, 4)),
                        ("dynamic_obstacle", (n, 1, 5, 10)),
                        ("static_obstacle", (n, 1, 5, 10))):
        spec["agents", "observation", name] = UnboundedContinuousTensorSpec(shape, device=cfg.device)
    teacher_cfg = OmegaConf.create({"feature_extractor": {"learning_rate": .0005},
                                   "actor": {"learning_rate": .0005, "action_limit": 2.},
                                   "critic": {"learning_rate": .0005}})
    teacher = PPO(teacher_cfg, spec, SimpleNamespace(shape=(1, 3)), cfg.device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=cfg.device), strict=True)
    teacher.eval()
    teacher.requires_grad_(False)
    centers = env.fixed_scene["obstacle_center_xyz"].to(cfg.device)
    sizes = env.fixed_scene["obstacle_size_xyz"].to(cfg.device)
    captured = {}
    predict = student.model.predict_action

    def capture(latent, goal, proprio, past, mask):
        prediction = predict(latent, goal, proprio, past, mask)
        captured.update(latent=latent.detach(), goal=goal.detach(),
                        proprio=proprio.detach(), prediction=prediction.detach().clone())
        return prediction

    student.model.predict_action = capture
    chunks = {key: [] for key in ("latent", "goal", "proprio", "target", "student_first", "route", "physics_step")}
    active = torch.ones(n, dtype=torch.bool, device=cfg.device)
    outcomes = ["timeout"] * n
    td = env.reset()
    with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
        for step in range(int(cfg.env.max_episode_length)):
            student.set_active(active)
            due = step % 10 == 0
            if due:
                teacher_sensor.update(float(cfg.sim.dt), force_recompute=True)
                expert_td = td.clone()
                distances = (teacher_sensor.data.ray_hits_w - teacher_sensor.data.pos_w[:, None]).norm(dim=-1)
                distances = torch.nan_to_num(distances, nan=4., posinf=4.).clamp(0., 4.)
                expert_td["agents", "observation", "lidar"] = 4. - distances.reshape(n, 1, 36, 4)
                expert_td["agents", "observation", "dynamic_obstacle"] = torch.zeros(n, 1, 5, 10, device=cfg.device)
                state = td["info", "drone_state"][:, 0, :13]
                expert_td["agents", "observation", "static_obstacle"] = static_observation(
                    state[:, :3], expert_td["agents", "observation", "direction"], centers, sizes)
                teacher(expert_td)
                world = expert_td["agents", "action"].reshape(n, 3)
                target = student._world_to_ego(world, state[:, 3:7]) / 2.
            student(td)  # Student's complete chunk, NOT the teacher, drives Lee.
            if due and student.inference_calls > student.bootstrap_action_chunks:
                valid = active & ~student.goal_braking
                valid &= torch.isfinite(target).all(dim=-1)
                for key in ("latent", "goal", "proprio"):
                    chunks[key].append(captured[key][valid].cpu().float())
                chunks["target"].append(target[valid].cpu())
                chunks["student_first"].append(captured["prediction"][valid, 0].cpu().float())
                chunks["route"].append(torch.arange(n, device=cfg.device)[valid].cpu())
                chunks["physics_step"].append(torch.full((int(valid.sum()),), step, dtype=torch.long))
            td = env.step(td)["next"]
            stats = td["stats"]
            success = stats["reach_goal"].flatten().bool()
            collision = stats["collision"].flatten().bool()
            oob = stats["out_of_bounds"].flatten().bool()
            distance = (env.target_pos[:, 0] - td["info", "drone_state"][:, 0, :3]).norm(dim=-1)
            brake_failed = student.goal_braking & (distance > student.goal_stop_distance)
            finished = active & (success | collision | oob | brake_failed)
            for index in finished.nonzero().flatten().tolist():
                outcomes[index] = ("collision" if collision[index] else "out_of_bounds" if oob[index]
                                   else "goal_brake_failed" if brake_failed[index] else "success")
            active &= ~finished
            if (step + 1) % 100 == 0 or not active.any():
                print(f"[DAgger] physics_step={step+1} active={int(active.sum())}/{n}", flush=True)
            if not active.any():
                break
    data = {key: torch.cat(value) for key, value in chunks.items()}
    if not len(data["target"]):
        raise RuntimeError("No valid student-controlled chunk-start states")
    data["latent"] = data["latent"].half()
    metadata = {"format": "action-dagger-first-step-v1", "student_checkpoint": str(checkpoint),
                "teacher_sha256": teacher_sha, "samples": len(data["target"]),
                "map_seed": int(cfg.env.map_seed), "route_seed": int(cfg.env.route_seed),
                "outcomes": {key: outcomes.count(key) for key in set(outcomes)},
                "teacher_speed_over_2_fraction": float((data["target"].norm(dim=-1) > 1.).float().mean()),
                "first_action_mae_mps": float((data["student_first"] - data["target"]).abs().mean()*2),
                "student_first_units": "normalized (cloned before physical scaling)",
                "teacher_controls_student": False, "bootstrap_and_braking_excluded": True}
    data["metadata"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(data, temporary)
    os.replace(temporary, output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # Avoid known Isaac 2023 plugin-unload crashes after saving.


if __name__ == "__main__":
    main()
