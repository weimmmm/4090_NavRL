"""Fly, query PPO every K=10, synchronize one learner update, then act.

PPO NEVER controls the drone. Only action index zero is supervised online.
The fresh complete chunk is generated AFTER the synchronized weight update;
its remaining nine actions are held fixed until the next replan. No gradient
passes through Isaac. A separate four-GPU learner computes supervised grads.
"""
from pathlib import Path
from multiprocessing.connection import Client
from types import SimpleNamespace
import hashlib
import json
import os
import sys

from collect_action_dagger import static_observation
from wam_closed_loop_eval import WamLeePolicy, CONFIG_DIR, _resolve
import hydra
from omni.isaac.kit import SimulationApp


def apply_tail(student, packet):
    import torch
    parameters = dict(student.model.named_parameters())
    names = packet["names"]
    branch = "diffusion" if getattr(student.model, "flow_only", False) else "policy"
    allowed = tuple(branch+suffix for suffix in
                    (".blocks.6.", ".blocks.7.", ".norm.", ".out."))
    prefixes = allowed[:2] if packet.get("freeze_policy_output_head", False) else allowed
    expected = [name for name in parameters if name.startswith(prefixes)]
    if names != expected:
        raise ValueError("Learner and actor frozen/trainable split differs")
    array = packet["weights"]
    if sum(parameters[name].numel() for name in names) != array.size:
        raise ValueError("Malformed synchronized tail weights")
    flat = torch.from_numpy(array).to(student.device)
    offset = 0
    with torch.no_grad():
        for name in names:
            parameter = parameters[name]
            length = parameter.numel()
            parameter.copy_(flat[offset:offset+length].view_as(parameter))
            offset += length


def install_inference_microbatches(student, limit):
    """Bound actor memory without changing the synchronized online batch.

Frozen VAE/policy normalization is per sample, not BatchNorm. We assemble
every drone's inputs before one learner request, and split only inference.
"""
    import torch
    encode = student.vae.encode
    predict = student.model.predict_action

    def encode_chunks(image, *args, **kwargs):
        if len(image) <= limit:
            return encode(image, *args, **kwargs)
        modes = [encode(image[start:start+limit], *args, **kwargs).latent_dist.mode()
                 for start in range(0, len(image), limit)]
        mode = torch.cat(modes)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: mode))

    def predict_chunks(latent, goal, proprio, *legacy, **kwargs):
        if len(latent) <= limit:
            return predict(latent, goal, proprio, *legacy, **kwargs)
        parts = []
        for start in range(0, len(latent), limit):
            options = dict(kwargs)
            if options.get("initial_noise") is not None:
                options["initial_noise"] = options["initial_noise"][start:start+limit]
            parts.append(predict(*(value[start:start+limit] for value in
                                   (latent, goal, proprio, *legacy)), **options))
        return torch.cat(parts)

    student.vae.encode = encode_chunks
    return predict_chunks


@hydra.main(config_path=CONFIG_DIR, config_name="train", version_base=None)
def main(cfg):
    import torch
    root = Path(__file__).resolve().parents[2]
    checkpoint = _resolve(root, cfg.wam_eval.checkpoint)
    teacher_path = _resolve(root, cfg.online.teacher_checkpoint)
    output = _resolve(root, cfg.online.output)
    if output.exists():
        raise FileExistsError(output)
    if float(cfg.sim.dt) != .016 or int(cfg.wam_eval.replan_interval_steps) != 10:
        raise ValueError("Keep original dt=.016 and K=10")
    if int(cfg.env_dyn.num_obstacles) != 0:
        raise ValueError("Static-map pilot only")
    teacher_sha = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    if teacher_sha != "1fb660d0f7ce3f136daa6967095149b8ad1a34a32eb3e67216bef22d856a323e":
        raise ValueError("Teacher checkpoint differs from the dataset PPO")
    app = SimulationApp({"headless": True, "anti_aliasing": 1,
                         "active_gpu": int(os.environ.get("ISAAC_ACTIVE_GPU", "0")),
                         "physics_gpu": int(str(cfg.device).split(":")[-1]), "multi_gpu": False})
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
    from generate_fixed_environments import generate_routes, tensor_dict
    from lidar_wam.runner.online_supervision import online_teacher_target

    env = NavigationEnv(cfg)
    env.eval()
    env.set_seed(int(cfg.seed))
    lee = LeePositionController(9.81, env.drone.params).to(cfg.device)
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
    for name, shape in (("lidar", (n, 1, 36, 4)), ("dynamic_obstacle", (n, 1, 5, 10)),
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
    local_learner = bool(cfg.online.get("local_learner", False))
    if local_learner:
        from local_flow_online_sft import LocalFlowSFT
        connection = LocalFlowSFT(cfg, checkpoint, output.parent)
    else:
        connection = Client(str(cfg.online.socket), family="AF_UNIX")
    packet = connection.recv()
    target_mode = str(cfg.online.get("teacher_target_mode", "raw"))
    if packet.get("teacher_target_mode", "raw") != target_mode:
        raise ValueError("Actor and learner teacher-label modes differ")
    if packet.get("supervision_horizon", 1) != int(cfg.online.get("supervision_horizon", 1)):
        raise ValueError("Actor and learner supervision horizons differ")
    version = int(packet["step"])
    phase_end = int(cfg.online.target_updates)
    physics_total = queries = samples = 0
    episode_summaries = []
    output.parent.mkdir(parents=True, exist_ok=True)

    if bool(cfg.online.get("refill_enabled", False)):
        from online_refill import run_refill
        metadata = run_refill(cfg, env, WamLeePolicy, lee, checkpoint, teacher, teacher_sensor,
                              connection, packet, centers, sizes,
                              connection.attach if local_learner else apply_tail,
                              install_inference_microbatches, static_observation)
        output.write_text(json.dumps(metadata, indent=2)+"\n")
        print(json.dumps(metadata), flush=True)
        if local_learner:
            connection.finalize(metadata)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
        for episode in range(int(cfg.online.get("max_episodes", 8))):
            # Keep this phase's map fixed; use new opposite-edge routes at every
            # episode reset. The held-out baseline evaluation map is never used.
            if episode:
                env.fixed_scene.update(tensor_dict(generate_routes(
                    int(cfg.env.route_seed)+episode, int(env.fixed_scene["num_drones"]))))
            student = WamLeePolicy(cfg, env, lee, checkpoint, checkpoint)
            if student.use_past_actions or student.lidar_history_frames != 1:
                raise ValueError("Expected baseline's single-frame/no-past deployment")
            apply_tail(student, packet)
            pending = {}
            predict = install_inference_microbatches(student, int(cfg.online.get("inference_batch_size", 128)))

            def online_predict(latent, goal, proprio, past, mask):
                nonlocal packet, version, samples
                valid = pending["valid"]
                if version < phase_end and bool(valid.any()):
                    fresh = {"latent": latent[valid].float().cpu().numpy(),
                             "goal": goal[valid].float().cpu().numpy(),
                             "proprio": proprio[valid].float().cpu().numpy(),
                             "target": pending["target"][valid].float().cpu().numpy()}
                    connection.send({"command": "update", "batch": fresh,
                                     "student_version": version, "physics_step": physics_total,
                                     "episode": episode})
                    packet = connection.recv()
                    if int(packet["step"]) != version+1:
                        raise RuntimeError("Exactly one learner update is required per fresh chunk")
                    version += 1
                    samples += int(valid.sum())
                    apply_tail(student, packet)
                    if version == 1 or version % 10 == 0:
                        print(json.dumps({"update": version, "physics_step": physics_total,
                                          "episode": episode, "fresh_states": int(valid.sum()),
                                          "teacher_queries": queries, "student_controls": True}), flush=True)
                # No PPO action is substituted here: the UPDATED student
                # generates all ten actions, held constant within this chunk.
                return predict(latent, goal, proprio, past, mask)

            student.model.predict_action = online_predict
            active = torch.ones(n, dtype=torch.bool, device=cfg.device)
            outcomes = ["unfinished"]*n
            td = env.reset()
            for local_step in range(int(cfg.env.max_episode_length)):
                student.set_active(active)
                if local_step % 10 == 0:
                    teacher_sensor.update(float(cfg.sim.dt), force_recompute=True)
                    expert_td = td.clone()
                    distance = (teacher_sensor.data.ray_hits_w-teacher_sensor.data.pos_w[:, None]).norm(dim=-1)
                    distance = torch.nan_to_num(distance, nan=4., posinf=4.).clamp(0., 4.)
                    expert_td["agents", "observation", "lidar"] = 4.-distance.reshape(n, 1, 36, 4)
                    expert_td["agents", "observation", "dynamic_obstacle"] = torch.zeros(n, 1, 5, 10, device=cfg.device)
                    state = td["info", "drone_state"][:, 0, :13]
                    expert_td["agents", "observation", "static_obstacle"] = static_observation(
                        state[:, :3], expert_td["agents", "observation", "direction"], centers, sizes)
                    teacher(expert_td)
                    queries += 1
                    target = student._world_to_ego(expert_td["agents", "action"].reshape(n, 3), state[:, 3:7])/2.
                    target = online_teacher_target(target, target_mode)
                    valid = active & ~student.goal_braking & torch.isfinite(target).all(-1)
                    if student.goal_stop_enabled:
                        valid &= (env.target_pos[:, 0]-state[:, :3]).norm(dim=-1)>student.goal_stop_distance
                    if student.inference_calls < student.bootstrap_action_chunks:
                        valid.zero_()
                    pending.update(valid=valid, target=target)
                before = version
                student(td)
                if local_step % 10:
                    assert before == version, "No weight update inside a chunk"
                td = env.step(td)["next"]
                physics_total += 1
                stats = td["stats"]
                success = stats["reach_goal"].flatten().bool()
                collision = stats["collision"].flatten().bool()
                oob = stats["out_of_bounds"].flatten().bool()
                goal_distance = (env.target_pos[:, 0]-td["info", "drone_state"][:, 0, :3]).norm(dim=-1)
                brake_failed = student.goal_braking & (goal_distance>student.goal_stop_distance)
                finished = active & (success | collision | oob | brake_failed)
                for index in finished.nonzero().flatten().tolist():
                    outcomes[index] = ("collision" if collision[index] else "out_of_bounds" if oob[index]
                                       else "goal_brake_failed" if brake_failed[index] else "success")
                active &= ~finished
                if (local_step+1) % 100 == 0:
                    print(f"[online actor] episode={episode} local_physics={local_step+1} "
                          f"active={int(active.sum())}/{n} learner_updates={version}", flush=True)
                if not active.any() or (version >= phase_end and (local_step+1) % 10 == 0):
                    break
            partial = version >= phase_end and bool(active.any())
            if not partial:
                for index in active.nonzero().flatten().tolist():
                    outcomes[index] = "timeout"
            summary = {"episode": episode, "partial": partial,
                       "counts": {key: outcomes.count(key) for key in set(outcomes)}}
            episode_summaries.append(summary)
            print(json.dumps({"training_rollout_not_heldout_eval": summary}), flush=True)
            # Remove the hook before constructing the next reset policy.
            student.model.predict_action = predict
            if version >= phase_end:
                break
        if version < phase_end:
            raise RuntimeError("Episode budget exhausted before the requested online updates")
    command = "finish" if bool(cfg.online.get("final_session", False)) else "session_done"
    connection.send({"command": command, "episodes": len(episode_summaries)})
    ack = connection.recv()
    if int(ack["step"]) != version:
        raise RuntimeError("Learner/actor final version differs")
    connection.close()
    metadata = {"updates": version, "physics_steps": physics_total, "teacher_queries": queries,
                "num_envs": n, "inference_microbatch_size": int(cfg.online.get("inference_batch_size", 128)),
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                "fresh_supervised_states": samples, "query_interval_steps": 10,
                "supervised_action_index": 0, "student_update_before_chunk_generation": True,
                "teacher_target_mode": target_mode,
                "weights_fixed_inside_chunk": True, "teacher_controls_student": False,
                "training_episodes": episode_summaries, "map_seed": int(cfg.env.map_seed)}
    output.write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
