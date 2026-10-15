"""Per-slot online SFT reset utilities; never reset the rest of the fleet.

Refill is performed at the next K=10 boundary (at most nine physics steps
after termination). Reset observations are refreshed without advancing
physics or double-counting unreset environments' reward/success counters.
"""
import numpy as np
import os
from pathlib import Path
import torch
from lidar_wam.runner.online_supervision import online_teacher_target, target_diagnostics


def sample_routes(count, rng, edge=24., height=2.):
    route = rng.integers(0, 4, count, dtype=np.int64)
    start = np.zeros((count, 3), np.float32)
    goal = np.zeros_like(start)
    start[:, 2] = goal[:, 2] = height
    lateral_start = rng.uniform(-edge, edge, count)
    lateral_goal = rng.uniform(-edge, edge, count)
    for direction in range(4):
        keep = route == direction
        axis = 1 if direction < 2 else 0
        other = 1-axis
        sign = 1 if direction in (0, 3) else -1
        start[keep, axis], goal[keep, axis] = sign*edge, -sign*edge
        start[keep, other], goal[keep, other] = lateral_start[keep], lateral_goal[keep]
    yaw = np.arctan2(goal[:, 1]-start[:, 1], goal[:, 0]-start[:, 0])
    quaternion = np.zeros((count, 4), np.float32)
    quaternion[:, 0], quaternion[:, 3] = np.cos(yaw/2), np.sin(yaw/2)
    return {"route_id": torch.from_numpy(route), "start_position": torch.from_numpy(start),
            "goal_position": torch.from_numpy(goal), "initial_orientation_wxyz": torch.from_numpy(quaternion)}


def reset_policy_slots(student, ids, state):
    """Clear only selected rows; keep global K=10 clock/model/other rows intact."""
    if student.required_history_frames != 1 or student.use_past_actions:
        raise ValueError("Refill currently supports the single-frame/no-past baseline only")
    for name in ("past", "past_world", "past_mask", "chunk", "goal_braking", "reference_started",
                 "previous_reference_velocity", "safety_heading", "last_ego_command",
                 "last_world_command", "last_reference_velocity_world"):
        getattr(student, name)[ids] = 0
    for image in student.history:
        image[ids] = 0
    if student.previous_velocity is not None:
        student.previous_velocity[ids] = state[ids, 7:10]
    student.reference_position[ids] = state[ids, :3]
    student.reference_yaw[ids] = student._quaternion_yaw(state[ids, 3:7])
    student.chunk_frame_yaw[ids] = student._goal_frame_yaw(state[:, :3])[ids]
    student.warmup_chunks[ids] = student.bootstrap_action_chunks


def refill_slots(env, student, td, ids, rng):
    """Refresh sensor/teacher conditions after a partial physical reset."""
    from tensordict import TensorDict
    if not len(ids):
        return td
    cpu_ids = ids.cpu().long()
    for key, values in sample_routes(len(ids), rng).items():
        env.fixed_scene[key][cpu_ids] = values
    # NavigationEnv computes rewards and terminal counters while creating
    # observations. Save them before reset, and restore the untouched rows.
    saved_stats = env.stats.clone()
    fields = ("goal_stop_count", "prev_drone_vel_w", "reward", "terminated", "truncated")
    saved = {key: getattr(env, key).clone() for key in fields}
    untouched = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    untouched[ids] = False
    before_state = td["info", "drone_state"].clone()
    mask = ~untouched
    reset_td = TensorDict({"_reset": mask[:, None]}, [env.num_envs], device=env.device)
    env.reset(reset_td)
    env.lidar.update(env.dt, force_recompute=True)
    fresh = env._compute_state_and_obs()
    env.stats.copy_(saved_stats)
    env.stats[ids] = 0.
    for key, value in saved.items():
        destination = getattr(env, key)
        destination.copy_(value)
        destination[ids] = 0
    fresh["stats"] = env.stats.clone()
    fresh.update(env._compute_reward_and_done())
    state = fresh["info", "drone_state"][:, 0, :13]
    if not torch.allclose(fresh["info", "drone_state"][untouched], before_state[untouched], atol=1e-5, rtol=0):
        raise RuntimeError("Partial refill changed unreset drones' physical states")
    expected = env.fixed_scene["start_position"][cpu_ids].to(env.device)
    if not torch.allclose(state[ids, :3], expected, atol=1e-4, rtol=0):
        raise RuntimeError("Reset slots did not reach their new start poses")
    if not torch.all(env.progress_buf[ids] == 0):
        raise RuntimeError("Per-slot episode age did not reset")
    reset_policy_slots(student, ids, state)
    return fresh


def make_refill_policy(base_class, *args):
    class RefillPolicy(base_class):
        def __init__(self, *args):
            super().__init__(*args)
            self.warmup_chunks = torch.full((self.num_envs,), self.bootstrap_action_chunks,
                                           dtype=torch.long, device=self.device)

        def _sample_chunk(self, state, acceleration_world):
            warm = self.warmup_chunks > 0
            super()._sample_chunk(state, acceleration_world)
            self.chunk[warm] = 0
            self.chunk[warm, :, 0] = self.bootstrap_forward_speed
            self.warmup_chunks[warm] -= 1
    return RefillPolicy(*args)


def select_fresh(valid, capacity, generator):
    """Select unique CURRENT states; reserve environments never become replay."""
    available = int(valid.sum())
    if available <= capacity:
        return valid.clone(), available
    eligible = valid.nonzero().flatten()
    selected = eligible[torch.randperm(available, generator=generator, device=valid.device)[:capacity]]
    mask = torch.zeros_like(valid)
    mask[selected] = True
    return mask, available


def run_refill(cfg, env, base_policy, lee, checkpoint, teacher, teacher_sensor,
               connection, packet, centers, sizes, apply_weights, inference_chunks,
               static_features):
    if int(cfg.online.get("supervision_horizon", 1)) == 10:
        from online_chunk_sft import run_chunk_supervised
        return run_chunk_supervised(cfg, env, base_policy, lee, checkpoint, teacher, teacher_sensor,
                                    connection, packet, centers, sizes, apply_weights,
                                    inference_chunks, static_features)
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    n = env.num_envs
    student = make_refill_policy(base_policy, cfg, env, lee, checkpoint, checkpoint)
    target_mode = str(cfg.online.get("teacher_target_mode", "raw"))
    if packet.get("teacher_target_mode", "raw") != target_mode:
        raise ValueError("Actor and learner teacher-label modes differ")
    if target_mode == "student-norm" and (
            abs(student.max_reference_speed-2.)>1e-6
            or not torch.allclose(student.action_target_scale,
                                  torch.full_like(student.action_target_scale, 2.))):
        raise ValueError("Projected labels require the original 2 m/s student contract")
    apply_weights(student, packet)
    predict = inference_chunks(student, int(cfg.online.get("inference_batch_size", 128)))
    version = initial_version = int(packet["step"])
    end = int(cfg.online.target_updates)
    capacity = int(cfg.online.learn_batch_capacity)
    minimum_batch = int(cfg.online.get("minimum_batch_size", 1))
    if not 1 <= minimum_batch <= min(n, capacity):
        raise ValueError("Minimum fresh batch must fit the fleet/batch capacity")
    route_rng = np.random.default_rng(int(cfg.env.route_seed)+1701)
    selector_rng = torch.Generator(device=env.device).manual_seed(int(cfg.seed)+31)
    active = torch.ones(n, dtype=torch.bool, device=env.device)
    pending_reset = torch.zeros_like(active)
    pending = {}
    physics = queries = samples = replacements = completed = skipped_small = 0
    counts = {key: 0 for key in ("success", "collision", "out_of_bounds", "goal_brake_failed", "timeout")}
    test_resets = 0

    def online_predict(latent, goal, proprio, past, mask):
        nonlocal version, packet, samples
        valid = pending["valid"]
        if version < end and int(valid.sum()) >= minimum_batch:
            fresh = {"latent": latent[valid].float().cpu().numpy(),
                     "goal": goal[valid].float().cpu().numpy(),
                     "proprio": proprio[valid].float().cpu().numpy(),
                     "target": pending["target"][valid].float().cpu().numpy()}
            diagnostics = {"available_online_states": pending["available"],
                           "fresh_batch_deficit": max(0, capacity-int(valid.sum())),
                           "warmup_slots": int((student.warmup_chunks>0).sum()),
                           "refilled_slots": replacements, "completed_trajectories": completed,
                           "skipped_small_batches": skipped_small,
                           "actor_pid": os.getpid(), "full_fleet_resets": 1}
            diagnostics.update(target_diagnostics(pending["raw_target"], pending["target"], valid))
            connection.send({"command": "update", "batch": fresh, "student_version": version,
                             "physics_step": physics, "episode": completed, "diagnostics": diagnostics})
            packet = connection.recv()
            if int(packet["step"]) != version+1:
                raise RuntimeError("One fresh update is required per K=10 boundary")
            version += 1
            samples += int(valid.sum())
            apply_weights(student, packet)
            if version == initial_version+1 or version % 10 == 0:
                print({"update": version, "physics_step": physics, "fresh_states": int(valid.sum()),
                       **diagnostics}, flush=True)
        return predict(latent, goal, proprio, past, mask)

    student.model.predict_action = online_predict
    td = env.reset()
    initial = td["info", "drone_state"][:, 0, :3]
    expected = env.fixed_scene["start_position"][:n].to(env.device)
    if not torch.allclose(initial, expected, atol=1e-4, rtol=0):
        raise RuntimeError("Cloning spacing leaked into actual world-frame flight routes")
    budget = (end-initial_version+student.bootstrap_action_chunks+32)*10
    stop_file = str(cfg.online.get("stop_file", ""))
    with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
        while physics < budget:
            if physics % 10 == 0:
                if stop_file and Path(stop_file).exists():
                    # Graceful cancellation is only honored at a real chunk
                    # boundary. It is NOT a validation/reset control channel.
                    break
                ids = pending_reset.nonzero().flatten()
                if len(ids):
                    td = refill_slots(env, student, td, ids, route_rng)
                    replacements += len(ids)
                    active[ids] = True
                    pending_reset[ids] = False
                    print(f"[refill] physics={physics} slots={len(ids)} total={replacements}", flush=True)
                teacher_sensor.update(env.dt, force_recompute=True)
                expert_td = td.clone()
                distance = (teacher_sensor.data.ray_hits_w-teacher_sensor.data.pos_w[:, None]).norm(dim=-1)
                distance = torch.nan_to_num(distance, nan=4., posinf=4.).clamp(0., 4.)
                expert_td["agents", "observation", "lidar"] = 4.-distance.reshape(n, 1, 36, 4)
                expert_td["agents", "observation", "dynamic_obstacle"] = torch.zeros(n, 1, 5, 10, device=env.device)
                state = td["info", "drone_state"][:, 0, :13]
                expert_td["agents", "observation", "static_obstacle"] = static_features(
                    state[:, :3], expert_td["agents", "observation", "direction"], centers, sizes)
                teacher(expert_td)
                queries += 1
                raw_target = student._world_to_ego(expert_td["agents", "action"].reshape(n, 3), state[:, 3:7])/2.
                target = online_teacher_target(raw_target, target_mode)
                valid = active & ~student.goal_braking & (student.warmup_chunks == 0)
                valid &= torch.isfinite(target).all(-1)
                if student.goal_stop_enabled:
                    valid &= (env.target_pos[:, 0]-state[:, :3]).norm(dim=-1)>student.goal_stop_distance
                selected, available = select_fresh(valid, capacity, selector_rng)
                if 0 < int(selected.sum()) < minimum_batch:
                    skipped_small += 1
                pending.update(valid=selected, target=target, raw_target=raw_target, available=available)
            student.set_active(active)
            before = version
            student(td)
            if physics % 10:
                assert before == version, "Weights changed inside a committed chunk"
            td = env.step(td)["next"]
            physics += 1
            stats = td["stats"]
            success = stats["reach_goal"].flatten().bool()
            collision = stats["collision"].flatten().bool()
            oob = stats["out_of_bounds"].flatten().bool()
            timeout = td["truncated"].flatten().bool()
            goal_distance = (env.target_pos[:, 0]-td["info", "drone_state"][:, 0, :3]).norm(dim=-1)
            brake_failed = student.goal_braking & (goal_distance>student.goal_stop_distance)
            finished = active & (success | collision | oob | brake_failed | timeout)
            # Match the existing evaluator's terminal-type priority.
            remaining = finished.clone()
            for key, mask in (("collision", collision), ("out_of_bounds", oob),
                              ("goal_brake_failed", brake_failed), ("success", success),
                              ("timeout", timeout)):
                hit = remaining & mask
                counts[key] += int(hit.sum())
                remaining &= ~hit
            completed += int(finished.sum())
            active[finished] = False
            pending_reset |= finished
            # Test-only reset injection validates row isolation/warmup early,
            # without pretending these artificial terminations are real SR.
            if bool(cfg.online.get("test_force_refill", False)) and physics == 20:
                forced = torch.tensor([0, n-1], device=env.device)
                active[forced] = False
                pending_reset[forced] = True
                test_resets += len(forced)
            if physics % 100 == 0:
                print(f"[refill actor] physics={physics} active={int(active.sum())}/{n} "
                      f"updates={version} completed={completed} refilled={replacements}", flush=True)
            if version >= end and physics % 10 == 0:
                break
    requested_stop = bool(stop_file and Path(stop_file).exists())
    if version < end and not requested_stop:
        raise RuntimeError("No-progress safety budget exceeded in online refill loop")
    if test_resets and replacements < test_resets:
        raise RuntimeError("Forced-reset smoke test never exercised the refill path")
    command = "finish" if bool(cfg.online.get("final_session", False)) else "session_done"
    connection.send({"command": command, "episodes": completed})
    ack = connection.recv()
    if int(ack["step"]) != version:
        raise RuntimeError("Final actor/learner versions differ")
    connection.close()
    return {"updates": version, "start_update": initial_version, "physics_steps": physics,
            "teacher_queries": queries, "fresh_supervised_states": samples,
            "num_envs": n, "online_batch_capacity": capacity,
            "actor_pid": os.getpid(), "full_fleet_resets": 1,
            "stopped_by_request": requested_stop,
            "minimum_online_batch": minimum_batch, "skipped_small_batches": skipped_small,
            "uses_offline_replay": False, "refill_enabled": True,
            "teacher_target_mode": target_mode,
            "completed_trajectories": completed, "refilled_slots": replacements,
            "forced_test_resets": test_resets, "training_counts_not_heldout_eval": counts,
            "query_interval_steps": 10, "supervised_action_index": 0,
            "student_update_before_chunk_generation": True, "weights_fixed_inside_chunk": True,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
