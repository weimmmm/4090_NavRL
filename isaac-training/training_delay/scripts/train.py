import argparse
import os
import sys

OMNIDRONES_SOURCE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "OmniDrones")
)
if OMNIDRONES_SOURCE not in sys.path:
    sys.path.insert(0, OMNIDRONES_SOURCE)

import hydra
import datetime
import time
import wandb
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp




FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    run_id = cfg.wandb.get("run_id")
    checkpoint_path = cfg.get("checkpoint_path")
    if run_id is not None and checkpoint_path is None:
        raise ValueError(
            "Resuming a W&B run requires checkpoint_path; refusing to restart "
            "the policy while appending to an existing run."
        )

    # Simulation App
    sim_app = SimulationApp(
        {
            "headless": cfg.headless,
            "anti_aliasing": 1,
            "active_gpu": int(str(cfg.device).split(":")[-1]),
            "physics_gpu": int(str(cfg.device).split(":")[-1]),
            "multi_gpu": False,
        }
    )

    from env import NavigationEnv
    from omni_drones.utils.torchrl import EpisodeStats, SyncDataCollector
    from ppo import PPO
    from torchrl.envs.utils import ExplorationType

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Use Wandb to monitor training
    if run_id is None:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=wandb_cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )

    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=wandb_cfg,
            mode=cfg.wandb.mode,
            id=run_id,
            resume="must"
        )

    def summary_value(key):
        value = run.summary.get(key, 0.0)
        return float(value) if value is not None else 0.0

    policy_frame_offset = summary_value("policy_frames")
    iteration_offset = int(summary_value("training_iteration"))
    if run_id is not None:
        iteration_offset += 1
    cumulative_sim_time_seconds = summary_value("sim_time_seconds")
    cumulative_physics_frames = summary_value("physics_frames")
    wall_time_offset = summary_value("wall_time_seconds")

    # Navigation Training Environment
    env = NavigationEnv(cfg)

    # NavigationEnv delays cmd_vel before calling the unchanged velocity
    # controller on every physics tick.
    transformed_env = env.train()
    transformed_env.set_seed(cfg.seed)    
    # PPO Policy
    policy = PPO(
        cfg.algo,
        transformed_env.observation_spec,
        transformed_env.action_spec,
        cfg.device,
        cfg.timing.reference_dt,
    )

    if checkpoint_path is not None:
        checkpoint = to_absolute_path(checkpoint_path)
        policy.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
        print(f"[NavRL]: loaded training checkpoint: {checkpoint}")

    # checkpoint = "/home/zhefan/catkin_ws/src/navigation_runner/scripts/ckpts/checkpoint_2500.pt"
    # checkpoint = "/home/xinmingh/RLDrones/navigation/scripts/nav-ros/navigation_runner/ckpts/checkpoint_36000.pt"
    # policy.load_state_dict(torch.load(checkpoint))
    
    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector
    remaining_policy_frames = int(cfg.max_frame_num - policy_frame_offset)
    if remaining_policy_frames <= 0:
        raise ValueError(
            f"Run already has {policy_frame_offset:.0f} policy frames, which "
            f"reaches max_frame_num={cfg.max_frame_num}."
        )
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=remaining_policy_frames,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )

    training_start_time = time.perf_counter()

    # Training Loop
    for i, data in enumerate(collector):
        # print("data: ", data)
        # print("============================")
        # Log Info
        transition_dt = data["next", "stats", "transition_dt"].float()
        inference_delay = data["next", "stats", "inference_delay"].float()
        command_delay = data["next", "stats", "command_delay"].float()
        publisher_wait_delay = data[
            "next", "stats", "publisher_wait_delay"
        ].float()
        transport_delay = data["next", "stats", "transport_delay"].float()
        total_delay = data["next", "stats", "total_delay"].float()
        sampled_inference_delay = data[
            "next", "stats", "sampled_inference_delay"
        ].float()
        sampled_command_delay = data[
            "next", "stats", "sampled_command_delay"
        ].float()
        sampled_total_delay = data[
            "next", "stats", "sampled_total_delay"
        ].float()
        pending_command_age = data[
            "next", "stats", "pending_command_age"
        ].float()
        command_queue_depth = data[
            "next", "stats", "command_queue_depth"
        ].float()
        command_publish_count = data[
            "next", "stats", "command_publish_count"
        ].float()
        batch_sim_time_seconds = (
            transition_dt.sum().item() / transformed_env.num_envs
        )
        batch_physics_frames = (
            transition_dt / float(cfg.timing.reference_dt)
        ).sum().item()
        cumulative_sim_time_seconds += batch_sim_time_seconds
        cumulative_physics_frames += batch_physics_frames
        mean_transition_dt = transition_dt.mean().item()
        mean_time_scale = mean_transition_dt / float(cfg.timing.reference_dt)
        policy_frames = policy_frame_offset + collector._frames
        training_iteration = iteration_offset + i
        info = {
            "training_iteration": training_iteration,
            "env_frames": policy_frames,
            "policy_frames": policy_frames,
            "physics_frames": cumulative_physics_frames,
            "sim_time_seconds": cumulative_sim_time_seconds,
            "wall_time_seconds": (
                wall_time_offset + time.perf_counter() - training_start_time
            ),
            "rollout_fps": collector._fps,
            "rollout_physics_fps": collector._fps * mean_time_scale,
            "timing/mean_transition_dt": mean_transition_dt,
            "timing/mean_measured_inference_delay": (
                inference_delay.mean().item()
            ),
            "timing/mean_measured_command_delay": command_delay.mean().item(),
            "timing/mean_publisher_wait_delay": (
                publisher_wait_delay.mean().item()
            ),
            "timing/mean_measured_transport_delay": (
                transport_delay.mean().item()
            ),
            "timing/mean_measured_total_delay": total_delay.mean().item(),
            "timing/mean_sampled_inference_delay": (
                sampled_inference_delay.mean().item()
            ),
            "timing/mean_sampled_command_delay": (
                sampled_command_delay.mean().item()
            ),
            "timing/mean_sampled_total_delay": (
                sampled_total_delay.mean().item()
            ),
            "timing/mean_pending_command_age": (
                pending_command_age.mean().item()
            ),
            "timing/mean_command_queue_depth": (
                command_queue_depth.mean().item()
            ),
            "timing/mean_command_publish_count": (
                command_publish_count.mean().item()
            ),
        }

        # Train Policy
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats) # log training loss info

        # Calculate and log training episode stats
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs: # log once all agents have finished an episode
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # Update wand info
        info["wall_time_seconds"] = (
            wall_time_offset + time.perf_counter() - training_start_time
        )
        run.log(info)


        # Save Model
        if training_iteration % cfg.save_interval == 0:
            ckpt_path = os.path.join(
                run.dir, f"checkpoint_{training_iteration}.pt"
            )
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", training_iteration)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    
