import argparse
import hashlib
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
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp

# Some torchrun versions pass --local-rank on argv while others expose only
# LOCAL_RANK. Hydra does not need the CLI copy because we read the environment.
_filtered_argv = []
_skip_torchrun_rank = False
for _arg in sys.argv:
    if _skip_torchrun_rank:
        _skip_torchrun_rank = False
        continue
    if _arg in ("--local-rank", "--local_rank"):
        _skip_torchrun_rank = True
        continue
    if _arg.startswith("--local-rank=") or _arg.startswith("--local_rank="):
        continue
    _filtered_argv.append(_arg)
sys.argv[:] = _filtered_argv




FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


def _init_distributed(cfg):
    """Initialize torchrun metadata and split the vectorized environment."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return rank, 1, local_rank, False

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA.")
    distributed_cfg = cfg.get("distributed", {})
    backend = str(distributed_cfg.get("backend", "nccl"))
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed could not be initialized.")

    torch.cuda.set_device(local_rank)
    # With CUDA_VISIBLE_DEVICES=5,6,... each rank addresses its local visible
    # device as cuda:0, cuda:1, ... . Isaac Sim uses the same local index.
    cfg.device = f"cuda:{local_rank}"

    global_num_envs = int(cfg.env.num_envs)
    if global_num_envs % world_size != 0:
        raise ValueError(
            f"env.num_envs={global_num_envs} must be divisible by "
            f"WORLD_SIZE={world_size}."
        )
    cfg.env.num_envs = global_num_envs // world_size
    return rank, world_size, local_rank, True


def _broadcast_resume_state(values, device, distributed):
    if not distributed:
        return values
    state = torch.tensor(values, dtype=torch.float64, device=device)
    dist.broadcast(state, src=0)
    return state.tolist()


def _ordered_distributed_keys(values, device, world_size, distributed):
    """Return a stable key order and reject mismatched rank payloads."""
    keys = sorted(values)
    if not distributed:
        return keys

    payload = "\0".join(keys).encode("utf-8")
    signature = torch.tensor(
        list(hashlib.sha256(payload).digest()),
        dtype=torch.uint8,
        device=device,
    )
    signatures = [torch.empty_like(signature) for _ in range(world_size)]
    dist.all_gather(signatures, signature)
    if any(not torch.equal(signatures[0], item) for item in signatures[1:]):
        raise RuntimeError(
            "Metric keys differ across distributed ranks; refusing to mix "
            "unrelated values during log reduction."
        )
    return keys


def _reduce_info(info, device, world_size, distributed):
    """Average scalar metrics across rollout workers by stable key order."""
    keys = _ordered_distributed_keys(info, device, world_size, distributed)
    if not distributed:
        return {key: float(info[key]) for key in keys}

    values = torch.tensor(
        [float(info[key]) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values.div_(world_size)
    return {key: values[index].item() for index, key in enumerate(keys)}


def _reduce_episode_stats(stats, device, world_size, distributed):
    """Compute episode means weighted by each rank's completed episodes."""
    local_sums = {}
    local_count = None
    for key, value in stats.items(True, True):
        log_key = "train/" + (".".join(key) if isinstance(key, tuple) else key)
        value = value.float()
        metric_count = int(value.shape[0])
        if local_count is None:
            local_count = metric_count
        elif metric_count != local_count:
            raise RuntimeError(
                f"Episode metric {log_key} has {metric_count} samples, "
                f"expected {local_count}."
            )
        if value.numel() != metric_count:
            raise RuntimeError(
                f"Episode metric {log_key} must contain one scalar per episode."
            )
        local_sums[log_key] = value.sum().item()

    if local_count is None or local_count <= 0:
        raise RuntimeError("Cannot reduce an empty episode statistics batch.")

    keys = _ordered_distributed_keys(
        local_sums, device, world_size, distributed
    )
    packed = torch.tensor(
        [float(local_count)] + [float(local_sums[key]) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    total_count = packed[0].clamp_min(1.0)
    return {
        key: (packed[index + 1] / total_count).item()
        for index, key in enumerate(keys)
    }


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    rank, world_size, local_rank, distributed = _init_distributed(cfg)
    is_main = rank == 0

    run_id = cfg.wandb.get("run_id")
    checkpoint_path = cfg.get("checkpoint_path")
    if run_id is not None and checkpoint_path is None:
        raise ValueError(
            "Resuming a W&B run requires checkpoint_path; refusing to restart "
            "the policy while appending to an existing run."
        )

    # Simulation App. In distributed mode active_gpu is the local visible
    # device selected by LOCAL_RANK (CUDA_VISIBLE_DEVICES controls the mapping).
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

    # Only rank 0 owns the W&B run and checkpoint files.
    if is_main:
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
    else:
        run = None

    def summary_value(key):
        if not is_main:
            return 0.0
        value = run.summary.get(key, 0.0)
        return float(value) if value is not None else 0.0

    resume_state = _broadcast_resume_state(
        [
            summary_value("policy_frames"),
            summary_value("training_iteration"),
            summary_value("sim_time_seconds"),
            summary_value("physics_frames"),
            summary_value("wall_time_seconds"),
        ],
        cfg.device,
        distributed,
    )
    policy_frame_offset = resume_state[0]
    iteration_offset = int(resume_state[1])
    if run_id is not None:
        iteration_offset += 1
    cumulative_sim_time_seconds = resume_state[2]
    cumulative_physics_frames = resume_state[3]
    wall_time_offset = resume_state[4]

    # Navigation Training Environment
    env = NavigationEnv(cfg)

    # NavigationEnv delays cmd_vel before calling the unchanged velocity
    # controller on every physics tick.
    transformed_env = env.train()
    transformed_env.set_seed(int(cfg.seed) + rank * 1009)

    # Keep initial policy parameters identical across ranks while allowing the
    # simulator random streams to differ.
    if distributed:
        torch.manual_seed(int(cfg.seed))
        torch.cuda.manual_seed_all(int(cfg.seed))

    # PPO Policy
    policy = PPO(
        cfg.algo,
        transformed_env.observation_spec,
        transformed_env.action_spec,
        cfg.device,
        cfg.timing.reference_dt,
        distributed=distributed,
    )

    if checkpoint_path is not None:
        checkpoint = to_absolute_path(checkpoint_path)
        policy.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
        print(f"[NavRL]: loaded training checkpoint: {checkpoint}")
    policy.sync_distributed_parameters()

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
    global_remaining_policy_frames = int(cfg.max_frame_num - policy_frame_offset)
    if distributed:
        if global_remaining_policy_frames % world_size != 0:
            raise ValueError(
                "max_frame_num minus the resumed policy_frames must be "
                f"divisible by WORLD_SIZE={world_size}."
            )
        remaining_policy_frames = global_remaining_policy_frames // world_size
    else:
        remaining_policy_frames = global_remaining_policy_frames
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
        ).sum().item() * world_size
        cumulative_sim_time_seconds += batch_sim_time_seconds
        cumulative_physics_frames += batch_physics_frames
        mean_transition_dt = transition_dt.mean().item()
        mean_time_scale = mean_transition_dt / float(cfg.timing.reference_dt)
        policy_frames = policy_frame_offset + collector._frames * world_size
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
            "rollout_fps": collector._fps * world_size,
            "rollout_physics_fps": collector._fps * mean_time_scale * world_size,
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
        # The project collector keeps a rollout-side policy copy when a
        # policy device is specified. Refresh it after every PPO update so
        # the next batch is sampled with the current synchronized policy.
        collector.update_policy_weights_()
        info.update(train_loss_stats) # log training loss info

        # Calculate and log training episode stats
        episode_stats.add(data)
        local_stats_ready = len(episode_stats) >= transformed_env.num_envs
        if distributed:
            stats_ready = torch.tensor(
                int(local_stats_ready), device=cfg.device, dtype=torch.int32
            )
            dist.all_reduce(stats_ready, op=dist.ReduceOp.MIN)
            stats_ready = bool(stats_ready.item())
        else:
            stats_ready = local_stats_ready

        info = _reduce_info(info, cfg.device, world_size, distributed)
        if stats_ready: # log once all agents have finished an episode
            info.update(
                _reduce_episode_stats(
                    episode_stats.pop(),
                    cfg.device,
                    world_size,
                    distributed,
                )
            )

        # Update wand info
        info["wall_time_seconds"] = (
            wall_time_offset + time.perf_counter() - training_start_time
        )
        if is_main:
            run.log(info)


        # Save Model
        if distributed:
            dist.barrier()
        if is_main and training_iteration % cfg.save_interval == 0:
            ckpt_path = os.path.join(
                run.dir, f"checkpoint_{training_iteration}.pt"
            )
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", training_iteration)
        if distributed:
            dist.barrier()

    if distributed:
        dist.barrier()
    if is_main:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)
        wandb.finish()
    sim_app.close()
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
    
