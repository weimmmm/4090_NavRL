"""Train SAC and save a checkpoint every configured number of steps."""
import os
import random
import sys
from pathlib import Path


def _bootstrap_env():
    desired = os.environ.copy()
    desired.setdefault("CUDA_VISIBLE_DEVICES", "0")
    desired.setdefault("PYTHONHASHSEED", "3407")
    desired.setdefault("OMNI_KIT_RENDERER", "Vulkan")
    desired.setdefault("OMNI_KIT_NO_OPENGL_RENDERING", "1")
    if desired != os.environ and os.environ.get("NAVRL_BOOTSTRAPPED") != "1":
        desired["NAVRL_BOOTSTRAPPED"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], desired)


_bootstrap_env()

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
REPLAY_KEYS = (
    ("agents", "observation"), ("agents", "action_normalized"),
    ("next", "agents", "observation"), ("next", "agents", "reward"),
    ("next", "done"), ("next", "terminated"), ("next", "truncated"),
)


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _mean(value):
    """Convert a tensor metric to a scalar suitable for W&B."""
    return value.detach().float().mean().item()


def batch_metrics(batch):
    """Per-time-step diagnostics. These are not episode success rates."""
    return {
        "train/step.reward_mean": _mean(batch["next", "agents", "reward"]),
        "train/step.done_rate": _mean(batch["next", "done"]),
        "train/step.reach_goal_rate": _mean(batch["next", "stats", "reach_goal"]),
        "train/step.collision_rate": _mean(batch["next", "stats", "collision"]),
        "train/step.out_of_bound_rate": _mean(batch["next", "stats", "out_of_bound"]),
    }


def completed_episode_metrics(episode_stats):
    return {
        "train/" + (".".join(key) if isinstance(key, tuple) else str(key)): _mean(value)
        for key, value in episode_stats.items(True, True)
    }


@hydra.main(config_path=str(ROOT / "cfg"), config_name="train_sac", version_base=None)
def main(cfg):
    seed = int(cfg.seed)
    if int(cfg.algo.n_step) != 1:
        raise ValueError("This simplified trainer supports algo.n_step=1 only.")

    configure_seed(seed)
    sim_app = SimulationApp({"headless": bool(cfg.headless), "anti_aliasing": 1, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0})
    wandb_run = None
    try:
        from env import NavigationEnv
        from sac import SAC
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl import EpisodeStats, SyncDataCollector
        from omni_drones.utils.torchrl.transforms import VelController
        from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        from torchrl.envs.transforms import Compose, TransformedEnv
        from torchrl.envs.utils import ExplorationType

        wandb_kwargs = {
            "project": str(cfg.wandb.project),
            "name": str(cfg.wandb.name),
            "mode": str(cfg.wandb.mode),
        }
        # W&B accepts an OmegaConf object poorly in some versions; plain values
        # are enough here and keep this trainer independent from old run state.
        wandb_kwargs["config"] = {
            "seed": seed,
            "num_envs": int(cfg.env.num_envs),
            "max_frame_num": int(cfg.max_frame_num),
            "save_interval": int(cfg.simple_eval.save_interval),
            "algo": OmegaConf.to_container(cfg.algo, resolve=True),
        }
        if cfg.wandb.entity is not None:
            wandb_kwargs["entity"] = str(cfg.wandb.entity)
        if cfg.wandb.dir is not None:
            wandb_kwargs["dir"] = str(cfg.wandb.dir)
        if cfg.wandb.run_id is not None:
            wandb_kwargs["id"] = str(cfg.wandb.run_id)
        wandb_run = wandb.init(**wandb_kwargs)

        base_env = NavigationEnv(cfg)
        controller = LeePositionController(9.81, base_env.drone.params).to(cfg.device)
        env = TransformedEnv(base_env, Compose(VelController(controller, yaw_control=False))).train()
        env.set_seed(seed)
        configure_seed(seed)
        policy = SAC(cfg.algo, env.observation_spec, env.action_spec, cfg.device).train()
        replay_storage_device = torch.device(str(cfg.replay_buffer.storage_device))
        if replay_storage_device.type != "cpu":
            raise ValueError("This trainer keeps replay_buffer.storage_device=cpu; SAC batches are moved to GPU only for updates.")
        replay = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=int(cfg.algo.buffer_size), device=replay_storage_device),
            batch_size=int(cfg.algo.batch_size),
            pin_memory=bool(cfg.replay_buffer.pin_memory),
            prefetch=int(cfg.replay_buffer.prefetch),
        )
        collector = SyncDataCollector(
            env, policy=policy,
            frames_per_batch=int(cfg.env.num_envs) * int(cfg.algo.training_frame_num),
            total_frames=int(cfg.max_frame_num), device=cfg.device,
            return_same_td=True, exploration_type=ExplorationType.RANDOM,
        )
        episode_keys = [
            key for key in env.observation_spec.keys(True, True)
            if isinstance(key, tuple) and key[0] == "stats"
        ]
        completed_episodes = EpisodeStats(episode_keys)
        episode_batch_size = int(cfg.env.num_envs)
        output_dir = ROOT / "checkpoints"
        output_dir.mkdir(exist_ok=True)
        interval = int(cfg.simple_eval.save_interval)
        min_replay = max(int(cfg.algo.warmup_steps), int(cfg.algo.batch_size))
        print(f"[SAC] train={int(cfg.env.num_envs)} envs | save every {interval} steps | no in-training evaluation")
        print("[SAC] replay buffer=CPU (pinned) | sampled batches are copied to GPU for SAC updates")

        for step, batch in enumerate(collector, start=1):
            # The simulator emits GPU tensors. Keep the full replay buffer in
            # host memory, then SAC.update() copies only each sampled batch to GPU.
            replay.extend(batch.select(*REPLAY_KEYS).reshape(-1).detach().to(replay_storage_device))
            logs = batch_metrics(batch)
            logs["train/replay_size"] = len(replay)
            logs["train/collector_step"] = step
            completed_episodes.add(batch.reshape(-1))
            if len(completed_episodes) >= episode_batch_size:
                logs.update(completed_episode_metrics(completed_episodes.pop()))
            if len(replay) >= min_replay:
                logs.update({f"sac/{key}": value for key, value in policy.update(
                    replay, batch_size=int(cfg.algo.batch_size), tau=float(cfg.algo.tau)
                ).items()})
            logs["train/frames"] = int(collector._frames)
            wandb.log(logs, step=int(collector._frames))
            if step % interval != 0:
                continue
            checkpoint = output_dir / f"checkpoint_step_{step:07d}.pt"
            torch.save(policy.state_dict(), checkpoint)
            wandb.log({"checkpoint/saved": 1, "checkpoint/step": step}, step=int(collector._frames))
            print(f"[SAC] saved: {checkpoint} | train_frames={int(collector._frames)}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        sim_app.close()


if __name__ == "__main__":
    main()