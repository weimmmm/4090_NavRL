"""Train SAC and save a checkpoint every configured number of steps."""
import os
import random
import sys
import types
from pathlib import Path


def _bootstrap_env():
    desired = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[4])
    repo_omnidrones = str(
        Path(__file__).resolve().parents[3] / "third_party" / "OmniDrones"
    )
    python_paths = [item for item in desired.get("PYTHONPATH", "").split(os.pathsep) if item]
    if not python_paths or python_paths[0] != repo_omnidrones:
        desired["PYTHONPATH"] = os.pathsep.join(
            [repo_omnidrones, *[item for item in python_paths if item != repo_omnidrones]]
        )
    # torchrun needs every requested GPU to remain visible. A plain one-process
    # launch keeps the historical physical-GPU-0 default.
    if int(desired.get("WORLD_SIZE", "1")) == 1:
        desired.setdefault("CUDA_VISIBLE_DEVICES", "0")
    desired.setdefault("PYTHONHASHSEED", "3407")
    desired.setdefault("OMNI_KIT_RENDERER", "Vulkan")
    desired.setdefault("OMNI_KIT_NO_OPENGL_RENDERING", "1")
    # Docker bind mounts have host ownership, which makes W&B's optional Git
    # probe fail before training starts. Source tracking is not needed here.
    desired.setdefault("WANDB_DISABLE_GIT", "true")
    # The bind-mounted repository is owned by the host user, while Isaac runs
    # as root. Pass Git's safe.directory through process-local config so old
    # W&B versions can inspect the working tree without changing global config.
    git_config_count = int(desired.get("GIT_CONFIG_COUNT", "0"))
    safe_directories = {
        desired.get(f"GIT_CONFIG_VALUE_{index}")
        for index in range(git_config_count)
        if desired.get(f"GIT_CONFIG_KEY_{index}") == "safe.directory"
    }
    if repo_root not in safe_directories:
        desired[f"GIT_CONFIG_KEY_{git_config_count}"] = "safe.directory"
        desired[f"GIT_CONFIG_VALUE_{git_config_count}"] = repo_root
        desired["GIT_CONFIG_COUNT"] = str(git_config_count + 1)
    if desired != os.environ and os.environ.get("NAVRL_BOOTSTRAPPED") != "1":
        desired["NAVRL_BOOTSTRAPPED"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], desired)


_bootstrap_env()

import hydra
import numpy as np
import torch
import torch.distributed as dist
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


class TerminalBalancedReplay:
    """Sample a small, persistent fraction of episode-ending transitions.

    The main FIFO replay is intentionally kept at the configured per-GPU size.
    A compact second FIFO prevents successful and collision terminal targets
    from disappearing completely when the online policy changes late in a run.
    """

    def __init__(self, main, terminal, terminal_fraction):
        self.main = main
        self.terminal = terminal
        self.terminal_fraction = float(terminal_fraction)
        if not 0.0 <= self.terminal_fraction <= 1.0:
            raise ValueError(
                f"replay_buffer.terminal_fraction must be in [0, 1], got {self.terminal_fraction}"
            )

    def __len__(self):
        return len(self.main)

    def extend(self, batch):
        self.main.extend(batch)
        done = batch["next", "done"].reshape(-1).to(torch.bool)
        if bool(done.any()):
            self.terminal.extend(batch[done])

    def sample(self, batch_size):
        batch_size = int(batch_size)
        if len(self.terminal) == 0 or self.terminal_fraction <= 0.0:
            return self.main.sample(batch_size)

        terminal_count = min(
            len(self.terminal),
            max(1, int(round(batch_size * self.terminal_fraction))),
        )
        main_count = batch_size - terminal_count
        samples = []
        if main_count:
            samples.append(self.main.sample(main_count))
        samples.append(self.terminal.sample(terminal_count))
        if len(samples) == 1:
            return samples[0]
        return torch.cat(samples, dim=0)


def avoid_optional_omnidrones_env_imports():
    """Import only isaac_env, without optional Nucleus-dependent tasks."""
    if "omni_drones.envs" in sys.modules:
        return
    import omni_drones

    package_dir = Path(omni_drones.__file__).resolve().parent / "envs"
    package = types.ModuleType("omni_drones.envs")
    package.__path__ = [str(package_dir)]
    package.__file__ = str(package_dir / "__init__.py")
    sys.modules["omni_drones.envs"] = package


def avoid_nucleus_asset_probe():
    """Prevent unused Orbit assets from blocking on an absent Nucleus server."""
    from omni.isaac.core.utils import nucleus as nucleus_utils

    nucleus_utils.get_assets_root_path = lambda: "omniverse://localhost"


def initialize_distributed(cfg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = bool(cfg.distributed.enabled) or world_size > 1
    expected_world_size = int(cfg.distributed.expected_world_size)
    if not enabled:
        return False, 0, 0, 0, 1
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Expected {expected_world_size} torchrun workers, got WORLD_SIZE={world_size}. "
            "Launch with scripts/train_sac_3gpu.sh."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed SAC requires CUDA and the NCCL backend.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 1:
        cuda_index = 0
    elif visible_gpus >= world_size and local_rank < visible_gpus:
        cuda_index = local_rank
    else:
        raise RuntimeError(
            f"Worker local_rank={local_rank} sees {visible_gpus} CUDA devices; expected "
            "one isolated device or all torchrun devices."
        )
    torch.cuda.set_device(cuda_index)
    dist.init_process_group(backend=str(cfg.distributed.backend), init_method="env://")
    return True, rank, local_rank, cuda_index, world_size


def apply_local_worker_config(cfg, world_size, rank, cuda_index):
    """Convert global config totals to the original per-GPU SAC workload."""
    global_values = {
        "num_envs": int(cfg.env.num_envs),
        "buffer_size": int(cfg.algo.buffer_size),
        "max_frame_num": int(cfg.max_frame_num),
    }
    for name, value in global_values.items():
        if value % world_size != 0:
            raise ValueError(f"Global {name}={value} must be divisible by world_size={world_size}.")

    cfg.env.num_envs = global_values["num_envs"] // world_size
    cfg.algo.buffer_size = global_values["buffer_size"] // world_size
    cfg.max_frame_num = global_values["max_frame_num"] // world_size
    cfg.device = f"cuda:{cuda_index}"
    cfg.sim.device = cfg.device
    cfg.seed = int(cfg.seed) + rank
    return global_values


def mean_metrics_across_workers(metrics, device, distributed):
    if not distributed:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values.div_(dist.get_world_size())
    return {key: values[index].item() for index, key in enumerate(keys)}


def mean_optional_metrics_across_workers(metrics, keys, device, distributed):
    if not distributed:
        return metrics
    present = float(bool(metrics))
    values = torch.tensor(
        [float(metrics.get(key, 0.0)) for key in keys] + [present],
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = values[-1].item()
    if count == 0:
        return {}
    return {key: (values[index] / count).item() for index, key in enumerate(keys)}


@hydra.main(config_path=str(ROOT / "cfg"), config_name="train_sac", version_base=None)
def main(cfg):
    if int(cfg.algo.n_step) != 1:
        raise ValueError("This simplified trainer supports algo.n_step=1 only.")

    distributed = False
    rank = 0
    local_rank = 0
    cuda_index = 0
    world_size = 1
    sim_app = None
    wandb_run = None
    try:
        distributed, rank, local_rank, cuda_index, world_size = initialize_distributed(cfg)
        global_cfg = OmegaConf.to_container(cfg, resolve=True)
        global_values = apply_local_worker_config(cfg, world_size, rank, cuda_index)
        seed = int(cfg.seed)
        configure_seed(seed)
        sim_app = SimulationApp({
            "headless": bool(cfg.headless),
            "anti_aliasing": 1,
            "multi_gpu": False,
            "active_gpu": local_rank,
            "physics_gpu": cuda_index,
        })
        avoid_optional_omnidrones_env_imports()
        avoid_nucleus_asset_probe()
        # Navigation uses the locally generated /World/ground terrain. The
        # extra visual grid plane is unused and references a Nucleus asset.
        cfg.skip_default_ground_plane = True
        from env import NavigationEnv
        from sac import SAC
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl import EpisodeStats, SyncDataCollector
        from omni_drones.utils.torchrl.transforms import VelController
        from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        from torchrl.envs.transforms import Compose, TransformedEnv
        from torchrl.envs.utils import ExplorationType

        # Isaac Sim 2023 can segfault when several processes enter the first
        # PhysX reset at exactly the same time. Build one worker's world at a
        # time; this only serializes startup, not simulation or training.
        base_env = None
        if distributed:
            for startup_rank in range(world_size):
                if rank == startup_rank:
                    print(
                        f"[SAC][rank {rank}] initializing Isaac/PhysX on "
                        f"cuda:{cuda_index} (launcher GPU {local_rank})",
                        flush=True,
                    )
                    base_env = NavigationEnv(cfg)
                dist.barrier()
        else:
            base_env = NavigationEnv(cfg)
        if base_env is None:
            raise RuntimeError(f"Rank {rank} did not initialize its environment.")
        controller = LeePositionController(9.81, base_env.drone.params).to(cfg.device)
        env = TransformedEnv(base_env, Compose(VelController(controller, yaw_control=False))).train()
        env.set_seed(seed)
        configure_seed(seed)
        policy = SAC(cfg.algo, env.observation_spec, env.action_spec, cfg.device).train()
        if distributed:
            policy.enable_distributed()

        # Initialize external logging only after every worker has built Isaac
        # and joined the first model broadcast. Rank 0 must not be able to exit
        # early while the other workers are still inside PhysX initialization.
        if rank == 0:
            wandb_kwargs = {
                "project": str(cfg.wandb.project),
                "name": str(cfg.wandb.name),
                "mode": str(cfg.wandb.mode),
                "config": global_cfg,
                # W&B 0.12 probes Git for program_relpath even when disable_git
                # is true. Supplying it prevents bind-mount ownership errors.
                "settings": wandb.Settings(
                    disable_git=True,
                    program_relpath="scripts/train_sac.py",
                ),
            }
            if cfg.wandb.entity is not None:
                wandb_kwargs["entity"] = str(cfg.wandb.entity)
            if cfg.wandb.dir is not None:
                wandb_kwargs["dir"] = str(cfg.wandb.dir)
            if cfg.wandb.run_id is not None:
                wandb_kwargs["id"] = str(cfg.wandb.run_id)
            wandb_run = wandb.init(**wandb_kwargs)
        replay_storage_device = torch.device(str(cfg.replay_buffer.storage_device))
        if replay_storage_device.type != "cpu":
            raise ValueError("This trainer keeps replay_buffer.storage_device=cpu; SAC batches are moved to GPU only for updates.")
        main_replay = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=int(cfg.algo.buffer_size), device=replay_storage_device),
            # SAC.update() always passes an explicit sample size. Leaving the
            # default unset avoids warnings when the balanced sampler requests
            # a smaller main-replay slice.
            batch_size=None,
            pin_memory=bool(cfg.replay_buffer.pin_memory),
            prefetch=int(cfg.replay_buffer.prefetch),
        )
        terminal_replay = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=int(cfg.replay_buffer.terminal_capacity),
                device=replay_storage_device,
            ),
            batch_size=None,
            pin_memory=bool(cfg.replay_buffer.pin_memory),
            prefetch=int(cfg.replay_buffer.prefetch),
        )
        replay = TerminalBalancedReplay(
            main_replay,
            terminal_replay,
            cfg.replay_buffer.terminal_fraction,
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
        episode_metric_keys = [
            "train/" + (".".join(key) if isinstance(key, tuple) else str(key))
            for key in episode_keys
        ]
        completed_episodes = EpisodeStats(episode_keys)
        episode_batch_size = int(cfg.env.num_envs)
        output_dir = Path(
            os.environ.get("NAVRL_CHECKPOINT_DIR", str(ROOT / "checkpoints"))
        ).expanduser()
        if rank == 0:
            output_dir.mkdir(exist_ok=True)
        interval = int(cfg.simple_eval.save_interval)
        parameter_sync_interval = max(1, int(cfg.distributed.parameter_sync_interval))
        min_replay = max(int(cfg.algo.warmup_steps), int(cfg.algo.batch_size))
        best_train_reach = float("-inf")
        best_train_step = 0
        best_train_min_delta = float(cfg.best_train.min_delta)
        latest_train_reach = None
        if rank == 0:
            print(
                f"[SAC] distributed={world_size} GPUs | global envs={global_values['num_envs']} "
                f"| local envs/GPU={int(cfg.env.num_envs)}"
            )
            print(
                f"[SAC] global replay={global_values['buffer_size']} | local replay/GPU="
                f"{int(cfg.algo.buffer_size)} (CPU pinned) | terminal replay="
                f"{int(cfg.replay_buffer.terminal_capacity)} | gradients=mean all-reduce"
            )
            print(f"[SAC] save every {interval} steps | no in-training evaluation")

        for step, batch in enumerate(collector, start=1):
            # The simulator emits GPU tensors. Keep the full replay buffer in
            # host memory, then SAC.update() copies only each sampled batch to GPU.
            replay_batch = batch.select(*REPLAY_KEYS).reshape(-1).detach().to(replay_storage_device)
            replay.extend(replay_batch)
            logs = mean_metrics_across_workers(batch_metrics(batch), cfg.device, distributed)
            logs["train/replay_size"] = len(replay) * world_size
            logs["train/replay_size_per_gpu"] = len(replay)
            logs["train/terminal_replay_size"] = len(terminal_replay) * world_size
            logs["train/terminal_replay_size_per_gpu"] = len(terminal_replay)
            logs["train/collector_step"] = step
            completed_episodes.add(batch.reshape(-1))
            episode_logs = {}
            if len(completed_episodes) >= episode_batch_size:
                episode_logs = completed_episode_metrics(completed_episodes.pop())
            logs.update(mean_optional_metrics_across_workers(
                episode_logs, episode_metric_keys, cfg.device, distributed
            ))
            current_train_reach = logs.get("train/stats.reach_goal")
            if current_train_reach is not None:
                latest_train_reach = float(current_train_reach)
            if len(replay) >= min_replay:
                update_logs = policy.update(
                    replay, batch_size=int(cfg.algo.batch_size), tau=float(cfg.algo.tau)
                )
                update_logs = mean_metrics_across_workers(update_logs, cfg.device, distributed)
                logs.update({f"sac/{key}": value for key, value in update_logs.items()})
                if distributed and step % parameter_sync_interval == 0:
                    logs["distributed/param_max_abs_diff"] = policy.synchronize_parameters()
            global_frames = int(collector._frames) * world_size
            logs["train/frames"] = global_frames
            if rank == 0:
                wandb.log(logs, step=global_frames)
            if step % interval != 0:
                continue
            if rank == 0:
                checkpoint = output_dir / f"checkpoint_step_{step:07d}.pt"
                torch.save(policy.state_dict(), checkpoint)
                wandb.log({"checkpoint/saved": 1, "checkpoint/step": step}, step=global_frames)
                print(f"[SAC] saved: {checkpoint} | global_frames={global_frames}", flush=True)
                if latest_train_reach is not None and latest_train_reach > best_train_reach + best_train_min_delta:
                    best_train_reach = latest_train_reach
                    best_train_step = step
                    best_checkpoint = output_dir / "checkpoint_best_train.pt"
                    torch.save(policy.state_dict(), best_checkpoint)
                    wandb.log(
                        {
                            "checkpoint/best_train_saved": 1,
                            "checkpoint/best_train_reach_goal": best_train_reach,
                            "checkpoint/best_train_step": best_train_step,
                        },
                        step=global_frames,
                    )
                    print(
                        f"[SAC] saved best train: {best_checkpoint} | "
                        f"reach_goal={best_train_reach:.4f} | step={best_train_step}",
                        flush=True,
                    )

        if rank == 0:
            checkpoint = output_dir / "checkpoint_final.pt"
            torch.save(policy.state_dict(), checkpoint)
            if latest_train_reach is not None and latest_train_reach > best_train_reach + best_train_min_delta:
                best_train_reach = latest_train_reach
                best_train_step = step
                best_checkpoint = output_dir / "checkpoint_best_train.pt"
                torch.save(policy.state_dict(), best_checkpoint)
                print(
                    f"[SAC] saved best train: {best_checkpoint} | "
                    f"reach_goal={best_train_reach:.4f} | step={best_train_step}",
                    flush=True,
                )
            elif best_train_step == 0:
                best_checkpoint = output_dir / "checkpoint_best_train.pt"
                torch.save(policy.state_dict(), best_checkpoint)
            print(f"[SAC] saved final: {checkpoint}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if sim_app is not None:
            sim_app.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
