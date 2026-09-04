import datetime
import json
import os
import random as rn
import socket
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMNI_KIT_RENDERER", "Vulkan")
os.environ.setdefault("OMNI_KIT_NO_OPENGL_RENDERING", "1")

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
TRAINING_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_SCHEMA_VERSION = 1


def configure_seed(seed: int):
    rn.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_sim_launch_config(cfg):
    sim_launch = {
        "headless": cfg.headless,
        "anti_aliasing": 1,
        "multi_gpu": False,
    }
    device_str = str(getattr(cfg, "device", "cuda:0"))
    sim_device_str = str(getattr(getattr(cfg, "sim", {}), "device", device_str))
    if device_str != sim_device_str:
        raise ValueError(
            "[build_eval_world]: cfg.device and cfg.sim.device must match. "
            f"Got device={device_str}, sim.device={sim_device_str}."
        )
    if not device_str.startswith("cuda:"):
        return sim_launch

    try:
        cuda_idx = int(device_str.split(":", 1)[1])
    except ValueError:
        cuda_idx = 0

    visible_devices = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if visible_devices and cuda_idx >= len(visible_devices):
        raise ValueError(
            "[build_eval_world]: cfg.device points to a hidden CUDA ordinal. "
            f"cfg.device={device_str}, CUDA_VISIBLE_DEVICES={','.join(visible_devices)}."
        )

    active_gpu = cuda_idx
    if visible_devices:
        try:
            active_gpu = int(visible_devices[cuda_idx])
        except ValueError:
            active_gpu = cuda_idx

    sim_launch["active_gpu"] = active_gpu
    sim_launch["physics_gpu"] = cuda_idx
    print(
        "[build_eval_world]: SimulationApp GPU mapping | "
        f"active_gpu={active_gpu}, physics_gpu={cuda_idx}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    return sim_launch


def resolve_world_path(cfg) -> Path:
    raw_path = cfg.eval_world.get("path", "eval_worlds/eval_3407.pt")
    raw_path = str(raw_path)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = TRAINING_ROOT / path
    return path.resolve()


def apply_eval_world_config(cfg):
    """Apply the frozen-world specification without changing training defaults."""
    world_cfg = cfg.eval_world
    cfg.env.num_envs = int(world_cfg.num_envs)
    cfg.env.num_obstacles = int(world_cfg.num_obstacles_static)
    cfg.env_dyn.num_obstacles = int(world_cfg.num_obstacles_dynamic)


def record_dynamic_obstacle_trajectory(env, horizon: int):
    num_obstacles = int(env.cfg.env_dyn.num_obstacles)
    traj_state = torch.zeros((horizon, num_obstacles, 13), dtype=torch.float32)
    traj_vel = torch.zeros((horizon, num_obstacles, 3), dtype=torch.float32)

    if num_obstacles == 0:
        return traj_state, traj_vel

    for step in range(horizon):
        env.move_dynamic_obstacle()
        traj_state[step] = env.dyn_obs_state.detach().cpu()
        traj_vel[step] = env.dyn_obs_vel.detach().cpu()
        if (step + 1) % 200 == 0 or step + 1 == horizon:
            print(f"[build_eval_world]: recorded dynamic trajectory {step + 1}/{horizon}")

    return traj_state, traj_vel


@hydra.main(config_path=FILE_PATH, config_name="train_sac", version_base=None)
def main(cfg):
    apply_eval_world_config(cfg)
    world_seed = int(cfg.eval_world.get("seed", cfg.seed))
    world_name = str(cfg.eval_world.get("name", "eval_3407"))
    out_path = resolve_world_path(cfg)
    overwrite = bool(cfg.eval_world.get("overwrite", False))

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"[build_eval_world]: {out_path} already exists. "
            "Set eval.overwrite_world=True to replace it."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configure_seed(world_seed)
    sim_app = SimulationApp(build_sim_launch_config(cfg))
    configure_seed(world_seed)

    try:
        from env import NavigationEnv

        print(f"[build_eval_world]: building world={world_name}, seed={world_seed}")
        print(f"[build_eval_world]: output={out_path}")

        env = NavigationEnv(cfg).train()
        env.set_seed(world_seed)
        env.dyn_obs_seed = world_seed
        env.dyn_obs_initialized = False
        env.dyn_obs_step_count = 0
        env.reset()

        drone_pos_init = env.drone.pos.detach().clone().cpu()
        drone_rot_init = env.drone.rot.detach().clone().cpu()
        target_pos_init = env.target_pos.detach().clone().cpu()
        target_dir_init = env.target_dir.detach().clone().cpu()
        height_range_init = env.height_range.detach().clone().cpu()

        dyn_obs_origin = torch.zeros((0, 3), dtype=torch.float32)
        dyn_obs_state_init = torch.zeros((0, 13), dtype=torch.float32)
        dyn_obs_goal_init = torch.zeros((0, 3), dtype=torch.float32)
        dyn_obs_vel_init = torch.zeros((0, 3), dtype=torch.float32)
        if int(cfg.env_dyn.num_obstacles) > 0:
            dyn_obs_origin = env.dyn_obs_origin.detach().clone().cpu()
            dyn_obs_state_init = env.dyn_obs_state.detach().clone().cpu()
            dyn_obs_goal_init = env.dyn_obs_goal.detach().clone().cpu()
            dyn_obs_vel_init = env.dyn_obs_vel.detach().clone().cpu()

        horizon = int(env.max_episode_length)
        dyn_obs_traj_state, dyn_obs_traj_vel = record_dynamic_obstacle_trajectory(env, horizon)

        snapshot = {
            "drone_pos_init": drone_pos_init,
            "drone_rot_init": drone_rot_init,
            "target_pos_init": target_pos_init,
            "target_dir_init": target_dir_init,
            "height_range_init": height_range_init,
            "dyn_obs_origin": dyn_obs_origin,
            "dyn_obs_state_init": dyn_obs_state_init,
            "dyn_obs_goal_init": dyn_obs_goal_init,
            "dyn_obs_vel_init": dyn_obs_vel_init,
            "dyn_obs_traj_state": dyn_obs_traj_state,
            "dyn_obs_traj_vel": dyn_obs_traj_vel,
            "meta": {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "world_name": world_name,
                "world_seed": world_seed,
                "num_envs": int(env.num_envs),
                "max_episode_length": int(env.max_episode_length),
                "num_obstacles_static": int(cfg.env.num_obstacles),
                "num_obstacles_dynamic": int(cfg.env_dyn.num_obstacles),
                "trajectory_length": int(horizon),
                "device": str(cfg.device),
                "built_at_iso": datetime.datetime.now().isoformat(timespec="seconds"),
                "hostname": socket.gethostname(),
                "user": os.environ.get("USER") or os.environ.get("LOGNAME") or "",
                "env_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "torch_version": str(getattr(torch, "__version__", "")),
                "torch_cuda_version": str(getattr(torch.version, "cuda", "")),
                "cli_overrides": list(sys.argv[1:]),
                "resolved_config": OmegaConf.to_container(cfg, resolve=True),
            },
        }

        torch.save(snapshot, out_path)
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps({**snapshot["meta"], "snapshot_path": str(out_path)}, indent=2),
            encoding="utf-8",
        )
        size_mb = out_path.stat().st_size / 1024 / 1024
        print("[build_eval_world]: SUCCESS")
        print(f"[build_eval_world]: world={out_path}")
        print(f"[build_eval_world]: meta={meta_path}")
        print(f"[build_eval_world]: size={size_mb:.2f} MB")
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()