"""Short PPO+LiDAR+physics capacity test; it does not create dataset scenes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaac_eval.bootstrap import activate_vendored_sources

activate_vendored_sources()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "isaac_eval/checkpoints/ppo_dataset_collector.pt")
    args = parser.parse_args()
    physical_gpu = int(args.device.split(":")[-1])
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(physical_gpu))
    from omni.isaac.kit import SimulationApp
    app = SimulationApp({"headless": True, "anti_aliasing": 1,
                         "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
    env = None
    completed = False
    try:
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from torchrl.envs.transforms import Compose, TransformedEnv
        from torchrl.envs.utils import ExplorationType, set_exploration_type
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        from isaac_eval.navigation_env import NavigationEnv
        from isaac_eval.ppo_policy import load_ppo_expert

        with initialize_config_dir(config_dir=str(ROOT / "isaac_eval/cfg"), version_base=None):
            cfg = compose(config_name="train")
        cfg.device = "cuda:0"
        cfg.headless = True
        cfg.seed = args.seed
        cfg.env.num_envs = args.num_envs
        cfg.env.max_episode_length = 2200
        cfg.env.num_obstacles = 350
        cfg.env.randomize_routes = True
        cfg.env_dyn.num_obstacles = 0
        cfg.enable_eval = False
        cfg.record_eval_video = False
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        env = NavigationEnv(cfg)
        env.eval()
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        transformed = TransformedEnv(
            env, Compose(VelController(controller, yaw_control=False))).eval()
        transformed.set_seed(args.seed)
        td = transformed.reset()
        policy = load_ppo_expert(
            args.checkpoint, cfg.algo, transformed.observation_spec,
            transformed.action_spec, cfg.device, cfg.sensor.lidar_range)
        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            for _ in range(args.steps):
                policy(td)
                td = transformed.step(td)["next"]
        torch.cuda.synchronize()
        print(json.dumps({
            "passed": True, "num_envs": args.num_envs, "steps": args.steps,
            "static_obstacles": 350, "raw_lidar_resolution": list(env.lidar_raw_resolution),
            "peak_torch_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_torch_cuda_bytes": int(torch.cuda.max_memory_reserved()),
        }, indent=2), flush=True)
        completed = True
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        if completed:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
