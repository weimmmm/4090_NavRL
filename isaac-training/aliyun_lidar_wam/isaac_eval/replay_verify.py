"""Replay a route subset from a v2 snapshot and compare its initial LiDAR."""

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
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from isaac_eval.environment_file import load_environment, mesh_sha256, slice_environment
    complete = load_environment(args.environment)
    fixed = slice_environment(complete, range(args.count))

    from omni.isaac.kit import SimulationApp
    gpu = int(args.device.split(":")[-1])
    app = SimulationApp({"headless": True, "anti_aliasing": 1,
                         "active_gpu": gpu, "physics_gpu": gpu, "multi_gpu": False})
    env = None
    completed = False
    try:
        import h5py
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from isaac_eval.collect_dataset import _stage_mesh
        from isaac_eval.navigation_env import NavigationEnv
        from isaac_eval.policy import lidar_range_image

        with initialize_config_dir(config_dir=str(ROOT / "isaac_eval" / "cfg"), version_base=None):
            cfg = compose(config_name="train")
        cfg.device = args.device
        cfg.headless = True
        cfg.seed = int(fixed["terrain_seed"])
        cfg.env.num_envs = args.count
        cfg.env.max_episode_length = int(fixed["max_steps"])
        cfg.env.num_obstacles = int(fixed["static_obstacles"])
        cfg.env.randomize_routes = False
        cfg.env_dyn.num_obstacles = 0
        cfg.enable_eval = False
        cfg.record_eval_video = False
        env = NavigationEnv(cfg, eval_environment=fixed)
        env.eval()
        td = env.reset()
        replay = lidar_range_image(env).detach().cpu().numpy()
        with h5py.File(args.dataset, "r") as data:
            frames = data["frames"]
            scene = frames["scene_id"][:]
            frame = frames["frame_index"][:]
            rows = [int(np.flatnonzero((scene == index) & (frame == 0))[0])
                    for index in range(args.count)]
            reference = frames["range_values"][rows]
        error = np.abs(replay - reference)
        vertices, faces = _stage_mesh()
        stage_hash = mesh_sha256(vertices, faces)
        expected_hash = fixed["terrain_mesh"]["sha256"]
        result = {
            "environment": str(args.environment), "dataset": str(args.dataset),
            "routes_replayed": args.count, "max_abs_lidar_error": float(error.max()),
            "mean_abs_lidar_error": float(error.mean()), "tolerance": args.tolerance,
            "mesh_sha256": stage_hash, "mesh_hash_matches": stage_hash == expected_hash,
            "start_positions_match": bool(torch.equal(
                td["info", "drone_state"][:, :, :3].cpu(), fixed["start_positions"])),
        }
        if result["max_abs_lidar_error"] > args.tolerance:
            raise RuntimeError(f"initial LiDAR error exceeds tolerance: {result}")
        if not result["mesh_hash_matches"] or not result["start_positions_match"]:
            raise RuntimeError(f"snapshot replay mismatch: {result}")
        print(json.dumps(result, indent=2))
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
