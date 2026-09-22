"""Check actual static-obstacle hits and collision detection in Isaac Sim."""

import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", "..", "third_party", "OmniDrones")))

import hydra
import numpy as np
import torch
from omni.isaac.kit import SimulationApp


@hydra.main(config_path="../cfg", config_name="train", version_base=None)
def main(cfg):
    cfg.env.num_envs = 1
    cfg.env_dyn.num_obstacles = 0
    cfg.headless = True
    if cfg.env.num_obstacles <= 0:
        raise ValueError("The hit check requires static obstacles")
    gpu_id = int(str(cfg.device).split(":")[-1])
    app = SimulationApp({"headless": True, "active_gpu": gpu_id,
                         "physics_gpu": gpu_id, "multi_gpu": False})
    env = None
    try:
        from env import NavigationEnv
        from omni.isaac.orbit import sim as sim_utils
        from omni_drones.utils.torch import euler_to_quaternion
        from pxr import UsdGeom

        env = NavigationEnv(cfg)
        env.reset()
        if env.lidar.cfg.mesh_prim_paths != ["/World/ground"]:
            raise RuntimeError("LiDAR is not scanning the generated terrain")
        prim = sim_utils.get_first_matching_child_prim(
            "/World/ground", lambda child: child.GetTypeName() == "Mesh"
        )
        if prim is None:
            raise RuntimeError("Generated terrain mesh not found")
        mesh = UsdGeom.Mesh(prim)
        vertices = torch.as_tensor(np.array(mesh.GetPointsAttr().Get(), copy=True),
                                   dtype=torch.float32, device=env.device)
        faces = torch.as_tensor(np.array(mesh.GetFaceVertexIndicesAttr().Get(), copy=True),
                                dtype=torch.long, device=env.device).reshape(-1, 3)
        triangles = vertices[faces]
        centers = triangles.mean(dim=1)
        normals = torch.cross(triangles[:, 1] - triangles[:, 0],
                              triangles[:, 2] - triangles[:, 0], dim=-1)
        lengths = normals.norm(dim=-1)
        normals = normals / lengths.clamp_min(1e-6).unsqueeze(-1)
        # Probe vertical obstacle faces above the floor, within the flight-height range.
        candidates = ((lengths > 1e-6) & (normals[:, 2].abs() < 0.1)
                      & (centers[:, 2] > 0.5) & (centers[:, 2] < 3.0)).nonzero().flatten()
        if candidates.numel() == 0:
            raise RuntimeError("No obstacle side faces found in the terrain")
        expected_shape = (1, 1, env.lidar_vbeams, env.lidar_hbeams)

        def probe(center, normal, offset):
            position = (center + offset * normal).reshape(1, 1, 3)
            rpy = torch.zeros(1, 1, 3, device=env.device)
            rpy[..., 2] = torch.atan2(-normal[1], -normal[0])
            env.drone.set_world_poses(position, euler_to_quaternion(rpy))
            env.lidar.update(env.dt, force_recompute=True)
            hits = env.lidar.data.ray_hits_w
            distances = (hits - env.lidar.data.pos_w.unsqueeze(1)).norm(dim=-1)
            obstacle_hits = torch.isfinite(distances) & (hits[..., 2] > 0.2) & (distances < 1.0)
            observation = env._compute_state_and_obs()
            image = observation["agents", "observation", "lidar"]
            if image.shape != expected_shape or not torch.isfinite(image).all():
                raise RuntimeError("Invalid range-image observation")
            return obstacle_hits.sum().item(), image.min().item(), env.stats["collision"].item()

        for index in candidates[:64].tolist():
            center, normal = centers[index], normals[index]
            hit_count, far_distance, far_collision = probe(center, normal, 0.5)
            if hit_count == 0 or far_distance >= 1.0 or far_collision:
                continue
            near_count, near_distance, near_collision = probe(center, normal, 0.15)
            if near_count > 0 and near_distance < 0.3 and near_collision:
                print("PASS: scanned mesh:", prim.GetPath())
                print("PASS: obstacle hits:", hit_count,
                      "range-image shape:", expected_shape)
                print("PASS: nearest distances: %.3f m -> %.3f m; collision: 0 -> 1"
                      % (far_distance, near_distance))
                return
        raise RuntimeError("Could not verify obstacle hits and the 0.3 m collision threshold")
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.sim.stop()
        app.close()


if __name__ == "__main__":
    main()
