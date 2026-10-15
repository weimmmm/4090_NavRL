"""Render Isaac-Sim RGB with GT and strict autoregressive Future-DiT LiDAR.

The obstacle appearance comes from Isaac Sim, not from interpolation of the
sparse LiDAR.  The model receives three past VAE latents once; its t+1:t+3
outputs are autoregressive.  Future GT is used only for overlays and metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_eval.bootstrap import activate_vendored_sources

activate_vendored_sources()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--future-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--route", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument("--frame-interval", type=int, default=10)
    parser.add_argument("--min-front-hits", type=int, default=24)
    parser.add_argument("--max-search-steps", type=int, default=600)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    for name in ("environment", "ppo_checkpoint", "future_checkpoint",
                 "latent_metadata"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_file():
            parser.error(f"missing {name}: {value}")
        setattr(args, name, value)
    args.output = args.output.expanduser().resolve()
    return args


def load_future(checkpoint, device):
    import torch
    from lidar_wam.models.lidar_video_dit import LiDARVideoDiT

    payload = torch.load(checkpoint, map_location="cpu")
    cfg = payload["architecture"]
    model = LiDARVideoDiT(
        width=int(cfg["width"]), depth=int(cfg["depth"]),
        heads=int(cfg["heads"]), mlp_ratio=float(cfg["mlp_ratio"]),
    ).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def configure_camera(width, height):
    import omni.replicator.core as rep
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom

    camera_path = "/World/ForwardCamera"
    if not prim_utils.is_prim_path_valid(camera_path):
        prim_utils.create_prim(camera_path, prim_type="Camera")
    camera = UsdGeom.Camera(get_current_stage().GetPrimAtPath(camera_path))
    horizontal_aperture = 20.955
    focal_length = 18.0
    camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
    camera.GetVerticalApertureAttr().Set(
        horizontal_aperture * float(height) / float(width))
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
    render_product = rep.create.render_product(camera_path, (width, height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach([render_product])
    horizontal_fov = math.degrees(2.0 * math.atan(
        horizontal_aperture / (2.0 * focal_length)))
    return camera_path, annotator, render_product, horizontal_fov


def capture_rgb(env, camera_path, annotator):
    import numpy as np
    from omni.isaac.core.utils.viewports import set_camera_view

    position = env.drone.pos[0, 0].detach().cpu().numpy().astype(np.float64)
    heading = env.drone.heading[0, 0].detach().cpu().numpy().astype(np.float64)
    heading[2] = 0.0
    heading /= max(float(np.linalg.norm(heading)), 1e-6)
    eye = position + 0.20 * heading + np.array([0.0, 0.0, 0.12])
    target = eye + 10.0 * heading
    set_camera_view(eye=eye, target=target, camera_prim_path=camera_path)
    # Two renders make both the camera transform and RTX result visible to the
    # Replicator annotator in the same simulation step.
    env.sim.render()
    env.sim.render()
    image = annotator.get_data()
    image = np.frombuffer(image, dtype=np.uint8).reshape(*image.shape)
    return np.asarray(image, dtype=np.uint8).copy()


def lidar_pixels(frame, threshold, width, height, horizontal_fov):
    import numpy as np

    azimuth = np.arange(108, dtype=np.float32) * (360.0 / 108.0)
    azimuth = (azimuth + 180.0) % 360.0 - 180.0
    elevation = np.linspace(-7.0, 52.0, 18, dtype=np.float32)
    azimuth, elevation = np.meshgrid(azimuth, elevation, indexing="ij")
    valid = frame[1, :, :18] > float(threshold)
    valid &= np.abs(azimuth) <= horizontal_fov / 2.0
    distance = np.clip((frame[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
    horizontal = np.deg2rad(azimuth[valid])
    vertical = np.deg2rad(elevation[valid])
    focal = (width / 2.0) / math.tan(math.radians(horizontal_fov / 2.0))
    # Camera looks along +X, with +Y appearing on image left and +Z upward.
    u = width / 2.0 - focal * np.tan(horizontal)
    v = height / 2.0 - focal * np.tan(vertical) / np.cos(horizontal)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside], v[inside], distance[valid][inside]


def render_report(rgb_frames, truth, prediction, records, horizontal_fov,
                  destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(3, 3, figsize=(17, 10), dpi=180)
    scatter = None
    for horizon in range(3):
        rgb = rgb_frames[horizon]
        axes[0, horizon].imshow(rgb)
        axes[0, horizon].set_title(f"t+{horizon+1} · Isaac Sim RGB", fontsize=12)
        for row, (frame, threshold, label) in enumerate((
                (truth[horizon], 0.0, "GT LiDAR"),
                (prediction[horizon], 1.5, "Future-DiT prediction")), start=1):
            axes[row, horizon].imshow(rgb)
            u, v, distance = lidar_pixels(
                frame, threshold, rgb.shape[1], rgb.shape[0], horizontal_fov)
            scatter = axes[row, horizon].scatter(
                u, v, c=distance, cmap="turbo", vmin=0, vmax=10,
                s=26, marker="o", edgecolors="black", linewidths=.25,
                alpha=.9)
            suffix = ("" if row == 1 else
                      f" · CD={records[horizon]['cd_paper_m2']:.3f} m²")
            axes[row, horizon].set_title(
                f"t+{horizon+1} · {label}{suffix}", fontsize=11)
        for row in range(3):
            axes[row, horizon].set_xlim(0, rgb.shape[1])
            axes[row, horizon].set_ylim(rgb.shape[0], 0)
            axes[row, horizon].axis("off")
    color_axis = figure.add_axes((.925, .16, .014, .68))
    figure.colorbar(scatter, cax=color_axis, label="LiDAR range (m)")
    figure.suptitle(
        "Isaac Sim forward RGB with strict autoregressive Future-DiT LiDAR\n"
        "row 1: scene · row 2: GT returns · row 3: predicted returns",
        y=.99, fontsize=15)
    figure.subplots_adjust(left=.02, right=.91, bottom=.03, top=.91,
                           hspace=.18, wspace=.04)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    from isaac_eval.environment_file import load_environment, slice_environment

    fixed = load_environment(args.environment)
    if args.route >= int(fixed["num_envs"]):
        raise ValueError("route is outside the saved environment")
    fixed = dict(slice_environment(fixed, [args.route]))
    fixed["max_steps"] = max(int(fixed["max_steps"]), args.max_search_steps + 50)
    terrain_seed = int(fixed["terrain_seed"])

    from omni.isaac.kit import SimulationApp

    gpu_id = int(str(args.device).split(":")[-1])
    app = SimulationApp({
        "headless": True, "anti_aliasing": 2,
        "active_gpu": gpu_id, "physics_gpu": gpu_id, "multi_gpu": False,
        "width": args.width, "height": args.height,
    })
    env = None
    try:
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from torchrl.envs.transforms import Compose, TransformedEnv
        from torchrl.envs.utils import ExplorationType, set_exploration_type
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController

        WAM_ROOT = ROOT / "lidar_WAM"
        if str(WAM_ROOT) not in sys.path:
            sys.path.insert(0, str(WAM_ROOT))
        from isaac_eval.navigation_env import NavigationEnv
        from isaac_eval.policy import lidar_range_image, load_circular_vae
        from isaac_eval.ppo_policy import load_ppo_expert
        from lidar_wam.runner.lidar_video_dit import sample_next
        from lidar_wam.runner.world_direct_horizon import frame_metrics

        with initialize_config_dir(
                config_dir=str(ROOT / "isaac_eval" / "cfg"),
                version_base=None):
            cfg = compose(config_name="train")
        cfg.device = args.device
        cfg.headless = True
        cfg.seed = terrain_seed
        cfg.env.num_envs = 1
        cfg.env.max_episode_length = int(fixed["max_steps"])
        cfg.env.num_obstacles = int(fixed["static_obstacles"])
        cfg.env.randomize_routes = False
        cfg.env_dyn.num_obstacles = 0
        cfg.enable_eval = False
        cfg.record_eval_video = False
        # RGB is captured with an independent offscreen Replicator camera.
        # Do not enable the UI viewport: it requires GLFW/X11 on this server.

        np.random.seed(terrain_seed)
        torch.manual_seed(terrain_seed)
        torch.cuda.manual_seed_all(terrain_seed)
        env = NavigationEnv(cfg, eval_environment=fixed)
        env.eval()
        env.enable_render(False)
        camera_path, rgb_annotator, render_product, horizontal_fov = (
            configure_camera(args.width, args.height))
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        transformed = TransformedEnv(
            env, Compose(VelController(controller, yaw_control=False))).eval()
        transformed.set_seed(terrain_seed)
        td = transformed.reset()
        policy = load_ppo_expert(
            args.ppo_checkpoint, cfg.algo, transformed.observation_spec,
            transformed.action_spec, cfg.device, cfg.sensor.lidar_range)
        future, payload = load_future(args.future_checkpoint, args.device)
        vae = load_circular_vae(torch.device(args.device))
        scale = float(json.loads(args.latent_metadata.read_text())[
            "scaling_factor"])

        history = []
        predicted = None
        prediction_step = None
        truth_frames, rgb_frames = [], []
        generator = torch.Generator(device=args.device).manual_seed(271828)
        front = np.abs((np.arange(108) * (360.0 / 108.0) + 180) % 360 - 180) <= 45

        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            initial = lidar_range_image(env)
            latent = vae.encode(initial).latent_dist.mode() * scale
            history.append(latent)
            for step in range(1, args.max_search_steps + 31):
                policy(td)
                td = transformed.step(td)["next"]
                if step % args.frame_interval != 0:
                    continue
                frame = lidar_range_image(env)
                latent = vae.encode(frame).latent_dist.mode() * scale
                if predicted is None:
                    history.append(latent)
                    history = history[-3:]
                    front_hits = int((frame[0, 1, front, :18] > 0).sum())
                    if (len(history) == 3 and front_hits >= args.min_front_hits):
                        rollout = torch.stack(history, dim=1)
                        predictions = []
                        for _ in range(3):
                            value = sample_next(
                                future, rollout, args.flow_steps,
                                generator=generator)
                            predictions.append(value)
                            rollout = torch.cat(
                                (rollout[:, 1:], value[:, None]), dim=1)
                        predicted_latent = torch.stack(predictions, dim=1)
                        decoded = vae.decode(
                            predicted_latent.flatten(0, 1).float() / scale
                        ).sample.float().cpu().numpy()
                        predicted = decoded.reshape(3, 2, 108, 20)
                        prediction_step = step
                        print(json.dumps({
                            "event": "prediction_created", "step": step,
                            "front_hits": front_hits}), flush=True)
                else:
                    horizon = (step - prediction_step) // args.frame_interval
                    if 1 <= horizon <= 3:
                        truth_frames.append(frame[0].float().cpu().numpy())
                        rgb_frames.append(capture_rgb(
                            env, camera_path, rgb_annotator))
                        print(json.dumps({
                            "event": "future_captured", "horizon": horizon,
                            "step": step}), flush=True)
                    if horizon >= 3:
                        break
        if predicted is None or len(truth_frames) != 3:
            raise RuntimeError("could not capture a complete future sequence")

        metrics = []
        for horizon in range(3):
            row = frame_metrics(predicted[horizon], truth_frames[horizon])
            metrics.append({
                "horizon": horizon + 1,
                "cd_paper_m2": float(row["cd_paper_m2"]),
                "valid_range_mae_m": (
                    float(row["valid_range_abs_error_sum_m"])
                    / max(float(row["valid_range_count"]), 1.0)),
            })
        args.output.mkdir(parents=True, exist_ok=True)
        figure = args.output / "isaac_rgb_gt_prediction_ar3.png"
        render_report(rgb_frames, truth_frames, predicted, metrics,
                      horizontal_fov, figure)
        for horizon, image in enumerate(rgb_frames, start=1):
            from PIL import Image
            Image.fromarray(image).save(
                args.output / f"isaac_rgb_tplus{horizon}.png")
        report = {
            "environment": str(args.environment), "route": args.route,
            "future_checkpoint": str(args.future_checkpoint),
            "future_checkpoint_step": int(payload.get("step", -1)),
            "prediction_sim_step": prediction_step,
            "frame_interval": args.frame_interval,
            "flow_steps": args.flow_steps,
            "horizontal_camera_fov_deg": horizontal_fov,
            "protocol": (
                "strict autoregressive prediction from three past GT LiDAR "
                "frames; future GT is overlay/metric only"),
            "metrics": metrics, "figure": str(figure),
        }
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
