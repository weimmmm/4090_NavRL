"""Annotate REAL student LiDAR observations with synchronous ego commands.

Red arrows are a navigation HUD: yaw-only ego XY, +X up / +Y left.
They are NOT optical flow or a projection of vertical velocity. The reported
3-D norm includes Z. No synthetic scene, PPO action, or interpolated geometry
is substituted for the recorded LiDAR or student's executed command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def select_examples(data, lidar_range):
    steps = np.asarray(data["lidar_step"], dtype=np.int64)
    images = data["lidar_image"]
    if images.shape[0] != len(steps) or images.shape[2:] != (2, 108, 20):
        raise ValueError("Require recorded LiDAR [frames,drones,2,108,20]")
    if np.any(steps % 10) or steps.max() >= len(data["ego_velocity_command"]):
        raise ValueError("LiDAR and command clocks do not align to K=10")
    velocity = np.asarray(data["ego_velocity_command"][steps], np.float32)
    position = np.asarray(data["root_state_wxyz"][steps, :, :3], np.float32)
    distance = np.linalg.norm(position-data["target_position"][None], axis=-1)
    norm = np.linalg.norm(velocity, axis=-1)
    azimuth = (np.arange(108)*360./108.+180.) % 360.-180.
    front = np.flatnonzero(np.abs(azimuth) <= 60.+1e-5)
    # Camera left is +Y / positive azimuth. Preserve sensor samples exactly.
    front = front[np.argsort(azimuth[front])[::-1]]
    frontal = images[:, :, :, front, :18]
    valid = frontal[:, :, 1] > 0
    ranges = (frontal[:, :, 0].astype(np.float32)+1.)*(lidar_range/2.)
    hit_fraction = valid.mean(axis=(-1, -2))
    nearest = np.where(valid, ranges, lidar_range).min(axis=(-1, -2))
    alive = steps[:, None]+100 < data["terminal_step"][None]
    eligible = ((steps[:, None] >= 100) & alive & (distance > 3.)
                & (data["outcome"][None] == "success")
                & (norm > .8) & (norm <= 2.001)
                & (hit_fraction > .12) & (nearest > .45))
    x, y = velocity[..., 0], velocity[..., 1]
    definitions = (
        ("left", "Left-forward", (x > .6) & (y > .45)),
        ("forward", "Forward", (x > 1.) & (np.abs(y) < .10)),
        ("right", "Right-forward", (x > .6) & (y < -.45)),
    )
    samples, used_routes = [], set()
    for slug, title, condition in definitions:
        candidates = eligible & condition
        if used_routes:
            candidates[:, list(used_routes)] = False
        if not candidates.any():
            raise RuntimeError(f"No genuine recorded example for {title}")
        score = hit_fraction*.8 - .12*np.abs(nearest-1.7)
        score += (.5*np.minimum(np.abs(y), 1.2) if slug != "forward"
                  else .3*np.minimum(x, 1.8)-np.abs(y))
        index, route = np.unravel_index(np.where(candidates, score, -np.inf).argmax(), candidates.shape)
        used_routes.add(int(route))
        step = int(steps[index])
        image = ranges[index, route].copy()
        image[~valid[index, route]] = lidar_range
        command = velocity[index, route].astype(float)
        samples.append({"slug": slug, "direction": title, "route": int(route),
                        "source_route_index": int(data["source_route_index"][route]),
                        "physics_step": step, "time_s": step*float(data["physics_dt"]),
                        "command_ego_xyz_mps": command.tolist(),
                        "command_speed_3d_mps": float(np.linalg.norm(command)),
                        "goal_distance_m": float(distance[index, route]),
                        "front_hit_fraction": float(hit_fraction[index, route]),
                        "frontal_range_m": image.T[::-1],
                        "observation_index": int(index)})
    return samples


def render_one(ax, sample, lidar_range, *, origin=(.51, .24), gain=.30):
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyArrowPatch

    image = sample["frontal_range_m"]
    ax.imshow(image, cmap="gray", vmin=0, vmax=lidar_range,
              interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    vx, vy, vz = sample["command_ego_xyz_mps"]
    start = origin
    # Fractions are not isotropic on a rectangular image. Correct the X scale
    # so that one m/s has the SAME pixel length on either HUD axis.
    bounds = ax.get_window_extent()
    end = (start[0]-gain*vy*bounds.height/bounds.width, start[1]+gain*vx)
    arrow = FancyArrowPatch(start, end, transform=ax.transAxes,
                            arrowstyle="-|>", mutation_scale=30,
                            linewidth=3.5, color="#ed3b32", zorder=5)
    arrow.set_path_effects([pe.Stroke(linewidth=6, foreground="white"), pe.Normal()])
    ax.add_patch(arrow)
    ax.plot(*start, "o", transform=ax.transAxes, color="#ed3b32", ms=6,
            markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    ax.text(.035, .94, "5 m LiDAR | front 120 deg", transform=ax.transAxes,
            fontsize=14, va="top", color="#202020",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 4})
    ax.text(.035, .065, "Arrow: ego XY command (HUD)", transform=ax.transAxes,
            fontsize=14, va="bottom", color="#202020",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 4})


def create_images(samples, output, lidar_range):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    for sample in samples:
        fig = plt.figure(figsize=(5.6, 4.5), dpi=160, facecolor="white")
        ax = fig.add_axes((.025, .25, .95, .57))
        render_one(ax, sample, lidar_range)
        fig.text(.5, .985, sample["direction"], ha="center", va="top", fontsize=19)
        fig.text(.5, .860, f"Route {sample['route']} | t = {sample['time_s']:.2f} s",
                 ha="center", fontsize=15)
        fig.text(.5, .175, f"Total speed = {sample['command_speed_3d_mps']:.3f} m/s",
                 ha="center", fontsize=21)
        vx, vy, vz = sample["command_ego_xyz_mps"]
        fig.text(.5, .100, f"vx {vx:+.3f}   vy {vy:+.3f}   vz {vz:+.3f} m/s",
                 ha="center", fontsize=16)
        fig.text(.5, .035, "x: forward | y: left | z: up", ha="center", fontsize=14, color="#404040")
        fig.savefig(output/f"velocity_{sample['slug']}.png", dpi=160)
        plt.close(fig)
    from PIL import Image
    panels = [Image.open(output/f"velocity_{s['slug']}.png").convert("RGB") for s in samples]
    combined = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
    offset = 0
    for panel in panels:
        combined.paste(panel, (offset, 0))
        offset += panel.width
    combined.save(output/"velocity_triptych.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lidar-range", type=float, default=5.)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    if metrics["action_frame"] not in {
            "yaw_only_ego_flu", "current_observation_yaw_only_ego_flu"}:
        raise ValueError("Do not display another action frame as ego FLU")
    args.output.mkdir(parents=True, exist_ok=True)
    # Only trusted, locally generated evaluator archives contain outcome objects.
    data = np.load(args.trajectory, allow_pickle=True)
    samples = select_examples(data, args.lidar_range)
    create_images(samples, args.output, args.lidar_range)
    summary = {"checkpoint": metrics["checkpoint"], "checkpoint_step": metrics["checkpoint_step"],
               "source_trajectory": str(args.trajectory), "action_frame": metrics["action_frame"],
               "arrow_semantics": "ego XY navigation HUD: forward up, left left; NOT optical flow",
               "speed_semantics": "3-D norm of student's commanded velocity, not measured motion",
               "depth_semantics": "actual recorded radial LiDAR range; 0 m black, 5 m/no hit white",
               "samples": [{k: v for k, v in sample.items() if k != "frontal_range_m"} for sample in samples]}
    (args.output/"samples.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
