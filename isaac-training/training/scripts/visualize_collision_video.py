"""Replay saved collision routes; never rerun inference or the simulator.

``prepare`` needs numpy/torch in the training container. ``render`` only
needs Pillow and ffmpeg, and can run locally with a Chinese-capable font.
The final red cross is the drone center at the collision decision, not a
measured rigid-body contact point. LiDAR is sample-and-held, never invented.
"""
from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import subprocess


def prepare(args):
    import numpy as np
    import torch
    from PIL import Image

    metrics = json.loads(args.metrics.read_text())
    data = np.load(args.trajectory, allow_pickle=True)
    route = args.route
    terminal = int(data["terminal_step"][route])
    if data["outcome"][route] != "collision":
        raise ValueError("Select a collision route")
    scene_path = Path(metrics["scene_path"])
    scene = torch.load(scene_path, map_location="cpu", weights_only=False)
    if scene["map_sha256"] != metrics["map_sha256"]:
        raise ValueError("Scene differs from the evaluated map")
    centers = scene["obstacle_center_xyz"].numpy()
    sizes = scene["obstacle_size_xyz"].numpy()
    states = data["root_state_wxyz"][:terminal+1, route]
    position = states[-1, :3]
    gaps = np.maximum(np.abs(position-centers)-sizes/2, 0)
    distances = np.linalg.norm(gaps, axis=1)
    closest_box = int(distances.argmin())
    az = (np.arange(108)*360./108.+180.) % 360.-180.
    front_order = np.flatnonzero(np.abs(az) <= 60.+1e-5)
    front_order = front_order[np.argsort(az[front_order])[::-1]]
    full_order = np.argsort(az)[::-1]
    inverse = np.argsort(full_order)

    def encode(image):
        picture = Image.fromarray(np.clip(np.rint(image/5*255), 0, 255).astype(np.uint8))
        buffer = BytesIO()
        picture.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    frames = []
    # NPZ entries are decompressed on each lookup: load the archive array once.
    lidar_images = data["lidar_image"]
    lidar_steps = data["lidar_step"]
    for index in np.flatnonzero(lidar_steps < terminal).tolist():
        step = int(lidar_steps[index])
        raw = lidar_images[index, route]
        ranges = (raw[0, :, :18].astype(np.float32)+1)*2.5
        valid = raw[1, :, :18] > 0
        masked = np.where(valid, ranges, np.inf)
        ray, elevation = np.unravel_index(masked.argmin(), masked.shape)
        nearest = float(masked[ray, elevation])
        ranges[~valid] = 5
        frames.append({"step": step,
                       "front_png": encode(ranges[front_order].T[::-1]),
                       "pano_png": encode(ranges[full_order].T[::-1]),
                       "nearest_range_m": nearest if math.isfinite(nearest) else None,
                       "nearest_azimuth_deg": float(az[ray]) if math.isfinite(nearest) else None,
                       "nearest_pixel": [int(inverse[ray]), 17-int(elevation)] if math.isfinite(nearest) else None})
    output = {"route": route, "dt": float(data["physics_dt"]),
              "terminal_step": terminal, "outcome": "collision",
              "checkpoint": metrics["checkpoint"], "checkpoint_step": metrics["checkpoint_step"],
              "scene_path": str(scene_path), "map_sha256": metrics["map_sha256"],
              "source_trajectory": str(args.trajectory),
              "position": states[:, :3].astype(float).tolist(),
              "velocity_world": states[:, 7:10].astype(float).tolist(),
              "command_ego": data["ego_velocity_command"][:terminal, route].astype(float).tolist(),
              "command_world": data["lee_target_velocity_world"][:terminal, route].astype(float).tolist(),
              "start": data["start_position"][route].astype(float).tolist(),
              "goal": data["target_position"][route].astype(float).tolist(),
              "obstacle_centers": centers.astype(float).tolist(),
              "obstacle_sizes": sizes.astype(float).tolist(),
              "nearest_terminal_box": closest_box,
              "nearest_terminal_box_surface_distance_m": float(distances[closest_box]),
              "terminal_position_semantics": "drone center at range-based collision decision, NOT contact point",
              "lidar_semantics": "real observations strictly before terminal; held until next recorded observation",
              "frames": frames}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, separators=(",", ":"), ensure_ascii=False))
    print(json.dumps({"route": route, "terminal_step": terminal,
                      "observations": len(frames), "nearest_box": closest_box,
                      "box_surface_distance_m": float(distances[closest_box]),
                      "json_bytes": args.output.stat().st_size}))


def render(args):
    from PIL import Image, ImageDraw, ImageFont

    data = json.loads(args.data.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    video_path = args.output/f"route_{data['route']:03d}_collision_replay.mp4"
    if video_path.exists():
        raise FileExistsError(video_path)
    font_path = args.font
    fonts = {size: ImageFont.truetype(str(font_path), size) for size in (18, 20, 22, 24, 26, 28, 30, 36)}
    width, height, fps = 1440, 900, args.fps
    background, ink = "#ffffff", "#16212c"
    blue, red, green, gray = "#146bb0", "#d52d2d", "#18834a", "#abb1b7"
    dt = data["dt"]
    terminal = data["terminal_step"]
    duration = terminal*dt
    observations = data["frames"]
    obs_steps = [f["step"] for f in observations]
    fronts, panos = [], []
    for frame in observations:
        fronts.append(Image.open(BytesIO(base64.b64decode(frame["front_png"]))).convert("RGB"))
        panos.append(Image.open(BytesIO(base64.b64decode(frame["pano_png"]))).convert("RGB"))
    final = data["position"][-1]
    first_slow = max(0., duration-3.)
    clock = [(min(i/fps, first_slow), "normal") for i in range(math.ceil(first_slow*fps))]
    clock += [(min(first_slow+i*.25/fps, duration), "slow") for i in range(math.ceil((duration-first_slow)*fps/.25))]
    clock += [(duration, "end") for _ in range(3*fps)]
    obstacle_pairs = list(zip(data["obstacle_centers"], data["obstacle_sizes"]))

    def text(draw, xy, value, size=24, fill=ink, anchor=None):
        draw.text(xy, value, font=fonts[size], fill=fill, anchor=anchor)

    def map_image(step, center, span, side, *, terminal_marker=False, small=False):
        image = Image.new("RGB", (side, side), background)
        draw = ImageDraw.Draw(image)
        half = span/2
        def xy(point):
            return ((point[0]-center[0]+half)/span*side,
                    (center[1]+half-point[1])/span*side)
        tick = 10 if span > 25 else 2 if span > 10 else 1
        lowx, highx = center[0]-half, center[0]+half
        lowy, highy = center[1]-half, center[1]+half
        for v in range(math.ceil(lowx/tick)*tick, math.floor(highx/tick)*tick+1, tick):
            x, _ = xy((v, 0))
            draw.line((x, 0, x, side), fill="#e7e9eb")
            if not small:
                text(draw, (x+4, side-28), str(v), size=18)
        for v in range(math.ceil(lowy/tick)*tick, math.floor(highy/tick)*tick+1, tick):
            _, y = xy((0, v))
            draw.line((0, y, side, y), fill="#e7e9eb")
            if not small:
                text(draw, (4, y+2), str(v), size=18)
        for index, (center_box, size) in enumerate(obstacle_pairs):
            x0, y0 = xy((center_box[0]-size[0]/2, center_box[1]+size[1]/2))
            x1, y1 = xy((center_box[0]+size[0]/2, center_box[1]-size[1]/2))
            if x1 < 0 or x0 > side or y1 < 0 or y0 > side:
                continue
            draw.rectangle((x0, y0, x1, y1), fill=gray)
            if terminal_marker and index == data["nearest_terminal_box"]:
                draw.rectangle((x0, y0, x1, y1), outline=red, width=3)
        path = [xy(point) for point in data["position"][:step+1]]
        if len(path) > 1:
            draw.line(path, fill=blue, width=2 if small else 4)
        sx, sy = xy(data["start"])
        gx, gy = xy(data["goal"])
        for px, py in ((sx, sy), (gx, gy)):
            draw.ellipse((px-4, py-4, px+4, py+4), fill=green)
        if not small:
            if 0 < sx < side and 0 < sy < side:
                text(draw, (sx+7, min(side-60, sy+6)), "起点", size=20, fill=green)
            if 0 < gx < side and 0 < gy < side:
                text(draw, (gx+7, max(0, gy-30)), "目标", size=20, fill=green)
        px, py = xy(data["position"][step])
        if terminal_marker:
            radius = 7 if small else 13
            draw.line((px-radius, py-radius, px+radius, py+radius), fill=red, width=3 if small else 5)
            draw.line((px-radius, py+radius, px+radius, py-radius), fill=red, width=3 if small else 5)
            if not small:
                draw.ellipse((px-.3/span*side, py-.3/span*side, px+.3/span*side, py+.3/span*side), outline=red, width=2)
                text(draw, (min(side-240, px+20), max(5, py-42)), "碰撞判定位置", size=28, fill=red)
        else:
            draw.ellipse((px-6, py-6, px+6, py+6), fill=blue, outline=background, width=2)
        if not small:
            command = data["command_world"][min(step, terminal-1)]
            seconds_scale = .7
            ex, ey = xy((data["position"][step][0]+command[0]*seconds_scale,
                         data["position"][step][1]+command[1]*seconds_scale))
            dx, dy = ex-px, ey-py
            norm = math.hypot(dx, dy)
            if norm > 1:
                ux, uy = dx/norm, dy/norm
                draw.line((px, py, ex, ey), fill=red, width=4)
                head = 12
                draw.polygon(((ex, ey), (ex-ux*head-uy*6, ey-uy*head+ux*6), (ex-ux*head+uy*6, ey-uy*head-ux*6)), fill=red)
        draw.rectangle((0, 0, side-1, side-1), outline="#8b949e", width=1)
        return image

    def make_frame(seconds, phase):
        step = min(terminal, int(round(seconds/dt)))
        observation_index = max(0, bisect_right(obs_steps, step)-1)
        obs = observations[observation_index]
        done = phase == "end"
        canvas = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(canvas)
        label = "碰撞判定 · 定格" if done else "最后3秒 · 0.25倍慢放" if phase == "slow" else "真实轨迹 · 1倍播放"
        text(draw, (30, 15), f"Route {data['route']}  碰撞前飞行回放", size=36)
        text(draw, (1405, 25), f"{seconds:06.3f} / {duration:.3f} s  |  {label}", size=24, anchor="ra", fill=red if done else ink)
        text(draw, (45, 76), "真实地图与轨迹（俯视：上为世界Y+，右为X+）", size=24)
        # Short episodes also start with an overview. Follow the actual drone
        # during zoom so its initial position cannot fall outside the crop.
        zoom_start = max(0., duration-6.)
        zoom_end = max(zoom_start+.5, duration-3.)
        zoom = min(1., max(0., (seconds-zoom_start)/(zoom_end-zoom_start)))
        zoom = zoom*zoom*(3-2*zoom)
        span = 50*(1-zoom)+7*zoom
        center = [data["position"][step][axis]*zoom for axis in (0, 1)]
        overview = map_image(step, center, span, 735, terminal_marker=done)
        canvas.paste(overview, (45, 115))
        if zoom > .01:
            inset = map_image(step, (0, 0), 50, 190, terminal_marker=done, small=True)
            canvas.paste(inset, (55, 650))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((55, 620, 245, 649), fill=background)
            text(draw, (60, 621), "全程轨迹", size=20)
        text(draw, (870, 76), "前方120°雷达：近黑、远白", size=26)
        front = fronts[observation_index].resize((518, 252), Image.Resampling.NEAREST)
        canvas.paste(front, (870, 120))
        text(draw, (870, 385), "完整360°雷达", size=22)
        pano = panos[observation_index].resize((518, 86), Image.Resampling.NEAREST)
        canvas.paste(pano, (870, 420))
        draw = ImageDraw.Draw(canvas)
        marker = obs["nearest_pixel"]
        if marker:
            mx, my = 870+(marker[0]+.5)*518/108, 420+(marker[1]+.5)*86/18
            draw.ellipse((mx-6, my-6, mx+6, my+6), outline=blue, width=3)
        for index, direction in enumerate(("后", "左", "前", "右", "后")):
            anchor = "la" if index == 0 else "ra" if index == 4 else "ma"
            text(draw, (870+index*518/4, 510), direction, size=20, anchor=anchor)
        text(draw, (870, 548), f"雷达帧：{obs['step']*dt:.3f} s（每0.16 s更新）", size=22)
        near = obs["nearest_range_m"]
        if near is not None:
            text(draw, (870, 585), f"最近回波 {near:.3f} m，方位 {obs['nearest_azimuth_deg']:+.1f}°", size=24)
        command = data["command_ego"][min(step, terminal-1)]
        norm = math.sqrt(sum(v*v for v in command))
        measured = math.sqrt(sum(v*v for v in data["velocity_world"][step]))
        text(draw, (870, 630), f"速度指令 {norm:.3f} m/s   实测 {measured:.3f} m/s", size=24)
        text(draw, (870, 673), f"前后 vx {command[0]:+.3f}   左右 vy {command[1]:+.3f}", size=22)
        text(draw, (870, 710), f"上下 vz {command[2]:+.3f} m/s   高度 {data['position'][step][2]:.2f} m", size=22)
        if done:
            text(draw, (870, 758), "红叉：触发碰撞判定时的无人机中心", size=24, fill=red)
            text(draw, (870, 799), f"最近实体表面距离 ≈ {data['nearest_terminal_box_surface_distance_m']:.3f} m", size=22)
            text(draw, (870, 833), "不是已测量的物理接触点", size=20)
        else:
            text(draw, (870, 763), "蓝线：已经飞过的轨迹", size=24, fill=blue)
            text(draw, (870, 805), "红箭头：当前水平速度指令", size=24, fill=red)
        text(draw, (45, 864), "灰块：障碍物俯视投影    绿色：起点 / 目标", size=20)
        return canvas

    for seconds, phase, name in ((0., "normal", "first_frame.png"), (duration-1., "slow", "before_collision.png"), (duration, "end", "final_collision.png")):
        make_frame(seconds, phase).save(args.output/name)
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required")
    command = [executable, "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-threads", "4", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index, (seconds, phase) in enumerate(clock):
            process.stdin.write(make_frame(seconds, phase).tobytes())
            if index % (fps*10) == 0:
                print(f"rendered {index}/{len(clock)} frames", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoding failed")
    except BaseException:
        process.terminate()
        process.wait()
        raise
    metadata = {"route": data["route"], "source": str(args.data),
                "terminal_step": terminal, "terminal_time_s": duration,
                "final_position_xyz_m": final, "video_frames": len(clock), "fps": fps,
                "playback": "1x, last 3 seconds 0.25x, final hold 3 seconds",
                "terminal_marker": data["terminal_position_semantics"],
                "nearest_terminal_box": data["nearest_terminal_box"],
                "nearest_terminal_box_surface_distance_m": data["nearest_terminal_box_surface_distance_m"],
                "last_lidar_lead_s": (terminal-obs_steps[-1])*dt,
                "video": str(video_path)}
    (args.output/"video_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("prepare")
    first.add_argument("--trajectory", type=Path, required=True)
    first.add_argument("--metrics", type=Path, required=True)
    first.add_argument("--route", type=int, default=21)
    first.add_argument("--output", type=Path, required=True)
    second = sub.add_parser("render")
    second.add_argument("--data", type=Path, required=True)
    second.add_argument("--output", type=Path, required=True)
    second.add_argument("--fps", type=int, default=20)
    second.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else render(args)


if __name__ == "__main__":
    main()
