"""Render reproducible fifth-frame GT and autoregressive LiDAR predictions."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.executed_residual import ExecutedLatents
from lidar_wam.runner.lidar_geometry import frame_points, load_rays
from evaluate_autoregressive_5 import eligible_starts
from evaluate_representative import fetch, metric, save_json


def select_cases(report):
    rows = [r for r in report["rows"] if r["horizon"] == 5]
    cases = {}
    for label, subset in (("overall_median", rows),
                          ("seed19_median", [r for r in rows if r["seed"] == 19])):
        ordered = sorted(subset, key=lambda r: r["autoregressive"]["chamfer_m"])
        cases[label] = ordered[(len(ordered) - 1) // 2]
    return cases


def save_ply(path, xyz):
    with path.open("w") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(xyz)}\n")
        file.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(file, xyz, fmt="%.6f")


def render(path, truth, prediction, score, seed, frame):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    clouds = (truth, prediction)
    names = ("Ground truth", "Autoregressive prediction")
    for row, x_axis, y_axis, y_label, y_lim in ((0, 0, 1, "Y (m)", (-10, 10)),
                                                 (1, 0, 2, "Z (m)", (-3, 9))):
        for col, (cloud, name) in enumerate(zip(clouds, names)):
            axis = axes[row, col]
            axis.scatter(cloud[:, x_axis], cloud[:, y_axis], c=cloud[:, 2],
                         cmap="viridis", vmin=-3, vmax=9, s=4, alpha=.8,
                         linewidths=0, rasterized=True)
            axis.set_title(f"{name} ({len(cloud)} points)")
            axis.set_xlim(-10, 10)
            axis.set_ylim(*y_lim)
            axis.set_aspect("equal")
            axis.set_xlabel("X (m)")
            axis.set_ylabel(y_label)
            axis.grid(alpha=.18)
        axis = axes[row, 2]
        axis.scatter(truth[:, x_axis], truth[:, y_axis], c="#252525", s=6,
                     alpha=.48, linewidths=0, label="GT", rasterized=True)
        axis.scatter(prediction[:, x_axis], prediction[:, y_axis], c="#f26b38",
                     s=5, alpha=.43, linewidths=0, label="Prediction", rasterized=True)
        axis.set_title("Overlay")
        axis.set_xlim(-10, 10)
        axis.set_ylim(*y_lim)
        axis.set_aspect("equal")
        axis.set_xlabel("X (m)")
        axis.set_ylabel(y_label)
        axis.grid(alpha=.18)
        axis.legend(markerscale=2)
    fig.suptitle(f"Test seed {seed}, frame {frame} + 5 (0.80 s) | Chamfer {score:.3f} m\n"
                 "Top: XY view   Bottom: XZ view   Sensor coordinates", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--report", type=Path, default=stage1.OUT / "representative_baseline" /
                        "test_lagen_unet_8h_autoregressive_5.json")
    parser.add_argument("--output", type=Path, default=stage1.OUT / "representative_baseline" /
                        "figures" / "autoregressive_5")
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    args.raw_root = (args.raw_root or stage1.DATA.parent).expanduser().resolve()
    report = json.loads(args.report.read_text())
    cases = select_cases(report)
    manifest = json.loads((args.report.parent / "sample_manifest.json").read_text())
    data = ExecutedLatents("test")
    with h5py.File(stage1.DATA / "navrl_static_test.h5", "r") as h5:
        starts, trajectories, _ = eligible_starts(manifest, data.indices, h5)
    lookup = {int(index): position for position, index in enumerate(data.indices)}
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())["scaling_factor"]
    threshold = report["mask_logit_threshold"]
    model = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    checkpoint = stage1.load_model(stage1.OUT / report["checkpoint_run"] /
                                   report["checkpoint_name"], model)
    if checkpoint["step"] != report["checkpoint_step"]:
        raise ValueError("Checkpoint changed since metric evaluation")
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    vae = stage1.load_circular_vae()
    outputs = {}
    with h5py.File(stage1.DATA / "navrl_static_test.h5", "r") as h5:
        for label, record in cases.items():
            seed = record["seed"]
            locations = np.array([i for i, row in enumerate(starts) if row["seed"] == seed])
            offset = next(i for i, location in enumerate(locations)
                          if starts[location]["source_index"] == record["source_index"])
            batch_start = offset // args.batch_size * args.batch_size
            chosen = locations[batch_start:batch_start + args.batch_size]
            within = offset - batch_start
            latent = data.previous[[lookup[starts[i]["source_index"]]
                                    for i in chosen]].to(stage1.DEVICE)
            initial_indices = np.array([starts[i]["source_index"] for i in chosen])
            state = torch.from_numpy(stage1.causal_state(
                fetch(h5, "prev_ego_feats", initial_indices))).to(stage1.DEVICE)
            for h in range(5):
                indices = trajectories[chosen, h]
                actions = torch.from_numpy(fetch(h5, "normalized_action_sequence",
                                                indices)).to(stage1.DEVICE)
                latent = stage1.generate(model, scheduler, latent, actions, state,
                                         42 + seed * 10000 + batch_start + h * 100000,
                                         init_strength=1.0, num_steps=20)
            predicted_frame = vae.decode(latent / scale).sample[within].cpu().numpy()
            target = h5["range_values"][record["target_source_index"]]
            rays = load_rays(args.raw_root, "test", seed)[0]
            measured = metric(predicted_frame, target, rays, threshold)["chamfer_m"]
            expected = record["autoregressive"]["chamfer_m"]
            if abs(measured - expected) > 1e-4:
                raise ValueError(f"Cannot reproduce {label}: measured {measured}, expected {expected}")
            truth_xyz = frame_points(target, rays)
            predicted_xyz = frame_points(predicted_frame, rays, threshold)
            stem = f"{label}_seed{seed}_source{record['source_index']}_h5"
            figure = args.output / f"{stem}.png"
            gt_ply = args.output / f"{stem}_gt.ply"
            predicted_ply = args.output / f"{stem}_prediction.ply"
            args.output.mkdir(parents=True, exist_ok=True)
            render(figure, truth_xyz, predicted_xyz, measured, seed,
                   record["frame_idx"])
            save_ply(gt_ply, truth_xyz)
            save_ply(predicted_ply, predicted_xyz)
            outputs[label] = {"seed": seed, "source_index": record["source_index"],
                              "target_source_index": record["target_source_index"],
                              "chamfer_m": measured, "gt_points": len(truth_xyz),
                              "predicted_points": len(predicted_xyz),
                              "figure": str(figure), "gt_ply": str(gt_ply),
                              "prediction_ply": str(predicted_ply)}
            print(json.dumps({label: outputs[label]}), flush=True)
    save_json(args.output / "selected_examples.json", outputs)


if __name__ == "__main__":
    main()
