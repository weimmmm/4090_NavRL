"""Visualize single-noise versus antithetic LiDAR Future-DiT rollouts.

The comparison is strictly autoregressive: both methods receive the same
three observed latents, then feed only their own prediction back at t+2 and
t+3.  The positive member of the antithetic pair uses exactly the same
initial noise as the single-noise prediction at every horizon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_video_dit import HistoryLatentDataset, sample_next
from lidar_wam.runner.world_direct_horizon import frame_metrics
from evaluate_lidar_video_dit_autoregressive import collect_rollout_rows, load_model


def range_image_360(frame: np.ndarray, prediction: bool) -> np.ndarray:
    """Return a 360-degree metric range image; near black, far/miss white."""
    threshold = 1.5 if prediction else 0.0
    hit = frame[1, :, :18] > threshold
    distance = np.clip((frame[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
    display = np.where(hit, distance, 10.0)

    # Stored azimuth is [0, 360). Reorder it to the intuitive [-180, 180)
    # convention, with 0 degrees (the forward direction) in the centre.
    azimuth = np.arange(108, dtype=np.float32) * (360.0 / 108.0)
    signed = (azimuth + 180.0) % 360.0 - 180.0
    order = np.argsort(signed)
    return display[order].T


def compact_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    row = frame_metrics(prediction, truth)
    return {
        "cd_paper_m2": float(row["cd_paper_m2"]),
        "depth_mae_m": float(row["valid_range_abs_error_sum_m"])
        / max(int(row["valid_range_count"]), 1),
        "mask_f1": (2.0 * int(row["tp"]))
        / max(2 * int(row["tp"]) + int(row["fp"]) + int(row["fn"]), 1),
        "gt_hit_count": int(row["gt_hit_count"]),
        "pred_hit_count": int(row["pred_hit_count"]),
    }


@torch.no_grad()
def rollout(model, history: torch.Tensor, horizons: int, steps: int,
            mode: str, noise_seed: int) -> torch.Tensor:
    outputs = []
    for horizon in range(horizons):
        # Reset per horizon so single noise equals the positive member of the
        # antithetic pair. Histories still differ after the first prediction.
        generator = torch.Generator(device=history.device).manual_seed(
            int(noise_seed) + 1009 * horizon)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = sample_next(
                model, history, steps, generator=generator,
                sampling_mode=mode)
        outputs.append(prediction)
        history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("visualization requires CUDA")
    device = torch.device(args.device)
    dataset = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=True)
    samples = collect_rollout_rows(dataset, args.count, args.seed, 3)
    history = torch.stack([
        torch.from_numpy(np.asarray(
            sample["latents"][sample["history_rows"]], dtype=np.float32))
        for sample in samples
    ]).to(device)
    truth = np.stack([
        np.asarray(sample["images"][sample["future_rows"]], dtype=np.float32)
        for sample in samples
    ])

    model, payload = load_model(args.checkpoint, device)
    vae = stage1.load_circular_vae().to(device).eval()
    scale = float(dataset.vae_metadata["scaling_factor"])
    single_latent = rollout(
        model, history.clone(), 3, args.flow_steps, "independent",
        args.noise_seed)
    antithetic_latent = rollout(
        model, history.clone(), 3, args.flow_steps, "antithetic",
        args.noise_seed)

    def decode(value: torch.Tensor) -> np.ndarray:
        flat = value.flatten(0, 1).float() / scale
        decoded = vae.decode(flat).sample.float().cpu().numpy()
        return decoded.reshape(args.count, 3, 2, 108, 20)

    predictions = {
        "single": decode(single_latent),
        "antithetic_x2": decode(antithetic_latent),
    }
    records = []
    for sample_index, sample in enumerate(samples):
        item = {
            "sample": sample_index + 1,
            "validation_index": int(sample["global_index"]),
            "shard": int(sample["shard"]),
            "future_rows": [int(v) for v in sample["future_rows"]],
            "single": [],
            "antithetic_x2": [],
        }
        for horizon in range(3):
            for name in ("single", "antithetic_x2"):
                item[name].append(compact_metrics(
                    predictions[name][sample_index, horizon],
                    truth[sample_index, horizon]))
        records.append(item)

    args.output.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for horizon in range(3):
        figure, axes = plt.subplots(
            args.count, 3, figsize=(15.5, 2.05 * args.count), dpi=180,
            sharex=True, sharey=True, squeeze=False)
        last_image = None
        for sample_index, record in enumerate(records):
            panels = (
                ("GT", truth[sample_index, horizon], False, None),
                ("single noise", predictions["single"][sample_index, horizon],
                 True, record["single"][horizon]),
                ("antithetic x2",
                 predictions["antithetic_x2"][sample_index, horizon], True,
                 record["antithetic_x2"][horizon]),
            )
            for column, (name, frame, prediction, metric) in enumerate(panels):
                axis = axes[sample_index, column]
                last_image = axis.imshow(
                    range_image_360(frame, prediction), origin="lower",
                    aspect="auto", cmap="gray", vmin=0.0, vmax=10.0,
                    extent=(-180.0, 180.0, -7.0, 52.0),
                    interpolation="nearest")
                axis.axvline(0.0, color="#d62728", linewidth=.55, alpha=.7)
                if sample_index == 0:
                    axis.set_title(name, fontsize=12, weight="bold")
                if metric is not None:
                    axis.text(
                        .012, .96,
                        f"CD {metric['cd_paper_m2']:.3f} m²  |  "
                        f"MAE {metric['depth_mae_m']:.2f} m",
                        transform=axis.transAxes, va="top", ha="left",
                        fontsize=8.2,
                        bbox={"facecolor": "white", "edgecolor": "none",
                              "alpha": .82, "pad": 1.8})
                if column == 0:
                    axis.set_ylabel(
                        f"sample {sample_index + 1}\nval {record['validation_index']}\n"
                        "elevation (deg)", fontsize=8.5)
                if sample_index == args.count - 1:
                    axis.set_xlabel("azimuth (deg; 0° forward)", fontsize=9)
                axis.set_xticks((-180, -90, 0, 90, 180))
        color_axis = figure.add_axes((.932, .10, .012, .82))
        figure.colorbar(
            last_image, cax=color_axis,
            label="range: 0 m black → 10 m / no return white")
        figure.suptitle(
            f"Strict autoregressive Future-DiT · t+{horizon + 1} · "
            f"checkpoint step {payload.get('step', '?')}\n"
            "fixed validation samples and matched initial noise; full 360° LiDAR",
            fontsize=13, y=.995)
        figure.subplots_adjust(
            left=.085, right=.91, bottom=.055, top=.94,
            hspace=.23, wspace=.08)
        image_path = args.output / f"t{horizon + 1}_single_vs_antithetic_x2_n{args.count}.png"
        figure.savefig(image_path, bbox_inches="tight")
        plt.close(figure)
        image_paths.append(str(image_path))

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "protocol": "strict autoregressive t+1:t+3",
        "selection": "fixed seeded validation sample; no metric cherry-picking",
        "count": int(args.count),
        "selection_seed": int(args.seed),
        "noise_seed": int(args.noise_seed),
        "flow_steps": int(args.flow_steps),
        "display": "full 360-degree range image; near black, far/miss white",
        "images": image_paths,
        "samples": records,
    }
    report_path = args.output / "single_vs_antithetic_x2_samples.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "checkpoint_step": report["checkpoint_step"],
        "count": report["count"],
        "images": image_paths,
        "report": str(report_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--video-index-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--noise-seed", type=int, default=271828)
    parser.add_argument("--flow-steps", type=int, default=20)
    main(parser.parse_args())
