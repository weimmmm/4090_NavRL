"""Compare single-sample and latent-mean-8 autoregressive Future-DiT rollouts.

The two methods receive the same initial three GT latents.  At each horizon
they also start from the same first noise seed.  Predicted t+1 and t+2 latents
are fed back into their respective histories; future GT is visualization and
metrics only.
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
from visualize_lidar_video_dit_ar3 import load_model, select_sequences


def front_depth(frame: np.ndarray, prediction: bool):
    """Return a forward metric depth image: near=black, far/no-return=white."""
    threshold = 1.5 if prediction else 0.0
    hit = frame[1, :, :18] > threshold
    distance = np.clip((frame[0, :, :18] + 1.0) * 5.0, 0.0, 10.0)
    azimuth = np.arange(108, dtype=np.float32) * (360.0 / 108.0)
    signed = (azimuth + 180.0) % 360.0 - 180.0
    chosen = np.flatnonzero(np.abs(signed) <= 60.0)
    chosen = chosen[np.argsort(signed[chosen])]
    # A LiDAR miss means no obstacle within the 10 m sensing range.
    depth = np.where(hit, distance, 10.0)[chosen].T
    return depth, signed[chosen]


@torch.no_grad()
def rollout(model, initial_history, horizons: int, flow_steps: int,
            samples: int, noise_seed: int):
    history = initial_history
    predictions = []
    for horizon in range(horizons):
        # Reset by horizon so the first draw is identical between the single
        # and mean-8 methods, while their autoregressive histories may differ.
        generator = torch.Generator(device=history.device).manual_seed(
            int(noise_seed) + 1009 * horizon)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = sample_next(
                model, history, flow_steps, generator=generator,
                num_samples=samples)
        predictions.append(predicted)
        history = torch.cat((history[:, 1:], predicted[:, None]), dim=1)
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("visualization requires CUDA")
    device = torch.device(args.device)
    dataset = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=False)
    sequences = select_sequences(dataset, args.count, args.seed)
    initial = torch.from_numpy(np.stack([
        row["history"] for row in sequences])).to(device)
    truth = np.stack([row["truth"] for row in sequences])

    model, payload = load_model(args.checkpoint, device)
    vae = stage1.load_circular_vae().to(device).eval()
    scale = float(dataset.vae_metadata["scaling_factor"])

    single_latent = rollout(
        model, initial, 3, args.flow_steps, 1, args.noise_seed)
    mean8_latent = rollout(
        model, initial, 3, args.flow_steps, 8, args.noise_seed)

    def decode(latent):
        flat = latent.flatten(0, 1).float() / scale
        return vae.decode(flat).sample.float().cpu().numpy().reshape(
            args.count, 3, 2, 108, 20)

    predictions = {
        "single": decode(single_latent),
        "latent_mean8": decode(mean8_latent),
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    records = []
    figure_paths = []
    for sample, sequence in enumerate(sequences):
        metrics = {name: [] for name in predictions}
        for name, images in predictions.items():
            for horizon in range(3):
                row = frame_metrics(images[sample, horizon], truth[sample, horizon])
                metrics[name].append({
                    "cd_paper_m2": float(row["cd_paper_m2"]),
                    "depth_mae_m": float(row["valid_range_abs_error_sum_m"])
                    / max(int(row["valid_range_count"]), 1),
                    "pred_hit_count": int(row["pred_hit_count"]),
                    "gt_hit_count": int(row["gt_hit_count"]),
                })

        figure, axes = plt.subplots(
            3, 3, figsize=(15, 8.6), dpi=190, sharex=True, sharey=True)
        plot = None
        rows = (("GT", truth[sample], False, None),
                ("Single sample", predictions["single"][sample], True,
                 "single"),
                ("Latent mean ×8", predictions["latent_mean8"][sample], True,
                 "latent_mean8"))
        for row_index, (label, frames, is_prediction, metric_name) in enumerate(rows):
            for horizon in range(3):
                depth, signed = front_depth(frames[horizon], is_prediction)
                plot = axes[row_index, horizon].imshow(
                    depth, origin="lower", aspect="auto", cmap="gray",
                    vmin=0.0, vmax=10.0,
                    extent=(float(signed.min()), float(signed.max()), -7, 52),
                    interpolation="nearest")
                title = f"{label} · t+{horizon + 1}"
                if metric_name is not None:
                    value = metrics[metric_name][horizon]
                    title += (f"\nCD={value['cd_paper_m2']:.3f} m² · "
                              f"MAE={value['depth_mae_m']:.3f} m")
                axes[row_index, horizon].set_title(title, fontsize=10)
                axes[row_index, horizon].axvline(
                    0, color="#e05252", linewidth=.65, alpha=.75)
                axes[row_index, horizon].set_xlabel("azimuth (deg; 0° forward)")
                axes[row_index, horizon].set_ylabel("elevation angle (deg)")
        color_axis = figure.add_axes((.925, .13, .014, .72))
        figure.colorbar(plot, cax=color_axis, label="range (m): near black → far white")
        figure.suptitle(
            f"Future-DiT autoregressive comparison · validation sample "
            f"{sequence['index']}\n"
            "same initial GT history; later predictions use predicted history only",
            y=.985, fontsize=13)
        figure.subplots_adjust(
            left=.065, right=.9, bottom=.075, top=.88, hspace=.5, wspace=.18)
        path = output / f"ar3_single_vs_mean8_sample_{sequence['index']:06d}.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        figure_paths.append(str(path))
        records.append({
            "dataset_index": int(sequence["index"]),
            "shard": int(sequence["shard"]),
            "future_rows": [int(v) for v in sequence["future_rows"]],
            "metrics": metrics,
            "figure": str(path),
        })

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "protocol": "strict autoregressive t+1:t+3",
        "single_samples": 1,
        "ensemble_samples": 8,
        "flow_steps": int(args.flow_steps),
        "selection_seed": int(args.seed),
        "noise_seed": int(args.noise_seed),
        "samples": records,
        "figures": figure_paths,
    }
    report_path = output / "ar3_single_vs_latent_mean8_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--video-index-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--noise-seed", type=int, default=271828)
    parser.add_argument("--flow-steps", type=int, default=20)
    main(parser.parse_args())
