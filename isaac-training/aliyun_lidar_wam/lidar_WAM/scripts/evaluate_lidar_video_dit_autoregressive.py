"""Evaluate history-only Video-DiT with a strictly autoregressive rollout."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_video_dit import HistoryLatentDataset, sample_next, _decode
from lidar_wam.runner.world_direct_horizon import frame_metrics


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = payload["architecture"]
    model = LiDARVideoDiT(
        width=int(cfg["width"]), depth=int(cfg["depth"]),
        heads=int(cfg["heads"]), mlp_ratio=float(cfg["mlp_ratio"]),
    ).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def successor_table(dataset, shard):
    path = dataset.dataset_root / dataset.entries[shard]["dataset"]
    with h5py.File(path, "r") as handle:
        frames = handle["frames"] if "frames" in handle else handle
        tokens = _decode(frames["token"][:])
        previous = _decode(frames["prev_token"][:])
        images = np.asarray(frames["range_values"][:], dtype=np.float32)
    # Every non-empty prev_token identifies its unique successor.
    successor = {}
    for row, token in enumerate(previous):
        if token:
            successor[token] = row
    return {token: successor.get(token, -1) for token in tokens}, images, tokens


def collect_rollout_rows(dataset, count, seed, horizon):
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(len(dataset))
    tables, raw_images, row_tokens = {}, {}, {}
    selected = []
    for global_index in candidates.tolist():
        shard = bisect.bisect_right(dataset.ends, int(global_index))
        start = 0 if shard == 0 else int(dataset.ends[shard - 1])
        rows, latents = dataset._open(shard)
        chain = rows[int(global_index) - start]
        if shard not in tables:
            tables[shard], raw_images[shard], row_tokens[shard] = successor_table(
                dataset, shard)
        table = tables[shard]
        current = int(chain[2])
        future = []
        for _ in range(horizon):
            following = table.get(row_tokens[shard][current], -1)
            if following < 0:
                break
            future.append(int(following))
            current = int(following)
        if len(future) != horizon:
            continue
        selected.append({
            "global_index": int(global_index), "shard": int(shard),
            "history_rows": [int(v) for v in chain[:3]],
            "future_rows": future, "latents": latents,
            "images": raw_images[shard],
        })
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"only found {len(selected)} validation chains with {horizon} successors")
    return selected


def overlay(ax, truth, prediction, title, cd, pred_threshold=1.5):
    gt = stage1.to_points(truth, 0.0)
    pred = stage1.to_points(prediction, pred_threshold)
    if len(gt):
        ax.scatter(gt[:, 0], gt[:, 1], s=10, c="#2878b5", label="GT")
    if len(pred):
        ax.scatter(pred[:, 0], pred[:, 1], s=12, c="#d84a3a", marker="x",
                   label="autoregressive prediction")
    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title} | CD {cd:.3f} m²\nGT {len(gt)} / pred {len(pred)}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.2)


@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=True)
    samples = collect_rollout_rows(dataset, args.count, args.seed, args.horizon)
    model, payload = load_model(args.checkpoint, device)
    vae = stage1.load_circular_vae().to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    scale = float(dataset.vae_metadata["scaling_factor"])

    # Only these three frames are observed at rollout start. All later input
    # frames are produced by the model itself.
    histories = torch.stack([
        torch.from_numpy(np.asarray(sample["latents"][sample["history_rows"]], dtype=np.float32))
        for sample in samples]).to(device)
    gt_latents = [
        torch.from_numpy(np.asarray(sample["latents"][sample["future_rows"]], dtype=np.float32))
        for sample in samples
    ]
    gt_latents = torch.stack(gt_latents).to(device)
    predicted_latents = []
    rollout_history = histories
    for horizon_index in range(args.horizon):
        predicted = sample_next(model, rollout_history, args.flow_steps)
        predicted_latents.append(predicted)
        rollout_history = torch.cat((rollout_history[:, 1:], predicted[:, None]), dim=1)
    predicted_latents = torch.stack(predicted_latents, dim=1)

    flat_pred = predicted_latents.reshape(-1, 4, 27, 5)
    flat_gt = gt_latents.reshape(-1, 4, 27, 5)
    predicted_images = vae.decode(flat_pred / scale).sample.float().cpu().numpy()
    gt_images = vae.decode(flat_gt / scale).sample.float().cpu().numpy()
    predicted_images = predicted_images.reshape(args.count, args.horizon, 2, 108, 20)
    gt_images = gt_images.reshape(args.count, args.horizon, 2, 108, 20)

    per_horizon = []
    for horizon_index in range(args.horizon):
        rows = []
        for sample_index, sample in enumerate(samples):
            metric = frame_metrics(predicted_images[sample_index, horizon_index],
                                   sample["images"][sample["future_rows"][horizon_index]])
            rows.append(float(metric["cd_paper_m2"]))
        per_horizon.append({
            "horizon": horizon_index + 1,
            "cd_mean_m2": float(np.mean(rows)),
            "cd_median_m2": float(np.median(rows)),
            "cd_p90_m2": float(np.percentile(rows, 90)),
            "values_m2": rows,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Detailed rollout for the first sampled route: each row is a horizon and
    # uses only predictions generated in the autoregressive loop.
    figure, axes = plt.subplots(args.horizon, 2, figsize=(7, 3.1 * args.horizon), dpi=150)
    if args.horizon == 1:
        axes = np.asarray([axes])
    first = samples[0]
    for horizon_index in range(args.horizon):
        prediction = predicted_images[0, horizon_index]
        truth = first["images"][first["future_rows"][horizon_index]]
        metric = per_horizon[horizon_index]["values_m2"][0]
        overlay(axes[horizon_index, 0], truth, truth,
                f"horizon {horizon_index + 1}: GT", 0.0,
                pred_threshold=0.0)
        overlay(axes[horizon_index, 1], truth, prediction,
                f"horizon {horizon_index + 1}: autoregressive", metric)
    figure.suptitle(
        f"Strict autoregressive rollout | checkpoint step {payload.get('step', '?')} | "
        f"route val index {first['global_index']}", y=0.995)
    figure.tight_layout()
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)

    plot = args.output.with_name(args.output.stem + "_metrics.png")
    figure, ax = plt.subplots(figsize=(7, 4), dpi=150)
    x = [row["horizon"] for row in per_horizon]
    ax.plot(x, [row["cd_mean_m2"] for row in per_horizon], "o-", label="mean")
    ax.plot(x, [row["cd_median_m2"] for row in per_horizon], "s--", label="median")
    ax.plot(x, [row["cd_p90_m2"] for row in per_horizon], "^:", label="p90")
    ax.set_xlabel("autoregressive horizon (frame)"); ax.set_ylabel("squared Chamfer (m²)")
    ax.set_xticks(x); ax.grid(alpha=0.25); ax.legend()
    ax.set_title(f"Autoregressive validation, N={args.count}")
    figure.tight_layout(); figure.savefig(plot, bbox_inches="tight"); plt.close(figure)

    summary = {
        "checkpoint": str(args.checkpoint), "checkpoint_step": int(payload.get("step", -1)),
        "count": int(args.count), "seed": int(args.seed), "flow_steps": int(args.flow_steps),
        "horizon": int(args.horizon), "initial_observation": "GT/sensor history t-2,t-1,t",
        "future_inputs": "previous model prediction only; no future GT is fed back",
        "per_horizon": per_horizon,
        "first_route_val_index": int(first["global_index"]),
        "output": str(args.output), "metrics_plot": str(plot),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--video-index-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--flow-steps", type=int, default=20)
    main(parser.parse_args())
