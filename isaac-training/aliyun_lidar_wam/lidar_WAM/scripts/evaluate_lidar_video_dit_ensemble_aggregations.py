"""Compare robust aggregation rules for eight Future-DiT latent samples."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evaluate_lidar_video_dit_samplers import integrate, load_model, summarize
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_video_dit import HistoryLatentDataset
from lidar_wam.runner.world_direct_horizon import frame_metrics


def aggregate(values):
    """Aggregate [K,B,C,H,W] samples without consulting ground truth."""
    count, batch = values.shape[:2]
    flattened = values.permute(1, 0, 2, 3, 4).reshape(batch, count, -1)
    mean = values.mean(0)

    # Medoid: select the actual generated endpoint with the smallest summed
    # squared distance to all other generated endpoints.
    pairwise = (flattened[:, :, None] - flattened[:, None, :]).square().mean(-1)
    medoid_index = pairwise.sum(-1).argmin(-1)
    batch_index = torch.arange(batch, device=values.device)
    by_batch = values.permute(1, 0, 2, 3, 4)
    medoid = by_batch[batch_index, medoid_index]

    # Trimmed mean: reject the four endpoints furthest from the eight-sample
    # mean independently for each batch item, then average the central four.
    center = flattened.mean(1, keepdim=True)
    central_index = (flattened - center).square().mean(-1).topk(
        4, dim=1, largest=False).indices
    gather_index = central_index[..., None, None, None].expand(
        -1, -1, *values.shape[2:])
    central = torch.gather(by_batch, 1, gather_index).mean(1)

    # Coordinate median is robust to outliers but may not itself lie on the
    # learned latent manifold.  Use the conventional average of middle values.
    ordered = values.sort(dim=0).values
    median = 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])
    return {
        "single_first": values[0],
        "latent_mean8": mean,
        "latent_medoid8": medoid,
        "latent_trimmed_mean4of8": central,
        "latent_coordinate_median8": median,
    }


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    dataset = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=True)
    count = min(int(args.eval_samples), len(dataset))
    rng = np.random.default_rng(int(args.seed))
    indices = np.sort(rng.choice(len(dataset), count, replace=False))
    loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=args.eval_batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    model, payload = load_model(args.checkpoint, device)
    vae = stage1.load_circular_vae().to(device).eval()
    scale = float(dataset.vae_metadata["scaling_factor"])
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)
    rows = {name: [] for name in (
        "single_first", "latent_mean8", "latent_medoid8",
        "latent_trimmed_mean4of8", "latent_coordinate_median8")}
    started = time.time()
    for history, _, target in loader:
        history = history.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            samples = torch.stack([
                integrate(model, history, args.flow_steps, "euler", generator)
                for _ in range(8)], dim=0)
            aggregated = aggregate(samples)
        names = list(aggregated)
        latents = torch.cat([aggregated[name] for name in names], dim=0)
        decoded = vae.decode(latents.float() / scale).sample.float().cpu().numpy()
        decoded = decoded.reshape(len(names), len(history), 2, 108, 20)
        target = target.numpy()
        for method_index, name in enumerate(names):
            for prediction, truth in zip(decoded[method_index], target):
                rows[name].append(frame_metrics(prediction, truth, 1.5))

    results = {name: summarize(value) for name, value in rows.items()}
    best = min(results, key=lambda name: results[name][
        "nonempty_gt_cd_paper_m2"])
    report = {
        "format": "navrl-future-dit-ensemble-aggregation-ablation-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "eval_samples": count,
        "nonempty_gt_samples": results[best]["nonempty_gt_samples"],
        "flow_steps": int(args.flow_steps),
        "ensemble_size": 8,
        "selection_seed": int(args.seed),
        "noise_seed": int(args.noise_seed),
        "best_method": best,
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--video-index-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=12345)
    parser.add_argument("--flow-steps", type=int, default=20)
    main(parser.parse_args())
