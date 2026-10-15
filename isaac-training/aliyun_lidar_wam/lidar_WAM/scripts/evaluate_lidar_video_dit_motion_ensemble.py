"""Evaluate single/ensemble sampling for LiDAR Future-DiT checkpoints.

Unlike the legacy sampler ablation, this evaluator also supports checkpoints
conditioned on causal action history, current proprioception and goal.  Every
ensemble size uses the same validation windows and starts from the same noise
seed, so differences are attributable to aggregation rather than data choice.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_video_dit import (
    HistoryLatentDataset,
    MotionHistoryLatentDataset,
    configure_prediction_from_payload,
    sample_next,
)
from lidar_wam.runner.world_direct_horizon import frame_metrics
from evaluate_lidar_video_dit_samplers import summarize


def load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture = dict(payload.get("architecture", {}))
    motion = bool(architecture.get("motion_conditioning", False))
    model = LiDARVideoDiT(
        width=int(architecture.get("width", 512)),
        depth=int(architecture.get("depth", 8)),
        heads=int(architecture.get("heads", 8)),
        mlp_ratio=float(architecture.get("mlp_ratio", 4.0)),
        motion_conditioning=motion,
        past_horizon=int(architecture.get("past_horizon", 30)),
        proprio_dim=int(architecture.get("proprio_dim", 10)),
        goal_dim=int(architecture.get("goal_dim", 4)),
    ).to(device).eval()
    model.load_state_dict(payload.get("model", payload), strict=True)
    configure_prediction_from_payload(model, payload)
    return model, payload, architecture, motion


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    model, payload, architecture, motion = load_model(args.checkpoint, device)
    dataset_type = MotionHistoryLatentDataset if motion else HistoryLatentDataset
    dataset_kwargs = {"past_horizon": int(architecture.get("past_horizon", 30))} \
        if motion else {}
    dataset = dataset_type(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=True, **dataset_kwargs)
    count = min(int(args.eval_samples), len(dataset))
    offset = int(args.selection_offset)
    if offset < 0 or offset + count > len(dataset):
        raise ValueError(
            f"selection range [{offset},{offset + count}) exceeds "
            f"validation size {len(dataset)}")
    # One seeded permutation plus non-overlapping offsets gives reproducible,
    # disjoint validation subsets for robustness sweeps.
    permutation = np.random.default_rng(args.seed).permutation(len(dataset))
    indices = np.sort(permutation[offset:offset + count])
    loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=args.eval_batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    vae = stage1.load_circular_vae().to(device).eval()
    scale = float(dataset.vae_metadata["scaling_factor"])
    motion_stats = architecture.get("motion_stats") if motion else None
    if motion and not motion_stats:
        raise ValueError("motion-conditioned checkpoint has no motion_stats")

    results = {}
    methods = [(f"latent_mean{size}", int(size))
               for size in args.ensemble_sizes]
    if args.include_zero_noise:
        methods.insert(0, ("zero_noise", 0))
    if args.include_antithetic:
        methods.insert(1 if args.include_zero_noise else 0,
                       ("antithetic_mean2", -2))
    for method_name, ensemble_size in methods:
        generator = torch.Generator(device=device).manual_seed(args.noise_seed)
        rows = []
        started = time.time()
        for batch in loader:
            if motion:
                history, _, past, mask, proprio, goal, target_image = batch
                past = past.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                proprio = proprio.to(device, non_blocking=True)
                goal = goal.to(device, non_blocking=True)
            else:
                history, _, target_image = batch
                past = mask = proprio = goal = None
            history = history.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                conditions = dict(
                    past_actions=past, past_mask=mask, proprio=proprio,
                    goal=goal, motion_stats=motion_stats)
                sampling_mode = (
                    "zero" if ensemble_size == 0 else
                    "antithetic" if ensemble_size == -2 else
                    "independent")
                prediction = sample_next(
                    model, history, args.flow_steps, generator,
                    num_samples=max(ensemble_size, 1),
                    sampling_mode=sampling_mode, **conditions)
            decoded = vae.decode(prediction.float() / scale).sample.cpu().numpy()
            for predicted_image, truth in zip(decoded, target_image.numpy()):
                rows.append(frame_metrics(predicted_image, truth, 1.5))
        result = summarize(rows)
        result.update({"ensemble_size": int(ensemble_size),
                       "elapsed_seconds": time.time() - started})
        results[method_name] = result
        print(json.dumps({"method": method_name, **result}), flush=True)

    best = min(results, key=lambda key: results[key][
        "nonempty_gt_cd_paper_m2"])
    report = {
        "format": "navrl-future-dit-motion-ensemble-ablation-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "motion_conditioning": motion,
        "prediction_target": architecture.get("prediction_target", "absolute_latent"),
        "eval_samples": count,
        "flow_steps": int(args.flow_steps),
        "selection_seed": int(args.seed),
        "selection_offset": offset,
        "noise_seed": int(args.noise_seed),
        "best_method": best,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "best_method": best,
                      "best": results[best]}, indent=2), flush=True)


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
    parser.add_argument("--selection-offset", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=12345)
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument("--ensemble-sizes", type=int, nargs="*",
                        default=(1, 2, 4, 8))
    parser.add_argument("--include-zero-noise", action="store_true")
    parser.add_argument("--include-antithetic", action="store_true")
    evaluate(parser.parse_args())
