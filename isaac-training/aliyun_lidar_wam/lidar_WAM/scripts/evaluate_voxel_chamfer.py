"""Evaluate trial voxel-anchor and LaGen-style fixed-1500 Chamfer metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from lidar_wam.data_v2 import V2WindowDataset
from lidar_wam.runner import stage1
from lidar_wam.runner.world_direct_horizon import (
    DirectHorizonWorldModel,
    WorldV2View,
    evaluate_world_v2,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim-steps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = V2WindowDataset(
        "val", args.dataset_root, args.latent_root, args.index_root,
        samples_per_seed=args.samples, random_seed=args.seed)
    view = WorldV2View(dataset)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DirectHorizonWorldModel().to(device).float()
    model.load_state_dict(payload["model"], strict=True)
    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    evaluation_args = SimpleNamespace(
        eval_batch_size=args.batch_size,
        workers=args.workers,
        precision="bf16",
        seed=args.seed,
        ddim_steps=args.ddim_steps,
        mask_threshold=1.5,
    )
    try:
        result = evaluate_world_v2(model, vae, view, evaluation_args, device)
    finally:
        dataset.close()
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(payload.get("step", -1)),
        "voxel_size_m": 0.20,
        "fixed_point_count": 1500,
        "fixed_point_padding": "lagen_prefix_copy_zero_pad",
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
