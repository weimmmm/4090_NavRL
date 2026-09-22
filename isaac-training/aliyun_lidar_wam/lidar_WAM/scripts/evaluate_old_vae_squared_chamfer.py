"""Compute strict squared point-cloud Chamfer for the historical two-channel VAE."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1


def chamfer_squared(prediction, target, threshold):
    pred = stage1.to_points(prediction, threshold)
    truth = stage1.to_points(target, 0.0)
    if not len(pred) and not len(truth):
        return 0.0
    if not len(pred) or not len(truth):
        return 200.0
    return float(np.square(cKDTree(pred).query(truth)[0]).mean()
                 + np.square(cKDTree(truth).query(pred)[0]).mean())


@torch.no_grad()
def evaluate(args):
    dataset = stage1.Frames(args.split, limit=args.limit)
    vae = stage1.load_circular_vae()
    rows = []
    for batch_start in range(0, len(dataset), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(dataset))
        images = torch.stack([dataset[i] for i in range(batch_start, batch_end)]).to(stage1.DEVICE)
        recon = vae.decode(vae.encode(images).latent_dist.mode()).sample.cpu().numpy()
        for j, index in enumerate(range(batch_start, batch_end)):
            target = dataset.image[index]
            rows.append({"source_index": int(dataset.indices[index]),
                         "seed": int(dataset.seeds[index]),
                         "cd_paper_m2": chamfer_squared(recon[j], target, args.threshold),
                         "gt_points": int(len(stage1.to_points(target))),
                         "recon_points": int(len(stage1.to_points(recon[j], args.threshold)))})
        if batch_start == 0 or batch_end == len(dataset):
            print(json.dumps({"processed": batch_end, "total": len(dataset)}), flush=True)
    result = {"split": args.split, "samples": len(rows),
              "vae_checkpoint": str(stage1.CIRCULAR_VAE / "diffusion_pytorch_model.safetensors"),
              "vae_step": json.loads((stage1.CIRCULAR_VAE / "best_validation.json").read_text())["step"],
              "mask_threshold": args.threshold,
              "definition": "sum of GT-to-reconstruction and reconstruction-to-GT mean squared nearest-neighbor distances, m^2",
              "empty_cloud_policy": "200 m^2 when exactly one cloud is empty; 0 when both are empty",
              "mean_cd_paper_m2": float(np.mean([row["cd_paper_m2"] for row in rows])),
              "median_cd_paper_m2": float(np.median([row["cd_paper_m2"] for row in rows])),
              "empty_reconstruction_frames": int(sum(row["recon_points"] == 0 and row["gt_points"] > 0 for row in rows)),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("mean_cd_paper_m2", "median_cd_paper_m2",
                                               "empty_reconstruction_frames", "samples")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=stage1.OUT / "evaluation" / "old_vae_squared_chamfer.json")
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    evaluate(args)


if __name__ == "__main__":
    main()
