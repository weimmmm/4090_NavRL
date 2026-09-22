"""Evaluate both NavRL VAE checkpoints on exactly the same held-out frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lidar_wam.runner import stage1


def reference_vae():
    return stage1.load_circular_vae()


@torch.no_grad()
def evaluate(model, frames, threshold, batch_size):
    distance_sum = point_count = tp = fp = fn = 0.0
    chamfers = []
    for start in range(0, len(frames), batch_size):
        target = frames.image[start:start + batch_size]
        x = torch.from_numpy(target).to(stage1.DEVICE)
        reconstruction = model.decode(model.encode(x).latent_dist.mode()).sample.cpu().numpy()
        for truth, pred in zip(target, reconstruction):
            valid = truth[1, :, :18] > 0
            pred_valid = pred[1, :, :18] > threshold
            distance_sum += float((np.abs(pred[0, :, :18] - truth[0, :, :18])
                                   * valid).sum() * 5)
            point_count += int(valid.sum())
            tp += int((valid & pred_valid).sum())
            fp += int((~valid & pred_valid).sum())
            fn += int((valid & ~pred_valid).sum())
            chamfers.append(stage1.chamfer(stage1.to_points(pred, threshold),
                                           stage1.to_points(truth)))
    return {"samples": len(frames), "threshold_logit": threshold,
            "target_valid_radial_mae_m": distance_sum / max(point_count, 1),
            "mask_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "chamfer_m": float(np.mean(chamfers)),
            "chamfer_median_m": float(np.median(chamfers)),
            "chamfer_p95_m": float(np.percentile(chamfers, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    frames = stage1.Frames(args.split, limit=args.samples)
    external = reference_vae()
    external_result = evaluate(external, frames, 1.5, args.batch_size)
    del external
    torch.cuda.empty_cache()
    local = stage1.make_vae(circular=False).to(stage1.DEVICE).float().eval()
    stage1.load_model(stage1.OUT / "vae_full" / "best.pt", local)
    local_result = evaluate(local, frames, 2.0, args.batch_size)
    result = {"split": args.split, "source_indices": frames.indices.tolist(),
              "external": external_result, "local": local_result}
    output = stage1.OUT / "vae_full" / f"direct_comparison_{args.split}_n{len(frames)}.json"
    stage1.save_json(output, result)
    print(json.dumps({"split": args.split, "external": external_result,
                      "local": local_result}), flush=True)


if __name__ == "__main__":
    main()
