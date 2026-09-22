"""Single-channel sentinel-range VAE for NavRL LiDAR.

This is intentionally isolated from the historical two-channel range+mask
experiments.  The source HDF5 remains unchanged; conversion happens in memory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from lidar_wam.runner import stage1, utils


NO_RETURN = -1.0
VALID_MIN = -0.8
VALID_THRESHOLD = -0.9
MAX_RANGE_M = 10.0
RUN_ROOT = stage1.OUT / "dense_sentinel"


def encode_dense(frame):
    """Convert historical [range, hit-mask] data to one sentinel channel."""
    hit = frame[:, 1:2] > 0
    fraction = np.clip((frame[:, 0:1] + 1.0) / 2.0, 0.0, 1.0)
    encoded = VALID_MIN + (1.0 - VALID_MIN) * fraction
    return np.where(hit, encoded, NO_RETURN).astype(np.float32)


def decode_distance(encoded):
    """Convert predicted valid values to metric range; validity is decided first."""
    return np.clip((encoded - VALID_MIN) / (1.0 - VALID_MIN) * MAX_RANGE_M,
                   0.0, MAX_RANGE_M)


class DenseFrames(Dataset):
    def __init__(self, split, overfit=False, limit=None):
        with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
            indices, self.counts = stage1.valid_indices(h5, split)
            if overfit:
                indices = indices[:128]
            elif limit is not None:
                indices = indices[:limit]
            source = h5["range_values"][indices].astype(np.float32)
        self.image = encode_dense(source)
        self.indices = indices

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return torch.from_numpy(self.image[index])


def make_vae():
    config = json.loads((stage1.CIRCULAR_VAE / "config.json").read_text())
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    config.update({"sample_size": [108, 20], "in_channels": 1,
                   "out_channels": 1, "scaling_factor": 1.0})
    model = stage1.AutoencoderKL(**config)
    utils.replace_down(model)
    utils.replace_conv(model)
    utils.replace_attn(model)
    return model


def loss_fn(model, target):
    posterior = model.encode(target).latent_dist
    prediction = model.decode(posterior.sample()).sample
    target = target[:, :, :, :18]
    prediction_physical = prediction[:, :, :, :18]
    hit = target > VALID_THRESHOLD
    no_return = ~hit
    error = (prediction_physical - target).abs()
    valid_l1 = error[hit].mean()
    no_return_l1 = error[no_return].mean()
    hit_margin = F.relu(VALID_THRESHOLD - prediction_physical[hit]).mean()
    no_return_margin = F.relu(prediction_physical[no_return] - VALID_THRESHOLD).mean()
    kl = posterior.kl().mean()
    total = (3.0 * valid_l1 + no_return_l1 + 0.5 * hit_margin
             + 0.5 * no_return_margin + 1e-6 * kl)
    return total, {"valid_l1": valid_l1, "no_return_l1": no_return_l1,
                   "hit_margin": hit_margin, "no_return_margin": no_return_margin,
                   "kl": kl}


def cloud(encoded):
    encoded = encoded[0, :, :18]
    valid = encoded > VALID_THRESHOLD
    distance = decode_distance(encoded)
    return (stage1.DIRECTIONS[valid] * distance[valid, None]).astype(np.float32)


def squared_chamfer(prediction, target):
    pred = cloud(prediction)
    truth = cloud(target)
    if not len(pred) and not len(truth):
        return 0.0
    if not len(pred) or not len(truth):
        return 2.0 * MAX_RANGE_M ** 2
    return float(np.square(cKDTree(pred).query(truth)[0]).mean()
                 + np.square(cKDTree(truth).query(pred)[0]).mean())


@torch.no_grad()
def evaluate(model, dataset, batch_size):
    model.eval()
    valid_abs_m = valid_count = tp = fp = fn = 0
    false_empty = true_empty = below = above = values = 0
    chamfer_sum = 0.0
    for target in DataLoader(dataset, batch_size=batch_size):
        target = target.to(stage1.DEVICE)
        prediction = model.decode(model.encode(target).latent_dist.mode()).sample
        physical_target = target[:, :, :, :18]
        physical_prediction = prediction[:, :, :, :18]
        actual = physical_target > VALID_THRESHOLD
        predicted = physical_prediction > VALID_THRESHOLD
        target_m = decode_distance(physical_target.cpu().numpy())
        prediction_m = decode_distance(physical_prediction.cpu().numpy())
        valid_abs_m += float((np.abs(prediction_m - target_m) *
                              actual.cpu().numpy()).sum())
        valid_count += int(actual.sum())
        tp += int((predicted & actual).sum())
        fp += int((predicted & ~actual).sum())
        fn += int((~predicted & actual).sum())
        actual_frame = actual.flatten(1).any(1)
        predicted_frame = predicted.flatten(1).any(1)
        false_empty += int((actual_frame & ~predicted_frame).sum())
        true_empty += int((~actual_frame).sum())
        below += int((physical_prediction < -1.0).sum())
        above += int((physical_prediction > 1.0).sum())
        values += physical_prediction.numel()
        for p, t in zip(prediction.cpu().numpy(), target.cpu().numpy()):
            chamfer_sum += squared_chamfer(p, t)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"samples": len(dataset),
            "hit_range_mae_m": valid_abs_m / max(valid_count, 1),
            "hit_precision": precision, "hit_recall": recall,
            "hit_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "false_empty_frames": false_empty, "true_empty_frames": true_empty,
            "cd_paper_m2": chamfer_sum / len(dataset),
            "below_minus_one_fraction": below / max(values, 1),
            "above_plus_one_fraction": above / max(values, 1)}


def train(args):
    train_set = DenseFrames("train", overfit=args.overfit)
    validation_set = (DenseFrames("train", overfit=True) if args.overfit else
                      DenseFrames("val", limit=args.validation_samples))
    run = RUN_ROOT / ("vae_overfit" if args.overfit else "vae_full")
    run.mkdir(parents=True, exist_ok=True)
    stage1.save_json(run / "config.json", {
        "representation": {"no_return": NO_RETURN, "valid_min": VALID_MIN,
                           "valid_threshold": VALID_THRESHOLD,
                           "max_range_m": MAX_RANGE_M},
        "input_channels": 1, "latent_channels": 4,
        "loss": "3*valid_l1 + no_return_l1 + .5*hit_hinge + .5*no_return_hinge + 1e-6*KL",
        "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.lr, "seed": args.seed,
        "train_samples": len(train_set), "validation_samples": len(validation_set),
        "source_hdf5_modified": False})
    stage1.seed_everything(args.seed)
    model = make_vae().to(stage1.DEVICE).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = stage1.infinite(DataLoader(train_set, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True))
    first_step = 1
    best = math.inf
    history = []
    if args.resume:
        first_step = stage1.resume_training(run / "latest.pt", model, optimizer) + 1
        if (run / "best_metrics.json").exists():
            best = json.loads((run / "best_metrics.json").read_text())["selection_score"]
        if (run / "history.json").exists():
            history = json.loads((run / "history.json").read_text())
    for step in range(first_step, args.steps + 1):
        model.train()
        target = next(loader).to(stage1.DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss, components = loss_fn(model, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite VAE loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            record = {"step": step, "loss": loss.item(), "grad_norm": grad.item(),
                      **{name: value.item() for name, value in components.items()}}
            if stage1.DEVICE.type == "cuda":
                record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            history.append(record)
            print(json.dumps(record), flush=True)
        if step == args.steps or step % args.eval_every == 0:
            metrics = evaluate(model, validation_set, args.eval_batch_size)
            score = (metrics["hit_range_mae_m"] + 0.2 * (1 - metrics["hit_recall"])
                     + 5.0 * metrics["false_empty_frames"] / len(validation_set))
            print(json.dumps({"step": step, "validation": metrics,
                              "selection_score": score}), flush=True)
            stage1.save_model(run / "latest.pt", model, optimizer, step, metrics)
            if score < best:
                best = score
                stage1.save_model(run / "best.pt", model, optimizer, step, metrics)
                stage1.save_json(run / "best_metrics.json", {"step": step,
                                 "selection_score": score, **metrics})
            stage1.save_json(run / "history.json", history)
    metrics = json.loads((run / "best_metrics.json").read_text())
    metrics["passed"] = (metrics["false_empty_frames"] == 0
                         and metrics["hit_range_mae_m"] <= 0.05
                         and metrics["hit_recall"] >= 0.995)
    stage1.save_json(run / "gate.json", metrics)
    print(json.dumps({"run": str(run), "gate": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="resume the full/overfit run from latest.pt")
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    train(args)


if __name__ == "__main__":
    main()
