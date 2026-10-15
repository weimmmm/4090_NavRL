"""Frozen, reproducible action validation; never train or replay these samples.

Run in a separate process/GPU against an immutable checkpoint. The existing
cache contains only full, valid ten-command windows from the original val
shards. Its fixed subset/order and original normalization are identical for
every evaluated checkpoint. The training learner never opens this cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from lidar_wam.runner.action_expert_closed_loop_sft import build_model
from lidar_wam.runner.online_supervision import student_feasible_target


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[4] / path


def load_fixed_samples(cache_path, payload, windows, seed):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata = cache["metadata"]
    if metadata.get("past_input") != "zero (same as deployment)":
        raise ValueError("Fixed cache's past-action contract differs from the student")
    if abs(float(metadata["scaling_factor"])-float(payload["vae"]["scaling_factor"])) > 1e-6:
        raise ValueError("Fixed validation VAE scale differs from checkpoint")
    vae_path = project_path(metadata["vae"])
    vae_sha = hashlib.sha256(vae_path.read_bytes()).hexdigest()
    if vae_sha != payload["vae"]["sha256"]:
        raise ValueError("Fixed cache's referenced VAE differs from checkpoint")
    reference = torch.load(project_path(metadata["checkpoint"]),
                           map_location="cpu", weights_only=False)
    if reference["condition_stats"] != payload["condition_stats"]:
        raise ValueError("Fixed cache and checkpoint normalization differ")
    del reference
    validation = cache["val"]  # NEVER draw from cache['train'].
    count = min(int(windows), len(validation["target"]))
    if count <= 0:
        raise ValueError("Fixed validation subset is empty")
    indices = torch.randperm(len(validation["target"]),
                             generator=torch.Generator().manual_seed(seed))[:count].sort().values
    data = {key: validation[key][indices].float()
            for key in ("latent", "goal", "proprio", "target")}
    if data["target"].shape != (count, 10, 3):
        raise ValueError("Expected complete ten-action validation labels")
    if not all(bool(torch.isfinite(value).all()) for value in data.values()):
        raise ValueError("Non-finite fixed validation inputs")
    identity = {"cache": str(cache_path), "windows": count, "seed": int(seed),
                "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                "indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                "vae_sha256": vae_sha, "samples_used_for_training": False}
    return data, identity


@torch.no_grad()
def fixed_metrics(model, data, device, batch_size):
    model.eval().requires_grad_(False)
    totals = torch.zeros(10, device=device, dtype=torch.float64)
    for start in range(0, len(data["target"]), batch_size):
        batch = {key: value[start:start+batch_size].to(device)
                 for key, value in data.items()}
        n = len(batch["target"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.predict_action(
                batch["latent"], batch["goal"], batch["proprio"],
                torch.zeros(n, 30, 3, device=device), torch.zeros(n, 30, device=device))
        prediction = prediction.float()
        target = batch["target"]
        values = []
        for reference in (target, student_feasible_target(target)):
            error = (prediction-reference).abs() * 2.
            values.extend((
                F.smooth_l1_loss(prediction[:, 0], reference[:, 0], beta=.25),
                F.smooth_l1_loss(prediction, reference, beta=.25),
                error[:, 0].mean(), error[:, 1:].mean(), error.mean()))
        values = torch.stack(values)
        totals += values.double() * n
    assert all(parameter.grad is None for parameter in model.parameters())
    keys = ("first_loss", "chunk_loss", "first_mae_mps", "later9_mae_mps", "chunk_mae_mps")
    return dict(zip([prefix+key for prefix in ("fixed_", "fixed_feasible_") for key in keys],
                    (totals / len(data["target"])).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.windows <= 0 or args.batch_size <= 0:
        parser.error("Validation windows and batch size must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda:0")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data, identity = load_fixed_samples(args.cache, payload, args.windows, args.seed)
    model = build_model(payload).to(device)
    metrics = fixed_metrics(model, data, device, args.batch_size)
    step = int(payload["online_sft"]["step"])
    result = {"step": step, "checkpoint": str(args.checkpoint),
              "fixed_samples": identity, "metrics": metrics,
              "offline_label_speed_over2_fraction": float(
                  (data["target"].norm(dim=-1)>1.0001).float().mean()),
              "feasible_metrics_reference": "same labels projected onto the student's unit-speed ball",
              "grad_enabled": False, "optimizer_created": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2)+"\n")
    os.replace(temporary, args.output)
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(args.tensorboard) as writer:
        for key, value in metrics.items():
            writer.add_scalar("validation/"+key, value, step)
        writer.add_scalar("validation/fixed_windows", len(data["target"]), step)
        writer.add_text("validation/fixed_subset", json.dumps(identity), step)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
