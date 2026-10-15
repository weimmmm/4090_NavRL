"""Validate Future DiT checkpoints on a separate GPU.

The training ranks only save atomic ``latest.pt`` checkpoints when launched
with ``--external-validation``.  This worker watches that file, chooses a new
random validation subset for every checkpoint, evaluates exactly
``--eval-samples`` windows, and appends the metrics to the run's TensorBoard
directory without using any training GPU.
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
    HistoryLatentDataset, _atomic_torch_save, configure_prediction_from_payload,
    validate)


def _load_model(checkpoint: Path, device: torch.device, payload=None):
    if payload is None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = dict(payload.get("architecture", {}))
    model = LiDARVideoDiT(
        width=int(architecture.get("width", 512)),
        depth=int(architecture.get("depth", 8)),
        heads=int(architecture.get("heads", 8)),
        mlp_ratio=float(architecture.get("mlp_ratio", 4.0)),
        prediction_frames=int(architecture.get("prediction_frames", 1)),
    ).to(device)
    model.load_state_dict(payload.get("model", payload), strict=True)
    configure_prediction_from_payload(model, payload)
    return model, payload


def _select_indices(length: int, count: int, seed: int):
    count = min(max(int(count), 1), int(length))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(length), count, replace=False)).tolist()


def _write_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=float) + "\n")
    temporary.replace(path)


def evaluate_checkpoint(args, dataset, vae, model, checkpoint: Path,
                        device: torch.device, evaluation_number: int,
                        writer, model_payload=None):
    if model_payload is None:
        model_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(model_payload.get("model", model_payload), strict=True)
    configure_prediction_from_payload(model, model_payload)
    model.eval()
    indices = _select_indices(
        len(dataset), args.eval_samples, args.seed + evaluation_number)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.eval_batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    metrics = validate(
        model, loader, vae, float(dataset.vae_metadata["scaling_factor"]),
        device, args.precision, args.flow_steps, args.flow_ensemble_size,
        motion_stats=None, flow_sampling_mode=args.flow_sampling_mode)
    step = int(model_payload.get("step", -1))
    metrics = {
        key: (int(value) if key in ("samples", "nonempty_gt_samples",
                                    "depth_valid_samples") else float(value))
        for key, value in metrics.items()}
    metrics.update({"step": step, "evaluation_number": int(evaluation_number),
                    "random_seed": int(args.seed + evaluation_number),
                    "sampled_windows": len(indices)})
    run_dir = Path(args.run_dir)
    _write_json(run_dir / f"external_validation_step{step:08d}.json", metrics)
    _write_json(run_dir / "external_validation_latest.json", metrics)
    # latest.pt can advance while validation runs. Save the payload actually
    # evaluated, so the best checkpoint always matches its reported metrics.
    best_path = run_dir / "best_metrics.json"
    previous = json.loads(best_path.read_text()) if best_path.exists() else {}
    selection_metric = model_payload.get("config", {}).get(
        "selection_metric", "generation_loss")
    if metrics[selection_metric] < previous.get("best_value", float("inf")):
        model_payload["metrics"] = metrics
        model_payload.setdefault("best_metrics", {})["selection_score"] = (
            metrics[selection_metric])
        _atomic_torch_save(model_payload, run_dir / "best.pt")
        _write_json(best_path, {
            **metrics, "optimized_metric": selection_metric,
            "best_value": metrics[selection_metric]})
    for key, value in metrics.items():
        if key not in ("step", "evaluation_number", "random_seed"):
            writer.add_scalar(f"validation_20480/{key}", value, step)
    writer.flush()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--video-index-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-samples", type=int, default=20480)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument("--flow-ensemble-size", type=int, default=1)
    parser.add_argument("--flow-sampling-mode", choices=("independent", "zero", "antithetic"), default="independent")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--stop-step", type=int, default=50000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("External Future DiT validation requires CUDA")
    device = torch.device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run_dir / "latest.pt"
    dataset = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=True)
    vae = stage1.load_circular_vae().to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.run_dir / "tensorboard")
        model = None
        last_step = -1
        evaluation_number = 0
        previous_validation = args.run_dir / "external_validation_latest.json"
        if previous_validation.exists():
            previous = json.loads(previous_validation.read_text())
            last_step = int(previous.get("step", -1))
            evaluation_number = int(previous.get("evaluation_number", -1)) + 1
        while True:
            if checkpoint.exists():
                try:
                    payload = torch.load(
                        checkpoint, map_location="cpu", weights_only=False)
                    step = int(payload.get("step", -1))
                except Exception:
                    step = -1
                if step > last_step:
                    if model is None:
                        model, _ = _load_model(checkpoint, device, payload)
                    metrics = evaluate_checkpoint(
                        args, dataset, vae, model, checkpoint, device,
                        evaluation_number, writer, payload)
                    print(json.dumps({"external_validation": metrics}),
                          flush=True)
                    last_step = step
                    evaluation_number += 1
                    if step >= int(args.stop_step):
                        break
            time.sleep(float(args.poll_seconds))
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
