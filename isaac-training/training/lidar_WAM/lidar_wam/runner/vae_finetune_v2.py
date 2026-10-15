"""Fine-tune the circular LiDAR VAE on a manifest-v2 shard dataset.

The source VAE was trained on normalized 10 m range images.  The L20 dataset
uses the same two-channel ``[2,108,20]`` representation, but maps 0--5 m to
``[-1,1]``.  This runner keeps complete trajectories on one side of a stable
train/validation split, starts from the source VAE, and reports metrics in the
correct 5 m physical scale.  It never overwrites the source checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from lidar_wam.data_v2 import file_sha256, read_manifest
from lidar_wam.runner import stage1


RANGE_M = 5.0
RANGE_SCALE_M = RANGE_M / 2.0
MASK_THRESHOLDS = np.linspace(-1.0, 3.0, 17, dtype=np.float32)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def stable_validation_scene(dataset: str, scene_id: int, fraction: float) -> bool:
    token = f"{dataset}:{int(scene_id)}".encode()
    bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return bucket / float(2**64) < fraction


def frames(handle):
    return handle["frames"] if "frames" in handle else handle


class ShardFrameStore:
    """Chunk-friendly train batches and a fixed trajectory-disjoint holdout."""

    def __init__(self, root: Path, validation_fraction: float,
                 validation_frames_per_condition: int, seed: int):
        self.root = Path(root)
        self.entries = [row for row in read_manifest(root)["entries"]
                        if row["split"] == "train"]
        if not self.entries:
            raise ValueError("manifest contains no train entries")
        self.train_rows: list[np.ndarray] = []
        self.val_rows: list[np.ndarray] = []
        self.handles: dict[int, h5py.File] = {}
        scene_totals = {"train": 0, "validation": 0}
        frame_totals = {"train": 0, "validation": 0}

        for entry in self.entries:
            with h5py.File(self.root / entry["dataset"], "r") as handle:
                data = frames(handle)
                scene_id = np.asarray(data["scene_id"][:], dtype=np.int32)
            unique = np.unique(scene_id)
            held_out = np.asarray([
                stable_validation_scene(entry["dataset"], int(value),
                                        validation_fraction)
                for value in unique
            ], dtype=bool)
            if not held_out.any():
                # Tiny shards still need at least one whole held-out trajectory.
                held_out[int(hashlib.sha256(entry["dataset"].encode()).hexdigest()[:8], 16)
                         % len(held_out)] = True
            if held_out.all():
                held_out[0] = False
            validation_mask = held_out[scene_id]
            train = np.flatnonzero(~validation_mask).astype(np.int32)
            validation = np.flatnonzero(validation_mask).astype(np.int32)
            self.train_rows.append(train)
            self.val_rows.append(validation)
            scene_totals["validation"] += int(held_out.sum())
            scene_totals["train"] += int(len(held_out) - held_out.sum())
            frame_totals["validation"] += int(len(validation))
            frame_totals["train"] += int(len(train))

        self.train_counts = np.asarray([len(rows) for rows in self.train_rows],
                                       dtype=np.float64)
        self.train_probabilities = self.train_counts / self.train_counts.sum()
        self.validation_selection = self._validation_selection(
            validation_frames_per_condition, seed)
        self.summary = {
            "split_unit": "complete trajectory",
            "split_hash": "sha256(dataset-relative-path:scene_id)",
            "validation_fraction": validation_fraction,
            "scenes": scene_totals,
            "frames": frame_totals,
            "evaluated_validation_frames": len(self.validation_selection),
        }

    def _validation_selection(self, per_condition: int, seed: int):
        rng = np.random.default_rng(seed)
        selected: list[tuple[int, int]] = []
        conditions = sorted({int(row["static_obstacles"]) for row in self.entries})
        for condition in conditions:
            shards = [
                shard for shard, entry in enumerate(self.entries)
                if int(entry["static_obstacles"]) == condition
                and len(self.val_rows[shard])
            ]
            if not shards:
                raise ValueError(f"condition {condition} has no validation frames")
            available = np.asarray(
                [len(self.val_rows[shard]) for shard in shards], np.int64)
            target = min(int(per_condition), int(available.sum()))
            exact = target * available / available.sum()
            allocation = np.floor(exact).astype(np.int64)
            allocation = np.minimum(allocation, available)
            # Resolve rounding while keeping proportional coverage across all
            # shards.  Each selected region is contiguous in the compact list
            # of held-out rows, avoiding thousands of unrelated HDF5 chunks.
            order = np.argsort(-(exact - allocation))
            remaining = target - int(allocation.sum())
            for index in order:
                if remaining <= 0:
                    break
                room = int(available[index] - allocation[index])
                if room:
                    allocation[index] += 1
                    remaining -= 1
            for shard, count in zip(shards, allocation):
                if not count:
                    continue
                rows = self.val_rows[shard]
                begin = int(rng.integers(0, len(rows) - int(count) + 1))
                selected.extend(
                    (shard, int(row))
                    for row in rows[begin:begin + int(count)])
        return selected

    def _data(self, shard: int):
        if shard not in self.handles:
            self.handles[shard] = h5py.File(
                self.root / self.entries[shard]["dataset"], "r")
        return frames(self.handles[shard])

    def training_batch(self, batch_size: int, rng: np.random.Generator):
        shard = int(rng.choice(len(self.entries), p=self.train_probabilities))
        rows = self.train_rows[shard]
        if len(rows) < batch_size:
            choice = np.sort(rng.choice(rows, batch_size, replace=True))
        else:
            begin = int(rng.integers(0, len(rows) - batch_size + 1))
            choice = rows[begin:begin + batch_size]
        # Rows are monotonic and nearly contiguous, preserving HDF5 chunk I/O.
        return np.asarray(self._data(shard)["range_values"][choice], np.float32)

    def validation_batches(self, batch_size: int):
        # Group by shard so each HDF5 fancy-index operation stays sorted.
        by_shard: dict[int, list[int]] = {}
        for shard, row in self.validation_selection:
            by_shard.setdefault(shard, []).append(row)
        for shard in sorted(by_shard):
            rows = np.asarray(sorted(by_shard[shard]), dtype=np.int64)
            for begin in range(0, len(rows), batch_size):
                choice = rows[begin:begin + batch_size]
                image = np.asarray(
                    self._data(shard)["range_values"][choice], np.float32)
                condition = int(self.entries[shard]["static_obstacles"])
                yield condition, image

    def scaling_batches(self, batch_size: int, samples: int,
                        rng: np.random.Generator):
        remaining = int(samples)
        while remaining > 0:
            count = min(batch_size, remaining)
            yield self.training_batch(count, rng)
            remaining -= count

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def reconstruction_loss(inputs, outputs, posterior):
    inputs = inputs[:, :, :, :18]
    outputs = outputs[:, :, :, :18]
    valid = (inputs[:, 1:2] > 0).to(outputs.dtype)
    invalid = 1.0 - valid
    range_error = torch.abs(outputs[:, :1] - inputs[:, :1])
    mask_error = F.binary_cross_entropy_with_logits(
        outputs[:, 1:2], valid, reduction="none")
    valid_range = masked_mean(range_error, valid)
    invalid_range = masked_mean(range_error, invalid)
    balanced_mask = 0.5 * (
        masked_mean(mask_error, valid) + masked_mean(mask_error, invalid))
    kl = posterior.kl().mean()
    total = (5.0 * valid_range + 2.0 * balanced_mask
             + 0.1 * invalid_range + 1e-6 * kl)
    return total, {
        "valid_range_l1": valid_range.detach(),
        "balanced_mask_bce": balanced_mask.detach(),
        "invalid_range_l1": invalid_range.detach(),
        "kl": kl.detach(),
    }


def empty_metrics():
    return {"range_sum": 0.0, "valid": 0.0,
            "counts": np.zeros((len(MASK_THRESHOLDS), 3), np.float64),
            "frames": 0}


def update_metrics(result, inputs, outputs):
    inputs = inputs[:, :, :, :18].float()
    outputs = outputs[:, :, :, :18].float()
    valid = inputs[:, 1] > 0
    result["range_sum"] += float(
        (torch.abs(outputs[:, 0] - inputs[:, 0]) * RANGE_SCALE_M)[valid].sum())
    result["valid"] += float(valid.sum())
    result["frames"] += int(len(inputs))
    for index, threshold in enumerate(MASK_THRESHOLDS):
        predicted = outputs[:, 1] > float(threshold)
        result["counts"][index] += (
            float((valid & predicted).sum()),
            float((~valid & predicted).sum()),
            float((valid & ~predicted).sum()),
        )


def finalize_metrics(value):
    counts = value["counts"]
    precision = counts[:, 0] / np.maximum(counts[:, 0] + counts[:, 1], 1)
    recall = counts[:, 0] / np.maximum(counts[:, 0] + counts[:, 2], 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    fixed = int(np.argmin(np.abs(MASK_THRESHOLDS - 1.5)))
    return {
        "frames": int(value["frames"]),
        "range_mae_m": value["range_sum"] / max(value["valid"], 1),
        "mask_f1": float(f1[best]),
        "mask_threshold": float(MASK_THRESHOLDS[best]),
        "mask_f1_at_1_5": float(f1[fixed]),
    }


@torch.inference_mode()
def evaluate(model, store: ShardFrameStore, batch_size: int, mixed: bool):
    model.eval()
    combined = empty_metrics()
    per_condition: dict[int, dict] = {}
    losses = []
    for condition, array in store.validation_batches(batch_size):
        image = torch.from_numpy(array).to(stage1.DEVICE)
        with autocast_context(mixed):
            posterior = model.encode(image).latent_dist
            reconstruction = model.decode(posterior.mode()).sample
            loss, _ = reconstruction_loss(image, reconstruction, posterior)
        losses.append(float(loss))
        update_metrics(combined, image, reconstruction)
        update_metrics(per_condition.setdefault(condition, empty_metrics()),
                       image, reconstruction)
    result = finalize_metrics(combined)
    result["loss"] = float(np.mean(losses))
    result["conditions"] = {
        str(key): finalize_metrics(value)
        for key, value in sorted(per_condition.items())
    }
    return result


@torch.inference_mode()
def recompute_scaling(model, store, batch_size, samples, seed, mixed):
    model.eval()
    total = square = count = 0.0
    rng = np.random.default_rng(seed)
    for array in store.scaling_batches(batch_size, samples, rng):
        image = torch.from_numpy(array).to(stage1.DEVICE)
        with autocast_context(mixed):
            latent = model.encode(image).latent_dist.sample().float()
        total += float(latent.sum())
        square += float(latent.square().sum())
        count += latent.numel()
    mean = total / count
    variance = max(square / count - mean * mean, 1e-12)
    return variance ** -0.5


def save_training_checkpoint(path, model, optimizer, step, best_score,
                             best_step, stale_evaluations):
    temporary = path.with_suffix(".partial")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": int(step), "best_score": float(best_score),
        "best_step": int(best_step),
        "stale_evaluations": int(stale_evaluations),
    }, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-frames-per-condition", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--early-stop-evaluations", type=int, default=6)
    parser.add_argument("--scaling-samples", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")

    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    mixed = not args.fp32

    store = ShardFrameStore(
        args.dataset_root, args.validation_fraction,
        args.validation_frames_per_condition, args.seed)
    source_identity = stage1.circular_vae_identity()
    config = {
        "format": "navrl-circular-vae-finetune-v2",
        "dataset_root": str(args.dataset_root),
        "dataset_manifest_sha256": file_sha256(args.dataset_root / "manifest.json"),
        "source_vae": source_identity,
        "range_m": RANGE_M,
        "normalization": "distance / 2.5 - 1; hit mask +/-1; last 2 columns padding",
        "input_shape": [2, 108, 20],
        "loss": "5*valid_l1 + 2*balanced_mask_bce + .1*invalid_l1 + 1e-6*kl",
        "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "precision": "fp32" if args.fp32 else "bf16",
        "validation": store.summary, "seed": args.seed,
    }
    atomic_json(args.output / "config.json", config)

    model = stage1.load_circular_vae().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=1e-6)
    start_step = 1
    best_score = math.inf
    best_step = 0
    stale = 0
    checkpoint = args.output / "checkpoint_last.pt"
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        best_score = float(state["best_score"])
        best_step = int(state["best_step"])
        stale = int(state.get("stale_evaluations", 0))

    history_path = args.output / "history.jsonl"
    baseline = evaluate(model, store, args.batch_size, mixed)
    baseline["step"] = start_step - 1
    baseline["selection_score"] = (
        baseline["range_mae_m"] + 1 - baseline["mask_f1_at_1_5"])
    print("VAE_VALIDATION=" + json.dumps(baseline), flush=True)
    if baseline["selection_score"] < best_score:
        best_score = baseline["selection_score"]
        best_step = start_step - 1
        model.save_pretrained(args.output / "best", safe_serialization=True)
        atomic_json(args.output / "best_metrics.json", baseline)

    rng = np.random.default_rng(args.seed + start_step)
    started = time.monotonic()
    stopped_early = False
    for step in range(start_step, args.steps + 1):
        model.train()
        image = torch.from_numpy(
            store.training_batch(args.batch_size, rng)).to(stage1.DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(mixed):
            posterior = model.encode(image).latent_dist
            reconstruction = model.decode(posterior.sample()).sample
            loss, components = reconstruction_loss(image, reconstruction, posterior)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            record = {
                "type": "train", "step": step, "loss": float(loss),
                **{key: float(value) for key, value in components.items()},
                "grad_norm": float(grad_norm),
                "elapsed_s": time.monotonic() - started,
            }
            with history_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print("VAE_TRAIN=" + json.dumps(record), flush=True)

        evaluate_now = step == args.steps or step % args.eval_every == 0
        if evaluate_now:
            metrics = evaluate(model, store, args.batch_size, mixed)
            metrics["step"] = step
            metrics["selection_score"] = (
                metrics["range_mae_m"] + 1 - metrics["mask_f1_at_1_5"])
            with history_path.open("a") as stream:
                stream.write(json.dumps({"type": "validation", **metrics}) + "\n")
            print("VAE_VALIDATION=" + json.dumps(metrics), flush=True)
            if metrics["selection_score"] < best_score:
                best_score = metrics["selection_score"]
                best_step = step
                stale = 0
                model.save_pretrained(args.output / "best", safe_serialization=True)
                atomic_json(args.output / "best_metrics.json", metrics)
            else:
                stale += 1

        if step == args.steps or step % args.save_every == 0:
            save_training_checkpoint(checkpoint, model, optimizer, step,
                                     best_score, best_step, stale)
        if evaluate_now and args.early_stop_evaluations > 0 \
                and stale >= args.early_stop_evaluations:
            stopped_early = True
            break

    best_weights = args.output / "best" / "diffusion_pytorch_model.safetensors"
    model.load_state_dict(load_file(str(best_weights)), strict=True)
    scaling = recompute_scaling(
        model, store, args.batch_size, args.scaling_samples,
        args.seed + 991, mixed)
    model.register_to_config(scaling_factor=scaling)
    model.save_pretrained(args.output / "best", safe_serialization=True)
    result = {
        "best_step": best_step, "best_score": best_score,
        "stopped_early": stopped_early,
        "scaling_factor": scaling,
        "best_metrics": json.loads((args.output / "best_metrics.json").read_text()),
        "source_vae_sha256": source_identity["vae_sha256"],
        "best_vae_sha256": file_sha256(best_weights),
    }
    atomic_json(args.output / "result.json", result)
    print("VAE_RESULT=" + json.dumps(result), flush=True)
    store.close()


if __name__ == "__main__":
    main()
