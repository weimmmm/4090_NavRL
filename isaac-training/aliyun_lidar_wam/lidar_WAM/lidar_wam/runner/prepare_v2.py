"""Validate the self-contained NavRL v2 dataset and cache frozen VAE latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from lidar_wam.coordinates import (
    goal_frame_causal_features,
    numpy_goal_frame_causal_features,
    numpy_normalized_to_world_velocity,
)
from lidar_wam.data_v2 import (
    SPLITS,
    build_all_indices,
    file_sha256,
    read_manifest,
    split_entries,
)
from lidar_wam.runner import stage1


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required") from exc
    return h5py


def _frames(handle):
    return handle["frames"] if "frames" in handle else handle


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def audit_actions(dataset_root: Path) -> dict:
    h5py = _h5py()
    maximum = total = 0.0
    count = rows = 0
    for entry in read_manifest(dataset_root)["entries"]:
        with h5py.File(dataset_root / entry["dataset"], "r") as handle:
            data = _frames(handle)
            for start in range(0, len(data["step_delta"]), 2048):
                stop = min(start + 2048, len(data["step_delta"]))
                normalized = np.asarray(
                    data["normalized_action_sequence"][start:stop], np.float32)
                world = np.asarray(data["world_action_sequence"][start:stop], np.float32)
                direction = np.asarray(data["target_dir_2d"][start:stop], np.float32)
                mask = np.asarray(data["action_mask"][start:stop], bool)
                reconstructed = numpy_normalized_to_world_velocity(normalized, direction)
                error = np.abs(reconstructed - world)[mask]
                if error.size:
                    maximum = max(maximum, float(error.max()))
                    total += float(error.sum())
                    count += int(error.size)
                rows += stop - start
    result = {"rows": rows, "components": count, "max_abs_error": maximum,
              "mean_abs_error": total / max(count, 1), "passed": maximum <= 1e-5}
    if not result["passed"]:
        raise RuntimeError(f"normalized/world action audit failed: {result}")
    return result


def audit_feature_parity(dataset_root: Path, samples: int = 4096) -> dict:
    h5py = _h5py()
    entries = read_manifest(dataset_root)["entries"]
    states, targets, directions = [], [], []
    remaining = samples
    for entry in entries:
        if remaining <= 0:
            break
        with h5py.File(dataset_root / entry["dataset"], "r") as handle:
            data = _frames(handle)
            count = min(remaining, len(data["drone_state"]))
            index = np.linspace(0, len(data["drone_state"])-1, count, dtype=np.int64)
            states.append(np.asarray(data["drone_state"][index], np.float32))
            targets.append(np.asarray(data["target_position"][index], np.float32))
            directions.append(np.asarray(data["target_dir_2d"][index], np.float32))
            remaining -= count
    state, target, direction = map(np.concatenate, (states, targets, directions))
    np_goal, np_proprio = numpy_goal_frame_causal_features(state, target, direction)
    torch_goal, torch_proprio = goal_frame_causal_features(
        torch.from_numpy(state), torch.from_numpy(target), torch.from_numpy(direction))
    maximum = max(float(np.abs(np_goal - torch_goal.numpy()).max()),
                  float(np.abs(np_proprio - torch_proprio.numpy()).max()))
    result = {"samples": len(state), "max_abs_error": maximum,
              "passed": maximum <= 1e-6}
    if not result["passed"]:
        raise RuntimeError(f"NumPy/Torch goal-frame parity failed: {result}")
    return result


def inspect_dataset(args):
    args.index_root.mkdir(parents=True, exist_ok=True)
    indices = build_all_indices(args.dataset_root, args.index_root, strict_expected=True)
    result = {
        "dataset_manifest": str(args.dataset_root / "manifest.json"),
        "dataset_manifest_sha256": file_sha256(args.dataset_root / "manifest.json"),
        "indices": indices,
        "action_audit": audit_actions(args.dataset_root),
        "feature_parity": audit_feature_parity(args.dataset_root),
    }
    _write_json(args.index_root / "inspection.json", result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


@torch.no_grad()
def vae_gate(dataset_root: Path, vae, batch_size: int, samples: int = 4096) -> dict:
    h5py = _h5py()
    entry = split_entries(dataset_root, "val")[0]
    valid_error = valid_count = tp = fp = fn = 0.0
    with h5py.File(dataset_root / entry["dataset"], "r") as handle:
        data = _frames(handle)
        indices = np.linspace(0, len(data["range_values"])-1,
                              min(samples, len(data["range_values"])), dtype=np.int64)
        # h5py requires monotonically increasing unique indices; linspace can
        # repeat when samples exceeds the row count.
        indices = np.unique(indices)
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start+batch_size]
            image = torch.from_numpy(
                np.asarray(data["range_values"][chosen], np.float32)).to(stage1.DEVICE)
            reconstruction = vae.decode(vae.encode(image).latent_dist.mode()).sample
            target = image[:, 1:2, :, :18] > 0
            prediction = reconstruction[:, 1:2, :, :18] > 1.5
            valid_error += float(
                ((reconstruction[:, :1, :, :18] - image[:, :1, :, :18]).abs()
                 * target).sum())
            valid_count += float(target.sum())
            tp += float((prediction & target).sum())
            fp += float((prediction & ~target).sum())
            fn += float((~prediction & target).sum())
    result = {
        "samples": int(len(indices)),
        "valid_range_mae": valid_error / max(valid_count, 1),
        "mask_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "mask_threshold": 1.5,
    }
    result["passed"] = result["valid_range_mae"] <= 0.10 and result["mask_f1"] >= 0.70
    return result


@torch.no_grad()
def cache_latents(args):
    if not (args.index_root / "inspection.json").is_file():
        inspect_dataset(args)
    inspection = json.loads((args.index_root / "inspection.json").read_text())
    if not inspection["action_audit"]["passed"] or not inspection["feature_parity"]["passed"]:
        raise RuntimeError("dataset inspection gates did not pass")

    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    gate = vae_gate(args.dataset_root, vae, args.batch_size)
    identity = stage1.circular_vae_identity()
    gate.update(identity)
    _write_json(args.output / "vae_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"frozen Circular VAE gate failed: {gate}")

    h5py = _h5py()
    args.output.mkdir(parents=True, exist_ok=True)
    scaling = float(vae.config.scaling_factor)
    cached = []
    for entry in read_manifest(args.dataset_root)["entries"]:
        seed = int(entry["seed"])
        source = args.dataset_root / entry["dataset"]
        destination = args.output / f"seed_{seed:04d}.npy"
        with h5py.File(source, "r") as handle:
            data = _frames(handle)
            count = len(data["range_values"])
            if args.resume and destination.is_file():
                existing = np.load(destination, mmap_mode="r")
                if tuple(existing.shape) == (count, 4, 27, 5) and existing.dtype == np.float16:
                    cached.append({"seed": seed, "frames": count,
                                   "file": destination.name, "resumed": True})
                    continue
            temporary = destination.with_suffix(".tmp.npy")
            output = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float16, shape=(count, 4, 27, 5))
            for start in range(0, count, args.batch_size):
                stop = min(start + args.batch_size, count)
                image = torch.from_numpy(np.asarray(
                    data["range_values"][start:stop], np.float32)).to(stage1.DEVICE)
                latent = vae.encode(image).latent_dist.mode() * scaling
                output[start:stop] = latent.cpu().numpy().astype(np.float16)
                if start == 0 or stop == count or stop % (args.batch_size * 100) == 0:
                    print(json.dumps({"seed": seed, "cached": stop, "total": count}),
                          flush=True)
            output.flush()
            del output
            os.replace(temporary, destination)
        cached.append({"seed": seed, "frames": count, "file": destination.name,
                       "sha256": file_sha256(destination), "resumed": False})

    metadata = {
        "format": "navrl-wam-latent-cache-v2",
        "dataset_manifest_sha256": file_sha256(args.dataset_root / "manifest.json"),
        "scaling_factor": scaling,
        "latent_definition": "Circular VAE posterior mode * scaling_factor",
        "storage_dtype": "float16", "shape": [4, 27, 5],
        **identity, "vae_gate": gate, "shards": cached,
    }
    _write_json(args.output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "cache-latents"):
        child = commands.add_parser(name)
        child.add_argument("--dataset-root", type=Path, required=True)
        child.add_argument("--index-root", type=Path, default=None)
        if name == "cache-latents":
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--batch-size", type=int, default=128)
            child.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    default_index = stage1.OUT / "wam_static_seed00_09" / "index"
    args.index_root = (default_index if args.index_root is None
                       else args.index_root.expanduser().resolve())
    if args.command == "inspect":
        inspect_dataset(args)
    else:
        args.output = args.output.expanduser().resolve()
        cache_latents(args)


if __name__ == "__main__":
    main()
