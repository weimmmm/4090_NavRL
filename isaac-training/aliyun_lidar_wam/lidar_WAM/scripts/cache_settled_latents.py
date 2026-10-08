"""Cache frozen Circular-VAE latents for the settled-goal shard dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from lidar_wam.runner import stage1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    latent_root = args.latent_root.expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["entries"]
    if not entries:
        raise RuntimeError("manifest has no entries")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    latent_root.mkdir(parents=True, exist_ok=True)
    device = stage1.DEVICE
    vae = stage1.load_circular_vae().to(device).eval()
    vae.requires_grad_(False)
    scaling = float(vae.config.scaling_factor)
    shards = []
    for number, entry in enumerate(entries, 1):
        source = dataset_root / entry["dataset"]
        destination = latent_root / entry["latent_cache"]
        with h5py.File(source, "r") as handle:
            frames = handle["frames"] if "frames" in handle else handle
            count = len(frames["range_values"])
            expected = (count, 4, 27, 5)
            if args.resume and destination.is_file():
                try:
                    existing = np.load(destination, mmap_mode="r")
                    valid_existing = (tuple(existing.shape) == expected
                                      and existing.dtype == np.float16)
                except (OSError, ValueError):
                    valid_existing = False
                if valid_existing:
                    print(json.dumps({"entry": number, "total": len(entries),
                                      "resumed": True, "frames": count}),
                          flush=True)
                    shards.append({"seed": int(entry["seed"]),
                                   "frames": count, "file": destination.name,
                                   "resumed": True})
                    continue
                destination.unlink(missing_ok=True)
            # A process-specific temporary avoids races with a stale cache
            # worker after an interrupted remote ``docker exec``.
            temporary = destination.parent / (
                f"{destination.name}.partial-{os.getpid()}.npy")
            temporary.unlink(missing_ok=True)
            output = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float16, shape=expected)
            for start in range(0, count, args.batch_size):
                stop = min(start + args.batch_size, count)
                image = torch.from_numpy(np.asarray(
                    frames["range_values"][start:stop], dtype=np.float32)).to(device)
                latent = vae.encode(image).latent_dist.mode() * scaling
                output[start:stop] = latent.detach().cpu().numpy().astype(np.float16)
            output.flush()
            del output
            if not temporary.is_file():
                candidates = sorted(destination.parent.glob(
                    f"{destination.name}.partial-*.npy"))
                raise FileNotFoundError(
                    f"latent temporary disappeared: {temporary}; "
                    f"candidates={candidates}")
            os.replace(temporary, destination)
        digest = sha256(destination)
        shards.append({"seed": int(entry["seed"]), "frames": count,
                       "file": destination.name, "sha256": digest,
                       "resumed": False})
        print(json.dumps({"entry": number, "total": len(entries),
                          "frames": count, "file": destination.name}), flush=True)

    identity = stage1.circular_vae_identity()
    metadata = {
        "format": "navrl-wam-latent-cache-v2",
        "dataset_manifest_sha256": sha256(manifest_path),
        "scaling_factor": scaling,
        "latent_definition": "Circular VAE posterior mode * scaling_factor",
        "storage_dtype": "float16",
        "shape": [4, 27, 5],
        **identity,
        "shards": shards,
    }
    write_json(latent_root / "metadata.json", metadata)
    print(json.dumps({"latent_root": str(latent_root),
                      "entries": len(shards),
                      "frames": sum(row["frames"] for row in shards),
                      "vae_step": identity.get("vae_step")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
