"""Train a history-only LiDAR Video DiT with rectified flow.

The model consumes frozen-VAE latents at t-2, t-1 and t and predicts t+1.
It intentionally has no action, state, goal or privileged simulator input.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner import stage1
from lidar_wam.runner.world_direct_horizon import frame_metrics


FORMAT = "navrl-lidar-video-dit-v1"
INDEX_FORMAT = "navrl-lidar-history3-next1-index-v1"


def _decode(values):
    return [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in values]


def _atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _entries_from_archive(path: Path):
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    return archive, metadata["entries"], metadata


def build_video_index(dataset_root: Path, base_index_root: Path,
                      output_root: Path, split: str):
    """Materialize strict t-2,t-1,t,t+1 row chains shard by shard."""
    output = output_root / split
    output.mkdir(parents=True, exist_ok=True)
    archive, entries, base_metadata = _entries_from_archive(
        base_index_root / f"{split}_windows.npz")
    all_shards = np.asarray(archive["shard"])
    base_rows = np.asarray(archive["rows"])
    counts, rejected = [], 0
    for shard_id, entry in enumerate(entries):
        destination = output / f"rows_{shard_id:04d}.npy"
        candidates = base_rows[all_shards == shard_id, :2]
        if destination.exists():
            counts.append(int(np.load(destination, mmap_mode="r").shape[0]))
            continue
        path = dataset_root / entry["dataset"]
        with h5py.File(path, "r") as handle:
            frames = handle["frames"] if "frames" in handle else handle
            tokens = _decode(frames["token"][:])
            previous = _decode(frames["prev_token"][:])
        token_to_row = {token: row for row, token in enumerate(tokens)}
        chains = []
        for current, following in candidates.tolist():
            first_token = previous[current]
            first = token_to_row.get(first_token, -1) if first_token else -1
            second_token = previous[first] if first >= 0 else ""
            second = token_to_row.get(second_token, -1) if second_token else -1
            if first < 0 or second < 0:
                rejected += 1
                continue
            chains.append((second, first, current, following))
        value = np.asarray(chains, dtype=np.int64).reshape(-1, 4)
        np.save(destination, value)
        counts.append(len(value))
        print(json.dumps({"split": split, "shard": shard_id,
                          "windows": len(value)}), flush=True)
    metadata = {
        "format": INDEX_FORMAT,
        "split": split,
        "counts": counts,
        "total_windows": int(sum(counts)),
        "rejected_missing_history": int(rejected),
        "entries": entries,
        "dataset_manifest_sha256": base_metadata["dataset_manifest_sha256"],
        "alignment": ["t-2", "t-1", "t", "t+1"],
        "conditions": [],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class HistoryLatentDataset(Dataset):
    def __init__(self, dataset_root: Path, latent_root: Path,
                 video_index_root: Path, split: str, return_target_image=False):
        self.dataset_root = Path(dataset_root)
        self.latent_root = Path(latent_root)
        self.root = Path(video_index_root) / split
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("format") != INDEX_FORMAT:
            raise ValueError("incompatible history index")
        self.entries = self.metadata["entries"]
        self.counts = np.asarray(self.metadata["counts"], dtype=np.int64)
        self.ends = np.cumsum(self.counts)
        self._rows = {}
        self._latents = {}
        self._files = {}
        self.return_target_image = bool(return_target_image)
        latent_metadata = json.loads((self.latent_root / "metadata.json").read_text())
        self.vae_metadata = {
            key: latent_metadata.get(key) for key in (
                "scaling_factor", "vae_variant", "vae_step",
                "vae_checkpoint", "vae_sha256")}

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getstate__(self):
        value = dict(self.__dict__)
        value["_rows"] = {}
        value["_latents"] = {}
        value["_files"] = {}
        return value

    def _open(self, shard: int):
        if shard not in self._rows:
            self._rows[shard] = np.load(
                self.root / f"rows_{shard:04d}.npy", mmap_mode="r")
            filename = self.entries[shard].get("latent_cache")
            if not filename:
                raise ValueError(f"manifest entry {shard} has no latent_cache")
            self._latents[shard] = np.load(
                self.latent_root / filename, mmap_mode="r")
        return self._rows[shard], self._latents[shard]

    def _target_image(self, shard: int, row: int) -> np.ndarray:
        """Read raw GT only for the small validation subset.

        Training stays entirely on the frozen-VAE latent cache.  Chamfer is
        deliberately measured against the physical range image, rather than a
        VAE reconstruction of it.
        """
        if shard not in self._files:
            self._files[shard] = h5py.File(
                self.dataset_root / self.entries[shard]["dataset"], "r")
        handle = self._files[shard]
        frames = handle["frames"] if "frames" in handle else handle
        return np.asarray(frames["range_values"][row], dtype=np.float32)

    def __getitem__(self, index):
        shard = bisect.bisect_right(self.ends, int(index))
        start = 0 if shard == 0 else int(self.ends[shard - 1])
        rows, latents = self._open(shard)
        chain = rows[int(index) - start]
        history = np.asarray(latents[chain[:3]], dtype=np.float32)
        target = np.asarray(latents[int(chain[3])], dtype=np.float32)
        if self.return_target_image:
            return (torch.from_numpy(history), torch.from_numpy(target),
                    torch.from_numpy(self._target_image(shard, int(chain[3]))))
        return torch.from_numpy(history), torch.from_numpy(target)


def flow_batch(model, history, target):
    noise = torch.randn_like(target)
    sigma = torch.rand(len(target), device=target.device, dtype=target.dtype)
    noisy = (1 - sigma[:, None, None, None]) * target + (
        sigma[:, None, None, None] * noise)
    velocity_target = noise - target
    velocity = model(history, noisy, sigma)
    flow_mse = (velocity - velocity_target).square().mean()
    predicted_clean = noisy - sigma[:, None, None, None] * velocity
    x0_l1 = (predicted_clean - target).abs().mean()
    return flow_mse + 0.1 * x0_l1, flow_mse.detach(), x0_l1.detach()


@torch.no_grad()
def sample_next(model, history, steps: int, generator=None):
    value = torch.randn(
        len(history), 4, 27, 5, device=history.device,
        dtype=history.dtype, generator=generator)
    schedule = torch.linspace(1, 0, steps + 1, device=history.device,
                              dtype=history.dtype)
    for current, following in zip(schedule[:-1], schedule[1:]):
        timestep = torch.full((len(history),), current, device=history.device,
                              dtype=history.dtype)
        velocity = model(history, value, timestep)
        value = value + (following - current) * velocity
    return value


@torch.no_grad()
def validate(model, loader, vae, scale, device, precision, flow_steps):
    """Validate a fixed random subset, including decoded squared Chamfer.

    ``cd_paper_m2`` includes every frame using the project-wide finite empty
    cloud convention. ``nonempty_gt_cd_paper_m2`` excludes only frames whose
    *ground truth* has no return; false-empty predictions on non-empty scenes
    are still penalised, so this metric cannot be gamed by predicting nothing.
    """
    model.eval()
    totals = torch.zeros(7, device=device, dtype=torch.float64)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if precision == "bf16" else contextlib.nullcontext())
    generator = torch.Generator(device=device).manual_seed(12345)
    for history, target, target_image in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with amp:
            loss, flow_mse, x0_l1 = flow_batch(model, history, target)
            predicted = sample_next(model, history, flow_steps, generator)
        # Decode in fp32: VAE mask logits around the hit threshold are too
        # sensitive for an evaluation metric in bf16.
        prediction_image = vae.decode(predicted.float() / scale).sample.cpu().numpy()
        target_image = target_image.numpy()
        cd_sum = nonempty_cd_sum = 0.0
        nonempty_count = 0
        for prediction, truth in zip(prediction_image, target_image):
            row = frame_metrics(prediction, truth)
            cd_sum += float(row["cd_paper_m2"])
            if not row["gt_empty"]:
                nonempty_cd_sum += float(row["cd_paper_m2"])
                nonempty_count += 1
        count = len(history)
        totals += torch.tensor(
            (float(flow_mse) * count, float(x0_l1) * count,
             float((predicted-target).abs().mean()) * count, cd_sum,
             nonempty_cd_sum, float(count), float(nonempty_count)),
            device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(totals)
    count = max(float(totals[5]), 1.0)
    nonempty_count = max(float(totals[6]), 1.0)
    model.train()
    return {"flow_mse": float(totals[0] / count),
            "x0_l1": float(totals[1] / count),
            "sample_l1": float(totals[2] / count),
            "selection_score": float(totals[2] / count),
            "cd_paper_m2": float(totals[3] / count),
            "nonempty_gt_cd_paper_m2": float(totals[4] / nonempty_count),
            "samples": int(count), "nonempty_gt_samples": int(totals[6])}


def checkpoint_payload(model, optimizer, step, bests, args, dataset):
    source = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "format": FORMAT,
        "step": int(step), "best_metrics": {key: float(value)
                                                for key, value in bests.items()},
        "model": source.state_dict(), "optimizer": optimizer.state_dict(),
        "architecture": {"history_frames": 3, "prediction_frames": 1,
                         "latent_shape": [4, 27, 5], "width": args.width,
                         "depth": args.depth, "heads": args.heads,
                         "mlp_ratio": args.mlp_ratio,
                         "conditioning": "history_lidar_only"},
        "vae": dataset.vae_metadata,
        "config": vars(args),
    }


def train(args):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    if rank == 0:
        for split in ("train", "val"):
            path = args.video_index_root / split / "metadata.json"
            if not path.exists():
                build_video_index(args.dataset_root, args.base_index_root,
                                  args.video_index_root, split)
    if dist.is_initialized():
        dist.barrier()
    training = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "train")
    validation_full = HistoryLatentDataset(
        args.dataset_root, args.latent_root, args.video_index_root, "val",
        return_target_image=True)
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(
        len(validation_full), min(args.eval_samples, len(validation_full)),
        replace=False))
    validation = Subset(validation_full, selected.tolist())
    sampler = DistributedSampler(
        training, world_size, rank, shuffle=True, seed=args.seed,
        drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(
        training, batch_size=args.micro_batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0)
    val_sampler = DistributedSampler(
        validation, world_size, rank, shuffle=False, drop_last=False
    ) if world_size > 1 else None
    val_loader = DataLoader(
        validation, batch_size=args.eval_batch_size, sampler=val_sampler,
        num_workers=max(0, args.workers // 2), pin_memory=True)
    model = LiDARVideoDiT(
        args.width, args.depth, args.heads, args.mlp_ratio).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95))
    # This VAE is strictly evaluation-only. It never contributes gradients to
    # Video DiT training and decodes only the requested random validation rows.
    vae = stage1.load_circular_vae().to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    scale = float(training.vae_metadata["scaling_factor"])
    run_dir = args.out / args.run_name
    writer = None
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir / "tensorboard")
        config = vars(args).copy()
        config.update({"global_batch_size": args.micro_batch_size * world_size,
                       "train_windows": len(training),
                       "val_windows": len(validation_full),
                       "val_subset": len(validation),
                       "vae": training.vae_metadata})
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n")
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.precision == "bf16" else contextlib.nullcontext())
    iterator, epoch = iter(train_loader), 0
    bests = {"selection_score": float("inf"), "cd_paper_m2": float("inf"),
             "nonempty_gt_cd_paper_m2": float("inf")}
    started = time.time()
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        try:
            history, target = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            history, target = next(iterator)
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp:
            loss, flow_mse, x0_l1 = flow_batch(model, history, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if rank == 0 and step % args.log_every == 0:
            row = {"step": step, "loss": float(loss),
                   "flow_mse": float(flow_mse), "x0_l1": float(x0_l1),
                   "grad_norm": float(grad_norm),
                   "peak_allocated_gib": (
                       torch.cuda.max_memory_allocated(device) / 2**30),
                   "elapsed_min": (time.time()-started)/60}
            print(json.dumps(row), flush=True)
            for key, value in row.items():
                if key not in ("step",):
                    writer.add_scalar(f"train/{key}", value, step)
        # A validation is also a checkpoint boundary. Keep the two intervals
        # equal by default so all four checkpoint files are coherent.
        evaluate = step % args.eval_every == 0
        save_latest = step % args.checkpoint_every == 0
        metrics = None
        if evaluate:
            metrics = validate(model, val_loader, vae, scale, device,
                               args.precision, args.flow_steps)
            if rank == 0:
                print(json.dumps({"step": step, "validation": metrics}), flush=True)
                for key, value in metrics.items():
                    if key != "samples":
                        writer.add_scalar(f"validation/{key}", value, step)
                checkpoint = checkpoint_payload(
                    model, optimizer, step, bests, args, training)
                updates = {
                    "selection_score": ("best.pt", "best_metrics.json"),
                    "cd_paper_m2": ("best_chamfer_m2.pt",
                                     "best_chamfer_m2_metrics.json"),
                    "nonempty_gt_cd_paper_m2": (
                        "best_nonempty_chamfer_m2.pt",
                        "best_nonempty_chamfer_m2_metrics.json"),
                }
                for metric, (weight_name, json_name) in updates.items():
                    if metrics[metric] < bests[metric]:
                        bests[metric] = metrics[metric]
                        checkpoint = checkpoint_payload(
                            model, optimizer, step, bests, args, training)
                        _atomic_torch_save(checkpoint, run_dir / weight_name)
                        (run_dir / json_name).write_text(
                            json.dumps({"step": step, "optimized_metric": metric,
                                        "best_value": metrics[metric], **metrics},
                                       indent=2) + "\n")
        if rank == 0 and (save_latest or evaluate):
            _atomic_torch_save(
                checkpoint_payload(model, optimizer, step, bests, args, training),
                run_dir / "latest.pt")
            writer.flush()
    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-index")
    training = sub.add_parser("train")
    for item in (prepare, training):
        item.add_argument("--dataset-root", type=Path, required=True)
        item.add_argument("--latent-root", type=Path, required=True)
        item.add_argument("--base-index-root", type=Path, required=True)
        item.add_argument("--video-index-root", type=Path, required=True)
    prepare.add_argument("--splits", nargs="+", default=("train", "val"))
    training.add_argument("--out", type=Path, default=Path("outputs"))
    training.add_argument("--run-name", default="lidar_video_dit_history3_next1")
    training.add_argument("--steps", type=int, default=50000)
    training.add_argument("--micro-batch-size", type=int, default=16)
    training.add_argument("--eval-batch-size", type=int, default=16)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--width", type=int, default=512)
    training.add_argument("--depth", type=int, default=8)
    training.add_argument("--heads", type=int, default=8)
    training.add_argument("--mlp-ratio", type=float, default=4.0)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--weight-decay", type=float, default=1e-2)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    training.add_argument("--flow-steps", type=int, default=20)
    training.add_argument("--eval-samples", type=int, default=1024)
    training.add_argument("--eval-every", type=int, default=200)
    training.add_argument("--checkpoint-every", type=int, default=200)
    training.add_argument("--log-every", type=int, default=20)
    training.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare-index":
        for split in args.splits:
            result = build_video_index(
                args.dataset_root, args.base_index_root,
                args.video_index_root, split)
            print(json.dumps(result, indent=2), flush=True)
    else:
        train(args)


if __name__ == "__main__":
    main()
