"""Train the true frozen-VAE Future ``LiDARVideoDiT`` on SUPER HDF5.

By default the target is the immediate next LiDAR latent ``t+1``: one LiDAR
interval, or 0.16 s at the fixed 0.016 s physics step.  Unlike
``train_super_future.py`` this module does not import or construct a UNet or a
DDPM scheduler: it uses the rectified-flow objective implemented by
``LiDARVideoDiT``. Validation uses an independent root or whole shards held
out from the validated 5 m cache; every random subset is recorded.

No simulator, SUPER, or Lee process is started by this module.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from lidar_wam.models.lidar_video_dit import LiDARVideoDiT
from lidar_wam.runner import stage1
from lidar_wam.runner.super_dual_dit_data import fixed_eval_indices
from lidar_wam.runner.super_future_data import _range_image


FORMAT_PREFIX = "navrl-super-lidar-video-dit"


class WorldOnlyDataset(Dataset):
    """Three causal LiDAR frames and a configurable future target.

    ``target_offset=1`` gives the required next-frame mapping
    ``[t-2,t-1,t] -> t+1``.  The old direct-t+3 experiment remains
    reproducible by explicitly passing ``target_offset=3``.
    """

    def __init__(self, root: Path, max_windows: int | None = None,
                 random_seed: int = 42, target_offset: int = 1):
        self.root = Path(root).expanduser().resolve()
        self.target_offset = int(target_offset)
        if self.target_offset < 1:
            raise ValueError("target_offset must be at least one LiDAR frame")
        self.scene_paths = sorted(self.root.glob("scene_*/trajectories.h5"))
        if not self.scene_paths:
            raise FileNotFoundError(
                f"no scene_*/trajectories.h5 under {self.root}")
        self._handles: dict[int, h5py.File] = {}
        self._windows: list[tuple[int, int, int]] = []
        for scene_index, path in enumerate(self.scene_paths):
            with h5py.File(path, "r") as handle:
                lidar_steps = np.asarray(handle["lidar/step"][:], dtype=np.int32)
                lidar_episode = np.asarray(
                    handle["lidar/episode_index"][:], dtype=np.int32)
                offsets = np.asarray(
                    handle["episodes/lidar_offset"][:], dtype=np.int64)
                lengths = np.asarray(
                    handle["episodes/lidar_length"][:], dtype=np.int32)
                for episode_index, (offset, length) in enumerate(
                    zip(offsets.tolist(), lengths.tolist())
                ):
                    for current_local in range(2, int(length) - self.target_offset):
                        rows = [
                            int(offset) + current_local - 2,
                            int(offset) + current_local - 1,
                            int(offset) + current_local,
                            int(offset) + current_local + self.target_offset,
                        ]
                        if not np.all(lidar_episode[rows] == episode_index):
                            raise ValueError(f"non-contiguous episode in {path}")
                        expected = np.asarray(
                            [10, 10, 10 * self.target_offset], dtype=np.int32)
                        if np.array_equal(np.diff(lidar_steps[rows]), expected):
                            self._windows.append(
                                (scene_index, episode_index, current_local))
        if not self._windows:
            raise RuntimeError(
                f"no valid history3/t+{self.target_offset} windows under {self.root}")
        if max_windows is not None and len(self._windows) > int(max_windows):
            generator = np.random.default_rng(int(random_seed))
            chosen = np.sort(generator.choice(
                len(self._windows), int(max_windows), replace=False))
            self._windows = [self._windows[int(index)] for index in chosen]

    def __len__(self):
        return len(self._windows)

    def _handle(self, scene_index: int):
        handle = self._handles.get(scene_index)
        if handle is None:
            handle = h5py.File(self.scene_paths[scene_index], "r")
            self._handles[scene_index] = handle
        return handle

    def __getstate__(self):
        value = dict(self.__dict__)
        value["_handles"] = {}
        return value

    def __getitem__(self, index):
        scene_index, episode_index, current_local = self._windows[int(index)]
        handle = self._handle(scene_index)
        offset = int(handle["episodes/lidar_offset"][episode_index])
        history_rows = [
            offset + current_local - 2,
            offset + current_local - 1,
            offset + current_local,
        ]
        target_row = offset + current_local + self.target_offset
        history = np.stack([
            _range_image(handle["lidar/range_m"][row]) for row in history_rows
        ])
        target = _range_image(handle["lidar/range_m"][target_row])
        return torch.from_numpy(history), torch.from_numpy(target)

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--latent-cache", type=Path)
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--loss-space", choices=('latent', 'pixel'), default='latent')
    parser.add_argument("--decoder-chunk-size", type=int, default=8)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--target-offset", type=int, default=1,
        help="LiDAR-frame target offset; 1 means immediate next frame (0.16 s)")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--eval-windows", type=int, default=32768,
        help="fresh random validation windows per evaluation (without replacement)")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--patience", type=int, default=8,
                        help="validation checks without improvement before early stop")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def encode_images(vae, images: torch.Tensor, scale: float, device,
                  precision: str) -> torch.Tensor:
    """Encode [B,T,2,108,20] images with the frozen circular VAE."""
    batch, frames = images.shape[:2]
    if images.ndim == 5 and tuple(images.shape[2:]) == (4, 27, 5):
        # Cache stores frozen VAE posterior modes with scale already applied.
        return images.to(device, non_blocking=True).float()
    flat = images.to(device, non_blocking=True).reshape(
        batch * frames, 2, 108, 20).float()
    with torch.no_grad():
        # VAE weights stay fp32; autocast only reduces activation memory on GPU.
        with autocast_context(device, precision):
            latent = vae.encode(flat).latent_dist.mode() * float(scale)
    return latent.reshape(batch, frames, 4, 27, 5)


def flow_batch(model, vae, history_image, target_image, scale, device,
               precision, loss_space='latent', target_pixels=None,
               decoder_chunk_size=8):
    history = encode_images(vae, history_image, scale, device, precision)
    target = encode_images(vae, target_image[:, None], scale, device, precision)[:, 0]
    noise = torch.randn_like(target)
    sigma = torch.rand(len(target), device=device, dtype=target.dtype)
    view = sigma[:, None, None, None]
    noisy = (1.0 - view) * target + view * noise
    with autocast_context(device, precision):
        velocity = model(history, noisy, sigma)
        predicted = noisy.float() - view.float() * velocity.float()
        if loss_space == 'pixel':
            if target_pixels is None:
                raise ValueError('pixel loss requires the true recorded next-frame pixels')
            from torch.utils.checkpoint import checkpoint
            def decode(value):
                output = vae.decode(value / float(scale)).sample.float()
                # VAE channel 1 is a validity logit (trained using BCE).
                # Convert it to the recorded [-1,1] image scale before MSE.
                return torch.cat((output[:, :1],
                                  2.0 * output[:, 1:2].sigmoid() - 1.0), dim=1)
            decoded = []
            for chunk in predicted.split(decoder_chunk_size):
                # Frozen weights, but differentiable output: propagate pixel
                # error through the decoder to the DiT. Checkpoint to bound RAM.
                decoded.append(checkpoint(decode, chunk, use_reentrant=False)
                               if torch.is_grad_enabled() else decode(chunk))
            image = torch.cat(decoded)
            gt = target_pixels.to(device, non_blocking=True).float()
            # Only the 18 measured elevation beams; exclude the 2 padded beams.
            error = image[..., :18] - gt[..., :18]
            pixel_mse = error.square().mean()
            range_mse_m2 = (error[:, 0] * 2.5).square().mean()
            mask_mse = error[:, 1].square().mean()
            return pixel_mse, range_mse_m2.detach(), mask_mse.detach()
        target_velocity = noise - target
        flow = (velocity.float() - target_velocity.float()).square().mean()
        x0_l1 = (predicted - target.float()).abs().mean()
        loss = flow + 0.1 * x0_l1
    return loss, flow.detach(), x0_l1.detach()


@torch.no_grad()
def evaluate(model, vae, loader, scale, device, precision,
             loss_space='latent', decoder_chunk_size=8):
    model.eval()
    loss_sum = flow_sum = x0_sum = samples = 0.0
    for batch in loader:
        history_image, target_image = batch[:2]
        loss, flow, x0 = flow_batch(
            model, vae, history_image, target_image, scale, device, precision,
            loss_space, batch[2] if len(batch) == 3 else None, decoder_chunk_size)
        count = float(len(history_image))
        flow_sum += float(flow) * count
        x0_sum += float(x0) * count
        samples += count
        loss_sum += float(loss) * count
    model.train()
    result = {
        "flow_mse": flow_sum / max(samples, 1.0),
        "x0_l1": x0_sum / max(samples, 1.0),
        "loss": loss_sum / max(samples, 1.0),
        "samples": int(samples),
    }
    if loss_space == 'pixel':
        result['range_mse_m2'] = result.pop('flow_mse')
        result['mask_mse'] = result.pop('x0_l1')
        result['pixel_mse'] = result['loss']
    return result


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def checkpoint_format(target_offset: int) -> str:
    return f"{FORMAT_PREFIX}-t{int(target_offset)}-v1"


def save_checkpoint(path, model, optimizer, step, metrics, args, scale):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": checkpoint_format(args.target_offset),
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "metrics": metrics,
        "architecture": {
            "history_frames": 3, "prediction_frames": 1,
            "latent_shape": [4, 27, 5], "width": args.width,
            "depth": args.depth, "heads": args.heads,
            "mlp_ratio": args.mlp_ratio,
            "conditioning": "history_lidar_only",
            "target": f"direct_t_plus_{args.target_offset}",
            "target_offset_lidar_frames": int(args.target_offset),
            "target_offset_physics_steps": int(args.target_offset * 10),
            "target_offset_seconds": float(args.target_offset * 10 * 0.016),
        },
        "vae": {"variant": "circular", "scaling_factor": float(scale),
                "frozen": True,
                "checkpoint": str(args.vae_checkpoint) if args.vae_checkpoint else None},
        "loss_space": args.loss_space,
        "config": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args, writer=None):
    args.train_root = args.train_root.expanduser().resolve()
    if args.validation_root is not None:
        args.validation_root = args.validation_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if args.train_root == args.validation_root:
        raise ValueError("train-root and validation-root must be independent")
    if args.latent_cache is None and (args.validation_root is None or not args.validation_root.exists()):
        raise FileNotFoundError(
            f"validation root missing: {args.validation_root}; no simulator was started")
    if args.target_offset < 1:
        raise ValueError("target-offset must be at least one LiDAR frame")
    if args.eval_windows < 1:
        raise ValueError("eval-windows must be positive")
    if args.decoder_chunk_size < 1:
        raise ValueError('decoder-chunk-size must be positive')
    if args.loss_space == 'pixel' and args.latent_cache is None:
        raise ValueError('pixel loss currently requires the validated 5 m cached dataset')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1.DEVICE = device

    split_info = {}
    if args.latent_cache is not None:
        from lidar_wam.runner.video_dit_cached_data import (
            CachedFutureDataset, load_cache_entries, split_entries)
        if args.vae_checkpoint is None:
            raise ValueError('latent-cache requires an explicit vae-checkpoint')
        if args.validation_root is not None:
            raise ValueError('cached mode uses whole-shard holdout; omit validation-root')
        args.latent_cache = args.latent_cache.expanduser().resolve()
        args.vae_checkpoint = args.vae_checkpoint.expanduser().resolve()
        entries, cache_metadata = load_cache_entries(
            args.train_root, args.latent_cache, args.vae_checkpoint)
        train_entries, val_entries = split_entries(entries, args.validation_fraction, args.seed)
        train_data = CachedFutureDataset(train_entries, args.target_offset,
                                        return_pixels=args.loss_space == 'pixel')
        validation_full = CachedFutureDataset(val_entries, args.target_offset,
                                             return_pixels=args.loss_space == 'pixel')
        split_info = {'validation_mode': 'whole_shard_holdout',
                      'train_shards': train_entries, 'validation_shards': val_entries,
                      'train_trajectories': train_data.trajectory_count,
                      'validation_trajectories': validation_full.trajectory_count,
                      'latent_source': 'validated_frozen_vae_cache',
                      'vae_sha256': cache_metadata['vae_sha256']}
    else:
        train_data = WorldOnlyDataset(
            args.train_root, max_windows=args.max_train_windows,
            random_seed=args.seed, target_offset=args.target_offset)
        validation_full = WorldOnlyDataset(
            args.validation_root, random_seed=args.seed,
            target_offset=args.target_offset)
    if len(validation_full) < args.eval_windows:
        raise ValueError(
            f"validation has {len(validation_full)} windows, but exactly "
            f"{args.eval_windows} are required")
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0)
    model = LiDARVideoDiT(
        args.width, args.depth, args.heads, args.mlp_ratio).to(device).float()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95))
    if args.vae_checkpoint is not None:
        from safetensors.torch import load_file
        vae_config = json.loads((args.vae_checkpoint.parent / 'config.json').read_text())
        vae = stage1.make_vae(circular=True)
        vae.load_state_dict(load_file(str(args.vae_checkpoint), device='cpu'), strict=True)
        vae.register_to_config(scaling_factor=float(vae_config['scaling_factor']))
        vae = vae.to(device).float().eval()
    else:
        vae = stage1.load_circular_vae().to(device).eval()
    vae.requires_grad_(False)
    scale = float(getattr(vae.config, "scaling_factor", 0.5281441482784038))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({
        "format": checkpoint_format(args.target_offset),
        "train_root": str(args.train_root),
        "validation_root": str(args.validation_root),
        "train_windows": len(train_data), "validation_windows": len(validation_full),
        "validation_subset": args.eval_windows,
        "validation_sampling": (
            "fresh random validation windows at every evaluation; "
            "without replacement; seed=base_seed+step"),
        "validation_index_dtype": "int64", "vae_scale": scale,
        "history_lidar_frames": 3,
        "target_offset_lidar_frames": int(args.target_offset),
        "target_offset_physics_steps": int(args.target_offset * 10),
        "target_offset_seconds": float(args.target_offset * 10 * 0.016),
        "super_started": False, "lee_started": False, **split_info, **vars(args),
    }, indent=2, default=str) + "\n")
    print(json.dumps({'train_windows': len(train_data),
                      'validation_windows': len(validation_full),
                      'validation_subset': args.eval_windows, **split_info}), flush=True)
    if writer is not None:
        writer.add_text('run/config', (args.out / 'config.json').read_text(), 0)

    start_step = 1
    best = math.inf
    no_improve = 0
    history = []
    if args.init_checkpoint is not None:
        if args.resume:
            raise ValueError('init-checkpoint and resume are mutually exclusive')
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        if initial.get('format') != checkpoint_format(args.target_offset):
            raise ValueError('initial checkpoint target offset mismatch')
        if not math.isclose(float(initial['vae']['scaling_factor']), scale, abs_tol=1e-12):
            raise ValueError('initial checkpoint VAE scale mismatch')
        model.load_state_dict(initial['model'], strict=True)
        print(json.dumps({'initialized_from': str(args.init_checkpoint),
                          'source_step': initial['step'], 'loss_space': args.loss_space}), flush=True)
        del initial
    if args.resume:
        checkpoint = torch.load(args.out / "latest.pt", map_location="cpu",
                                weights_only=False)
        if checkpoint.get("format") != checkpoint_format(args.target_offset):
            raise ValueError(
                "resume checkpoint target offset does not match this run")
        if checkpoint.get('loss_space', 'latent') != args.loss_space:
            raise ValueError('resume loss-space mismatch; use init-checkpoint for a new objective')
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        if (args.out / 'best.pt').is_file():
            best_payload = torch.load(args.out / 'best.pt', map_location='cpu', weights_only=False)
            best = float(best_payload.get('metrics', {}).get('loss', math.inf))
            del best_payload
        history_path = args.out / 'history.json'
        if history_path.is_file():
            history = json.loads(history_path.read_text())
        no_improve = int(checkpoint.get('metrics', {}).get('no_improve_checks', 0))
        del checkpoint

    iterator = iter(train_loader)
    started = time.monotonic()
    deadline = (started + args.max_hours * 3600
                if args.max_hours and args.max_hours > 0 else None)
    for step in range(start_step, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        history_image, target_image = batch[:2]
        optimizer.zero_grad(set_to_none=True)
        loss, flow, x0 = flow_batch(
            model, vae, history_image, target_image, scale, device, args.precision,
            args.loss_space, batch[2] if len(batch) == 3 else None, args.decoder_chunk_size)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite DiT loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"non-finite DiT gradient at step {step}")
        optimizer.step()
        if step == start_step or step % args.log_every == 0:
            row = {"step": step, "loss": float(loss), "flow_mse": float(flow),
                   "x0_l1": float(x0), "grad_norm": float(grad),
                   "elapsed_min": (time.monotonic() - started) / 60.0}
            if args.loss_space == 'pixel':
                row['range_mse_m2'] = row.pop('flow_mse')
                row['mask_mse'] = row.pop('x0_l1')
                row['pixel_mse'] = row['loss']
            history.append(row)
            print(json.dumps(row), flush=True)
            if writer is not None:
                keys = (('loss', 'pixel_mse', 'range_mse_m2', 'mask_mse', 'grad_norm')
                        if args.loss_space == 'pixel' else ('loss', 'flow_mse', 'x0_l1', 'grad_norm'))
                for key in keys:
                    writer.add_scalar('train/' + key, row[key], step)
                writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], step)
                if device.type == 'cuda':
                    writer.add_scalar('system/gpu_peak_allocated_gib',
                                      torch.cuda.max_memory_allocated(device)/(1024**3), step)
        do_eval = step % args.eval_every == 0 or step == args.steps
        if do_eval:
            # Do not tune to one permanently fixed held-out subset.  Every
            # evaluation draws a fresh, reproducible sample of exactly
            # ``eval_windows`` validation windows from the independent maps.
            eval_seed = int(args.seed + step)
            selected = fixed_eval_indices(
                len(validation_full), args.eval_windows, eval_seed)
            val_loader = DataLoader(
                Subset(validation_full, selected.tolist()),
                batch_size=args.eval_batch_size, shuffle=False,
                num_workers=max(0, min(args.workers, 2)),
                pin_memory=device.type == "cuda")
            np.save(args.out / f"validation_indices_step_{step:07d}.npy", selected)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else []):
                torch.manual_seed(eval_seed)
                metrics = evaluate(model, vae, val_loader, scale, device, args.precision,
                                   args.loss_space, args.decoder_chunk_size)
            metrics.update({"step": step, "seed": eval_seed,
                            "requested_samples": args.eval_windows})
            print(json.dumps({"validation": metrics}), flush=True)
            (args.out / f"evaluation_step_{step}.json").write_text(
                json.dumps(metrics, indent=2) + "\n")
            save_checkpoint(args.out / "latest.pt", model, optimizer, step,
                            metrics, args, scale)
            if metrics["loss"] < best - args.min_delta:
                best = metrics["loss"]
                no_improve = 0
                save_checkpoint(args.out / "best.pt", model, optimizer, step,
                                metrics, args, scale)
            else:
                no_improve += 1
            if writer is not None:
                keys = (('loss', 'pixel_mse', 'range_mse_m2', 'mask_mse')
                        if args.loss_space == 'pixel' else ('loss', 'flow_mse', 'x0_l1'))
                for key in keys:
                    writer.add_scalar('validation/' + key, metrics[key], step)
                writer.add_scalar('validation/best_loss', best, step)
                writer.add_scalar('validation/no_improve_checks', no_improve, step)
                writer.add_scalar('validation/samples', metrics['samples'], step)
                writer.flush()
            (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if no_improve >= args.patience:
                print(json.dumps({"early_stop": True, "step": step,
                                  "best_validation_loss": best}), flush=True)
                break
        elif step % args.checkpoint_every == 0:
            save_checkpoint(args.out / "latest.pt", model, optimizer, step,
                            {"step": step, "train_loss": float(loss)}, args, scale)
        if deadline is not None and time.monotonic() >= deadline:
            print(json.dumps({"max_hours_reached": args.max_hours, "step": step}), flush=True)
            break


def main():
    from torch.utils.tensorboard import SummaryWriter
    args = parse_args()
    log_dir = args.tensorboard_dir or args.out / 'tensorboard'
    with SummaryWriter(str(log_dir), flush_secs=15) as writer:
        train(args, writer)


if __name__ == "__main__":
    main()
