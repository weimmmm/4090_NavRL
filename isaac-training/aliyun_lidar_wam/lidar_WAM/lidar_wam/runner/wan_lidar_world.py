"""Wan2.2 Video DiT world model for NavRL LiDAR latent clips.

This runner is deliberately independent from the existing LaGen UNet runner.
It keeps the trained circular VAE frozen, constructs one current frame plus
four future ten-action transitions, and adapts Wan2.2's pretrained video DiT
to the four-channel LiDAR latent space.

The command is usable on the PPU environment:

  python -m lidar_wam.runner.wan_lidar_world inspect-sequences
  python -m lidar_wam.runner.wan_lidar_world check-load --wan-checkpoint PATH
  python -m lidar_wam.runner.wan_lidar_world train --overfit --wan-checkpoint PATH
  python -m lidar_wam.runner.wan_lidar_world evaluate --split val --checkpoint PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "third_party" / "diffusers" / "src"))
from lidar_wam.runner import stage1
from lidar_wam.models.wan_lidar import WanContinuousFlowMatchScheduler, WanVideoDiT
from lidar_wam.models.wan_lidar.state_dict_converters import (
    wan_video_dit_from_diffusers,
    wan_video_dit_state_dict_converter,
)
from lidar_wam.runner.lidar_geometry import frame_points, load_rays

OUT = PROJECT / "outputs" / "wan_lidar_1to4"
LATENT_DIR = "latents_circular"
HORIZONS = 4
VIDEO_FRAMES = 5
HORIZONTAL = 27
VERTICAL = 5
PAD_HORIZONTAL = 28
PAD_VERTICAL = 6
WAN_IN_DIM = 48
WAN_OUT_DIM = 48
WAN_HIDDEN = 3072
WAN_FFN = 14336
WAN_CONTEXT = 4096
WAN_FREQ = 256
DEVICE = stage1.DEVICE


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def causal_state(value):
    return stage1.causal_state(np.asarray(value, dtype=np.float32))


def _decode_token(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


class ClipDataset(Dataset):
    """Continuous five-frame clips backed by the immutable HDF5 cache."""

    def __init__(self, split: str, data_root: Path, latent_root: Path,
                 limit_per_seed: int | None = None, random_seed: int = 42,
                 overfit: bool = False, with_images: bool = False):
        self.split = split
        self.data_root = Path(data_root)
        self.latent_root = Path(latent_root)
        cache = np.load(self.latent_root / f"{split}.npz")
        metadata = json.loads((self.latent_root / "metadata.json").read_text())
        scale = float(metadata["scaling_factor"])
        source = cache["source_index"].astype(np.int64)
        source_pos = {int(index): position for position, index in enumerate(source)}
        h5_path = self.data_root / f"navrl_static_{split}.h5"
        with h5py.File(h5_path, "r") as h5:
            tokens = h5["token"][:]
            previous_tokens = h5["prev_token"][:]
            scenes = h5["scene_token"][:]
            frames = h5["frame_idx"][:]
            seeds = h5["terrain_seed"][:]
            action_mask = h5["action_mask"][:]
            step_delta = h5["step_delta"][:]
        valid = set(source.tolist())
        # Row j is the successor transition when its previous token equals
        # row i's target token.
        successor = {previous_tokens[i]: i for i in range(len(previous_tokens))}
        selected = []
        chains = []
        for index in source:
            index = int(index)
            chain = [index]
            # Each HDF5 row is one transition. Four rows therefore provide
            # t->t+1, ..., t+3->t+4 and the four ten-action chunks.
            for _ in range(HORIZONS - 1):
                next_index = successor.get(tokens[chain[-1]])
                if next_index is None or int(next_index) not in valid:
                    break
                chain.append(int(next_index))
            if len(chain) != HORIZONS:
                continue
            if any(scenes[j] != scenes[index] or
                   int(frames[j]) != int(frames[index]) + h
                   for h, j in enumerate(chain)):
                continue
            if any(int(step_delta[j]) != 10 or not bool(np.all(action_mask[j]))
                   for j in chain[:-1]):
                continue
            if any(j not in source_pos for j in chain):
                continue
            selected.append(index)
            chains.append(chain)
        if not chains:
            raise RuntimeError(f"No continuous {VIDEO_FRAMES}-frame clips in {split}")
        selected = np.asarray(selected, dtype=np.int64)
        chains = np.asarray(chains, dtype=np.int64)
        chain_pos = np.asarray([[source_pos[int(j)] for j in chain] for chain in chains])
        chain_seeds = seeds[selected].astype(np.int64)
        if overfit:
            keep = np.arange(min(128, len(chains)))
        elif limit_per_seed is not None:
            rng = np.random.default_rng(random_seed)
            pieces = []
            for seed in sorted(set(chain_seeds.tolist())):
                candidates = np.flatnonzero(chain_seeds == seed)
                count = min(limit_per_seed, len(candidates))
                if count < limit_per_seed:
                    raise RuntimeError(
                        f"{split} seed {seed} has {len(candidates)} clips, "
                        f"requested {limit_per_seed}")
                pieces.append(rng.choice(candidates, count, replace=False))
            keep = np.sort(np.concatenate(pieces))
        else:
            keep = np.arange(len(chains))
        self.source_indices = chains[keep]
        self.seeds = chain_seeds[keep]
        self.initial = torch.from_numpy(
            cache["previous"][chain_pos[keep, 0]].copy() * scale).float()
        self.future = torch.from_numpy(
            cache["target"][chain_pos[keep]].copy() * scale).float()
        self.actions = torch.from_numpy(
            cache["actions"][chain_pos[keep]].copy()).float()
        self.state = torch.from_numpy(
            causal_state(cache["state"][chain_pos[keep, 0]])).float()
        self.scale = scale
        self.with_images = bool(with_images)
        self._h5 = None
        if self.with_images:
            self._h5_path = h5_path

    def _open_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, index):
        item = (self.initial[index], self.future[index], self.actions[index],
                self.state[index], torch.from_numpy(self.source_indices[index]))
        if not self.with_images:
            return item
        h5 = self._open_h5()
        images = np.asarray(h5["range_values"][self.source_indices[index]],
                            dtype=np.float32)
        return item + (torch.from_numpy(images),)

    def __del__(self):
        if getattr(self, "_h5", None) is not None:
            self._h5.close()


def build_manifest(data_root: Path, latent_root: Path, output: Path,
                   split: str, per_seed: int = 256, seed: int = 42):
    data = ClipDataset(split, data_root, latent_root, per_seed,
                       random_seed=seed)
    rows = [{"source_indices": [int(v) for v in chain],
             "seed": int(seed_value)}
            for chain, seed_value in zip(data.source_indices, data.seeds)]
    save_json(output / f"{split}_clip_manifest.json", {
        "version": 1, "split": split, "random_seed": seed,
        "clips_per_seed": per_seed, "horizons": HORIZONS, "rows": rows})
    return data


def _pad_lidar(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5 or x.shape[-2:] != (HORIZONTAL, VERTICAL):
        raise ValueError(f"Expected [B,C,T,27,5], got {tuple(x.shape)}")
    # The first spatial latent axis is the circular azimuth axis.
    x = torch.cat([x, x[..., :1, :]], dim=-2)
    return F.pad(x, (0, 1, 0, 0, 0, 0))


def _unpad_lidar(x: torch.Tensor) -> torch.Tensor:
    return x[..., :HORIZONTAL, :VERTICAL]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / max(self.rank, 1)
        # LoRA is injected after the pretrained Wan backbone has already been
        # moved to its execution device.  Create the new parameters beside the
        # wrapped weight instead of leaving them on the CPU.
        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.down = nn.Linear(
            base.in_features, self.rank, bias=False, **factory_kwargs)
        self.up = nn.Linear(
            self.rank, base.out_features, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, value):
        return self.base(value) + self.scale * self.up(self.down(value))


def _replace_linear(parent: nn.Module, name: str, rank: int, alpha: float):
    child = getattr(parent, name)
    if isinstance(child, nn.Linear):
        setattr(parent, name, LoRALinear(child, rank, alpha))


def add_lora(dit: nn.Module, rank: int = 16, alpha: float = 16.0):
    for block in dit.blocks:
        for module_name in ("self_attn", "cross_attn"):
            module = getattr(block, module_name)
            for name in ("q", "k", "v", "o"):
                _replace_linear(module, name, rank, alpha)
        for name, module in list(block.ffn._modules.items()):
            if isinstance(module, nn.Linear):
                block.ffn._modules[name] = LoRALinear(module, rank, alpha)


class WanLiDARWorld(nn.Module):
    """Wan Video DiT with four-channel circular-VAE latent adapters."""

    def __init__(self, pretrained_path: Path | None, dtype=torch.bfloat16,
                 gradient_checkpointing: bool = True, allow_random: bool = False):
        super().__init__()
        self.dtype = dtype
        config = dict(
            hidden_dim=WAN_HIDDEN, in_dim=WAN_IN_DIM, ffn_dim=WAN_FFN,
            out_dim=WAN_OUT_DIM, text_dim=WAN_CONTEXT, freq_dim=WAN_FREQ,
            eps=1e-6, patch_size=(1, 2, 2), num_heads=24,
            attn_head_dim=128, num_layers=30, has_image_input=False,
            has_image_pos_emb=False, has_ref_conv=False,
            add_control_adapter=False, seperated_timestep=True,
            require_vae_embedding=False, require_clip_embedding=False,
            fuse_vae_embedding_in_latents=True, action_conditioned=False,
            video_attention_mask_mode="bidirectional",
            use_gradient_checkpointing=gradient_checkpointing,
        )
        self.dit = WanVideoDiT(**config)
        self.lidar_in = nn.Conv3d(5, WAN_IN_DIM, kernel_size=1)
        self.lidar_out = nn.Conv3d(WAN_OUT_DIM, 4, kernel_size=1)
        self.action_encoder = nn.Sequential(
            nn.Linear(3, 1024), nn.SiLU(), nn.Linear(1024, WAN_CONTEXT))
        self.state_encoder = nn.Sequential(
            nn.Linear(5, 1024), nn.SiLU(), nn.Linear(1024, WAN_CONTEXT))
        self.segment_embedding = nn.Parameter(torch.zeros(1, HORIZONS, 10, WAN_CONTEXT))
        self.step_embedding = nn.Parameter(torch.zeros(1, 1, 10, WAN_CONTEXT))
        self.null_context = nn.Parameter(torch.zeros(1, 1, WAN_CONTEXT))
        nn.init.normal_(self.segment_embedding, std=0.02)
        nn.init.normal_(self.step_embedding, std=0.02)
        nn.init.normal_(self.null_context, std=0.02)
        self.loaded_report = {}
        if pretrained_path is not None:
            self.loaded_report = self._load_pretrained(Path(pretrained_path))
        elif not allow_random:
            raise FileNotFoundError(
                "Wan checkpoint is required. Pass --wan-checkpoint or use "
                "--allow-random only for the random-init ablation.")
        self._freeze_dit()

    def _freeze_dit(self):
        for parameter in self.dit.parameters():
            parameter.requires_grad_(False)

    def _load_pretrained(self, path: Path):
        files = [path] if path.is_file() else sorted(
            list(path.glob("*.safetensors")) + list(path.glob("*.bin")) +
            list(path.glob("*.pt")))
        if not files:
            raise FileNotFoundError(f"No Wan checkpoint files found under {path}")
        state = {}
        for file in files:
            part = load_file(str(file), device="cpu") if file.suffix == ".safetensors" else \
                torch.load(file, map_location="cpu", weights_only=True)
            if isinstance(part, dict) and "state_dict" in part:
                part = part["state_dict"]
            state.update(part)
        normalized = {}
        for key, value in state.items():
            while key.startswith(("module.", "model.", "diffusion_model.")):
                key = key.split(".", 1)[1]
            normalized[key] = value
        state = normalized
        if any("attn1." in key for key in state):
            state = wan_video_dit_from_diffusers(state)
        compatible = {}
        skipped = []
        own = self.dit.state_dict()
        for key, value in state.items():
            if key in own and tuple(own[key].shape) == tuple(value.shape):
                compatible[key] = value
            elif key in own:
                skipped.append(key)
        missing, unexpected = self.dit.load_state_dict(compatible, strict=False)
        block_keys = [key for key in own if key.startswith("blocks.")]
        loaded_block = sum(key in compatible for key in block_keys)
        report = {
            "checkpoint": str(path), "files": [str(v) for v in files],
            "loaded_tensors": len(compatible), "skipped_shape": skipped[:20],
            "missing": list(missing)[:40], "unexpected": list(unexpected)[:20],
            "block_tensor_fraction": loaded_block / max(len(block_keys), 1),
        }
        if report["block_tensor_fraction"] < 0.7:
            raise RuntimeError(
                "Wan checkpoint did not load enough transformer block tensors: "
                + json.dumps(report))
        return report

    def add_lora(self, rank: int = 16):
        add_lora(self.dit, rank=rank)

    def condition(self, actions: torch.Tensor, state: torch.Tensor):
        # [B,4,10,3] -> [B,40,4096], preserving transition and step order.
        if actions.shape[1:] != (HORIZONS, 10, 3):
            raise ValueError(f"Expected actions [B,4,10,3], got {tuple(actions.shape)}")
        action = self.action_encoder(actions.to(dtype=self.dtype))
        action = action + self.segment_embedding.to(action.dtype) + \
            self.step_embedding.to(action.dtype)
        action = action.reshape(actions.shape[0], HORIZONS * 10, WAN_CONTEXT)
        state_token = self.state_encoder(state.to(dtype=self.dtype)).unsqueeze(1)
        null = self.null_context.to(dtype=action.dtype).expand(actions.shape[0], -1, -1)
        context = torch.cat([null, state_token, action], dim=1)
        mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        return context, mask

    def forward(self, noisy: torch.Tensor, actions: torch.Tensor,
                state: torch.Tensor, timestep: torch.Tensor):
        # noisy: [B,4,T,27,5], where frame zero is the clean observation.
        if noisy.ndim != 5 or noisy.shape[1:] != (4, VIDEO_FRAMES, 27, 5):
            raise ValueError(f"Expected noisy [B,4,5,27,5], got {tuple(noisy.shape)}")
        known = torch.zeros(
            (noisy.shape[0], 1, VIDEO_FRAMES, HORIZONTAL, VERTICAL),
            device=noisy.device, dtype=noisy.dtype)
        known[:, :, 0] = 1
        model_input = _pad_lidar(torch.cat([noisy, known], dim=1))
        model_input = self.lidar_in(model_input.to(dtype=self.dtype))
        context, context_mask = self.condition(actions, state)
        pred = self.dit(
            model_input, timestep.to(dtype=self.dtype), context, context_mask,
            fuse_vae_embedding_in_latents=True)
        return _unpad_lidar(self.lidar_out(pred).float())


def load_model(args, allow_random=False):
    checkpoint = None if args.wan_checkpoint is None else Path(args.wan_checkpoint)
    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = WanLiDARWorld(
        checkpoint, dtype=model_dtype,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        allow_random=allow_random,
    ).to(DEVICE, dtype=model_dtype)
    return model


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def trainable_state_dict(model):
    return {name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def load_training_checkpoint(model, checkpoint):
    if "trainable" in checkpoint:
        current = model.state_dict()
        current.update(checkpoint["trainable"])
        model.load_state_dict(current, strict=True)
    else:
        # Compatibility with the one-step smoke checkpoint created before the
        # compact checkpoint format was enabled.
        model.load_state_dict(checkpoint["model"], strict=True)


def flow_batch(model, batch, scheduler, vae=None, scale=1.0,
               decoded_aux=False, aux_weight=0.0):
    initial, future, actions, state, source = batch[:5]
    del source
    initial, future = initial.to(DEVICE), future.to(DEVICE)
    actions, state = actions.to(DEVICE), state.to(DEVICE)
    # [B,4,4,H,W] -> [B,4,T,H,W]
    clean = torch.cat([initial.unsqueeze(1), future], dim=1).permute(0, 2, 1, 3, 4)
    noise = torch.randn_like(clean)
    timestep = scheduler.sample_training_t(
        clean.shape[0], clean.device, clean.dtype)
    noisy = scheduler.add_noise(clean, noise, timestep)
    noisy[:, :, 0] = initial
    target = scheduler.training_target(clean, noise, timestep)
    prediction = model(noisy, actions, state, timestep)
    future_slice = (slice(None), slice(None), slice(1, None), slice(None), slice(None))
    flow_loss = F.mse_loss(prediction[future_slice], target[future_slice])
    sigma = (timestep / scheduler.num_train_timesteps).view(-1, 1, 1, 1, 1)
    x0 = noisy - sigma * prediction
    x0_future = x0[:, :, 1:]
    x0_loss = F.l1_loss(x0_future, future.permute(0, 2, 1, 3, 4))
    aux = torch.zeros((), device=DEVICE)
    if decoded_aux and vae is not None and len(batch) > 5:
        eligible = timestep < scheduler.num_train_timesteps * 0.5
        if bool(eligible.any()):
            # torch.flatnonzero is unavailable in the PPU PyTorch build.
            chosen = torch.nonzero(eligible, as_tuple=False).flatten()[:64]
            image = batch[-1][chosen.cpu()].to(DEVICE)
            chosen_device = chosen.to(DEVICE)
            decoded = vae.decode(
                x0_future[chosen_device].permute(0, 2, 1, 3, 4).reshape(-1, 4, 27, 5)
                / scale).sample
            decoded = decoded.reshape(len(chosen), HORIZONS, 2, 108, 20)
            target_image = image
            gt_hit = target_image[:, :, 1:2, :, :18] > 0
            pred_mask = decoded[:, :, 1:2, :, :18]
            pred_range = decoded[:, :, :1, :, :18]
            gt_range = target_image[:, :, :1, :, :18]
            pos = gt_hit.float()
            neg = (~gt_hit).float()
            mask_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(pred_mask, pos, reduction="none")
                .mul(pos).sum() / pos.sum().clamp_min(1)
                + F.binary_cross_entropy_with_logits(pred_mask, pos, reduction="none")
                .mul(neg).sum() / neg.sum().clamp_min(1))
            range_loss = (pred_range - gt_range).abs().mul(pos).sum() / pos.sum().clamp_min(1)
            predicted_count = (pred_mask > 1.5).float().flatten(2).sum(-1)
            true_count = pos.flatten(2).sum(-1)
            nonempty = true_count > 0
            if bool(nonempty.any()):
                empty_loss = F.relu(
                    1.0 - predicted_count[nonempty] /
                    true_count[nonempty].clamp_min(1)).mean()
            else:
                empty_loss = torch.zeros((), device=pred_mask.device)
            aux = aux_weight * (0.05 * mask_loss + 0.025 * range_loss +
                                0.025 * empty_loss)
    return flow_loss + 0.1 * x0_loss + aux, {
        "flow_mse": float(flow_loss.detach()),
        "x0_l1": float(x0_loss.detach()),
        "decoded_aux": float(aux.detach()),
    }


@torch.no_grad()
def _sample(model, initial, actions, state, scheduler, steps=20, seed=42):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int(seed))
    shape = (initial.shape[0], 4, VIDEO_FRAMES, HORIZONTAL, VERTICAL)
    sample = torch.randn(shape, device=DEVICE, dtype=initial.dtype,
                         generator=generator)
    sample[:, :, 0] = initial
    timesteps, deltas = scheduler.build_inference_schedule(
        steps, DEVICE, sample.dtype)
    for timestep, delta in zip(timesteps, deltas):
        t = timestep.expand(initial.shape[0])
        sample[:, :, 0] = initial
        velocity = model(sample, actions, state, t)
        sample = scheduler.step(velocity, delta, sample)
    sample[:, :, 0] = initial
    return sample[:, :, 1:].permute(0, 2, 1, 3, 4)


def _squared_chamfer(pred, target, rays, threshold=1.5):
    p = frame_points(pred, rays, threshold)
    q = frame_points(target, rays, 0.0)
    if not len(p) and not len(q):
        return 0.0, False
    if not len(p) or not len(q):
        return 200.0, bool(not len(p) and len(q))
    return float(np.square(cKDTree(p).query(q)[0]).mean() +
                 np.square(cKDTree(q).query(p)[0]).mean()), False


@torch.no_grad()
def evaluate(args):
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluate")
    data_root = Path(args.data_root).resolve()
    latent_root = Path(args.latent_root).resolve()
    data = ClipDataset(args.split, data_root, latent_root, args.per_seed,
                       random_seed=42, with_images=True)
    model = load_model(args)
    if args.lora_rank > 0:
        model.add_lora(args.lora_rank)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_training_checkpoint(model, checkpoint)
    model.eval()
    scheduler = WanContinuousFlowMatchScheduler(1000, shift=5.0)
    vae = stage1.load_circular_vae()
    scale = data.scale
    raw_root = Path(args.raw_root).resolve()
    rays = {int(seed): load_rays(raw_root, args.split, int(seed))[0]
            for seed in sorted(set(data.seeds.tolist()))}
    rows = []
    for start in range(0, len(data), args.eval_batch_size):
        stop = min(start + args.eval_batch_size, len(data))
        batch = [data[i] for i in range(start, stop)]
        initial, future, actions, state, source, images = zip(*batch)
        initial = torch.stack(initial).to(DEVICE)
        actions = torch.stack(actions).to(DEVICE)
        state = torch.stack(state).to(DEVICE)
        prediction = _sample(model, initial, actions, state, scheduler,
                             args.ddim_steps, args.seed + start)
        decoded = vae.decode(
            prediction.reshape(-1, 4, 27, 5) / scale).sample
        decoded = decoded.reshape(len(batch), HORIZONS, 2, 108, 20).cpu().numpy()
        if start == 0 and len(batch):
            current = vae.decode(initial[:1] / scale).sample[0].cpu().numpy()
            for preview_horizon in (0, HORIZONS - 1):
                stage1.save_preview(
                    Path(args.output) / f"{args.split}_preview_h{preview_horizon + 1}.png",
                    current, images[0][preview_horizon].numpy(),
                    decoded[0, preview_horizon], pred_mask_threshold=1.5)
        shuffled_decoded = None
        if args.action_sensitivity:
            shuffled_actions = torch.roll(actions, shifts=1, dims=2)
            shuffled = _sample(model, initial, shuffled_actions, state, scheduler,
                               args.ddim_steps, args.seed + start)
            shuffled_decoded = vae.decode(
                shuffled.reshape(-1, 4, 27, 5) / scale).sample
            shuffled_decoded = shuffled_decoded.reshape(
                len(batch), HORIZONS, 2, 108, 20).cpu().numpy()
        for j in range(len(batch)):
            for horizon in range(HORIZONS):
                seed = int(data.seeds[start + j])
                cd, false_empty = _squared_chamfer(
                    decoded[j, horizon], images[j][horizon].numpy(), rays[seed])
                gt_hit = images[j][horizon][1, :, :18].numpy() > 0
                pred_hit = decoded[j, horizon, 1, :, :18] > 1.5
                pred_range = np.clip((decoded[j, horizon, 0, :, :18] + 1) * 5, 0, 10)
                gt_range = np.clip((images[j][horizon][0, :, :18].numpy() + 1) * 5, 0, 10)
                row = {
                    "source_indices": [int(v) for v in source[j].numpy()],
                    "seed": seed, "horizon": horizon + 1,
                    "cd_paper_m2": cd, "empty_cloud": bool(false_empty),
                    "valid_range_mae_m": float(np.abs(pred_range[gt_hit] -
                                                       gt_range[gt_hit]).mean())
                    if gt_hit.any() else 0.0,
                    "pred_hit_count": int(pred_hit.sum()),
                    "gt_hit_count": int(gt_hit.sum()),
                    "tp": int((pred_hit & gt_hit).sum()),
                    "fp": int((pred_hit & ~gt_hit).sum()),
                    "fn": int((~pred_hit & gt_hit).sum()),
                }
                if shuffled_decoded is not None:
                    row["shuffled_cd_paper_m2"] = _squared_chamfer(
                        shuffled_decoded[j, horizon], images[j][horizon].numpy(),
                        rays[seed])[0]
                rows.append(row)
    summary = {}
    for horizon in range(1, HORIZONS + 1):
        selected = [row for row in rows if row["horizon"] == horizon]
        tp = sum(row["tp"] for row in selected)
        fp = sum(row["fp"] for row in selected)
        fn = sum(row["fn"] for row in selected)
        summary[f"horizon_{horizon}"] = {
            "samples": len(selected),
            "cd_paper_m2": float(np.mean([row["cd_paper_m2"] for row in selected])),
            "valid_range_mae_m": float(np.mean(
                [row["valid_range_mae_m"] for row in selected])),
            "pred_hit_count_mean": float(np.mean(
                [row["pred_hit_count"] for row in selected])),
            "gt_hit_count_mean": float(np.mean(
                [row["gt_hit_count"] for row in selected])),
            "mask_precision": tp / max(tp + fp, 1),
            "mask_recall": tp / max(tp + fn, 1),
            "mask_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "false_empty": int(sum(row["empty_cloud"] for row in selected)),
        }
        if args.action_sensitivity:
            summary[f"horizon_{horizon}"]["shuffled_cd_paper_m2"] = float(
                np.mean([row["shuffled_cd_paper_m2"] for row in selected]))
    result = {"split": args.split, "checkpoint": str(args.checkpoint),
              "rows": rows, "summary": summary}
    save_json(Path(args.output) / f"{args.split}_metrics.json", result)
    print(json.dumps(summary, indent=2), flush=True)


def inspect_sequences(args):
    data = ClipDataset(args.split, Path(args.data_root), Path(args.latent_root),
                       args.per_seed, random_seed=42)
    result = {
        "split": args.split, "clips": len(data),
        "source_shape": list(data.source_indices.shape),
        "initial_shape": list(data.initial.shape),
        "future_shape": list(data.future.shape),
        "actions_shape": list(data.actions.shape),
        "state_shape": list(data.state.shape),
        "seed_counts": {str(seed): int(np.sum(data.seeds == seed))
                        for seed in sorted(set(data.seeds.tolist()))},
    }
    save_json(Path(args.output) / f"{args.split}_sequence_inspection.json", result)
    print(json.dumps(result, indent=2), flush=True)


def prepare(args):
    output = Path(args.output).resolve()
    results = {}
    for split in ("val", "test"):
        data = build_manifest(Path(args.data_root), Path(args.latent_root),
                              output, split, args.per_seed, seed=42)
        results[split] = {"clips": len(data),
                          "seed_counts": {str(seed): int(np.sum(data.seeds == seed))
                                          for seed in sorted(set(data.seeds.tolist()))}}
    save_json(output / "manifest_config.json", {
        "random_seed": 42, "clips_per_seed": args.per_seed,
        "horizons": HORIZONS, "results": results})
    print(json.dumps(results, indent=2), flush=True)


def check_load(args):
    model = load_model(args)
    model.eval()
    batch = 1
    initial = torch.zeros(batch, 4, 27, 5, device=DEVICE,
                          dtype=model.dtype)
    noisy = torch.zeros(batch, 4, VIDEO_FRAMES, 27, 5,
                        device=DEVICE, dtype=model.dtype)
    noisy[:, :, 0] = initial
    actions = torch.zeros(batch, HORIZONS, 10, 3, device=DEVICE,
                          dtype=model.dtype)
    state = torch.zeros(batch, 5, device=DEVICE, dtype=model.dtype)
    with torch.no_grad():
        prediction = model(noisy, actions, state,
                           torch.ones(batch, device=DEVICE, dtype=model.dtype) * 500)
    result = {"loaded_report": model.loaded_report,
              "input_shape": list(noisy.shape),
              "output_shape": list(prediction.shape),
              "dtype": str(prediction.dtype)}
    save_json(Path(args.output) / "load_check.json", result)
    print(json.dumps(result, indent=2), flush=True)


def train(args):
    seed_everything(args.seed)
    data_root = Path(args.data_root).resolve()
    latent_root = Path(args.latent_root).resolve()
    run_dir = Path(args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_data = ClipDataset("train", data_root, latent_root,
                             overfit=args.overfit, with_images=args.decoded_aux)
    val_data = ClipDataset("val", data_root, latent_root, args.val_per_seed,
                           random_seed=42, with_images=False)
    model = load_model(args)
    if args.lora_rank > 0:
        model.add_lora(args.lora_rank)
    vae = stage1.load_circular_vae() if args.decoded_aux else None
    if vae is not None:
        for parameter in vae.parameters():
            parameter.requires_grad_(False)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    scheduler = WanContinuousFlowMatchScheduler(1000, shift=5.0)
    loader = DataLoader(train_data, batch_size=args.micro_batch,
                        shuffle=True, drop_last=True, num_workers=0)
    iterator = iter(loader)
    history = []
    best = float("inf")
    started = time.monotonic()
    config = {
        "model": "Wan2.2-TI2V-5B Video DiT adapted to NavRL LiDAR",
        "horizons": HORIZONS, "actions_per_transition": 10,
        "latent_shape": [4, 27, 5], "vae": stage1.circular_vae_identity(),
        "wan_checkpoint": str(args.wan_checkpoint) if args.wan_checkpoint else None,
        "lora_rank": args.lora_rank,
        "wan_loaded_report": model.loaded_report,
        "steps": args.steps, "micro_batch": args.micro_batch,
        "gradient_accumulation": args.grad_accum,
        "flow_shift": 5.0, "decoded_aux": args.decoded_aux,
    }
    save_json(run_dir / "config.json", config)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_stats = {}
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            loss, stats = flow_batch(
                model, batch, scheduler, vae=vae, scale=train_data.scale,
                decoded_aux=args.decoded_aux,
                aux_weight=args.aux_weight)
            (loss / args.grad_accum).backward()
            for key, value in stats.items():
                total_stats[key] = total_stats.get(key, 0.0) + value / args.grad_accum
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            record = {"step": step, "loss": float(loss.detach()),
                      "grad_norm": float(grad), **total_stats,
                      "elapsed_min": round((time.monotonic() - started) / 60, 2)}
            history.append(record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vals = []
                for i in range(min(len(val_data), args.val_batches * args.micro_batch)):
                    item = val_data[i]
                    initial, future, actions, state, source = item
                    clean = torch.cat([initial.unsqueeze(0).unsqueeze(1),
                                       future.unsqueeze(0)], dim=1).permute(0, 2, 1, 3, 4).to(DEVICE)
                    actions = actions.unsqueeze(0).to(DEVICE)
                    state = state.unsqueeze(0).to(DEVICE)
                    noise = torch.randn_like(clean)
                    t = torch.ones(1, device=DEVICE, dtype=clean.dtype) * 500
                    noisy = scheduler.add_noise(clean, noise, t)
                    noisy[:, :, 0] = clean[:, :, 0]
                    pred = model(noisy, actions, state, t)
                    vals.append(float(F.mse_loss(
                        pred[:, :, 1:], scheduler.training_target(clean, noise, t)[:, :, 1:])))
            score = float(np.mean(vals)) if vals else float("inf")
            print(json.dumps({"step": step, "validation_flow_mse": score}), flush=True)
            compact = {"trainable": trainable_state_dict(model),
                       "optimizer": optimizer.state_dict(),
                       "step": step, "validation_flow_mse": score}
            torch.save(compact, run_dir / "latest.pt")
            if score < best:
                best = score
                torch.save(compact, run_dir / "best.pt")
            save_json(run_dir / "history.json", history)
    print(json.dumps({"best_validation_flow_mse": best,
                      "checkpoint": str(run_dir / "best.pt")}), flush=True)


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, default=stage1.DATA)
    common.add_argument("--latent-root", type=Path,
                        default=stage1.OUT / LATENT_DIR)
    common.add_argument("--output", type=Path, default=OUT)
    common.add_argument("--wan-checkpoint", type=Path)
    common.add_argument("--allow-random", action="store_true")
    common.add_argument("--bf16", action="store_true", default=True)
    common.add_argument("--no-gradient-checkpointing", action="store_true")
    common.add_argument("--seed", type=int, default=42)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("inspect-sequences", "check-load"):
        sub.add_parser(name, parents=[common])
    prep = sub.add_parser("prepare", parents=[common])
    prep.add_argument("--per-seed", type=int, default=256)
    tr = sub.add_parser("train", parents=[common])
    tr.add_argument("--steps", type=int, default=20000)
    tr.add_argument("--micro-batch", type=int, default=16)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--lora-rank", type=int, default=16)
    tr.add_argument("--eval-every", type=int, default=500)
    tr.add_argument("--log-every", type=int, default=20)
    tr.add_argument("--val-per-seed", type=int, default=64)
    tr.add_argument("--val-batches", type=int, default=8)
    tr.add_argument("--overfit", action="store_true")
    tr.add_argument("--decoded-aux", action="store_true")
    tr.add_argument("--aux-weight", type=float, default=1.0)
    ev = sub.add_parser("evaluate", parents=[common])
    ev.add_argument("--split", choices=("val", "test"), required=True)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--per-seed", type=int, default=256)
    ev.add_argument("--eval-batch-size", type=int, default=4)
    ev.add_argument("--ddim-steps", type=int, default=20)
    ev.add_argument("--raw-root", type=Path)
    ev.add_argument("--lora-rank", type=int, default=16)
    ev.add_argument("--action-sensitivity", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if hasattr(args, "raw_root") and args.raw_root is None:
        args.raw_root = Path(args.data_root).resolve().parent
    if args.command == "inspect-sequences":
        inspect_sequences(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "check-load":
        check_load(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
