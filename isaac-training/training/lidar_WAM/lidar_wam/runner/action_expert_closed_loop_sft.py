"""Four-GPU DAgger SFT: exact online first-action labels + offline 10-step BC.

Frozen: VAE (cache only), observation CNN, flow branch, policy input encoders,
queries and first six blocks. Trainable: last two policy blocks, norm/output.
Uses the baseline's fixed normalization, coordinate frame and action scale.
No future/privileged features are model inputs. Saved format remains deployable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from lidar_wam.models.action_expert import SuperActionExpertModel
from lidar_wam.runner.super_action_expert import (
    _load_frozen_vae, _encode_current, _normalise, _atomic_save, _autocast)
from lidar_wam.runner.super_lee_action_data import SuperLeeActionDataset, discover_shard_names


def build_model(payload, *, freeze_policy_output_head=False):
    arch = payload["architecture"]
    from lidar_wam.models.flow_action_expert import FLOW_ACTION_EXPERT_FORMAT, FlowActionExpertModel
    if payload.get("format") == FLOW_ACTION_EXPERT_FORMAT:
        if (payload["action_interface"].get("normalized_output_constraint") != "unit_box_after_euler"
                or list(payload["action_interface"]["model_output_scale_mps_xyz"]) != [2., 2., 2.]):
            raise ValueError("Flow SFT must retain the per-axis +/-2 m/s deployment contract")
        model = FlowActionExpertModel(
            width=arch["width"], depth=arch["depth"], heads=arch["heads"],
            ffn_width=arch["ffn_width"], horizon=arch["action_horizon"],
            action_dim=arch["action_dim"], proprio_dim=arch["proprio_dim"],
            sample_steps=int(arch.get("sampling_steps", 10)))
        if len(model.diffusion.blocks) != 8 or model.sample_steps != 10:
            raise ValueError("Online Flow SFT requires the baseline's eight blocks / ten Euler steps")
        model.load_state_dict(payload["model"], strict=True)
        model.requires_grad_(False)
        for block in model.diffusion.blocks[-2:]:
            block.requires_grad_(True)
        model.diffusion.norm.requires_grad_(not freeze_policy_output_head)
        model.diffusion.out.requires_grad_(not freeze_policy_output_head)
        return model
    model = SuperActionExpertModel(
        width=arch["width"], depth=arch["depth"], heads=arch["heads"],
        ffn_width=arch["ffn_width"], horizon=arch["action_horizon"],
        action_dim=arch["action_dim"], proprio_dim=arch["proprio_dim"],
        past_horizon=arch["past_horizon"], lidar_history_frames=1)
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    for block in model.policy.blocks[-2:]:
        block.requires_grad_(True)
    model.policy.norm.requires_grad_(not freeze_policy_output_head)
    model.policy.out.requires_grad_(not freeze_policy_output_head)
    return model


class DirectPolicy(nn.Module):
    def __init__(self, model, project_output=True):
        super().__init__()
        self.expert = model
        self.project_output = bool(project_output)

    def forward(self, latent, goal, proprio):
        n = len(latent)
        return self.expert.predict_action(
            latent, goal, proprio, latent.new_zeros(n, 30, 3),
            latent.new_zeros(n, 30), project_output=self.project_output)


def prepare(args):
    if args.cache.exists():
        raise FileExistsError(args.cache)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    resolved_config = json.loads((args.checkpoint.parent/"config.json").read_text())
    validation_shards = resolved_config["validation_shards"]
    if not validation_shards:
        raise ValueError("Baseline validation-shard list is empty")
    train_shards = [name for name in discover_shard_names(args.data_root)
                    if name not in validation_shards]
    device = torch.device("cuda:0")
    vae, scale = _load_frozen_vae(args.vae_checkpoint, device)
    if hashlib.sha256(args.vae_checkpoint.read_bytes()).hexdigest() != payload["vae"]["sha256"]:
        raise ValueError("VAE weights do not match the baseline SHA256")
    if abs(scale - float(payload["vae"]["scaling_factor"])) > 1e-6:
        raise ValueError("VAE scale does not match baseline")
    stats = payload["condition_stats"]
    output = {}
    for split, count in (("train", args.cache_windows), ("val", args.val_windows)):
        shards = train_shards if split == "train" else validation_shards
        dataset = SuperLeeActionDataset(args.data_root, shards, max_windows=count,
                                       random_seed=args.seed + (split == "val"))
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
        parts = {key: [] for key in ("latent", "goal", "proprio", "target")}
        for index, batch in enumerate(loader):
            image, target, goal, proprio = batch[:4]
            latent = _encode_current(vae, image, scale, device, "bf16")
            values = {"latent": latent.cpu().half(),
                      "target": target / torch.tensor([2., 2., 2.]),
                      "goal": _normalise(goal, stats, "goal", "cpu"),
                      "proprio": _normalise(proprio, stats, "proprio", "cpu")}
            for key in parts:
                parts[key].append(values[key])
            if index % 25 == 0:
                print(f"[cache] {split} {min((index+1)*128, len(dataset))}/{len(dataset)}", flush=True)
        output[split] = {key: torch.cat(value) for key, value in parts.items()}
        dataset.close()
    output["metadata"] = {"checkpoint": str(args.checkpoint), "vae": str(args.vae_checkpoint),
                          "vae_sha256": payload["vae"]["sha256"],
                          "scaling_factor": scale, "past_input": "zero (same as deployment)",
                          "train_samples": len(output["train"]["target"]),
                          "val_samples": len(output["val"]["target"])}
    _atomic_save(output, args.cache)
    print(json.dumps(output["metadata"]), flush=True)


def draw(data, count, generator, device):
    indices = torch.randint(len(data["target"]), (count,), generator=generator)
    return {key: data[key][indices].to(device).float()
            for key in ("latent", "goal", "proprio", "target")}


def supervised_loss(prediction, target):
    """Keep baseline's first-step/hard/temporal emphasis for offline BC."""
    weights = prediction.new_ones(10)
    weights[0] = 4.
    weights /= weights.mean()
    error = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none", beta=.25)
    hard = (target[..., 1:].abs().amax(dim=(1, 2)) * 2 >= .5).float()
    sample_weight = 1. + hard
    sample_weight /= sample_weight.mean()
    direct = ((error * weights[None, :, None]).mean((1, 2))*sample_weight).mean()
    delta = F.smooth_l1_loss(prediction[:, 1:]-prediction[:, :-1],
                            target[:, 1:]-target[:, :-1], beta=.25)
    return direct + .05*delta


@torch.no_grad()
def validate(model, data, device):
    model.eval()
    sums = torch.zeros(3, device=device)
    for start in range(0, len(data["target"]), 64):
        batch = {key: value[start:start+64].to(device).float() for key, value in data.items()}
        with _autocast(device, "bf16"):
            prediction = model(batch["latent"], batch["goal"], batch["proprio"]).float()
        n = len(prediction)
        sums += torch.stack(((prediction-batch["target"]).abs().mean()*n*2,
                             (prediction[:, 0]-batch["target"][:, 0]).abs().mean()*n*2,
                             prediction.new_tensor(n)))
    model.train()
    return {"offline_chunk_mae_mps": float(sums[0]/sums[2]),
            "offline_first_mae_mps": float(sums[1]/sums[2])}


def train(args):
    local = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed + rank)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expert = build_model(payload).to(device)
    model = DirectPolicy(expert).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local], broadcast_buffers=False)
    parameters = [p for p in expert.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01, betas=(.9, .95))
    if "closed_loop_sft_optimizer" in payload:
        optimizer.load_state_dict(payload["closed_loop_sft_optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    start_step = int(payload.get("closed_loop_sft", {}).get("step", 0))
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if abs(float(cache["metadata"]["scaling_factor"])-float(payload["vae"]["scaling_factor"])) > 1e-6:
        raise ValueError("Offline cache VAE scale differs from student checkpoint")
    offline = cache["train"]
    sources = [torch.load(path, map_location="cpu", weights_only=False) for path in args.online]
    online = {key: torch.cat([source[key] for source in sources])
              for key in ("latent", "goal", "proprio", "target")}
    generator = torch.Generator().manual_seed(args.seed + rank*100003 + start_step)
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        args.out.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.out/"tensorboard")
        writer.add_text("config/freeze", "VAE/CNN/flow/input/blocks1-6 frozen; blocks7-8/norm/output trainable", start_step)
        writer.add_scalar("replay/online_samples", len(online["target"]), start_step)
        writer.add_scalar("replay/offline_samples", len(offline["target"]), start_step)
        info = {"world_size": world, "trainable_parameters": sum(p.numel() for p in parameters),
                "per_gpu_offline_batch": args.batch_size, "per_gpu_online_batch": args.batch_size,
                "lr": args.lr, "online_weight": args.online_weight, "start_step": start_step,
                "online_samples": len(online["target"])}
        print(json.dumps(info), flush=True)
        (args.out/f"config_round_{start_step}.json").write_text(json.dumps(info, indent=2)+"\n")
        initial = validate(model.module if world > 1 else model, cache["val"], device)
        for key, value in initial.items():
            writer.add_scalar("validation/"+key, value, start_step)
        print(json.dumps({"step": start_step, "validation": initial}), flush=True)
    if world > 1:
        dist.barrier()
    model.train()
    started = time.time()
    for update in range(1, args.steps+1):
        a = draw(offline, args.batch_size, generator, device)
        b = draw(online, args.batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, "bf16"):
            prediction = model(*(torch.cat((a[key], b[key])) for key in ("latent", "goal", "proprio")))
            offline_loss = supervised_loss(prediction[:args.batch_size], a["target"])
            online_loss = F.smooth_l1_loss(prediction[args.batch_size:, 0].float(),
                                         b["target"], beta=.25)
            loss = offline_loss + args.online_weight*online_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite SFT loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        step = start_step + update
        if update == 1 or update % 10 == 0:
            values = torch.stack((loss.detach(), offline_loss.detach(), online_loss.detach(),
                                 (prediction[args.batch_size:, 0].float()-b["target"]).abs().mean().detach()*2))
            if world > 1:
                dist.all_reduce(values)
                values /= world
            if rank == 0:
                row = dict(zip(("loss", "offline_bc", "online_first_bc", "online_first_mae_mps"), values.tolist()))
                row.update(grad_norm=float(grad), lr=args.lr,
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           elapsed_s=time.time()-started)
                print(json.dumps({"step": step, **row}), flush=True)
                for key, value in row.items():
                    writer.add_scalar("train/"+key, value, step)
                writer.flush()
    if rank == 0:
        metrics = validate(model.module if world > 1 else model, cache["val"], device)
        print(json.dumps({"step": step, "validation": metrics}), flush=True)
        for key, value in metrics.items():
            writer.add_scalar("validation/"+key, value, step)
        saved = {key: value for key, value in payload.items() if key not in ("optimizer", "model", "closed_loop_sft_optimizer")}
        saved.update(model=expert.state_dict(), step=step,
                     architecture={**payload["architecture"], "training_action_diffusion": False},
                     closed_loop_sft_optimizer=optimizer.state_dict(),
                     closed_loop_sft={"step": step, "source_checkpoint": str(args.checkpoint),
                                      "lr": args.lr, "online_weight": args.online_weight,
                                      "first_action_only": True, "execution_k": 10,
                                      "replay_files": [str(p) for p in args.online], "metrics": metrics})
        _atomic_save(saved, args.out/f"sft_step_{step}.pt")
        _atomic_save(saved, args.out/"latest.pt")
        writer.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "smoke"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--cache-windows", type=int, default=32768)
    parser.add_argument("--val-windows", type=int, default=4096)
    parser.add_argument("--online", type=Path, nargs="+", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--online-weight", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.command == "smoke":
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        expert = build_model(payload).cuda()
        policy = DirectPolicy(expert)
        prediction = policy(torch.randn(2, 4, 27, 5, device="cuda"),
                            torch.randn(2, 4, device="cuda"),
                            torch.randn(2, 10, device="cuda"))
        prediction.square().mean().backward()
        assert prediction.shape == (2, 10, 3)
        assert expert.policy.out.weight.grad.norm() > 0
        assert all(p.grad is None for p in expert.observation.parameters())
        assert all(p.grad is None for p in expert.policy.blocks[0].parameters())
        print("Freeze/gradient checks passed; trainable parameters:",
              sum(p.numel() for p in expert.parameters() if p.requires_grad), flush=True)
    elif args.command == "prepare":
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
