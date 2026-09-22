"""Train and evaluate NWM CDiT on causal NavRL LiDAR transitions."""

import argparse
from contextlib import nullcontext
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_wam.runner import stage1
from lidar_wam.runner.executed_residual import ExecutedLatents, condition_stats, latent_metadata
from lidar_wam.runner.nwm_predictor import NavRLCDiT, model_kwargs
from third_party.nwm.diffusion import create_diffusion


RUN_NAME = "world_nwm_cdit"
VERSION = "nwm_cdit_3f6cd8e_navrl_executed_v1"


def run_dir(args):
    return stage1.OUT / (RUN_NAME + ("_overfit" if args.overfit else "_full"))


def make_model(args):
    return NavRLCDiT(args.model).to(stage1.DEVICE).float()


def one_loss(model, diffusion, batch, precision="fp32"):
    previous, target, actions, state = (v.to(stage1.DEVICE) for v in batch)
    target = NavRLCDiT.pad_latent(target)
    t = torch.randint(diffusion.num_timesteps, (len(target),),
                      device=target.device, dtype=torch.long)
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if precision == "bf16" else nullcontext())
    with autocast:
        losses = diffusion.training_losses(
            model, target, t, model_kwargs(previous, actions, state))
    return losses["loss"].mean(), {k: v.detach().mean().item() for k, v in losses.items()}


@torch.no_grad()
def validate(model, diffusion, dataset, precision="fp32"):
    model.eval()
    values = []
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]
                               if stage1.DEVICE.type == "cuda" else []):
        torch.manual_seed(2917)
        for batch in loader:
            _, parts = one_loss(model, diffusion, batch, precision)
            values.append(parts)
    return {key: float(np.mean([v[key] for v in values])) for key in values[0]}


def train(args):
    stats = condition_stats()
    dataset = ExecutedLatents("train", overfit=args.overfit)
    validation = (ExecutedLatents("train", overfit=True) if args.overfit else
                  ExecutedLatents("val", limit=256))
    output = run_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    meta = {"version": VERSION, "upstream_commit": "3f6cd8e70d6f2d1e2b9684acff510710135f0f41",
            "model": args.model, "vae_sha256": stats["vae_sha256"],
            "prediction": "next scaled VAE latent", "actions": "ten executed 3D velocity commands",
            "state": "previous drone state [2:13]", "time_horizon_s": 0.16,
            "train_examples": len(dataset), "validation_examples": len(validation),
            "diffusion_steps": 1000, "lr": args.lr, "batch_size": args.batch_size,
            "precision": args.precision}
    config = output / "config.json"
    if args.resume:
        existing = json.loads(config.read_text())
        for key in ("version", "model", "vae_sha256", "precision"):
            if existing[key] != meta[key]:
                raise ValueError(f"Cannot resume: {key} differs")
    else:
        stage1.save_json(config, meta)
    stage1.seed_everything(args.seed)
    model = make_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    diffusion = create_diffusion("")
    start, best = 1, math.inf
    if args.resume:
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"] + 1
        best = saved.get("best_loss", math.inf)
    loader = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             drop_last=True, num_workers=0))
    for step in range(start, args.steps + 1):
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, drop_last=True, num_workers=0))
            batch = next(loader)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = one_loss(model, diffusion, batch, args.precision)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite NWM loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "train": parts,
                              "grad_norm": float(grad)}), flush=True)
        if step == args.steps or step % args.eval_every == 0:
            metrics = validate(model, diffusion, validation, args.precision)
            print(json.dumps({"step": step, "validation": metrics}), flush=True)
            is_best = metrics["loss"] < best
            best = min(best, metrics["loss"])
            payload = {"step": step, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "best_loss": best,
                       "validation": metrics, "config": meta}
            torch.save(payload, output / "latest.pt")
            if is_best:
                torch.save(payload, output / "best.pt")
                stage1.save_json(output / "best_metrics.json",
                                 {"step": step, "validation": metrics})


@torch.no_grad()
def evaluate(args):
    stage1.seed_everything(args.seed)
    output = run_dir(args)
    config = json.loads((output / "config.json").read_text())
    if config["vae_sha256"] != latent_metadata()["vae_sha256"]:
        raise ValueError("NWM checkpoint and circular VAE differ")
    args.model = config["model"]
    data = (ExecutedLatents("train", overfit=True) if args.split == "train" else
            ExecutedLatents(args.split, limit=args.samples))
    model = make_model(args).eval()
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    vae = stage1.load_circular_vae()
    vae_scale = latent_metadata()["scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    diffusion = create_diffusion(f"ddim{args.ddim_steps}")
    with h5py.File(stage1.DATA / f"navrl_static_{args.split}.h5", "r") as h5:
        previous_images = h5["prev_range_values"][data.indices]
        target_images = h5["range_values"][data.indices]
    rows = []
    rng = np.random.default_rng(args.seed)
    for seed in sorted(stage1.EXPECTED_SEEDS[args.split]):
        positions = np.flatnonzero(data.seeds == seed)
        if args.split == "train":
            positions = positions[:args.samples]
        if not len(positions):
            continue
        shuffled = None
        if len(positions) > 1:
            permutation = rng.permutation(len(positions))
            while np.any(permutation == np.arange(len(positions))):
                permutation = rng.permutation(len(positions))
            shuffled = positions[permutation]
        for start in range(0, len(positions), args.batch_size):
            selected = positions[start:start + args.batch_size]
            previous = data.previous[selected].to(stage1.DEVICE)
            actions = data.actions[selected].to(stage1.DEVICE)
            state = data.state[selected].to(stage1.DEVICE)
            noise = torch.randn_like(NavRLCDiT.pad_latent(previous))
            def generate(a):
                result = diffusion.ddim_sample_loop(
                    model, noise.shape, noise=noise.clone(), clip_denoised=False,
                    model_kwargs=model_kwargs(previous, a, state),
                    device=stage1.DEVICE, progress=False)
                latent = NavRLCDiT.crop_latent(result)
                return vae.decode(latent / vae_scale).sample.cpu().numpy()
            prediction = generate(actions)
            shuffled_prediction = (generate(
                data.actions[shuffled[start:start + len(selected)]].to(stage1.DEVICE))
                if shuffled is not None else None)
            for j, index in enumerate(selected):
                truth = stage1.to_points(target_images[index])
                rows.append({"source_index": int(data.indices[index]),
                             "terrain_seed": int(seed),
                             "copy_chamfer_m": stage1.chamfer(
                                 stage1.to_points(previous_images[index]), truth),
                             "prediction_chamfer_m": stage1.chamfer(
                                 stage1.to_points(prediction[j], threshold), truth),
                             "shuffled_action_chamfer_m": (
                                 stage1.chamfer(stage1.to_points(
                                     shuffled_prediction[j], threshold), truth)
                                 if shuffled_prediction is not None else None)})
            if start == 0:
                stage1.save_preview(output / f"{args.split}_step{saved['step']}_seed{seed}.png",
                                    previous_images[selected[0]], target_images[selected[0]],
                                    prediction[0], threshold)
    means = {}
    for key in ("copy_chamfer_m", "prediction_chamfer_m", "shuffled_action_chamfer_m"):
        values = [row[key] for row in rows if row[key] is not None]
        means[key] = float(np.mean(values)) if values else None
    means["prediction_improvement"] = 1 - means["prediction_chamfer_m"] / means["copy_chamfer_m"]
    means["shuffle_degradation"] = (means["shuffled_action_chamfer_m"] /
        means["prediction_chamfer_m"] - 1 if means["shuffled_action_chamfer_m"] is not None else None)
    report = {"split": args.split, "step": saved["step"], "samples": len(rows),
              "ddim_steps": args.ddim_steps, "summary": means, "rows": rows}
    path = output / f"evaluation_{args.split}_step{saved['step']}_n{len(rows)}.json"
    stage1.save_json(path, report)
    print(json.dumps({"report": str(path), **means}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--model", choices=("CDiT-S/2", "CDiT-B/2", "CDiT-L/2", "CDiT-XL/2"),
                        default="CDiT-B/2")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--ddim-steps", type=int, default=20)
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
