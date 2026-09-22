"""Causal history-to-future latent prior followed by the existing diffusion UNet.

WorldVLN motivates the history -> predicted future latent -> downstream model
ordering. This LiDAR experiment does not use WorldVLN's RGB tokenizer, language
instruction, action decoder, or weights. Future actions enter only the frozen
NavRL diffusion UNet. All reported generated frames pass through DDIM.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.epona_history import LidarHistoryMST
from lidar_wam.runner.lidar_geometry import load_rays
from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch
from try_epona_mst_history import FiveHistoryDataset


def latent_errors(model, data, batch_size):
    model.eval()
    squared = copied = elements = 0.0
    with torch.no_grad():
        for history, past, target, _, _ in DataLoader(data, batch_size=batch_size):
            history, past, target = (x.to(stage1.DEVICE) for x in (history, past, target))
            predicted = model(history, past)
            squared += F.mse_loss(predicted, target, reduction="sum").item()
            copied += F.mse_loss(history[:, -1], target, reduction="sum").item()
            elements += target.numel()
    return {"predicted_mse": squared / elements, "copy_mse": copied / elements}


def train_prior(train, validation, args):
    stage1.seed_everything(args.seed)
    model = LidarHistoryMST(width=args.width, blocks=args.blocks).to(stage1.DEVICE).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loader = stage1.infinite(DataLoader(train, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True))
    path = args.out / "future_latent_best.pt"
    best = float("inf")
    records = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        history, past, target, _, _ = (
            x.to(stage1.DEVICE) for x in next(loader))
        predicted = model(history, past)
        loss = F.mse_loss(predicted, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite latent loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "train_latent_mse": loss.item(),
                              "elapsed_min": round((time.monotonic() - started) / 60, 2)}),
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            score = latent_errors(model, validation, args.eval_batch_size)
            records.append({"step": step, **score})
            if score["predicted_mse"] < best:
                best = score["predicted_mse"]
                temporary = path.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "step": step,
                            "width": args.width, "blocks": args.blocks,
                            "validation": score}, temporary)
                temporary.replace(path)
            stage1.save_json(args.out / "training.json", {"records": records,
                             "best_validation_latent_mse": best})
            print(json.dumps({"step": step, "validation": score,
                              "best_validation_latent_mse": best}), flush=True)
    selected = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    return model.eval(), selected


@torch.no_grad()
def evaluate(split, data, prior, unet, vae, strengths, args, scale, threshold):
    with h5py.File(args.data_root / f"navrl_static_{split}.h5", "r") as h5:
        target_images = fetch(h5, "range_values", data.source)
    rays = {int(seed): load_rays(args.raw_root, split, int(seed))[0]
            for seed in sorted(set(data.seeds))}
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    rows = [{"source_index": int(source), "seed": int(seed),
             "history_source_indices": [int(v) for v in hist]}
            for source, seed, hist in zip(data.source, data.seeds,
                                          data.history_source)]
    methods = ["pure_noise"]
    for strength in strengths:
        methods.extend((f"copy_init_{strength:g}", f"learned_init_{strength:g}"))
    for start in range(0, len(data), args.eval_batch_size):
        stop = min(start + args.eval_batch_size, len(data))
        history = data.history[start:stop].to(stage1.DEVICE)
        past = data.past_actions[start:stop].to(stage1.DEVICE)
        actions = data.future_actions[start:stop].to(stage1.DEVICE)
        state = data.state[start:stop].to(stage1.DEVICE)
        previous = history[:, -1]
        learned = prior(history, past)
        generated = {}
        for method in methods:
            if method == "pure_noise":
                strength, initial = 1.0, None
            else:
                strength = float(method.rsplit("_", 1)[1])
                initial = previous if method.startswith("copy_") else learned
            output = stage1.generate(unet, scheduler, previous, actions, state,
                                     seed=args.seed + start, init_strength=strength,
                                     num_steps=args.ddim_steps, initial_latent=initial)
            generated[method] = vae.decode(output / scale).sample.cpu().numpy()
        for local, index in enumerate(range(start, stop)):
            target = target_images[index]
            ray = rays[int(data.seeds[index])]
            for method in methods:
                rows[index][method] = lidar_metric(generated[method][local], target,
                                                   ray, threshold)
        if stop % max(args.eval_batch_size * 8, 1) == 0 or stop == len(data):
            print(json.dumps({"split": split, "evaluated": stop,
                              "total": len(data)}), flush=True)
    summaries = {}
    for group_name, group in (("all", rows), *(
            (f"seed_{seed}", [r for r in rows if r["seed"] == seed])
            for seed in sorted(set(data.seeds)))):
        paired = [r for r in group if all(not r[m]["empty_cloud"] for m in methods)]
        summaries[group_name] = {
            "samples": len(group), "paired_nonempty": len(paired),
            "paired_cd_paper_m2": {m: float(np.mean([r[m]["cd_paper_m2"]
                                                    for r in paired])) for m in methods},
            "all_samples": {m: summarize([r[m] for r in group]) for m in methods}}
    return {"split": split, "methods": methods, "strengths": strengths,
            "latent_mse": latent_errors(prior, data, args.eval_batch_size),
            "summary": summaries, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=stage1.OUT / "worldvln_latent_prior")
    p.add_argument("--checkpoint", type=Path, default=stage1.OUT /
                   "world_circular_causal_8h" / "best.pt")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-per-seed", type=int, default=64)
    p.add_argument("--test-per-seed", type=int, default=64)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--ddim-steps", type=int, default=20)
    p.add_argument("--strengths", type=float, nargs="+", default=[0.35, 0.6, 0.8])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if any(not 0 < s < 1 for s in args.strengths):
        p.error("strengths must be in (0,1)")
    args.out.mkdir(parents=True, exist_ok=True)
    stage1.DATA = args.data_root
    train = FiveHistoryDataset("train", args.data_root)
    val = FiveHistoryDataset("val", args.data_root, args.val_per_seed)
    test = FiveHistoryDataset("test", args.data_root, args.test_per_seed)
    for split, data in (("val", val), ("test", test)):
        stage1.save_json(args.out / f"{split}_manifest.json", [{
            "source_index": int(s), "seed": int(seed),
            "history_source_indices": [int(v) for v in hist]}
            for s, seed, hist in zip(data.source, data.seeds, data.history_source)])
    meta = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())
    stage1.save_json(args.out / "config.json", {
        "train_examples": len(train), "validation_examples": len(val),
        "test_examples": len(test), "seed": args.seed,
        "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.lr, "history_frames": 5,
        "future_actions_in_prior": False, "future_actions_in_diffusion": True,
        "ddim_steps": args.ddim_steps, "strengths": args.strengths,
        "vae_sha256": meta["vae_sha256"], "unet_checkpoint": str(args.checkpoint),
        "selection": "best validation latent MSE, then lowest paired validation Chamfer among learned strengths"})
    prior, selected = train_prior(train, val, args)
    unet = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    unet_checkpoint = stage1.load_model(args.checkpoint, unet)
    vae = stage1.load_circular_vae()
    scale = meta["scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    val_results = evaluate("val", val, prior, unet, vae, args.strengths,
                           args, scale, threshold)
    stage1.save_json(args.out / "validation.json", val_results)
    paired = val_results["summary"]["all"]["paired_cd_paper_m2"]
    chosen_strength = min(args.strengths, key=lambda s: paired[f"learned_init_{s:g}"])
    stage1.save_json(args.out / "selection.json", {
        "prior_step": selected["step"], "unet_step": unet_checkpoint["step"],
        "strength": chosen_strength, "validation_paired_cd_paper_m2": paired})
    print(json.dumps({"selected_prior_step": selected["step"],
                      "selected_strength": chosen_strength,
                      "validation_paired_cd_paper_m2": paired}), flush=True)
    test_results = evaluate("test", test, prior, unet, vae, [chosen_strength],
                            args, scale, threshold)
    stage1.save_json(args.out / "test.json", test_results)
    print(json.dumps({"test_paired_cd_paper_m2":
                      test_results["summary"]["all"]["paired_cd_paper_m2"],
                      "test_paired_n": test_results["summary"]["all"]["paired_nonempty"]}),
          flush=True)


if __name__ == "__main__":
    main()
