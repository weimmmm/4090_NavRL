"""Two-frame generated-history fine-tuning of the existing NavRL diffusion UNet.

The first frame is sampled with the same 20-step DDIM path as inference,
without target-frame information or gradient.  The generated latent conditions
the second frame's ordinary epsilon loss.  An ordinary first-frame epsilon
loss is retained.  This is an Epona-inspired probe, not Epona's flow objective.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_geometry import load_rays
from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch, save_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "epona_probe"


class TwoFramePairs(Dataset):
    def __init__(self, split, data_root, indices=None):
        cached = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
        source = cached["source_index"].astype(np.int64)
        scale = json.loads((stage1.OUT / stage1.LATENT_DIR /
                            "metadata.json").read_text())["scaling_factor"]
        with h5py.File(data_root / f"navrl_static_{split}.h5", "r") as h5:
            token = fetch(h5, "token", source)
            previous_token = fetch(h5, "prev_token", source)
            scene = fetch(h5, "scene_token", source)
            frame = fetch(h5, "frame_idx", source)
            successor = {value: j for j, value in enumerate(previous_token)}
            first = np.array([i for i, key in enumerate(token)
                              if key in successor and
                              scene[i] == scene[successor[key]] and
                              frame[successor[key]] == frame[i] + 1], dtype=np.int64)
            second = np.array([successor[token[i]] for i in first], dtype=np.int64)
            if not len(first):
                raise ValueError(f"No linked two-frame trajectories in {split}")
            if indices is not None:
                first, second = first[indices], second[indices]
            targets = fetch(h5, "range_values", source[second]) if split == "val" else None
        self.source = source[first]
        self.next_source = source[second]
        self.seed = cached["seeds"][first]
        self.initial = torch.from_numpy(cached["previous"][first].copy() * scale)
        self.first = torch.from_numpy(cached["target"][first].copy() * scale)
        self.second = torch.from_numpy(cached["target"][second].copy() * scale)
        self.action1 = torch.from_numpy(cached["actions"][first].copy())
        self.action2 = torch.from_numpy(cached["actions"][second].copy())
        self.state = torch.from_numpy(stage1.causal_state(cached["state"][first]))
        self.target_image = targets
        if not torch.allclose(self.first[:min(64, len(first))],
                              torch.from_numpy(cached["previous"][second[:min(64, len(second))]].copy() * scale),
                              atol=1e-5):
            raise ValueError("Adjacent VAE latents do not match")

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        return (self.initial[i], self.first[i], self.second[i],
                self.action1[i], self.action2[i], self.state[i])


def choose_validation_indices(split, data_root, per_seed=32):
    all_pairs = TwoFramePairs(split, data_root)
    rng = np.random.default_rng(42)
    chosen = np.concatenate([rng.choice(np.flatnonzero(all_pairs.seed == seed),
                                        per_seed, replace=False)
                             for seed in sorted(np.unique(all_pairs.seed))])
    return chosen, all_pairs


@torch.no_grad()
def evaluate(model, data, raw_root, batch_size, seed=701):
    model.eval()
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    vae = stage1.load_circular_vae()
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR /
                        "metadata.json").read_text())["scaling_factor"]
    rays = {int(s): load_rays(raw_root, "val", int(s))[0]
            for s in np.unique(data.seed)}
    scores = []
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        z0 = data.initial[start:end].to(stage1.DEVICE)
        a1 = data.action1[start:end].to(stage1.DEVICE)
        a2 = data.action2[start:end].to(stage1.DEVICE)
        state = data.state[start:end].to(stage1.DEVICE)
        z1 = stage1.generate(model, scheduler, z0, a1, state,
                             seed + start * 100, num_steps=20)
        z2 = stage1.generate(model, scheduler, z1, a2, state,
                             seed + start * 100 + 1, num_steps=20)
        image = vae.decode(z2 / scale).sample.cpu().numpy()
        for local, idx in enumerate(range(start, end)):
            metric = lidar_metric(image[local], data.target_image[idx],
                                  rays[int(data.seed[idx])], 1.5)
            scores.append({"source_index": int(data.source[idx]),
                           "seed": int(data.seed[idx]), **metric})
    del vae
    return {"all": summarize(scores),
            **{f"seed_{s}": summarize([r for r in scores if r["seed"] == s])
               for s in sorted(np.unique(data.seed))}}


def train(args):
    stage1.DATA = args.data_root
    stage1.seed_everything(args.seed)
    train_data = TwoFramePairs("train", args.data_root)
    val_indices, _ = choose_validation_indices("val", args.data_root,
                                                per_seed=args.val_per_seed)
    val_data = TwoFramePairs("val", args.data_root, indices=val_indices)
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "chain_val_manifest.json", {
        "seed": 42, "sources": val_data.source.tolist(),
        "next_sources": val_data.next_source.tolist()})
    model = stage1.WorldModel().to(stage1.DEVICE).float()
    ckpt = stage1.load_model(args.checkpoint, model)
    print(json.dumps({"train_pairs": len(train_data), "val_pairs": len(val_data),
                      "source_checkpoint_step": ckpt["step"]}), flush=True)
    original = evaluate(model, val_data, args.raw_root, args.eval_batch_size)
    save_json(args.out / "chain_original_val.json", original)
    print(json.dumps({"original_val": original["all"]}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    noise_scheduler = stage1.DDPMScheduler(num_train_timesteps=1000,
                                           prediction_type="epsilon")
    sample_scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                            prediction_type="epsilon", clip_sample=False)
    loader = stage1.infinite(DataLoader(train_data, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True))
    history = []
    best = original["all"]["cd_paper_m2"]
    for step in range(1, args.steps + 1):
        z0, z1, z2, a1, a2, state = [v.to(stage1.DEVICE) for v in next(loader)]
        model.eval()
        generated = stage1.generate(model, sample_scheduler, z0, a1, state,
                                    seed=args.seed * 100000 + step,
                                    num_steps=args.generation_steps).detach()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t = torch.randint(0, 1000, (len(z0),), device=stage1.DEVICE,
                          dtype=torch.long)
        epsilon_first = torch.randn_like(z1)
        epsilon_second = torch.randn_like(z2)
        pred_first = model(noise_scheduler.add_noise(z1, epsilon_first, t),
                           z0, a1, state, t)
        pred_second = model(noise_scheduler.add_noise(z2, epsilon_second, t),
                            generated, a2, state, t)
        first_loss = F.mse_loss(pred_first, epsilon_first)
        second_loss = F.mse_loss(pred_second, epsilon_second)
        loss = (1 - args.generated_weight) * first_loss + \
               args.generated_weight * second_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at {step}")
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"Non-finite gradient at {step}")
        optimizer.step()
        record = {"step": step, "first_noise_mse": float(first_loss.item()),
                  "second_generated_context_noise_mse": float(second_loss.item()),
                  "grad_norm": float(gradient.item())}
        history.append(record)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            score = evaluate(model, val_data, args.raw_root, args.eval_batch_size)
            print(json.dumps({"step": step, "val": score["all"]}), flush=True)
            save_json(args.out / "chain_latest_val.json", score)
            if score["all"]["cd_paper_m2"] < best:
                best = score["all"]["cd_paper_m2"]
                torch.save({"model": model.state_dict(),
                            "source_checkpoint": str(args.checkpoint),
                            "source_step": ckpt["step"], "fine_tune_step": step},
                           args.out / "chain_best.pt")
                save_json(args.out / "chain_best_val.json", score)
            save_json(args.out / "chain_history.json", history)
    save_json(args.out / "chain_config.json", {
        "source_checkpoint": str(args.checkpoint), "source_step": ckpt["step"],
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "generation_steps": args.generation_steps, "eval_generation_steps": 20,
        "generated_weight": args.generated_weight, "seed": args.seed,
        "ego_state_rollout": "initial state fixed; isolates generated-history effect",
        "sampling": "full DDIM from random Gaussian; no target-frame leakage"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" /
                   "world_circular_causal_8h" / "best.pt")
    p.add_argument("--out", type=Path, default=RUN)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--val-per-seed", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--generated-weight", type=float, default=0.5)
    p.add_argument("--generation-steps", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
