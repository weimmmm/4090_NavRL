"""Five observed LiDAR frames condition the frozen executed-action residual UNet.

This adapts the already-vendored Epona MST block, not a new generator. The
residual UNet and circular VAE remain frozen. It predicts the next frame after
the recorded next 10 actions, never using a future observation or future pose.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch
from lidar_wam.runner import stage1
from lidar_wam.runner.epona_history import LidarHistoryMST
from lidar_wam.runner.executed_residual import condition_stats
from lidar_wam.runner.lidar_geometry import load_rays
from try_epona_mst_history import FiveHistoryDataset


class ResidualFiveHistory(Dataset):
    def __init__(self, split, data_root, stats, per_seed=None):
        base = FiveHistoryDataset(split, data_root)
        with h5py.File(data_root / f"navrl_static_{split}.h5", "r") as h5:
            actions = fetch(h5, "action_sequence", base.source)
            state = fetch(h5, "prev_drone_state", base.source)[:, 2:13]
        finite = np.isfinite(actions).all(axis=(1, 2)) & np.isfinite(state).all(axis=1)
        selected = np.flatnonzero(finite)
        if per_seed is not None:
            rng = np.random.default_rng(42)
            selected = np.concatenate([rng.choice(selected[base.seeds[selected] == seed],
                                                  size=per_seed, replace=False)
                                       for seed in sorted(stage1.EXPECTED_SEEDS[split])])
        self.source = base.source[selected].copy()
        self.seeds = base.seeds[selected].copy()
        self.history_source = base.history_source[selected].copy()
        self.history = base.history[selected].clone()
        self.past = base.past_actions[selected].clone()
        self.previous = base.previous[selected].clone()
        self.target = base.target[selected].clone()
        self.future = torch.from_numpy(((actions[selected] - stats["action_mean"]) /
                                        stats["action_std"]).astype(np.float32))
        self.state = torch.from_numpy(((state[selected] - stats["state_mean"]) /
                                       stats["state_std"]).astype(np.float32))
        if not all(torch.isfinite(x).all() for x in (
                self.history, self.past, self.previous, self.target,
                self.future, self.state)):
            raise ValueError("Nonfinite residual-history input")

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        return (self.history[i], self.past[i], self.previous[i],
                self.target[i], self.future[i], self.state[i])


@torch.no_grad()
def sample(world, scheduler, previous, condition, actions, state, scale, seed):
    # Exact residual-DDIM convention used by stage1.generate at strength 0.05,
    # except the UNet conditioning latent may come from five observed frames.
    torch.manual_seed(seed)
    scheduler.set_timesteps(400, device=stage1.DEVICE)
    timesteps = scheduler.timesteps[-20:]
    residual = scheduler.add_noise(torch.zeros_like(previous),
                                   torch.randn_like(previous),
                                   timesteps[0].repeat(len(previous)))
    for timestep in timesteps:
        predicted = world(residual, condition, actions, state, timestep)
        residual = scheduler.step(predicted, timestep, residual, eta=0.0).prev_sample
    return previous + residual / scale


@torch.no_grad()
def evaluate(world, adapter, vae, data, split, args, stats, threshold, scale):
    world.eval()
    adapter.eval()
    with h5py.File(args.data_root / f"navrl_static_{split}.h5", "r") as h5:
        frames = fetch(h5, "range_values", data.source)
        previous_frames = fetch(h5, "prev_range_values", data.source)
    rays = {int(seed): load_rays(args.raw_root, split, int(seed))[0]
            for seed in sorted(set(data.seeds))}
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    rows = [{"source_index": int(index), "seed": int(seed),
             "history_source_indices": [int(x) for x in history]}
            for index, seed, history in zip(data.source, data.seeds, data.history_source)]
    for start in range(0, len(data), args.eval_batch_size):
        stop = min(start + args.eval_batch_size, len(data))
        history, past, previous, _, actions, state = (
            x.to(stage1.DEVICE) for x in (
                data.history[start:stop], data.past[start:stop],
                data.previous[start:stop], data.target[start:stop],
                data.future[start:stop], data.state[start:stop]))
        condition = adapter(history, past)
        for name, cond in (("residual_single", previous),
                           ("residual_five_history", condition)):
            latent = sample(world, scheduler, previous, cond, actions, state,
                            stats["residual_scale"], seed=42 + start)
            prediction = vae.decode(latent / scale).sample.cpu().numpy()
            for local, position in enumerate(range(start, stop)):
                rows[position][name] = lidar_metric(
                    prediction[local], frames[position], rays[int(data.seeds[position])],
                    threshold)
        for position in range(start, stop):
            rows[position]["copy"] = lidar_metric(
                previous_frames[position], frames[position],
                rays[int(data.seeds[position])], 0)
    methods = ("copy", "residual_single", "residual_five_history")
    groups = {"all": rows}
    groups.update({f"seed_{seed}": [r for r in rows if r["seed"] == seed]
                   for seed in sorted(set(data.seeds))})
    summary = {}
    for name, group in groups.items():
        paired = [r for r in group if all(not r[m]["empty_cloud"] for m in methods)]
        summary[name] = {"samples": len(group), "paired_nonempty": len(paired),
                         "paired_cd_paper_m2": {m: float(np.mean([
                             r[m]["cd_paper_m2"] for r in paired])) for m in methods},
                         "all_samples": {m: summarize([r[m] for r in group])
                                         for m in methods}}
    return {"split": split, "summary": summary, "rows": rows}


def train(world, adapter, train_data, val_data, args, stats, vae, scale, threshold):
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    scheduler = stage1.DDPMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon")
    loader = stage1.infinite(DataLoader(train_data, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True))
    history = []
    best = float("inf")
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        adapter.train()
        hist, past, previous, target, future, state = (
            tensor.to(stage1.DEVICE) for tensor in next(loader))
        clean = (target - previous) * stats["residual_scale"]
        noise = torch.randn_like(clean)
        timestep = torch.randint(0, 1000, (len(clean),),
                                 device=stage1.DEVICE, dtype=torch.long)
        noisy = scheduler.add_noise(clean, noise, timestep)
        optimizer.zero_grad(set_to_none=True)
        condition = adapter(hist, past)
        prediction = world(noisy, condition, future, state, timestep)
        loss = F.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at {step}")
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step, "noise_mse": loss.item(),
                              "minutes": round((time.monotonic()-started)/60, 2)}),
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(world, adapter, vae, val_data, "val", args,
                              stats, threshold, scale)
            cd = result["summary"]["all"]["paired_cd_paper_m2"]
            history.append({"step": step, **cd})
            print(json.dumps({"step": step, "validation_cd_paper_m2": cd}), flush=True)
            stage1.save_json(args.out / "validation_latest.json", result)
            if cd["residual_five_history"] < best:
                best = cd["residual_five_history"]
                path = args.out / "best.pt"
                temp = path.with_suffix(".tmp")
                torch.save({"model": adapter.state_dict(), "step": step,
                            "validation_cd_paper_m2": best}, temp)
                temp.replace(path)
                stage1.save_json(args.out / "validation_best.json", result)
            stage1.save_json(args.out / "history.json", history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=stage1.OUT / "residual_mst_five_history")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-per-seed", type=int, default=256)
    parser.add_argument("--test-per-seed", type=int, default=256)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stage1.DATA = args.data_root
    stage1.seed_everything(42)
    stats = condition_stats()
    train_data = ResidualFiveHistory("train", args.data_root, stats)
    val_data = ResidualFiveHistory("val", args.data_root, stats, args.val_per_seed)
    test_data = ResidualFiveHistory("test", args.data_root, stats, args.test_per_seed)
    for split, data in (("val", val_data), ("test", test_data)):
        stage1.save_json(args.out / f"{split}_manifest.json", [{
            "source_index": int(s), "seed": int(seed),
            "history_source_indices": [int(v) for v in hist]}
            for s, seed, hist in zip(data.source, data.seeds, data.history_source)])
    checkpoint_path = stage1.OUT / "world_circular_executed_residual_full" / "best.pt"
    world = stage1.WorldModel(state_dim=11).to(stage1.DEVICE).float().eval()
    checkpoint = stage1.load_model(checkpoint_path, world)
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    vae = stage1.load_circular_vae()
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR /
                        "metadata.json").read_text())["scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    stage1.save_json(args.out / "config.json", {
        "train_examples": len(train_data), "val_examples": len(val_data),
        "test_examples": len(test_data), "frozen_residual_checkpoint_step": checkpoint["step"],
        "vae_sha256": stats["vae_sha256"], "history_frames": 5,
        "history_actions": "normalized PPO actions for already observed intervals",
        "future_actions": "finite executed world-frame commands for next 10 steps",
        "future_pose_input": False, "selection": "validation paired decoded squared Chamfer",
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "residual_init_strength": 0.05, "ddim_steps": 20})
    adapter = LidarHistoryMST().to(stage1.DEVICE)
    train(world, adapter, train_data, val_data, args, stats, vae, scale, threshold)
    selected = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
    adapter.load_state_dict(selected["model"])
    test = evaluate(world, adapter, vae, test_data, "test", args,
                    stats, threshold, scale)
    stage1.save_json(args.out / "test.json", test)
    stage1.save_json(args.out / "selection.json", {
        "step": selected["step"],
        "validation_cd_paper_m2": selected["validation_cd_paper_m2"],
        "test_paired_cd_paper_m2": test["summary"]["all"]["paired_cd_paper_m2"]})
    print(json.dumps({"selected_step": selected["step"],
                      "test": test["summary"]["all"]["paired_cd_paper_m2"]}),
          flush=True)


if __name__ == "__main__":
    main()
