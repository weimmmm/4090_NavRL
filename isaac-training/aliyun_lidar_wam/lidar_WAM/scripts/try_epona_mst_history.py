"""Train Epona MST history adapters for the frozen NavRL diffusion UNet.

Uses 3 or 5 *observed* prior LiDAR frames and their past command chunks. The
future ten actions enter the original UNet. Model selection uses validation
noise MSE; test maps are evaluated only after both adapters are selected.
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.epona_history import LidarHistoryMST
from lidar_wam.runner.lidar_geometry import load_rays
from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch

ROOT = Path(__file__).resolve().parents[1]


class FiveHistoryDataset(Dataset):
    """Linked episodes only; predecessor rows are already valid 10-step pairs."""

    def __init__(self, split, data_root, per_seed=None, random_seed=42):
        cached = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
        source = cached["source_index"].astype(np.int64)
        with h5py.File(data_root / f"navrl_static_{split}.h5", "r") as h5:
            token = fetch(h5, "token", source)
            previous_token = fetch(h5, "prev_token", source)
            scene = fetch(h5, "scene_token", source)
            frame = fetch(h5, "frame_idx", source)
        lookup = {key: index for index, key in enumerate(token)}
        selected, history_positions = [], []
        for current in range(len(source)):
            chain, pointer = [], current
            for _ in range(5):
                preceding = lookup.get(previous_token[pointer])
                if (preceding is None or scene[preceding] != scene[pointer]
                        or frame[preceding] + 1 != frame[pointer]):
                    break
                chain.append(preceding)
                pointer = preceding
            if len(chain) == 5:
                selected.append(current)
                history_positions.append(chain[::-1])
        selected = np.asarray(selected, dtype=np.int64)
        history_positions = np.asarray(history_positions, dtype=np.int64)
        if per_seed is not None:
            rng = np.random.default_rng(random_seed)
            seeds = cached["seeds"][selected]
            subset = np.concatenate([
                rng.choice(np.flatnonzero(seeds == s), per_seed, replace=False)
                for s in sorted(np.unique(seeds))])
            selected, history_positions = selected[subset], history_positions[subset]
        if len(selected) == 0:
            raise ValueError(f"No five-history examples in {split}")
        scale = json.loads((stage1.OUT / stage1.LATENT_DIR /
                            "metadata.json").read_text())["scaling_factor"]
        self.source = source[selected]
        self.seeds = cached["seeds"][selected].astype(np.int64)
        self.history_source = source[history_positions]
        self.history = torch.from_numpy(cached["target"][history_positions].copy() * scale)
        self.previous = torch.from_numpy(cached["previous"][selected].copy() * scale)
        self.history[:, -1] = self.previous
        self.past_actions = torch.from_numpy(cached["actions"][history_positions].copy())
        self.target = torch.from_numpy(cached["target"][selected].copy() * scale)
        self.future_actions = torch.from_numpy(cached["actions"][selected].copy())
        self.state = torch.from_numpy(stage1.causal_state(cached["state"][selected]))
        if not all(torch.isfinite(value).all() for value in (
                self.history, self.past_actions, self.target,
                self.future_actions, self.state)):
            raise ValueError("Nonfinite history inputs")

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return (self.history[index], self.past_actions[index], self.target[index],
                self.future_actions[index], self.state[index])


@torch.no_grad()
def noise_score(unet, adapter, data, frames, scheduler, batch_size=16,
                shuffle_older=False):
    if adapter is not None:
        adapter.eval()
    torch.manual_seed(2917)
    total = elements = 0
    for history, past, target, future, state in DataLoader(data, batch_size=batch_size):
        history, past, target, future, state = (
            value.to(stage1.DEVICE) for value in (history, past, target, future, state))
        if adapter is None:
            condition = history[:, -1]
        else:
            recent, old_actions = history[:, -frames:].clone(), past[:, -frames:].clone()
            if shuffle_older:
                # Keep the newest observation and all future-action inputs fixed.
                recent[:, :-1] = torch.roll(recent[:, :-1], shifts=1, dims=0)
                old_actions[:, :-1] = torch.roll(old_actions[:, :-1], shifts=1, dims=0)
            condition = adapter(recent, old_actions)
        noise = torch.randn_like(target)
        timestep = torch.randint(0, 1000, (len(target),),
                                 device=stage1.DEVICE, dtype=torch.long)
        prediction = unet(scheduler.add_noise(target, noise, timestep),
                          condition, future, state, timestep)
        total += F.mse_loss(prediction, noise, reduction="sum").item()
        elements += noise.numel()
    return total / elements


def train_adapter(unet, train, val, frames, args, run_dir):
    # Keep initialization and minibatch order identical for the 3/5-frame ablation.
    stage1.seed_everything(args.seed)
    adapter = LidarHistoryMST(width=args.width, blocks=args.blocks).to(stage1.DEVICE)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    scheduler = stage1.DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    loader = stage1.infinite(DataLoader(train, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True))
    baseline = noise_score(unet, None, val, frames, scheduler,
                           batch_size=args.eval_batch_size)
    best = float("inf")
    records = []
    path = run_dir / f"mst_{frames}_best.pt"
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        adapter.train()
        history, past, target, future, state = (
            value.to(stage1.DEVICE) for value in next(loader))
        noise = torch.randn_like(target)
        timestep = torch.randint(0, 1000, (len(target),),
                                 device=stage1.DEVICE, dtype=torch.long)
        noisy = scheduler.add_noise(target, noise, timestep)
        optimizer.zero_grad(set_to_none=True)
        condition = adapter(history[:, -frames:], past[:, -frames:])
        prediction = unet(noisy, condition, future, state, timestep)
        loss = F.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"frames": frames, "step": step,
                              "train_noise_mse": loss.item(),
                              "elapsed_min": round((time.monotonic()-started)/60, 2)}),
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            score = noise_score(unet, adapter, val, frames, scheduler,
                                batch_size=args.eval_batch_size)
            records.append({"step": step, "validation_noise_mse": score})
            if score < best:
                best = score
                temporary = path.with_suffix(".tmp")
                torch.save({"model": adapter.state_dict(), "step": step,
                            "frames": frames, "width": args.width,
                            "blocks": args.blocks, "validation_noise_mse": score},
                           temporary)
                temporary.replace(path)
            stage1.save_json(run_dir / f"mst_{frames}_training.json", {
                "baseline_noise_mse": baseline, "best_noise_mse": best,
                "best_checkpoint": str(path), "records": records,
                "train_examples": len(train), "validation_examples": len(val)})
            print(json.dumps({"frames": frames, "step": step,
                              "validation_noise_mse": score,
                              "baseline_noise_mse": baseline,
                              "best_noise_mse": best}), flush=True)
    chosen = torch.load(path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(chosen["model"])
    return adapter.eval(), chosen


@torch.no_grad()
def pointcloud_scores(unet, adapters, data, split, data_root, raw_root,
                      batch_size, threshold):
    with h5py.File(data_root / f"navrl_static_{split}.h5", "r") as h5:
        target_images = fetch(h5, "range_values", data.source)
    rays = {int(seed): load_rays(raw_root, split, int(seed))[0]
            for seed in sorted(set(data.seeds))}
    vae = stage1.load_circular_vae()
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR /
                        "metadata.json").read_text())["scaling_factor"]
    rows = [{"source_index": int(source), "seed": int(seed),
             "history_source_indices": [int(item) for item in hist]}
            for source, seed, hist in zip(data.source, data.seeds,
                                          data.history_source)]
    for name, adapter in (("single_frame", None), *adapters.items()):
        if adapter is not None:
            adapter.eval()
        frames = int(name.split("_")[1]) if adapter is not None else 1
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            history = data.history[start:stop].to(stage1.DEVICE)
            past = data.past_actions[start:stop].to(stage1.DEVICE)
            future = data.future_actions[start:stop].to(stage1.DEVICE)
            state = data.state[start:stop].to(stage1.DEVICE)
            condition = (history[:, -1] if adapter is None else
                         adapter(history[:, -frames:], past[:, -frames:]))
            output = stage1.generate(unet, scheduler, condition, future, state,
                                     seed=42 + start, num_steps=20,
                                     init_strength=1.0)
            images = vae.decode(output / scale).sample.cpu().numpy()
            for j, index in enumerate(range(start, stop)):
                rows[index][name] = lidar_metric(images[j], target_images[index],
                                                 rays[int(data.seeds[index])], threshold)
        print(json.dumps({"split": split, "method": name,
                          "samples": len(data)}), flush=True)
    methods = ("single_frame", *adapters.keys())
    groups = {"all": rows}
    for seed in sorted(set(data.seeds)):
        groups[f"seed_{seed}"] = [row for row in rows if row["seed"] == seed]
    summary = {}
    for key, group in groups.items():
        paired = [row for row in group if all(not row[m]["empty_cloud"] for m in methods)]
        summary[key] = {"samples": len(group), "paired_nonempty": len(paired),
                        "paired_cd_paper_m2": {m: float(np.mean([
                            row[m]["cd_paper_m2"] for row in paired])) for m in methods},
                        "all_samples": {m: summarize([row[m] for row in group])
                                        for m in methods}}
    return {"split": split, "methods": methods, "summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=stage1.OUT / "epona_probe" / "mst_history")
    parser.add_argument("--checkpoint", type=Path, default=stage1.OUT /
                        "world_circular_causal_8h" / "best.pt")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--val-per-seed", type=int, default=64)
    parser.add_argument("--test-per-seed", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stage1.DATA = args.data_root
    train = FiveHistoryDataset("train", args.data_root)
    val = FiveHistoryDataset("val", args.data_root, args.val_per_seed)
    test = FiveHistoryDataset("test", args.data_root, args.test_per_seed)
    for split, data in (("val", val), ("test", test)):
        stage1.save_json(args.out / f"{split}_manifest.json", [{
            "source_index": int(s), "seed": int(seed),
            "history_source_indices": [int(v) for v in history]}
            for s, seed, history in zip(data.source, data.seeds, data.history_source)])
    latent_meta = json.loads((stage1.OUT / stage1.LATENT_DIR /
                              "metadata.json").read_text())
    stage1.save_json(args.out / "config.json", {
        "history_frames": [3, 5], "steps_per_model": args.steps,
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "width": args.width, "blocks": args.blocks,
        "validation_per_seed": args.val_per_seed,
        "test_per_seed": args.test_per_seed, "seed": args.seed,
        "train_examples_with_five_valid_predecessors": len(train),
        "source_unet_checkpoint": str(args.checkpoint),
        "vae_sha256": latent_meta["vae_sha256"],
        "selection": "best validation noise MSE independently for 3 and 5 frames",
        "history_condition": "five preceding observed latents and past action chunks, no future ground truth",
        "final_generator": "frozen LaGen-style UNet, 20-step DDIM"})
    unet = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    source_checkpoint = stage1.load_model(args.checkpoint, unet)
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    print(json.dumps({"checkpoint_step": source_checkpoint["step"],
                      "train": len(train), "val": len(val), "test": len(test)}), flush=True)
    adapters = {}
    selected = {}
    for count in (3, 5):
        adapter, chosen = train_adapter(unet, train, val, count, args, args.out)
        adapters[f"mst_{count}"] = adapter
        selected[f"mst_{count}"] = {"step": chosen["step"],
                                     "validation_noise_mse": chosen["validation_noise_mse"]}
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    for split, data in (("val", val), ("test", test)):
        result = pointcloud_scores(unet, adapters, data, split, args.data_root,
                                   args.raw_root, args.eval_batch_size, threshold)
        diagnostic_scheduler = stage1.DDPMScheduler(
            num_train_timesteps=1000, prediction_type="epsilon")
        result["noise_mse"] = {"single_frame": noise_score(
            unet, None, data, 1, diagnostic_scheduler,
            batch_size=args.eval_batch_size)}
        for name, adapter in adapters.items():
            count = int(name.split("_")[1])
            result["noise_mse"][name] = noise_score(
                unet, adapter, data, count, diagnostic_scheduler,
                batch_size=args.eval_batch_size)
            result["noise_mse"][f"{name}_older_history_shuffled"] = noise_score(
                unet, adapter, data, count, diagnostic_scheduler,
                batch_size=args.eval_batch_size, shuffle_older=True)
        result["selected_checkpoints"] = selected
        result["source_checkpoint_step"] = source_checkpoint["step"]
        stage1.save_json(args.out / f"{split}_pointcloud.json", result)
        print(json.dumps({"split": split,
                          "paired": result["summary"]["all"]["paired_cd_paper_m2"],
                          "paired_n": result["summary"]["all"]["paired_nonempty"]}),
              flush=True)


if __name__ == "__main__":
    main()
