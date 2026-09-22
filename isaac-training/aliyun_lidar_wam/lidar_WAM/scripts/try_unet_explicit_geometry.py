"""Fine-tune the original full-next-latent UNet with causal motion-warp conditions.

The output still starts from Gaussian noise.  The warp is a *condition*, never
a future-pose input or a replacement for the diffusion output.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_geometry import load_rays, warp_frame
from evaluate_lagen_metrics_10 import lidar_metric, summarize
from evaluate_representative import fetch, save_json
from try_epona_motion import next_state, transform_from_initial
from try_epona_guided_diffusion import load_motion

ROOT = Path(__file__).resolve().parents[1]
RUN = stage1.OUT / "unet_explicit_geometry_pilot"


class GeometryUNet(stage1.WorldModel):
    def __init__(self):
        super().__init__()
        old = self.unet.conv_in
        self.unet.conv_in = nn.Conv2d(13, old.out_channels, kernel_size=old.kernel_size,
                                     stride=old.stride, padding=old.padding)
        self.motion = nn.Sequential(nn.Linear(5, 768), nn.SiLU(), nn.Linear(768, 768))

    def initialize_from_original(self, path):
        original = torch.load(path, map_location="cpu", weights_only=False)["model"]
        own = self.state_dict()
        for key, value in original.items():
            if key == "unet.conv_in.weight":
                own[key].zero_()
                own[key][:, :8].copy_(value)
            else:
                own[key].copy_(value)
        nn.init.zeros_(self.motion[-1].weight)
        nn.init.zeros_(self.motion[-1].bias)
        self.load_state_dict(own)

    def forward(self, noisy_next, previous, actions, state, timestep,
                guide_latent, guide_mask, motion):
        tokens = torch.cat((self.condition(actions, state),
                            self.motion(motion).unsqueeze(1)), dim=1)
        x = torch.cat((noisy_next, previous, guide_latent, guide_mask), dim=1)
        return self.unet(x, timestep, encoder_hidden_states=tokens).sample


def geometry_inputs(h5, indices, split, rays, motion_model):
    previous = fetch(h5, "prev_range_values", indices).astype(np.float32)
    actions = fetch(h5, "action_sequence", indices).astype(np.float64)
    states = fetch(h5, "prev_drone_state", indices).astype(np.float64)
    seeds = fetch(h5, "terrain_seed", indices).astype(int)
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Nonfinite current-state or executed-action condition")
    if not np.all(fetch(h5, "step_delta", indices) == 10):
        raise ValueError("A sample does not cover exactly ten simulator steps")
    warps, motion_tokens = [], []
    for image, command, state, seed in zip(previous, actions, states, seeds):
        future = next_state(state, command, motion_model)
        transform = transform_from_initial(state, future)
        ray_grid, azimuth, elevation = rays[int(seed)]
        warped = warp_frame(image, transform, ray_grid, azimuth, elevation)
        # The VAE was trained with -1 in empty distance cells. The geometric
        # projection's default 10m empty distance (+1) must be normalized here.
        warped[0][warped[1] < 0] = -1.0
        warps.append(warped)
        angle = np.arctan2(transform[1, 0], transform[0, 0])
        motion_tokens.append([*transform[:3, 3], np.sin(angle), np.cos(angle)])
    return np.stack(warps), np.asarray(motion_tokens, dtype=np.float32), previous, seeds


def conditions(vae, warps, motion_tokens, scale):
    image = torch.from_numpy(warps).to(stage1.DEVICE)
    guide = vae.encode(image).latent_dist.mode() * scale
    valid = (image[:, 1:2] > 0).float()
    mask = F.adaptive_max_pool2d(valid, guide.shape[-2:])
    motion = torch.from_numpy(motion_tokens).to(stage1.DEVICE)
    return guide.detach(), mask, motion


@torch.no_grad()
def sample(model, scheduler, previous, actions, state, guide, mask, motion,
           seed, steps):
    torch.manual_seed(seed)
    scheduler.set_timesteps(steps, device=stage1.DEVICE)
    x = torch.randn_like(previous)
    for t in scheduler.timesteps:
        prediction = model(x, previous, actions, state, t, guide, mask, motion)
        x = scheduler.step(prediction, t, x, eta=0.0).prev_sample
    return x


@torch.no_grad()
def evaluate(model, vae, scheduler, split, count, args, motion_model, scale, threshold,
             ablate_geometry=False):
    manifest = json.loads((stage1.OUT / "representative_baseline" /
                           "sample_manifest.json").read_text())
    selection = []
    selected_per_seed = {}
    for row in manifest["splits"][split]:
        seed = row["seed"]
        if count is None or selected_per_seed.get(seed, 0) < count // 2:
            selection.append(row)
            selected_per_seed[seed] = selected_per_seed.get(seed, 0) + 1
    indices = np.array([row["source_index"] for row in selection], dtype=np.int64)
    cached = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
    lookup = {int(index): i for i, index in enumerate(cached["source_index"])}
    positions = np.array([lookup[int(i)] for i in indices])
    ray_cache = {seed: load_rays(args.raw_root, split, seed)
                 for seed in sorted(set(row["seed"] for row in selection))}
    rows = []
    with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
        for start in range(0, len(indices), args.batch_size):
            sl = slice(start, start + args.batch_size)
            batch_indices = indices[sl]
            batch_pos = positions[sl]
            warped, motion, initial, seeds = geometry_inputs(
                h5, batch_indices, split, ray_cache, motion_model)
            target = fetch(h5, "range_values", batch_indices)
            guide, mask, motion = conditions(vae, warped, motion, scale)
            if ablate_geometry:
                guide = torch.zeros_like(guide)
                mask = torch.zeros_like(mask)
                motion = torch.zeros_like(motion)
            previous = torch.from_numpy(cached["previous"][batch_pos].copy() * scale).to(stage1.DEVICE)
            actions = torch.from_numpy(cached["actions"][batch_pos].copy()).to(stage1.DEVICE)
            state = torch.from_numpy(stage1.causal_state(cached["state"][batch_pos])).to(stage1.DEVICE)
            generated = sample(model, scheduler, previous, actions, state, guide, mask,
                               motion, args.seed + start, args.ddim_steps)
            images = vae.decode(generated / scale).sample.cpu().numpy()
            for j, index in enumerate(batch_indices):
                ray_grid = ray_cache[int(seeds[j])][0]
                rows.append({"source_index": int(index), "seed": int(seeds[j]),
                             "model": lidar_metric(images[j], target[j], ray_grid, threshold),
                             "copy": lidar_metric(initial[j], target[j], ray_grid, 0),
                             "warp": lidar_metric(warped[j], target[j], ray_grid, 0)})
    summary = {name: {"all": summarize([row[name] for row in rows]),
                      **{f"seed_{seed}": summarize([row[name] for row in rows if row["seed"] == seed])
                         for seed in ray_cache}} for name in ("model", "copy", "warp")}
    return {"split": split, "samples": len(rows), "summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation-only", action="store_true")
    args = parser.parse_args()
    stage1.DATA = args.data_root.resolve()
    args.raw_root = args.raw_root.resolve()
    RUN.mkdir(parents=True, exist_ok=True)
    motion_path = stage1.OUT / "epona_probe" / "motion_ridge.npz"
    motion_model = load_motion(motion_path)
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())["scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR / "oracle_val.json").read_text())["selected_threshold"]
    source = stage1.OUT / "world_circular_causal_8h" / "best.pt"
    model = GeometryUNet()
    model.initialize_from_original(source)
    model = model.to(stage1.DEVICE).float()
    vae = stage1.load_circular_vae()
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    train_scheduler = stage1.DDPMScheduler(num_train_timesteps=1000,
                                           prediction_type="epsilon")
    if args.ablation_only:
        model.load_state_dict(torch.load(RUN / "best.pt", map_location="cpu",
                                         weights_only=False)["model"])
        model.to(stage1.DEVICE).eval()
        result = evaluate(model, vae, scheduler, "val", 128, args, motion_model,
                          scale, threshold, ablate_geometry=True)
        save_json(RUN / "validation_128_geometry_ablation.json", result)
        print(json.dumps({"ablation_cd_paper_m2": result["summary"]["model"]["all"]["cd_paper_m2"]}),
              flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cached = np.load(stage1.OUT / stage1.LATENT_DIR / "train.npz")
    with h5py.File(stage1.DATA / "navrl_static_train.h5", "r") as h5:
        indices = cached["source_index"]
        finite = np.isfinite(fetch(h5, "action_sequence", indices)).all(axis=(1, 2))
        finite &= np.isfinite(fetch(h5, "prev_drone_state", indices)).all(axis=1)
        finite &= fetch(h5, "step_delta", indices) == 10
    eligible = np.flatnonzero(finite)
    if len(eligible) < 1000:
        raise ValueError("Too few eligible training transitions")
    ray_cache = {seed: load_rays(args.raw_root, "train", seed)
                 for seed in sorted(stage1.EXPECTED_SEEDS["train"])}
    save_json(RUN / "config.json", {"source_checkpoint": str(source),
              "motion_model": str(motion_path), "steps": args.steps,
              "batch_size": args.batch_size, "lr": args.lr,
              "condition": "predicted-pose warp VAE latent + projected-hit mask + pose token",
              "output": "full next latent, pure-Gaussian DDIM sampling",
              "validation_selection": "lowest squared Chamfer on first 128 fixed validation samples",
              "raw_HDF5_modified": False})
    rng = np.random.default_rng(args.seed)
    best = float("inf")
    with h5py.File(stage1.DATA / "navrl_static_train.h5", "r") as h5:
        for step in range(1, args.steps + 1):
            chosen = rng.choice(eligible, args.batch_size, replace=False)
            chosen.sort()
            warped, motion, _, _ = geometry_inputs(h5, indices[chosen], "train",
                                                    ray_cache, motion_model)
            with torch.no_grad():
                guide, mask, motion = conditions(vae, warped, motion, scale)
            previous = torch.from_numpy(cached["previous"][chosen].copy() * scale).to(stage1.DEVICE)
            target = torch.from_numpy(cached["target"][chosen].copy() * scale).to(stage1.DEVICE)
            actions = torch.from_numpy(cached["actions"][chosen].copy()).to(stage1.DEVICE)
            state = torch.from_numpy(stage1.causal_state(cached["state"][chosen])).to(stage1.DEVICE)
            noise = torch.randn_like(target)
            t = torch.randint(0, 1000, (len(target),), device=stage1.DEVICE)
            model.train()
            prediction = model(train_scheduler.add_noise(target, noise, t), previous,
                               actions, state, t, guide, mask, motion)
            loss = F.mse_loss(prediction, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 50 == 0:
                print(json.dumps({"step": step, "noise_mse": loss.item()}), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                validation = evaluate(model, vae, scheduler, "val", 128, args,
                                      motion_model, scale, threshold)
                score = validation["summary"]["model"]["all"]["cd_paper_m2"]
                print(json.dumps({"step": step, "validation_cd_paper_m2": score}), flush=True)
                save_json(RUN / f"validation_step{step}.json", validation)
                if score < best:
                    best = score
                    torch.save({"model": model.cpu().state_dict(), "step": step,
                                "validation_cd_paper_m2": score}, RUN / "best.pt")
                    model.to(stage1.DEVICE)
                    save_json(RUN / "best_selection.json", {"step": step, "cd_paper_m2": score})
    model.load_state_dict(torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)["model"])
    model.to(stage1.DEVICE).eval()
    validation = evaluate(model, vae, scheduler, "val", None, args, motion_model, scale, threshold)
    save_json(RUN / "validation_full.json", validation)
    test = evaluate(model, vae, scheduler, "test", None, args, motion_model, scale, threshold)
    save_json(RUN / "test.json", test)
    print(json.dumps({"best_step": json.loads((RUN / "best_selection.json").read_text())["step"],
                      "validation": validation["summary"], "test": test["summary"]}), flush=True)


if __name__ == "__main__":
    main()
