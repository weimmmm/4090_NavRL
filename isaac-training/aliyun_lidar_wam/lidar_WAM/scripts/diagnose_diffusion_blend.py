"""Test whether the current world model predicts a useful latent change direction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lidar_wam.runner import stage1


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--init-strength", type=float, default=1.0)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--alphas", type=float, nargs="*",
                        default=(0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0))
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    stage1.seed_everything(args.seed)

    identity = stage1.circular_vae_identity()
    meta = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())
    config = json.loads((stage1.OUT / stage1.WORLD_FULL_DIR / "config.json").read_text())
    if any(meta.get(key) != value for key, value in identity.items()):
        raise ValueError("Latent cache does not match circular VAE weights")
    if config.get("vae_sha256") != identity["vae_sha256"]:
        raise ValueError("World checkpoint does not match circular VAE weights")

    frames = stage1.Frames(args.split, limit=args.samples, need_prev=True)
    latents = stage1.Latents(args.split, stage1.OUT, limit=args.samples)
    if not np.array_equal(frames.indices, latents.indices):
        raise ValueError("Frame and latent cache rows differ")
    world = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    step = stage1.load_model(stage1.OUT / stage1.WORLD_FULL_DIR / "best.pt", world)["step"]
    vae = stage1.load_circular_vae()
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000, prediction_type="epsilon",
                                    clip_sample=False)
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    alphas = sorted(set(args.alphas))
    if any(alpha < 0 or alpha > 1 for alpha in alphas):
        raise ValueError("Blend alphas must be within [0, 1]")
    rows = []
    for terrain_seed in sorted(stage1.EXPECTED_SEEDS[args.split]):
        positions = np.flatnonzero(frames.seeds == terrain_seed)
        for start in range(0, len(positions), args.batch_size):
            ix = positions[start:start + args.batch_size]
            previous = latents.previous[ix].to(stage1.DEVICE)
            target = latents.target[ix].to(stage1.DEVICE)
            actions = latents.actions[ix].to(stage1.DEVICE)
            state = latents.state[ix].to(stage1.DEVICE)
            generated = stage1.generate(world, scheduler, previous, actions, state,
                                        args.seed + terrain_seed * 10000 + start,
                                        args.init_strength, args.ddim_steps)
            true_delta = (target - previous).flatten(1)
            predicted_delta = (generated - previous).flatten(1)
            cosine = torch.nn.functional.cosine_similarity(
                true_delta, predicted_delta, dim=1, eps=1e-8).cpu().numpy()
            decoded = {}
            for alpha in alphas:
                blended = previous + alpha * (generated - previous)
                decoded[alpha] = vae.decode(blended / meta["scaling_factor"]).sample.cpu().numpy()
            predicted_frame = (decoded[1.0] if 1.0 in decoded else
                               vae.decode(generated / meta["scaling_factor"]).sample.cpu().numpy())
            for j, index in enumerate(ix):
                truth_cloud = stage1.to_points(frames.image[index])
                row = {"source_index": int(frames.indices[index]),
                       "seed": int(terrain_seed),
                       "copy_chamfer_m": stage1.chamfer(
                           stage1.to_points(frames.prev[index]), truth_cloud),
                       "latent_direction_cosine": float(cosine[j])}
                for alpha in alphas:
                    row[f"blend_{alpha:g}_chamfer_m"] = stage1.chamfer(
                        stage1.to_points(decoded[alpha][j], threshold), truth_cloud)
                    image_blend = frames.prev[index].copy()
                    image_blend[0] += alpha * (predicted_frame[j, 0] - image_blend[0])
                    row[f"image_blend_{alpha:g}_chamfer_m"] = stage1.chamfer(
                        stage1.to_points(image_blend), truth_cloud)
                rows.append(row)
        print(f"diagnosed seed {terrain_seed}: {len(positions)} samples", flush=True)

    summary = {"samples": len(rows), "copy_chamfer_m": float(np.mean([
                   row["copy_chamfer_m"] for row in rows])),
               "latent_direction_cosine_mean": float(np.mean([
                   row["latent_direction_cosine"] for row in rows])),
               "latent_direction_cosine_positive_fraction": float(np.mean([
                   row["latent_direction_cosine"] > 0 for row in rows])),
               "blend_chamfer_m": {str(alpha): float(np.mean([
                   row[f"blend_{alpha:g}_chamfer_m"] for row in rows]))
                   for alpha in alphas},
               "image_blend_fixed_previous_mask_chamfer_m": {str(alpha): float(np.mean([
                   row[f"image_blend_{alpha:g}_chamfer_m"] for row in rows]))
                   for alpha in alphas}}
    report = {"split": args.split, "world_step": step,
              "init_strength": args.init_strength, "ddim_steps": args.ddim_steps,
              "mask_threshold": threshold, "summary": summary, "rows": rows}
    path = (stage1.OUT / "evaluation" /
            f"blend_{args.split}_step{step}_strength{args.init_strength:g}_n{len(rows)}_ddim{args.ddim_steps}.json")
    stage1.save_json(path, report)
    print(json.dumps(summary), flush=True)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
