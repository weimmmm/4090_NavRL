"""Test non-oracle K-sample aggregation for the direct-t+3 diffusion model."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_direct_t3_autoregressive_15 import read_rows, select_scenes, summarize
from lidar_wam.runner import stage1
from lidar_wam.runner.world_direct_horizon import (
    DirectHorizonWorldModel,
    frame_metrics,
    generate,
)


def evaluate_frames(frames, targets, selected, seeds):
    rows = []
    for index, (prediction, target) in enumerate(zip(frames, targets)):
        row = frame_metrics(prediction, target, 1.5)
        row["cd_released_code_m2"] = row["cd_paper_m2"] / 2.0
        row.update({"scene_token": selected[index][0], "seed": seeds[index]})
        rows.append(row)
    return {"summary": summarize(rows), "rows": rows}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--ddim-steps", type=int, default=20)
    args = parser.parse_args()

    stage1.DATA = args.data_root.resolve()
    cache = np.load(args.latent_root.resolve() / "test.npz")
    source = np.asarray(cache["source_index"], dtype=np.int64)
    source_position = {int(row): position for position, row in enumerate(source)}
    metadata = json.loads((args.latent_root / "metadata.json").read_text())
    scale = float(metadata["scaling_factor"])
    with h5py.File(args.data_root / "navrl_static_test.h5", "r") as h5:
        selected, manifest = select_scenes(h5, source, args.scenes)
        trajectory = np.asarray([rows for _, rows in selected], dtype=np.int64)
        target_rows = trajectory[:, 2]
        targets = read_rows(h5, "range_values", target_rows).astype(np.float32)
    positions = np.asarray([[source_position[int(row)] for row in rows]
                            for rows in trajectory], dtype=np.int64)
    previous = torch.from_numpy(
        np.asarray(cache["previous"][positions[:, 0]]) * scale).float().to(stage1.DEVICE)
    actions = torch.from_numpy(
        np.asarray(cache["actions"][positions[:, :3]])).float().to(stage1.DEVICE)
    state = torch.from_numpy(stage1.causal_state(
        np.asarray(cache["state"][positions[:, 0]]))).float().to(stage1.DEVICE)

    model = DirectHorizonWorldModel().to(stage1.DEVICE).float().eval()
    checkpoint = stage1.load_model(args.checkpoint, model)
    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    scheduler = stage1.DDIMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon", clip_sample=False)

    latent_samples = torch.stack([
        generate(model, scheduler, previous, actions, state,
                 args.base_seed + sample, args.ddim_steps)
        for sample in range(args.samples)
    ])
    decoded_samples = torch.stack([
        vae.decode(latent / scale).sample for latent in latent_samples
    ])

    flat = latent_samples.flatten(2).permute(1, 0, 2)
    pairwise = (flat[:, :, None] - flat[:, None, :]).square().mean(-1)
    medoid_indices = pairwise.mean(-1).argmin(-1)
    batch_indices = torch.arange(len(previous), device=stage1.DEVICE)
    latent_medoid = latent_samples.permute(1, 0, 2, 3, 4)[
        batch_indices, medoid_indices]

    methods = {
        **{f"sample_{i}": decoded_samples[i] for i in range(args.samples)},
        "latent_mean": vae.decode(latent_samples.mean(0) / scale).sample,
        "latent_medoid": vae.decode(latent_medoid / scale).sample,
        "decoded_mean": decoded_samples.mean(0),
        "decoded_median": decoded_samples.median(0).values,
    }
    result = {
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "samples_per_condition": args.samples,
        "base_seed": args.base_seed,
        "selection": manifest,
        "methods": {
            name: evaluate_frames(value.cpu().numpy(), targets, selected,
                                  manifest["seeds"])
            for name, value in methods.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage1.save_json(args.output, result)
    print(json.dumps({name: value["summary"]
                      for name, value in result["methods"].items()}), flush=True)


if __name__ == "__main__":
    main()
