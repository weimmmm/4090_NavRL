"""Evaluate original and decoded-aux diffusion checkpoints on a fixed manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.world_decoded_aux import (LatentsWithTargets,
                                                 evaluate_generation,
                                                 representative_indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    args.out = args.out.expanduser().resolve()
    stage1.DATA = args.data_root.expanduser().resolve()
    stage1.seed_everything(args.seed)
    meta = json.loads((args.out / stage1.LATENT_DIR / "metadata.json").read_text())
    data = LatentsWithTargets(args.split, args.out,
                              source_indices=representative_indices(args.out, args.split,
                                                                       args.samples))
    model = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    checkpoint = stage1.load_model(args.checkpoint.expanduser().resolve(), model)
    vae = stage1.load_circular_vae()
    metrics = evaluate_generation(model, vae, data, float(meta["scaling_factor"]),
                                  args.ddim_steps, args.batch_size, args.seed, 1.5)
    result = {"label": args.label, "checkpoint": str(args.checkpoint),
              "checkpoint_step": int(checkpoint["step"]), "split": args.split,
              "ddim_steps": args.ddim_steps, "mask_threshold": 1.5, **metrics}
    target = args.out / "world_circular_decoded_aux_ft" / f"{args.label}_{args.split}.json"
    stage1.save_json(target, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
