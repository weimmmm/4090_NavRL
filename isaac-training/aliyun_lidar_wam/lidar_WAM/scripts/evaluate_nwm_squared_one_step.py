"""Evaluate the existing NWM checkpoint on the fixed 512 test transitions in m²."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_lagen_metrics_10 import lidar_metric
from evaluate_representative import load_selected
from lidar_wam.runner import stage1
from lidar_wam.runner.lidar_geometry import load_rays
from lidar_wam.runner.nwm_predictor import NavRLCDiT, model_kwargs
from third_party.nwm.diffusion import create_diffusion


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    manifest = json.loads((stage1.OUT / "representative_baseline" /
                           "sample_manifest.json").read_text())
    rows, data, positions, arrays = load_selected(manifest, "test")
    rays = {seed: load_rays(args.raw_root, "test", seed)[0]
            for seed in sorted({r["seed"] for r in rows})}
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR /
                        "metadata.json").read_text())["scaling_factor"]
    vae = stage1.load_circular_vae()
    model = NavRLCDiT("CDiT-B/2").to(stage1.DEVICE).float().eval()
    ckpt = torch.load(stage1.OUT / "world_nwm_cdit_full" / "best.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    diffusion = create_diffusion("ddim20")
    output = [None] * len(rows)
    for seed in sorted(rays):
        locations = np.array([i for i, row in enumerate(rows) if row["seed"] == seed])
        for start in range(0, len(locations), args.batch_size):
            chosen = locations[start:start + args.batch_size]
            source = positions[chosen]
            previous = data.previous[source].to(stage1.DEVICE)
            actions = data.actions[source].to(stage1.DEVICE)
            state = data.state[source].to(stage1.DEVICE)
            torch.manual_seed(42 + seed * 10000 + start)
            noise = torch.randn_like(NavRLCDiT.pad_latent(previous))
            latent = diffusion.ddim_sample_loop(
                model, noise.shape, noise=noise,
                clip_denoised=False,
                model_kwargs=model_kwargs(previous, actions, state),
                device=stage1.DEVICE, progress=False)
            predicted = vae.decode(NavRLCDiT.crop_latent(latent) / scale).sample.cpu().numpy()
            for local, index in enumerate(chosen):
                target = arrays["range_values"][index]
                output[index] = {"source_index": int(rows[index]["source_index"]),
                                 "seed": int(seed),
                                 "copy": lidar_metric(arrays["prev_range_values"][index],
                                                      target, rays[seed], 0),
                                 "nwm": lidar_metric(predicted[local], target,
                                                     rays[seed], threshold)}
        print(json.dumps({"seed": seed, "done": len(locations)}), flush=True)
    all_mean = {name: float(np.mean([r[name]["cd_paper_m2"] for r in output]))
                for name in ("copy", "nwm")}
    paired = [r for r in output if not r["copy"]["empty_cloud"]
              and not r["nwm"]["empty_cloud"]]
    paired_mean = {name: float(np.mean([r[name]["cd_paper_m2"] for r in paired]))
                   for name in ("copy", "nwm")}
    report = {"split": "test", "samples": len(output),
              "checkpoint_step": ckpt["step"],
              "all_cd_paper_m2": all_mean, "paired_nonempty": len(paired),
              "paired_cd_paper_m2": paired_mean, "rows": output}
    path = stage1.OUT / "representative_baseline" / "test_nwm_squared_one_step.json"
    stage1.save_json(path, report)
    print(json.dumps({"all": all_mean, "paired": paired_mean,
                      "paired_n": len(paired)}), flush=True)


if __name__ == "__main__":
    main()
