"""Paired offline accuracy and GPU latency audit of one unchanged checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "lidar_WAM"),
                str(ROOT / "lidar_WAM/third_party/python_deps")]

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lidar_wam.models.action_expert import SuperActionExpertModel
from lidar_wam.models.action_flow_sampling import sample_action_flow
from lidar_wam.runner.super_action_expert import (
    _autocast, _load_frozen_vae, _prepare,
)
from lidar_wam.runner.super_lee_action_data import (
    SuperLeeActionDataset, fixed_eval_indices,
)

MODES = {"direct": 0, "flow10": 10, "flow20": 20}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def predict(model, mode, inputs, noise):
    latent, goal, proprio, past, mask = inputs
    if mode == "direct":
        return model.predict_action(latent, goal, proprio, past, mask)
    observation = model.encode_current(latent)
    return sample_action_flow(model.diffusion, observation, goal, proprio,
                              past, mask, steps=MODES[mode], initial_noise=noise)


def summary(values):
    value = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(value.mean()), "median_ms": float(np.median(value)),
            "p95_ms": float(np.percentile(value, 95)), "repeats": len(value)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise-seed", type=int, default=271828)
    parser.add_argument("--latency-repeats", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.windows <= 0 or args.batch_size <= 0:
        raise ValueError("windows and batch-size must be positive")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, arch = payload["config"], payload["architecture"]
    torch.manual_seed(int(config["seed"]))
    model = SuperActionExpertModel(
        **{name: int(arch[name]) for name in ("width", "depth", "heads", "ffn_width")},
        horizon=int(arch["action_horizon"]), action_dim=int(arch["action_dim"]),
        proprio_dim=int(arch["proprio_dim"]), past_horizon=int(arch["past_horizon"]),
        lidar_history_frames=int(arch["history_lidar_frames"])).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    vae, scale = _load_frozen_vae(Path(payload["vae"]["checkpoint"]), device)
    if not math.isclose(scale, float(payload["vae"]["scaling_factor"]),
                        rel_tol=0., abs_tol=1e-12):
        raise ValueError("checkpoint/VAE latent scale differs")
    data_root = Path(config.get("validation_root") or config["train_root"])
    # The checkpoint stores raw CLI args (empty shards means auto-split).
    # The adjacent run config records the actual resolved held-out shards.
    run_config_path = args.checkpoint.parent / "config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.is_file() else {}
    validation_shards = (run_config.get("validation_shards")
                         or config.get("validation_shards"))
    if not validation_shards:
        from lidar_wam.runner.super_lee_action_data import discover_shard_names
        available = discover_shard_names(data_root)
        if config.get("validation_root"):
            validation_shards = available
        else:
            groups = {}
            for name in available:
                groups.setdefault(name.split("/")[0], []).append(name)
            validation_shards = [sorted(names)[-1] for _, names in sorted(groups.items())]
    data = SuperLeeActionDataset(data_root, shard_names=validation_shards)
    indices = fixed_eval_indices(len(data), args.windows, int(config["seed"]) + 17)
    loader = DataLoader(Subset(data, indices.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=True)
    action_scale = torch.as_tensor(
        payload["action_interface"]["model_output_scale_mps_xyz"],
        device=device, dtype=torch.float32).reshape(1, 1, 3)
    stats = payload["condition_stats"]
    sums = {name: dict(abs=0., squared=0., elements=0, first=0., first_elements=0,
                        near=0., near_elements=0, paired_to_direct=0.) for name in MODES}
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)
    fixed_inputs = fixed_image = None
    started = time.time()
    for batch_index, batch in enumerate(loader):
        latent, _target, target, goal, proprio, past, mask = _prepare(
            batch, vae, scale, action_scale, stats, device, "bf16")
        # Match the saved deployment contract, not teacher-forced history.
        if not payload.get("past_action_conditioning", {}).get(
                "deployment_use_past_actions", True):
            past, mask = torch.zeros_like(past), torch.zeros_like(mask)
        inputs = (latent, goal, proprio, past, mask)
        if fixed_inputs is None:
            fixed_inputs = tuple(value.clone() for value in inputs)
            fixed_image = batch[0].to(device).float()
        # One shared initial noise tensor for both discretizations, per sample.
        noise = torch.randn((len(target), 10, 3), device=device, generator=generator)
        predictions = {}
        for mode in MODES:
            with _autocast(device, "bf16"):
                prediction = predict(model, mode, inputs, noise).float() * action_scale
            # The Isaac adapter reapplies the physical 2 m/s norm bound; this
            # also removes BF16 rounding overshoot from the Direct projection.
            prediction = prediction * (2.0 / prediction.norm(
                dim=-1, keepdim=True).clamp_min(1e-6)).clamp_max(1.0)
            predictions[mode] = prediction
        near = batch[2][:, 3].to(device) <= 3.0
        for mode, prediction in predictions.items():
            error, row = prediction - target, sums[mode]
            row["abs"] += float(error.abs().sum())
            row["squared"] += float(error.square().sum())
            row["elements"] += error.numel()
            row["first"] += float(error[:, 0].abs().sum())
            row["first_elements"] += error[:, 0].numel()
            row["near"] += float(error[near].abs().sum())
            row["near_elements"] += error[near].numel()
            row["paired_to_direct"] += float(
                (prediction - predictions["direct"]).abs().sum())
        if batch_index % 8 == 0:
            print(json.dumps({"batch": batch_index, "windows_seen":
                sums["direct"]["elements"] // 30, "elapsed_s": time.time() - started}),
                  flush=True)
    result = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
              "checkpoint_step": int(payload["step"]), "vae": payload["vae"],
              "validation_shards": list(data.shard_names), "available_windows": len(data),
              "windows": len(indices), "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
              "past_actions": "checkpoint deployment setting; disabled means zeros plus invalid mask",
              "initial_noise": "identical per window for Flow10 and Flow20; Gaussian std=1",
              "noise_seed": args.noise_seed, "precision": "bf16 autocast, fp32 Euler state",
              "final_projection": "unit L2 ball, checkpoint physical scaling, adapter 2 m/s norm bound",
              "timing_note": "GPU synchronized wall clock; no file IO; same cached current image and conditions",
              "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "metrics": {}}
    for mode, row in sums.items():
        result["metrics"][mode] = {
            "action_mae_mps": row["abs"] / row["elements"],
            "action_rmse_mps": math.sqrt(row["squared"] / row["elements"]),
            "first_action_mae_mps": row["first"] / row["first_elements"],
            "near_goal_action_mae_mps": row["near"] / max(row["near_elements"], 1),
            "near_goal_windows": row["near_elements"] // 30,
            "difference_from_direct_mae_mps": row["paired_to_direct"] / row["elements"],
            "latency": {}}
    for size in (1, min(128, len(fixed_inputs[0]))):
        inputs = tuple(value[:size] for value in fixed_inputs)
        image = fixed_image[:size]
        noise = torch.randn((size, 10, 3), device=device,
                            generator=torch.Generator(device=device).manual_seed(args.noise_seed))
        for mode in MODES:
            timings = []
            for iteration in range(3 + args.latency_repeats):
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                with _autocast(device, "bf16"):
                    latent = vae.encode(image).latent_dist.mode() * scale
                    predict(model, mode, (latent, *inputs[1:]), noise)
                torch.cuda.synchronize(device)
                if iteration >= 3:
                    timings.append((time.perf_counter() - tick) * 1000.)
            result["metrics"][mode]["latency"][f"batch_{size}_vae_encoder_policy"] = summary(timings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
