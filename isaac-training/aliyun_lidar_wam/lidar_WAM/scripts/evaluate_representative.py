"""Fixed, representative NavRL LiDAR benchmarks using existing checkpoints only.

No command in this file trains a model or changes the source HDF5 files.
Run `prepare`, evaluate each method on validation, then repeat on test after
the validation protocol has been fixed. `summarize` joins per-sample reports.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_wam.runner import stage1
from lidar_wam.runner.executed_residual import ExecutedLatents, condition_stats
from lidar_wam.runner.lidar_geometry import (
    frame_points, load_rays, predict_transform, project_points, warp_frame,
)
from lidar_wam.runner.nwm_predictor import NavRLCDiT, model_kwargs
from third_party.nwm.diffusion import create_diffusion


METHODS = ("copy", "vae", "oracle_pose", "velocity_pose", "lagen_unet",
           "lagen_residual", "nwm")
ACTION_METHODS = {"lagen_unet", "lagen_residual", "nwm"}
SPLITS = ("val", "test")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fetch(h5, key, indices):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    return h5[key][indices[order]][np.argsort(order)]


def prepare(output):
    manifest = {"version": 1, "random_seed": 42, "samples_per_seed": 256,
                "selection": "uniform without replacement among finite executed-action, 10-step transitions",
                "splits": {}}
    for split in SPLITS:
        data = ExecutedLatents(split)
        rng = np.random.default_rng(42)
        with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
            state = fetch(h5, "prev_drone_state", data.indices)
            relative_pose = fetch(h5, "prev_trans_mat", data.indices)
        finite_geometry = (np.isfinite(state).all(axis=1) &
                           np.isfinite(relative_pose).all(axis=(1, 2)))
        selected = np.concatenate([
            rng.choice(np.flatnonzero((data.seeds == seed) & finite_geometry),
                       256, replace=False)
            for seed in sorted(stage1.EXPECTED_SEEDS[split])])
        source = np.sort(data.indices[selected])
        with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
            frames = h5["frame_idx"][source]
            seeds = h5["terrain_seed"][source]
            scenes = h5["scene_token"][source]
            assert np.all(h5["step_delta"][source] == 10)
            assert np.all(h5["action_mask"][source])
            assert np.isfinite(h5["action_sequence"][source]).all()
        rows = [{"source_index": int(i), "seed": int(seed),
                 "frame_idx": int(frame), "scene_token": scene.decode() if isinstance(scene, bytes) else scene}
                for i, seed, frame, scene in zip(source, seeds, frames, scenes)]
        manifest["splits"][split] = rows
    save_json(output / "sample_manifest.json", manifest)
    print(json.dumps({split: {"samples": len(rows),
                              "early_frame_fraction": sum(r["frame_idx"] <= 2 for r in rows) / len(rows)}
                      for split, rows in manifest["splits"].items()}), flush=True)


def manifest_hash(manifest):
    canonical = json.dumps(manifest, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def metric(prediction, target, rays, threshold=0.0):
    valid = target[1, :, :18] > 0
    hit = prediction[1, :, :18] > threshold
    distance = np.clip((prediction[0, :, :18] + 1) * 5, 0, 10)
    truth_distance = np.clip((target[0, :, :18] + 1) * 5, 0, 10)
    return {"chamfer_m": stage1.chamfer(
                frame_points(prediction, rays, threshold), frame_points(target, rays)),
            "valid_range_mae_m": float(np.abs(distance[valid] - truth_distance[valid]).mean())
                if valid.any() else 0.0,
            "valid_range_abs_sum_m": float(np.abs(distance[valid] - truth_distance[valid]).sum()),
            "true_hits": int(valid.sum()), "tp": int((hit & valid).sum()),
            "fp": int((hit & ~valid).sum()), "fn": int((~hit & valid).sum())}


def summarize_metrics(records):
    result = {"samples": len(records),
              "chamfer_m": float(np.mean([r["chamfer_m"] for r in records])),
              "valid_range_mae_m": sum(r["valid_range_abs_sum_m"] for r in records) /
                                   max(sum(r["true_hits"] for r in records), 1)}
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    result["mask_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    return result


def grouped_summary(rows, key="metrics"):
    groups = {"all": rows,
              "early_frame_1_2": [r for r in rows if r["frame_idx"] <= 2],
              "later_frame_3_plus": [r for r in rows if r["frame_idx"] > 2]}
    for seed in sorted({r["seed"] for r in rows}):
        groups[f"seed_{seed}"] = [r for r in rows if r["seed"] == seed]
    return {label: summarize_metrics([r[key] for r in group])
            for label, group in groups.items() if group}


def load_selected(manifest, split):
    rows = manifest["splits"][split]
    data = ExecutedLatents(split)
    lookup = {int(source): position for position, source in enumerate(data.indices)}
    source = np.array([row["source_index"] for row in rows], dtype=np.int64)
    if any(int(i) not in lookup for i in source):
        raise ValueError("Manifest includes an invalid or stale transition")
    positions = np.array([lookup[int(i)] for i in source])
    with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
        arrays = {key: fetch(h5, key, source) for key in (
            "prev_range_values", "range_values", "prev_drone_state",
            "normalized_action_sequence", "prev_ego_feats", "prev_trans_mat")}
    return rows, data, positions, arrays


def shuffle_positions(rows):
    result = np.arange(len(rows))
    for seed in sorted({r["seed"] for r in rows}):
        locations = np.array([i for i, r in enumerate(rows) if r["seed"] == seed])
        rng = np.random.default_rng(1042 + seed)
        perm = rng.permutation(len(locations))
        while np.any(perm == np.arange(len(locations))):
            perm = rng.permutation(len(locations))
        result[locations] = locations[perm]
    return result


def identity_check(frame, rays, azimuth, elevation):
    recovered = project_points(frame_points(frame, rays), azimuth, elevation)
    original_mask = frame[1, :, :18] > 0
    recovered_mask = recovered[1, :, :18] > 0
    errors = np.abs((recovered[0, :, :18] - frame[0, :, :18]) * 5)[original_mask]
    mismatch = int(np.count_nonzero(original_mask != recovered_mask))
    maximum = float(errors.max()) if len(errors) else 0.0
    if mismatch or maximum > 1e-4:
        raise ValueError(f"Identity ray reprojection failed: mask={mismatch}, distance={maximum}")
    return maximum


def evaluate_simple(method, rows, arrays, rays_by_seed, source_root, split):
    output = []
    vae = stage1.load_circular_vae() if method == "vae" else None
    selected_latents = None
    if method == "vae":
        # The existing latent cache stores posterior mode before scaling.
        cached = np.load(stage1.OUT / stage1.LATENT_DIR / f"{split}.npz")
        lookup = {int(source): i for i, source in enumerate(cached["source_index"])}
        selected_latents = cached["target"][[lookup[r["source_index"]] for r in rows]]
    identity_max = 0.0
    with torch.no_grad():
        for i, row in enumerate(rows):
            rays, azimuth, elevation = rays_by_seed[row["seed"]]
            previous = arrays["prev_range_values"][i]
            target = arrays["range_values"][i]
            if method in ("oracle_pose", "velocity_pose"):
                identity_max = max(identity_max, identity_check(previous, rays, azimuth, elevation))
                transform = (arrays["prev_trans_mat"][i] if method == "oracle_pose" else
                             predict_transform(arrays["prev_drone_state"][i]))
                prediction = warp_frame(previous, transform, rays, azimuth, elevation)
            elif method == "copy":
                prediction = previous
            else:
                z = torch.from_numpy(selected_latents[i:i + 1]).to(stage1.DEVICE)
                prediction = vae.decode(z).sample[0].cpu().numpy()
            record = dict(row)
            record["metrics"] = metric(prediction, target, rays,
                                       1.5 if method == "vae" else 0.0)
            if method == "velocity_pose":
                true_t = arrays["prev_trans_mat"][i]
                record["pose_translation_error_m"] = float(np.linalg.norm(
                    transform[:3, 3] - true_t[:3, 3]))
                record["pose_rotation_error_deg"] = float(np.rad2deg(
                    Rotation.from_matrix(transform[:3, :3] @ true_t[:3, :3].T).magnitude()))
            output.append(record)
    extra = {"identity_max_distance_error_m": identity_max} if method in (
        "oracle_pose", "velocity_pose") else {}
    return output, extra


@torch.no_grad()
def evaluate_model(method, rows, data, positions, arrays, rays_by_seed, batch_size,
                   unet_run=None, checkpoint_name="best.pt"):
    vae = stage1.load_circular_vae()
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR / "metadata.json").read_text())[
        "scaling_factor"]
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR / "oracle_val.json").read_text())[
        "selected_threshold"]
    if method == "nwm":
        model = NavRLCDiT("CDiT-B/2").to(stage1.DEVICE).float().eval()
        checkpoint = torch.load(stage1.OUT / "world_nwm_cdit_full" / "best.pt",
                                map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        diffusion = create_diffusion("ddim20")
    else:
        model = stage1.WorldModel(state_dim=11 if method == "lagen_residual" else 5)
        model = model.to(stage1.DEVICE).float().eval()
        run = ("world_circular_executed_residual_full" if method == "lagen_residual" else
               (unet_run or "world_circular_causal_full"))
        checkpoint = stage1.load_model(stage1.OUT / run /
                                       (checkpoint_name if method == "lagen_unet" else "best.pt"), model)
        scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                          prediction_type="epsilon", clip_sample=False)
    shuffled = shuffle_positions(rows)
    predictions = [None] * len(rows)
    sensitivities = [None] * len(rows)
    for seed in sorted({r["seed"] for r in rows}):
        locations = np.array([i for i, r in enumerate(rows) if r["seed"] == seed])
        for start in range(0, len(locations), batch_size):
            chosen = locations[start:start + batch_size]
            source = positions[chosen]
            previous = data.previous[source].to(stage1.DEVICE)
            state = data.state[source].to(stage1.DEVICE)
            if method == "lagen_unet":
                actions = torch.from_numpy(arrays["normalized_action_sequence"][chosen]).to(stage1.DEVICE)
                state = torch.from_numpy(stage1.causal_state(arrays["prev_ego_feats"][chosen])).to(stage1.DEVICE)
                shuffled_actions = torch.from_numpy(arrays["normalized_action_sequence"][shuffled[chosen]]).to(stage1.DEVICE)
            else:
                actions = data.actions[source].to(stage1.DEVICE)
                shuffled_actions = data.actions[positions[shuffled[chosen]]].to(stage1.DEVICE)
            seed_number = 42 + seed * 10000 + start
            if method == "nwm":
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seed_number)
                    noise = torch.randn_like(NavRLCDiT.pad_latent(previous))
                def sample(a):
                    z = diffusion.ddim_sample_loop(
                        model, noise.shape, noise=noise.clone(), clip_denoised=False,
                        model_kwargs=model_kwargs(previous, a, state),
                        device=stage1.DEVICE, progress=False)
                    return NavRLCDiT.crop_latent(z)
            else:
                residual_scale = (condition_stats()["residual_scale"]
                                  if method == "lagen_residual" else None)
                def sample(a):
                    return stage1.generate(
                        model, scheduler, previous, a, state, seed_number,
                        0.05 if method == "lagen_residual" else 1.0,
                        20, residual_scale=residual_scale)
            result = vae.decode(sample(actions) / scale).sample.cpu().numpy()
            changed = vae.decode(sample(shuffled_actions) / scale).sample.cpu().numpy()
            for local, global_index in enumerate(chosen):
                predictions[global_index] = result[local]
                sensitivities[global_index] = changed[local]
        print(f"{method}: seed {seed} complete", flush=True)
    output = []
    for i, row in enumerate(rows):
        rays = rays_by_seed[row["seed"]][0]
        target = arrays["range_values"][i]
        record = dict(row)
        record["metrics"] = metric(predictions[i], target, rays, threshold)
        record["shuffled_action_metrics"] = metric(sensitivities[i], target, rays, threshold)
        output.append(record)
    return output, {"checkpoint_step": checkpoint["step"],
                    "ddim_steps": 20, "mask_logit_threshold": threshold,
                    "init_strength": 0.05 if method == "lagen_residual" else 1.0,
                    "action_shuffle_interpretation": "input sensitivity only; not a counterfactual accuracy test"}


def evaluate(args, manifest):
    rows, data, positions, arrays = load_selected(manifest, args.split)
    rays_by_seed = {seed: load_rays(args.raw_root, args.split, seed)
                    for seed in sorted(stage1.EXPECTED_SEEDS[args.split])}
    if args.method in ACTION_METHODS:
        result, extra = evaluate_model(args.method, rows, data, positions, arrays,
                                       rays_by_seed, args.batch_size, args.unet_run,
                                       args.checkpoint_name)
    else:
        result, extra = evaluate_simple(args.method, rows, arrays, rays_by_seed,
                                        args.raw_root, args.split)
    report = {"version": 1, "split": args.split, "method": args.method,
              "manifest_sha256": manifest_hash(manifest), "summary": grouped_summary(result),
              "rows": result, **extra}
    if args.method == "lagen_unet":
        report["checkpoint_run"] = args.unet_run or "world_circular_causal_full"
        report["checkpoint_name"] = args.checkpoint_name
    if args.method in ACTION_METHODS:
        report["shuffled_action_summary"] = grouped_summary(result, "shuffled_action_metrics")
    suffix = f"_{args.report_tag}" if args.report_tag else ""
    path = args.output / f"{args.split}_{args.method}{suffix}.json"
    save_json(path, report)
    print(json.dumps({"report": str(path), "all": report["summary"]["all"],
                      "shuffled_action": report.get("shuffled_action_summary", {}).get("all")}), flush=True)


def summarize(args, manifest):
    reports = {}
    for method in METHODS:
        path = args.output / f"{args.split}_{method}.json"
        report = json.loads(path.read_text())
        if report["manifest_sha256"] != manifest_hash(manifest):
            raise ValueError(f"Stale report: {path}")
        reports[method] = report
    expected = [r["source_index"] for r in manifest["splits"][args.split]]
    combined = []
    for i, item in enumerate(manifest["splits"][args.split]):
        row = dict(item)
        row["methods"] = {}
        for method, report in reports.items():
            source = report["rows"][i]
            if source["source_index"] != expected[i]:
                raise ValueError(f"Sample order mismatch: {method}")
            row["methods"][method] = source["metrics"]
            if "shuffled_action_metrics" in source:
                row["methods"][method + "_shuffled"] = source["shuffled_action_metrics"]
        combined.append(row)
    summary = {method: report["summary"] for method, report in reports.items()}
    for method, report in reports.items():
        if "shuffled_action_summary" in report:
            summary[method + "_shuffled"] = report["shuffled_action_summary"]
    copy = summary["copy"]["all"]["chamfer_m"]
    oracle = summary["oracle_pose"]["all"]["chamfer_m"]
    result = {"version": 1, "split": args.split,
              "manifest_sha256": manifest_hash(manifest), "samples": len(combined),
              "methods": summary,
              "oracle_pose_improvement_vs_copy": 1 - oracle / copy,
              "geometry_gate_10_percent": oracle <= 0.9 * copy,
              "future_pose_is_diagnostic_only": True,
              "decision": ("corrected pose warp clears the 10% gate" if oracle <= 0.9 * copy
                           else "corrected pose warp fails the 10% gate; investigate visibility and projection"),
              "rows": combined}
    path = args.output / f"summary_{args.split}.json"
    save_json(path, result)
    print(json.dumps({"report": str(path), "copy": copy, "oracle_pose": oracle,
                      "geometry_gate_10_percent": result["geometry_gate_10_percent"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "summarize"))
    parser.add_argument("--split", choices=SPLITS, default="val")
    parser.add_argument("--method", choices=METHODS, default="copy")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unet-run", default=None,
                        help="Output subdirectory holding an alternative LaGen UNet best.pt")
    parser.add_argument("--checkpoint-name", default="best.pt", choices=("best.pt", "latest.pt"),
                        help="Which LaGen UNet checkpoint to evaluate")
    parser.add_argument("--report-tag", default=None,
                        help="Suffix for a separate method report without overwriting the baseline")
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, default=stage1.OUT / "representative_baseline")
    args = parser.parse_args()
    stage1.DATA = args.data_root.expanduser().resolve()
    args.raw_root = (args.raw_root or stage1.DATA.parent).expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.command == "prepare":
        prepare(args.output)
        return
    manifest = json.loads((args.output / "sample_manifest.json").read_text())
    if args.command == "evaluate":
        evaluate(args, manifest)
    else:
        summarize(args, manifest)


if __name__ == "__main__":
    main()
