"""Causal action-to-pose baseline and calibrated LiDAR rollout diagnostics.

Uses finite executed world-frame commands.  Future drone states are read only
for supervised fitting and scoring, never as rollout inputs.  No HDF5 is edited.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner.lidar_geometry import load_rays, warp_frame
from evaluate_lagen_metrics_10 import lidar_metric
from evaluate_representative import fetch, save_json

ROOT = Path(__file__).resolve().parents[1]
DT = 0.16


def angle_delta(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def yaw(state):
    q = state[3:7]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2]


def features(states, actions):
    """Only previous state and this interval's ten executed commands."""
    q = states[:, 3:7]
    angles = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_euler("xyz")[:, 2]
    return np.concatenate((states[:, 2:3], states[:, 7:13],
                           np.sin(angles)[:, None], np.cos(angles)[:, None],
                           actions.reshape(len(actions), -1),
                           actions.mean(axis=1), actions[:, -1]), axis=1)


def residual_targets(previous, current):
    p = current[:, :3] - previous[:, :3] - previous[:, 7:10] * DT
    yp = Rotation.from_quat(previous[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2]
    yc = Rotation.from_quat(current[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2]
    d_yaw = angle_delta(yc, yp) - previous[:, 12] * DT
    d_velocity = current[:, 7:10] - previous[:, 7:10]
    d_yaw_rate = current[:, 12] - previous[:, 12]
    return np.column_stack((p, d_yaw, d_velocity, d_yaw_rate))


def next_state(previous, actions, model=None):
    result = previous.copy()
    if model is None:
        correction = np.zeros(8, dtype=np.float64)
    else:
        x = features(previous[None], actions[None])[0]
        correction = ((x - model["mean"]) / model["std"]) @ model["weight"]
        correction += model["bias"]
    result[:3] += previous[7:10] * DT + correction[:3]
    dq = Rotation.from_euler("z", previous[12] * DT + correction[3])
    q = previous[3:7]
    old = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    new_q = (dq * old).as_quat()
    result[3:7] = new_q[[3, 0, 1, 2]]
    result[7:10] += correction[4:7]
    result[12] += correction[7]
    return result


def fit_model(x, y, alpha):
    mean = x.mean(axis=0)
    std = x.std(axis=0).clip(1e-5)
    xn = (x - mean) / std
    yn = y.mean(axis=0)
    weights = np.linalg.solve(xn.T @ xn + alpha * np.eye(x.shape[1]),
                              xn.T @ (y - yn))
    return {"mean": mean, "std": std, "weight": weights, "bias": yn,
            "alpha": alpha}


def load_pairs(data_root, split):
    cache = np.load(ROOT / "outputs" / "latents_circular" / f"{split}.npz")
    indices = cache["source_index"]
    with h5py.File(data_root / f"navrl_static_{split}.h5", "r") as h5:
        actions = fetch(h5, "action_sequence", indices)
        previous = fetch(h5, "prev_drone_state", indices)
        current = fetch(h5, "drone_state", indices)
    finite = (np.isfinite(actions).all(axis=(1, 2)) &
              np.isfinite(previous).all(axis=1) &
              np.isfinite(current).all(axis=1))
    return indices[finite], previous[finite].astype(np.float64), \
        current[finite].astype(np.float64), actions[finite].astype(np.float64)


def pose_summary(previous, current, actions, model):
    errors = []
    for p, c, a in zip(previous, current, actions):
        pred = next_state(p, a, model)
        errors.append((np.linalg.norm(pred[:3] - c[:3]),
                       abs(angle_delta(yaw(pred), yaw(c))),
                       np.linalg.norm(pred[7:10] - c[7:10])))
    err = np.asarray(errors)
    return {"position_mae_m": float(err[:, 0].mean()),
            "yaw_mae_rad": float(err[:, 1].mean()),
            "velocity_mae_mps": float(err[:, 2].mean())}


def transform_from_initial(initial, predicted):
    # The collection LiDAR is attached with yaw only; body roll/pitch must
    # not rotate its rays.  This is checked against the cached true transform.
    ri = Rotation.from_euler("z", yaw(initial)).as_matrix()
    rp = Rotation.from_euler("z", yaw(predicted)).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rp.T @ ri
    transform[:3, 3] = rp.T @ (initial[:3] - predicted[:3])
    return transform


def rollout(data_root, raw_root, manifest_path, model, limit):
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["rows"][:limit] if limit else manifest["rows"]
    trajectories = np.asarray([r["trajectory_indices"] for r in rows], dtype=np.int64)
    flat = trajectories.reshape(-1)
    with h5py.File(data_root / "navrl_static_test.h5", "r") as h5:
        unique, inverse = np.unique(flat, return_inverse=True)
        actions = h5["action_sequence"][unique][inverse].reshape(len(rows), 10, 10, 3)
        targets = h5["range_values"][unique][inverse].reshape(len(rows), 10, 2, 108, 20)
        actual_states = h5["drone_state"][unique][inverse].reshape(len(rows), 10, 13)
        initial = fetch(h5, "prev_drone_state", trajectories[:, 0])
        initial_frames = fetch(h5, "prev_range_values", trajectories[:, 0])
        first_true_transform = fetch(h5, "prev_trans_mat", trajectories[:, 0])
    ray_cache = {seed: load_rays(raw_root, "test", seed)
                 for seed in sorted({r["seed"] for r in rows})}
    output = []
    excluded = 0
    transform_max_translation_difference_m = 0.0
    transform_max_rotation_difference_rad = 0.0
    for i, row in enumerate(rows):
        if not np.isfinite(actions[i]).all() or not np.isfinite(initial[i]).all():
            excluded += 1
            continue
        rays, azimuth, elevation = ray_cache[row["seed"]]
        constructed = transform_from_initial(initial[i], actual_states[i, 0])
        transform_max_translation_difference_m = max(
            transform_max_translation_difference_m,
            float(np.linalg.norm(constructed[:3, 3] - first_true_transform[i, :3, 3])))
        transform_max_rotation_difference_rad = max(
            transform_max_rotation_difference_rad,
            float(Rotation.from_matrix(constructed[:3, :3] @
                 first_true_transform[i, :3, :3].T).magnitude()))
        if (transform_max_translation_difference_m > 1e-3 or
                transform_max_rotation_difference_rad > 1e-3):
            raise ValueError("Drone-to-LiDAR transform disagrees with cached true transform")
        states = {"velocity": initial[i].copy(), "learned": initial[i].copy()}
        for h in range(10):
            for method in states:
                states[method] = next_state(states[method], actions[i, h],
                                            model if method == "learned" else None)
            if h + 1 not in (1, 5, 10):
                continue
            item = {"source_index": row["source_index"], "seed": row["seed"],
                    "horizon": h + 1}
            for method, state in states.items():
                transform = transform_from_initial(initial[i], state)
                projected = warp_frame(initial_frames[i], transform,
                                       rays, azimuth, elevation)
                item[method] = lidar_metric(projected, targets[i, h], rays, 0)
                item[f"{method}_position_error_m"] = float(
                    np.linalg.norm(state[:3] - actual_states[i, h, :3]))
                item[f"{method}_yaw_error_rad"] = float(abs(angle_delta(
                    yaw(state), yaw(actual_states[i, h]))))
            oracle = warp_frame(initial_frames[i],
                                transform_from_initial(initial[i], actual_states[i, h]),
                                rays, azimuth, elevation)
            item["oracle_pose"] = lidar_metric(oracle, targets[i, h], rays, 0)
            item["copy"] = lidar_metric(initial_frames[i], targets[i, h], rays, 0)
            output.append(item)
    summary = {}
    for h in (1, 5, 10):
        selected = [r for r in output if r["horizon"] == h]
        summary[str(h)] = {method: {
            "cd_paper_m2": float(np.mean([r[method]["cd_paper_m2"] for r in selected])),
            "mask_f1": float(2 * sum(r[method]["tp"] for r in selected) /
                             max(1, sum(2 * r[method]["tp"] + r[method]["fp"] +
                                        r[method]["fn"] for r in selected))),
            "empty_cloud_cases": int(sum(r[method]["empty_cloud"] for r in selected)),
            **({"position_mae_m": float(np.mean([
                r[f"{method}_position_error_m"] for r in selected])),
                "yaw_mae_rad": float(np.mean([
                    r[f"{method}_yaw_error_rad"] for r in selected]))}
               if method in ("velocity", "learned") else {})
        } for method in ("copy", "velocity", "learned", "oracle_pose")}
    return {"rows": output, "summary": summary, "selected": len(rows) - excluded,
            "excluded_nonfinite_actions_or_state": excluded,
            "first_step_transform_check": {
                "max_translation_difference_m": transform_max_translation_difference_m,
                "max_rotation_difference_rad": transform_max_rotation_difference_rad},
            "manifest": str(manifest_path),
            "note": "Pose oracle is diagnostic only; all methods warp the initial observed cloud."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs" /
                        "representative_baseline" / "test_10frame_sample_manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "epona_probe")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train_idx, p_train, c_train, a_train = load_pairs(args.data_root, "train")
    val_idx, p_val, c_val, a_val = load_pairs(args.data_root, "val")
    x_train = features(p_train, a_train)
    y_train = residual_targets(p_train, c_train)
    scores = {}
    models = {}
    for alpha in (1.0, 100.0, 10000.0):
        candidate = fit_model(x_train, y_train, alpha)
        scores[str(alpha)] = pose_summary(p_val, c_val, a_val, candidate)
        models[alpha] = candidate
    best_alpha = min(models, key=lambda alpha: scores[str(alpha)]["position_mae_m"])
    model = models[best_alpha]
    np.savez(args.out / "motion_ridge.npz", **model)
    report = {"train_pairs": len(train_idx), "val_pairs": len(val_idx),
              "action_source": "finite action_sequence; world-frame executed commands",
              "best_alpha_val_only": best_alpha, "validation": scores,
              "validation_velocity": pose_summary(p_val, c_val, a_val, None)}
    shuffled = np.random.default_rng(42).permutation(len(a_val))
    report["validation_shuffled_action_sensitivity"] = pose_summary(
        p_val, c_val, a_val[shuffled], model)
    save_json(args.out / "motion_fit.json", report)
    result = rollout(args.data_root, args.raw_root, args.manifest, model, args.limit)
    result["fit"] = report
    save_json(args.out / "test_motion_rollout.json", result)
    print(json.dumps({"fit": report, "test_rollout": result["summary"],
                      "selected": result["selected"], "excluded": result[
                          "excluded_nonfinite_actions_or_state"]}), flush=True)


if __name__ == "__main__":
    main()
