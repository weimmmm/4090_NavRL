"""Paired, nonempty LaGen-style squared Chamfer summary of saved rollouts."""

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "representative_baseline"
METHODS = ("model", "copy", "velocity")


def cluster_interval(rows, scenes, n_boot=2000):
    """Bootstrap scene tokens within each test seed to respect overlapping starts."""
    groups = {seed: {} for seed in (18, 19)}
    for row in rows:
        scene = scenes[row["source_index"]]
        groups[row["seed"]].setdefault(scene, []).append(row)
    rng = random.Random(42)
    estimates = []
    for _ in range(n_boot):
        total = count = 0
        for by_scene in groups.values():
            labels = list(by_scene)
            for _ in labels:
                selected = by_scene[rng.choice(labels)]
                total += sum(r["model"]["cd_paper_m2"] -
                             r["velocity"]["cd_paper_m2"] for r in selected)
                count += len(selected)
        estimates.append(total / count)
    estimates.sort()
    return [estimates[int(.025 * n_boot)], estimates[int(.975 * n_boot)]]


def main():
    report = json.loads((ROOT / "test_lagen_style_10frame_metrics.json").read_text())
    manifest = json.loads((ROOT / "test_10frame_sample_manifest.json").read_text())
    scenes = {r["source_index"]: r["scene_token"] for r in manifest["rows"]}
    summary = {}
    for horizon in range(1, 11):
        all_rows = [r for r in report["rows"] if r["horizon"] == horizon]
        paired = [r for r in all_rows if all(not r[m]["empty_cloud"] for m in METHODS)]
        values = {method: sum(r[method]["cd_paper_m2"] for r in paired) / len(paired)
                  for method in METHODS}
        summary[str(horizon)] = {
            "samples_total": len(all_rows), "samples_paired_nonempty": len(paired),
            "empty_cloud_cases": {m: sum(r[m]["empty_cloud"] for r in all_rows)
                                  for m in METHODS},
            "paired_cd_paper_m2": values,
            "paired_cd_released_code_m2": {m: value / 2 for m, value in values.items()},
            "model_minus_velocity_cd_m2": values["model"] - values["velocity"],
            "model_minus_velocity_scene_bootstrap_95pct_ci_m2":
                cluster_interval(paired, scenes)}
    output = {"version": 1, "definition": "Squared Chamfer on the same starts where model, copy, velocity and GT have nonempty point clouds; paper formula sums both directions. Other ray and mask metrics in main report use all 512 starts.",
              "summary": summary}
    path = ROOT / "test_lagen_style_10frame_paired_summary.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for horizon in range(1, 11):
        s = summary[str(horizon)]
        m = report["summary"][str(horizon)]["model"]["all"]
        print(horizon, s["samples_paired_nonempty"],
              *(round(s["paired_cd_paper_m2"][method], 3) for method in METHODS),
              round(m["gt_hit_l1_m"], 3),
              round(m["gt_hit_absrel_percent"], 1),
              round(m["mask_f1"], 3),
              s["empty_cloud_cases"],
              [round(v, 3) for v in s["model_minus_velocity_scene_bootstrap_95pct_ci_m2"]])
    print(path)


if __name__ == "__main__":
    main()
