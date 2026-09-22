"""Join existing fixed-manifest one-step reports by original HDF5 row index."""

import json
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SOURCES = {
    "legacy": OUT / "representative_baseline/test_legacy_squared_one_step.json",
    "nwm": OUT / "representative_baseline/test_nwm_squared_one_step.json",
    "delta": OUT / "action_latent_delta/test_one_step.json",
    "residual": OUT / "geometry_residual_diffusion/test_one_step.json",
    "hard": OUT / "hard_geometry_inpaint/test.json",
    "gate": OUT / "completion_multisample/test.json",
    "oracle": OUT / "completion_oracle/test.json",
}
METHODS = {
    "copy": ("legacy", "copy"),
    "velocity_pose": ("legacy", "velocity_pose"),
    "action_pose": ("gate", "warp"),
    "lagen_unet_8h": ("legacy", "lagen_unet_8h"),
    "lagen_residual": ("legacy", "lagen_residual"),
    "nwm_cdit": ("nwm", "nwm"),
    "deterministic_latent": ("delta", "deterministic"),
    "geometry_residual_diffusion": ("residual", "prediction"),
    "hard_warp_diffusion": ("hard", "all_hits"),
    "multisample_gate_k8": ("gate", "k8_c6_s0.25"),
    "gt_completion_oracle_diagnostic": ("oracle", "oracle_add"),
}


def main():
    reports = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    lookup = {name: {int(row["source_index"]): row for row in report["rows"]}
              for name, report in reports.items()}
    keys = set.intersection(*(set(rows) for rows in lookup.values()))
    if len(keys) != 512:
        raise ValueError(f"Expected 512 common fixed-manifest source rows, got {len(keys)}")
    manifest = json.loads((OUT / "representative_baseline/sample_manifest.json").read_text())
    selected = {int(row["source_index"]) for row in manifest["splits"]["test"]}
    if keys != selected:
        raise ValueError("Report sample set differs from fixed manifest")
    paired = [index for index in sorted(keys) if all(
        not lookup[src][index][field]["empty_cloud"]
        for src, field in METHODS.values())]
    methods = {}
    for name, (src, field) in METHODS.items():
        rows = lookup[src]
        methods[name] = {
            "all_512_cd_paper_m2": fmean(
                rows[i][field]["cd_paper_m2"] for i in keys),
            "common_nonempty_cd_paper_m2": fmean(
                rows[i][field]["cd_paper_m2"] for i in paired),
            "empty_cloud_cases": sum(rows[i][field]["empty_cloud"] for i in keys),
            "source": str(SOURCES[src].relative_to(ROOT)),
        }
    summary = {"definition": "sum of two mean squared nearest-neighbor distances, m^2",
               "all_samples": 512, "common_nonempty_samples": len(paired),
               "empty_cloud_policy": "if only one cloud is empty, 200 m^2; both empty, 0",
               "split": "test", "seeds": [18, 19],
               "gt_oracle_is_deployable": False, "methods": methods,
               "common_source_indices": paired}
    path = OUT / "one_step_squared_overview.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"common_nonempty": len(paired), "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
