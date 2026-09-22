"""Pair geometry and original UNet rollouts on exactly the same test starts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs"
MOTION = ROOT / "epona_probe" / "test_motion_rollout.json"
ORIGINAL = ROOT / "representative_baseline" / "test_lagen_style_10frame_metrics.json"


def main():
    motion = json.loads(MOTION.read_text())
    original = json.loads(ORIGINAL.read_text())
    lookup = {(row["source_index"], row["horizon"]): row
              for row in original["rows"]}
    methods = ("original_unet", "copy", "velocity", "learned", "oracle_pose")
    result = {"definition": "paper-formula bidirectional squared Chamfer, m^2; "
                            "paired samples require all five point clouds nonempty",
              "transform_check": motion["first_step_transform_check"],
              "horizons": {}}
    for horizon in (1, 5, 10):
        rows = [row for row in motion["rows"] if row["horizon"] == horizon]
        def metric(row, method):
            if method == "original_unet":
                return lookup[(row["source_index"], horizon)]["model"]
            return row[method]
        groups = {"all": rows}
        for seed in sorted({row["seed"] for row in rows}):
            groups[f"seed_{seed}"] = [row for row in rows if row["seed"] == seed]
        result["horizons"][str(horizon)] = {}
        for label, group in groups.items():
            paired = [row for row in group
                      if all(not metric(row, method)["empty_cloud"]
                             for method in methods)]
            if not paired:
                continue
            result["horizons"][str(horizon)][label] = {
                "total": len(group), "paired_nonempty": len(paired),
                "paired_cd_paper_m2": {
                    method: sum(metric(row, method)["cd_paper_m2"] for row in paired) /
                            len(paired) for method in methods},
                "empty_cases": {method: sum(metric(row, method)["empty_cloud"]
                                            for row in group) for method in methods}}
    path = ROOT / "epona_probe" / "paired_comparison.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"report": str(path), "horizons": {
        h: result["horizons"][h]["all"] for h in result["horizons"]}}))


if __name__ == "__main__":
    main()
