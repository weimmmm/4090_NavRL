"""Audit non-finite executed velocity commands and their evaluation impact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from lidar_wam.runner import stage1


def mean(rows, key):
    return float(np.mean([row[key] for row in rows])) if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=stage1.DATA)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    report = {}
    for split in ("train", "val", "test"):
        with h5py.File(data_root / f"navrl_static_{split}.h5", "r") as h5:
            valid, _ = stage1.valid_indices(h5, split)
            finite = np.isfinite(h5["action_sequence"][valid]).all(axis=(1, 2))
            clean = np.zeros(len(h5["action_sequence"]), dtype=bool)
            clean[valid] = finite
        result = {"valid_normalized_action_rows": len(valid),
                  "finite_executed_action_rows": int(finite.sum()),
                  "nonfinite_executed_action_rows": int((~finite).sum())}
        if split != "train":
            for strength in ("1", "0.25"):
                path = (stage1.OUT / "evaluation" /
                        f"{split}_step5000_strength{strength}_n256_circular_causal_ddim20.json")
                if not path.exists():
                    continue
                evaluation = json.loads(path.read_text())
                rows = [row for seed_rows in evaluation["per_seed"].values()
                        for row in seed_rows]
                groups = {"finite": [row for row in rows if clean[row["source_index"]]],
                          "nonfinite": [row for row in rows if not clean[row["source_index"]]]}
                result[f"evaluation_strength_{strength}"] = {
                    name: {"samples": len(group),
                           "copy_chamfer_m": mean(group, "copy_chamfer_m"),
                           "prediction_chamfer_m": mean(group, "prediction_chamfer_m"),
                           "shuffled_action_chamfer_m": mean(group, "shuffled_action_chamfer_m")}
                    for name, group in groups.items()}
        report[split] = result
    path = stage1.OUT / "evaluation" / "executed_action_finite_audit.json"
    stage1.save_json(path, report)
    print(json.dumps(report), flush=True)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
