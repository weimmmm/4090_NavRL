"""Produce paired, nonempty comparisons for geometry-guided DDIM outputs."""

import json
from pathlib import Path

ROOT = (Path(__file__).resolve().parents[1] / "outputs" / "epona_probe" /
        "guided_diffusion")


def summarize(rows, methods):
    paired = [row for row in rows if all(not row[m]["empty_cloud"] for m in methods)]
    return {"total": len(rows), "paired_nonempty": len(paired),
            "paired_cd_paper_m2": {
                method: sum(row[method]["cd_paper_m2"] for row in paired) /
                        len(paired) for method in methods},
            "empty_cases": {method: sum(row[method]["empty_cloud"]
                                        for row in rows) for method in methods}}


def main():
    result = {}
    for split in ("validation", "test"):
        data = json.loads((ROOT / f"{split}.json").read_text())
        methods = tuple(data["summary"])
        rows = data["rows"]
        result[split] = {"all": summarize(rows, methods)}
        for seed in sorted({row["seed"] for row in rows}):
            result[split][f"seed_{seed}"] = summarize(
                [row for row in rows if row["seed"] == seed], methods)
    path = ROOT / "paired_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(path), "validation": result["validation"]["all"],
                      "test": result["test"]["all"]}))


if __name__ == "__main__":
    main()
