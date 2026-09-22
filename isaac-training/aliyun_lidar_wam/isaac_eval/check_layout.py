"""Check that the evaluator does not require another NavRL source directory."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "isaac_eval"
REQUIRED = [
    ROOT / "lidar_WAM" / "lidar_wam" / "vae" / "circular" / "diffusion_pytorch_model.safetensors",
    EVAL / "navigation_env.py",
    EVAL / "environment_file.py",
    EVAL / "environments" / "static350_seed18_n256.pt",
    EVAL / "evaluate_ppo.py",
    EVAL / "ppo_policy.py",
    EVAL / "ppo_lidar_encoder.py",
    EVAL / "collect_dataset.py",
    EVAL / "capacity_smoke.py",
    EVAL / "collect_all.py",
    EVAL / "dataset_io.py",
    EVAL / "finalize_dataset.py",
    EVAL / "replay_verify.py",
    EVAL / "checkpoints" / "ppo_dataset_collector.pt",
    EVAL / "policy.py",
    EVAL / "cfg" / "train.yaml",
    EVAL / "third_party" / "OmniDrones" / "omni_drones",
    EVAL / "third_party" / "orbit" / "source" / "extensions" / "omni.isaac.orbit",
    EVAL / "third_party" / "rl" / "torchrl",
    EVAL / "third_party" / "rl" / "torchrl" / "_torchrl.so",
    EVAL / "third_party" / "tensordict" / "tensordict",
    EVAL / "third_party" / "tensordict" / "tensordict" / "_tensordict.so",
]
FORBIDDEN_IMPORTS = {"training_legan", "training_lidar", "training", "baseline"}


def main():
    missing = [str(path) for path in REQUIRED if not path.exists()]
    forbidden = []
    for path in EVAL.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in FORBIDDEN_IMPORTS:
                    forbidden.append({"file": path.name, "import": name})
    report = {
        "project_root": str(ROOT),
        "missing_required_paths": missing,
        "forbidden_external_project_imports": forbidden,
        "passed": not missing and not forbidden,
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
