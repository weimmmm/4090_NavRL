"""Batch-evaluate SAC checkpoints on one frozen evaluation world.

The simulator is launched once, then every ``checkpoint_step_*.pt`` whose
step is at least ``--start-step`` is evaluated deterministically.  A result is
flushed to the output txt after each checkpoint, so re-running the script
automatically skips checkpoints that already have a successful result.
"""
import argparse
import os
import random
import re
import site
import sys
import traceback
import types
from pathlib import Path

# This project deliberately evaluates on physical GPU 0, just like training.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("OMNI_KIT_RENDERER", "Vulkan")
os.environ.setdefault("OMNI_KIT_NO_OPENGL_RENDERING", "1")


ROOT = Path(__file__).resolve().parents[1]


def bootstrap_project_paths():
    """Make Isaac Sim's embedded Python see this project's training packages.

    The normal NavRL conda interpreter loads these via ``.pth`` files. Isaac's
    ``python.sh`` does not, so add the same site-packages directory explicitly
    and then the three vendored project packages. ``NAVRL_SITE_PACKAGES`` can
    override the default when the conda env lives elsewhere.
    """
    site_packages = Path(
        os.environ.get(
            "NAVRL_SITE_PACKAGES",
            str(Path.home() / "miniconda3/envs/NavRL/lib/python3.10/site-packages"),
        )
    )
    if site_packages.is_dir():
        site.addsitedir(str(site_packages))

    training_root = ROOT.parents[1]
    for path in (
        training_root / "third_party" / "tensordict",
        training_root / "third_party" / "rl",
        training_root / "third_party" / "OmniDrones",
    ):
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)


bootstrap_project_paths()


def avoid_optional_omnidrones_env_imports():
    """Load only ``isaac_env`` instead of unrelated Nucleus-dependent tasks.

    Importing ``omni_drones.envs.isaac_env`` normally executes
    ``omni_drones.envs.__init__`` first. That initializer imports optional
    pinball/forest tasks, which probe a Nucleus asset server even though the
    navigation environment does not use either task. Presenting the directory
    as a package avoids that probe and leaves the required ``isaac_env`` module
    available through Python's normal submodule loading.
    """
    if "omni_drones.envs" in sys.modules:
        return
    package_dir = ROOT.parents[1] / "third_party" / "OmniDrones" / "omni_drones" / "envs"
    package = types.ModuleType("omni_drones.envs")
    package.__path__ = [str(package_dir)]
    package.__file__ = str(package_dir / "__init__.py")
    sys.modules["omni_drones.envs"] = package


def avoid_nucleus_asset_probe():
    """Keep evaluation offline when importing Orbit.

    Navigation's drone USD files are stored locally in OmniDrones.  Orbit still
    probes a Nucleus server while defining a few unused asset-path constants;
    on this machine that synchronous probe can block for a very long time.
    Provide a harmless placeholder only for this evaluator before Orbit is
    imported.  No evaluation asset is read from that placeholder.
    """
    from omni.isaac.core.utils import nucleus as nucleus_utils

    nucleus_utils.get_assets_root_path = lambda: "omniverse://localhost"

import numpy as np
import torch
from omni.isaac.kit import SimulationApp
from omegaconf import OmegaConf


CHECKPOINT_RE = re.compile(r"^checkpoint_step_(\d+)\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all SAC checkpoints from a given training step."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=ROOT / "checkpoints",
        help="Directory containing checkpoint_step_XXXXXXX.pt files.",
    )
    parser.add_argument(
        "--world",
        type=Path,
        default=ROOT / "eval_worlds" / "eval_3407_2048.pt",
        help="Frozen evaluation world (.pt).",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=3000,
        help="Only evaluate checkpoints whose numeric step is at least this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "checkpoints" / "eval_3407_2048_reach_goal_from_3000.txt",
        help="Append-only text result file.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Evaluate again even if a successful result for that step is in --output.",
    )
    return parser.parse_args()


def configure_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def checkpoint_paths(checkpoints_dir: Path, start_step: int):
    paths = []
    for path in checkpoints_dir.glob("checkpoint_step_*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if match is not None and int(match.group(1)) >= start_step:
            paths.append((int(match.group(1)), path))
    return sorted(paths, key=lambda item: item[0])


def completed_steps(output_path: Path):
    """Read successful records only; failed checkpoints will be retried."""
    if not output_path.is_file():
        return set()
    completed = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3 and fields[0].isdigit() and fields[2] != "ERROR":
            completed.add(int(fields[0]))
    return completed


def append_result(output_path: Path, step: int, checkpoint: Path, reach_goal, extra=""):
    new_file = not output_path.exists() or output_path.stat().st_size == 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        if new_file:
            file.write("step\tcheckpoint\treach_goal\tcollision\tout_of_bound\treturn\tnote\n")
        if reach_goal == "ERROR":
            file.write(f"{step}\t{checkpoint.name}\tERROR\t\t\t\t{extra}\n")
        else:
            file.write(
                f"{step}\t{checkpoint.name}\t{reach_goal:.6f}\t"
                f"{extra['collision']:.6f}\t{extra['out_of_bound']:.6f}\t"
                f"{extra['return']:.6f}\t\n"
            )
        file.flush()
        os.fsync(file.fileno())


def rollout_metrics(trajs):
    """Use the first terminal transition from each of the frozen 2048 worlds."""
    done = trajs["next", "done"].to(torch.bool)
    if done.ndim > 2 and done.shape[-1] == 1:
        done = done.squeeze(-1)
    has_done = done.any(dim=1)
    first_done = torch.argmax(done.long(), dim=1)
    first_done = torch.where(
        has_done,
        first_done,
        torch.full_like(first_done, done.shape[1] - 1),
    )

    def at_first_done(value):
        # ``first_done`` is [num_envs], while a stat is typically
        # [num_envs, time, 1].  ``take_along_dim`` requires the index to have
        # the same rank as the input, hence [num_envs, 1, 1] here (and the
        # analogous shape for any trailing stat dimensions).
        index = first_done.reshape(first_done.shape + (1,) * (value.ndim - 1))
        return torch.take_along_dim(value, index, dim=1).reshape(-1)

    result = {"finished_rate": has_done.float().mean().item()}
    for key, value in trajs["next", "stats"].items():
        result[str(key)] = at_first_done(value.float()).mean().item()
    return result


def main():
    args = parse_args()
    checkpoints_dir = args.checkpoints_dir.resolve()
    world_path = args.world.resolve()
    output_path = args.output.resolve()
    if not checkpoints_dir.is_dir():
        raise NotADirectoryError(checkpoints_dir)
    if not world_path.is_file():
        raise FileNotFoundError(world_path)

    candidates = checkpoint_paths(checkpoints_dir, args.start_step)
    if not candidates:
        raise RuntimeError(
            f"No checkpoint_step_*.pt with step >= {args.start_step} in {checkpoints_dir}"
        )
    finished = set() if args.rerun else completed_steps(output_path)
    candidates = [(step, path) for step, path in candidates if step not in finished]
    if not candidates:
        print(f"[BATCH-EVAL] all requested checkpoints are already in {output_path}")
        return

    snapshot = torch.load(world_path, map_location="cpu")
    meta = snapshot["meta"]
    cfg = OmegaConf.create(meta["resolved_config"])
    cfg.env.num_envs = int(meta["num_envs"])
    cfg.env.max_episode_length = int(meta["max_episode_length"])
    cfg.env.num_obstacles = int(meta["num_obstacles_static"])
    cfg.env_dyn.num_obstacles = int(meta["num_obstacles_dynamic"])
    cfg.seed = int(meta["world_seed"])
    # The actual collision/lidar terrain is generated locally at /World/ground.
    # Do not instantiate Orbit's optional Nucleus-hosted visual grid plane.
    cfg.skip_default_ground_plane = True
    configure_seed(int(cfg.seed))

    print(
        f"[BATCH-EVAL] {len(candidates)} checkpoints | steps "
        f"{candidates[0][0]}..{candidates[-1][0]} | world={world_path.name}",
        flush=True,
    )
    sim_app = SimulationApp(
        {
            "headless": True,
            "anti_aliasing": 1,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    try:
        avoid_optional_omnidrones_env_imports()
        avoid_nucleus_asset_probe()
        from evalenv import EvalEnv
        from sac import SAC
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        from torchrl.envs.transforms import Compose, TransformedEnv
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        base_env = EvalEnv(cfg, str(world_path))
        controller = LeePositionController(9.81, base_env.drone.params).to(cfg.device)
        env = TransformedEnv(
            base_env, Compose(VelController(controller, yaw_control=False))
        ).eval()
        policy = SAC(cfg.algo, env.observation_spec, env.action_spec, cfg.device).eval()

        for ordinal, (step, checkpoint) in enumerate(candidates, start=1):
            print(
                f"[BATCH-EVAL] [{ordinal}/{len(candidates)}] step={step} "
                f"checkpoint={checkpoint.name}",
                flush=True,
            )
            try:
                # Resetting the frozen environment returns every drone and each
                # dynamic obstacle to exactly the same saved trajectory start.
                env.set_seed(int(cfg.seed))
                state_dict = torch.load(checkpoint, map_location=cfg.device)
                policy.load_state_dict(state_dict)
                policy.eval()
                with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
                    trajs = env.rollout(
                        max_steps=int(cfg.env.max_episode_length),
                        policy=policy,
                        auto_reset=True,
                        break_when_any_done=False,
                        return_contiguous=False,
                    )
                values = rollout_metrics(trajs)
                append_result(
                    output_path,
                    step,
                    checkpoint,
                    values["reach_goal"],
                    {
                        "collision": values["collision"],
                        "out_of_bound": values["out_of_bound"],
                        "return": values["return"],
                    },
                )
                print(
                    f"[BATCH-EVAL] step={step} reach_goal="
                    f"{values['reach_goal']:.6f} (written to {output_path.name})",
                    flush=True,
                )
            except Exception as exc:
                detail = f"{type(exc).__name__}: {str(exc).replace(chr(9), ' ')}"
                append_result(output_path, step, checkpoint, "ERROR", detail)
                print(f"[BATCH-EVAL] step={step} FAILED: {detail}", flush=True)
                traceback.print_exc()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()