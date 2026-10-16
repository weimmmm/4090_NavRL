"""Offline SFT update shrinkage; never modifies a source or flight controller.

This is a separately evaluated algorithm ablation, not evidence of improved
navigation. Only the two explicitly fine-tuned Flow blocks may differ.
"""
import argparse
import copy
from pathlib import Path

import torch

ALLOWED = ("diffusion.blocks.6.", "diffusion.blocks.7.")


def shrink_checkpoint(base, tuned, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Shrinkage must lie in [0,1]")
    for name in ("format", "architecture", "condition_stats", "vae_checkpoint", "action_interface"):
        if base.get(name) != tuned.get(name):
            raise ValueError(f"Checkpoint contract differs: {name}")
    if base["model"].keys() != tuned["model"].keys():
        raise ValueError("Model keys differ")
    state, changed = {}, []
    for name, original in base["model"].items():
        value = tuned["model"][name]
        if original.shape != value.shape or original.dtype != value.dtype:
            raise ValueError(f"Tensor contract differs: {name}")
        if not torch.isfinite(original).all() or not torch.isfinite(value).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
        if torch.equal(original, value):
            state[name] = original.clone()
            continue
        if not name.startswith(ALLOWED) or not original.is_floating_point():
            raise ValueError(f"A frozen parameter changed: {name}")
        changed.append(name)
        state[name] = (original.clone() if alpha == 0 else value.clone() if alpha == 1
                       else torch.lerp(original.float(), value.float(), alpha).to(original.dtype))
    result = copy.deepcopy({k: v for k, v in tuned.items()
                            if k not in ("model", "optimizer", "online_sft_optimizer")})
    result["model"] = state
    result["sft_update_shrinkage"] = {
        "alpha": float(alpha), "base_step": int(base["step"]),
        "sft_updates": int(tuned["step"]), "changed_tensor_names": changed,
        "optimizer_removed": True, "closed_loop_improvement_verified": False,
    }
    return result


def main():
    from lidar_wam.runner.super_action_expert import _atomic_save

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in (args.base.resolve(), args.tuned.resolve()):
        parser.error("Output must be a new path, never an existing/source checkpoint")
    result = shrink_checkpoint(torch.load(args.base, map_location="cpu", weights_only=False),
                               torch.load(args.tuned, map_location="cpu", weights_only=False),
                               args.alpha)
    result["sft_update_shrinkage"].update(base_path=str(args.base), tuned_path=str(args.tuned))
    _atomic_save(result, args.output)
    print(result["sft_update_shrinkage"], flush=True)


if __name__ == "__main__":
    main()
