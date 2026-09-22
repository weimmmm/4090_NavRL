# Fast-WAM attribution

The joint world/action flow-training design in
`lidar_wam/models/action_expert.py` is adapted from the official Fast-WAM
implementation:

- Repository: https://github.com/yuantianyuan01/FastWAM
- Reference files: `src/fastwam/models/wan22/fastwam.py`, `action_dit.py`, and
  `mot.py`
- Retrieved: 2026-09-20
- License: MIT; preserved in `LICENSE`

The local implementation does not copy the Wan video expert.  It replaces
Fast-WAM's layer-wise Video-DiT/Action-DiT mixture with a shared causal LiDAR
observation encoder because this project uses a LaGen 2-D UNet world expert.
It retains separate video/action flow objectives and the rule that the action
expert can attend only to the clean current observation, never future ground
truth observations.
