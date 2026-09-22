# Vendored runtime sources

The evaluator keeps the following source trees locally so it does not import
another directory from the surrounding NavRL checkout:

| Local path | Copied from | License file |
| --- | --- | --- |
| `third_party/OmniDrones/omni_drones` | `isaac-training/third_party/OmniDrones/omni_drones` | `third_party/OmniDrones/LICENSE` |
| `third_party/orbit/source` | `isaac-training/third_party/orbit/source` | `third_party/orbit/LICENSE` |
| `third_party/rl/torchrl` | `isaac-training/third_party/rl/torchrl` | `third_party/rl/LICENSE` |
| `third_party/tensordict/tensordict` | `isaac-training/third_party/tensordict/tensordict` | `third_party/tensordict/LICENSE` |

The vendored TensorDict and TorchRL folders also include their Python 3.10
Linux extension modules, `_tensordict.so` and `_torchrl.so`, copied from the
matching `/opt/navrl-deps` builds in the Isaac Sim 2023 container. These
binaries are platform-specific and must be rebuilt if the Python/PyTorch
runtime changes.

`navigation_env.py`, `cfg/`, and the small functions in `nav_utils.py` were
copied from this NavRL repository's `training_legan` implementation for the
standalone evaluator. The corresponding NavRL license is also available at
`../lidar_WAM/scripts/NAVRL_LICENSE`.

Isaac Sim is not redistributed here. It is supplied by the user's licensed
Isaac Sim 2023 container.
