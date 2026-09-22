# Bundled source provenance

This project runs without importing the sibling `LaGen` checkout.

| Local path | Source | Purpose |
| --- | --- | --- |
| `third_party/diffusers/src/` and `third_party/diffusers/LICENSE` | `LaGen/third_party/diffusers/` | Vendored Diffusers `AutoencoderKL`, `UNet2DConditionModel`, DDPM/DDIM schedulers and dependencies |
| `lidar_wam/vae/config.json` | `LaGen/lagen/vae/config.json` | Original two-channel VAE architecture configuration |
| `lidar_wam/runner/utils.py` and `lidar_wam/vae/circular/utils.py` | `LaGen_NavRL/lagen/runner/utils.py` | Circular horizontal convolution, downsampling and attention replacement for the active NavRL VAE |
| `lidar_wam/vae/circular/{config.json,diffusion_pytorch_model.safetensors,best_validation.json}` | `lagen_navrl_runtime/outputs/vae_navrl_108x20/` | Active 29,500-step circular VAE checkpoint |
| `scripts/convert_static_dataset.py`, `scripts/validate_cache.py` | `NavRL/isaac-training/training_legan/scripts/` | Conversion and validation of the original NavRL LiDAR dataset |
| `third_party/nwm/models.py`, `third_party/nwm/diffusion/` | [facebookresearch/nwm](https://github.com/facebookresearch/nwm), commit `3f6cd8e70d6f2d1e2b9684acff510710135f0f41` | NWM CDiT backbone and original diffusion training/sampling implementation |
| `third_party/fastwam/` and `lidar_wam/models/action_expert.py` | [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM), retrieved 2026-09-20 | MIT-licensed reference for joint world/action flow training and the causal ActionDiT interface; adapted to the existing LaGen UNet without Wan |

The copied Diffusers source retains its license in `third_party/diffusers/LICENSE`. The circular VAE utility and NavRL conversion source retain their upstream licenses in `lidar_wam/vae/circular/LICENSE` and `scripts/NAVRL_LICENSE`.
NWM's copied source and model weights are CC BY-NC 4.0; its license is retained at `third_party/nwm/LICENSE.md`. The adapter in `lidar_wam/runner/nwm_predictor.py` calls the copied CDiT blocks without changing upstream code.
Fast-WAM's license and detailed adaptation notes are retained in `third_party/fastwam/LICENSE` and `third_party/fastwam/UPSTREAM.md`.
