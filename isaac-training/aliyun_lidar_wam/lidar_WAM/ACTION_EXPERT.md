# LaGen UNet + Fast-WAM-style Action Expert

This experiment uses the existing direct-`t+3` LaGen UNet checkpoint. It does
not load Wan. The implementation borrows Fast-WAM's causal joint-training
principle and adapts it to a 2-D UNet world expert.

```text
clean current LiDAR latent ── shared circular encoder ─┬─ Action Flow Transformer
                                                      │       ↓
goal + current proprio + previous actions ────────────┘   future 30 PPO actions

shared LiDAR summary + GT action chunks + noisy t+3 latent
                         └─────────────────────────────── LaGen UNet world loss
```

The action branch never receives the true future latent, future range image,
future state, or the future target pose. During training the world branch uses
the demonstrated future action chunks, as the existing action-conditioned WM
does. The shared encoder and a zero-initialized UNet conditioning residual are
the coupling between the branches. At initialization that residual is exactly
zero, so loading the existing WM does not change its output.

The policy target is three ordered `10×3` normalized PPO action chunks. Action
flow is trained in standardized logit space and converted back with a sigmoid,
so exported actions remain in `[0,1]`. At deployment, generate 30 actions but
execute only the first 10 simulator steps, observe a new LiDAR frame, and plan
again.

## Inputs

- Current circular-VAE latent: `[4,27,5]`.
- Goal: body-frame relative XYZ and Euclidean distance.
- Proprioception: height, body-frame linear/angular velocity, and body-frame
  gravity vector.
- Previous action chunk: `[10,3]` plus a validity mask. It is zero/masked at
  the start of an episode.

Goal metadata comes from immutable `frames.jsonl`; it is deliberately not
invented from the HDF5 cache. World targets and actions still come from the
existing latent/HDF5 caches and preserve the seed split.

## Commands

```bash
cd /mnt/workspace/wam_trining/trining/lidar_WAM
PY=/mnt/workspace/LaGen/.venv-ppu/bin/python
DATA=/mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache
RAW=/mnt/workspace/wam_trining/data/navrl_static_100k
WM=outputs/world_direct_t3/latest.pt

$PY -m lidar_wam.runner.action_expert_joint inspect \
  --data-root "$DATA" --raw-data-root "$RAW"

$PY -m lidar_wam.runner.action_expert_joint smoke \
  --data-root "$DATA" --raw-data-root "$RAW" --world-checkpoint "$WM"

# Required 128-trajectory fitting check.
$PY -u -m lidar_wam.runner.action_expert_joint train --overfit \
  --data-root "$DATA" --raw-data-root "$RAW" --world-checkpoint "$WM" \
  --steps 1000 --batch-size 128

# Full joint run.
$PY -u -m lidar_wam.runner.action_expert_joint train \
  --data-root "$DATA" --raw-data-root "$RAW" --world-checkpoint "$WM" \
  --steps 30000 --batch-size 256

$PY -m lidar_wam.runner.action_expert_joint evaluate \
  --data-root "$DATA" --raw-data-root "$RAW" \
  --checkpoint outputs/action_expert_joint/best.pt --split val
```

The full run uses action/shared/world learning rates `1e-4`, `5e-5`, and
`1e-6`. Use `--freeze-world` for the ablation where only the shared adapter and
Action Expert train. Checkpoint selection uses validation action MAE and also
reports the mean-action and repeat-last-action baselines. Simulation success
rate still requires deployment in Isaac Sim; offline imitation error alone is
not a navigation-success result.
