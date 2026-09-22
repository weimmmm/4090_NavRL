# Wan2.2 LiDAR 1-to-4 world model

This experiment keeps the trained circular NavRL VAE frozen and adapts the
Wan2.2-TI2V-5B Video DiT to five latent frames: the current observation and
four future observations. Each future transition has its own ten normalized
PPO actions. The existing LaGen UNet experiments are not modified.

The required Wan DiT checkpoint is intentionally supplied by the user with
`--wan-checkpoint`; this prevents an accidental random-init run. A random
model is available only with `--allow-random` for the pretrained-versus-random
ablation.

```bash
cd /mnt/workspace/wam_trining/trining/lidar_WAM
export PY=/mnt/workspace/LaGen/.venv-ppu/bin/python
export DATA=/mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache
export WAN=/mnt/workspace/models/Wan2.2-TI2V-5B

$PY -m lidar_wam.runner.wan_lidar_world inspect-sequences \
  --data-root "$DATA" --latent-root outputs/latents_circular

$PY -m lidar_wam.runner.wan_lidar_world prepare \
  --data-root "$DATA" --latent-root outputs/latents_circular \
  --output outputs/wan_lidar_1to4

$PY -m lidar_wam.runner.wan_lidar_world check-load \
  --data-root "$DATA" --latent-root outputs/latents_circular \
  --wan-checkpoint "$WAN"

# First fit only 128 clips. This is the required shape/conditioning gate.
$PY -m lidar_wam.runner.wan_lidar_world train --overfit --steps 1000 \
  --data-root "$DATA" --latent-root outputs/latents_circular \
  --wan-checkpoint "$WAN" --output outputs/wan_lidar_1to4_overfit

# Full adapter + LoRA training. Add --decoded-aux only after the overfit gate.
$PY -m lidar_wam.runner.wan_lidar_world train --steps 20000 \
  --data-root "$DATA" --latent-root outputs/latents_circular \
  --wan-checkpoint "$WAN" --output outputs/wan_lidar_1to4

$PY -m lidar_wam.runner.wan_lidar_world evaluate --split val \
  --data-root "$DATA" --latent-root outputs/latents_circular \
  --raw-root /mnt/workspace/wam_trining/data/navrl_static_100k \
  --wan-checkpoint "$WAN" --checkpoint outputs/wan_lidar_1to4/best.pt
```

The evaluation writes per-sample and per-horizon metrics to the independent
`outputs/wan_lidar_1to4/` directory. It uses calibrated ray directions,
squared bidirectional nearest-neighbour Chamfer in `m^2`, valid-range MAE in
metres, mask precision/recall/F1 and false-empty counts.
