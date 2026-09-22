# 直接预测 0.48 秒后的 LiDAR

`lidar_wam.runner.world_direct_horizon` 保留现有双通道 Circular VAE 和
LaGen 风格条件 UNet，使用当前 latent、当前因果状态和连续三段
`10×3` 归一化 PPO 动作，直接预测第三个未来帧。它不会先生成
`t+1/t+2`，因此没有三次自回归造成的累计误差。

连续样本必须属于同一 scene、帧号连续、三段 `step_delta` 均为 10，
而且三段动作全部有效且有限。当前缓存可构造训练 75,626 条、验证
9,462 条、测试 9,472 条轨迹。验证和测试分别按 seed 42 从每个 seed
固定抽取 256 条，清单保存在实验目录中。

```bash
cd /mnt/workspace/wam_trining/trining/lidar_WAM
PY=/mnt/workspace/LaGen/.venv-ppu/bin/python
DATA=/mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache
OUT=/mnt/workspace/wam_trining/trining/lidar_WAM/outputs
INIT=$OUT/world_circular_causal_8h/best.pt

# 检查连续轨迹数量
$PY -m lidar_wam.runner.world_direct_horizon inspect \
  --data-root "$DATA" --out "$OUT"

# batch 2 前向、反向和短 DDIM 检查
$PY -m lidar_wam.runner.world_direct_horizon smoke \
  --data-root "$DATA" --out "$OUT" --init-checkpoint "$INIT"

# 128 条轨迹过拟合，默认 1,000 步、batch 128、lr 1e-4
$PY -m lidar_wam.runner.world_direct_horizon train \
  --data-root "$DATA" --out "$OUT" --init-checkpoint "$INIT" --overfit

# 完整训练，默认 20,000 步、batch 1024、基础 lr 1e-5
$PY -m lidar_wam.runner.world_direct_horizon train \
  --data-root "$DATA" --out "$OUT" --init-checkpoint "$INIT"

# 锁定验证最佳 checkpoint 后评估；最后才运行 test
$PY -m lidar_wam.runner.world_direct_horizon evaluate \
  --data-root "$DATA" --out "$OUT" --split val
$PY -m lidar_wam.runner.world_direct_horizon evaluate \
  --data-root "$DATA" --out "$OUT" --split test
```

完整评估在相同样本上比较复制帧、直接 `t+3`、一步 UNet 自回归三次、
0.48 秒恒速度/角速度重投影以及目标帧 VAE 重建。输出包含平方 Chamfer
`m²`、线性对称最近邻距离 `m`、有效点 Range MAE、mask 指标、空点云、
latent MSE、逐 seed/逐样本数据和对照图。

