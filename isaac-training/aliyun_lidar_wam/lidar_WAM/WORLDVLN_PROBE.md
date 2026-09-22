# 历史 latent 预测与扩散生成试验

这是受 [WorldVLN](https://github.com/EmbodiedCity/WorldVLN.code)“先由观测历史预测未来 latent，再交给下游模块”启发的 **NavRL LiDAR 适配试验**，不是 WorldVLN 模型或权重的复现。WorldVLN 的 RGB、语言指令、离散多尺度 token 和动作解码器没有移植过来。最终雷达图始终由现有 LaGen 风格条件 UNet 做 20 步 DDIM 生成，满足本项目保留 diffusion 的要求。

## 因果输入和训练

预测器直接复用本目录 `LidarHistoryMST`：输入当前及前四张**已观测** LiDAR 的 VAE latent，以及通向这些已观测帧的动作序列；输出下一张 latent 的预测。它不读取未来 LiDAR、真实未来位姿或待预测区间的动作。待预测区间实际执行的 10 个归一化动作只进入冻结的 diffusion UNet。使用原训练地图 71,615 条有完整五帧历史的样本，FP32、AdamW、batch 32 训练 1,000 次更新，按验证 latent MSE 选 checkpoint。VAE、UNet、HDF5 均未改动。

脚本：`scripts/try_worldvln_latent_prior.py`。选中的 checkpoint 是第 1,000 步。验证集预测 latent MSE **0.2078**，复制当前 latent 的 MSE **0.6348**；测试集对应 **0.1763 / 0.5676**。这些是 VAE latent 空间的无量纲误差，不能当作点云精度。

在验证 seed 16/17 各固定抽 64 条（128 条）上，分别从纯噪声、复制当前 latent、预测 latent 初始化同一个 UNet；复制和预测初值使用相同 DDIM 强度和随机种子。验证只在 `0.35 / 0.6 / 0.8` 中选择一次强度，选中 **0.35**。测试 seed 18/19 各 64 条只按该强度评估。

| 固定样本 | 联合非空数 | 纯噪声 diffusion | 复制 latent 初值 + diffusion | 历史预测 latent 初值 + diffusion |
| --- | ---: | ---: | ---: | ---: |
| 验证 | 128 | 0.752 | 0.763 | **0.700** |
| 测试 | 126 | 2.138 | **1.067** | 1.083 |

表中是同一批非空样本的双向**平方** Chamfer，单位 m²。测试集历史 latent 初值没有超过复制初值。测试 mask F1 是复制 **0.8254**、历史预测 **0.8586**；命中点距离 MAE 是 **1.119 / 0.794 m**。Chamfer、mask、径向 MAE 衡量不同误差，这次不能只按其中一项宣布整体改进。测试纯噪声 Chamfer 的均值受 seed 18 大误差样本影响明显，不应拿它替代全测试集结论。详细逐样本、逐 seed 记录在 `outputs/worldvln_latent_prior_1000/{validation,test}.json`，固定索引在同目录 `val_manifest.json`、`test_manifest.json`。

## 与动作预测几何初值结合

`scripts/try_history_geometry_mix.py` 复用已训练的动作到位姿 ridge 模型，只用上一帧状态及待执行的 10 个**实际世界坐标速度指令**预测下一传感器位姿，将上一帧 LiDAR 重投影，编码成几何 latent。几何 latent 与上述历史预测 latent 线性混合后加噪，交给相同 UNet 做 20 步 DDIM。混合系数 0 为纯几何初值，1 为纯历史预测初值；验证尝试 `0 / 0.25 / 0.5 / 0.75 / 1`。真实未来位姿只在旧诊断中用过，**这里没有进入推理**。

latent 缓存只保证归一化 PPO 动作有限，实际世界坐标指令可能是非有限值。因此这项几何试验先从完整五帧历史候选中筛出指令及上一帧状态有限的样本，再在每个验证/测试 seed 固定随机抽 64 条；每个方法在相同样本上比较。验证 seed 的有限候选数为 3,587 / 3,350，测试为 3,776 / 3,313。下表同样是配对非空的双向平方 Chamfer，单位 m²：

| 固定样本 | 联合非空数 | 直接重投影 | 几何初值 + diffusion | 最佳混合初值 + diffusion | 历史预测初值 + diffusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| 验证 | 128 | **0.454** | 0.592 | 0.592（系数 0） | 0.643 |
| 测试 | 127 | **0.441** | 0.553 | 0.553（系数 0） | 0.575 |

验证选出系数 **0**，所以测试上的“最佳混合”就是纯几何初值；测试仍计算系数 1 作诊断。测试逐 seed 的几何初值 / 历史初值 Chamfer 为 seed 18 的 **0.581 / 0.590**、seed 19 的 **0.525 / 0.560**。直接重投影 Chamfer 更低，但测试命中点距离 MAE 是 **1.088 m**，几何初值 + diffusion 为 **0.802 m**；diffusion 对部分可见性和距离误差仍有帮助。完整结果和选参在 `outputs/history_geometry_mix_finite_64/`。这个样本集按有限实际指令重新抽取，与上一个 128 样本 latent 试验不同，不能跨表直接相减。

## 判断

历史预测 latent 的 MSE 大幅优于复制，但这种表示空间收益没有转化为测试 Chamfer 收益。几何重投影在相同样本上仍比所有采样初值更准；只在**推理时**更换 DDIM 初值无法让原 UNet 学会保留正确几何、只生成遮挡后新出现的雷达回波。现有试验不支持继续扩大历史 Transformer 或反复调初值强度来宣称提升。

下一项有意义的模型实验是：把**因果、动作预测的几何重投影 latent**显式作为训练条件，扩散只学习重投影误差和新显露区域，同时继续监督 mask、有效点距离和最终点云；在同一有限指令样本清单上比较直接重投影、原 UNet、推理引导版、训练条件版，并分别统计可重投影区域和新显露区域。训练时不能使用真实未来位姿；动作专家要输出与该位姿预测器一致的执行指令，才能用于规划。

复现命令（PPU 环境）：

```bash
cd /mnt/workspace/wam_trining/trining/lidar_WAM
PY=/mnt/workspace/LaGen/.venv-ppu/bin/python
DATA=/mnt/workspace/wam_trining/data/navrl_static_100k
$PY scripts/try_worldvln_latent_prior.py --data-root "$DATA/lagen_cache" --raw-root "$DATA" --out outputs/worldvln_latent_prior_1000 --steps 1000 --batch-size 32 --val-per-seed 64 --test-per-seed 64
$PY scripts/try_history_geometry_mix.py --data-root "$DATA/lagen_cache" --raw-root "$DATA" --prior-run outputs/worldvln_latent_prior_1000 --out outputs/history_geometry_mix_finite_64 --per-seed 64
```

## 五帧 latent 自回归误差

`scripts/evaluate_latent_prior_5.py` 在测试 seed 18、19 各固定抽取 256 条连续的五个 10 步转换（共 512 条；每个 seed 分别有 4,210 / 4,127 个候选）。初始输入是五张真实已观测雷达帧。之后每预测一张，就把**预测的 latent** 放回五帧历史窗口；待预测区间的动作只在该区间结束后进入动作历史，不提前输入预测器。未来第五帧距离初始观测约 0.8 秒。下表是每个样本 540 个经 VAE scaling factor 缩放的 latent 数值上的 MSE，**没有米的单位**。

| 未来帧 | 真正自回归 | 每步真实历史诊断 | 复制初始 latent |
| ---: | ---: | ---: | ---: |
| 1 | 0.1862 | 0.1862 | 0.5909 |
| 2 | 0.2767 | 0.1877 | 1.0296 |
| 3 | 0.3758 | 0.1908 | 1.2831 |
| 4 | 0.4886 | 0.1970 | 1.3757 |
| 5 | **0.5964** | 0.1986 | 1.3982 |

第五帧按测试地图分别为 seed 18 的 **0.6497**、seed 19 的 **0.5431**。真实历史诊断的第五帧 **0.1986** 使用了中间四张真实帧，只用于分离误差累积，不能当作可部署的五帧预测成绩。这里的第一帧 **0.1862** 与前面 **0.1763** 都是一步测试，但抽取的样本不同，不能误认为训练使指标变化。逐样本指标、轨迹索引、候选计数保存在 `outputs/worldvln_latent_prior_1000/test_autoregressive_5.json` 和相邻的 `test_autoregressive_5_manifest.json`。

复现：

```bash
$PY scripts/evaluate_latent_prior_5.py --data-root "$DATA/lagen_cache" --per-seed 256 --batch-size 32
```
