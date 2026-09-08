# NavRL 延时策略评估报告（物理交互量对齐）

## 1. 评估目的

本次评估比较以下两份权重在相同延时环境中的表现：

- baseline：在无延时环境中训练的策略；
- delay：在双阶段随机延时环境中训练的策略。

这里的 baseline 仅表示“训练时没有延时”。评估时 baseline 和 delay
策略都承受相同的推理延时与命令传输延时。因此，本实验比较的是两种训练方式
在延时测试环境中的鲁棒性，不是“无延时测试环境”和“延时测试环境”的直接比较。

评估日期为 2026-09-08。Isaac Sim 容器使用 UTC，因此 YAML 文件中的时间也是
UTC。

## 2. 权重与训练交互量

### 2.1 评估权重

```text
baseline:
isaac-training/wandb/run-20260831_141527-yza88ur1/files/checkpoint_final.pt

delay:
isaac-training/training_delay/wandb/run-20260903_040330-2wl73jsv/files/checkpoint_final.pt
```

### 2.2 物理交互量核对

baseline 训练共采集 `1,200,029,696` 个 policy transition。无延时训练中，一个
policy transition 对应一个约 16 ms 的 physics tick，因此约等于 1.200B 个
物理交互。

delay 训练日志中的 collector 明确记录：

```text
total_frames (400000000) ... 97280 additional frames will be collected
frames_per_batch = 3072 * 32 = 98304
```

因此这次 delay 权重实际训练了：

```text
400,000,000 + 97,280 = 400,097,280 policy transitions
400,097,280 / 98,304 = 4,070 training iterations
```

日志中没有加载旧训练 checkpoint 的记录。延时训练每个 transition 平均推进约
3 个 physics tick（平均 transition 时间约 48 ms，参考 tick 为 16 ms），所以：

```text
400,097,280 * 3 = 1,200,291,840 equivalent physics ticks
```

两份权重的物理交互量相差约 0.022%，可以视为基本对齐。

delay W&B summary 中的 `policy_frames=1.2B` 和 `physics_frames=3.52B` 含有历史
计数偏移，不能直接视为该 checkpoint 从随机初始化开始实际经历的训练量。本报告以
本地 collector 启动日志、batch 数和 checkpoint 保存次数为准。

## 3. 评估设置

| 项目 | 设置 |
|---|---|
| 固定场景 | `delay_value/environments/fixed_scenarios.pt` |
| 并行环境数 | 256 |
| 静态障碍物 | 350 |
| 动态障碍物 | 80 |
| 评估种子 | 0、1 |
| 最大 nominal horizon | 2200 步，即 35.2 s |
| 推理延时范围 | 16-96 ms |
| 命令延时范围 | 16-32 ms |
| 延时变化概率 | 0.2 |
| 单次最大变化 | 1 个 16 ms 参考步 |
| 视频录制 | 关闭 |

每个种子下，两份策略使用同一个固定场景集。延时调度使用独立且固定种子的随机数
生成器，避免策略动作或 episode reset 改变延时采样序列。

本次测试包含两种延时执行方式：

1. 按步长延时：`continuous_in_eval=false`，延时以约 16 ms physics tick 执行。
2. 随机毫秒延时：`continuous_in_eval=true`，环境使用 1 ms physics tick 执行真实
   毫秒延时；策略看到的延时仍按 `floor(delay / 16 ms)` 量化。

## 4. 完整结果

### 4.1 按约 16 ms 步长执行延时

| Seed | 策略 | 到达率 | 碰撞率 | Return | Episode Time |
|---:|---|---:|---:|---:|---:|
| 0 | baseline | 37.50% | 56.25% | 5370.25 | 20.91 s |
| 0 | delay | **82.81%** | **16.41%** | **9199.04** | 31.88 s |
| 1 | baseline | 37.11% | 54.69% | 5326.24 | 21.23 s |
| 1 | delay | **82.03%** | **16.41%** | **9063.83** | 31.63 s |
| 平均 | baseline | 37.30% | 55.47% | 5348.24 | 21.07 s |
| 平均 | delay | **82.42%** | **16.41%** | **9131.44** | 31.76 s |

两种策略的两 seed 均值差异：

- 到达率：delay 提高 45.12 个百分点；
- 碰撞率：delay 降低 39.06 个百分点；
- Return：delay 提高 3783.19，约提高 70.74%。

### 4.2 按随机毫秒执行延时

| Seed | 策略 | 到达率 | 碰撞率 | Return | Episode Time |
|---:|---|---:|---:|---:|---:|
| 0 | baseline | 38.67% | 55.47% | 5001.00 | 20.56 s |
| 0 | delay | **77.73%** | **19.53%** | **8736.99** | 30.49 s |
| 1 | baseline | 37.50% | 57.03% | 5185.97 | 20.66 s |
| 1 | delay | **78.13%** | **17.19%** | **8875.87** | 31.01 s |
| 平均 | baseline | 38.09% | 56.25% | 5093.48 | 20.61 s |
| 平均 | delay | **77.93%** | **18.36%** | **8806.43** | 30.75 s |

两种策略的两 seed 均值差异：

- 到达率：delay 提高 39.84 个百分点；
- 碰撞率：delay 降低 37.89 个百分点；
- Return：delay 提高 3712.95，约提高 72.90%。

## 5. 结果分析

在物理交互量基本对齐后，delay 策略在两种延时执行方式下仍显著优于 baseline。
因此，性能差距不能解释为 delay 策略经历了约三倍的训练物理交互。主要原因是：

1. baseline 训练时没有经历观测陈旧和命令传输滞后，评估时出现了明显的分布偏移；
2. delay 策略训练时已经适应随机延时，并通过额外的延时状态判断观测和命令的时效；
3. 两份策略的 Episode Time 不同主要是因为 baseline 更早发生碰撞，而不是评估给
   delay 策略提供了更长的最大物理时间。

随机毫秒模式比按步长模式更严格。对 delay 策略而言，两 seed 平均到达率从
82.42% 降至 77.93%，下降 4.49 个百分点；碰撞率从 16.41% 升至 18.36%，增加
1.95 个百分点。这说明只按 16 ms 整步模拟延时会略微高估 delay 策略的性能，但不
改变 delay 策略明显优于 baseline 的总体结论。

当前结论来自一份 baseline 训练权重、一份 delay 训练权重和两个评估 seed。若用于
论文或正式报告，还应使用多个独立训练 seed，并报告均值、标准差或置信区间。

## 6. 原始结果文件

```text
# Seed 0，按步长
delay_value/results/comparison_step_delay/evaluation_20260908_030647.yaml

# Seed 0，随机毫秒
delay_value/results/comparison_random_delay_baseline/evaluation_20260908_043038.yaml
delay_value/results/comparison_random_delay_delay/evaluation_20260908_043026.yaml

# Seed 1，按步长
delay_value/results/aligned_step_seed1/evaluation_20260908_051239.yaml

# Seed 1，随机毫秒
delay_value/results/aligned_random_seed1_baseline/evaluation_20260908_060005.yaml
delay_value/results/aligned_random_seed1_delay/evaluation_20260908_060135.yaml
```

这些目录属于生成的评估产物，已通过 `.gitignore` 排除，不应提交到 Git。

## 7. 复现命令

以下命令在 Isaac Sim 容器内、目录
`/workspace/NavRL/isaac-training/delay_value` 中执行。若容器只映射了一张物理 GPU，
容器内应使用 `gpu_id=0`。

公共参数：

```text
baseline_checkpoint=/workspace/NavRL/isaac-training/wandb/run-20260831_141527-yza88ur1/files/checkpoint_final.pt
delay_checkpoint=/workspace/NavRL/isaac-training/training_delay/wandb/run-20260903_040330-2wl73jsv/files/checkpoint_final.pt
eval.dataset_path=/workspace/NavRL/isaac-training/delay_value/environments/fixed_scenarios.pt
eval.record_video=false
eval.max_steps=2200
env.num_obstacles=350
env_dyn.num_obstacles=80
```

按步长评估两份权重：

```bash
/isaac-sim/python.sh scripts/eval_random_delay.py \
  gpu_id=0 seed=1 \
  baseline_checkpoint=/workspace/NavRL/isaac-training/wandb/run-20260831_141527-yza88ur1/files/checkpoint_final.pt \
  delay_checkpoint=/workspace/NavRL/isaac-training/training_delay/wandb/run-20260903_040330-2wl73jsv/files/checkpoint_final.pt \
  eval.dataset_path=/workspace/NavRL/isaac-training/delay_value/environments/fixed_scenarios.pt \
  eval.policy=both eval.record_video=false eval.max_steps=2200 \
  env.num_obstacles=350 env_dyn.num_obstacles=80 \
  timing.continuous_in_eval=false sim.dt=0.016 sim.substeps=1 \
  eval.result_dir=results/aligned_step_seed1
```

随机毫秒评估可以将两份策略放到两张 GPU 上并行执行。baseline 进程使用：

```bash
/isaac-sim/python.sh scripts/eval_random_delay.py \
  gpu_id=0 seed=1 \
  baseline_checkpoint=/workspace/NavRL/isaac-training/wandb/run-20260831_141527-yza88ur1/files/checkpoint_final.pt \
  delay_checkpoint=/workspace/NavRL/isaac-training/training_delay/wandb/run-20260903_040330-2wl73jsv/files/checkpoint_final.pt \
  eval.dataset_path=/workspace/NavRL/isaac-training/delay_value/environments/fixed_scenarios.pt \
  eval.policy=baseline eval.record_video=false eval.max_steps=2200 \
  env.num_obstacles=350 env_dyn.num_obstacles=80 \
  timing.continuous_in_eval=true sim.dt=0.001 sim.substeps=16 \
  eval.result_dir=results/aligned_random_seed1_baseline
```

delay 进程使用相同参数，将下面两项改为：

```text
eval.policy=delay
eval.result_dir=results/aligned_random_seed1_delay
```
