# NavRL 延时训练对照实验结果与分析

本文档说明当前仓库中 `training`（无延时）和 `training_delay`（随机 Actor 推理延时 + 50 Hz 发布 + 异步命令传输延时）两套实验的代码流程、配置、统计口径和运行方法。

> **版本说明（2026-09-01）**：当前代码使用“双阶段随机延时”模型。Actor 完成后更新 latest output，独立 50 Hz 发布器重复发布该值，再经过有序的随机延时通道到达控制器。0.2～0.4 节保留的是旧版实验，只能用于历史追溯。当前延时网络使用 10 维状态（基线 8 维 + 两个最近实测量），旧版 checkpoint 与当前网络不兼容。

## 0. 实验结果摘要

### 0.1 当前双阶段随机延时评估（2026-08-31）

使用以下两个 `checkpoint_8000.pt` 进行对照：

- 无延时训练基线：`training/scripts/wandb/run-20260827_204049-ma206url/files/checkpoint_8000.pt`
- 双阶段延时训练：`training_delay/scripts/wandb/run-20260831_012406-81d8oenb/files/checkpoint_8000.pt`

两份策略均在物理 GPU 1 上运行，使用同一批 1024 个固定场景、`seed=0`、完整 2200 policy step rollout、相同的 Actor 推理延时序列和相同的 command 传输延时序列。这里的“无延时训练基线”只表示它在无延时环境中训练；本次评估本身对两份策略都施加了双阶段随机延时。

下表是历史评估使用的延时条件；当前配置中的 command 范围已经改为较小的实机控制链路估计值 0.060～0.150 s。

| 延时阶段 | 配置范围 | physics tick | 本次 2200 步序列均值 |
|---|---:|---:|---:|
| Actor 推理延时 | 0.016～0.096 s | 1～6 | 0.04102 s |
| command 传输延时（历史评估） | 0.016～0.032 s | 1～2 | 0.02324 s |

延时采用 `change_probability=0.2`、`max_step_change=1` 的随机游走。推理延时 1～6 tick 在本次序列中分别出现 `648/556/462/285/140/109` 次；command 延时 1/2 tick 分别出现 `1204/996` 次。

完整结果：

| 指标 | 无延时训练基线 | 双阶段延时训练 | 差值（双阶段 - 基线） |
|---|---:|---:|---:|
| Reach Goal | 0.6943 | **0.8770** | **+0.1826** |
| Collision | 0.2715 | **0.1211** | **-0.1504** |
| Return | 6620.27 | **7567.75** | **+947.48** |
| Episode Time | 29.40 s | **32.99 s** | +3.58 s |
| `stats.episode_len` | 1822.22 | 2044.30 | +222.08 |
| 等效 nominal Episode Length（`episode_time/0.016`） | 1837.62 | 2061.65 | +224.03 |
| Decision Count | 620.43 | 692.84 | +72.41 |
| Command Update Count | 619.43 | 691.84 | +72.41 |
| Controller Update Count | 1822.94 | 2045.18 | +222.24 |
| Truncated | 0.7158 | 0.8789 | +0.1631 |
| rollout 平均 transition dt | 0.038560 s | 0.038568 s | +0.000008 s |
| 视频 FPS | 13 | 13 | 0 |

主要结果是 Reach Goal 提高 18.26 个百分点，Collision 降低 15.04 个百分点。两份策略的 rollout 平均 transition dt 几乎相同，说明性能差异不是因为双阶段策略在评估时获得了更高控制频率或更短延时。双阶段策略的 Episode Time 更长，主要与它碰撞更少、更多 episode 能继续飞行到目标或时间上限一致，不能解释为它飞得更慢。

`stats.episode_len` 与“等效 nominal Episode Length”存在约 15～17 步差异，是统计顺序造成的：前者对每个环境的 `episode_time/0.016` 先截断到 2200 再求平均；后者先对 `episode_time` 求平均再除以 `0.016`，因此会保留最后一个变长 transition 越过 35.2 s 边界的少量超出时间。两者都不是 Actor 决策次数；严格比较物理时间时优先使用 `Episode Time`。

为什么加入延时训练后 Reach Goal 反而更高：

1. **比较的是两种训练方式在同一个双延时测试环境中的表现。** 延时并没有让任务本身变简单；无延时基线遇到了训练时没有见过的观测陈旧和命令滞后，而双阶段策略训练时已覆盖这些扰动。
2. **当前模型允许策略看到因果上可获得的时序状态。** 10 维状态只在基线 8 维状态后追加最近一次实际观测到 Actor 推理延时和最近一次 Actor 输出到控制器采用的实际延时；8 维基线没有这些信息。
3. **随机延时训练是一种时序域随机化。** 它牺牲理想无延时条件下的单一最优控制，换取对一组推理和传输时延的鲁棒性，因此在随机延时测试分布上高于无延时基线是合理结果。
4. **历史提升不是旧版动作保持的低通滤波假象。** command 传输倒计时与下一次推理重叠；`VelController` 每个 physics tick 都闭环运行，保持的是当前 active `cmd_vel`，不是电机动作。当前 50 Hz 发布模型需要重新训练和评估，不能直接沿用本表数值。
5. **结论仍受单 seed 限制。** 当前差值足以说明这一次固定场景下存在明显鲁棒性收益，但论文结论仍需多 seed 均值、标准差或置信区间。

原始 YAML：

```text
delay_value/results/checkpoint_8000_two_stage_random/evaluation_20260831_104431.yaml
```

视频：

```text
delay_value/videos/checkpoint_8000_two_stage_random/baseline.mp4
delay_value/videos/checkpoint_8000_two_stage_random/delay.mp4
```

### 0.2 历史训练曲线结果（旧版动作保持模型）

从当前 W&B 历史记录读取的早期 `reach_goal` 如下。数值为对应训练阶段的平均值，step 为 W&B 训练 iteration：

| W&B step | 无延时 | 随机延时 |
|---:|---:|---:|
| 约 250 | 0.510 | 0.890 |
| 约 500 | 0.648 | 0.841 |
| 约 750 | 0.710 | 0.877 |
| 约 1000 | 0.705 | 0.875 |
| 约 1250 | 0.786 | 0.920 |
| 约 1500 | 0.680 | 0.871 |

在原始 W&B step 横轴下，随机延时曲线在前 1500 step 大多数时间高于无延时曲线。但这不能直接解释为延时策略在相同训练量下更强，因为每个 delay policy step 会推进多个 physics step。

### 0.3 历史同一 checkpoint 跨环境评估

使用无延时训练得到的 `checkpoint_8000.pt`，在固定评估场景 `eval_env.pt` 上得到：

| 评估条件 | Reach Goal | Collision | Episode Length | Return |
|---|---:|---:|---:|---:|
| 无延时 | 0.8945 | 0.0713 | 2104.73 | 7180.66 |
| 随机延时 | 0.8877 | 0.0723 | 约 610.83* | 8010.49* |

带 `*` 的随机延时数据来自统计口径修正前的评估输出：`Episode Length` 实际是 actor 决策次数，`Return` 也使用了修正前的平滑惩罚计算，不能与当前修正后的新实验直接混合比较。其真实物理时长约为 `33.98 s`，对应 `33.98 / 0.016 = 2123.94` 个 nominal 等效步。

### 0.4 历史随机动作保持对照评估（2026-08-28）

2026-08-28 使用 `training_delay` 的 `checkpoint_8000.pt` 进行了随机延时评估。为保证对照公平，基线 checkpoint 和 delay checkpoint 都在同一批 `eval_env.pt` 场景、同一随机种子和同一随机延时序列下运行，评估最大 rollout 为 800 个 policy step，延时范围为 `0.032~0.128 s`。

| 指标 | 无延时训练 checkpoint* | 延时训练 checkpoint |
|---|---:|---:|
| Reach Goal | 0.8877 | **0.9248** |
| Collision | 0.0723 | **0.0645** |
| Episode Length（等效 nominal 步） | 2105.10 | 2113.10 |
| Decision Count（实际策略决策次数） | 610.83 | 613.60 |
| Episode Time | 33.98 s | 34.11 s |
| Return | 7995.41 | 7991.86 |
| Truncated | 0.9189 | 0.9346 |
| 平均 command dt | 0.06443 s | 0.06443 s |

\* 这里的“无延时训练 checkpoint”指基线训练得到的 checkpoint，但本次对照评估本身仍在随机延时环境中进行，不是无延时环境评估。

本次结果中，延时训练策略相对基线的 Reach Goal 提高 `0.0371`，Collision 降低 `0.0078`。两组的 Episode Time、等效 Episode Length 和 Decision Count 接近，说明当前 episode 统计口径已经统一。视频保存在远程服务器：

```text
/home/wei/End_to_End/NavRL/isaac-training/training_delay/eval_videos/checkpoint_8000_delay_random/baseline.mp4
/home/wei/End_to_End/NavRL/isaac-training/training_delay/eval_videos/checkpoint_8000_delay_random/delay.mp4
```

该结果可以说明：在这一次固定场景和随机延时序列下，延时训练 checkpoint 对随机命令间隔表现出更好的适应性。但这是单个 checkpoint、单个 seed 的评估结果，不能单独作为普遍优越性的统计结论；正式实验仍应使用多个 seed，并报告均值和方差。

### 0.5 新旧结果的综合分析

1. **旧版早期曲线不能按 W&B step 直接比较。** 两套实验每个 batch 都采集 `1024 * 32` 个策略转移，但无延时每个策略 step 是 `0.016 s`，旧随机动作保持模型每个 step 可能覆盖 `0.032～0.128 s`。因此同一个 W&B step，旧 delay 已经历更多物理仿真时间。
2. **延时曲线较高的主要原因是训练横轴不等价。** 例如 delay 在约 1489 step 达到 `reach_goal=0.871`，无延时在约 6728 step 达到约 `0.876`；按物理训练时间对齐后，两者处于相近水平。
3. **旧动作保持模型可能带来额外稳定性。** 历史结果中的 delay 实现是随机 zero-order hold：一个动作被 VelController 转成电机动作后保持 2～8 个 physics step。这会减少高频动作变化，但它不是当前双阶段延时代码。
4. **旧奖励实现会使 delay 的 return 偏高。** 之前平滑惩罚按决策次数计算，而主要奖励按 `time_scale` 放大，delay 在相同物理时间内支付的平滑惩罚次数更少。当前代码已将平滑项一起乘以 `time_scale`，后续实验应使用新代码重新训练。
5. **当前双阶段结果与旧结果不能纵向比较数值。** 两者的网络输入、延时因果关系、奖励累计和评估 horizon 均不同。历史结果使用的是旧的 19 维 FIFO 实现；新的训练应以 10 维、50 Hz 发布实现重新产生 checkpoint 和评估结果。
6. **训练曲线仍应按物理时间对齐。** 当前 W&B 已记录 `policy_frames`、`physics_frames`、`sim_time_seconds` 和 `wall_time_seconds`。比较样本效率、仿真经历和计算成本时应分别选择对应横轴。

### 0.6 本实验的结论边界

当前结果可以支持：双阶段随机延时会降低平均 Actor 更新频率；无延时训练策略在推理和 command 传输延时同时存在时性能明显下降；双阶段随机延时训练在本次固定场景和 seed 下将 Reach Goal 从 `0.6943` 提升到 `0.8770`，并将 Collision 从 `0.2715` 降低到 `0.1211`。

当前结果还不能支持：双阶段延时训练在任意延时分布、任意场景或实机上必然优于基线，也不能证明它在相同 policy frames、累计仿真时间和墙钟成本下训练效率更高。要形成正式结论，应至少使用多个训练 seed 和多个评估 seed，报告均值、标准差或置信区间，并分别按 `policy_frames`、`sim_time_seconds` 和 `wall_time_seconds` 对齐训练成本。

## 1. 实验目标

比较两种策略：

1. 无延时训练策略：actor 每个 nominal control step 输出一次速度指令。
2. 双阶段延时训练策略：分别随机化“观测到 Actor 输出”的推理延时和“Actor 输出到控制器应用”的 command 传输延时，使策略适应真实系统中的两类时序不确定性。

当前代码只在 Isaac Sim 中训练和评估，没有接入 ROS。`VelController` 仍使用原来的 `LeePositionController`，其源码没有修改；两段延时均位于 actor 与 `VelController` 之间，低层控制器始终按固定 physics tick 闭环运行。

## 2. 代码目录

```text
isaac-training/
├── training/                         # 无延时基线
│   ├── cfg/train.yaml
│   ├── cfg/eval.yaml
│   └── scripts/{env,ppo,train,eval}.py
├── training_delay/                   # 双阶段延时训练
│   ├── cfg/train.yaml
│   ├── cfg/eval_random.yaml
│   └── scripts/{env,timing,ppo,train,eval,eval_random_delay,utils}.py
└── delay_value/                      # 双阶段延时评估归档
    └── 历史结果与旧版评估脚本
```

当前配置默认从 `isaac-training/training/cfg/eval_env.pt` 加载固定评估场景。它只固定起点、目标点、地形随机种子和障碍物配置；仿真、观测、奖励、控制器和双延时执行逻辑仍来自 `training_delay` 的 `NavigationEnv`。

## 3. 环境与仿真参数

两套配置的主要环境参数相同：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `env.num_envs` | 1024 | 并行无人机环境数量 |
| `env.max_episode_length` | 2200 | nominal control step 上限 |
| `env.num_obstacles` | 200 | 静态障碍物数量 |
| `env_dyn.num_obstacles` | 0 | 当前不启用动态障碍物 |
| `sim.dt` | 0.016 s | physics step 时间间隔 |
| `sim.substeps` | 1 | 每个 nominal step 的 physics 子步数 |
| nominal control rate | 62.5 Hz | `1 / 0.016` |
| nominal episode horizon | 35.2 s | `2200 * 0.016` |

当前训练使用 GPU：

```text
training:       cfg/train.yaml 中默认 cuda:0
training_delay: cfg/train.yaml 中 gpu_id=1，即 cuda:1
```

训练脚本启动 Isaac Sim 时会把 `active_gpu` 和 `physics_gpu` 设置为对应的 `gpu_id`。修改 GPU 时应同时检查配置中的 `device`。

## 4. 动作和控制器流程

两套实验都使用同一套高层动作和控制器：

```text
观测
  -> PPO feature extractor
  -> Beta actor 输出 3 维归一化动作
  -> 映射到 [-2, 2] m/s 的局部速度
  -> 转换到世界坐标系
  -> Actor 推理延时（控制器继续执行旧 cmd_vel）
  -> 推理完成，只更新 latest_actor_cmd_vel
  -> 立即开始下一次 Actor 推理
  -> 独立 50 Hz 发布器周期性读取 latest_actor_cmd_vel
  -> 每次发布重新采样 command 传输延时
  -> 有序传输到期后更新 active cmd_vel
  -> VelController（始终每个 physics tick 闭环运行）
  -> LeePositionController
  -> 四旋翼电机动作
  -> Isaac Sim physics
```

当前采用 `overlapping_transport`。动作基于推理开始时的观测；推理完成只覆盖发布器持有的 latest output，不会立刻创建传输任务。50 Hz 时钟到达后，发布器读取当时最新的 Actor 输出，并采样该 Actor 输出到控制器采用的目标延时；目标延时减去已经等待的发布器时间后，才作为发布后的传输倒计时。传输倒计时与下一次推理和后续发布同时进行；后发报文不能越过先发报文，同一 tick 到期时只应用最新发布。`LeePositionController` 源码没有修改，并且在每个 `0.016 s` physics tick 都根据当前无人机状态重新计算电机动作，因此保持的是速度设定值，不是电机输出。

一次只存在一个 Actor 推理任务，但可同时存在推理任务和多条传输中的发布报文。发布器会重复发布同一 Actor 输出，直到新推理完成后 latest output 被覆盖。由于 1024 个 Isaac 环境共享仿真时钟，当前整批环境共享同一条延时采样序列；每个环境的在途报文有独立有效状态和序号，episode reset 会单独清除该环境的旧报文。

## 5. 双阶段延时机制

### 5.1 问题定义

实机从一次观测到速度命令真正被控制器采用，当前拆为以下时间点：

```text
t_obs       t_actor_done       t_publish       t_controller_apply
  |---------------|---------------|-------------------|
    inference       publisher wait      transport
```

定义：

```text
d_infer    = t_actor_done - t_obs
d_pub_wait = t_publish - t_actor_done
d_transport = t_controller_apply - t_publish
d_total    = d_infer + d_pub_wait + d_transport
```

`d_infer` 表示图像/激光预处理和 Actor 前向推理造成的观测陈旧；`d_pub_wait` 是推理完成后等待下一次 50 Hz 发布时刻的时间；`d_transport` 是发布后到控制器采用该 `cmd_vel` 的时间。三者不能简单实现成一个动作保持，因为等待期间 Actor、发布器和控制器仍按各自时钟运行。

### 5.2 当前配置

`training_delay/cfg/train.yaml` 和 `training_delay/cfg/eval_random.yaml` 的核心配置为：

```yaml
timing:
  enabled: true
  mode: overlapping_transport
  sampling_mode: random_walk_steps
  command_sampling_mode: random_walk_steps
  command_publish_hz: 50.0
  randomize_in_eval: true
  reference_dt: 0.016
  inference_delay:
    min: 0.016
    max: 0.096
    eval: 0.032
  command_delay:
    min: 0.060
    max: 0.150
    eval: 0.100
  change_probability: 0.2
  max_step_change: 1
```

训练时 `env.training=true`，两段延时始终随机；评估配置将 `randomize_in_eval` 设为 `true`，使 `env.training=false` 时仍随机采样。若评估时将其设为 `false`，则固定使用两个 `eval` 值。

`sim.dt=0.016 s`，因此：

| 项目 | 时间范围 | tick 范围 | 对应频率/含义 |
|---|---:|---:|---|
| Actor 推理周期 | 0.016～0.096 s | 1～6 | 62.5～10.42 Hz 的策略更新频率 |
| cmd_vel 发布周期 | 平均 0.020 s | 受 physics tick 量化 | 独立于 Actor 更新频率 |
| Actor 输出到控制器采用 | 0.060～0.150 s | 4～9 | 每次 50 Hz 发布单独采样，包含发布等待 |
| 观测到控制器采用 | 约 0.080～0.272 s | 推理 + 0～0.032 s 发布等待 + 传输 | 不含机体动力学响应 |

两个随机阶段各自维护随机游走状态。推理延时在每次 Actor 决策开始时采样；command 延时在每次 50 Hz 发布时采样。每次以概率 `0.2` 尝试改变，变化量从 `-1/0/+1` tick 中采样并裁剪到范围内。当前 1024 个并行环境共享仿真时钟和采样值；每个环境的在途报文有独立 valid mask，单个 episode reset 不会把其他环境的报文清空。

### 5.3 Actor、50 Hz 发布和有序传输

一次外层 PPO transition 对应一次 Actor 推理周期，而不是整段端到端延时：

```text
观测 O_k
  -> Actor 推理 d_infer(k)
  -> 产生 cmd_k，覆盖 latest_actor_cmd_vel
  -> 立即开始基于新观测的下一次 Actor 推理

同时：
50 Hz 发布时钟到达
  -> 快照 latest_actor_cmd_vel
  -> 采样 d_transport 并进入有序通道
  -> 报文到期后更新 active_cmd_vel
  -> VelController 从下一 physics tick 起使用新设定值
```

因此发布与传输都和下一次 Actor 推理重叠。每次 50 Hz 发布都会为当前 latest Actor 输出重新采样 `0.060~0.150 s` 的端到端目标延时，再扣除该输出已经等待发布器的时间；为保持 ROS 有序传输，若前一报文尚未到期，后一报文不会提前越过它。多个报文同 tick 到期时只采用最新报文。重复发布同一个 Actor 输出可以刷新控制器收到的速度值，但不会被统计成新的 Actor command，也不会重置上一条不同命令的执行时长。episode reset 时，该环境尚未执行的报文会失效。

### 5.4 `VelController` 的行为

延时发生在 `VelController` 之前。控制器每个固定 `0.016 s` physics tick 执行：

```text
当前无人机状态 + active_cmd_vel
  -> LeePositionController
  -> 新的四旋翼电机动作
  -> Isaac Sim physics
```

等待新命令时保持的是 `active_cmd_vel` 速度设定值，而不是旧电机输出。无人机状态持续变化，所以即使 command 没更新，`LeePositionController` 仍会重新闭环计算电机动作。这比旧版“Actor 输出后计算一次电机动作，再保持 N 个 tick”的 zero-order hold 更接近真实控制系统。

### 5.5 因果观测与不可见信息

策略只观察最近已经完成的 Actor 推理时长，以及最近一次 Actor 输出到控制器采用的实际延时。它不观察发布器相位、在途 payload、剩余传输时间或采样到的未来值，避免把仿真器内部状态泄漏给策略。

采样到但尚未发生的延时只写入 `stats.sampled_inference_delay`、`stats.sampled_command_delay` 和 `stats.sampled_total_delay`，用于实验核对，不进入策略输入。

### 5.6 公平评估中的随机序列

当前 `training_delay` 评估在每个 rollout 前调用 `env.set_seed(seed)` 并重置 timing schedule。使用 `ExplorationType.MEAN` 时，基线和延时策略会从同一 seed 开始采样同一条延时序列；如果改用随机探索，策略本身消耗的随机数可能改变后续时序采样，不能直接视为严格配对。

2026-08-31 评估的 2200 点延时序列为：

```text
inference tick 1/2/3/4/5/6: 648 / 556 / 462 / 285 / 140 / 109
command tick   1/2:         1204 / 996
mean inference delay:       0.04102 s
mean command delay:         0.02324 s
```

结果 YAML 中的 `sampled_*` 是各环境第一次 episode 结束时的瞬时值，不代表整条 2200 步序列的均值；整段延时分布应使用 timing RNG 序列统计。

这些范围是保持旧实验总时间范围的初始设置，不代表实机测量值。最终配置应使用 `t_obs`、`t_actor_done` 和 `t_controller_apply` 三个时间戳分别统计两个延时。

## 6. 观测差异

无延时 PPO 使用 8 维无人机状态：相对目标方向、水平距离、垂直距离和目标坐标系速度。

延时 PPO 使用 10 维状态。前 8 维保持不变，随后增加：

```text
last_measured_inference_delay / reference_dt
interval_between_command_apply_events / reference_dt
```

策略只看到已经完成的两段延时，不会看到“尚需多久到达”或本次未来采样值。这些采样值只写入 `stats.sampled_*` 和 W&B 用于核对分布。

## 7. PPO 训练流程

训练入口分别为：

```text
training/scripts/train.py
training_delay/scripts/train.py
```

每次训练 iteration 的流程是：

1. `SyncDataCollector` 在 1024 个环境中采集 32 个策略 step。
2. 每个 batch 包含 `1024 * 32 = 32768` 个环境转移。
3. actor 使用随机探索采样动作（`ExplorationType.RANDOM`）。
4. PPO 计算 GAE、return、actor loss 和 critic loss。
5. 每个 batch 进行 4 个 epoch、16 个 minibatch 的更新。
6. 将训练指标写入 W&B。
7. 每 500 个 iteration 保存一个 checkpoint。

共同的 PPO 超参数为：

```yaml
feature_extractor.learning_rate: 5e-4
actor.learning_rate: 5e-4
critic.learning_rate: 5e-4
actor.clip_ratio: 0.1
critic.clip_ratio: 0.1
entropy_loss_coefficient: 1e-3
gamma: 0.99
gae_lambda: 0.95
training_frame_num: 32
training_epoch_num: 4
num_minibatches: 16
actor.action_limit: 2.0
```

总训练上限为 `600000000` 个 policy frames，可以通过修改 `max_frame_num` 或停止进程提前结束。

## 8. 延时奖励和 GAE

奖励在每个固定 `0.016 s` physics tick 计算：

```python
tick_reward = reward_rate - 0.1 * penalty_smooth
transition_reward = sum(gamma**j * tick_reward[j])
```

这样不会再用一次转移最终状态的奖励乘以时间倍数，碰撞、目标和轨迹奖励按真实物理推进累计。

外层 PPO transition 对应一次 Actor 推理，持续时间为该次推理延时（最少一个 nominal tick）。50 Hz 发布和命令传输通过跨 transition 的持久状态表示，不再延长 Actor transition。GAE 使用：

```python
time_scale = transition_dt / reference_dt
discount = gamma ** time_scale
trace_decay = lambda_ ** time_scale
```

无延时 PPO 使用普通的固定步长 GAE。
延时 PPO 在 `done = terminated | truncated` 边界停止 bootstrap，避免从超过物理 episode horizon 后的状态计算 value target。

## 9. 统计指标口径

延时环境当前同时记录：

| 指标 | 含义 |
|---|---|
| `stats.episode_len` | 每个环境的 `episode_time/0.016`，先截断到 2200 后再参与 batch 平均 |
| `equivalent_episode_len` | 评估脚本用 batch 平均 `episode_time/0.016` 计算，会保留最后一个变长 transition 的边界超出量 |
| `stats.decision_count` | 实际 actor 决策次数，随机延时下通常明显小于 2200 |
| `stats.episode_time` | 真实累计物理时间，最大约 35.2 s |
| `stats.inference_delay` | 最近一次已完成的 Actor 推理延时 |
| `stats.command_delay` | 最近一次 Actor 输出完成到控制器采用的实际延时，也是网络第 10 维的原始值 |
| `stats.publisher_wait_delay` | 该 Actor 输出完成后等待下一次 50 Hz 发布的实际时间 |
| `stats.transport_delay` | 该报文从发布到控制器采用的实际等待时间 |
| `stats.total_delay` | 对应命令从观测到控制器采用的实际总时间：推理 + 发布等待 + 传输 |
| `stats.sampled_*` | 最近一次 Actor 或发布事件采样到的随机值，仅用于核对，不进入策略 |
| `stats.transition_dt` | 当前 PPO transition 覆盖的物理时间 |
| `stats.command_age_at_update` | 新命令生效时旧命令的年龄 |
| `stats.pending_command_age` | 最早即将到达的在途命令已经等待的时间 |
| `stats.command_queue_depth` | 当前仍在途的 50 Hz 发布报文数量 |
| `stats.command_publish_count` | episode 内实际执行的 50 Hz 发布次数 |
| `stats.command_update_count` | episode 内不同 Actor command 实际生效次数；重复发布不计数 |
| `stats.controller_update_count` | 低层控制器实际闭环更新次数 |
| `stats.reach_goal` | episode 内是否到达目标 |
| `stats.collision` | episode 结束时碰撞状态 |
| `stats.truncated` | 是否因达到物理时间上限而截断 |
| `stats.return` | episode 累计奖励 |

例如随机延时下可能出现：

```text
decision_count 约 600
episode_time    约 34 s
episode_len     约 2100（等效 nominal 步）
```

这不是矛盾：600 是真实策略决策次数，2100 是把 34 秒换算成 `34/0.016` 后的等效步数。

## 10. 公平比较训练曲线

不能直接比较两条曲线在相同 W&B `_step` 下的 `reach_goal`，因为两边每个策略 step 覆盖的物理时间不同：

```text
无延时：每个策略 step = 0.016 s
延时：  每个策略 step = 0.016~0.096 s（仅 Actor 推理周期）
```

建议同时记录以下横轴：

1. `policy_iteration`：优化器更新次数。
2. `policy_frames`：actor 决策样本数量。
3. `sim_time_seconds`：每个并行环境平均累计的 `transition_dt`。
4. `physics_frames`：所有并行环境的 `transition_dt / 0.016` 之和。
5. `wall_time_seconds`：实际训练耗时。

比较“同样数据量”时用 `policy_frames`；比较“同样仿真经历”时用 `sim_time_seconds` 或 `physics_frames`；比较实际训练成本时用 `wall_time_seconds`。只看 W&B iteration 时，delay 可能看起来收敛更快，但这不等于在同等物理训练量下学得更快。

## 11. 训练运行命令

### 无延时训练

```bash
cd /home/wei/End_to_End/NavRL/isaac-training/training
source /home/wei/miniconda3/etc/profile.d/conda.sh
conda activate NavRL
python scripts/train.py
```

### 延时训练

```bash
cd /home/wei/End_to_End/NavRL/isaac-training/training_delay
source /home/wei/miniconda3/etc/profile.d/conda.sh
conda activate NavRL
python scripts/train.py
```

延时训练中断后以已保存权重续训：

```bash
python scripts/train.py \
  wandb.run_id=<run_id> \
  checkpoint_path=/path/to/checkpoint_N.pt
```

`wandb.run_id` 和 `checkpoint_path` 必须同时提供，否则脚本会拒绝在旧 W&B run 中追加一条从随机网络开始的曲线。现有 checkpoint 保存网络和 value normalization 状态，不包含 Adam optimizer 动量，所以属于权重续训，不是逐位完全恢复。

默认使用配置文件中的 GPU。修改 GPU 时优先修改 `cfg/train.yaml` 的 `gpu_id`，并使 `device`、Isaac Sim 的 `active_gpu` 和 `physics_gpu` 保持一致。

## 12. 固定场景评估

当前双阶段评估使用：

```text
isaac-training/training/cfg/eval_env.pt
```

该文件不是环境代码，只保存固定起点、目标点、terrain seed 和障碍场景元数据。实际环境类是 `training_delay/scripts/env.py` 中的 `NavigationEnv`。

正式对照应保持 `eval_env.pt`、`seed`、`num_envs=1024`、`num_obstacles=200`、`env_dyn.num_obstacles=0`、rollout horizon 和 timing 配置一致。场景固定与延时随机并不冲突：前者控制空间环境，后者测试时序鲁棒性。

## 13. 随机延时评估

当前评估入口位于独立归档目录：

```bash
cd /home/yimingwei/NavRL/isaac-training/training_delay
source /home/wei/miniconda3/etc/profile.d/conda.sh
conda activate NavRL
python -u scripts/eval_random_delay.py \
  baseline_checkpoint=/path/to/training/checkpoint.pt \
  delay_checkpoint=/path/to/training_delay/checkpoint.pt \
  gpu_id=1 device=cuda:1 \
  eval.max_steps=2200
```

脚本会让策略使用固定场景、随机种子和双阶段延时配置。`eval_random_delay.py` 可同时加载原始 8 维基线和当前 10 维延时策略：基线输入自动截取前 8 维，两者都直接使用 `NavigationEnv` 内部的低层控制器；旧版 19 维 FIFO checkpoint 无法加载到当前网络。单模型 `eval.py` 使用 `delay_checkpoint`。运行完成后，视频和指标由当前配置决定。

## 14. 视频帧率

视频不是 physics 的真实帧率，而是抽帧后的播放编码：

- 无延时评估使用 `RenderCallback(interval=2)`，约为 `31.25 FPS`。
- 随机延时下每 2 个策略决策保存一帧，视频帧率取实测平均 `transition_dt` 计算。
- 按 `0.016～0.096 s` Actor transition 计算，每 2 个策略决策保存一帧时，视频帧率范围约为 `5.2～31.25 FPS`。

随机延时视频使用平均帧率近似，不是严格的可变帧率视频。分析控制性能时应优先看 `episode_time`、`reach_goal` 和 `collision`，不能根据视频播放速度推断 actor 的真实控制频率。

## 15. 实验记录建议

每次实验建议保存：

```text
完整 Hydra 配置
代码版本或 git commit
checkpoint 文件
W&B run id
GPU id 和 seed
eval_env.pt
reach_goal / collision / episode_len / decision_count / episode_time
sim_time_seconds 和 wall_time_seconds
评估视频
```

论文或报告中的主对照建议使用：

```text
相同评估场景 + 相同 seed + 相同物理 episode horizon
主指标：reach_goal、collision
辅助指标：episode_time、equivalent episode length、decision_count
训练曲线横轴：累计 sim_time，而不是单独使用 W&B step
```
