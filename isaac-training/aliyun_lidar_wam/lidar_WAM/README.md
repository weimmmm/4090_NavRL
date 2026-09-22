# NavRL LiDAR WAM：离线版最佳方法、完整流程与实测结果

## 当前结论（2026-09-22）

本文档描述的是**上一版纯离线方法**：使用 PPO 数据集监督训练，在
Isaac Sim 中只做固定环境闭环评估，**没有**使用 DAgger、高层 PPO
selector、在线强化学习或任何 Isaac Sim 权重微调。

当前有实测依据的最佳部署方法是：

```text
冻结 Circular VAE
        +
goal-frame Action-only v2 Flow Action Expert（step 4000）
        +
每次预测 30 个动作、只执行前 10 个动作、重新读取真实 LiDAR 后再规划
```

对应权重为：

```text
outputs/action_expert_action_only_v2/best.pt
```

选择它而不是更复杂的联合权重，原因如下：

- Action-only 在 seed 8 固定环境前 128 条路线的重复实测中，reach goal
  为 **28.125%–34.375%**；最近用于 WAM/数据动作对照的结果为
  **28.125%（36/128）**。
- 独立 Future Model 的 `step 8500` 已经通过离线 copy/shuffle 门槛，适合
  做未来 LiDAR 预测实验。
- 但联合训练检查中，Action Expert 虽然拟合良好，联合后的 world 分支
  `cd_paper_m2=1.6588`，差于 copy 的 `1.2515`，world gate 未通过。
- 因此当前正式部署应使用 Action-only；Future 和 Joint 结果作为独立实验，
  不能因为结构更复杂而标成最佳策略。

## 1. 方法总览

```text
PPO 完整轨迹（seed 0-9，每 seed 512 条）
        │
        ├─ 固定 .pt：mesh、起点、目标、姿态、路线方向
        └─ HDF5：物理 LiDAR、状态、10 步 PPO action、终止信息
                         │
                  时间索引和坐标审计
                         │
            冻结 Circular VAE 编码 LiDAR latent
                         │
       ┌─────────────────┴─────────────────┐
       │                                   │
Action-only v2                         Future Model
预测未来 30 步动作                     直接预测 t+3 LiDAR
正式闭环控制                           离线生成/诊断
       │                                   │
执行前 10 步后重新感知               Joint（实验，不作正式部署）
       │
Isaac 固定 .pt 环境闭环评估（不更新权重）
```

### 1.1 问题定义与方法选择原则

目标不是在离线数据上把某一个误差降到最低，而是学习一个能够闭环执行的
导航策略：给定时刻 `t` 的真实 LiDAR、无人机状态、固定目标和上一段动作，
产生未来 30 个速度指令；仿真器执行前 10 个指令后，在 `t+1` 重新感知并
规划。用符号表示为：

```text
o_t = {LiDAR_t, goal_t, proprio_t, action_{t-1}, mask_{t-1}}
pi_theta(o_t, noise) -> a_hat_{t+1:t+3} in [0,1]^(30x3)
execute(a_hat_{t+1}[0:10]) -> observe o_{t+1} -> replan
```

其中一个数据帧间隔代表 10 个物理仿真步，所以 `t+1/t+2/t+3` 分别对应
未来 10/20/30 步。方法遵循四条原则：

1. **因果性**：策略只能看到当前及历史信息，未来 LiDAR、未来状态和终止
   标签不得进入策略输入。
2. **训练—部署同义**：离线和在线必须使用相同的时间索引、goal frame、
   动作缩放、LiDAR 排列和执行长度。
3. **滚动时域控制**：虽然预测 30 步，但只开环执行 10 步，避免把较远期
   预测误差连续积累 0.48 秒。
4. **闭环结果优先**：离线 MAE、Chamfer 用于排错和筛选；最终优劣由固定
   `.pt` 环境中的 reach goal、collision、OOB 和 timeout 决定。

“最佳方法”按证据强度选择，而不是按模型复杂度选择：同一固定路线的大样本
闭环结果，高于离线验证指标；离线验证指标又高于训练 loss。按照这一原则，
当前最佳仍是 Action-only step 4000，而不是未通过 world gate 的 Joint。

### 1.2 数据集

数据目录：

```text
../datasets/wam_static_seed00_09/
├── environments/static350_seed00_n512.pt ... static350_seed09_n512.pt
├── train/seed_0000 ... seed_0007
├── val/seed_0008
├── test/seed_0009
└── manifest.json
```

数据集包含 **624,640 个物理 LiDAR 帧、5,120 条完整 episode**。每张地图
有 350 个静态障碍物、无动态障碍物，每 10 个仿真步（0.16 秒）保存一帧。
PPO 数据动作在全部十个 seed 上的终止统计为 4,315 reach goal、555
collision、249 timeout、1 out of bounds。

固定划分为：

- seed 0–7：训练，共 480,752 个有效三段窗口；
- seed 8：验证，共 62,784 个有效三段窗口；
- seed 9：最终测试，共 61,325 个有效三段窗口。

Action-control 的一段窗口数量分别为 488,928 / 63,806 / 62,345。seed 8/9
不进入训练统计量。索引禁止跨 scene、跳帧和不完整动作。

### 1.3 严格时间对齐与样本构造

对于当前记录帧 `i`：

- 当前 LiDAR、状态和目标来自 `i`；
- previous action 是 `i` 保存的上一物理帧到当前帧的 10 步动作；
- Action Expert 的 30 步标签来自 successor `i+1、i+2、i+3`；
- Future Model 的目标是 `i+3` 的 LiDAR，即 30 个物理步、0.48 秒以后；
- scene 首帧 previous action 置零，同时 mask 置零；
- 未来 30 步内 collision/OOB 的窗口不用于 world target，但更早的有效避障
  片段仍然保留。

全数据 normalized action 到 world action 的最大重建误差为
`4.76837e-7`，离线与在线 goal-frame feature 最大误差为 `0`。

窗口构造不是按 HDF5 行号简单切片，而是沿 `next_token` 建链。对每个候选
窗口依次检查：scene 相同、`frame_index` 连续、每段 `step_delta=10`、30 个
动作 mask 完整且所有数值有限。这样能够防止 episode 边界附近出现
off-by-one 或把下一架无人机的动作拼进当前样本。

失败轨迹不会整条删除。只有未来监督段本身已经 collision/OOB 的窗口被排除，
碰撞前较早的低净空和转弯片段仍保留。这一点很重要：若只保留成功轨迹，模型
会看不到接近障碍物时 PPO 如何纠偏；若保留碰撞发生后的无效控制，则又会把
失败动作当作教师标签。

### 1.4 坐标系与动作协议

训练和部署调用同一套固定 goal frame：

```text
x_goal = 起点指向目标点的固定 target_dir_2d
y_goal = world_z × x_goal
z_goal = 世界坐标向上
```

goal vector、线速度、角速度和 gravity 都转换到这个 frame；LiDAR 保持
原始 sensor-yaw 排列。Action Expert 输出 `[B,30,3]`、范围 `[0,1]`，语义
是 goal-frame normalized velocity。

部署转换只能执行一次：

```text
v_goal  = 4 * action - 2             # 每轴 [-2, 2] m/s
v_world = goal_basis @ v_goal        # goal frame → world frame
```

在线只执行 `action[:,0:10]`，随后重新读取真实 LiDAR 并再次规划。不能把
输出重复旋转，也不能把它当作 body-frame action。

策略条件的具体定义为：

```text
goal[4]     = [relative_goal_x, relative_goal_y, relative_goal_z, distance]
proprio[10] = [world_height,
                velocity_goal_xyz,
                angular_velocity_goal_xyz,
                gravity_goal_xyz]
```

`target_dir_2d` 在环境 reset 时确定，此后整条路线不随无人机朝向变化。因此
这里的 goal frame 不是机体系。LiDAR 仍按传感器 yaw 排列，网络通过周期方位
位置编码辨别左右障碍物；不能再用机体四元数旋转一次 LiDAR 或 action。

### 1.5 冻结 Circular VAE

使用 `lidar_wam/vae/circular/` 的 step 29,500 权重，输入为
`[2,108,20]` range image，最后两列是 padding。VAE 始终冻结，只负责把
LiDAR 编码为 `[4,27,5]` latent。

seed 8 固定 4,096 帧的 gate：

| 指标 | 实测 | 门槛 |
|---|---:|---:|
| valid range MAE | 0.006094 | ≤ 0.10 |
| mask F1 | 0.999576 | ≥ 0.70 |

VAE 的方法角色是**固定感知压缩器**，而不是控制器。原始 range image 的两
通道分别表达归一化距离和有效回波 mask；108 个方位格是环形维度，因此编码器
在方位方向使用 circular padding，在仰角方向使用普通零 padding。训练 Action
Expert 时不更新 VAE，原因是：

- 保持所有离线 latent cache 与在线编码语义一致；
- 避免 action loss 为了降低动作误差而破坏几何重建；
- 把控制失败定位到策略，而不是同时变化的感知表征。

经 VAE 编码后：

```text
LiDAR [B,2,108,20] -> latent z_t [B,4,27,5]
```

后两列仰角 padding 不参与有效距离指标。缓存同时绑定数据集 manifest、VAE
权重和 scaling factor 的哈希，防止训练时误用另一套 VAE latent。

### 1.6 Action-only v2（当前推荐控制器）

#### 1.6.1 网络输入与结构

Action Expert 输入当前 LiDAR latent、goal-frame goal/proprio、上一段
`10×3` action 和 mask，输出未来 `30×3` normalized action。网络的数据流为：

```text
z_t [B,4,27,5]
  -> 3 层 circular Conv（4->128->256->512）
  -> flatten + LayerNorm + 周期方位/仰角位置编码
  -> observation tokens [B,135,512]

context = concat(
    135 个 LiDAR tokens,
    1 个 goal token,
    1 个 proprio token,
    10 个 previous-action+mask tokens
)

30 个 noisy action tokens
  -> 8 层 Transformer，width=512，8 heads，FFN=2048
  -> action self-attention
  -> 对 causal context 做 cross-attention
  -> flow velocity [B,30,3]
```

位置编码显式包含 `sin(yaw)`、`cos(yaw)`、二倍角和 elevation。它解决了把
二维特征展平后注意力无法区分“左侧障碍物”和“右侧障碍物”的问题，同时在
360° 接缝处连续。上一段 action 的 mask 作为第 4 个输入量嵌入，使 scene
首帧的全零占位不会被误解为真实的零速度历史。

#### 1.6.2 动作变换与条件归一化

PPO action 接近边界，直接在 `[0,1]` 空间做高斯 flow 会产生大量越界值。
训练前先做逐轴 logit 和标准化：

```text
u       = clip(a, 1e-4, 1-1e-4)
ell     = log(u) - log(1-u)
x       = (ell - mean_train) / std_train
```

goal 和 proprio 同样只使用 seed 0–7 的均值、标准差归一化。推理结束后执行
逆标准化和 sigmoid，保证输出回到 `[0,1]`。seed 8/9 绝不参与这些统计量，
因此验证和测试不是对目标 seed 的拟合。

#### 1.6.3 Rectified Flow 训练目标

令 `x` 为 GT 30 步动作的 logit-normalized 表示，`epsilon ~ N(0,I)`，随机
采样 `sigma ~ U(0,1)`，构造插值状态：

```text
x_sigma = (1-sigma) * x + sigma * epsilon
v_star  = epsilon - x
```

网络预测 `v_theta(x_sigma, sigma, o_t)`，主损失为 flow velocity MSE。由预测
速度重建干净动作 `x0_hat = x_sigma - sigma*v_theta`，再加入 endpoint 和相邻
动作变化约束：

```text
L_flow  = mean(w_h * ||v_theta - v_star||^2)
L_x0    = mean(w_h * |x0_hat - x|)
L_delta = mean(w_h * |Delta(x0_hat) - Delta(x)|)

L_action = L_flow + 0.25*L_x0 + 0.05*L_delta
```

当前最佳 step 4000 对应的已保存训练配置中，三个 10 步段的 `w_h` 是
`1.5/1.25/1.0`，用来加强实际会执行的前 10 步。**复现该权重时应以
`outputs/action_expert_action_only_v2/config.json` 为准**；当前源码后续实验
已把默认 horizon weight 改为等权，不能把两个实验混为一谈。

`L_delta` 不是要求动作平滑到不转弯，而是要求预测的动作变化与 PPO 的动作
变化一致，减少只会复制上一动作的惯性解。

#### 1.6.4 困难样本采样

采样比例：

```text
50% 全部有效窗口
25% 最低净空 25% 窗口
25% 转弯/action-change 最大 25% 窗口
```

Action-only 损失使用 flow matching，并加入 x0 和 action-delta 约束；三个
十步 chunk 的权重为 `1.5 / 1.25 / 1.0`，让真正执行的前十步优先。训练配置
是 7 卡 BF16、global batch 11,760；虽然配置目标为 20,000 step，验证最优
checkpoint 出现在 **step 4000**。

三个采样池允许重叠：一个低净空且急转弯的样本可以同时被两个困难池抽到。
目的不是改变真实数据分布，而是提高单次优化中安全关键纠偏片段的出现频率。
验证集仍按固定样本计算，不使用训练采样权重。

#### 1.6.5 Flow 推理

推理从固定 policy seed 生成的高斯噪声 `x_1` 开始，用 10 个 Euler 步从
`sigma=1` 积分到 `0`：

```text
x_next = x_current + (sigma_next-sigma_current)
                         * v_theta(x_current, sigma_current, o_t)
a_hat  = sigmoid(x_0 * std_train + mean_train)
```

固定 route ID 和 policy seed 后，同一路线的噪声确定，便于逐 checkpoint
公平比较。部署只采用 `a_hat[:,0:10]`；后 20 步提供长期趋势监督，但不会在
没有新观测时直接执行。

step 4000 在 seed 8 的 4,096 个离线窗口上：

| 指标 | Action Expert | repeat-last | 改善 |
|---|---:|---:|---:|
| 30 步 overall MAE | 0.035738 | 0.043575 | 17.99% |
| 前 10 步 MAE | 0.026009 | 0.026228 | 0.83% |
| 低净空 MAE | 0.071663 | 0.083164 | 13.83% |
| 转弯 MAE | 0.085911 | 0.103051 | 16.63% |

normalized action MAE 乘以 4 即对应速度误差。例如前 10 步 MAE
`0.026009` 对应约 `0.1040 m/s`。各动作轴 MAE 为：

| 轴 | normalized MAE | 速度 MAE |
|---|---:|---:|
| goal-x | 0.010684 | 0.04274 m/s |
| goal-y | 0.055837 | 0.22335 m/s |
| world-z | 0.040691 | 0.16276 m/s |

这里最重要的限制是：**前 10 步只比 repeat-last 好 0.83%**。这解释了模型
离线 overall MAE 看起来不错，但闭环碰撞率仍高；它仍然过多依赖动作惯性，
横向和高度修正能力不足。

### 1.7 Future Model（独立有效，但不是当前正式控制器）

Future Model 使用当前 latent、连续三段真实 `10×3` 动作和因果状态，直接
预测 `t+3` LiDAR latent，不递归生成 t+1/t+2。当前最优为：

```text
outputs/world_direct_t3_v2/best.pt（checkpoint step 8500）
```

网络把未来 latent 的带噪版本与当前 latent 在通道维拼接：

```text
[z_{t+3,sigma}, z_t] -> [B,8,27,5]
```

三段动作分别嵌入，并加 step-within-segment 与 segment ID 编码，形成 30 个
768 维 action tokens；当前因果 ego state 形成第 31 个 token。条件 UNet 的
通道为 `256/512/512`、每层 4 个 residual block，通过 cross-attention 预测
扩散噪声。它只直接学习：

```text
p(z_{t+3} | z_t, a_{t+1:t+3}, causal_state_t)
```

而不是递归调用三次单帧模型。直接 t+3 避免 t+1 的解码/再编码误差逐次累积，
也正好覆盖 Action Expert 生成的完整 30 步候选。训练时输入真实 PPO 动作；
评估 shuffle-action 对照用于确认模型不是无视 action、只复制当前 LiDAR。

训练目标以 DDPM epsilon MSE 为主，并在可稳定解码的 timestep 子集加入 latent
L1、mask BCE、range L1、presence 与 empty-frame 辅助项。模型选择不只看
diffusion loss，而要求点云指标同时超过 copy current，并在打乱动作后退化。

step 8500 在 4,096 个固定验证样本上的结果：

| 指标 | Future prediction | copy current | shuffle action |
|---|---:|---:|---:|
| `cd_paper_m2`（代码中的双向平方和） | 1.10443 | 1.23276 | 1.21967 |
| 论文式 `1/2 × 双向平方和` | 0.55222 | 0.61638 | 0.60984 |
| symmetric NN distance | 0.55236 m | 0.87110 m | 0.63132 m |
| valid range MAE | 0.95549 m | 3.85233 m | 1.34050 m |
| mask F1 | 0.81728 | 0.56647 | 0.78018 |
| false-empty frames | 1 | 6 | 2 |

这里的论文式平方 Chamfer 为：

```text
0.5 * (mean_{p in P} min_{q in Q} ||p-q||²
     + mean_{q in Q} min_{p in P} ||q-p||²)
```

当前代码字段 `cd_paper_m2` 保存的是括号内两项之和；若按上式报告，需要
除以 2。无论是否除以 2，相对排序和改善比例不变。prediction 相对 copy 的
平方 Chamfer 改善约 **10.41%**，shuffle action 相对 prediction 恶化约
**10.43%**，说明模型确实使用了动作条件。

### 1.8 Joint WAM 的定位

联合训练加载 Action-only 和 Future step 8500，通过共享 observation tokens
和 `observation_to_world` adapter 让 world loss 影响共享编码器。训练期间
Future UNet 前 500 step 冻结，world loss 权重从 0.05 增加到 0.25，最终
UNet LR 为 `1e-6`。

但该联合版本目前只应视作实验：500-step 检查中 Action 分支通过门槛，
world 分支却退化到 `cd_paper_m2=1.65878`，比 copy 的 `1.25149` 更差，且
shuffle 只恶化 1.07%，因此总 gate 为 false。不能把
`outputs/action_expert_joint_v2_future8500/latest.pt` 当作已验证优于
Action-only 的正式权重。

联合结构中，当前 LiDAR 只编码一次。相同的 135 个 observation tokens 一路
进入 Action Transformer；另一路做 token mean pooling，经零初始化的
`512->768` adapter 加到 world condition。零初始化保证联合训练第 0 步的
Future 输出严格等于已加载的 step 8500，而不是因随机 adapter 立刻破坏它。

理想联合目标为：

```text
L_joint = L_action + lambda_world * L_world
```

Future target 只能进入 world loss，绝不作为 Action Expert 的输入，因此没有
“训练时看到未来、部署时看不到”的信息泄漏。联合训练是否真正有效，需要同时
满足两类证据：Action 前 10 步改善，以及 Future 优于 copy 且对 shuffled
action 敏感。当前 joint 只满足前者的一部分，没有满足 world gate，所以按
方法选择原则回退到 Action-only。

### 1.9 完整训练阶段与停止条件

上一版方法的完整离线流程分成五个有硬门槛的阶段：

1. **数据审计**：建立 token 链，验证窗口数量、动作 mask、scene 边界、坐标
   转换和 `.pt` 环境哈希。失败则停止，不能靠训练“吸收”错位。
2. **VAE gate 与 latent cache**：冻结 VAE，在 seed 8 固定帧上验证重建；
   通过后一次性缓存 latent。VAE 不参与后续反向传播。
3. **Action-only**：只优化 observation encoder 和 Action Flow Expert；每
   500 step 用 seed 8 固定窗口验证，保留候选，不查看 seed 9。
4. **Future-only**：从已有 world UNet 初始化，用新数据的严格 `t+3` 目标
   重训；必须同时通过 copy 和 shuffled-action 因果门槛。
5. **Joint 实验**：只有前两支各自通过门槛才允许联合；若任何一支退化，正式
   部署回退到对应的独立权重。

Action checkpoint 的离线选择分数定义为：

```text
score = 0.40 * first10_MAE
      + 0.25 * low_clearance_first10_MAE
      + 0.25 * turning_first10_MAE
      + 0.10 * overall_MAE
```

这里分数越低越好。repeat-last 和 mean-action 都是必须报告的 baseline：
repeat-last 检验模型是否只是延续惯性，mean-action 检验模型是否退化为数据集
平均速度。仅仅训练 loss 下降，或者 30 步 overall MAE 较低，不能证明它学会
了闭环避障。

### 1.10 在线闭环执行算法

上一版正式 Action-only 的单次规划逻辑如下：

```python
while not terminated and sim_step < 2200:
    lidar = read_physical_lidar()
    latent = frozen_vae.encode(lidar)

    goal, proprio = goal_frame_features(
        drone_state, target_position, fixed_target_dir_2d
    )
    action_chunk = flow_policy.sample(
        latent, goal, proprio, previous_10_actions, previous_mask,
        flow_steps=10, policy_seed=f(route_id)
    )                           # [30,3], normalized goal-frame velocity

    for action in action_chunk[:10]:
        velocity_goal = 4 * action - 2
        velocity_world = goal_to_world_once(
            velocity_goal, fixed_target_dir_2d
        )
        isaac_step(velocity_world)
        if collision or out_of_bounds or reach_goal:
            break

    previous_10_actions = actually_executed_normalized_actions
```

终止帧可能不足 10 步，所以历史 mask 必须只标记实际执行动作。到达目标后立即
终止，不需要额外悬停。评估 512 条路线时按 `0/128/256/384` 四个 route slice
依次运行，并保留原 route ID；不能一次并发 512 台后把 PhysX 容量异常当成
策略 OOB。

### 1.11 方法为何有效、又为何目前只有约 28%–34%

这套方法已经解决三类基础错误：数据动作可以在固定环境复现 PPO 的 84.375%；
训练和在线坐标完全一致；Future 确实对 action 条件敏感。因此当前低成功率
不能再归因于环境文件或坐标旋转。

它仍未接近 PPO 的核心原因是：闭环真正执行的是前 10 步，而 step 4000 的
前 10 步 MAE只比 repeat-last 改善 **0.83%**。30 步 overall 改善 17.99%
主要包含更远期片段，并不能直接挽救下一次碰撞。y/z 轴误差又明显大于前向轴，
说明横向绕障和高度修正仍弱。Future Model 的作用限于独立预训练和联合优化；
Isaac 部署时只运行联合训练后的 Action Expert，不再额外生成或筛选动作候选。

因此，本文所说的“最佳”是**现有上一版离线实验中证据最强、最可复现的模型**，
不是已经达到最终 70% 目标的模型。这个区分也是整套方法论的重要组成部分。

## 2. Isaac Sim 闭环评估结果（无在线微调）

### 2.1 指标定义

- `reach_goal`：无人机与目标点距离 `< 0.5 m`；**不要求悬停**。
- `collision`：LiDAR/环境检测到与障碍物碰撞。
- `out_of_bounds`：高度 `<0.2 m` 或 `>4.0 m`。
- `timeout`：达到 2,200 个仿真步仍未结束。
- `mean_episode_steps`：终止前平均物理步数。
- `mean_path_length_m`：实际飞行轨迹长度，不是起终点直线距离。
- `mean_min_clearance_m`：每条轨迹最小 LiDAR 净空的平均值。

每条路线只记录一个最终终止原因，因此四种终止率应满足：

```text
reach_goal_rate + collision_rate + out_of_bounds_rate + timeout_rate = 1
rate(reason) = count(reason) / 固定路线总数
```

聚合时分母必须是预先指定的全部 route，而不是只统计成功写出日志的 episode。
四个 128-route 分块合并时按 count 求和后除以 512，不能简单平均缺失或长度
不同的 JSON。

这些指标分成三层，含义不能互相替代：

| 层级 | 指标 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| 感知重建 | range MAE、mask F1、Chamfer | VAE/Future 是否保留几何 | 策略是否会避障 |
| 动作克隆 | first-10/axis/hard-case MAE | 是否拟合 PPO 局部动作 | 误差闭环累积后是否成功 |
| 闭环导航 | reach/collision/OOB/timeout | 最终导航能力 | 单独定位是哪一模块失效 |

`mean_episode_steps` 或 `mean_path_length_m` 较大只说明无人机飞得久或绕得远，
不代表到达目标。碰撞前长距离飞行、在目标附近穿过但没有进入 0.5 m 球、以及
持续游走到 timeout，都会拉高轨迹长度但仍计为失败。相反，较短路线快速到达
可以得到更高成功率和更低平均步数。

固定路线评估还必须固定以下变量：环境 `.pt`、route ID、checkpoint、policy
seed、Flow/DDIM steps、执行 horizon 和终止阈值。否则两个百分比不是严格的
同条件对照。

### 2.2 数据与坐标正确性基准

直接重放 seed 8 数据集保存的 world actions，512 条路线与 `.pt` 环境逐元素
对齐：

| 指标 | 结果 |
|---|---:|
| reach goal | 84.375%（432/512） |
| collision | 10.9375%（56/512） |
| timeout | 4.6875%（24/512） |
| termination reason match | 100% |
| initial LiDAR max error | 0 |
| position/velocity/LiDAR replay error | 0 |

这证明数据集动作、固定环境和坐标转换能够复现；WAM 的低成功率不是 `.pt`
环境或 normalized→world action 转换错误。

### 2.3 当前最佳 Action-only 闭环结果

现存三份 step 4000、seed 8 前 128 条固定路线、policy seed 42、Flow 10 步
的评估记录存在波动：

| 运行 | Reach goal | Collision | OOB | Timeout |
|---|---:|---:|---:|---:|
| 对照运行 | 28.125% | 69.531% | 0% | 2.344% |
| 另一次完整运行 | 33.594% | 64.844% | 0% | 1.563% |
| 当前记录最高 | 34.375% | 64.063% | 0% | 1.563% |

为了避免挑最好的一次作为结论，当前 README 将 **28.125%** 作为保守可复现
基线。对应平均轨迹为 507.62 步、14.10 m；成功轨迹平均 586.81 步、
17.35 m。

512 台同时仿真时曾出现约 75% OOB 的容量异常，与 128 台结果不一致；这些
512 并发结果不作为策略质量指标。正式 512 路线评估必须按每次 128 台分块，
并保留 source route ID。当前还没有一份四个分块都通过一致性检查的正式
Action-only 512 路线总成绩，因此不能宣称已达到 70% 目标。

## 3. 推荐的离线训练与评估顺序

普通训练使用 `lidar-wam-train` 的 Conda Python，不需要 Isaac Sim Python：

```bash
docker start lidar-wam-train
docker exec -it lidar-wam-train bash
cd /workspace/NavRL/isaac-training/aliyun_lidar_wam/lidar_WAM

PY=/opt/conda/bin/python
ROOT=/workspace/NavRL/isaac-training/aliyun_lidar_wam
DATA=$ROOT/datasets/wam_static_seed00_09
LATENTS=$PWD/outputs/wam_static_seed00_09/latents
INDEX=$PWD/outputs/wam_static_seed00_09/index
GPUS=0,1,2,3,4,5,7
```

### 3.1 检查数据并缓存冻结 VAE latent

```bash
$PY -m lidar_wam.runner.prepare_v2 inspect \
  --dataset-root "$DATA" --index-root "$INDEX"

CUDA_VISIBLE_DEVICES=0 $PY -m lidar_wam.runner.prepare_v2 cache-latents \
  --dataset-root "$DATA" --index-root "$INDEX" --output "$LATENTS" \
  --batch-size 128 --resume
```

### 3.2 训练 Action-only

```bash
CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run \
  --standalone --nproc_per_node=7 \
  -m lidar_wam.runner.action_expert_joint train-action \
  --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
  --steps 20000 --micro-batch-size 1680 --grad-accumulation 1 \
  --precision bf16 --eval-every 500 --eval-batch-size 256 --workers 8
```

优先检查 `candidate_step_004000.pt`，但新训练不能因为旧实验 step 4000 最好
就跳过验证；仍应按 seed 8 的离线 first-10/low-clearance/turning 指标选权重。

### 3.3 可选：独立训练 Future Model

```bash
CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run \
  --standalone --nproc_per_node=7 \
  -m lidar_wam.runner.world_direct_horizon train-v2 \
  --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
  --init-checkpoint outputs/action_expert_joint/best.pt \
  --steps 20000 --micro-batch-size 384 --grad-accumulation 1 \
  --precision bf16 --eval-every 500 --eval-batch-size 64 --workers 8
```

只有 copy improvement ≥10% 且 shuffle degradation ≥5% 才保留。当前满足
该条件的是 step 8500。

### 3.4 固定环境闭环评估

Isaac Sim 只在这里启动，不反向传播、不更新 checkpoint：

```bash
docker start navrl-train
docker exec -it navrl-train bash
cd /workspace/NavRL/isaac-training/aliyun_lidar_wam

/isaac-sim/python.sh -m isaac_eval.evaluate \
  --device cuda:0 \
  --environment datasets/wam_static_seed00_09/environments/static350_seed08_n512.pt \
  --checkpoint lidar_WAM/outputs/action_expert_action_only_v2/best.pt \
  --route-start 0 --route-limit 128 --episodes 128 \
  --flow-steps 10 \
  --output isaac_eval/results/offline_action_only_seed8
```

seed 8 只用于模型选择；所有配置锁定后才允许对 seed 9 做一次最终测试。

## 4. 当前限制和正确结论

1. 当前最好的纯离线 WAM 仍远低于 PPO 数据动作的 84.375%。
2. 最大瓶颈是实际执行的前 10 步：离线只比 repeat-last 改善 0.83%。
3. Future step 8500 能预测 t+3，但“能预测未来”不等于“Action Expert 已经
   使用未来信息做出更安全的动作”。
4. 联合梯度目前会损伤 world 质量，因此 Joint 权重不应替代 Action-only。
5. 28.125% 是当前保守的 128-route 基线，不是完整 seed 8 成绩，也不是
   70% 目标已经完成。

## 5. 指标与结论的原始文件

- 数据集总清单：`../datasets/wam_static_seed00_09/manifest.json`
- 时间/坐标审计：`outputs/wam_static_seed00_09/index/inspection.json`
- VAE gate：`outputs/wam_static_seed00_09/latents/vae_gate.json`
- Action-only step 4000：
  `outputs/action_expert_action_only_v2/best_metrics.json`
- Future step 8500：`outputs/world_direct_t3_v2/best_metrics.json`
- seed 8 数据动作 512 路线精确重放：
  `../isaac_eval/results/dataset_action_replay_seed8/world_routes000_511.json`
- Action-only 28.125% 闭环记录：
  `../isaac_eval/results/wam_vs_dataset_seed8/action_expert_best_seed8_n128_routes000_k1_policy42.json`
---

## 历史实验记录

以下内容保留旧数据、早期单步 world model 和其他架构实验，不能覆盖上面的
当前离线版结论。

> 新采集的 seed 0–9、512 路线自包含数据集请使用
> [TRAINING_V2.md](TRAINING_V2.md)。该入口修复了旧 Action Expert 的
> body-frame 条件/goal-frame 输出错配，并提供 DDP、分层采样、坐标语义
> checkpoint 与固定 `.pt` 闭环评估。下文保留的是旧数据实验记录。

Epona 思路的验证（动作预测位姿、几何引导扩散、两帧生成历史微调，以及使用 3／5 张历史雷达帧的 MST 适配）见 [EPONA_PROBE.md](EPONA_PROBE.md)。

最新的固定代表性样本、修正后的雷达几何基线和现有模型同批复评结果见 [REPRESENTATIVE_BASELINE.md](REPRESENTATIVE_BASELINE.md)。下文旧版“每个 seed 前 128 条”基线仅用于历史对照；其中依赖旧 `estimate_range_out` 的真实位姿重投影受角度错位影响，不应再用于路线判断。

新增 NWM CDiT 预测器的独立适配实验：见 [NWM_ADAPTER.md](NWM_ADAPTER.md)。它复用 NWM 官方 CDiT 和扩散训练/采样代码，接入现有环形 VAE 与真实的 10 步动作。已完成 BF16、batch 256、5,000 次更新的训练：验证/测试 256 样本点云 Chamfer 分别为 0.416/0.419 米，均未超过复制上一帧基线 0.207/0.242 米；详见实验文档和 `outputs/world_nwm_cdit_full/`。

本目录包含运行所需的 LaGen Diffusers 源码和 VAE 配置，不从外层 `LaGen` 目录导入代码。**当前默认 VAE 是 `lidar_wam/vae/circular/` 中已训练 29,500 步的环形 NavRL VAE**：先构建双通道 `AutoencoderKL`，再应用本目录的 `replace_down`、`replace_conv`、`replace_attn`，最后加载本目录的 safetensors 权重。世界模型仍使用本目录 Diffusers 的 `UNet2DConditionModel`，动作条件 UNet 为 `256/512/512` 通道、每层 4 个残差块、768 维交叉注意力。官方 nuScenes VAE 权重不用于此数据集。

每个样本输入上一帧 `2×108×20` range image、10 个连续仿真步的动作和上一帧自身状态；目标是这 10 步执行完之后的一帧。缓存字段 `prev_ego_feats` 虽有 5 维，但其中第 3、4 维由下一帧状态计算；因果训练入口将其置零，只使用其余 3 维上一帧信息。最初的 `world_full` 权重曾使用全部 5 维，含未来信息泄漏；旧普通 VAE 的修正实验在 `world_causal_full`，当前环形 VAE 的修正实验在 `world_circular_causal_full`。距离通道按 `(距离米/5)-1` 归一化，掩码通道为 `±1`。最后两个仰角列为填充，训练损失与指标均不计入。

## 环境与数据

目录布局与 LaGen 相似：`lidar_wam/runner/stage1.py` 是训练与评估实现，`lidar_wam/runner/utils.py` 保存环形卷积等替换代码，`third_party/diffusers/src/` 是随项目保存的 Diffusers 源码，`lidar_wam/vae/circular/` 保存**当前使用的**环形 VAE 配置和权重。`lidar_wam/vae/config.json` 与 `outputs/vae_full/` 保留旧普通 VAE，供历史结果复核。`scripts/` 放数据转换和诊断入口，`outputs/` 放检查点与指标；根目录 `stage1.py` 是兼容入口。

先准备与 PPU 匹配的 PyTorch，再安装 `requirements.txt` 中的其余 Python 依赖。训练数据可放在本目录的 `data/lagen_cache/`，也可用 `--data-root` 指向外部 HDF5 目录。当前阿里云数据仍在原位置，运行示例：

```bash
cd /mnt/workspace/wam_trining/trining/lidar_WAM
export LIDAR_WAM_DATA_ROOT=/mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache
PY=python
ENTRY=stage1.py
$PY $ENTRY inspect
```

数据目录中应有 `navrl_static_{train,val,test}.h5`。固定按 seed 划分：训练 0–15，验证 16–17，测试 18–19。仅保留 `action_mask` 全为 1、`step_delta=10`、两帧图像及归一化动作与自身状态均有限的样本。当前有效数为训练 78,329、验证 9,794、测试 9,800。若只保留原始 `.npz`，可用 `scripts/convert_static_dataset.py` 构建 HDF5；转换代码也已放入本目录。

这里的“有效”只保证 `normalized_action_sequence` 有限。PPO 的归一化输出还会依据目标方向旋转成实际世界坐标速度；HDF5 的 `action_sequence` 才是采集时保存的实际指令。审计发现其中部分转换有非有限值，因此使用实际指令的实验会额外过滤这些转换，不能把 78,329 对都视为“10 个实际指令有限”。

```bash
# 只有原始数据时才需要转换；输出目录可设为本项目的 data/lagen_cache
$PY scripts/convert_static_dataset.py /path/to/navrl_static_100k --output data/lagen_cache
$PY scripts/validate_cache.py data/lagen_cache --source-dataset /path/to/navrl_static_100k
```

已有 HDF5 缓存可以直接通过 `--data-root` 或 `LIDAR_WAM_DATA_ROOT` 使用，不需要重新转换。验证脚本提供 `--source-dataset`，用于项目移动后覆盖旧缓存清单中的绝对来源路径。

原始数据是训练好的 PPO 在 Isaac Sim 静态地图上闭环飞行采集的连续轨迹：每个仿真步执行一个三维动作，通常每 10 步（0.16 秒）记录一帧 LiDAR。碰撞、到达或终止可提前保存末帧；缓存只构造同一回合的相邻帧转换，本入口再排除不足 10 步或动作不完整的转换。每个 seed 对应一张不同地图，同一 seed 内有多次重置。原始雷达是 `18×108`、量程 10 米；缓存为 `2×108×20`，其中最后两列是填充。

逐步动作不在点云 `.bin` 中，而在**当前记录帧**的 `.npz` 中，键为 `velocity_commands` 和 `normalized_actions`。采集循环在每个仿真步执行环境前追加一次 PPO 输出，保存雷达帧时写入数组；缓存转换器只复制到 `action_sequence` 和 `normalized_action_sequence`，并用 `action_mask` 标记真实长度。已直接检查原始 `seed-0016-scene-000001-frame-000022.npz`：这两个数组均为 `10×3`，且十行各不相同。

已核对三个 HDF5 文件的全部 **98,302** 行：每行的 `prev_token` 与 `token` 都属于同一 `scene_token`，帧编号恰好相差 1。满足 10 步条件的样本中，训练集有 **78,315/78,329** 行的十个动作并非全部相同，验证和测试集则分别为 **9,794/9,794** 和 **9,800/9,800**；因此加载器保留逐步动作序列，不将其合并。

## 训练顺序

```bash
# 1. 验证环形 VAE 并重新编码 train/val/test；结果写入 latents_circular
$PY $ENTRY cache-latents --batch-size 128

# 2. 校准点云 mask 阈值；结果写入 vae_circular/oracle_val.json
$PY $ENTRY evaluate-vae-chamfer --split val --samples 256

# 3. 128 个样本、200 次更新的动作条件扩散模型拟合检查
$PY $ENTRY train-world --overfit --steps 200 --batch-size 128 --eval-every 50

# 4. 用新的 latent 从头训练动作条件扩散模型；当前已完成 5,000 步
$PY $ENTRY train-world --steps 20000 --batch-size 256 --eval-every 500

# 如果要从当前 5,000 步继续到 20,000 步，用下面这条代替上面的从头训练命令
$PY $ENTRY train-world --steps 20000 --batch-size 256 --eval-every 500 --resume

# 5. 按留出 seed 评估；默认每组 seed 取 128 个样本，共 256 个
$PY $ENTRY evaluate-baselines --split val --samples 256
$PY $ENTRY evaluate --split val --samples 256
$PY $ENTRY evaluate --split test --samples 256
```

默认使用单卡 PPU、FP32、AdamW、学习率 `1e-4`、扩散拟合检查批量 128／完整训练批量 256。扩散主干约 2.74 亿参数（274M，0.274B）；输入 latent 为 `27×5`。显存不够时可减小 `--batch-size`。每次验证保存 `latest.pt`，中断后用同一命令加 `--resume` 从最新保存点续训；不加 `--resume` 会从头训练。新的扩散输出在 `outputs/world_circular_causal_full/`，旧 `world_causal_full/` 保留。`outputs/latents_circular/metadata.json` 记录环形 VAE 的权重 SHA-256 与配置中的 latent scaling factor **0.528144**；训练和评估会检查权重指纹，防止误用旧 latent。当前 `train-vae` 命令仅用于另行从头训练一个环形 VAE 实验，不会替换默认加载的已有权重。

VAE 门槛为验证集有效距离归一化 MAE ≤ 0.10、掩码 F1 ≥ 0.70。`cache-latents` 自动复核并写入 `outputs/vae_circular/gate.json`。还要检查 `vae_circular/oracle_val.json`：真实目标帧经 VAE 编码/解码后的 Chamfer 应低于复制上一帧基线。更换 VAE 权重后必须重新执行 `cache-latents` 并从头训练扩散模型。扩散评估使用 DDIM 20 步，始终以真实上一帧和真实动作作为条件；默认 `--init-strength 1` 从纯噪声采样，也可用较小强度从加噪上一帧 latent 初始化。`evaluation/{val,test}.json` 保存最近一次评估，带 step、strength 和样本数的 JSON 保存逐样本和总体 Chamfer 距离（米），同时与直接复制上一帧及打乱动作条件比较。阶段一通过要求预测相对复制基线降低至少 10%，且打乱动作后误差至少上升 5%。若门槛未通过，报告实际结果，不把冒烟测试视作模型有效。

在验证集两张留出地图各取 128 个有效转换对时，复制上一帧点云的对称 Chamfer 平均为 **0.20709 米**，有效点距离 MAE 为 **0.38881 米**、掩码 F1 为 **0.90474**。该基线与世界模型使用相同的上一帧信息，但不使用动作或自身状态；静态地图和短采样间隔使它相当强。使用真实下一帧位姿变换上一帧点云得到 **0.23252 米**，还用到了未来状态，因此只作诊断，不参与阶段一通过判定。指标和逐样本值在 `outputs/evaluation/baselines_val_n256.json`。

测试集两张未见过的地图各取 128 个有效转换对，复制帧 Chamfer 为 **0.19603 米**、有效点距离 MAE **0.34682 米**、掩码 F1 **0.90411**；详见 `outputs/evaluation/baselines_test_n256.json`。测试 seed 只在模型及验证设置确定后用于最终评估。

旧 `world_causal_full` 使用训练 8,000 步的普通 VAE，完整验证集有效点距离 MAE **0.10547 米**、重建点云 Chamfer **0.10561 米**、mask F1 **0.99600**。当前环形 VAE 约 **13.33M** 参数，在相同帧上分别为 **0.02990 米**、**0.02948 米**、**0.99961**。环形 VAE 世界模型已从头训练 5,000 步：验证集纯噪声生成 Chamfer **0.31748 米**，旧普通 VAE 版本 **0.35539 米**，复制帧基线 **0.20709 米**。完整验证、测试及动作打乱对照见 `RESULTS.md`；当前世界模型尚未通过阶段一验收。

## 实际指令与残差扩散实验

`scripts/run_executed_residual.py` 继续使用本目录的环形 VAE、Diffusers DDPM/DDIM 和同一个条件 UNet。新实验把去噪目标改为“下一帧 latent 减上一帧 latent”，条件改为 10 个实际世界坐标速度指令，以及上一帧无人机的高度、四元数、线速度和角速度。它只使用采集时可获得的上一帧信息与两帧之间的真实指令。对实际指令非有限的转换会过滤；目前训练集剩余 60,368 对，验证集 7,718 对，测试集 7,804 对。统计量仅从训练 seed 计算。

```bash
$PY scripts/audit_action_commands.py --data-root "$LIDAR_WAM_DATA_ROOT"
$PY scripts/run_executed_residual.py prepare --data-root "$LIDAR_WAM_DATA_ROOT"
$PY scripts/run_executed_residual.py train --data-root "$LIDAR_WAM_DATA_ROOT" --overfit --steps 200 --batch-size 128 --eval-every 50
$PY scripts/run_executed_residual.py train --data-root "$LIDAR_WAM_DATA_ROOT" --steps 5000 --batch-size 256 --eval-every 500
$PY scripts/run_executed_residual.py evaluate --data-root "$LIDAR_WAM_DATA_ROOT" --split val --samples 256
$PY scripts/run_executed_residual.py evaluate --data-root "$LIDAR_WAM_DATA_ROOT" --split test --samples 256
```

结果在 `outputs/world_circular_executed_residual_{overfit,full}/` 和 `outputs/evaluation/executed_residual_*.json`，不会覆盖旧实验。评估始终在同一批有限实际指令转换上计算预测、复制帧和打乱动作的 Chamfer；达到既定的相对复制帧降低 10%、打乱动作恶化 5% 两项标准才视为通过。

近期的五帧历史预测 latent、几何重投影与 diffusion 初值组合试验及实际验证/测试结果见 [WORLDVLN_PROBE.md](WORLDVLN_PROBE.md)。

Wan2.2 Video DiT 的独立 1→4 LiDAR 实验见 [WAN_LIDAR.md](WAN_LIDAR.md)。它复用冻结的环形 VAE，输入当前帧和四段真实的十步动作，使用本项目内适配的 Wan2.2-TI2V-5B Video DiT 联合生成四个未来 latent；不会覆盖现有 UNet checkpoint。

直接预测 0.48 秒后单帧的 UNet diffusion 实验见 [DIRECT_T3.md](DIRECT_T3.md)。它输入当前 latent 和连续三段真实动作，直接生成 `t+3`，并与复制帧、一步模型三次自回归和恒速度运动重投影在固定样本上统一比较。

不使用 Wan、直接把 Fast-WAM 风格 Action Expert 接到现有 `t+3` UNet 的联合训练入口见 [ACTION_EXPERT.md](ACTION_EXPERT.md)。它共同优化世界 diffusion 与 30 步动作 flow matching；动作分支只读取当前 LiDAR、当前目标/自身状态和过去动作，不读取未来雷达或未来状态。
