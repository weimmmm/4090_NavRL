# Bidirectional LaGen UNet + Action Flow Expert (v3)

This experiment uses the existing direct-`t+3` LaGen UNet checkpoint. It does
not load Wan. The implementation borrows Fast-WAM's causal joint-training
principle and adapts it to a 2-D UNet world expert.

```text
current LiDAR ─ frozen VAE ─ shared spatial tokens ───────────────┐
                                                                 │
noisy action ─ Action Expert provisional pass ─ clean estimate ──┤
                                                                 ▼
noisy t+3 latent + current latent ─────────────── action-conditioned UNet
                                                                 │
                                                       predicted future noise
                                                                 │
                                                       FutureTokenAdapter
                                                                 │
                                                                 ▼
current/goal/state/history ─ Action Expert final pass ← future cross-attention
                                      │
                               30 action velocities
```

The world UNet already implements action-to-future conditioning. Version 3
adds the missing future-to-action direction. Its spatial denoising prediction
is converted to 135 directional tokens and read by zero-gated cross-attention
in only the last two Action Transformer blocks. The gate starts at zero, so
the merged pretrained checkpoints retain their exact action behavior before
the first update.

The action branch never receives the clean future latent, future range image,
future state, or future target pose. The UNet is conditioned by the clean
action estimate reconstructed from the provisional action-flow velocity, not
by privileged PPO actions (`--world-action-teacher-forcing` defaults to zero).
Training and deployment consequently use the same bidirectional denoising
path.

The policy target is three ordered `10×3` normalized PPO action chunks. Action
flow is trained in standardized logit space and converted back with a sigmoid,
so exported actions remain in `[0,1]`. At deployment, generate 30 actions but
execute only the first 10 simulator steps, observe a new LiDAR frame, and plan
again.

## Inputs

- Current circular-VAE latent: `[4,27,5]`.
- Goal: fixed goal-frame relative XYZ and Euclidean distance.
- Proprioception: height, goal-frame linear/angular velocity, and goal-frame
  gravity vector.
- Previous action chunk: `[10,3]` plus a validity mask. It is zero/masked at
  the start of an episode.

Goal metadata comes from immutable `frames.jsonl`; it is deliberately not
invented from the HDF5 cache. World targets and actions still come from the
existing latent/HDF5 caches and preserve the seed split.

## Current dataset commands

```bash
cd /workspace/NavRL/isaac-training/aliyun_lidar_wam/lidar_WAM
export PYTHONPATH=.:third_party/diffusers/src
DATA=../datasets/wam_static_seed00_09
LATENTS=outputs/wam_static_seed00_09/latents
INDEX=outputs/wam_static_seed00_09/index
ACTION=outputs/action_expert_action_only_v2/best.pt
WORLD=outputs/world_direct_t3_v2/best.pt

python -m lidar_wam.runner.action_expert_joint train-joint \
  --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
  --action-checkpoint "$ACTION" --world-checkpoint "$WORLD" \
  --run-name joint_bidirectional_v3 --steps 10000 \
  --future-attention-layers 2 --fusion-warmup-steps 1000 \
  --world-action-teacher-forcing 0
```

During the first 1,000 updates, the Action backbone, shared observation encoder
and UNet have learning rate zero; only the new adapter, cross-attention and
gates learn. After warmup the existing staged world schedule applies. Action
loss uses relative horizon weights `1.0/0.3/0.1`, normalized to preserve its
overall scale, because only the first ten commands are executed.

V3 checkpoints retain the complete UNet and must be deployed directly. The
compact Action-only exporter intentionally rejects them. Isaac inference uses
the same coupled DDIM/action-flow loop and discards only the generated future
latent after the final action chunk is produced.

## TensorBoard

`train-action` and `train-joint` write TensorBoard events from DDP rank 0 only.
Events are stored inside each run directory under `tensorboard/`; resuming a
run with `--resume` keeps the same step axis and purges stale events after the
checkpoint step.

Important panels are:

- `train/loss`, `train/action_loss`, and `train/world_loss`;
- `train/action_flow_mse`, `train/action_x0_l1`, and
  `train/action_delta_l1`;
- `train/world_epsilon_mse`, `train/world_x0_l1`, and
  `train/world_to_shared_grad_norm`;
- `train/future_attention_gates/*`, adapter gradient/weight norms, and peak
  allocated GPU memory;
- `validation/action/*`, including first-10, low-clearance, turning, axis and
  repeat-last metrics;
- `validation/world/{prediction,copy,shuffled}/*`, including squared Chamfer,
  range MAE, mask F1 and the world-model gate.

For the current seven-GPU run, start TensorBoard inside the training container:

```bash
docker exec -d lidar-wam-train tensorboard \
  --logdir /workspace/NavRL/isaac-training/aliyun_lidar_wam/lidar_WAM/outputs/action_expert_joint_bidirectional_v3_mb256 \
  --host 0.0.0.0 --port 6006
```

Point `--logdir` at the run directory one level above `tensorboard/`. The UI
then lists the run as `tensorboard`; pointing directly at the event directory
names the run `.`, which some TensorBoard frontends fail to list.

The container uses bridge networking. From the client machine, forward the
container endpoint through the SSH host and then open
`http://127.0.0.1:6006`:

```bash
ssh -N -L 6006:172.17.0.12:6006 yimingwei_4090
```

The container IP may change after the container is recreated. Query the current
value with:

```bash
ssh yimingwei_4090 \
  "docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' lidar-wam-train"
```
