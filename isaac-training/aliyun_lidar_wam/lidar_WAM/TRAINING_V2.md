# WAM v2 goal-frame offline retraining

This workflow trains only from the self-contained
`datasets/wam_static_seed00_09` dataset.  It does not import any of the sibling
`training_lidar` or `training_legan` trees.

Training is fully offline.  Isaac Sim is used only after checkpoints have been
selected, for fixed-route closed-loop evaluation and an optional DAgger stage.

## Invariants

- Observation row `i` uses row `i` as its previous ten actions.
- Its 30 future labels are token successors `i+1`, `i+2`, and `i+3`.
- The world target is the LiDAR at successor `i+3`.
- Goal, velocity, angular velocity, gravity, and normalized PPO action all use
  the fixed start-to-target goal frame.
- A policy emits 30 normalized actions, executes 10, and then replans.
- New checkpoints carry coordinate semantics and dataset/VAE hashes.  The
  deployment code rejects legacy checkpoints unless
  `--allow-legacy-body` is given.

## Runtime and paths

Training uses the same normal CUDA/PyTorch interpreter as the earlier LaGen
inference image, not the Isaac Sim Python runtime:

```bash
docker start lidar-wam-train
docker exec -it lidar-wam-train bash
```

Inside `lidar-wam-train`:

```bash
cd /workspace/NavRL/isaac-training/aliyun_lidar_wam/lidar_WAM
PY=/opt/conda/bin/python
ROOT=/workspace/NavRL/isaac-training/aliyun_lidar_wam
DATA=$ROOT/datasets/wam_static_seed00_09
LATENTS=$PWD/outputs/wam_static_seed00_09/latents
INDEX=$PWD/outputs/wam_static_seed00_09/index
GPUS=0,1,2,3,4,5,7
```

GPU 6 is intentionally omitted while it is occupied by another process.

After the smoke gates pass, the exact 20 GiB-per-rank production sequence can
be launched with:

```bash
scripts/run_offline_v2_training.sh all
```

It runs Action Expert first and automatically starts future-LiDAR training
afterward.  Stage-specific invocations are `action` and `world`.

## 1. Data inspection and frozen VAE cache

```bash
$PY -m lidar_wam.runner.prepare_v2 inspect \
  --dataset-root "$DATA" --index-root "$INDEX"

CUDA_VISIBLE_DEVICES=0 $PY -m lidar_wam.runner.prepare_v2 cache-latents \
  --dataset-root "$DATA" --index-root "$INDEX" --output "$LATENTS" \
  --batch-size 128 --resume
```

The inspection must report the unfiltered counts `480792 / 62792 / 61327`,
action conversion error at most `1e-5`, and offline/online feature error at most
`1e-6`.  The cache command stops unless VAE range MAE is at most `0.10` and mask
F1 is at least `0.70`.

## 2. Action-only overfit and full training

```bash
CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run \
  --standalone --nproc_per_node=7 \
  -m lidar_wam.runner.action_expert_joint train-action \
  --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
  --steps 1000 --micro-batch-size 1680 --grad-accumulation 1 \
  --precision bf16 --eval-every 100 --eval-batch-size 256 \
  --workers 8 --overfit

CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run \
  --standalone --nproc_per_node=7 \
  -m lidar_wam.runner.action_expert_joint train-action \
  --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
  --steps 20000 --micro-batch-size 1680 --grad-accumulation 1 \
  --precision bf16 --eval-every 500 --eval-batch-size 256 \
  --workers 8
```

Each validation keeps the best four compact candidates in
`outputs/action_expert_action_only_v2/`.  Offline selection uses 50% overall,
25% low-clearance, and 25% turning MAE.  On the 24 GiB RTX 4090 host,
`micro-batch-size=1680` measured 20.56 GiB peak PyTorch allocation per training
rank (about 22.8 GiB including the CUDA context and reserved blocks).  Do not
reuse this batch size for the larger world or joint models.

## 3. Future LiDAR generation and joint Action Expert

The existing full joint checkpoint supplies the old world UNet warm start:

```bash
cd "$ROOT/lidar_WAM"
CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run \
  --standalone --nproc_per_node=7 \
  -m lidar_wam.runner.world_direct_horizon train-v2 \
  --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
  --init-checkpoint outputs/action_expert_joint/best.pt \
  --steps 20000 --micro-batch-size 384 --grad-accumulation 1 \
  --precision bf16 --eval-every 500 --eval-batch-size 64 \
  --workers 8
```

The seven-rank DDP probe measured 20.53 GiB peak PyTorch allocation per GPU at
`micro-batch-size=384`.  A 448 batch crossed 22 GiB during auxiliary VAE
decoding and is intentionally rejected as unsafe.

`world_gate.json` must show at least 10% improvement over raw-frame copy and at
least 5% degradation after shuffling actions. Joint training refuses a world
checkpoint that did not pass this causal gate.

```bash
scripts/run_joint_future_action_v2.sh
```

The launcher probes joint micro-batches 256/320/384, selects the largest one
below 20.5 GiB allocated per rank, runs the 500-step shared-gradient check, and
then starts 10,000 steps. The world UNet has zero LR through step 500 while the
world weight ramps 0.05→0.15; by step 1500 it reaches weight 0.25 and LR 1e-6.

Checkpoint selection is
`0.40*first10 + 0.25*low_clearance_first10 +
0.25*turning_first10 + 0.10*overall`. A candidate is retained only when all
three first-10 slices improve at least 10% over repeat-last and squared Chamfer
beats copy by 10% with at least 5% shuffled-action degradation. Each retained
step has an Action-only compact checkpoint and a full joint checkpoint for
offline validation and reproducible initialization.

For the official 512-route Isaac evaluation, run four slices of the original
environment with `--route-start 0/128/256/384 --route-limit 128`. Route IDs are
preserved, so action and world noise do not change with chunking. Merge them
with `python -m isaac_eval.aggregate_route_chunks`. Seed 9 remains untouched
until seed 8 reaches the final acceptance threshold.
