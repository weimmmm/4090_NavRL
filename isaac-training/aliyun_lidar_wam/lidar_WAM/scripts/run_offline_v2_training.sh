#!/usr/bin/env bash
set -euo pipefail

# Pure PyTorch offline training.  Run this inside the persistent
# ``lidar-wam-train`` container built from the earlier LaGen inference image.
# No Isaac Sim module or simulator process is imported by this script.

cd /workspace/NavRL/isaac-training/aliyun_lidar_wam/lidar_WAM

PY=${PY:-/opt/conda/bin/python}
ROOT=/workspace/NavRL/isaac-training/aliyun_lidar_wam
DATA=${DATA:-$ROOT/datasets/wam_static_seed00_09}
LATENTS=${LATENTS:-$PWD/outputs/wam_static_seed00_09/latents}
INDEX=${INDEX:-$PWD/outputs/wam_static_seed00_09/index}
GPUS=${GPUS:-0,1,2,3,4,5,7}
NPROC=${NPROC:-7}
STAGE=${1:-all}

export CUDA_VISIBLE_DEVICES=$GPUS
export LAGEN_ATTENTION_BACKEND=sdpa

mkdir -p outputs/action_expert_action_only_v2 outputs/world_direct_t3_v2

train_action() {
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        -m lidar_wam.runner.action_expert_joint train-action \
        --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
        --steps 20000 --micro-batch-size 1680 --grad-accumulation 1 \
        --precision bf16 --eval-every 500 --eval-samples 4096 \
        --eval-batch-size 256 --flow-steps 10 --workers 8 --log-every 20 \
        > outputs/action_expert_action_only_v2/train.log 2>&1
}

train_world() {
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        -m lidar_wam.runner.world_direct_horizon train-v2 \
        --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
        --init-checkpoint outputs/action_expert_joint/best.pt \
        --steps 20000 --micro-batch-size 384 --grad-accumulation 1 \
        --precision bf16 --eval-every 500 --samples-per-seed 4096 \
        --eval-batch-size 64 --ddim-steps 20 --workers 8 --log-every 20 \
        > outputs/world_direct_t3_v2/train.log 2>&1
}

case "$STAGE" in
    action) train_action ;;
    world) train_world ;;
    all)
        train_action
        train_world
        ;;
    *)
        echo "usage: $0 [all|action|world]" >&2
        exit 2
        ;;
esac
