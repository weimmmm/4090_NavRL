#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NavRL/isaac-training/aliyun_lidar_wam/lidar_WAM

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

# Keep the pre-restoration experiment and its checkpoints intact.
run_name="action_expert_residual_future_t1_from89_step16000_restored_direct_b512"
mkdir -p "outputs/${run_name}"

exec /opt/conda/bin/torchrun --standalone --nproc_per_node=4 \
  -m lidar_wam.runner.action_expert_future \
  --dataset-root /workspace/NavRL/isaac-training/aliyun_lidar_wam/datasets/wam_40x40_obs200_350_settled_4m \
  --latent-root outputs/wam_40x40_obs200_350_settled_4m/latents \
  --index-root outputs/wam_40x40_obs200_350_settled_4m/index_past30 \
  --action-checkpoint outputs/action_expert_settled_4m/eval_checkpoints/step_16000.pt \
  --future-checkpoint outputs/lidar_video_dit_residual_antithetic_retrain/best_nonempty_chamfer_m2.pt \
  --out outputs \
  --run-name "${run_name}" \
  --steps 6000 \
  --batch-size 512 \
  --eval-batch-size 32 \
  --workers 4 \
  --adapter-lr 3e-6 \
  --attention-lr 1e-5 \
  --future-steps 20 \
  --future-adapter-mode direct \
  --future-horizon 1 \
  --action-steps 10 \
  --eval-samples 1024 \
  --eval-every 500 \
  --closed-loop-every 500 \
  --closed-loop-start 1000 \
  --log-every 10 \
  --precision bf16 \
  --seed 42 \
  2>&1 | tee "outputs/${run_name}/train.log"
