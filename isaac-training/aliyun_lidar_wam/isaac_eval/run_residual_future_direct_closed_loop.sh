#!/usr/bin/env bash
set -euo pipefail

cd /home/yimingwei/NavRL/isaac-training/aliyun_lidar_wam
run_name="action_expert_residual_future_t1_from89_step16000_restored_direct_b512"
result_dir="isaac_eval/results/${run_name}"
mkdir -p "${result_dir}"

# Physical GPU1 is not a training device. Wait for other jobs to release it;
# never terminate unrelated processes or move evaluation onto a training GPU.
exec python3 -u isaac_eval/watch_joint_checkpoints_gpu_flexible.py \
  --run-dir "lidar_WAM/outputs/${run_name}" \
  --output "${result_dir}" \
  --environment datasets/wam_40x40_obs200_350_settled_4m/environments/map40_obs0350.pt \
  --gpus 1 \
  --routes 128 \
  --flow-steps 10 \
  --min-step 1000 \
  --final-step 6000 \
  --poll-seconds 20 \
  --max-used-mib 2000
