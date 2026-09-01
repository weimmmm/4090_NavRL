#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/workspace/NavRL/isaac-training"
LOG_DIR="${PROJECT_DIR}/logs"
WANDB_RUN_MODE="${WANDB_MODE:-online}"
export OMNI_DISABLE_NUCLEUS_CHECK="${OMNI_DISABLE_NUCLEUS_CHECK:-1}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

if pgrep -af 'training_delay/scripts/train.py' >/dev/null || \
   pgrep -af 'training/scripts/train.py' >/dev/null; then
    echo "A NavRL training process is already running:"
    pgrep -af 'training_delay/scripts/train.py|training/scripts/train.py' || true
    exit 1
fi

echo "Starting training_delay on GPU 7..."
nohup /isaac-sim/python.sh training_delay/scripts/train.py \
  device=cuda:7 \
  enable_eval=false \
  record_eval_video=false \
  "wandb.mode=${WANDB_RUN_MODE}" \
  > "${LOG_DIR}/train_delay_gpu7.log" 2>&1 &
echo "training_delay PID: $!"

echo "Starting training on GPU 5..."
nohup /isaac-sim/python.sh training/scripts/train.py \
  device=cuda:5 \
  enable_eval=false \
  record_eval_video=false \
  "wandb.mode=${WANDB_RUN_MODE}" \
  > "${LOG_DIR}/train_gpu5.log" 2>&1 &
echo "training PID: $!"

echo "Both training jobs started. Logs: ${LOG_DIR}"
