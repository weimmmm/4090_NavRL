#!/usr/bin/env bash
set -euo pipefail

project=/workspace/NavRL/isaac-training/aliyun_lidar_wam/lidar_WAM
dataset=/workspace/NavRL/isaac-training/aliyun_lidar_wam/datasets/wam_40x40_obs200_350_settled_5m
cache="$project/outputs/wam_40x40_obs200_350_settled_5m"
run_name=lidar_video_dit_residual_settled5m_scratch_b128
run_dir="$project/outputs/$run_name"
mode="${1:-train}"

cd "$project"
export PYTHONPATH="$project:$project/third_party/diffusers/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
mkdir -p "$run_dir"

case "$mode" in
  train)
    # An existing run must never be silently overwritten or resumed.
    if [[ -e "$run_dir/config.json" || -e "$run_dir/latest.pt" ]]; then
      echo "Run already exists: $run_dir" >&2
      exit 1
    fi
    export CUDA_VISIBLE_DEVICES=0,1,3,7
    exec torchrun --standalone --nproc_per_node=4 \
      -m lidar_wam.runner.lidar_video_dit train \
      --dataset-root "$dataset" --latent-root "$cache/latents" \
      --base-index-root "$cache/index_trainval_1pct_v1" \
      --video-index-root "$cache/video_index_h3n1" \
      --out "$project/outputs" --run-name "$run_name" \
      --steps 50000 --micro-batch-size 128 --eval-batch-size 256 \
      --workers 4 --width 512 --depth 8 --heads 8 --mlp-ratio 4 \
      --lr 1e-4 --min-lr 2e-7 --warmup-steps 200 \
      --prediction-target normalized_residual \
      --selection-metric generation_loss --precision bf16 \
      --flow-steps 20 --flow-ensemble-size 1 \
      --flow-sampling-mode independent --eval-samples 20480 \
      --eval-every 500 --checkpoint-every 500 --log-every 20 \
      --external-validation > "$run_dir/train.log" 2>&1
    ;;
  validate)
    export CUDA_VISIBLE_DEVICES=2
    exec python3 -u scripts/validate_lidar_video_dit_external.py \
      --dataset-root "$dataset" --latent-root "$cache/latents" \
      --video-index-root "$cache/video_index_h3n1" \
      --run-dir "$run_dir" --device cuda:0 --eval-samples 20480 \
      --eval-batch-size 256 --workers 4 --precision bf16 \
      --flow-steps 20 --flow-ensemble-size 1 \
      --flow-sampling-mode independent --poll-seconds 15 \
      --seed 4200 --stop-step 50000 > "$run_dir/validation.log" 2>&1
    ;;
  resume)
    if [[ ! -e "$run_dir/latest.pt" ]]; then
      echo "Resume checkpoint missing: $run_dir/latest.pt" >&2
      exit 1
    fi
    if [[ -e "$run_dir/config_50000.json" ]]; then
      echo "Continuation already configured: $run_dir" >&2
      exit 1
    fi
    cp "$run_dir/config.json" "$run_dir/config_50000.json"
    cp "$run_dir/latest.pt" "$run_dir/step_50000_before_extension.pt"
    export CUDA_VISIBLE_DEVICES=0,1,3,7
    exec torchrun --standalone --nproc_per_node=4 \
      -m lidar_wam.runner.lidar_video_dit train \
      --dataset-root "$dataset" --latent-root "$cache/latents" \
      --base-index-root "$cache/index_trainval_1pct_v1" \
      --video-index-root "$cache/video_index_h3n1" \
      --out "$project/outputs" --run-name "$run_name" \
      --steps 100000 --micro-batch-size 128 --eval-batch-size 256 \
      --workers 4 --width 512 --depth 8 --heads 8 --mlp-ratio 4 \
      --lr 5e-6 --min-lr 2e-7 --warmup-steps 500 \
      --prediction-target normalized_residual \
      --selection-metric generation_loss --precision bf16 \
      --flow-steps 20 --flow-ensemble-size 1 \
      --flow-sampling-mode independent --eval-samples 20480 \
      --eval-every 500 --checkpoint-every 500 --log-every 20 \
      --init-checkpoint "$run_dir/step_50000_before_extension.pt" \
      --resume-optimizer --restart-lr-schedule --external-validation \
      > "$run_dir/train_50000_to_100000.log" 2>&1
    ;;
  validate-resume)
    export CUDA_VISIBLE_DEVICES=2
    exec python3 -u scripts/validate_lidar_video_dit_external.py \
      --dataset-root "$dataset" --latent-root "$cache/latents" \
      --video-index-root "$cache/video_index_h3n1" \
      --run-dir "$run_dir" --device cuda:0 --eval-samples 20480 \
      --eval-batch-size 256 --workers 4 --precision bf16 \
      --flow-steps 20 --flow-ensemble-size 1 \
      --flow-sampling-mode independent --poll-seconds 15 \
      --seed 4200 --stop-step 100000 \
      > "$run_dir/validation_50000_to_100000.log" 2>&1
    ;;
  tensorboard)
    exec python3 scripts/run_tensorboard_compatible.py \
      --logdir "$run_dir/tensorboard" \
      --host 0.0.0.0 --port 6006 --reload_interval 30 \
      > "$run_dir/tensorboard.log" 2>&1
    ;;
  *)
    echo "Usage: $0 {train|validate|resume|validate-resume|tensorboard}" >&2
    exit 2
    ;;
esac
