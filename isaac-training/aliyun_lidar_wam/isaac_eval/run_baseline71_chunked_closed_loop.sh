#!/usr/bin/env bash
set -euo pipefail

# Sequential 71%-baseline fine-tuning/evaluation loop.  Each training chunk
# finishes before Isaac Sim is started on GPU0, so the same GPU is never shared
# by training and closed-loop evaluation.
HOST=/home/yimingwei/NavRL
ROOT=/workspace/NavRL
CODE="$ROOT/.runtime/baseline71/isaac-training/aliyun_lidar_wam/lidar_WAM"
DATA="$ROOT/isaac-training/aliyun_lidar_wam/datasets/wam_40x40_obs200_350_settled_4m"
LATENTS="$ROOT/isaac-training/aliyun_lidar_wam/lidar_WAM/outputs/wam_40x40_obs200_350_settled_4m/latents"
INDEX="$ROOT/isaac-training/aliyun_lidar_wam/lidar_WAM/outputs/wam_40x40_obs200_350_settled_4m/video_index_h3n1"
WORLD="$ROOT/isaac-training/aliyun_lidar_wam/lidar_WAM/outputs/lidar_video_dit_history3_next1_50k_v2metrics/best.pt"
START="$ROOT/isaac-training/aliyun_lidar_wam/lidar_WAM/outputs/history_world_action_goal_spatial_baseline71_newdata_2k/latest.pt"
RUN="$ROOT/isaac-training/aliyun_lidar_wam/lidar_WAM/outputs/history_world_action_goal_spatial_baseline71_newdata_30k"
EVAL_ROOT="$ROOT/isaac-training/aliyun_lidar_wam/isaac_eval/results/baseline71_newdata_30k_every400"
ENV="$ROOT/isaac-training/aliyun_lidar_wam/isaac_eval/environments/static350_seed18_n256.pt"
HOST_RUN="$HOST/isaac-training/aliyun_lidar_wam/lidar_WAM/outputs/history_world_action_goal_spatial_baseline71_newdata_30k"
HOST_EVAL_ROOT="$HOST/isaac-training/aliyun_lidar_wam/isaac_eval/results/baseline71_newdata_30k_every400"
TARGET=30000
CHUNK=400

mkdir -p "$HOST_RUN/closed_loop_queue" "$HOST_EVAL_ROOT"
step=$(docker exec lidar-wam-train python - "$START" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(p.get("step", 0)))
PY
)
start_step="$step"
best_score=-1
best_step=0
echo "starting_global_step=$step target=$TARGET"

while [ "$step" -lt "$TARGET" ]; do
    local_steps=$CHUNK
    remaining=$((TARGET - step))
    if [ "$remaining" -lt "$local_steps" ]; then
        local_steps=$remaining
    fi
    next_step=$((step + local_steps))
    ckpt="$RUN/latest.pt"
    if [ "$step" -eq "$start_step" ]; then
        ckpt="$START"
    fi
    echo "[train] global $step -> $next_step"
    docker exec -e CUDA_VISIBLE_DEVICES=0 lidar-wam-train bash -lc \
      "cd $CODE && /opt/conda/bin/python -u -m lidar_wam.runner.history_action_dit train \
      --dataset-root $DATA --latent-root $LATENTS --action-index-root $INDEX \
      --world-checkpoint $WORLD --resume-checkpoint $ckpt --out $ROOT/isaac-training/aliyun_lidar_wam/lidar_WAM/outputs \
      --run-name history_world_action_goal_spatial_baseline71_newdata_30k \
      --steps $local_steps --micro-batch-size 48 --eval-batch-size 16 --workers 2 \
      --width 512 --depth 8 --heads 8 --mlp-ratio 4.0 --past-horizon 30 --shared-world-depth 6 \
      --world-lr 1e-6 --action-lr 2e-5 --weight-decay 1e-2 --grad-clip 1.0 \
      --world-weight 1.0 --action-weight 1.0 --precision bf16 \
      --eval-samples 512 --condition-stat-samples 20000 --action-sample-steps 10 \
      --action-sample-windows 128 --eval-every 200 --log-every 20 --seed 42" 

    actual_step=$(docker exec lidar-wam-train python - "$RUN/latest.pt" <<'PY'
import sys, torch
print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", 0)))
PY
)
    if [ "$actual_step" -ne "$next_step" ]; then
        echo "checkpoint step mismatch: expected $next_step got $actual_step" >&2
        exit 2
    fi
    snapshot_host="$HOST_RUN/closed_loop_queue/step_$(printf '%06d' "$actual_step").pt"
    snapshot="$RUN/closed_loop_queue/step_$(printf '%06d' "$actual_step").pt"
    cp "$HOST_RUN/latest.pt" "$snapshot_host"
    eval_dir_host="$HOST_EVAL_ROOT/step_$(printf '%06d' "$actual_step")"
    eval_dir="$EVAL_ROOT/step_$(printf '%06d' "$actual_step")"
    mkdir -p "$eval_dir_host"
    echo "[closed_loop] step=$actual_step gpu=0"
    docker exec -e CUDA_VISIBLE_DEVICES=0 navrl-train bash -lc \
      "cd $ROOT/isaac-training/aliyun_lidar_wam && /isaac-sim/python.sh isaac_eval/evaluate.py \
      --checkpoint $snapshot --output $eval_dir --environment $ENV --device cuda:0 \
      --route-limit 128 --flow-steps 10"
    result="$eval_dir/action_expert_step_$(printf '%06d' "$actual_step")_seed18_n128_routes000_policy42.json"
    score=$(python - "$result" <<'PY'
import json, sys
print(float(json.load(open(sys.argv[1]))["summary"]["success_rate"]))
PY
)
    echo "$actual_step,$score" >> "$HOST_EVAL_ROOT/closed_loop_scores.csv"
    if python - "$score" "$best_score" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
    then
        best_score="$score"
        best_step="$actual_step"
        cp "$snapshot_host" "$HOST_RUN/closed_loop_best.pt"
        printf '{"step":%d,"success_rate":%.9f,"checkpoint":"%s"}\n' \
          "$best_step" "$best_score" "$HOST_RUN/closed_loop_best.pt" > "$HOST_RUN/closed_loop_best.json"
    fi
    step="$actual_step"
done

echo "completed_global_step=$step best_step=$best_step best_success_rate=$best_score"
