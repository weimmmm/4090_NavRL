#!/usr/bin/env bash
set -euo pipefail

# Pure PyTorch joint training.  Isaac Sim is not imported or launched here.
cd /workspace/NavRL/isaac-training/aliyun_lidar_wam/lidar_WAM

PY=${PY:-/opt/conda/bin/python}
ROOT=${ROOT:-/workspace/NavRL/isaac-training/aliyun_lidar_wam}
DATA=${DATA:-$ROOT/datasets/wam_static_seed00_09}
LATENTS=${LATENTS:-$PWD/outputs/wam_static_seed00_09/latents}
INDEX=${INDEX:-$PWD/outputs/wam_static_seed00_09/index}
ACTION=${ACTION:-$PWD/outputs/action_expert_action_only_v2/best.pt}
WORLD=${WORLD:-$PWD/outputs/world_direct_t3_v2/best.pt}
GPUS=${GPUS:-0,1,2,3,4,5,7}
NPROC=${NPROC:-7}
RUN_NAME=${RUN_NAME:-action_expert_joint_v2_future8500}
MEMORY_LIMIT_GIB=${MEMORY_LIMIT_GIB:-20.5}

export CUDA_VISIBLE_DEVICES=$GPUS
export LAGEN_ATTENTION_BACKEND=sdpa

selected=${JOINT_BATCH:-}
mkdir -p outputs/joint_batch_calibration
if [[ -z "$selected" ]]; then
    for batch in 256 320 384; do
        log="outputs/joint_batch_calibration/mb${batch}.log"
        if "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
            -m lidar_wam.runner.action_expert_joint train-joint \
            --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
            --action-checkpoint "$ACTION" --world-checkpoint "$WORLD" \
            --micro-batch-size "$batch" --grad-accumulation 1 --precision bf16 \
            --workers 8 --overfit --memory-smoke \
            --out outputs/joint_batch_calibration --run-name "mb${batch}" \
            >"$log" 2>&1; then
            peak=$(
                "$PY" -c 'import json,sys
rows=[]
for line in open(sys.argv[1]):
    try:
        value=json.loads(line)
    except Exception:
        continue
    if "peak_allocated_gib" in value:
        rows.append(float(value["peak_allocated_gib"]))
print(max(rows) if rows else 1e9)' "$log"
            )
            if "$PY" -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)' \
                    "$peak" "$MEMORY_LIMIT_GIB"; then
                selected=$batch
            fi
        fi
    done
fi

if [[ -z "$selected" ]]; then
    echo "No candidate micro-batch stayed below ${MEMORY_LIMIT_GIB} GiB" >&2
    exit 1
fi

echo "running 500-step shared-representation overfit check with batch $selected"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m lidar_wam.runner.action_expert_joint train-joint \
    --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
    --action-checkpoint "$ACTION" --world-checkpoint "$WORLD" \
    --steps 500 --micro-batch-size "$selected" --grad-accumulation 1 \
    --precision bf16 --eval-every 500 --eval-samples 4096 \
    --eval-batch-size 64 --flow-steps 10 --ddim-steps 20 \
    --workers 8 --log-every 20 --overfit --run-name "${RUN_NAME}_check" \
    >"outputs/joint_batch_calibration/overfit.log" 2>&1

"$PY" -c 'import json,sys
history=json.load(open(sys.argv[1]))
rows=[row for row in history if row.get("step", 0) >= 20]
if not rows or max(row.get("world_to_shared_grad_norm", 0.0) for row in rows) <= 1e-8:
    raise SystemExit("Future loss never reached the shared observation encoder")
if max(row.get("world_adapter_weight_norm", 0.0) for row in rows) <= 0.0:
    raise SystemExit("zero-initialized world adapter never left zero")' \
    "outputs/${RUN_NAME}_check_overfit/history.json"

mkdir -p "outputs/$RUN_NAME"
echo "selected joint micro-batch: $selected"
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m lidar_wam.runner.action_expert_joint train-joint \
    --dataset-root "$DATA" --latent-root "$LATENTS" --index-root "$INDEX" \
    --action-checkpoint "$ACTION" --world-checkpoint "$WORLD" \
    --steps 10000 --micro-batch-size "$selected" --grad-accumulation 1 \
    --precision bf16 --eval-every 500 --eval-samples 4096 \
    --eval-batch-size 64 --flow-steps 10 --ddim-steps 20 \
    --workers 8 --log-every 20 --run-name "$RUN_NAME" \
    >"outputs/$RUN_NAME/train.log" 2>&1
