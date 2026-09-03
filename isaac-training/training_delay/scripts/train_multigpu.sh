#!/usr/bin/env bash
set -euo pipefail

# GPU_LIST contains GPU indices visible inside the current container. For a
# container created with --gpus 'device=5,6', use GPU_LIST=0,1 (or omit it).
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_GPUS="${#GPU_IDS[@]}"
MASTER_PORT="${MASTER_PORT:-29501}"
RDZV_ID="${RDZV_ID:-navrl-$$}"

if (( NUM_GPUS < 2 )); then
    echo "GPU_LIST must contain at least two GPU indices, got: ${GPU_LIST}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
exec /isaac-sim/python.sh -m torch.distributed.run \
    --rdzv_backend=c10d \
    --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
    --rdzv_id="${RDZV_ID}" \
    --nproc_per_node="${NUM_GPUS}" \
    "$(dirname "$0")/train.py" \
    "$@"
