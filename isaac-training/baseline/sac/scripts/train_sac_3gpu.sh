#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_TRAINING_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
OMNIDRONES_DIR="${ISAAC_TRAINING_DIR}/third_party/OmniDrones"
export PYTHONPATH="${OMNIDRONES_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# NVIDIA Container Runtime already exposes exactly the selected physical GPUs.
# Re-applying host indices inside this Isaac image hides every CUDA device.
if [[ ! -f /.dockerenv ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
fi

if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
elif [[ -x /isaac-sim/python.sh ]]; then
    PYTHON_CMD=(/isaac-sim/python.sh)
else
    echo "No Python interpreter found; activate NavRL or run inside the Isaac Sim container." >&2
    exit 1
fi

exec "${PYTHON_CMD[@]}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=3 \
    "${SCRIPT_DIR}/train_sac.py" \
    distributed.enabled=true \
    distributed.expected_world_size=3 \
    "$@"
