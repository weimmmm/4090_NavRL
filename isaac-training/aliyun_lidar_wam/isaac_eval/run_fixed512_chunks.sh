#!/usr/bin/env bash
set -euo pipefail

# Evaluate one policy on all 512 saved routes without the unstable 512-env
# simultaneous physics configuration. Run inside navrl-train.
cd /workspace/NavRL/isaac-training/aliyun_lidar_wam
PY=${PY:-/isaac-sim/python.sh}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to an Action or joint policy checkpoint}
SEED=${SEED:-8}
POLICY_SEED=${POLICY_SEED:-42}
ENVIRONMENT=${ENVIRONMENT:-datasets/wam_static_seed00_09/environments/static350_seed$(printf '%02d' "$SEED")_n512.pt}
OUT=${OUT:-isaac_eval/results/fixed512_seed${SEED}}

files=()
for start in 0 128 256 384; do
    destination="$OUT/chunk_${start}"
    "$PY" -m isaac_eval.evaluate \
        --checkpoint "$CHECKPOINT" --environment "$ENVIRONMENT" \
        --route-start "$start" --route-limit 128 --episodes 128 \
        --policy-seed "$POLICY_SEED" \
        --output "$destination"
    files+=("$destination"/*.json)
done

"$PY" -m isaac_eval.aggregate_route_chunks "${files[@]}" \
    --output "$OUT/aggregate.json" --expected-routes 512
