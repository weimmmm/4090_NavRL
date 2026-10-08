#!/usr/bin/env bash
set -u

# This script is executed inside one of the root Isaac collection containers.
# Usage: launch_collection_worker.sh <root> <gpu> <condition> <shard-base> <primary>
# condition is 200, 350, or auto (GPU 4 balancing worker).

ROOT=${1:?dataset root is required}
GPU=${2:?container-local GPU id is required}
CONDITION=${3:?condition is required}
BASE=${4:?shard base is required}
PRIMARY=${5:-0}
PY=/isaac-sim/python.sh
CHECKPOINT=/workspace/NavRL/isaac-training/aliyun_lidar_wam/isaac_eval/checkpoints/ppo_dataset_collector.pt
cd /workspace/NavRL/isaac-training/aliyun_lidar_wam || exit 1
mkdir -p "$ROOT/logs"

sum_frames() {
    local obstacles=$1
    "$PY" -c 'import glob,json,sys; print(sum(int(json.load(open(p))["summary"]["frames"]) for p in glob.glob(sys.argv[1]+"/train/obs_"+sys.argv[2]+"/shard_*/summary.json")))' \
        "$ROOT" "$(printf '%04d' "$obstacles")"
}

run_one() {
    local obstacles=$1 shard=$2 seed=$3 route=$4 keep=$5
    local keep_arg=()
    if [ "$keep" -eq 1 ]; then
        keep_arg=(--keep-environment)
    fi
    "$PY" -u -m isaac_eval.collect_dataset \
        --seed "$seed" --route-seed "$route" --shard-id "$shard" \
        --static-obstacles "$obstacles" --opposite-fraction 0.5 \
        --num-envs 512 --max-steps 2500 --sample-interval 10 \
        --goal-radius 0.5 --settle-speed 0.1 --settle-steps 10 \
        --split train --device "cuda:$GPU" --output-root "$ROOT" \
        --checkpoint "$CHECKPOINT" "${keep_arg[@]}"
}

for ((i=0; i<80; i++)); do
    if [ "$CONDITION" = auto ]; then
        total200=$(sum_frames 200)
        total350=$(sum_frames 350)
        if [ "$total200" -lt 2000000 ] && { [ "$total350" -ge 2000000 ] || [ $((i % 2)) -eq 0 ]; }; then
            obstacles=200
            shard=$((BASE + i * 5))
            seed=$((4000000 + shard * 100))
            route=$((6000000 + shard * 100))
        else
            obstacles=350
            shard=$((BASE + i * 5 + 1))
            seed=$((4100000 + shard * 100))
            route=$((6100000 + shard * 100))
        fi
        [ "$obstacles" -eq 200 ] && [ "$total200" -ge 2000000 ] && continue
        [ "$obstacles" -eq 350 ] && [ "$total350" -ge 2000000 ] && continue
    else
        obstacles=$CONDITION
        total=$(sum_frames "$obstacles")
        [ "$total" -ge 2000000 ] && exit 0
        shard=$((BASE + i * 5))
        if [ "$obstacles" -eq 200 ]; then
            seed=$((4000000 + shard * 100))
            route=$((6000000 + shard * 100))
        else
            seed=$((4100000 + shard * 100))
            route=$((6100000 + shard * 100))
        fi
    fi
    keep=0
    [ "$PRIMARY" -eq 1 ] && [ "$i" -eq 0 ] && keep=1
    log="$ROOT/logs/worker_gpu${GPU}_obs${obstacles}_shard${shard}.log"
    run_one "$obstacles" "$shard" "$seed" "$route" "$keep" >"$log" 2>&1 || exit 1
done
