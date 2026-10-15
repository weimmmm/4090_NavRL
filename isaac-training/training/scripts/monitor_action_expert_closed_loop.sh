#!/usr/bin/env bash
set -u

project=/workspace/NavRL/isaac-training
run_name="${1:-super_action_expert_deterministic_v1}"
run_dir="$project/training/lidar_WAM/outputs/$run_name"
log="$project/training/lidar_WAM/outputs/${run_name}_closed_loop_monitor.log"
scene=training/data/maps/scene_350_04_map.pt

run_eval() {
    local checkpoint="$1"
    local output="$2"
    local bootstrap_speed="$3"
    local bootstrap_chunks="$4"
    mkdir -p "$output"
    cd "$project" || return 1
    ISAAC_ACTIVE_GPU=0 PYTHONPATH=training/lidar_WAM /isaac-sim/python.sh \
        training/scripts/wam_closed_loop_eval.py \
        device=cuda:0 headless=true env.num_envs=16 \
        env.max_episode_length=4000 env.scene_path="$scene" \
        wam_eval.checkpoint="$checkpoint" \
        wam_eval.bootstrap_forward_speed_mps="$bootstrap_speed" \
        wam_eval.bootstrap_action_chunks="$bootstrap_chunks" \
        wam_eval.safety_filter.enabled=false \
        wam_eval.metrics_path="$output/metrics.json" \
        wam_eval.trajectory_path="$output/trajectories.npz"
}

echo "monitor_start $(date --iso-8601=seconds)" >> "$log"
for step in $(seq 5000 500 20000); do
    checkpoint="$run_dir/step_${step}.pt"
    while [[ ! -s "$checkpoint" ]]; do
        sleep 20
    done
    base="$run_dir/closed_loop_step${step}_n16"
    if [[ ! -s "$base/metrics.json" ]]; then
        echo "eval_start step=$step bootstrap=0 $(date --iso-8601=seconds)" >> "$log"
        if ! run_eval "$checkpoint" "$base" 0.0 0 >> "$log" 2>&1; then
            echo "eval_failed step=$step bootstrap=0 $(date --iso-8601=seconds)" >> "$log"
            continue
        fi
    fi
    if /isaac-sim/python.sh "$project/training/scripts/check_birth_stuck.py" \
            "$base/trajectories.npz" >> "$log" 2>&1; then
        seeded="$run_dir/closed_loop_step${step}_bootstrap1m_n16"
        echo "eval_start step=$step bootstrap=1mps $(date --iso-8601=seconds)" >> "$log"
        run_eval "$checkpoint" "$seeded" 1.0 3 >> "$log" 2>&1 || \
            echo "eval_failed step=$step bootstrap=1mps $(date --iso-8601=seconds)" >> "$log"
    fi
    echo "eval_done step=$step $(date --iso-8601=seconds)" >> "$log"
done
echo "monitor_done $(date --iso-8601=seconds)" >> "$log"
