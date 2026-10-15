"""Round-based student rollout + four-GPU first-action SFT pilot.

Run ON THE 4090 HOST: python3 .../run_action_dagger_sft.py
Every subprocess is checked. Failure stops the pipeline; existing checkpoints
are never overwritten. GPUs 1/2/3/7 train, GPU0 simulates/evaluates. No external
GPU process is stopped. DDP GPU memory is bounded by the partial-freeze design.
"""
from pathlib import Path
import subprocess
import json
import argparse
import os
import shutil

ROOT = Path("/workspace/NavRL/isaac-training")
HOST = Path("/home/yimingwei/NavRL/isaac-training")
PROJECT = ROOT/"training/lidar_WAM"
TRAIN = "lidar-wam-train"
SIM = "navrl-eval-future3-gpu0-20260929-v6"
PY = "/opt/conda/bin/python3"
DEPS = str(ROOT/"aliyun_lidar_wam/lidar_WAM/third_party/diffusers/src")
RUNNER = "lidar_wam.runner.action_expert_closed_loop_sft"


def run(container, command, log, environment):
    args = ["docker", "exec", "-w", str(ROOT)]
    for key, value in environment.items():
        args += ["-e", f"{key}={value}"]
    args += [container, *map(str, command)]
    print("[pipeline]", log.name, flush=True)
    with log.open("w") as stream:
        subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="action_closed_loop_sft_first_step_20261014")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--updates-per-round", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    out = PROJECT/"outputs"/args.run_name
    host_out = HOST/"training/lidar_WAM/outputs"/args.run_name
    host_out.mkdir(parents=True, exist_ok=True)
    checkpoint = PROJECT/"outputs/action_dit_final_lee_ego_history1_5m_v3/best_closed_loop.pt"
    cache = out/"offline_cache.pt"
    teacher = PROJECT/"outputs/privileged_ppo_checkpoint_27000.pt"
    train_env = {"CUDA_VISIBLE_DEVICES": "1,2,3,7", "OMP_NUM_THREADS": "4",
                 "PYTHONPATH": f"{PROJECT}:{DEPS}", "PYTHONUNBUFFERED": "1"}
    sim_env = {"ISAAC_ACTIVE_GPU": "0", "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": f"{PROJECT}:{DEPS}",
               "NAVRL_ASSETS_ROOT": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0"}
    state = {"stage": "offline_cache", "round": 0, "training_gpus": [1, 2, 3, 7], "simulation_gpu": 0}

    def status(stage, round_index):
        state.update(stage=stage, round=round_index)
        (host_out/"status.json").write_text(json.dumps(state, indent=2)+"\n")

    if not (host_out/"offline_cache.pt").exists():
        run(TRAIN, [PY, "-m", RUNNER, "prepare", "--checkpoint", checkpoint,
                    "--cache", cache, "--data-root", ROOT/"training/datasets/wam_40x40_obs200_350_settled_5m/train",
                    "--vae-checkpoint", PROJECT/"outputs/vae_5m_finetune_v1/best/diffusion_pytorch_model.safetensors"],
            host_out/"cache.log", {**train_env, "CUDA_VISIBLE_DEVICES": "0"})
    online = []
    best_sr = .8125  # Reproduced baseline: do not call a worse SFT checkpoint "best".
    if not (host_out/"best_closed_loop.pt").exists():
        shutil.copyfile(HOST/"training/lidar_WAM/outputs/action_dit_final_lee_ego_history1_5m_v3/best_closed_loop.pt",
                        host_out/"best_closed_loop.pt")
    for index in range(1, args.rounds+1):
        status("student_rollout_ppo_first_step_labels", index)
        buffer = out/f"online_round_{index}.pt"
        online.append(buffer)
        common = ["headless=true", "device=cuda:0", "seed=20261014", "env.num_envs=128",
                  "env.max_episode_length=2500", "env_dyn.num_obstacles=0",
                  f"wam_eval.checkpoint={checkpoint}", "wam_eval.bootstrap_forward_speed_mps=1.0",
                  "wam_eval.bootstrap_action_chunks=1", "wam_eval.goal_stop_adapter.enabled=true",
                  "wam_eval.goal_stop_adapter.enter_distance_m=0.5",
                  "wam_eval.goal_stop_adapter.enter_speed_mps=0.2", "wam_eval.safety_filter.enabled=false"]
        obstacles = 200 if index % 2 == 1 else 350
        run(SIM, ["/opt/navrl/bin/isaac-python", "training/scripts/collect_action_dagger.py", *common,
                  f"env.num_obstacles={obstacles}", f"env.runtime_scene_id=dagger_round_{index}",
                  f"env.map_seed={8100000+index}", f"env.route_seed={9100000+index}",
                  f"+dagger.teacher_checkpoint={teacher}", f"+dagger.output={buffer}"],
            host_out/f"collect_round_{index}.log", sim_env)
        status("four_gpu_sft", index)
        run(TRAIN, [PY, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                    "-m", RUNNER, "train", "--checkpoint", checkpoint, "--cache", cache,
                    "--online", *online, "--out", out, "--steps", args.updates_per_round,
                    "--batch-size", args.batch_size, "--lr", "0.000005", "--online-weight", ".5"],
            host_out/f"train_round_{index}.log", train_env)
        step = args.updates_per_round*index
        checkpoint = out/f"sft_step_{step}.pt"
        status("heldout_closed_loop_eval", index)
        evaluation = out/f"closed_loop_step_{step}"
        (host_out/f"closed_loop_step_{step}").mkdir(exist_ok=True)
        eval_common = [value for value in common if not value.startswith("wam_eval.checkpoint=")]
        run(SIM, ["/opt/navrl/bin/isaac-python", "training/scripts/wam_closed_loop_eval.py", *eval_common,
                  f"wam_eval.checkpoint={checkpoint}", "env.num_obstacles=350",
                  "env.scene_path=training/datasets/wam_40x40_obs200_350_settled_5m/environments/map40_obs0350.pt",
                  f"wam_eval.metrics_path={evaluation}/metrics.json",
                  f"wam_eval.trajectory_path={evaluation}/trajectories.npz"],
            host_out/f"eval_step_{step}.log", sim_env)
        metrics = json.loads((host_out/f"closed_loop_step_{step}/metrics.json").read_text())
        sr = float(metrics["rates"]["sr"])
        if sr > best_sr:
            best_sr = sr
            shutil.copyfile(host_out/f"sft_step_{step}.pt", host_out/"best_closed_loop.pt")
        # Record closed-loop metrics into the SAME TensorBoard run.
        run(TRAIN, [PY, "-c", "import json,sys; from torch.utils.tensorboard import SummaryWriter; "
                    "m=json.load(open(sys.argv[1])); w=SummaryWriter(sys.argv[2]); "
                    "[w.add_scalar('closed_loop/'+k,v,int(sys.argv[3])) for k,v in m.get('rates',{}).items()]; "
                    "[w.add_scalar('closed_loop/count_'+k,v,int(sys.argv[3])) for k,v in m.get('counts',{}).items()]; w.close()",
                    evaluation/"metrics.json", out/"tensorboard", step],
            host_out/f"tensorboard_eval_{step}.log", train_env)
        print(f"[pipeline] step={step} sr={sr} counts={metrics.get('counts')}", flush=True)
    status("complete", args.rounds)


if __name__ == "__main__":
    main()
