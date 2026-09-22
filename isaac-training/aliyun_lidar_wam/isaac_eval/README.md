# Isaac Sim closed-loop Action Expert evaluation

This directory evaluates the trained LiDAR Action Expert as a closed-loop
navigation policy. Source code does not import `training_legan`,
`training_lidar`, or another project directory outside `aliyun_lidar_wam`.
The required Navigation environment, configuration, OmniDrones, Orbit,
TorchRL, and TensorDict sources are vendored below this directory. Isaac Sim,
the NVIDIA driver, CUDA, and ordinary Python packages remain runtime
dependencies supplied by the `navrl-train` container.

The policy observes the live raw `108 x 18` LiDAR grid, constructs the exact
two-channel `2 x 108 x 20` Circular-VAE input used for training, and generates
30 normalized PPO actions. It executes the first 10 actions, observes again,
and replans. Its normalized output is converted to the original goal-frame
velocity range `[-2, 2] m/s`, rotated into the world frame, and passed to the
same Lee velocity controller used during data collection.

## 1. Enter the existing Isaac Sim 2023 container

On the host:

```bash
docker start -ai navrl-train
```

The project is mounted at `/workspace/NavRL` in the container.

## 2. Check the independent project layout

Inside the container:

```bash
cd /workspace/NavRL/isaac-training/aliyun_lidar_wam
/isaac-sim/python.sh -m isaac_eval.check_layout
```

If the container does not yet have the lightweight model-loading packages,
install them without replacing Isaac's PyTorch, CUDA, or NumPy builds:

```bash
bash isaac_eval/install_dependencies.sh
```

## 3. Use the original training checkpoint

Evaluation loads `lidar_WAM/outputs/action_expert_joint/best.pt` directly. No
compact checkpoint export is required. The original checkpoint is read on CPU,
then its trained LiDAR observation encoder and Action Expert weights are loaded
onto the evaluation device together with the original normalization statistics.

## 4. Create and reuse a fixed evaluation environment

The checked-in default environment contains the terrain recipe plus fixed start
and goal positions for 256 drones. It was created inside the Isaac container:

```bash
/isaac-sim/python.sh -m isaac_eval.create_environment \
  --output isaac_eval/environments/static350_seed18_n256.pt \
  --num-envs 256 --terrain-seed 18 --route-seed 18 \
  --static-obstacles 350 --max-steps 2200
```

Run evaluation with that environment and the original training checkpoint:

```bash
/isaac-sim/python.sh -m isaac_eval.evaluate \
  --environment isaac_eval/environments/static350_seed18_n256.pt \
  --checkpoint lidar_WAM/outputs/action_expert_joint/best.pt \
  --device cuda:0
```

Because this is the default environment file, the `--environment` option may
be omitted. Pass `--random-environment` only when a newly sampled route set is
intentionally required.

Results are written to `isaac_eval/results/action_expert_seed<seed>_n<N>.json`.
They include success, collision, out-of-bounds and timeout rates; path length;
episode duration; minimum LiDAR clearance; and Action Expert replanning
latency. Add `--render` to open the Isaac viewer for a small visual run.

## PPO collector baseline

The exact deterministic PPO expert recorded in the `wam_data` metadata is
copied into `isaac_eval/checkpoints/ppo_dataset_collector.pt`. Evaluate it on
the same fixed 256-route environment with:

```bash
/isaac-sim/python.sh -m isaac_eval.evaluate_ppo \
  --environment isaac_eval/environments/static350_seed18_n256.pt \
  --checkpoint isaac_eval/checkpoints/ppo_dataset_collector.pt \
  --device cuda:0
```

The evaluator uses `ExplorationType.MEAN`, matching dataset collection. PPO
results are saved below `isaac_eval/results/ppo_fixed_env/`.

## Evaluation protocol

- Seeds 18 and 19 are the held-out test terrains from the dataset.
- The environment contains 350 static obstacles and no dynamic obstacles.
- The `.pt` file fixes all 256 start and goal routes. Its route distribution
  was sampled from the same independent four-edge distribution as training.
- A goal is reached within 0.5 m.
- Collision uses the environment's 0.3 m LiDAR threshold.
- The default limit is 2,200 simulation steps at 0.016 seconds per step.
- Action-flow sampling is reproducible through `--policy-seed`.

## Complete-trajectory PPO dataset (v2)

The standalone collector stores one complete PPO episode for every drone.  It
never resets or replaces a completed drone.  It records the initial frame,
every tenth physics step, and an additional terminal frame when termination
does not fall on a tenth step.  Each seed uses one compressed HDF5 file and one
self-contained `navrl-isaac-environment-v2` snapshot containing the exact
triangle mesh and all 512 routes.

Run the mandatory capacity tests first.  They execute PPO, LiDAR and physics
for 25 steps at each size and write logs only (no partial trajectory dataset),
so a capacity probe cannot be mistaken for production data:

```bash
cd /workspace/NavRL/isaac-training/aliyun_lidar_wam
/isaac-sim/python.sh -m isaac_eval.collect_all \
  --mode smoke --gpus 0 --seeds 0
```

If 512 environments exhaust a PhysX GPU buffer, increase the buffer in
`isaac_eval/cfg/sim.yaml` and repeat the 512 test.  The scripts intentionally
do not reduce the requested environment count.

Collect seeds 0--7 on GPUs 0--7, then assign seeds 8 and 9 to the first GPUs
that finish:

```bash
/isaac-sim/python.sh -m isaac_eval.collect_all \
  --mode collect --gpus 0,1,2,3,4,5,6,7 --seeds 0-9 \
  --output-root datasets/wam_static_seed00_09
```

For a one-GPU or resumable manual run, launch one seed at a time.  Existing
`.pt`, `.h5`, or summary files are never overwritten:

```bash
/isaac-sim/python.sh -m isaac_eval.collect_dataset \
  --seed 0 --num-envs 512 --device cuda:0 --split train \
  --output-root datasets/wam_static_seed00_09
```

Validate one completed seed without launching Isaac:

```bash
/isaac-sim/python.sh -m isaac_eval.validate_seed \
  --environment datasets/wam_static_seed00_09/environments/static350_seed00_n512.pt \
  --dataset datasets/wam_static_seed00_09/train/seed_0000/trajectories.h5 \
  --num-scenes 512
```

After all ten seeds exist, regenerate and verify the global and split
manifests:

```bash
/isaac-sim/python.sh -m isaac_eval.finalize_dataset \
  --output-root datasets/wam_static_seed00_09
```

The acceptance replay for seeds 0, 5 and 9 uses 16 routes each.  Change the
split in the paths for seed 9 (`test`) and run:

```bash
/isaac-sim/python.sh -m isaac_eval.replay_verify \
  --environment datasets/wam_static_seed00_09/environments/static350_seed00_n512.pt \
  --dataset datasets/wam_static_seed00_09/train/seed_0000/trajectories.h5 \
  --count 16 --tolerance 1e-5 --device cuda:0
```

The final layout is:

```text
datasets/wam_static_seed00_09/
├── environments/static350_seed00_n512.pt ... static350_seed09_n512.pt
├── train/seed_0000 ... seed_0007
├── val/seed_0008
├── test/seed_0009
└── manifest.json
```

The training directory currently contains `best.pt` and `normalization.json`
but not `best_metrics.json`, `history.json`, or `config.json`. Successful
checkpoint loading only proves structural compatibility; navigation quality
must be judged from these closed-loop results.

## Independence boundary

Everything specific to this evaluator lives below `aliyun_lidar_wam`:

```text
aliyun_lidar_wam/
├── lidar_WAM/                 # VAE, model definitions, trained weights
└── isaac_eval/
    ├── cfg/                   # copied NavigationEnv configuration
    ├── checkpoints/           # self-contained PPO collector checkpoint
    ├── environments/          # serialized, reusable evaluation environments
    ├── third_party/           # vendored runtime source dependencies
    ├── create_environment.py  # fixed .pt environment generator
    ├── environment_file.py    # .pt schema, validation, and loading
    ├── navigation_env.py      # copied static NavigationEnv
    ├── nav_utils.py           # copied coordinate and LiDAR utilities
    ├── policy.py              # inference preprocessing and policy wrapper
    ├── ppo_lidar_encoder.py   # collector PPO range-image encoder
    ├── ppo_policy.py          # collector PPO inference model
    ├── evaluate_ppo.py        # fixed-environment PPO evaluator
    ├── export_policy.py       # optional legacy compact checkpoint exporter
    └── evaluate.py            # closed-loop Isaac evaluation
```

Isaac Sim itself is proprietary runtime software and is intentionally provided
by the container rather than copied into the project.
