# Range-image LiDAR PPO

The LiDAR encoder adapts LaGen's range representation and circular-convolution
boundary handling to NavRL's lightweight PPO network. It uses only PyTorch;
LaGen, Diffusers, and pretrained generation weights are not required.

The `agents.observation.lidar` observation is a metric range image with shape
`[batch, 1, elevation, azimuth]`: rows are vertical beams, columns are horizontal
directions, and pixels store distances in meters. Inside the encoder, distance is normalized
to `[-1, 1]` using `sensor.lidar_range`. Each of the three convolutions wraps
azimuth and zero-pads elevation. The default `[batch, 1, 6, 36]` input produces
`[batch, 128]` features for the existing state and dynamic-obstacle fusion.
This is a single-frame encoder with no recurrent hidden state.

Sampling follows P2M's defaults: 10 m range, 360-degree horizontal FOV,
and vertical angles from -7 to 52 degrees. The ray caster samples a 108-by-18
angular grid. Each 3-by-3 block is reduced to its minimum distance, producing
the 36-by-6 angular policy grid, transposed to a 6-row, 36-column image.
Unhit rays or distances beyond the range are represented by 10 m (the maximum
configured range); no separate validity mask is included.
`sensor.lidar_hres` remains the policy grid's angular
step in degrees (10), while `lidar_h_sample` and `lidar_v_sample` specify raw
oversampling factors (3 each). P2M's optical-flow branch is not included.

The environment's reward and collision calculations still use internal
`lidar_range - distance` scans. Train a new policy: the image layout, input
meaning, and convolution kernels have changed, so earlier CNN checkpoints
cannot be loaded directly. Evaluation
must use a checkpoint from this version with matching sensor settings.

LiDAR scans `/World/ground`, whose generated terrain mesh contains the floor
and static obstacles. `/World/defaultGroundPlane` is a separate floor and is
not the LiDAR target. Dynamic obstacles still use the separate state branch.

Run from `NavRL/isaac-training` inside the existing Isaac Sim environment:

```bash
/isaac-sim/python.sh training_lidar/scripts/train.py \
  device=cuda:0 env.num_envs=32 env.num_obstacles=20 \
  env_dyn.num_obstacles=0 wandb.mode=offline enable_eval=false \
  max_frame_num=1024
```

Encoder tests require PyTorch but not Isaac Sim:

```bash
python -m unittest discover -s training_lidar/tests -v
```

The raycast regression additionally requires NumPy and Warp and exercises the
repository's Orbit raycast kernel on a floor-plus-obstacle mesh fixture.
For actual scene validation, run the following inside Isaac Sim. It creates
one environment without dynamic obstacles or W&B, locates generated obstacle
side faces, and checks sensor hits, range-image pixels, and the 0.3 m collision
threshold at two probe positions:

```bash
/isaac-sim/python.sh training_lidar/scripts/check_lidar_hits.py device=cuda:0
```

References:
- https://github.com/szzhou88/LaGen/blob/main/lagen/runner/utils.py
- https://github.com/szzhou88/LaGen/blob/main/lagen/dataset/pipeline_utils.py
- https://github.com/arclab-hku/P2M/blob/main/cfg/task/train_env.yaml
- https://github.com/arclab-hku/P2M/blob/main/resources/envs/single/env.py
