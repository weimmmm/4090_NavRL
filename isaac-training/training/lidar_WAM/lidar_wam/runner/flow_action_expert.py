"""Train a standalone Action Flow policy, with ten Euler steps at inference.

New runs load only the selected frozen VAE; the observation CNN and Action
DiT start from scratch. --resume-from restores a full training checkpoint,
including Adam moments, statistics and the global step. No Direct Transformer
or Future model is constructed.

Example (does not load any old action checkpoint)::

    torchrun --nproc_per_node=4 -m lidar_wam.runner.flow_action_expert \
        --train-root DATA/train --vae-checkpoint VAE/best/diffusion_pytorch_model.safetensors \
        --out outputs --run-name action_flow_history1 --sample-steps 10
"""

from lidar_wam.runner.super_action_expert import parse_args, train


if __name__ == "__main__":
    train(parse_args(default_policy_type="flow"))
