"""Use the shared 9-feature command-delay PPO from training_delay."""
from shared import load_training_module

PPO = load_training_module("ppo").PPO
