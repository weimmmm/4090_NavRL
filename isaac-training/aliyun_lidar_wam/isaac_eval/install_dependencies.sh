#!/usr/bin/env bash
set -euo pipefail

ISAAC_PYTHON="${ISAAC_PYTHON:-/isaac-sim/python.sh}"

# Do not replace the PyTorch, CUDA, or NumPy builds shipped with Isaac Sim.
"$ISAAC_PYTHON" -m pip install \
  safetensors==0.4.0 \
  accelerate==0.23.0 \
  huggingface-hub==0.17.3 \
  transformers==4.33.3 \
  hydra-core==1.3.2 \
  einops==0.7.0 \
  packaging==23.2 \
  Pillow==10.0.1
