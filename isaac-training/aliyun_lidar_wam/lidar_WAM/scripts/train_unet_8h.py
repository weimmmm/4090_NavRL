"""Run stage1's existing LaGen diffusion UNet in a separate checkpoint directory.

The training code, data filter, optimizer, noise schedule and model are exactly
those of stage1.py; only the output directory is changed to preserve the
original 5,000-step checkpoint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_wam.runner import stage1


if __name__ == "__main__":
    stage1.WORLD_FULL_DIR = "world_circular_causal_8h"
    stage1.main()
