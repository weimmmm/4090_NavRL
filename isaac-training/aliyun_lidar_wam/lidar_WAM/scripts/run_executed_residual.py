"""Entry point for the executed-action residual diffusion experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_wam.runner.executed_residual import main


if __name__ == "__main__":
    main()
