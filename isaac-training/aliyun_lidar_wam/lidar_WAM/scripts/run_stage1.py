"""Run the standalone NavRL stage-one training and evaluation CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar_wam.runner.stage1 import main


if __name__ == "__main__":
    main()
