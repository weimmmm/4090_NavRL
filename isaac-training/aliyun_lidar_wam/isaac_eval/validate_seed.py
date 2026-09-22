"""Validate a single environment/HDF5 pair without launching Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaac_eval.dataset_io import validate_dataset
from isaac_eval.environment_file import load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--num-scenes", type=int)
    args = parser.parse_args()
    environment = load_environment(args.environment)
    result = validate_dataset(args.dataset, environment, args.num_scenes, 10)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
