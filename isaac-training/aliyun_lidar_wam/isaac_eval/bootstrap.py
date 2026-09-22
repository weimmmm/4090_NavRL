"""Prefer dependency sources vendored inside ``aliyun_lidar_wam``."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
THIRD_PARTY = ROOT / "third_party"


def activate_vendored_sources():
    paths = [
        THIRD_PARTY / "OmniDrones",
        THIRD_PARTY / "tensordict",
        THIRD_PARTY / "rl",
    ]
    orbit_extensions = THIRD_PARTY / "orbit" / "source" / "extensions"
    if orbit_extensions.is_dir():
        paths.extend(sorted(path for path in orbit_extensions.iterdir() if path.is_dir()))
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing vendored dependency directories: " + ", ".join(missing))
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

