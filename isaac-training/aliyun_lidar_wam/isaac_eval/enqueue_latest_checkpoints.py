"""Snapshot an atomically-written ``latest.pt`` for asynchronous evaluation.

This helper is intentionally separate from the trainer so it can be attached
to an already-running job.  It only copies a checkpoint when its embedded
optimizer step changes and uses an atomic rename for the destination queue.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    latest = args.run_dir / "latest.pt"
    queue = args.run_dir / "closed_loop_queue"
    queue.mkdir(parents=True, exist_ok=True)
    last_mtime_ns = -1

    while True:
        try:
            stat = latest.stat()
        except FileNotFoundError:
            time.sleep(args.poll_seconds)
            continue
        if stat.st_mtime_ns == last_mtime_ns:
            time.sleep(args.poll_seconds)
            continue

        # The trainer publishes latest.pt with os.replace, so a successful
        # load always observes one complete checkpoint rather than a partial
        # write.
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        step = int(payload["step"])
        last_mtime_ns = stat.st_mtime_ns
        if step % args.every == 0:
            destination = queue / f"step_{step:06d}.pt"
            if not destination.exists():
                temporary = destination.with_suffix(".pt.tmp")
                shutil.copyfile(latest, temporary)
                os.replace(temporary, destination)
                print(json.dumps({
                    "event": "checkpoint_enqueued",
                    "step": step,
                    "path": str(destination),
                }), flush=True)
        if step >= args.final_step:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
