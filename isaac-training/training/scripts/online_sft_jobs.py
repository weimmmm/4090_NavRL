"""Standard-library scheduling helpers for independent SFT/validation jobs."""
import json
import os
from pathlib import Path
import subprocess
import time


def snapshot_steps(start, end, interval, include_start=False):
    if interval <= 0 or end <= start:
        raise ValueError("Snapshot interval must be positive and end must exceed start")
    first = (start // interval + 1) * interval
    steps = list(range(first, end, interval)) + [end]
    return ([start] if include_start else []) + steps


def validate_gpu_roles(training, simulation, evaluation, validation):
    devices = list(training) + [simulation, evaluation]
    if validation is not None:
        devices.append(validation)
    if len(set(devices)) != len(devices) or any(gpu < 0 for gpu in devices):
        raise ValueError("Learner, training simulation, evaluation and validation GPUs must be distinct")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def wait_snapshot(path, run_dir, poll_seconds=1.):
    """Wait for atomic checkpoint publication, never send anything to Isaac."""
    path = Path(path)
    run_dir = Path(run_dir)
    while not path.is_file():
        if read_json(run_dir / "training_finished.json"):
            return False
        time.sleep(poll_seconds)
    return True


def docker_command(container, command, environment, workdir):
    args = ["docker", "exec", "-w", str(workdir)]
    for key, value in environment.items():
        args += ["-e", f"{key}={value}"]
    return args + [container, *map(str, command)]


def start_logged(command, log):
    with Path(log).open("w") as stream:
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)


def run_logged(command, log, timeout=None):
    with Path(log).open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)
