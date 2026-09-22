"""Load training implementations without shadowing the evaluator's modules."""
import importlib.util
from pathlib import Path
import sys


TRAINING_SCRIPTS = Path(__file__).resolve().parents[2] / "training_delay" / "scripts"
_DEPENDENCIES = {"env": ("utils", "timing", "startup"), "ppo": ("utils",)}


def load_training_module(name):
    key = "_navrl_training_delay_" + name
    if key in sys.modules:
        return sys.modules[key]
    dependencies = {item: load_training_module(item)
                    for item in _DEPENDENCIES.get(name, ())}
    previous = {item: sys.modules.get(item) for item in dependencies}
    spec = importlib.util.spec_from_file_location(key, TRAINING_SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    sys.modules.update(dependencies)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    finally:
        for item, value in previous.items():
            if value is None:
                sys.modules.pop(item, None)
            else:
                sys.modules[item] = value
    return module
