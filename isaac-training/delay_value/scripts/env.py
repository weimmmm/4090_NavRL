"""Fixed-scene evaluation using the training command-delay environment."""

from copy import deepcopy
import math
from pathlib import Path

from shared import load_training_module
from startup import navigation_viewport_enabled


TrainingNavigationEnv = load_training_module("env").NavigationEnv
PROJECT_PATH = Path(__file__).resolve().parents[1]


def evaluation_config(cfg):
    """Resolve evaluator-relative scenarios without changing the caller."""
    cfg = deepcopy(cfg)
    if not math.isclose(float(cfg.sim.dt), 0.002, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Command-delay evaluation requires sim.dt=0.002")
    if int(cfg.sim.substeps) != 1:
        raise ValueError("Command-delay evaluation requires sim.substeps=1")
    if str(cfg.timing.command_delay.distribution) != "uniform_ticks":
        raise ValueError("Command-delay evaluation requires uniform_ticks")
    path = cfg.get("eval", {}).get("dataset_path")
    if path and not Path(str(path)).is_absolute():
        cfg.eval.dataset_path = str(PROJECT_PATH / str(path))
    return cfg


class CommandDelayEvaluationEnv(TrainingNavigationEnv):
    """Same physics, current-state observations and controller delay as training."""

    def __init__(self, cfg):
        super().__init__(evaluation_config(cfg))

    def _create_viewport_render_product(self):
        self.enable_viewport = navigation_viewport_enabled(self.cfg)
        if self.enable_viewport:
            return super()._create_viewport_render_product()


# IsaacEnv registers subclasses by class name. Keep evaluator's import name
# without re-registering training's NavigationEnv class.
NavigationEnv = CommandDelayEvaluationEnv
