"""Evaluation adapter around the training_delay navigation environment."""

from env import NavigationEnv


class TwoStageDelayEvalEnv(NavigationEnv):
    """Use the training environment with fixed scenes and two random delays."""

    def __init__(self, cfg):
        eval_cfg = cfg.get("eval", {})
        timing_cfg = cfg.get("timing", {})
        if not bool(eval_cfg.get("enabled", False)):
            raise ValueError("Evaluation requires eval.enabled=true")
        if not bool(timing_cfg.get("enabled", False)):
            raise ValueError("Two-stage evaluation requires timing.enabled=true")
        if str(timing_cfg.get("mode", "")) != "overlapping_transport":
            raise ValueError(
                "Two-stage evaluation requires timing.mode=overlapping_transport"
            )
        if not bool(timing_cfg.get("randomize_in_eval", False)):
            raise ValueError(
                "Two-stage evaluation requires timing.randomize_in_eval=true"
            )
        for name in ("inference_delay", "command_delay"):
            stage = timing_cfg.get(name)
            if stage is None:
                raise ValueError(f"Evaluation timing is missing timing.{name}")
            if float(stage.max) <= 0.0:
                raise ValueError(f"timing.{name}.max must be positive")
        super().__init__(cfg)
