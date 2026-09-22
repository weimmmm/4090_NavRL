"""Evaluation adapter for the command-only delay environment."""
from env import NavigationEnv


class CommandDelayEvalEnv(NavigationEnv):
    def __init__(self, cfg):
        if not bool(cfg.get("eval", {}).get("enabled", False)):
            raise ValueError("Evaluation requires eval.enabled=true")
        super().__init__(cfg)
