import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TimingStep:
    inference_steps: int
    command_steps: int
    elapsed_steps: int


class TwoStageDelaySchedule:
    """Quantized inference and command delays for a shared simulation clock.

    One outer environment transition represents one actor inference. Command
    transport is handled by a persistent FIFO in the environment, so it can
    overlap the following inference instead of extending the actor period.
    """

    def __init__(self, cfg, physics_dt: float, nominal_steps: int):
        self.enabled = bool(cfg.enabled)
        self.mode = str(cfg.get("mode", "overlapping_transport"))
        if self.mode != "overlapping_transport":
            raise ValueError("Only timing.mode=overlapping_transport is supported")
        self.randomize_in_eval = bool(cfg.get("randomize_in_eval", False))
        self.change_probability = float(cfg.change_probability)
        self.max_step_change = int(cfg.max_step_change)
        self.physics_dt = float(physics_dt)
        self.nominal_steps = int(nominal_steps)

        if self.physics_dt <= 0.0:
            raise ValueError("physics_dt must be positive")
        if self.nominal_steps <= 0:
            raise ValueError("nominal_steps must be positive")
        if not 0.0 <= self.change_probability <= 1.0:
            raise ValueError("timing.change_probability must be in [0, 1]")
        if self.max_step_change < 0:
            raise ValueError("timing.max_step_change must be non-negative")

        self.inference_range = self._parse_range(cfg.inference_delay, "inference_delay")
        self.command_range = self._parse_range(cfg.command_delay, "command_delay")
        self.reset()

    def _parse_range(self, cfg, name):
        minimum = float(cfg.min)
        maximum = float(cfg.max)
        evaluation = float(cfg.eval)
        if minimum < 0.0 or maximum < minimum:
            raise ValueError(f"timing.{name} must satisfy 0 <= min <= max")

        min_steps = max(0, math.ceil(minimum / self.physics_dt - 1e-9))
        max_steps = max(min_steps, math.floor(maximum / self.physics_dt + 1e-9))
        eval_steps = min(
            max_steps,
            max(min_steps, round(evaluation / self.physics_dt)),
        )
        return min_steps, max_steps, eval_steps

    def reset(self):
        self.inference_steps = self.inference_range[2]
        self.command_steps = self.command_range[2]

    def _maybe_change(self, current: int, bounds):
        minimum, maximum, _ = bounds
        if (
            self.max_step_change > 0
            and torch.rand(1).item() < self.change_probability
        ):
            change = int(
                torch.randint(
                    -self.max_step_change,
                    self.max_step_change + 1,
                    (1,),
                ).item()
            )
            current = min(maximum, max(minimum, current + change))
        return current

    def sample(self, training: bool) -> TimingStep:
        if not self.enabled:
            return TimingStep(
                inference_steps=0,
                command_steps=0,
                elapsed_steps=self.nominal_steps,
            )

        if not training and not self.randomize_in_eval:
            inference_steps = self.inference_range[2]
            command_steps = self.command_range[2]
        else:
            self.inference_steps = self._maybe_change(
                self.inference_steps, self.inference_range
            )
            self.command_steps = self._maybe_change(
                self.command_steps, self.command_range
            )
            inference_steps = self.inference_steps
            command_steps = self.command_steps

        return TimingStep(
            inference_steps=inference_steps,
            command_steps=command_steps,
            elapsed_steps=max(self.nominal_steps, inference_steps),
        )
