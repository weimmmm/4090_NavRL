import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TimingStep:
    inference_steps: int
    command_steps: int
    elapsed_steps: int
    inference_delay: float
    command_delay: float
    observed_inference_steps: int
    observed_command_steps: int


class TwoStageDelaySchedule:
    """Inference and command delays on separate simulation/observation clocks.

    One outer environment transition represents one actor inference. Command
    transport is handled by a persistent FIFO in the environment, so it can
    overlap the following inference instead of extending the actor period.
    Evaluation can execute delays on a fine physics clock while exposing only
    floor-quantized values to the policy.
    """

    def __init__(
        self,
        cfg,
        physics_dt: float,
        nominal_steps: int,
        observation_dt: float = None,
    ):
        self.enabled = bool(cfg.enabled)
        self.mode = str(cfg.get("mode", "overlapping_transport"))
        if self.mode != "overlapping_transport":
            raise ValueError("Only timing.mode=overlapping_transport is supported")
        self.randomize_in_eval = bool(cfg.get("randomize_in_eval", False))
        self.continuous_in_eval = bool(cfg.get("continuous_in_eval", False))
        self.delay_resolution = float(cfg.get("delay_resolution", 0.001))
        self.observation_quantization = str(
            cfg.get("observation_quantization", "floor")
        )
        self.change_probability = float(cfg.change_probability)
        self.max_step_change = int(cfg.max_step_change)
        self.physics_dt = float(physics_dt)
        self.observation_dt = float(
            physics_dt if observation_dt is None else observation_dt
        )
        self.nominal_steps = int(nominal_steps)
        # Evaluation timing must not depend on random numbers consumed by
        # environment resets or by a particular policy trajectory.
        self.generator = torch.Generator(device="cpu")

        if self.physics_dt <= 0.0:
            raise ValueError("physics_dt must be positive")
        if self.observation_dt <= 0.0:
            raise ValueError("observation_dt must be positive")
        if self.nominal_steps <= 0:
            raise ValueError("nominal_steps must be positive")
        if not 0.0 <= self.change_probability <= 1.0:
            raise ValueError("timing.change_probability must be in [0, 1]")
        if self.max_step_change < 0:
            raise ValueError("timing.max_step_change must be non-negative")
        if self.delay_resolution <= 0.0:
            raise ValueError("timing.delay_resolution must be positive")
        if self.observation_quantization != "floor":
            raise ValueError("Only timing.observation_quantization=floor is supported")

        self.inference_range = self._parse_range(cfg.inference_delay, "inference_delay")
        self.command_range = self._parse_range(cfg.command_delay, "command_delay")
        self.inference_seconds = self._parse_seconds_range(
            cfg.inference_delay, "inference_delay"
        )
        self.command_seconds = self._parse_seconds_range(
            cfg.command_delay, "command_delay"
        )
        self.reset(seed=0)

    @staticmethod
    def _parse_seconds_range(cfg, name):
        minimum = float(cfg.min)
        maximum = float(cfg.max)
        evaluation = float(cfg.eval)
        if minimum < 0.0 or maximum < minimum:
            raise ValueError(f"timing.{name} must satisfy 0 <= min <= max")
        if not minimum <= evaluation <= maximum:
            raise ValueError(f"timing.{name}.eval must be inside [min, max]")
        return minimum, maximum, evaluation

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

    def reset(self, seed=None):
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.inference_steps = self.inference_range[2]
        self.command_steps = self.command_range[2]
        self.inference_delay = self.inference_seconds[2]
        self.command_delay = self.command_seconds[2]

    def _maybe_change(self, current: int, bounds):
        minimum, maximum, _ = bounds
        if (
            self.max_step_change > 0
            and torch.rand(1, generator=self.generator).item()
            < self.change_probability
        ):
            change = int(
                torch.randint(
                    -self.max_step_change,
                    self.max_step_change + 1,
                    (1,),
                    generator=self.generator,
                ).item()
            )
            current = min(maximum, max(minimum, current + change))
        return current

    def _maybe_change_seconds(self, current: float, bounds):
        minimum, maximum, _ = bounds
        if (
            self.max_step_change > 0
            and torch.rand(1, generator=self.generator).item()
            < self.change_probability
        ):
            max_change = max(
                1,
                round(
                    self.max_step_change
                    * self.observation_dt
                    / self.delay_resolution
                ),
            )
            change_units = int(
                torch.randint(
                    -max_change,
                    max_change + 1,
                    (1,),
                    generator=self.generator,
                ).item()
            )
            current += change_units * self.delay_resolution
        return min(maximum, max(minimum, current))

    def _from_seconds(self, inference_delay: float, command_delay: float):
        inference_steps = max(
            0, math.ceil(inference_delay / self.physics_dt - 1e-9)
        )
        command_steps = max(0, math.ceil(command_delay / self.physics_dt - 1e-9))
        observed_inference_steps = max(
            0, math.floor(inference_delay / self.observation_dt + 1e-9)
        )
        observed_command_steps = max(
            0, math.floor(command_delay / self.observation_dt + 1e-9)
        )
        return TimingStep(
            inference_steps=inference_steps,
            command_steps=command_steps,
            elapsed_steps=max(self.nominal_steps, inference_steps),
            inference_delay=inference_delay,
            command_delay=command_delay,
            observed_inference_steps=observed_inference_steps,
            observed_command_steps=observed_command_steps,
        )

    def sample(self, training: bool) -> TimingStep:
        if not self.enabled:
            return self._from_seconds(0.0, 0.0)

        if not training and self.continuous_in_eval:
            if not self.randomize_in_eval:
                self.inference_delay = self.inference_seconds[2]
                self.command_delay = self.command_seconds[2]
            else:
                self.inference_delay = self._maybe_change_seconds(
                    self.inference_delay, self.inference_seconds
                )
                self.command_delay = self._maybe_change_seconds(
                    self.command_delay, self.command_seconds
                )
            return self._from_seconds(self.inference_delay, self.command_delay)

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

        return self._from_seconds(
            inference_steps * self.physics_dt,
            command_steps * self.physics_dt,
        )
