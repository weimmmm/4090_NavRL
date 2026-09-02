import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TimingStep:
    inference_steps: int
    command_steps: int
    elapsed_steps: int
    # Requested delay values are retained for logging. The environment derives
    # measured delays from simulated event timestamps.
    inference_delay: float = None
    command_delay: float = None


class TwoStageDelaySchedule:
    """Quantized inference and command delays for a shared simulation clock.

    One outer environment transition represents one Actor inference. The
    environment samples command delay separately for each periodic cmd_vel
    publication, so transport can overlap later inference and publication.
    """

    def __init__(self, cfg, physics_dt: float, nominal_steps: int):
        self.enabled = bool(cfg.enabled)
        self.mode = str(cfg.get("mode", "overlapping_transport"))
        if self.mode != "overlapping_transport":
            raise ValueError("Only timing.mode=overlapping_transport is supported")
        self.sampling_mode = str(cfg.get("sampling_mode", "random_walk_steps"))
        self.inference_sampling_mode = str(
            cfg.get("inference_sampling_mode", self.sampling_mode)
        )
        self.command_sampling_mode = str(
            cfg.get("command_sampling_mode", self.sampling_mode)
        )
        valid_sampling_modes = {"random_walk_steps", "continuous_uniform"}
        for name, mode in (
            ("inference_sampling_mode", self.inference_sampling_mode),
            ("command_sampling_mode", self.command_sampling_mode),
        ):
            if mode not in valid_sampling_modes:
                raise ValueError(
                    f"timing.{name} must be random_walk_steps or continuous_uniform"
                )
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
        self.inference_seconds = self._parse_seconds(
            cfg.inference_delay, "inference_delay"
        )
        self.command_seconds = self._parse_seconds(
            cfg.command_delay, "command_delay"
        )
        self.reset()

    @staticmethod
    def _parse_seconds(cfg, name):
        minimum = float(cfg.min)
        maximum = float(cfg.max)
        evaluation = float(cfg.eval)
        if minimum < 0.0 or maximum < minimum:
            raise ValueError(f"timing.{name} must satisfy 0 <= min <= max")
        if not minimum <= evaluation <= maximum:
            raise ValueError(f"timing.{name}.eval must lie within min and max")
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

    def _sample_uniform_seconds(self, bounds):
        minimum, maximum, _ = bounds
        return minimum + (maximum - minimum) * torch.rand(1).item()

    def _quantize_continuous_delay(self, delay_seconds, bounds):
        """Map a continuous sample to a neighboring valid physics tick.

        Stochastic rounding avoids always rounding the same fractional delay
        in one direction, while clamping keeps the configured tick bounds.
        """
        minimum, maximum, _ = bounds
        exact_steps = delay_seconds / self.physics_dt
        lower = math.floor(exact_steps + 1e-12)
        upper = math.ceil(exact_steps - 1e-12)
        lower = min(maximum, max(minimum, lower))
        upper = min(maximum, max(minimum, upper))
        if lower == upper:
            return lower
        fraction = exact_steps - math.floor(exact_steps)
        return upper if torch.rand(1).item() < fraction else lower

    def _sample_stage(self, training, sampling_mode, bounds, seconds, state_name):
        if not self.enabled:
            return 0, 0.0
        if not training and not self.randomize_in_eval:
            return bounds[2], seconds[2]
        if sampling_mode == "continuous_uniform":
            delay = self._sample_uniform_seconds(seconds)
            return self._quantize_continuous_delay(delay, bounds), delay

        steps = self._maybe_change(getattr(self, state_name), bounds)
        setattr(self, state_name, steps)
        return steps, steps * self.physics_dt

    def sample_inference(self, training: bool):
        """Sample one Actor inference delay."""
        return self._sample_stage(
            training,
            self.inference_sampling_mode,
            self.inference_range,
            self.inference_seconds,
            "inference_steps",
        )

    def sample_command(self, training: bool):
        """Sample one delay for a command publication."""
        return self._sample_stage(
            training,
            self.command_sampling_mode,
            self.command_range,
            self.command_seconds,
            "command_steps",
        )

    def sample(self, training: bool) -> TimingStep:
        inference_steps, inference_delay = self.sample_inference(training)
        command_steps, command_delay = self.sample_command(training)

        return TimingStep(
            inference_steps=inference_steps,
            command_steps=command_steps,
            elapsed_steps=max(self.nominal_steps, inference_steps),
            inference_delay=inference_delay,
            command_delay=command_delay,
        )
