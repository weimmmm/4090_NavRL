import importlib.util
import sys
import types
import unittest
from pathlib import Path


try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    # Fixed-evaluation schedule tests do not call random tensor functions.
    sys.modules["torch"] = types.ModuleType("torch")


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "timing.py"
SPEC = importlib.util.spec_from_file_location("training_delay_timing", MODULE_PATH)
TIMING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TIMING
SPEC.loader.exec_module(TIMING)


class Config(dict):
    __getattr__ = dict.__getitem__


def make_config(enabled=True, mode="overlapping_transport"):
    return Config(
        enabled=enabled,
        mode=mode,
        randomize_in_eval=False,
        change_probability=0.2,
        max_step_change=1,
        inference_delay=Config(min=0.016, max=0.096, eval=0.032),
        command_delay=Config(min=0.016, max=0.032, eval=0.016),
    )


class TwoStageDelayScheduleTest(unittest.TestCase):
    def test_fixed_eval_transition_covers_inference_only(self):
        schedule = TIMING.TwoStageDelaySchedule(
            make_config(), physics_dt=0.016, nominal_steps=1
        )

        step = schedule.sample(training=False)

        self.assertEqual(step.inference_steps, 2)
        self.assertEqual(step.command_steps, 1)
        self.assertEqual(step.elapsed_steps, 2)

    def test_disabled_timing_preserves_nominal_period(self):
        schedule = TIMING.TwoStageDelaySchedule(
            make_config(enabled=False), physics_dt=0.016, nominal_steps=2
        )

        step = schedule.sample(training=True)

        self.assertEqual(step.inference_steps, 0)
        self.assertEqual(step.command_steps, 0)
        self.assertEqual(step.elapsed_steps, 2)

    def test_ranges_are_quantized_to_physics_ticks(self):
        schedule = TIMING.TwoStageDelaySchedule(
            make_config(), physics_dt=0.016, nominal_steps=1
        )

        self.assertEqual(schedule.inference_range, (1, 6, 2))
        self.assertEqual(schedule.command_range, (1, 2, 1))

    def test_real_command_delay_range_is_quantized_to_4_9_ticks(self):
        cfg = make_config()
        cfg.command_delay = Config(min=0.060, max=0.150, eval=0.100)
        schedule = TIMING.TwoStageDelaySchedule(
            cfg, physics_dt=0.016, nominal_steps=1
        )

        self.assertEqual(schedule.command_range, (4, 9, 6))
        step = schedule.sample(training=False)
        self.assertEqual(step.command_steps, 6)
        self.assertAlmostEqual(step.command_delay, 0.100)

    def test_stage_samplers_can_run_on_different_clocks(self):
        schedule = TIMING.TwoStageDelaySchedule(
            make_config(), physics_dt=0.016, nominal_steps=1
        )

        inference_steps, inference_delay = schedule.sample_inference(
            training=False
        )
        command_steps, command_delay = schedule.sample_command(training=False)

        self.assertEqual((inference_steps, inference_delay), (2, 0.032))
        self.assertEqual((command_steps, command_delay), (1, 0.016))

    def test_old_serial_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlapping_transport"):
            TIMING.TwoStageDelaySchedule(
                make_config(mode="serial_single_flight"),
                physics_dt=0.016,
                nominal_steps=1,
            )


if __name__ == "__main__":
    unittest.main()
