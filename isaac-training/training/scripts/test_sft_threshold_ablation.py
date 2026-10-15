"""Pure scheduling checks; no GPU or training process is started."""
import unittest

from run_sft_threshold_ablation import command_for, threshold_values


class ThresholdAblationTests(unittest.TestCase):
    def test_requested_threshold_grid(self):
        self.assertEqual(threshold_values("0.1,0.2,0.3,0.4"), [.1, .2, .3, .4])
        for raw in ("0", "-0.1", "nan", "inf", "0.1,0.10", ""):
            with self.assertRaises(ValueError):
                threshold_values(raw)

    def test_only_threshold_and_output_name_change_between_runs(self):
        def invariant(threshold):
            command = command_for("test_"+str(threshold), threshold, "/original.pt", 500)
            for flag in ("--run-name", "--selective-correction-threshold-mps"):
                command[command.index(flag)+1] = "VARIANT"
            return command
        control = invariant(.3)
        for threshold in (.1, .2, .4):
            self.assertEqual(invariant(threshold), control)

    def test_source_and_frozen_head_and_resources(self):
        command = command_for("test", .3, "/original.pt", 500, evaluate_start=True)
        self.assertEqual(command[command.index("--start-step")+1], "0")
        self.assertEqual(command[command.index("--updates")+1], "500")
        self.assertEqual(command[command.index("--training-gpus")+1], "1,2,5,7")
        self.assertEqual(command[command.index("--sim-gpu")+1], "4")
        self.assertEqual(command[command.index("--eval-gpu")+1], "3")
        self.assertIn("--freeze-policy-output-head", command)
        self.assertIn("--skip-fixed-validation", command)
        self.assertIn("--evaluate-start", command)


if __name__ == "__main__":
    unittest.main()
