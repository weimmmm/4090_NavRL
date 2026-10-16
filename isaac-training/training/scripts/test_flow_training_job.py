"""CPU-only contract tests for the shared-GPU Flow training supervisor."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


path = Path(__file__).with_name("run_flow_action_training.py")
spec = importlib.util.spec_from_file_location("flow_training_job", path)
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


class FlowTrainingJobTests(unittest.TestCase):
    def test_isaac_is_scoped_to_gpu0_and_uses_ten_step_flow(self):
        root = Path("/home/yimingwei/NavRL")
        command = job.evaluation_command(root / "run/step_500.pt", root / "run/eval", root)
        self.assertIn("navrl-collect-gpu0", command)
        self.assertIn("ISAAC_ACTIVE_GPU=0", command)
        for item in ("device=cuda:0", "env.num_envs=128", "env.max_episode_length=2500",
                     "wam_eval.action_generation=flow", "wam_eval.sample_steps=10",
                     "wam_eval.use_past_actions=false"):
            self.assertIn(item, command)
        self.assertFalse(any("CUDA_VISIBLE_DEVICES=1" in item for item in command))

    def test_all_checkpoints_are_evaluated_in_order_and_best_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "isaac-training/training/lidar_WAM/outputs/test_flow"
            commands = []

            def spawn(command, log_path):
                commands.append(command)
                if "torchrun" in command:
                    run.mkdir(parents=True)
                    for step in (500, 1000):
                        (run / f"step_{step}.pt").write_bytes(str(step).encode())
                    return SimpleNamespace(pid=100, poll=lambda: 0, returncode=0)
                if "--logdir" in command:
                    return SimpleNamespace(pid=101, poll=lambda: None)
                checkpoint = next(item for item in command if item.startswith("wam_eval.checkpoint="))
                step = int(Path(checkpoint.split("=", 1)[1]).stem.split("_")[-1])
                success = 96 if step == 500 else 101
                metrics = {"checkpoint_step": step, "initial_conditions_sha256": "fixed",
                           "counts": {"success": success, "collision": 128 - success, "timeout": 0},
                           "rates": {"sr": success / 128}, "wall_time_s": 2.}
                (log_path.parent / "metrics.json").write_text(json.dumps(metrics))
                return SimpleNamespace(pid=200 + step, poll=lambda: 0, returncode=0)

            args = SimpleNamespace(host_root=root, run_name="test_flow", min_train_free_mib=4800,
                batch_size=128, steps=1000, ddp_port=29584, memory_fraction=.16,
                tb_port=6024, min_eval_free_mib=5000)
            with mock.patch.object(job, "gpu_free_mib", return_value=6000), \
                    mock.patch.object(job, "spawn", side_effect=spawn), \
                    mock.patch.object(job.time, "sleep"), \
                    mock.patch.object(job.subprocess, "run") as record:
                job.run(args)
            status = json.loads(run.with_name("test_flow_job").joinpath("status.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertEqual(list(status["evaluations"]), ["500", "1000"])
            self.assertTrue(all(value["state"] == "complete" for value in status["evaluations"].values()))
            self.assertEqual(status["best_closed_loop_step"], 1000)
            self.assertEqual((run / "best_closed_loop.pt").read_bytes(), b"1000")
            self.assertEqual(record.call_count, 2)
            self.assertIn("CUDA_VISIBLE_DEVICES=1,2,3,4,5,7", commands[0])
            self.assertEqual(status["global_batch"], 768)
            self.assertEqual(len(commands), 4)  # trainer + TensorBoard + two evaluations


if __name__ == "__main__":
    unittest.main()
