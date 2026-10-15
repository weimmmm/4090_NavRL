"""Scheduling/isolation tests; no GPU processes or external writes required."""
import ast
from pathlib import Path
import tempfile
import unittest

from online_sft_jobs import atomic_json, read_json, snapshot_steps, validate_gpu_roles, wait_snapshot


class ScheduleTests(unittest.TestCase):
    def test_grid_and_final_snapshot(self):
        self.assertEqual(snapshot_steps(10500, 10900, 200), [10600, 10800, 10900])
        self.assertEqual(snapshot_steps(5220, 10500, 500)[0], 5500)
        self.assertEqual(snapshot_steps(100, 500, 100), [200, 300, 400, 500])

    def test_fixed_validation_also_checks_initial_model(self):
        self.assertEqual(snapshot_steps(10500, 10900, 200, True), [10500, 10600, 10800, 10900])

    def test_overlapping_gpu_roles_rejected(self):
        validate_gpu_roles([1, 2, 3, 7], 0, 4, 5)
        with self.assertRaises(ValueError):
            validate_gpu_roles([1, 2, 3, 7], 0, 7, 5)
        with self.assertRaises(ValueError):
            validate_gpu_roles([1, 2, 3, 7], 0, 4, 4)

    def test_closed_loop_only_reserves_six_distinct_gpus(self):
        validate_gpu_roles([1, 2, 5, 7], 4, 3, None)
        with self.assertRaises(ValueError):
            validate_gpu_roles([1, 2, 5, 7], 4, 5, None)

    def test_only_atomic_published_snapshots_are_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"sft_step_5.pt.tmp").write_bytes(b"unfinished")
            atomic_json(root/"training_finished.json", {"success": False})
            self.assertFalse(wait_snapshot(root/"sft_step_5.pt", root))
            (root/"sft_step_5.pt").write_bytes(b"published")
            self.assertTrue(wait_snapshot(root/"sft_step_5.pt", root))
            self.assertEqual(read_json(root/"training_finished.json"), {"success": False})

    def test_actor_is_not_started_inside_a_loop(self):
        source = Path(__file__).with_name("run_action_online_sft.py")
        tree = ast.parse(source.read_text())
        found = []
        def visit(node, inside_loop=False):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "actor" for t in node.targets):
                if isinstance(node.value, ast.Call):
                    found.append(inside_loop)
            for child in ast.iter_child_nodes(node):
                visit(child, inside_loop or isinstance(node, (ast.For, ast.While)))
        visit(tree)
        self.assertEqual(found, [False])

    def test_evaluators_have_no_actor_control_channel(self):
        source = Path(__file__).with_name("monitor_online_sft.py").read_text()
        self.assertNotIn("multiprocessing.connection", source)
        self.assertNotIn("session_done", source)
        self.assertNotIn("env.reset", source)
        self.assertNotIn("learner.wait", source)

    def test_training_actor_only_initial_full_reset(self):
        tree = ast.parse(Path(__file__).with_name("online_refill.py").read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_refill")
        calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "reset"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args, [])


if __name__ == "__main__":
    unittest.main()
