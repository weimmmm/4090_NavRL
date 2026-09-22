"""Evaluation uses the same 2-ms command-delay dynamics as training."""

import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

try:
    import torch
    from tensordict import TensorDict
except ModuleNotFoundError:
    torch = None
    TensorDict = None


ROOT = Path(__file__).parents[1]


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
    __setattr__ = dict.__setitem__


def make_config():
    return Config(
        sim=Config(dt=0.002, substeps=1),
        timing=Config(command_delay=Config(distribution="uniform_ticks")),
        eval=Config(dataset_path="environments/fixed_scenarios.pt"),
    )


class EvaluationConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared = types.ModuleType("shared")
        shared.load_training_module = lambda name: types.SimpleNamespace(
            NavigationEnv=object
        )
        startup = types.ModuleType("startup")
        startup.navigation_viewport_enabled = lambda cfg: False
        with patch.dict(sys.modules, {"shared": shared, "startup": startup}):
            spec = importlib.util.spec_from_file_location(
                "command_eval_config", ROOT / "scripts/env.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        cls.evaluation_config = staticmethod(module.evaluation_config)

    def test_accepts_two_ms_policy_and_resolves_dataset(self):
        original = make_config()
        evaluated = self.evaluation_config(original)
        self.assertEqual(evaluated.sim.substeps, 1)
        self.assertTrue(Path(evaluated.eval.dataset_path).is_absolute())
        self.assertEqual(original.eval.dataset_path, "environments/fixed_scenarios.pt")

    def test_rejects_old_actor_cadence(self):
        cfg = make_config()
        cfg.sim.substeps = 50
        with self.assertRaisesRegex(ValueError, "substeps=1"):
            self.evaluation_config(cfg)

    def test_no_timing_feature_mismatch_configuration(self):
        text = (ROOT / "cfg/eval_random.yaml").read_text()
        self.assertNotIn("timing_feature_mismatch", text)
        self.assertNotIn("odom_delay", text)
        self.assertNotIn("command_queue_capacity", text)
        self.assertIn("max_steps: 352", text)


@unittest.skipUnless(torch is not None and TensorDict is not None, "PyTorch and TensorDict are required")
class BaselineAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ast.parse((ROOT / "scripts/eval_random_delay.py").read_text())
        classes = [
            node for node in source.body
            if isinstance(node, ast.ClassDef)
            and node.name in ("_BaselineObservationSpec", "_BaselinePolicyAdapter")
        ]
        namespace = {}
        exec(compile(ast.Module(body=classes, type_ignores=[]), "baseline_adapter", "exec"), namespace)
        cls.spec_view = staticmethod(namespace["_BaselineObservationSpec"])
        cls.adapter = staticmethod(namespace["_BaselinePolicyAdapter"])

    def test_baseline_sees_eight_values_without_mutating_environment_state(self):
        class FakeSpec:
            def zero(self):
                return TensorDict({"agents": {"observation": {
                    "state": torch.zeros(2, 12),
                }}}, [2])

        class FakePolicy:
            def __call__(self, inputs):
                assert inputs["agents", "observation", "state"].shape[-1] == 8
                inputs.set(("agents", "action"), torch.ones(2, 1, 3))
                return inputs

        self.assertEqual(
            self.spec_view(FakeSpec()).zero()["agents", "observation", "state"].shape[-1],
            8,
        )
        input_td = FakeSpec().zero()
        output_td = self.adapter(FakePolicy())(input_td)
        self.assertIs(output_td, input_td)
        self.assertEqual(input_td["agents", "observation", "state"].shape[-1], 12)
        self.assertEqual(input_td["agents", "action"].shape, (2, 1, 3))


if __name__ == "__main__":
    unittest.main()
