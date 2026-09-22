import __future__
import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    import numpy as np
    import warp as wp
except ImportError:
    wp = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lidar_encoder import min_pool_ranges


@unittest.skipIf(wp is None, "NumPy and Warp are required for the raycast regression")
class StaticObstacleRaycastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = tempfile.TemporaryDirectory(prefix="navrl-raycast-")
        cls.previous_cache = wp.config.kernel_cache_dir
        wp.config.kernel_cache_dir = cls.cache.name
        wp.init()
        orbit = Path(__file__).resolve().parents[2] / "third_party/orbit/source/extensions/omni.isaac.orbit"
        module_path = orbit / "omni/isaac/orbit/utils/warp/__init__.py"
        spec = importlib.util.spec_from_file_location(
            "_navrl_orbit_warp_test", module_path,
            submodule_search_locations=[str(module_path.parent)],
        )
        cls.ops = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.ops
        spec.loader.exec_module(cls.ops)

    @classmethod
    def tearDownClass(cls):
        wp.config.kernel_cache_dir = cls.previous_cache
        cls.cache.cleanup()

    def test_floor_only_misses_obstacle_but_terrain_mesh_detects_it(self):
        vertices = np.array([
            [-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0],
            [2, -1, 0], [2, 1, 0], [2, 1, 3], [2, -1, 3],
        ], dtype=np.float32)
        floor_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        terrain_faces = np.concatenate([floor_faces, [[4, 5, 6], [4, 6, 7]]])
        floor = self.ops.convert_to_warp_mesh(vertices, floor_faces, "cpu")
        terrain = self.ops.convert_to_warp_mesh(vertices, terrain_faces, "cpu")
        starts = torch.tensor([[[0., 0., 1.], [0., 0., 1.]]])
        directions = torch.tensor([[[1., 0., 0.], [0., 0., -1.]]])
        floor_hits = self.ops.raycast_mesh(starts, directions, floor, max_dist=10)[0]
        terrain_hits = self.ops.raycast_mesh(starts, directions, terrain, max_dist=10)[0]
        self.assertTrue(torch.isinf(floor_hits[0, 0]).all())
        torch.testing.assert_close(terrain_hits[0, 0], torch.tensor([2., 0., 1.]))
        torch.testing.assert_close(terrain_hits[0, 1], torch.tensor([0., 0., 0.]))

        starts = torch.tensor([[[1.8, 0., 1.]]])
        hit = self.ops.raycast_mesh(starts, directions[:, :1], terrain, max_dist=10)[0]
        distance = (hit - starts).norm(dim=-1)
        raw_ranges = distance.reshape(1, 1, 1, 1).expand(1, 1, 108, 18)
        pooled = min_pool_ranges(raw_ranges, 3, 3)
        image = pooled.transpose(-2, -1).contiguous()
        self.assertEqual(image.shape, (1, 1, 6, 36))
        torch.testing.assert_close(image, torch.full_like(image, 0.2))
        scan = 10 - pooled
        self.assertTrue((scan.amax(dim=(-2, -1)) > 10 - 0.3).item())

    def test_repository_generated_terrain_returns_obstacle_surface_hits(self):
        root = Path(__file__).resolve().parents[2]
        terrain_dir = root / "third_party/orbit/source/extensions/omni.isaac.orbit/omni/isaac/orbit/terrains/height_field"

        def load_function(filename, name):
            path = terrain_dir / filename
            node = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            node.decorator_list = []
            namespace = {"np": np}
            code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                           flags=__future__.annotations.compiler_flag)
            exec(code, namespace)
            return namespace[name]

        tree = ast.parse((Path(__file__).resolve().parents[1] / "scripts/env.py").read_text())
        config_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Name) and n.func.id == "HfDiscreteObstaclesTerrainCfg")
        config_values = {k.arg: ast.literal_eval(k.value) for k in config_call.keywords
                         if k.arg != "num_obstacles"}
        cfg = SimpleNamespace(**config_values, size=(40., 40.), num_obstacles=20)
        generate = load_function("hf_terrains.py", "discrete_obstacles_terrain")
        convert = load_function("utils.py", "convert_height_field_to_mesh")
        random_state = np.random.get_state()
        try:
            np.random.seed(0)
            heights = generate(0., cfg)
        finally:
            np.random.set_state(random_state)
        vertices, faces = convert(heights, cfg.horizontal_scale, cfg.vertical_scale, 0.75)
        triangles = vertices[faces]
        centers = triangles.mean(axis=1)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals / np.maximum(lengths[:, None], 1e-6)
        candidates = np.flatnonzero((lengths > 1e-6) & (np.abs(normals[:, 2]) < 0.1)
                                   & (centers[:, 2] > 0.5) & (centers[:, 2] < 3.))
        self.assertGreater(candidates.size, 0)
        center, normal = centers[candidates[0]], normals[candidates[0]]
        starts = torch.tensor(np.stack([center + 0.5 * normal, center + 0.15 * normal])[None])
        directions = torch.tensor(np.stack([-normal, -normal])[None])
        mesh = self.ops.convert_to_warp_mesh(vertices, faces, "cpu")
        hits = self.ops.raycast_mesh(starts, directions, mesh, max_dist=10)[0]
        self.assertTrue(torch.isfinite(hits).all())
        torch.testing.assert_close(hits, torch.tensor(np.stack([center, center])[None]))
        torch.testing.assert_close((hits - starts).norm(dim=-1), torch.tensor([[0.5, 0.15]]))


if __name__ == "__main__":
    unittest.main()
