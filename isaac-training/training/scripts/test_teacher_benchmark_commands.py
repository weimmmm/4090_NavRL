"""Pure CPU checks for the isolated PPO teacher diagnostic adapter."""
import unittest
import torch
from teacher_benchmark_commands import teacher_ego_chunk


class TeacherBenchmarkTests(unittest.TestCase):
    def test_native_command_is_not_silently_projected(self):
        world = torch.tensor([[2., 2., 0.]])
        quat = torch.tensor([[1., 0., 0., 0.]])
        out = teacher_ego_chunk(world, quat, "native", 10)
        self.assertTrue(torch.equal(out[:, 0], world))
        self.assertGreater(float(out[:, 0].norm()), 2.)

    def test_feasible_command_preserves_direction_and_speed_contract(self):
        world = torch.tensor([[2., 2., 1.], [.1, -.2, .3]])
        quat = torch.tensor([[1., 0., 0., 0.]]).expand(2, -1)
        out = teacher_ego_chunk(world, quat, "student-norm", 10)
        self.assertTrue(torch.all(out.norm(dim=-1)<=2.000001))
        self.assertTrue(torch.allclose(torch.nn.functional.cosine_similarity(out[:, 0], world), torch.ones(2)))
        self.assertTrue(torch.equal(out[1, 0], world[1]))

    def test_bootstrap_is_ten_physics_steps_for_both_query_intervals(self):
        world = torch.ones(1, 3)*2
        quat = torch.tensor([[1., 0., 0., 0.]])
        for step in (0, 1, 9):
            out = teacher_ego_chunk(world, quat, "native", step)
            self.assertTrue(torch.equal(out[:, 0], torch.tensor([[1., 0., 0.]])))
        self.assertTrue(torch.equal(teacher_ego_chunk(world, quat, "native", 10)[:, 0], world))

    def test_current_yaw_rotation_matches_existing_student_contract(self):
        world = torch.tensor([[0., 2., 0.]])
        quat = torch.tensor([[2**-.5, 0., 0., 2**-.5]])
        out = teacher_ego_chunk(world, quat, "native", 10)
        self.assertTrue(torch.allclose(out[0, 0], torch.tensor([2., 0., 0.]), atol=1e-6))
        self.assertTrue(torch.equal(out[:, 1:], out[:, :1].expand(-1, 9, -1)))

    def test_invalid_inputs_are_rejected(self):
        world = torch.ones(1, 3)
        quat = torch.tensor([[1., 0., 0., 0.]])
        with self.assertRaises(ValueError): teacher_ego_chunk(world, quat, "other", 10)
        with self.assertRaises(ValueError): teacher_ego_chunk(world, quat*0, "native", 10)
        with self.assertRaises(ValueError): teacher_ego_chunk(world*float('nan'), quat, "native", 10)


if __name__ == '__main__':
    unittest.main()
