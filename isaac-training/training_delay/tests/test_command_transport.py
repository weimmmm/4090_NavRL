import sys
import unittest
from pathlib import Path


try:
    import torch
except ModuleNotFoundError:
    torch = None


NAVIGATION_ENV = None
IMPORT_ERROR = "PyTorch is not installed"
if torch is not None and hasattr(torch, "tensor"):
    scripts_dir = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from env import NavigationEnv as NAVIGATION_ENV
    except (ImportError, ModuleNotFoundError) as exc:
        IMPORT_ERROR = str(exc)
    finally:
        sys.path.pop(0)


@unittest.skipUnless(
    NAVIGATION_ENV is not None,
    f"requires the Isaac Sim Python environment: {IMPORT_ERROR}",
)
class CommandTransportTest(unittest.TestCase):
    def make_transport(self):
        env = NAVIGATION_ENV.__new__(NAVIGATION_ENV)
        env.num_envs = 1
        env.device = torch.device("cpu")
        env.dt = 0.016
        env.active_cmd_vel = torch.zeros(1, 3)
        env.next_pending_cmd_vel = torch.zeros(1, 3)
        env.inference_delay = torch.zeros(1, 1)
        env.command_delay = torch.zeros(1, 1)
        env.publisher_wait_delay = torch.zeros(1, 1)
        env.transport_delay = torch.zeros(1, 1)
        env.total_delay = torch.zeros(1, 1)
        env.command_age_at_update = torch.zeros(1, 1)
        env.active_command_age = torch.zeros(1, 1)
        env.has_active_command = torch.zeros(1, dtype=torch.bool)
        env.command_sequence = torch.zeros(1, dtype=torch.long)
        env.last_applied_sequence = torch.full((1,), -1, dtype=torch.long)
        env.last_applied_actor_sequence = torch.full(
            (1,), -1, dtype=torch.long
        )
        env.pending_command_age = torch.zeros(1, 1)
        env.command_queue_depth = torch.zeros(1, 1)
        env.stats = {"command_update_count": torch.zeros(1, 1)}
        env._command_queue = []
        return env

    def advance(self, env, ticks=1):
        for _ in range(ticks):
            env.active_command_age += env.dt
            env._advance_command_transport(torch.tensor([[True]]))

    def enqueue(
        self,
        env,
        value,
        delay_steps,
        actor_sequence,
        inference_delay=0.0,
        publisher_wait_delay=0.0,
    ):
        command = torch.full((1, 3), float(value))
        env._enqueue_command(
            command,
            delay_steps,
            torch.tensor([[True]]),
            torch.full((1, 1), inference_delay),
            torch.full((1, 1), publisher_wait_delay),
            torch.tensor([actor_sequence]),
        )

    def test_each_publication_keeps_its_own_delay(self):
        env = self.make_transport()
        self.enqueue(env, value=1, delay_steps=3, actor_sequence=0)
        self.advance(env, 1)
        self.enqueue(env, value=2, delay_steps=4, actor_sequence=1)

        self.advance(env, 2)
        self.assertTrue(torch.equal(env.active_cmd_vel, torch.ones(1, 3)))

        self.advance(env, 2)
        self.assertTrue(torch.equal(env.active_cmd_vel, torch.full((1, 3), 2.0)))

    def test_later_publication_cannot_overtake_earlier_publication(self):
        env = self.make_transport()
        self.enqueue(env, value=1, delay_steps=4, actor_sequence=0)
        self.advance(env, 1)
        self.enqueue(env, value=2, delay_steps=1, actor_sequence=1)

        self.advance(env, 1)
        self.assertTrue(torch.equal(env.active_cmd_vel, torch.zeros(1, 3)))

        self.advance(env, 2)
        self.assertTrue(torch.equal(env.active_cmd_vel, torch.full((1, 3), 2.0)))
        self.assertEqual(env.stats["command_update_count"].item(), 1.0)

    def test_republishing_same_actor_command_does_not_reset_duration(self):
        env = self.make_transport()
        self.enqueue(env, value=1, delay_steps=1, actor_sequence=0)
        self.advance(env, 1)
        self.advance(env, 2)

        self.enqueue(env, value=1, delay_steps=1, actor_sequence=0)
        self.advance(env, 1)

        self.assertAlmostEqual(env.active_command_age.item(), 3 * env.dt)
        self.assertEqual(env.stats["command_update_count"].item(), 1.0)

    def test_total_delay_includes_publisher_wait(self):
        env = self.make_transport()
        self.enqueue(
            env,
            value=1,
            delay_steps=2,
            actor_sequence=0,
            inference_delay=0.032,
            publisher_wait_delay=0.016,
        )

        self.advance(env, 2)

        self.assertAlmostEqual(env.transport_delay.item(), 0.032)
        self.assertAlmostEqual(env.total_delay.item(), 0.080)


if __name__ == "__main__":
    unittest.main()
