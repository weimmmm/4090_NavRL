import argparse
import os

import torch


DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "environments", "fixed_scenarios.pt"
)


def create_eval_scenarios(
    num_envs: int,
    seed: int,
    terrain_seed: int,
    num_obstacles: int = 350,
    num_dynamic_obstacles: int = 80,
) -> dict:
    """Create fixed scenarios drawn from the training reset distribution."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if num_obstacles < 0 or num_dynamic_obstacles < 0:
        raise ValueError("obstacle counts must be non-negative")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample_boundary_positions():
        """Match the boundary-position distribution used by training.

        Training samples the start boundary and target boundary independently;
        they are not forced to be on opposite sides of the map.
        """
        side = torch.randint(0, 4, (num_envs,), generator=generator)
        lateral = torch.empty(num_envs).uniform_(-24.0, 24.0, generator=generator)
        height = torch.empty(num_envs).uniform_(0.5, 2.5, generator=generator)
        positions = torch.zeros(num_envs, 1, 3)
        y_positive = side == 0
        y_negative = side == 1
        x_positive = side == 2
        x_negative = side == 3
        positions[y_positive, 0, 0] = lateral[y_positive]
        positions[y_positive, 0, 1] = 24.0
        positions[y_negative, 0, 0] = lateral[y_negative]
        positions[y_negative, 0, 1] = -24.0
        positions[x_positive, 0, 0] = 24.0
        positions[x_positive, 0, 1] = lateral[x_positive]
        positions[x_negative, 0, 0] = -24.0
        positions[x_negative, 0, 1] = lateral[x_negative]
        positions[:, 0, 2] = height
        return positions

    # Keep the two draws independent, as they are in the training reset.
    start_pos = sample_boundary_positions()
    target_pos = sample_boundary_positions()

    return {
        "format_version": 1,
        "sampling": "training_independent_boundary_sides",
        "seed": seed,
        "terrain_seed": terrain_seed,
        "num_envs": num_envs,
        "num_obstacles": num_obstacles,
        "num_dynamic_obstacles": num_dynamic_obstacles,
        "start_pos": start_pos,
        "target_pos": target_pos,
    }


def main():
    parser = argparse.ArgumentParser(description="Create a fixed NavRL evaluation set")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain-seed", type=int, default=0)
    parser.add_argument("--num-obstacles", type=int, default=350)
    parser.add_argument("--num-dynamic-obstacles", type=int, default=80)
    args = parser.parse_args()

    scenarios = create_eval_scenarios(
        args.num_envs,
        args.seed,
        args.terrain_seed,
        args.num_obstacles,
        args.num_dynamic_obstacles,
    )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(scenarios, output)
    print(
        f"Saved {args.num_envs} fixed evaluation scenarios to {output} "
        f"(static={args.num_obstacles}, dynamic={args.num_dynamic_obstacles})"
    )


if __name__ == "__main__":
    main()
