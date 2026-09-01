import argparse
import os

import torch


DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "environments", "fixed_scenarios.pt"
)


def create_eval_scenarios(num_envs: int, seed: int, terrain_seed: int) -> dict:
    """Create fixed boundary-to-boundary navigation scenarios on CPU."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    side = torch.randint(0, 4, (num_envs,), generator=generator)
    start_lateral = torch.empty(num_envs).uniform_(-16.0, 16.0, generator=generator)
    target_lateral = torch.empty(num_envs).uniform_(-16.0, 16.0, generator=generator)
    start_height = torch.empty(num_envs).uniform_(0.5, 2.5, generator=generator)
    target_height = torch.empty(num_envs).uniform_(0.5, 2.5, generator=generator)

    start_pos = torch.zeros(num_envs, 1, 3)
    target_pos = torch.zeros(num_envs, 1, 3)

    # 0/1 cross the map along y; 2/3 cross it along x.
    y_positive = side == 0
    y_negative = side == 1
    x_positive = side == 2
    x_negative = side == 3

    start_pos[y_positive, 0, 0] = start_lateral[y_positive]
    start_pos[y_positive, 0, 1] = 24.0
    target_pos[y_positive, 0, 0] = target_lateral[y_positive]
    target_pos[y_positive, 0, 1] = -24.0

    start_pos[y_negative, 0, 0] = start_lateral[y_negative]
    start_pos[y_negative, 0, 1] = -24.0
    target_pos[y_negative, 0, 0] = target_lateral[y_negative]
    target_pos[y_negative, 0, 1] = 24.0

    start_pos[x_positive, 0, 0] = 24.0
    start_pos[x_positive, 0, 1] = start_lateral[x_positive]
    target_pos[x_positive, 0, 0] = -24.0
    target_pos[x_positive, 0, 1] = target_lateral[x_positive]

    start_pos[x_negative, 0, 0] = -24.0
    start_pos[x_negative, 0, 1] = start_lateral[x_negative]
    target_pos[x_negative, 0, 0] = 24.0
    target_pos[x_negative, 0, 1] = target_lateral[x_negative]

    start_pos[:, 0, 2] = start_height
    target_pos[:, 0, 2] = target_height

    return {
        "format_version": 1,
        "seed": seed,
        "terrain_seed": terrain_seed,
        "num_envs": num_envs,
        "num_obstacles": 200,
        "start_pos": start_pos,
        "target_pos": target_pos,
    }


def main():
    parser = argparse.ArgumentParser(description="Create a fixed NavRL evaluation set")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain-seed", type=int, default=0)
    args = parser.parse_args()

    scenarios = create_eval_scenarios(args.num_envs, args.seed, args.terrain_seed)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(scenarios, output)
    print(f"Saved {args.num_envs} fixed evaluation scenarios to {output}")


if __name__ == "__main__":
    main()
