"""Bounded, short GPU memory probe for the current Flow-only training path."""
import argparse
import gc
import json

import torch

from lidar_wam.runner import super_action_expert as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="16,32,64")
    parser.add_argument("--memory-fraction", type=float, default=.16)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    device = torch.device("cuda:0")
    vae, scale = runner._load_frozen_vae(runner.DEFAULT_VAE_CHECKPOINT, device)
    stats = {"goal_mean": [0.] * 4, "goal_std": [1.] * 4,
             "proprio_mean": [0.] * 10, "proprio_std": [1.] * 10}
    for size in map(int, args.batches.split(",")):
        model = optimizer = batch = loss = None
        try:
            torch.cuda.reset_peak_memory_stats()
            model = runner.FlowActionExpertModel().to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95))
            batch = (torch.zeros(size, 2, 108, 20), torch.ones(size, 10, 3) * .5,
                     torch.zeros(size, 4), torch.zeros(size, 10), torch.zeros(size, 4))
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = runner._loss_batch(
                    model, vae, batch, scale, torch.full((1, 1, 3), 2., device=device),
                    stats, device, "bf16", 1., 0., 0., 1., .5, 4., 2., 3., 0.)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            torch.cuda.synchronize()
            print(json.dumps({"batch_per_gpu": size, "state": "ok", "loss": float(loss),
                              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                              "reserved_gib": torch.cuda.memory_reserved() / 2**30}), flush=True)
        except torch.cuda.OutOfMemoryError as error:
            print(json.dumps({"batch_per_gpu": size, "state": "oom", "error": str(error)}), flush=True)
            break
        finally:
            model = optimizer = batch = loss = None
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
