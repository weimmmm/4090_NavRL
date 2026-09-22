# Command-delay policy evaluation

Evaluation reuses `training_delay`'s 2-ms physics, 100-ms Actor transition
(50 physical ticks), current-state twelve-value observation and
command-to-controller delivery model. Values 9–11 are the currently applied
command in the goal frame; value 12 is that command's transport delay divided
by 4 ms. Each Actor output has its own 2-ms or 4-ms transport time while the
controller holds its last received command. Stale late arrivals are ignored.
There is no odom delay, timing-feature mismatch switch or two-slot FIFO.

The default evaluation uses the fixed 1,024-scenario dataset, deterministic
mean actions, and 352 policy transitions (17,600 physical ticks, 35.2 s
maximum per episode).
Video recording is off by default. Paired baseline and command-delay policies
replay the same sampled delay stream when evaluated with the same seed. The
older eight-value baseline sees only the first eight state values.

Run inside Isaac Sim 2023 from this directory:

```bash
/isaac-sim/python.sh scripts/eval_random_delay.py \
  eval.policy=delay delay_checkpoint=/path/to/new_checkpoint.pt gpu_id=0
```

The new policy has a twelve-value state and requires retraining; older eight-,
nine- and 19-value command-delay checkpoints are incompatible. To compare against an
original eight-value baseline policy, supply `baseline_checkpoint` and use
`eval.policy=both`.

`timing.enabled=false` disables the command delay. With
`timing.randomize_in_eval=false`, command delay is fixed at 4 ms.
