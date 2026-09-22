# Changelog

## 2026-09-18 match 100-ms Actor transitions

- Reuse training's 50-tick action hold while preserving 2-ms physics and
  2–4-ms command delivery; evaluate 352 policy transitions per 35.2-s episode.

## 2026-09-17 twelve-value active-command evaluation

- Reuse training's twelve-value state, including the active controller command
  and its delay. The original baseline policy still receives only the first
  eight values; nine-value delay checkpoints are no longer compatible.
- Match training's 512-tick PPO configuration in the shared evaluation config.

## 2026-09-17 active command delay observation

- Reuse training's nine-value policy state and expose the currently applied
  command delay / 4 ms. Keep the older eight-value baseline evaluation path by
  passing only its original eight state values to that policy.

## 2026-09-17 2–4-ms controller reception delay

- Match training's uniform one-or-two-tick command arrival delay, replacing
  the former 62–160 ms distribution. Fixed-delay evaluation now uses 4 ms.

## 2026-09-17 command-only evaluation

- Reuse training's 2-ms command-to-controller delay without odometry delay or
  timing-feature observations. Evaluate new eight-value policies for up to
  17,600 steps; old two-stage-delay weights are incompatible.
- Remove the obsolete timing-feature mismatch ablation and two-slot FIFO
  evaluation configuration. Keep paired seed replay and fixed scenarios.

## Unreleased

- Add a default-off evaluation ablation that perturbs only the policy-visible
  odom and command delay features by continuous bounded jitter (1 ms by
  default). Keep the 2-ms physics clock, odom snapshots, command FIFO and
  physical delay draws unchanged through an independent third random stream;
  report signed, absolute and maximum feature mismatch metrics.
- Reuse training's progress/goal-hold reward, goal diagnostics, and bounded
  0--1 timing features. The physical delay distributions and two-slot FIFO are
  unchanged; retrain 19-feature delay policies before evaluation.
- Reuse training's reference-aligned 100-ms endpoint reward and navigation-step
  PPO discounts (gamma=0.99, lambda=0.95). Remove the unused PPO physical-clock
  constructor branch; retain 2-ms physics/delay scheduling and two-stage FIFO.
- Record `reward_dt`, `discount_dt`, and bounded timing-feature scales in
  evaluation metadata. Train new delay weights.

## Previous physical-tick normalization

- Use the same direct 2 ms feature/reward/PPO time base as training; remove
  `timing.reference_dt` and report `normalization_dt=sim.dt` in evaluation YAML.
  Evaluate newly trained physical-tick-normalized delay-policy weights.

## Previous reference-alignment cleanup

- Remove obsolete `inference_delay` and unused `odom_delay_capacity` config,
  and use training's cleaned odom/command-only sampler and queue interface.
  Reject removed inference configuration rather than silently adding a stage.
  Preserve active evaluation wrappers, checkpoint buffers and the 16 ms
  normalization reference; this value is not the observation period.
- Follow reference `e87db38`: condition the fitted command Weibull on the open
  interval (60,160) ms, recompute both CDF bounds, and quantize to 31–80 ticks
  (62–160 ms). Reject the obsolete fixed `truncation_mass` configuration.
- Match training's two-slot GPU FIFO; keep 2 ms physics, 100 ms navigation,
  46–54 ms odometry and the 19-feature delay-policy checkpoint layout.
- Reuse the training odometry capture window optimization without introducing
  the wheeled robot's 100 Hz locomotion layer into drone control.

## Previous alignment (before reference e87db38)

- Remove unused baseline training/update/optimizer code, the unused utility
  evaluator, Gaussian actor/distribution, minibatch and pattern helpers, and
  the evaluator's old reset alias. Preserve current entry points, baseline
  checkpoint buffers, two-stage timing, GPU FIFO, videos and fixed scenarios.
- Align evaluation with training_delay: 2 ms physics, 100 ms navigation period,
  delayed odometry (46–54 ms), truncated-Weibull command transport (52–200 ms
  after tick quantization), and an eight-slot per-environment GPU FIFO.
- Reuse the training environment, timing buffers and PPO instead of the legacy
  inference-delay/list-queue implementation. Actor and artificial LiDAR delays
  are disabled; command metadata binds the consumed odometry snapshot age.
- Replay independent odom/command random streams for baseline/delay comparisons,
  without affecting global policy/reset random state.
- Support disabled/fixed/random delays; fixed evaluation uses 50 ms odometry
  and 108 ms command transport. Keep the physical episode horizon at 35.2 s.
- Use the current 19-feature delay checkpoint layout; legacy 10-feature delay
  checkpoints require retraining. Baseline input remains eight features.
- Preserve fixed scenarios, first-episode metrics, video output and safe native
  simulator teardown. Historical experiment reports describe the old model.

## Previous legacy evaluator (before reference alignment)

- Reuse `training_delay`'s scoped Isaac startup/device compatibility, honoring
  nonzero CUDA devices without editing bundled third-party source files.
- Evaluate inference/command transport delays at 1 ms physics resolution while
  retaining the legacy policy's 16 ms floor-quantized timing observations.
- Generate reproducible fixed scenarios without starting Isaac Sim; reset the
  timing generator so baseline and delay-policy comparisons share a schedule.
- Use the bundled baseline PPO and configurable legacy 10-/19-feature delay
  observations; report first-episode statistics and measured-duration video FPS.
- Keep generated datasets, evaluation results, videos and checkpoints out of
  the code submission. Generate the default dataset with
  `python scripts/create_eval_env.py --num-envs 1024` before evaluation.
- This legacy inference/command evaluation module remained separate from the
  reference-aligned odometry/command environment in `training_delay`.
