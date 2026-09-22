# Epona-inspired NavRL probe

This is a small controlled experiment on the existing circular VAE and LaGen-style UNet, **not** a reproduction of Epona's camera-video architecture or weights. The original HDF5 and `world_circular_causal_8h/best.pt` were not changed.

## Action-to-pose and LiDAR reprojection

`scripts/try_epona_motion.py` fits a ridge residual model on the training maps' finite **executed world-frame commands** (`action_sequence`) and previous 13-dimensional drone state. It predicts the next position, yaw, velocity, and yaw rate from the ten commands, recursively using only its own predicted state after the first frame. The nominal component integrates previous velocity and yaw rate for 0.16 seconds. Ridge strength was selected only on validation-map one-step position error. Test-map future state is read only for metrics and the diagnostic true-pose projection.

The LiDAR was attached with yaw-only orientation during collection. The predicted drone pose is therefore converted to a yaw-only sensor transform before reprojection. On the 512 fixed test starts, the constructed true-pose first-step transform agreed with the HDF5 `prev_trans_mat` to at most `7.26e-6 m` translation and `7.15e-7 rad` rotation. The ray projection itself uses the calibrated ray directions, not the misleading azimuth labels.

Validation one-step position MAE was **0.00550 m** for the fitted model versus **0.01438 m** for constant velocity. On the 512 held-out test trajectories, the recursively predicted position MAE was **0.00797 / 0.06655 / 0.13195 m** at horizons 1 / 5 / 10. The same constant-velocity baseline had **0.01704 / 0.33087 / 0.92101 m** error.

As an input-sensitivity check, shuffling the validation commands among transitions raised the fitted model's position MAE from **0.00550 m** to **0.06927 m**. This shows that the fitted predictor uses its command input, but it is not a counterfactual flight experiment.

The table uses the *same starts and only jointly nonempty point clouds* for every method. It reports bidirectional **squared** Chamfer (paper formula), in **m²**. The fitted pose and true-pose diagnostics both reproject the **initial observed cloud** at each horizon, so they cannot reveal surfaces unseen initially.

| Horizon | Paired n | Original UNet autoregression | Copy initial | Constant-velocity pose | Action-predicted pose | True pose diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0.662 | 0.573 | 0.458 | 0.458 | 0.458 |
| 5 | 510 | 1.941 | 3.087 | 1.660 | **1.325** | 1.298 |
| 10 | 505 | 3.439 | 4.464 | 3.456 | **2.492** | 2.448 |

There were 7 original-UNet empty-cloud cases at horizon 10, 3 for predicted-pose reprojection, and 2 for copying. They are excluded only in the paired table and counted in the full metrics. The predicted-pose method is near its true-pose projection diagnostic at horizons 5 and 10; the remaining gap is mostly not pose error. This does not establish that reprojection can replace a generative model: disocclusion and new LiDAR returns are unavailable in the initial cloud.

Artifacts: `outputs/epona_probe/motion_ridge.npz`, `motion_fit.json`, `test_motion_rollout.json`, and `paired_comparison.json`. The latter has per-seed paired metrics. Reproduce with:

```bash
python scripts/try_epona_motion.py --data-root /mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache --raw-root /mnt/workspace/wam_trining/data/navrl_static_100k
python scripts/summarize_epona_probe.py
```

## Two-frame generated-history fine-tuning

`scripts/try_epona_chain.py` starts from `world_circular_causal_8h/best.pt` (step 10,200). It uses the *actual* 20-step DDIM sampler to generate frame 1 without gradients or target-frame data, then trains the ordinary DDPM epsilon objective for frame 2 conditioned on that generated latent. An equal-weight original one-step objective is retained. Ego features are frozen at the initial observed state in both validation and training, intentionally isolating generated-history feedback from state rollout. This is an Epona-inspired experiment, not Epona's flow-matching Chain-of-Forward formula.

With FP32, batch 16, learning rate `2e-6`, and 100 updates, the fixed 64-pair validation set gave:

| Fine-tune update | Two-frame Chamfer m² | Mask F1 |
| ---: | ---: | ---: |
| 0 (original checkpoint) | **4.164** | **0.8266** |
| 25 | 4.265 | 0.8149 |
| 50 | 4.332 | 0.8129 |
| 75 | 4.187 | 0.8112 |
| 100 | 4.195 | 0.8083 |

Exactly one empty cloud in every summary carries the evaluation's `200 m²` penalty. Removing that penalty gives **1.056 m²** for the original and **1.087 m²** at update 100 over the remaining 63 samples. This small probe did not improve validation quality, so no fine-tuned checkpoint was selected and test seed was not used to pick a model. It does not rule out larger or different rollout-training methods.

Artifacts: `outputs/epona_probe/chain_100/chain_val_manifest.json`, `chain_original_val.json`, `chain_latest_val.json`, `chain_history.json`, and `chain_config.json`. Reproduce with:

```bash
python scripts/try_epona_chain.py --data-root /mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache --raw-root /mnt/workspace/wam_trining/data/navrl_static_100k --out outputs/epona_probe/chain_100 --steps 100 --batch-size 16 --eval-batch-size 16 --val-per-seed 32 --lr 2e-6 --eval-every 25
```

The action-to-pose model consumes executed world-frame command setpoints, whereas the current UNet uses normalized PPO actions. An action expert must emit a defined command representation, or an explicit conversion is needed before these components can be connected. The next model experiment should condition a LiDAR correction/generation model on the causal, action-predicted reprojection and quantify newly visible regions separately.

## Geometry-guided diffusion sampling

The user requires diffusion to remain the LiDAR generator. `scripts/try_epona_guided_diffusion.py` therefore leaves the original UNet and VAE weights untouched. It predicts the next sensor pose from the ten executed commands, reprojects the observed LiDAR, encodes the reprojection with the circular VAE, adds DDIM-schedule noise, and runs the existing UNet for **20 denoising steps**. The final LiDAR is always decoded from the diffusion output. The original previous-frame latent, normalized actions, and causal ego features still condition the UNet. No ground-truth future pose or future frame enters the generator.

On the fixed representative subset, validation and test each used 64 examples per seed (128 total). Sampling strength `0.35` was selected from `0.35 / 0.6 / 0.8` using validation Chamfer only. The next table uses the same jointly nonempty samples for every method; squared two-way Chamfer is in **m²**.

| Split | Paired n | Copy | Predicted-pose warp | Pure-noise diffusion | Geometry-guided diffusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 126 | 0.589 | **0.465** | 0.736 | 0.635 |
| Test | 127 | 0.560 | **0.461** | 0.897 | 0.652 |

On test, guided diffusion also improved mask F1 from **0.868** to **0.877** and true-hit range MAE from **0.761 m** to **0.732 m** compared with pure-noise diffusion. It still loses to direct warping on point-cloud Chamfer. Thus geometry helps the diffusion sampler, but inference-only initialization does not yet produce a better point cloud than the geometric estimate. Training the diffusion predictor with a causal warped latent as an explicit condition is the next experiment; this probe does not claim it will succeed.

Artifacts: `outputs/epona_probe/guided_diffusion/validation.json`, `test.json`, `summary.json`, and `paired_summary.json`. The 8-sample-per-split code smoke test is separate in `guided_smoke/`. Reproduce with:

```bash
python scripts/try_epona_guided_diffusion.py --data-root /mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache --raw-root /mnt/workspace/wam_trining/data/navrl_static_100k --per-seed 64 --batch-size 16
python scripts/summarize_epona_guided.py
```

## Five-frame autoregressive guided diffusion

`scripts/evaluate_epona_guided_5.py` uses the fixed 512 test-map starts from the original ten-frame report. At each of five horizons it reads the next recorded ten-command chunk, recursively predicts the sensor pose, warps the **initial observed** LiDAR, and uses the encoded warp to initialize 20-step DDIM sampling at strength `0.35`. The previous **generated** latent is fed into the next diffusion prediction. The UNet's ego feature remains the initial causal feature, matching the original autoregressive report. Future ground-truth images and poses are used only for scoring; no model weights were changed.

The table reports the squared bidirectional point-cloud Chamfer in **m²** on samples where all three predictions have nonempty clouds. Each row compares the same starts and target frames. Frame 5 is approximately 0.8 seconds (50 simulation steps) after the initial observation.

| Predicted frame | Paired n | Original pure-noise diffusion | Geometry-guided diffusion | Direct predicted-pose warp |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0.662 | 0.598 | **0.458** |
| 2 | 512 | 0.969 | 0.792 | **0.708** |
| 3 | 512 | 1.297 | 1.018 | **0.891** |
| 4 | 510 | 1.584 | 1.296 | **1.136** |
| 5 | 510 | 1.941 | 1.567 | **1.325** |

At frame 5, guidance lowers Chamfer by **19.3%** against the original diffusion but remains **18.2%** above direct warping. The mask F1 is 0.700 / 0.726 / 0.661 for original diffusion / guided diffusion / direct warping, respectively, over all 512 starts. The original diffusion has two empty-cloud outputs at frame 5, so its all-sample mean includes the evaluator's empty-cloud penalty; the paired Chamfer table avoids that penalty. Per-seed frame-5 paired Chamfer is 1.689 / 1.426 / 1.522 on seed 18 and 2.194 / 1.707 / 1.128 on seed 19, in the same method order. The geometry-only ranking varies by map, while guided diffusion improves over pure noise on both.

The earlier one-step `0.897 / 0.652 / 0.461 m²` comparison used a separate 127-sample test subset. Its values should not be mixed with this 512-start autoregressive comparison.

Artifact: `outputs/epona_probe/guided_diffusion/test_autoregressive_5.json` contains per-sample and per-seed metrics for every horizon. Reproduce with:

```bash
python scripts/evaluate_epona_guided_5.py --data-root /mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache --raw-root /mnt/workspace/wam_trining/data/navrl_static_100k
```

## Three- and five-observation MST history adapter

`third_party/epona/stt.py` and its small dependencies are copied from the [official Epona source](https://github.com/Kevin-thu/Epona), with package-import and device compatibility edits documented in `third_party/epona/UPSTREAM.md`. `lidar_wam/runner/epona_history.py` uses Epona's `CausalTimeSpaceBlock` directly: causal temporal attention across observations followed by spatial attention over 135 VAE patches and one past-action token. The five-observation window includes the current observed LiDAR and four earlier frames (0.64 seconds of history); the three-observation window includes the current frame and two earlier frames. The associated past ten-action chunks are observed history. The **next** ten actions still enter the original UNet, and 20-step DDIM still generates the final LiDAR.

This is a LiDAR adaptation of the MST blocks, **not** the full Epona VisDiT or its trained weights. The MST maps its final-frame tokens to a residual correction of the previous latent. The circular VAE and existing step-10,200 UNet are frozen. Only the MST adapter is trained with the original DDPM epsilon loss, FP32, AdamW, batch 8, 1,000 updates per history length. The two window lengths share model initialization and minibatch order. The best checkpoint is chosen independently by validation noise MSE: step 200 for 3 frames and step 800 for 5 frames.

Links are formed using `prev_token`, `token`, `scene_token`, and consecutive `frame_idx`, and all five predecessor transitions must exist in the original valid 10-action latent cache. There are 71,615 training examples with five valid predecessor observations. Validation and test each use fixed seed 42 to draw 256 examples from each of their two held-out map seeds, then score all methods on the same target frame and diffusion noise.

| Split | Jointly nonempty | Single-frame UNet | 3-frame MST + UNet | 5-frame MST + UNet | 5-frame older history shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 510 | 0.7133 | 0.7089 | **0.7014** | 0.7071 |
| Test | 507 | 0.7085 | 0.7095 | **0.7071** | 0.7088 |

Entries are bidirectional **squared** Chamfer, in **m²**. Shuffling the four older observations and their past commands within each batch leaves the newest observation and future action chunk fixed. On test, 5-frame history improves Chamfer over the original by only **0.00137 m² (0.19%)**; shuffling older history increases its Chamfer by **0.00167 m²**. Test-map seed 18 shows 0.70893 / 0.70851 / 0.70926 and seed 19 shows 0.70810 / 0.70578 / 0.70837 for single-frame / intact 5-frame / shuffled 5-frame. The older frames therefore affect output, but the benefit is very small. A per-sample bootstrap stratified by map seed gives a descriptive 95% interval of **[-0.00264, 0.00581] m²** for single-frame minus intact five-frame Chamfer; it crosses zero and does not account for overlap among nearby trajectory samples. The test does not establish reliable improvement.

On all 512 test cases, 5-frame MST changes mask F1 from 0.86608 to 0.86625 and true-hit radial MAE from 0.78210 m to 0.77643 m. Both methods have five empty-cloud cases; the paired Chamfer table excludes them. Test noise MSE is 0.038873 for single-frame, 0.038865 for 5-frame MST, and 0.038889 after older-history shuffling. This is a sensitivity diagnostic, not a counterfactual action test.

Artifacts: `outputs/epona_probe/mst_history_1000/` has the fixed sample manifests, config, adapter checkpoints, training curves, per-sample/per-seed point-cloud metrics, and history-shuffle metrics. Reproduce with:

```bash
python scripts/try_epona_mst_history.py --data-root /mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache --raw-root /mnt/workspace/wam_trining/data/navrl_static_100k --out outputs/epona_probe/mst_history_1000 --steps 1000 --batch-size 8 --eval-batch-size 16 --eval-every 200 --val-per-seed 256 --test-per-seed 256
python scripts/evaluate_epona_mst_shuffle.py --data-root /mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache --raw-root /mnt/workspace/wam_trining/data/navrl_static_100k
```
