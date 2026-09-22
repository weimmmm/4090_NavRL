# NWM CDiT predictor for NavRL LiDAR

This experiment copies the official [Navigation World Models](https://github.com/facebookresearch/nwm) CDiT backbone and diffusion implementation into `third_party/nwm/` (commit `3f6cd8e`). The copied upstream files are unchanged. `lidar_wam/runner/nwm_predictor.py` adapts only the inputs and output shape. NWM's RGB checkpoint **cannot** be used as a LiDAR checkpoint; the CDiT is initialized from scratch. The existing circular VAE checkpoint and cached latents remain unchanged.

| Input | NavRL representation |
| --- | --- |
| Current observation | Circular VAE latent `[4,27,5]` |
| Executed actions | Ten separate normalized world-frame velocity commands `[10,3]` |
| Previous state | Eleven values from `prev_drone_state[2:13]` |
| Prediction target | Next scaled VAE latent `[4,27,5]`, 0.16 s later |

The adapter adds one wraparound row along the 360-degree angle axis and one zero column on the vertical axis, giving a `[4,28,6]` latent that NWM's 2×2 patch embedding accepts. Only the original `[4,27,5]` region is decoded for evaluation. Action tokens preserve all ten steps in order. The input uses no next-frame pose or state. Seed splits and finite-executed-action filtering reuse `ExecutedLatents`; training stats come only from training seeds.

Use the existing PPU environment and latent cache:

```bash
cd /mnt/workspace/wam_trining/trining/lidar_WAM
export LIDAR_WAM_DATA_ROOT=/mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache
PY=/mnt/workspace/LaGen/.venv-ppu/bin/python

# One-step integration smoke test, not a model-quality experiment.
$PY scripts/run_nwm.py train --overfit --model CDiT-B/2 --steps 1 --batch-size 2
$PY scripts/run_nwm.py evaluate --overfit --split val --samples 2 --ddim-steps 4

# Optional 128-example fit check; run only when ready to train.
$PY scripts/run_nwm.py train --overfit --model CDiT-B/2 --steps 200 --batch-size 8
$PY scripts/run_nwm.py train --model CDiT-B/2 --precision bf16 --steps 5000 --batch-size 256 --eval-every 500
$PY scripts/run_nwm.py evaluate --split val --samples 256 --ddim-steps 20
```

The `--overfit` and full runs have separate output directories. Checkpoints, validation diffusion losses, point-cloud Chamfer results, the copy-frame baseline, the action-shuffle control and comparison PNGs are saved below `outputs/world_nwm_cdit_{overfit,full}/`. A one-step checkpoint only demonstrates that data loading, training, sampling and decoding work; its numerical prediction error is not evidence of model quality. Compare a trained model on held-out seeds before claiming improvement.

The NWM source is distributed under CC BY-NC 4.0; see `third_party/nwm/LICENSE.md` before using the copied source in a commercial project.

For this PPU, a short benchmark measured approximately 0.381 s/update at batch 256 with FP32 and 0.199 s/update with BF16. BF16 uses about 10.2 GiB peak allocated memory at batch 256. Batch 512 improves samples/second slightly but nearly doubles time per optimizer update, so the full run uses batch 256. There is no speed benefit in filling all 98 GB of memory.

## Integration result (2026-09-16)

The `CDiT-B/2` PPU forward/backward, a real-data training update, checkpoint reload, NWM DDIM sampling, VAE decode and point-cloud evaluation all ran successfully. A 128-example fit check completed 200 updates: its validation diffusion loss fell from about `0.979` at update 1 to `0.169` at update 200. On 16 **training** examples at update 200 with 20 DDIM steps, copy-frame Chamfer was `0.155 m`, NWM prediction Chamfer was `3.195 m`, and shuffled-action prediction Chamfer was `3.262 m`. Thus the code is integrated but this short fit check **does not** show useful LiDAR prediction or action conditioning. These values are in `outputs/world_nwm_cdit_overfit/evaluation_train_n16.json`; they must not be presented as held-out evaluation.

## Full training result (2026-09-17)

The full run used 60,368 finite executed-action training transitions, `CDiT-B/2` (194,616,608 parameters), BF16, batch 256, AdamW at `8e-5` and 5,000 updates. The best checkpoint was update 5,000, selected using NWM diffusion loss across 256 validation transitions (`0.033349`; its noise MSE component was `0.032290`). The model predicts the next latent directly from pure noise and samples with 20 DDIM steps. There is no future-state input.

| Held-out split (128 transitions per seed) | Copy-frame Chamfer | NWM Chamfer | Shuffled-action Chamfer | NWM relative to copy | Shuffle degradation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation seeds 16–17 | 0.207 m | 0.416 m | 0.453 m | 100.8% worse | 9.0% worse |
| Test seeds 18–19 | 0.242 m | 0.419 m | 0.444 m | 73.4% worse | 5.9% worse |

This run **fails the phase-one criterion** requiring at least 10% lower Chamfer than copying the previous frame, even though the action shuffle raises error by more than 5% on both splits. Only about 8.6–14.1% of samples beat the copy baseline within each held-out seed. Diffusion noise loss is not a distance in metres, and it cannot substitute for point-cloud evaluation.

The final configuration, training log, `best.pt`, `latest.pt`, reports and comparison PNGs are under `outputs/world_nwm_cdit_full/`. The precise 256-sample reports are `evaluation_val_step5000_n256.json` and `evaluation_test_step5000_n256.json`.
