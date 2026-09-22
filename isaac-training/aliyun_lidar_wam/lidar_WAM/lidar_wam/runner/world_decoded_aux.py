"""Fine-tune the original NavRL conditional diffusion with decoded-space losses.

The historical two-channel circular VAE and the WorldModel topology are frozen and
unchanged.  Only the WorldModel is optimized.  In addition to epsilon MSE, clean
latent predictions at low-noise timesteps are decoded by the frozen VAE and
supervised in the representation that is ultimately converted to a point cloud.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from lidar_wam.runner import stage1


def squared_chamfer(prediction, target, mask_threshold=1.5):
    """LaGen-style bidirectional mean squared nearest-neighbour distance (m^2)."""
    pred = stage1.to_points(prediction, mask_threshold)
    truth = stage1.to_points(target, 0.0)
    if not len(pred) and not len(truth):
        return 0.0
    if not len(pred) or not len(truth):
        return 200.0
    return float(np.square(cKDTree(pred).query(truth)[0]).mean()
                 + np.square(cKDTree(truth).query(pred)[0]).mean())


class LatentsWithTargets(Dataset):
    """Cached scaled latents plus matching target images from the immutable HDF5."""

    def __init__(self, split, out, source_indices=None, overfit=False):
        cached = np.load(out / stage1.LATENT_DIR / f"{split}.npz")
        meta = json.loads((out / stage1.LATENT_DIR / "metadata.json").read_text())
        scale = float(meta["scaling_factor"])
        all_source = cached["source_index"]
        if source_indices is not None:
            location = {int(v): i for i, v in enumerate(all_source)}
            missing = [int(v) for v in source_indices if int(v) not in location]
            if missing:
                raise ValueError(f"Manifest indices absent from latent cache: {missing[:5]}")
            selection = np.asarray([location[int(v)] for v in source_indices], dtype=np.int64)
        elif overfit:
            selection = np.arange(min(128, len(all_source)), dtype=np.int64)
        else:
            selection = np.arange(len(all_source), dtype=np.int64)

        self.target = torch.from_numpy(cached["target"][selection].copy() * scale)
        self.previous = torch.from_numpy(cached["previous"][selection].copy() * scale)
        self.actions = torch.from_numpy(cached["actions"][selection].copy())
        self.state = torch.from_numpy(stage1.causal_state(cached["state"][selection]))
        self.seeds = cached["seeds"][selection].copy()
        self.indices = all_source[selection].copy()

        # h5py requires increasing fancy indices. Read once in sorted order, then
        # restore manifest order. The full training target tensor is about 1.3 GiB.
        order = np.argsort(self.indices)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
            sorted_images = h5["range_values"][self.indices[order]].astype(np.float32)
        self.image = torch.from_numpy(sorted_images[inverse].copy())

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        return (self.previous[i], self.target[i], self.actions[i], self.state[i],
                self.image[i])


def representative_indices(out, split, limit=None):
    path = out / "representative_baseline" / "sample_manifest.json"
    manifest = json.loads(path.read_text())
    rows = manifest["splits"][split]
    if limit is not None and limit < len(rows):
        # Preserve both seeds and the fixed ordering within each seed.
        seeds = sorted({int(row["seed"]) for row in rows})
        per_seed = max(1, limit // len(seeds))
        selected = [row for seed in seeds
                    for row in [r for r in rows if int(r["seed"]) == seed][:per_seed]]
        # Fill a non-divisible remainder deterministically without duplicating rows.
        chosen = {int(row["source_index"]) for row in selected}
        selected.extend(row for row in rows
                        if int(row["source_index"]) not in chosen)
        rows = selected[:limit]
    return [int(row["source_index"]) for row in rows]


def predicted_x0(noisy, epsilon, timesteps, scheduler):
    alpha_bar = scheduler.alphas_cumprod.to(noisy.device)[timesteps]
    alpha_bar = alpha_bar.view(-1, 1, 1, 1).to(noisy.dtype)
    return ((noisy - (1.0 - alpha_bar).sqrt() * epsilon)
            / alpha_bar.sqrt().clamp_min(1e-6))


def decoded_losses(vae, x0, target_image, scale, mask_threshold, topk, empty_margin):
    """Decoded-space losses with symmetric frame-state supervision.

    Non-empty frames:
      * BCE is computed only on non-empty frames, with pos_weight=4.
      * Presence loss only looks at GT-hit locations, so it cannot be satisfied
        by inventing hits at arbitrary rays.

    Empty frames:
      * A separately averaged all-negative BCE prevents rare empty frames from
        being diluted by the overwhelmingly non-empty training set.
      * Top-k suppression directly penalizes the highest mask logits, because
        a single logit above mask_threshold is enough to create a false point.
    """
    decoded = vae.decode(x0 / scale).sample[:, :, :, :18]
    truth = target_image[:, :, :, :18]
    gt_hit = truth[:, 1:2] > 0
    pred_range = decoded[:, :1]
    pred_mask = decoded[:, 1:2]

    frame_nonempty = gt_hit.flatten(1).any(1)
    frame_empty = ~frame_nonempty
    zero = pred_mask.sum() * 0.0

    nonempty_mask_bce = zero
    empty_mask_bce = zero
    range_loss = zero
    nonempty_presence_loss = zero
    empty_suppression_loss = zero

    if frame_nonempty.any():
        nonempty_mask_bce = F.binary_cross_entropy_with_logits(
            pred_mask[frame_nonempty],
            gt_hit[frame_nonempty].float(),
            pos_weight=pred_mask.new_tensor(4.0),
        )

        hit_error = (pred_range - truth[:, :1]).abs()
        hit_mask = gt_hit & frame_nonempty[:, None, None, None]
        range_loss = hit_error[hit_mask].mean()

        # Crucial: require confident predictions at TRUE hit locations only.
        # The old implementation used top-k over all rays, which could reward
        # spatially wrong false hits as long as the frame was not empty.
        presence_terms = []
        for idx in torch.nonzero(frame_nonempty, as_tuple=False).flatten():
            positive_logits = pred_mask[idx][gt_hit[idx]]
            k_pos = min(topk, int(positive_logits.numel()))
            if k_pos > 0:
                top_positive = positive_logits.topk(k_pos).values.mean()
                presence_terms.append(F.relu(mask_threshold - top_positive))
        if presence_terms:
            nonempty_presence_loss = torch.stack(presence_terms).mean()

    if frame_empty.any():
        empty_logits = pred_mask[frame_empty]
        empty_mask_bce = F.binary_cross_entropy_with_logits(
            empty_logits, torch.zeros_like(empty_logits)
        )

        # Inference uses mask > mask_threshold. Keep the largest logits below
        # a safety level instead of merely below the hard threshold.
        safe_logit = mask_threshold - empty_margin
        k_empty = min(topk, empty_logits[0].numel())
        top_empty = empty_logits.flatten(1).topk(k_empty, dim=1).values
        empty_suppression_loss = F.relu(top_empty - safe_logit).square().mean()

    stats = {
        "aux_nonempty_samples": int(frame_nonempty.sum()),
        "aux_empty_samples": int(frame_empty.sum()),
    }
    return (nonempty_mask_bce, empty_mask_bce, range_loss,
            nonempty_presence_loss, empty_suppression_loss, decoded, stats)


def select_aux_indices(timesteps, target_image, aux_t_max, aux_batch_max,
                       aux_empty_max):
    """Preferentially keep rare empty frames in the decoded-auxiliary subset."""
    eligible = torch.nonzero(timesteps < aux_t_max, as_tuple=False).flatten()
    if not len(eligible):
        return eligible

    gt_hit = target_image[:, 1:2, :, :18] > 0
    frame_empty = ~gt_hit.flatten(1).any(1)

    empty_idx = eligible[frame_empty[eligible]]
    nonempty_idx = eligible[~frame_empty[eligible]]

    def shuffled_take(index, count):
        if count <= 0 or not len(index):
            return index[:0]
        if len(index) <= count:
            return index
        order = torch.randperm(len(index), device=index.device)[:count]
        return index[order]

    empty_take = shuffled_take(empty_idx, min(aux_empty_max, aux_batch_max))
    remaining = aux_batch_max - len(empty_take)
    nonempty_take = shuffled_take(nonempty_idx, remaining)

    selected = torch.cat([empty_take, nonempty_take], dim=0)
    remaining = aux_batch_max - len(selected)

    # If there were unusually many empty frames and too few non-empty frames,
    # fill unused slots with the remaining empty candidates.
    if remaining > 0 and len(empty_idx) > len(empty_take):
        chosen = torch.zeros(len(target_image), dtype=torch.bool,
                             device=timesteps.device)
        chosen[empty_take] = True
        leftover_empty = empty_idx[~chosen[empty_idx]]
        selected = torch.cat(
            [selected, shuffled_take(leftover_empty, remaining)], dim=0
        )

    if len(selected) > 1:
        order = torch.randperm(len(selected), device=selected.device)
        selected = selected[order]
    return selected


@torch.no_grad()
def evaluate_generation(model, vae, dataset, scale, ddim_steps, batch_size,
                        seed, mask_threshold):
    model.eval()
    scheduler = stage1.DDIMScheduler(num_train_timesteps=1000,
                                     prediction_type="epsilon", clip_sample=False)
    rows = []
    for start in range(0, len(dataset), batch_size):
        batch = [value.to(stage1.DEVICE) for value in
                 next(iter(DataLoader(torch.utils.data.Subset(
                     dataset, range(start, min(start + batch_size, len(dataset)))),
                     batch_size=min(batch_size, len(dataset) - start))))]
        prev, target_latent, actions, state, target_image = batch
        generated = stage1.generate(model, scheduler, prev, actions, state,
                                    seed + start, num_steps=ddim_steps)
        prediction = vae.decode(generated / scale).sample.cpu().numpy()
        target_np = target_image.cpu().numpy()
        latent_errors = ((generated - target_latent).square()
                         .flatten(1).mean(1).cpu().numpy())

        for j in range(len(prediction)):
            pred_logits = prediction[j, 1, :, :18]
            pred_hit = pred_logits > mask_threshold
            gt_hit = target_np[j, 1, :, :18] > 0

            gt_empty = bool(not gt_hit.any())
            pred_is_empty = bool(not pred_hit.any())
            false_empty = bool((not gt_empty) and pred_is_empty)
            false_hit = bool(gt_empty and (not pred_is_empty))
            both_nonempty = bool((not gt_empty) and (not pred_is_empty))
            both_empty = bool(gt_empty and pred_is_empty)

            tp = int((pred_hit & gt_hit).sum())
            fp = int((pred_hit & ~gt_hit).sum())
            fn = int((~pred_hit & gt_hit).sum())
            rows.append({
                "source_index": int(dataset.indices[start + j]),
                "seed": int(dataset.seeds[start + j]),
                "cd_paper_m2": squared_chamfer(
                    prediction[j], target_np[j], mask_threshold),
                "latent_mse": float(latent_errors[j]),
                "gt_empty": gt_empty,
                "pred_empty": pred_is_empty,
                "false_empty_frame": false_empty,
                "false_hit_frame": false_hit,
                "both_nonempty": both_nonempty,
                "both_empty": both_empty,
                "gt_hit_count": int(gt_hit.sum()),
                "pred_hit_count": int(pred_hit.sum()),
                "max_mask_logit": float(pred_logits.max()),
                "tp": tp, "fp": fp, "fn": fn,
            })

    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    cds = np.asarray([row["cd_paper_m2"] for row in rows], dtype=np.float64)
    pair_cds = [row["cd_paper_m2"] for row in rows if row["both_nonempty"]]

    gt_empty_frames = sum(row["gt_empty"] for row in rows)
    gt_nonempty_frames = len(rows) - gt_empty_frames
    false_hit_frames = sum(row["false_hit_frame"] for row in rows)
    false_empty_frames = sum(row["false_empty_frame"] for row in rows)
    # For GT-empty frames, the number of erroneous predicted rays matters more
    # than a binary frame-level flag. A single false ray and a dense hallucinated
    # cloud must not be treated as equally bad by checkpoint selection.
    pred_hit_count = sum(row["pred_hit_count"] for row in rows)
    mean_pred_hit_count = pred_hit_count / max(gt_empty_frames, 1)

    return {
        "samples": len(rows),
        "cd_paper_m2": float(cds.mean()),
        "median_cd_paper_m2": float(np.median(cds)),
        "p90_cd_paper_m2": float(np.percentile(cds, 90)),
        "p95_cd_paper_m2": float(np.percentile(cds, 95)),
        "both_nonempty_cd_paper_m2": (
            float(np.mean(pair_cds)) if pair_cds else None),
        "latent_mse": float(np.mean([row["latent_mse"] for row in rows])),
        "mask_precision": tp / max(tp + fp, 1),
        "mask_recall": tp / max(tp + fn, 1),
        "mask_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "gt_empty_frames": gt_empty_frames,
        "gt_nonempty_frames": gt_nonempty_frames,
        "false_hit_frames": false_hit_frames,
        "false_empty_frames": false_empty_frames,
        "pred_hit_count": int(pred_hit_count),
        "mean_pred_hit_count": float(mean_pred_hit_count),
        "false_hit_rate_on_empty": false_hit_frames / max(gt_empty_frames, 1),
        "false_empty_rate_on_nonempty": (
            false_empty_frames / max(gt_nonempty_frames, 1)),
        "rows": rows,
    }


def find_empty_source_indices(out, split, limit=None):
    """Find cached source indices whose GT target mask is completely empty."""
    cached = np.load(out / stage1.LATENT_DIR / f"{split}.npz")
    source = np.asarray(cached["source_index"], dtype=np.int64)
    source = np.sort(source)
    found = []
    with h5py.File(stage1.DATA / f"navrl_static_{split}.h5", "r") as h5:
        for start in range(0, len(source), 4096):
            idx = source[start:start + 4096]
            images = h5["range_values"][idx].astype(np.float32)
            gt_hit = images[:, 1, :, :18] > 0
            empty = ~gt_hit.reshape(len(idx), -1).any(1)
            found.extend(int(v) for v in idx[empty])
            if limit is not None and len(found) >= limit:
                return found[:limit]
    return found if limit is None else found[:limit]


def selection_key(natural_metrics, empty_metrics):
    """Lexicographic checkpoint objective: state correctness before geometry."""
    false_hits = (empty_metrics["false_hit_frames"]
                  if empty_metrics is not None else 0)
    empty_pred_hits = (empty_metrics["pred_hit_count"]
                       if empty_metrics is not None else 0)
    false_empties = natural_metrics["false_empty_frames"]
    geometry = natural_metrics["both_nonempty_cd_paper_m2"]
    if geometry is None:
        geometry = math.inf
    return (int(false_hits), int(empty_pred_hits), int(false_empties),
            float(geometry))


def train(args):
    train_data = LatentsWithTargets("train", args.out, overfit=args.overfit)
    val_indices = representative_indices(args.out, "val", args.val_samples)
    val_data = LatentsWithTargets("val", args.out, source_indices=val_indices)
    empty_val_indices = find_empty_source_indices(
        args.out, "val", args.empty_val_samples)
    empty_val_data = (LatentsWithTargets("val", args.out,
                                        source_indices=empty_val_indices)
                      if empty_val_indices else None)
    meta = json.loads((args.out / stage1.LATENT_DIR / "metadata.json").read_text())
    scale = float(meta["scaling_factor"])
    run_dir = args.out / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "stage": "world_decoded_aux_finetune", "steps": args.steps,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "initial_checkpoint": str(args.init_checkpoint),
        "vae_sha256": meta["vae_sha256"], "scaling_factor": scale,
        "loss": {"epsilon_mse": 1.0, "x0_l1": args.latent_weight,
                 "nonempty_mask_bce": args.mask_weight,
                 "empty_mask_bce": args.empty_mask_weight,
                 "decoded_hit_range_l1": args.range_weight,
                 "nonempty_presence": args.presence_weight,
                 "empty_suppression": args.empty_suppress_weight},
        "aux_t_max": args.aux_t_max, "aux_batch_max": args.aux_batch_max,
        "aux_empty_max": args.aux_empty_max,
        "mask_threshold": args.mask_threshold,
        "state_topk": args.empty_topk, "empty_margin": args.empty_margin,
        "validation_manifest": "representative_baseline/sample_manifest.json",
        "validation_samples": len(val_data),
        "empty_validation_samples": len(empty_val_indices),
        "ddim_steps": args.ddim_steps,
        "selection_metric": (
            "lexicographic: false-hit frames on empty val, predicted hit count "
            "on empty val, false-empty on natural val, then both-nonempty Chamfer"),
        "architecture": "unchanged stage1.WorldModel and frozen circular two-channel VAE",
    }
    stage1.save_json(run_dir / "config.json", config)

    model = stage1.WorldModel().to(stage1.DEVICE).float()
    initial = stage1.load_model(args.init_checkpoint, model)
    vae = stage1.load_circular_vae()
    vae.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = stage1.DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    loader = stage1.infinite(DataLoader(train_data, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True,
                                        num_workers=args.workers, pin_memory=True))
    history = []
    best_key = (math.inf, math.inf, math.inf, math.inf)
    first_step = 1
    if args.resume:
        checkpoint = stage1.load_model(run_dir / "latest.pt", model)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        first_step = int(checkpoint["step"]) + 1
        history_path = run_dir / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        best_path = run_dir / "best_metrics.json"
        if best_path.exists():
            saved_best = json.loads(best_path.read_text())
            saved_key = saved_best.get("selection_key")
            # Do not compare a pre-pred_hit_count 3-tuple with the new
            # four-component objective. A new run directory is recommended,
            # but treating an old key as +inf keeps resume behavior safe.
            if isinstance(saved_key, list) and len(saved_key) == 4:
                best_key = tuple(saved_key)

    initial_metrics_path = run_dir / "initial_validation.json"
    if first_step == 1 and not args.skip_initial_eval:
        initial_metrics = evaluate_generation(
            model, vae, val_data, scale, args.ddim_steps,
            args.eval_batch_size, args.seed, args.mask_threshold)
        initial_empty_metrics = (
            evaluate_generation(
                model, vae, empty_val_data, scale, args.ddim_steps,
                args.eval_batch_size, args.seed + 100000, args.mask_threshold)
            if empty_val_data is not None else None)
        initial_record = {
            "natural": initial_metrics,
            "empty": initial_empty_metrics,
        }
        stage1.save_json(initial_metrics_path, initial_record)
        best_key = selection_key(initial_metrics, initial_empty_metrics)
        stage1.save_json(run_dir / "best_metrics.json", {
            "natural": {k: v for k, v in initial_metrics.items() if k != "rows"},
            "empty": ({k: v for k, v in initial_empty_metrics.items() if k != "rows"}
                      if initial_empty_metrics is not None else None),
            "selection_key": list(best_key),
            "step": 0,
            "external_checkpoint": str(args.init_checkpoint),
        })
        print(json.dumps({
            "initial_validation": {
                "natural": {k: v for k, v in initial_metrics.items() if k != "rows"},
                "empty": ({k: v for k, v in initial_empty_metrics.items()
                           if k != "rows"}
                          if initial_empty_metrics is not None else None),
                "selection_key": list(best_key),
            }}), flush=True)

    deadline = time.monotonic() + args.max_hours * 3600 if args.max_hours else None
    for step in range(first_step, args.steps + 1):
        model.train()
        prev, target, actions, state, target_image = [
            value.to(stage1.DEVICE, non_blocking=True) for value in next(loader)]
        noise = torch.randn_like(target)
        timesteps = torch.randint(0, 1000, (len(target),), device=stage1.DEVICE)
        noisy = scheduler.add_noise(target, noise, timesteps)
        optimizer.zero_grad(set_to_none=True)
        epsilon = model(noisy, prev, actions, state, timesteps)
        epsilon_loss = F.mse_loss(epsilon, noise)

        eligible = select_aux_indices(
            timesteps, target_image, args.aux_t_max, args.aux_batch_max,
            args.aux_empty_max)
        zero = epsilon_loss * 0.0
        latent_loss = nonempty_mask_bce = empty_mask_bce = zero
        range_loss = nonempty_presence_loss = empty_suppression_loss = zero
        aux_stats = {"aux_nonempty_samples": 0, "aux_empty_samples": 0}
        if len(eligible):
            x0 = predicted_x0(
                noisy[eligible], epsilon[eligible], timesteps[eligible], scheduler)
            latent_loss = F.l1_loss(x0, target[eligible])
            (nonempty_mask_bce, empty_mask_bce, range_loss,
             nonempty_presence_loss, empty_suppression_loss, _,
             aux_stats) = decoded_losses(
                vae, x0, target_image[eligible], scale,
                args.mask_threshold, args.empty_topk, args.empty_margin)

        loss = (
            epsilon_loss
            + args.latent_weight * latent_loss
            + args.mask_weight * nonempty_mask_bce
            + args.empty_mask_weight * empty_mask_bce
            + args.range_weight * range_loss
            + args.presence_weight * nonempty_presence_loss
            + args.empty_suppress_weight * empty_suppression_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            record = {
                "step": step, "loss": float(loss),
                "epsilon_mse": float(epsilon_loss),
                "x0_l1": float(latent_loss),
                "nonempty_mask_bce": float(nonempty_mask_bce),
                "empty_mask_bce": float(empty_mask_bce),
                "hit_range_l1_normalized": float(range_loss),
                "nonempty_presence_loss": float(nonempty_presence_loss),
                "empty_suppression_loss": float(empty_suppression_loss),
                "aux_samples": int(len(eligible)),
                **aux_stats,
                "grad_norm": float(grad),
            }
            if stage1.DEVICE.type == "cuda":
                record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            history.append(record)
            print(json.dumps(record), flush=True)

        reached_limit = deadline is not None and time.monotonic() >= deadline
        if step == args.steps or step % args.eval_every == 0 or reached_limit:
            metrics = evaluate_generation(
                model, vae, val_data, scale, args.ddim_steps,
                args.eval_batch_size, args.seed, args.mask_threshold)
            empty_metrics = (
                evaluate_generation(
                    model, vae, empty_val_data, scale, args.ddim_steps,
                    args.eval_batch_size, args.seed + 100000, args.mask_threshold)
                if empty_val_data is not None else None)

            current_key = selection_key(metrics, empty_metrics)
            summary = {
                "natural": {k: v for k, v in metrics.items() if k != "rows"},
                "empty": ({k: v for k, v in empty_metrics.items() if k != "rows"}
                          if empty_metrics is not None else None),
                "selection_key": list(current_key),
                "step": step,
                "initial_checkpoint_step": initial["step"],
            }
            print(json.dumps({"validation": summary}), flush=True)
            stage1.save_json(
                run_dir / f"validation_step_{step}.json",
                {"natural": metrics, "empty": empty_metrics,
                 "selection_key": list(current_key)})

            if not args.no_save_checkpoints:
                stage1.save_model(run_dir / "latest.pt", model, optimizer, step, summary)

            if current_key < best_key:
                best_key = current_key
                if not args.no_save_checkpoints:
                    stage1.save_model(run_dir / "best.pt", model, optimizer, step, summary)
                stage1.save_json(run_dir / "best_metrics.json", summary)

            stage1.save_json(run_dir / "history.json", history)
        if reached_limit:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=stage1.OUT)
    parser.add_argument("--run-name", default="world_circular_decoded_aux")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--empty-val-samples", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--mask-weight", type=float, default=0.1,
                        help="BCE weight for GT non-empty frames")
    parser.add_argument("--empty-mask-weight", type=float, default=0.1,
                        help="Separately averaged all-negative BCE on GT empty frames")
    parser.add_argument("--range-weight", type=float, default=0.05)
    parser.add_argument("--presence-weight", type=float, default=0.02,
                        help="Anti-false-empty hinge on true GT-hit locations only")
    parser.add_argument("--empty-suppress-weight", type=float, default=0.2,
                        help="Frame-level suppression of top mask logits on GT-empty frames")
    parser.add_argument("--aux-t-max", type=int, default=500)
    parser.add_argument("--aux-batch-max", type=int, default=64)
    parser.add_argument("--aux-empty-max", type=int, default=16,
                        help="Reserve decoded-aux slots for rare GT-empty frames")
    parser.add_argument("--empty-topk", type=int, default=8)
    parser.add_argument("--empty-margin", type=float, default=0.5,
                        help="For empty GT, keep top logits below threshold-margin")
    parser.add_argument("--mask-threshold", type=float, default=1.5)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true",
                        help="Diagnostics only: run training/evaluation without multi-GiB checkpoints")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.init_checkpoint = args.init_checkpoint.expanduser().resolve()
    stage1.DATA = args.data_root
    stage1.seed_everything(args.seed)
    print(f"device={stage1.DEVICE}", flush=True)
    train(args)


if __name__ == "__main__":
    main()
