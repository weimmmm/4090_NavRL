"""Five-step latent MSE for the frozen history-to-future predictor.

The recursive path feeds each predicted latent back into the five-observation
window. Ground-truth intermediate latents are used only for scoring and for a
separately labelled teacher-forced diagnostic. Every linked transition must be
an existing valid ten-simulator-step latent-cache row.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.epona_history import LidarHistoryMST
from evaluate_representative import fetch
from try_epona_mst_history import FiveHistoryDataset


def select_sequences(data, cache, h5, per_seed, random_seed):
    source = cache["source_index"].astype(np.int64)
    token = fetch(h5, "token", source)
    prev_token = fetch(h5, "prev_token", source)
    scene = fetch(h5, "scene_token", source)
    frame = fetch(h5, "frame_idx", source)
    successor = {}
    for position, key in enumerate(prev_token):
        if key in successor:
            raise ValueError("Duplicate successor token among valid transitions")
        successor[key] = position
    position = {int(index): i for i, index in enumerate(source)}
    candidates = {int(seed): [] for seed in sorted(set(data.seeds))}
    for data_index, first_source in enumerate(data.source):
        first = position[int(first_source)]
        chain = [first]
        for _ in range(4):
            following = successor.get(token[chain[-1]])
            if following is None or scene[following] != scene[first] or \
                    frame[following] != frame[chain[-1]] + 1:
                break
            chain.append(following)
        if len(chain) == 5:
            candidates[int(data.seeds[data_index])].append((data_index, chain))
    rng = np.random.default_rng(random_seed)
    selected = []
    for seed, available in candidates.items():
        for offset in rng.permutation(len(available)):
            data_index, chain = available[offset]
            linked = all(np.array_equal(
                h5["prev_range_values"][int(source[chain[h]])],
                h5["range_values"][int(source[chain[h - 1]])])
                for h in range(1, 5))
            if linked:
                selected.append((data_index, chain))
            if sum(int(data.seeds[i]) == seed for i, _ in selected) == per_seed:
                break
        if sum(int(data.seeds[i]) == seed for i, _ in selected) < per_seed:
            raise ValueError(f"Only {len(available)} five-step candidates for seed {seed}")
    selected.sort(key=lambda item: int(data.source[item[0]]))
    indices = np.asarray([item[0] for item in selected], dtype=np.int64)
    chains = np.asarray([item[1] for item in selected], dtype=np.int64)
    manifest = {"split": "test", "random_seed": random_seed,
                "per_seed": per_seed, "horizons": 5,
                "candidate_counts": {str(seed): len(rows)
                                     for seed, rows in candidates.items()},
                "rows": [{"initial_source_index": int(data.source[i]),
                          "seed": int(data.seeds[i]),
                          "history_source_indices": data.history_source[i].tolist(),
                          "future_source_indices": source[chain].tolist()}
                         for i, chain in selected]}
    return indices, chains, manifest


@torch.no_grad()
def evaluate(model, data, cache, selected, chains, scale, batch_size):
    targets = torch.from_numpy(cache["target"][chains].copy() * scale)
    next_actions = torch.from_numpy(cache["actions"][chains].copy())
    records = []
    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        history_free = data.history[selected[start:stop]].to(stage1.DEVICE)
        history_true = history_free.clone()
        actions_free = data.past_actions[selected[start:stop]].to(stage1.DEVICE)
        actions_true = actions_free.clone()
        truths = targets[start:stop].to(stage1.DEVICE)
        controls = next_actions[start:stop].to(stage1.DEVICE)
        initial = history_free[:, -1].clone()
        batch_mse = []
        for horizon in range(5):
            prediction_free = model(history_free, actions_free)
            prediction_teacher = model(history_true, actions_true)
            truth = truths[:, horizon]
            squared = lambda x: (x - truth).square().mean(dim=(1, 2, 3)).cpu().tolist()
            batch_mse.append({"recursive": squared(prediction_free),
                              "teacher_forced": squared(prediction_teacher),
                              "copy_initial": squared(initial),
                              "copy_previous_gt": squared(history_true[:, -1])})
            if horizon < 4:
                history_free = torch.cat((history_free[:, 1:],
                                          prediction_free.unsqueeze(1)), dim=1)
                history_true = torch.cat((history_true[:, 1:],
                                          truth.unsqueeze(1)), dim=1)
                actions_free = torch.cat((actions_free[:, 1:],
                                          controls[:, horizon:horizon + 1]), dim=1)
                actions_true = torch.cat((actions_true[:, 1:],
                                          controls[:, horizon:horizon + 1]), dim=1)
        for local, index in enumerate(selected[start:stop]):
            records.append({"initial_source_index": int(data.source[index]),
                            "seed": int(data.seeds[index]),
                            "horizons": [{"horizon": h + 1,
                                          **{name: values[local] for name, values in
                                             batch_mse[h].items()}}
                                         for h in range(5)]})
    methods = ("recursive", "teacher_forced", "copy_initial", "copy_previous_gt")
    summary = {}
    for horizon in range(5):
        summary[str(horizon + 1)] = {}
        for label, group in (("all", records), *(
                (f"seed_{seed}", [r for r in records if r["seed"] == seed])
                for seed in sorted(set(data.seeds[selected])))):
            summary[str(horizon + 1)][label] = {"samples": len(group),
                **{method: float(np.mean([r["horizons"][horizon][method]
                                          for r in group])) for method in methods}}
    return {"summary": summary, "rows": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=stage1.OUT /
                        "worldvln_latent_prior_1000")
    parser.add_argument("--per-seed", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = FiveHistoryDataset("test", args.data_root)
    cache = np.load(stage1.OUT / stage1.LATENT_DIR / "test.npz")
    scale = json.loads((stage1.OUT / stage1.LATENT_DIR /
                        "metadata.json").read_text())["scaling_factor"]
    with h5py.File(args.data_root / "navrl_static_test.h5", "r") as h5:
        selected, chains, manifest = select_sequences(data, cache, h5,
                                                       args.per_seed, args.seed)
    saved = torch.load(args.run / "future_latent_best.pt", map_location="cpu",
                       weights_only=False)
    model = LidarHistoryMST(width=saved["width"], blocks=saved["blocks"])
    model.load_state_dict(saved["model"])
    model = model.to(stage1.DEVICE).float().eval()
    report = evaluate(model, data, cache, selected, chains, scale, args.batch_size)
    report.update({"split": "test", "checkpoint_step": saved["step"],
                   "latent_scaling_factor": scale,
                   "metric": "mean squared error per scaled latent element, no physical unit",
                   "recursive": "each predicted latent becomes the newest history frame",
                   "teacher_forced": "intermediate ground-truth latents supplied as history; diagnostic only",
                   "future_action_use": "actions enter the history only after that interval; the model does not see the next interval's actions"})
    stage1.save_json(args.run / "test_autoregressive_5_manifest.json", manifest)
    stage1.save_json(args.run / "test_autoregressive_5.json", report)
    print(json.dumps({"samples": len(selected), "candidate_counts":
                      manifest["candidate_counts"], "summary": report["summary"]}),
          flush=True)


if __name__ == "__main__":
    main()
