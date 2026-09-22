"""Point-cloud sensitivity check for the selected five-frame MST adapter."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_wam.runner import stage1
from lidar_wam.runner.epona_history import LidarHistoryMST
from try_epona_mst_history import FiveHistoryDataset, pointcloud_scores


class ShuffleOlder(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, history, past_actions):
        history, past_actions = history.clone(), past_actions.clone()
        history[:, :-1] = torch.roll(history[:, :-1], shifts=1, dims=0)
        past_actions[:, :-1] = torch.roll(past_actions[:, :-1], shifts=1, dims=0)
        return self.adapter(history, past_actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=stage1.OUT /
                        "epona_probe" / "mst_history_1000")
    parser.add_argument("--per-seed", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    stage1.DATA = args.data_root
    unet = stage1.WorldModel().to(stage1.DEVICE).float().eval()
    stage1.load_model(stage1.OUT / "world_circular_causal_8h" / "best.pt", unet)
    adapter = LidarHistoryMST(width=256, blocks=2).to(stage1.DEVICE).eval()
    checkpoint = torch.load(args.out / "mst_5_best.pt", map_location="cpu",
                            weights_only=False)
    adapter.load_state_dict(checkpoint["model"])
    methods = {"mst_5": adapter, "mst_5_older_shuffled": ShuffleOlder(adapter).eval()}
    threshold = json.loads((stage1.OUT / stage1.CIRCULAR_VAE_DIR /
                            "oracle_val.json").read_text())["selected_threshold"]
    for split in ("val", "test"):
        data = FiveHistoryDataset(split, args.data_root, args.per_seed)
        result = pointcloud_scores(unet, methods, data, split, args.data_root,
                                   args.raw_root, args.batch_size, threshold)
        result["shuffle_definition"] = "older four observed frames and past commands rolled within each batch; latest frame, its past actions, and future ten actions unchanged"
        result["mst_checkpoint_step"] = checkpoint["step"]
        stage1.save_json(args.out / f"{split}_history_sensitivity.json", result)
        print(json.dumps({"split": split,
                          "paired_n": result["summary"]["all"]["paired_nonempty"],
                          "paired_cd_paper_m2": result["summary"]["all"]["paired_cd_paper_m2"]}),
              flush=True)


if __name__ == "__main__":
    main()
