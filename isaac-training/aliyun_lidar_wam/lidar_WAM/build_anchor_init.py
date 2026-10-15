from pathlib import Path
import copy
import torch

root = Path("/workspace/NavRL/isaac-training/aliyun_lidar_wam/lidar_WAM/outputs")
one_path = root / "action_expert_future_from89_step16000/closed_loop_queue/step_008500.pt"
direct_path = root / "lidar_video_dit_direct3_best_chamfer_m2.pt"
out_path = root / "action_expert_future_anchor_t1_direct3_far_gated_v1/initial_step0.pt"

one = torch.load(one_path, map_location="cpu")
direct = torch.load(direct_path, map_location="cpu")
one_state = one["model"]
direct_state = direct.get("model", direct)

state = {
    key: value for key, value in one_state.items()
    if not key.startswith("future_model.")
}
state.update({f"future_model.{key}": value for key, value in direct_state.items()})
state.update({
    f"anchor_future_model.{key[len('future_model.'):]}": value
    for key, value in one_state.items() if key.startswith("future_model.")
})
state["future_adapter.horizon_gate"] = torch.zeros(2, dtype=torch.float32)

payload = copy.deepcopy(one)
payload["model"] = state
payload["step"] = 0
payload.pop("optimizer", None)
architecture = dict(payload.get("architecture", {}))
direct_arch = dict(direct.get("architecture", {}))
architecture.update({
    "future_width": int(direct_arch.get("width", 512)),
    "future_depth": int(direct_arch.get("depth", 10)),
    "future_heads": int(direct_arch.get("heads", 8)),
    "future_mlp_ratio": float(direct_arch.get("mlp_ratio", 4.0)),
    "future_flow_steps": int(direct_arch.get("sampling_flow_steps", 4)),
    "future_prediction_frames": 3,
    "future_prediction_mode": "direct_joint",
    "future_prediction_target": str(
        direct_arch.get("prediction_target", "absolute_latent")),
    "future_adapter_mode": "temporal_change",
    "anchor_future_enabled": True,
    "anchor_future_prediction_frames": 1,
    "anchor_future_width": int(one["architecture"].get("future_width", 512)),
    "anchor_future_depth": int(one["architecture"].get("future_depth", 10)),
    "anchor_future_heads": int(one["architecture"].get("future_heads", 8)),
    "anchor_future_mlp_ratio": float(
        one["architecture"].get("future_mlp_ratio", 4.0)),
    "anchor_future_prediction_target": str(
        one["architecture"].get("future_prediction_target", "absolute_latent")),
    "anchor_future_flow_steps": int(
        one["architecture"].get("future_flow_steps", 4)),
    "temporal_horizon_gating": "t1_plus_tanh_gate_delta_t2_t3",
    "optimization_scope": "anchor_horizon_gates_only",
    "fusion": "one_way_anchor_t1_plus_temporal_change_future3_to_action_blocks_7_8",
})
payload["architecture"] = architecture
payload["source_checkpoints"] = {
    "one_frame_fusion": str(one_path),
    "direct_three_frame_future": str(direct_path),
}
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, out_path)
print(out_path)
print("state_tensors", len(state))
print("horizon_gate", state["future_adapter.horizon_gate"].tolist())
