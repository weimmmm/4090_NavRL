import torch

from lidar_wam.models.action_expert import (
    ActionOnlyModel, FutureChangeAdapter, FutureConditionedActionModel)
from lidar_wam.models.lidar_video_dit import LiDARVideoDiT


def _inputs(batch=2):
    return {
        "noisy_action": torch.randn(batch, 10, 3),
        "action_timestep": torch.full((batch,), 500.0),
        "current_latent": torch.randn(batch, 4, 27, 5),
        "goal": torch.randn(batch, 4),
        "proprio": torch.randn(batch, 10),
        "past_actions": torch.randn(batch, 30, 3),
        "past_mask": torch.ones(batch, 30),
    }


def test_future_change_adapter_shape():
    adapter = FutureChangeAdapter(width=32)
    tokens = adapter(torch.randn(2, 4, 27, 5), torch.randn(2, 4, 27, 5))
    assert tokens.shape == (2, 135, 32)


def test_zero_gate_exactly_preserves_action_checkpoint():
    torch.manual_seed(7)
    baseline = ActionOnlyModel(
        width=32, depth=2, heads=4, ffn_width=64, past_horizon=30).eval()
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    fused = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        past_horizon=30, future_attention_layers=1).eval()
    state = baseline.state_dict()
    incompatible = fused.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys
    inputs = _inputs()
    observation = baseline.encode_current(inputs["current_latent"])
    expected = baseline.action_expert(
        inputs["noisy_action"], inputs["action_timestep"], observation,
        inputs["goal"], inputs["proprio"], inputs["past_actions"],
        inputs["past_mask"])
    actual = fused.action_expert(
        inputs["noisy_action"], inputs["action_timestep"], observation,
        inputs["goal"], inputs["proprio"], inputs["past_actions"],
        inputs["past_mask"], future_tokens=torch.randn(2, 135, 32))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_gate_is_not_double_zero_locked():
    torch.manual_seed(11)
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    model = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        past_horizon=30, future_attention_layers=1)
    torch.nn.init.normal_(model.action_expert.out.weight, std=0.02)
    inputs = _inputs()
    observation = model.encode_current(inputs["current_latent"])
    output = model.action_expert(
        inputs["noisy_action"], inputs["action_timestep"], observation,
        inputs["goal"], inputs["proprio"], inputs["past_actions"],
        inputs["past_mask"], future_tokens=torch.randn(2, 135, 32))
    output.square().mean().backward()
    gate = model.action_expert.blocks[-1].future_gate
    assert gate.grad is not None
    assert torch.isfinite(gate.grad)
    assert gate.grad.abs() > 0
