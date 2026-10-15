import torch

from lidar_wam.models.action_expert import (
    ActionOnlyModel, FutureChangeAdapter, FutureConditionedActionModel,
    LiDARObservationEncoder)
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


def test_direct_future_adapter_uses_only_predicted_latent():
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    model = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        future_attention_layers=1, future_adapter_mode="direct").eval()
    current = torch.randn(2, 4, 27, 5)
    prediction = torch.randn_like(current)
    tokens = model.adapt_predicted_future(current, prediction)
    assert tokens.shape == (2, 135, 32)
    assert model.future_adapter.net[0].in_channels == 4
    torch.testing.assert_close(
        tokens, model.adapt_predicted_future(current + 10, prediction),
        rtol=0, atol=0)
    assert not torch.equal(
        tokens, model.adapt_predicted_future(current, prediction + 1))


def test_direct_future_adapter_receives_reconstructed_residual():
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    future.prediction_target = "normalized_residual"
    scale = torch.tensor([0.5, 1.0, 1.5, 2.0]).reshape(1, 4, 1, 1)
    future.register_buffer("prediction_residual_scale", scale, persistent=False)
    model = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        future_attention_layers=1, future_adapter_mode="direct").eval()
    history = torch.randn(2, 3, 4, 27, 5)
    noise = torch.randn(2, 4, 27, 5)
    adapter_inputs = []
    hook = model.future_adapter.register_forward_pre_hook(
        lambda module, inputs: adapter_inputs.append(inputs))
    try:
        tokens = model.encode_predicted_future(history, noise, steps=1)
    finally:
        hook.remove()
    assert tokens.shape == (2, 135, 32)
    assert len(adapter_inputs[0]) == 1
    torch.testing.assert_close(adapter_inputs[0][0], history[:, -1] + noise * scale)


def test_direct_training_loader_checkpoint_and_frozen_backbones():
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from lidar_wam.runner.action_expert_future import _checkpoint, _load_model

    baseline = ActionOnlyModel(
        width=32, depth=2, heads=4, ffn_width=64, past_horizon=30)
    torch.nn.init.normal_(baseline.action_expert.out.weight, std=0.02)
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    with tempfile.TemporaryDirectory(prefix="future-direct-test-") as directory:
        action_path = Path(directory) / "action.pt"
        future_path = Path(directory) / "future.pt"
        torch.save({"model": baseline.state_dict(), "architecture": {
            "width": 32, "depth": 2, "heads": 4, "ffn_width": 64,
            "past_horizon": 30}}, action_path)
        torch.save({"model": future.state_dict(), "architecture": {
            "width": 32, "depth": 1, "heads": 4, "mlp_ratio": 2.0}}, future_path)
        model, action_payload, future_payload, _, _ = _load_model(
            action_path, future_path, torch.device("cpu"))
        assert model.future_adapter_mode == "direct"
        assert model.future_adapter.net[0].in_channels == 4
        assert not any(p.requires_grad for p in model.future_model.parameters())
        assert not any(p.requires_grad for p in model.observation.parameters())
        assert not model.action_expert.out.weight.requires_grad
        inputs = _inputs()
        tokens = model.encode_predicted_future(
            torch.randn(2, 3, 4, 27, 5), torch.randn(2, 4, 27, 5), steps=1)
        model.action_expert(
            inputs["noisy_action"], inputs["action_timestep"],
            model.encode_current(inputs["current_latent"]), inputs["goal"],
            inputs["proprio"], inputs["past_actions"], inputs["past_mask"],
            future_tokens=tokens).square().mean().backward()
        gates = [block.future_gate for block in model.action_expert.blocks]
        assert all(g.grad is not None and torch.isfinite(g.grad) for g in gates)
        assert any(g.grad.abs() > 0 for g in gates)
        assert all(p.grad is None for p in model.future_model.parameters())
        assert all(p.grad is None for p in model.observation.parameters())
        optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
        args = SimpleNamespace(
            action_heads=4, action_ffn_width=64, future_mlp_ratio=2.0,
            future_steps=1, future_horizon=1, action_checkpoint=action_path,
            future_checkpoint=future_path)
        payload = _checkpoint(
            model, optimizer, 0, {}, args, {}, action_payload, future_payload)
        assert payload["architecture"]["future_adapter_mode"] == "direct"
        restored = FutureConditionedActionModel(
            LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0),
            width=32, depth=2, heads=4, ffn_width=64, past_horizon=30,
            future_attention_layers=2,
            future_adapter_mode=payload["architecture"]["future_adapter_mode"])
        restored.load_state_dict(payload["model"], strict=True)
        torch.testing.assert_close(
            model.adapt_predicted_future(inputs["current_latent"], inputs["current_latent"]),
            restored.adapt_predicted_future(inputs["current_latent"], inputs["current_latent"]))


def test_three_future_change_adapter_shape_and_order():
    torch.manual_seed(5)
    adapter = FutureChangeAdapter(width=32)
    current = torch.randn(2, 4, 27, 5)
    future = torch.randn(2, 3, 4, 27, 5)
    tokens = adapter(current, future)
    swapped = adapter(current, future[:, [1, 0, 2]])
    assert tokens.shape == (2, 405, 32)
    assert not torch.equal(tokens, swapped)


def test_three_future_predictions_are_autoregressive():
    torch.manual_seed(9)
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    model = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        past_horizon=30, future_attention_layers=1).eval()
    history = torch.randn(2, 3, 4, 27, 5)
    noise = torch.randn(2, 3, 4, 27, 5)
    calls = []
    original = model.predict_next

    def record(value, initial_noise, steps=4, gradient_steps=None):
        calls.append(value.clone())
        return original(value, initial_noise, steps, gradient_steps)

    model.predict_next = record
    tokens = model.encode_predicted_future(history, noise, steps=2)
    assert tokens.shape == (2, 405, 32)
    assert len(calls) == 3
    torch.testing.assert_close(calls[0], history)
    torch.testing.assert_close(calls[1][:, :2], history[:, 1:])
    torch.testing.assert_close(calls[2][:, 0], history[:, 2])


def test_predict_next_reconstructs_normalized_residual_latent():
    torch.manual_seed(13)
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    future.prediction_target = "normalized_residual"
    scale = torch.tensor([0.5, 1.0, 1.5, 2.0]).reshape(1, 4, 1, 1)
    future.register_buffer(
        "prediction_residual_scale", scale, persistent=False)
    model = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        past_horizon=30, future_attention_layers=1).eval()
    history = torch.randn(2, 3, 4, 27, 5)
    residual_noise = torch.randn(2, 4, 27, 5)

    # LiDARVideoDiT's zero-initialized output head leaves the initial flow
    # state unchanged, so this isolates the residual-to-absolute conversion.
    predicted = model.predict_next(history, residual_noise, steps=2)
    expected = history[:, -1] + residual_noise * scale
    torch.testing.assert_close(predicted, expected)


def test_predict_next_preserves_absolute_latent_semantics():
    torch.manual_seed(17)
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    future.prediction_target = "absolute_latent"
    model = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        past_horizon=30, future_attention_layers=1).eval()
    history = torch.randn(2, 3, 4, 27, 5)
    initial_noise = torch.randn(2, 4, 27, 5)

    predicted = model.predict_next(history, initial_noise, steps=2)
    torch.testing.assert_close(predicted, initial_noise)


def test_direct_three_residuals_reference_last_observed_frame():
    future = LiDARVideoDiT(
        width=32, depth=1, heads=4, mlp_ratio=2.0, prediction_frames=3)
    future.prediction_target = "normalized_residual"
    scale = torch.tensor([0.5, 1.0, 1.5, 2.0]).reshape(1, 4, 1, 1)
    future.register_buffer("prediction_residual_scale", scale, persistent=False)
    model = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        past_horizon=30, future_attention_layers=1).eval()
    history = torch.randn(2, 3, 4, 27, 5)
    noise = torch.randn(2, 3, 4, 27, 5)
    predicted = model.predict_next(history, noise, steps=1)
    assert predicted.shape == noise.shape
    torch.testing.assert_close(
        predicted, history[:, -1, None] + noise * scale[:, None])


def test_residual_autoregression_feedback_and_adapter_use_absolute_latents():
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    future.prediction_target = "normalized_residual"
    scale = torch.tensor([0.5, 1.0, 1.5, 2.0]).reshape(1, 4, 1, 1)
    future.register_buffer("prediction_residual_scale", scale, persistent=False)
    model = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        past_horizon=30, future_attention_layers=1).eval()
    history = torch.randn(2, 3, 4, 27, 5)
    noise = torch.randn(2, 3, 4, 27, 5)
    calls, adapter_inputs = [], []
    original = model.predict_next

    def record(value, initial_noise, steps=4, gradient_steps=None):
        calls.append(value.clone())
        return original(value, initial_noise, steps, gradient_steps)

    model.predict_next = record
    hook = model.future_adapter.register_forward_pre_hook(
        lambda module, inputs: adapter_inputs.append(inputs[1].clone()))
    try:
        tokens = model.encode_predicted_future(history, noise, steps=1)
    finally:
        hook.remove()
    expected = history[:, -1, None] + (noise * scale[:, None]).cumsum(1)
    assert tokens.shape == (2, 405, 32)
    torch.testing.assert_close(calls[1][:, -1], expected[:, 0])
    torch.testing.assert_close(calls[2][:, -1], expected[:, 1])
    torch.testing.assert_close(adapter_inputs[0], expected)


def test_residual_anchor_uses_anchor_model_scale():
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    anchor = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    anchor.prediction_target = "normalized_residual"
    scale = torch.tensor([0.25, 0.5, 1.0, 2.0]).reshape(1, 4, 1, 1)
    anchor.register_buffer("prediction_residual_scale", scale, persistent=False)
    model = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        past_horizon=30, future_attention_layers=1,
        anchor_future_model=anchor).eval()
    history = torch.randn(2, 3, 4, 27, 5)
    noise = torch.randn(2, 4, 27, 5)
    torch.testing.assert_close(model.predict_next(history, noise, steps=1), noise)
    torch.testing.assert_close(
        model.predict_anchor_next(history, noise, steps=1),
        history[:, -1] + noise * scale)


def test_residual_restoration_preserves_training_gradients():
    future = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2.0)
    future.prediction_target = "normalized_residual"
    future.register_buffer(
        "prediction_residual_scale", torch.ones(1, 4, 1, 1), persistent=False)
    model = FutureConditionedActionModel(
        future, width=32, depth=2, heads=4, ffn_width=64,
        past_horizon=30, future_attention_layers=1, freeze_future_model=False)
    history = torch.randn(2, 3, 4, 27, 5)
    noise = torch.randn(2, 4, 27, 5)
    model.predict_next(history, noise, steps=2, gradient_steps=1).square().mean().backward()
    grad = future.output_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_history_three_observation_encoder_shape_and_order():
    torch.manual_seed(3)
    encoder = LiDARObservationEncoder(width=32, history_frames=3)
    history = torch.randn(2, 3, 4, 27, 5)
    tokens = encoder(history)
    swapped = encoder(history[:, [1, 0, 2]])
    assert tokens.shape == (2, 405, 32)
    assert not torch.equal(tokens, swapped)


def test_history_three_action_only_model_keeps_other_inputs_unchanged():
    model = ActionOnlyModel(
        width=32, depth=2, heads=4, ffn_width=64, past_horizon=30,
        observation_history_frames=3)
    inputs = _inputs()
    velocity, world = model(
        inputs["noisy_action"], inputs["action_timestep"],
        torch.randn(2, 3, 4, 27, 5), inputs["goal"], inputs["proprio"],
        inputs["past_actions"], inputs["past_mask"])
    assert velocity.shape == (2, 10, 3)
    assert world is None


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
