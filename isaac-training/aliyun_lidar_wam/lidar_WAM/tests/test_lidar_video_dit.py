import torch

from lidar_wam.models.action_dit import ActionDiT, JointHistoryWorldActionDiT
from lidar_wam.models.lidar_video_dit import LiDARVideoDiT


def test_video_dit_shape_and_block_causal_mask():
    model = LiDARVideoDiT(width=64, depth=2, heads=4, mlp_ratio=2)
    history = torch.randn(2, 3, 4, 27, 5)
    noisy = torch.randn(2, 4, 27, 5)
    timestep = torch.rand(2)
    output = model(history, noisy, timestep)
    assert output.shape == noisy.shape
    boundary = 3 * 27 * 5
    assert model.block_causal_mask[:boundary, boundary:].all()
    assert not model.block_causal_mask[boundary:, :].any()
    assert not model.block_causal_mask[boundary:, boundary:].any()


def test_video_dit_rejects_wrong_history_length():
    model = LiDARVideoDiT(width=32, depth=1, heads=4, mlp_ratio=2)
    try:
        model(torch.randn(1, 2, 4, 27, 5),
              torch.randn(1, 4, 27, 5), torch.rand(1))
    except ValueError as error:
        assert "history" in str(error)
    else:
        raise AssertionError("two-frame history must be rejected")


def test_shared_history_tokens_match_world_and_action_shapes():
    world = LiDARVideoDiT(width=64, depth=1, heads=4, mlp_ratio=2)
    joint = JointHistoryWorldActionDiT(world, width=64, depth=1,
                                       heads=4, mlp_ratio=2,
                                       shared_world_depth=1)
    history = torch.randn(2, 3, 4, 27, 5)
    noisy_future = torch.randn(2, 4, 27, 5)
    noisy_action = torch.randn(2, 10, 3)
    goal = torch.randn(2, 4)
    proprio = torch.randn(2, 10)
    past = torch.randn(2, 30, 3)
    mask = torch.ones(2, 30)
    world_velocity, action_velocity = joint(
        history, noisy_future, torch.rand(2), noisy_action, torch.rand(2),
        goal, proprio, past, mask)
    assert joint.encode_history_features(history).shape == (2, 405, 64)
    assert world_velocity.shape == noisy_future.shape
    assert action_velocity.shape == noisy_action.shape


def test_action_dit_context_cannot_read_noisy_action():
    model = ActionDiT(width=64, depth=1, heads=4, mlp_ratio=2)
    context = model.context_tokens
    assert model.causal_mask[:context, context:].all()
    assert not model.causal_mask[context:, :].any()
