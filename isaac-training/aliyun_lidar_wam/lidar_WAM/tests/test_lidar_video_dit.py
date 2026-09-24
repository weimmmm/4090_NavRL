import torch

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

