import math

import torch
from torch import nn
from torch.nn import functional as F


def min_pool_ranges(distances, horizontal_sample, vertical_sample):
    """Keep the nearest return in each nonoverlapping angular block."""
    if horizontal_sample <= 0 or vertical_sample <= 0:
        raise ValueError("LiDAR sample factors must be positive")
    horizontal, vertical = distances.shape[-2:]
    if horizontal % horizontal_sample or vertical % vertical_sample:
        raise ValueError("Raw LiDAR dimensions must be divisible by sample factors")
    blocks = distances.reshape(
        *distances.shape[:-2],
        horizontal // horizontal_sample, horizontal_sample,
        vertical // vertical_sample, vertical_sample,
    )
    return blocks.amin(dim=(-3, -1))


class AzimuthCircularConv2d(nn.Conv2d):
    """Wrap azimuth only for inputs shaped [batch, channel, elevation, azimuth]."""

    def __init__(self, in_channels, out_channels, kernel_size=(3, 5), stride=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.azimuth_padding = self.kernel_size[1] // 2
        self.elevation_padding = self.kernel_size[0] // 2

    def forward(self, x):
        # LaGen's circular convolution uses distinct padding for the two axes.
        x = F.pad(x, (self.azimuth_padding, self.azimuth_padding, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.elevation_padding, self.elevation_padding), mode="constant")
        return super().forward(x)


class RangeImageEncoder(nn.Module):
    """Encode metric range images [batch, 1, elevation, azimuth] into 128 features."""

    def __init__(self, lidar_range):
        super().__init__()
        lidar_range = float(lidar_range)
        if not math.isfinite(lidar_range) or lidar_range <= 0:
            raise ValueError("lidar_range must be finite and positive")
        self.register_buffer("lidar_range", torch.tensor(lidar_range, dtype=torch.float32))
        self.network = nn.Sequential(
            AzimuthCircularConv2d(1, 4), nn.ELU(),
            AzimuthCircularConv2d(4, 16, stride=(1, 2)), nn.ELU(),
            AzimuthCircularConv2d(16, 16, stride=(2, 2)), nn.ELU(),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(128), nn.LayerNorm(128),
        )

    def forward(self, range_image):
        # Keep observations in meters; normalize [0, range] to [-1, 1] for the CNN.
        range_input = (2 * range_image / self.lidar_range - 1).clamp(-1, 1)
        return self.network(range_input)
