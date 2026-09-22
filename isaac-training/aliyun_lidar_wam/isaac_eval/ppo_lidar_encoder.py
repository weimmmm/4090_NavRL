"""LiDAR encoder used by the PPO expert that collected ``wam_data``."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class AzimuthCircularConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 5), stride=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.azimuth_padding = self.kernel_size[1] // 2
        self.elevation_padding = self.kernel_size[0] // 2

    def forward(self, value):
        value = F.pad(
            value, (self.azimuth_padding, self.azimuth_padding, 0, 0),
            mode="circular")
        value = F.pad(
            value, (0, 0, self.elevation_padding, self.elevation_padding),
            mode="constant")
        return super().forward(value)


class RangeImageEncoder(nn.Module):
    def __init__(self, lidar_range):
        super().__init__()
        lidar_range = float(lidar_range)
        if not math.isfinite(lidar_range) or lidar_range <= 0:
            raise ValueError("lidar_range must be finite and positive")
        self.register_buffer(
            "lidar_range", torch.tensor(lidar_range, dtype=torch.float32))
        self.network = nn.Sequential(
            AzimuthCircularConv2d(1, 4), nn.ELU(),
            AzimuthCircularConv2d(4, 16, stride=(1, 2)), nn.ELU(),
            AzimuthCircularConv2d(16, 16, stride=(2, 2)), nn.ELU(),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(128), nn.LayerNorm(128),
        )

    def forward(self, range_image):
        value = (2 * range_image / self.lidar_range - 1).clamp(-1, 1)
        return self.network(value)

