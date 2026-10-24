# Derived from MobileVLA-R1 llava/model/multimodal_projector/base_projector.py
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# New input guards: copyright reserved. See UPSTREAM-LICENSE-APACHE-2.0.txt.
import math

import torch
from torch import nn


class DownSampleBlock(nn.Module):
    """Keep source flatten/transpose order exactly, including odd-grid padding."""

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected [frames, square-patches, vision-hidden]")
        side = math.isqrt(x.shape[1])
        if not side or side * side != x.shape[1]:
            raise ValueError("Downsample requires a nonempty square patch grid")
        x = x.reshape(x.shape[0], side, side, x.shape[2])
        n, w, h, c = x.size()
        if w % 2:
            x = torch.cat([x, x.new_zeros(n, 1, h, c)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2:
            x = torch.cat([x, x.new_zeros(n, w, 1, c)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.view(n, w, h // 2, c * 2)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, h // 2, w // 2, c * 4)
        return x.reshape(n, -1, c * 4)


class MobileVLAProjector(nn.Module):
    """Source parameter names layers.{1,2,4}; TP=1 only in this first adapter."""

    def __init__(self, vision_hidden=1152, text_hidden=4096):
        super().__init__()
        self.layers = nn.Sequential(
            DownSampleBlock(),
            nn.LayerNorm(vision_hidden * 4),
            nn.Linear(vision_hidden * 4, text_hidden),
            nn.GELU(),
            nn.Linear(text_hidden, text_hidden),
        )

    def forward(self, x):
        return self.layers(x)
