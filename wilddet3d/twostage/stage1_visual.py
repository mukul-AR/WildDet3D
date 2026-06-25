"""Stage 1 (visual geometry): RGB + Depth crop -> visible 9-DoF OBB.

A compact convolutional encoder over the per-box 4-channel (RGB + metric
depth) crop, fused with a small geometric context vector (2D bbox, intrinsics,
median crop depth, back-projected anchor center). It regresses the box's
visible 9-DoF OBB in the camera frame:

    center (3, residual on the anchor) + size (3, positive) + rot6d (6)
    + presence logit (1).

This stands in for WildDet3D's heavy SAM3 + DINOv2 + dense head Stage 1 so the
full two-stage workflow is runnable without vis4d / pretrained weights; it
exposes the same per-box visual-OBB interface that Stage 2 consumes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _ConvBlock(nn.Module):
    """conv -> bn -> relu, optionally strided."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, 3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Stage1VisualNet(nn.Module):
    """RGB-D crop encoder -> visual 9-DoF OBB (camera frame).

    Args:
        in_ch: input channels (4 = RGB + depth).
        ctx_dim: dimensionality of the geometric context vector.
        width: base channel width.
        feat_dim: fused trunk feature dimension.
        center_delta_scale: max metric magnitude of the center residual (m).
    """

    def __init__(
        self,
        in_ch: int = 4,
        ctx_dim: int = 10,
        width: int = 32,
        feat_dim: int = 256,
        center_delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.center_delta_scale = center_delta_scale

        self.stem = _ConvBlock(in_ch, width, stride=1)
        self.enc = nn.Sequential(
            _ConvBlock(width, width * 2, stride=2),
            _ConvBlock(width * 2, width * 2, stride=1),
            _ConvBlock(width * 2, width * 4, stride=2),
            _ConvBlock(width * 4, width * 4, stride=1),
            _ConvBlock(width * 4, width * 8, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(width * 8 + 64, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        self.head_center = nn.Linear(feat_dim, 3)
        self.head_size = nn.Linear(feat_dim, 3)
        self.head_rot = nn.Linear(feat_dim, 6)
        self.head_presence = nn.Linear(feat_dim, 1)

        # Identity-ish rotation init (first two rows of I).
        nn.init.zeros_(self.head_rot.weight)
        self.head_rot.bias.data = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        )
        nn.init.zeros_(self.head_center.weight)
        nn.init.zeros_(self.head_center.bias)

    def forward(
        self, crop: Tensor, ctx: Tensor, anchor_center: Tensor
    ) -> dict[str, Tensor]:
        """Predict the visual OBB for each crop.

        Args:
            crop: ``[B, in_ch, H, W]`` RGB-D crops.
            ctx: ``[B, ctx_dim]`` geometric context.
            anchor_center: ``[B, 3]`` back-projected approximate center (m).

        Returns:
            dict with ``center`` ``[B, 3]``, ``size`` ``[B, 3]`` (positive),
            ``rot6d`` ``[B, 6]`` and ``presence`` ``[B]``.
        """
        feat = self.pool(self.enc(self.stem(crop))).flatten(1)
        ctx_feat = self.ctx_mlp(ctx)
        trunk = self.trunk(torch.cat([feat, ctx_feat], dim=-1))

        delta = torch.tanh(self.head_center(trunk)) * self.center_delta_scale
        center = anchor_center + delta
        size = nn.functional.softplus(self.head_size(trunk)) + 1e-3
        rot6d = self.head_rot(trunk)
        presence = self.head_presence(trunk).squeeze(-1)
        return {
            "center": center,
            "size": size,
            "rot6d": rot6d,
            "presence": presence,
        }
