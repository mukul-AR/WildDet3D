"""Dense conv detection head: fused FPN feature map -> per-cell predictions.

Output channels (per FPN cell):
    heatmap : 1   objectness logit (focal-trained center heatmap)
    reg     : 12  [du, dv, log_z, log_w, log_h, log_l, r0..r5]
                  du,dv  : sub-cell 2D center offset (units of FPN stride)
                  log_z  : log metric depth of the box center (camera frame)
                  log_*  : log box size (w, h, l)
                  r0..r5 : 6D continuous rotation (camera frame)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

REG_CHANNELS = 12


class DenseConvHead(nn.Module):
    """Shared conv tower with objectness + 9-DoF regression branches.

    Args:
        in_ch: input feature channels (fused FPN level, 256 for SAM3).
        feat_ch: conv tower width.
        n_convs: number of 3x3 conv-GN-ReLU blocks per branch.
        prior_prob: focal-loss bias prior for the heatmap (CenterNet/RetinaNet).
    """

    def __init__(
        self,
        in_ch: int = 256,
        feat_ch: int = 256,
        n_convs: int = 4,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()

        def tower() -> nn.Sequential:
            layers: list[nn.Module] = []
            c = in_ch
            for _ in range(n_convs):
                layers += [
                    nn.Conv2d(c, feat_ch, 3, padding=1, bias=False),
                    nn.GroupNorm(32, feat_ch),
                    nn.ReLU(inplace=True),
                ]
                c = feat_ch
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.reg_tower = tower()
        self.heatmap = nn.Conv2d(feat_ch, 1, 3, padding=1)
        self.reg = nn.Conv2d(feat_ch, REG_CHANNELS, 3, padding=1)

        # Focal prior bias on the objectness logit.
        nn.init.constant_(
            self.heatmap.bias, -math.log((1 - prior_prob) / prior_prob)
        )
        # Rotation init -> identity (first two rows of I); other reg -> 0.
        nn.init.zeros_(self.reg.weight)
        nn.init.zeros_(self.reg.bias)
        with torch.no_grad():
            self.reg.bias[6:12] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            )

    def forward(self, feat: Tensor) -> dict[str, Tensor]:
        """Args: feat ``[B, in_ch, Hf, Wf]``. Returns heatmap + reg maps."""
        return {
            "heatmap": self.heatmap(self.cls_tower(feat)),  # [B, 1, Hf, Wf]
            "reg": self.reg(self.reg_tower(feat)),  # [B, 12, Hf, Wf]
        }
