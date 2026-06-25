"""Stage 2 (actual geometry): visual OBB + walls + SKU -> corrected 9-DoF OBB.

The learned replacement for the design doc's classical Stage-2 optimiser. For
each Stage-1 box (in the base_link / container frame) it consumes:

    * the visual OBB           (center 3, size 3, rot6d 6),
    * the container walls       (left / right / bottom plane normal + offset),
    * the scene SKU candidates  (padded set of sorted W/H/L dimension priors),

and predicts the full **actual** 9-DoF OBB: occluded depth filled in, sizes
snapped toward the matching SKU, pose nudged to respect the container. A small
cross-attention from the box query to the SKU tokens implements the learned,
rotation-invariant SKU matching described in the doc.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Stage2GeometryNet(nn.Module):
    """Learned scene/SKU-aware refinement of a visual OBB to the actual OBB.

    Args:
        d_model: token / trunk width.
        max_sku: max number of SKU candidates per scene (padded).
        center_delta_scale: max metric magnitude of the center residual (m).
    """

    def __init__(
        self,
        d_model: int = 128,
        max_sku: int = 8,
        center_delta_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_sku = max_sku
        self.center_delta_scale = center_delta_scale

        # Box query from the visual OBB (12 = center3 + size3 + rot6d6).
        self.box_embed = nn.Sequential(
            nn.Linear(12, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        # SKU token from [sorted dims (3), valid (1)].
        self.sku_embed = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        # Wall context from 3 walls x [normal (3), offset (1), valid (1)].
        self.wall_embed = nn.Sequential(
            nn.Linear(3 * 5, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.trunk = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )
        self.head_center = nn.Linear(d_model, 3)
        self.head_logsize = nn.Linear(d_model, 3)
        self.head_rot = nn.Linear(d_model, 6)

        # Initialise as (near) identity refinement: output ~= visual OBB.
        for head in (self.head_center, self.head_logsize, self.head_rot):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        visual: dict[str, Tensor],
        walls: Tensor,
        skus: Tensor,
    ) -> dict[str, Tensor]:
        """Refine the visual OBB into the actual OBB.

        Args:
            visual: dict with ``center`` ``[B, 3]``, ``size`` ``[B, 3]``,
                ``rot6d`` ``[B, 6]`` (base_link frame).
            walls: ``[B, 3, 5]`` left/right/bottom (normal 3, offset 1, valid 1).
            skus: ``[B, max_sku, 4]`` sorted dims (3) + valid flag (1).

        Returns:
            dict with refined ``center`` ``[B, 3]``, ``size`` ``[B, 3]``
            (positive) and ``rot6d`` ``[B, 6]``.
        """
        center = visual["center"]
        size = visual["size"]
        rot6d = visual["rot6d"]
        box_vec = torch.cat([center, size, rot6d], dim=-1)  # [B, 12]
        box_tok = self.box_embed(box_vec)  # [B, d]

        sku_tok = self.sku_embed(skus)  # [B, S, d]
        # Cross-attention: box query attends over SKU tokens. Masked-out
        # (invalid) SKUs get -inf logits so they don't contribute.
        valid = skus[..., 3]  # [B, S]
        q = self.q_proj(box_tok).unsqueeze(1)  # [B, 1, d]
        k = self.k_proj(sku_tok)  # [B, S, d]
        v = self.v_proj(sku_tok)  # [B, S, d]
        logits = (q * k).sum(-1) / (self.d_model**0.5)  # [B, S]
        logits = logits.masked_fill(valid < 0.5, float("-inf"))
        # Scenes with no valid SKU: fall back to uniform attention.
        no_valid = (valid.sum(-1, keepdim=True) < 0.5)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.where(no_valid, torch.zeros_like(attn), attn)
        sku_ctx = (attn.unsqueeze(-1) * v).sum(1)  # [B, d]

        wall_ctx = self.wall_embed(walls.flatten(1))  # [B, d]

        trunk = self.trunk(torch.cat([box_tok, sku_ctx, wall_ctx], dim=-1))

        center_out = (
            center + torch.tanh(self.head_center(trunk)) * self.center_delta_scale
        )
        size_out = size * torch.exp(self.head_logsize(trunk).clamp(-2.0, 2.0))
        rot_out = rot6d + self.head_rot(trunk)
        return {
            "center": center_out,
            "size": size_out.clamp_min(1e-3),
            "rot6d": rot_out,
        }
