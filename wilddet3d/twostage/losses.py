"""Symmetry-aware 9-DoF OBB loss (center + size + rotation [+ presence]).

Follows the WildDet3D design doc loss formulation:
    * center (cx, cy, cz): smooth-L1 in metric (meter) space.
    * size (w, h, l): L1 in log-space (scale-invariant across SKU sizes).
    * rotation (6D continuous): cuboid symmetry-aware chordal distance
      (min over the 4-element D2 proper-rotation group); chordal is the
      safe, differentiable alternative to geodesic (no acos cliff).
    * presence (optional): BCE objectness/validity gate.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from wilddet3d.twostage.rotation_utils import (
    cuboid_symmetry_rotation_6d,
    rotation_6d_to_matrix,
    symmetry_min_geodesic,
)


def symmetry_chordal_loss(d6_pred: Tensor, d6_gt: Tensor) -> Tensor:
    """Per-sample min chordal (squared Frobenius) rotation loss ``[N]``.

    Minimised over the cuboid symmetry group so the model is not penalised for
    predicting a physically identical but differently-labelled rotation.
    """
    r_pred = rotation_6d_to_matrix(d6_pred)  # [N, 3, 3]
    variants = cuboid_symmetry_rotation_6d(d6_gt)  # [N, 4, 6]
    n, k, _ = variants.shape
    r_var = rotation_6d_to_matrix(variants.reshape(n * k, 6)).reshape(
        n, k, 3, 3
    )
    diff = r_pred.unsqueeze(1) - r_var  # [N, 4, 3, 3]
    chordal = diff.pow(2).sum(dim=(-1, -2))  # [N, 4]
    return chordal.min(dim=1).values  # [N]


class NineDoFLoss(nn.Module):
    """Weighted 9-DoF regression loss with cuboid symmetry-aware rotation.

    Args:
        w_center: weight on the center smooth-L1 term.
        w_size: weight on the log-size L1 term.
        w_rot: weight on the symmetry-aware chordal rotation term.
        w_presence: weight on the optional presence BCE term.
        size_eps: floor added before log to keep sizes positive.
    """

    def __init__(
        self,
        w_center: float = 1.0,
        w_size: float = 1.0,
        w_rot: float = 1.0,
        w_presence: float = 1.0,
        size_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.w_center = w_center
        self.w_size = w_size
        self.w_rot = w_rot
        self.w_presence = w_presence
        self.size_eps = size_eps
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=0.1)

    def forward(
        self,
        pred: dict[str, Tensor],
        target: dict[str, Tensor],
        weight: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the weighted loss and its components.

        Args:
            pred: dict with ``center`` ``[N, 3]``, ``size`` ``[N, 3]``,
                ``rot6d`` ``[N, 6]`` and optionally ``presence`` ``[N]``.
            target: dict with ``center`` ``[N, 3]``, ``size`` ``[N, 3]``,
                ``rot6d`` ``[N, 6]`` and optionally ``presence`` ``[N]``.
            weight: optional per-sample weight ``[N]`` (e.g. GT confidence).

        Returns:
            dict of scalar tensors: ``total``, ``center``, ``size``, ``rot``,
            ``presence`` and (detached) metric ``rot_deg``.
        """
        n = pred["center"].shape[0]
        device = pred["center"].device
        if weight is None:
            weight = torch.ones(n, device=device)
        wsum = weight.sum().clamp_min(1.0)

        center_l = (
            self.smooth_l1(pred["center"], target["center"]).sum(-1) * weight
        ).sum() / wsum

        pred_logsz = torch.log(pred["size"].clamp_min(self.size_eps))
        tgt_logsz = torch.log(target["size"].clamp_min(self.size_eps))
        size_l = (
            (pred_logsz - tgt_logsz).abs().sum(-1) * weight
        ).sum() / wsum

        rot_per = symmetry_chordal_loss(pred["rot6d"], target["rot6d"])
        rot_l = (rot_per * weight).sum() / wsum

        out: dict[str, Tensor] = {
            "center": center_l,
            "size": size_l,
            "rot": rot_l,
        }
        total = (
            self.w_center * center_l
            + self.w_size * size_l
            + self.w_rot * rot_l
        )

        if "presence" in pred and "presence" in target:
            presence_l = nn.functional.binary_cross_entropy_with_logits(
                pred["presence"], target["presence"].float()
            )
            out["presence"] = presence_l
            total = total + self.w_presence * presence_l

        out["total"] = total
        with torch.no_grad():
            out["rot_deg"] = symmetry_min_geodesic(
                pred["rot6d"], target["rot6d"]
            ).mean() * (180.0 / torch.pi)
        return out
