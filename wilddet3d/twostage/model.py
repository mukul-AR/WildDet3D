"""Two-stage 9-DoF model: Stage-1 visual -> frame transform -> Stage-2 actual.

Stage 1 predicts the visible OBB in the camera frame from an RGB-D crop. The
OBB is transformed into the container (base_link) frame using the per-box
extrinsics, then Stage 2 refines it into the actual OBB using the scene walls
and SKU candidates. The two stages are trained jointly, each with its own
9-DoF loss; Stage-2 is fed the (optionally detached + occlusion-augmented)
Stage-1 output so it learns to be robust to realistic Stage-1 error.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from wilddet3d.twostage.rotation_utils import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from wilddet3d.twostage.stage1_visual import Stage1VisualNet
from wilddet3d.twostage.stage2_geometry import Stage2GeometryNet


def transform_obb_cam_to_base(
    visual_cam: dict[str, Tensor], t_base_cam: Tensor
) -> dict[str, Tensor]:
    """Transform a camera-frame OBB into the base_link frame.

    Args:
        visual_cam: dict with ``center`` ``[B, 3]``, ``size`` ``[B, 3]``,
            ``rot6d`` ``[B, 6]`` in the camera frame.
        t_base_cam: ``[B, 4, 4]`` camera-to-base transforms.

    Returns:
        dict with the same keys in the base_link frame (size unchanged).
    """
    r_bc = t_base_cam[:, :3, :3]  # [B, 3, 3]
    t_bc = t_base_cam[:, :3, 3]  # [B, 3]
    center_base = (
        torch.bmm(r_bc, visual_cam["center"].unsqueeze(-1)).squeeze(-1) + t_bc
    )
    r_cam = rotation_6d_to_matrix(visual_cam["rot6d"])  # [B, 3, 3]
    r_base = torch.bmm(r_bc, r_cam)
    return {
        "center": center_base,
        "size": visual_cam["size"],
        "rot6d": matrix_to_rotation_6d(r_base),
    }


class TwoStage9DoF(nn.Module):
    """Full two-stage 9-DoF box predictor.

    Args:
        stage1_kwargs: kwargs forwarded to :class:`Stage1VisualNet`.
        stage2_kwargs: kwargs forwarded to :class:`Stage2GeometryNet`.
        detach_stage1: if True, detach the Stage-1 OBB before Stage 2 so the
            Stage-2 loss does not backprop into Stage 1.
    """

    def __init__(
        self,
        stage1_kwargs: dict | None = None,
        stage2_kwargs: dict | None = None,
        detach_stage1: bool = True,
    ) -> None:
        super().__init__()
        self.stage1 = Stage1VisualNet(**(stage1_kwargs or {}))
        self.stage2 = Stage2GeometryNet(**(stage2_kwargs or {}))
        self.detach_stage1 = detach_stage1

    @staticmethod
    def _occlusion_augment(visual: dict[str, Tensor]) -> dict[str, Tensor]:
        """Simulate partial observability: shrink a random size axis + jitter.

        Forces Stage 2 to rely on the SKU prior / walls to recover the actual
        geometry rather than copying the (here deliberately corrupted) Stage-1
        size. Applied to the Stage-2 *input* only (never to a target).
        """
        size = visual["size"]
        b = size.shape[0]
        device = size.device
        scale = torch.ones_like(size)
        axis = torch.randint(0, 3, (b,), device=device)
        factor = 0.35 + 0.6 * torch.rand(b, device=device)  # [0.35, 0.95]
        scale[torch.arange(b, device=device), axis] = factor
        center = visual["center"] + 0.02 * torch.randn_like(visual["center"])
        return {
            "center": center,
            "size": (size * scale).clamp_min(1e-3),
            "rot6d": visual["rot6d"] + 0.02 * torch.randn_like(visual["rot6d"]),
        }

    def forward(
        self,
        batch: dict[str, Tensor],
        teacher_visual_base: dict[str, Tensor] | None = None,
        teacher_force_p: float = 0.0,
        occlusion_aug: bool = False,
    ) -> dict[str, dict[str, Tensor]]:
        """Run both stages.

        Args:
            batch: dict with ``crop`` ``[B, 4, H, W]``, ``ctx`` ``[B, ctx]``,
                ``anchor_center`` ``[B, 3]`` (camera frame), ``T_base_cam``
                ``[B, 4, 4]``, ``walls`` ``[B, 3, 5]``, ``skus``
                ``[B, S, 4]``.
            teacher_visual_base: optional GT visual OBB (base frame) used to
                teacher-force the Stage-2 input with prob ``teacher_force_p``.
            teacher_force_p: probability of teacher forcing (train only).
            occlusion_aug: apply occlusion augmentation to the Stage-2 input.

        Returns:
            dict with ``stage1`` (camera frame), ``stage1_base`` (base frame)
            and ``stage2`` (base frame) OBB dicts.
        """
        visual_cam = self.stage1(
            batch["crop"], batch["ctx"], batch["anchor_center"]
        )
        visual_base = transform_obb_cam_to_base(
            visual_cam, batch["T_base_cam"]
        )

        s2_in = visual_base
        if self.detach_stage1:
            s2_in = {k: v.detach() for k, v in s2_in.items()}
        if (
            self.training
            and teacher_visual_base is not None
            and teacher_force_p > 0.0
            and torch.rand(()) < teacher_force_p
        ):
            s2_in = {k: v.clone() for k, v in teacher_visual_base.items()}
        if self.training and occlusion_aug:
            s2_in = self._occlusion_augment(s2_in)

        actual = self.stage2(s2_in, batch["walls"], batch["skus"])
        return {
            "stage1": visual_cam,
            "stage1_base": visual_base,
            "stage2": actual,
        }
