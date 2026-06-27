"""Match Stage-1 predicted visible boxes to GT for end-to-end Stage-2 training.

In ``real`` (non-teacher-forced) training, Stage 2 is fed Stage-1's *predicted*
visible boxes. Each predicted box is matched to its nearest GT visible box (by
camera-frame center distance); the matched GT supplies Stage 2's targets
(assignment + actual center/size/rotation). Predicted boxes with no GT within
``thresh`` are dropped (no Stage-2 supervision). Rotation is inherited from the
*predicted* box, so Stage 2 learns to place given imperfect inputs.
"""
from __future__ import annotations

import torch
from torch import Tensor


def match_predicted_to_gt(
    pred_centers: Tensor, gt_centers: Tensor, thresh: float = 0.15
) -> tuple[Tensor, Tensor]:
    """Greedy nearest-center match of predicted boxes to GT visible boxes.

    Args:
        pred_centers: predicted visible centers ``[M, 3]`` (camera frame).
        gt_centers: GT visible centers ``[N, 3]``.
        thresh: max center distance (m) to count as a match.

    Returns:
        ``keep`` bool mask ``[M]`` (predicted boxes with a GT within ``thresh``)
        and ``gt_idx`` long ``[M]`` (index of the nearest GT for each predicted
        box; meaningless where ``keep`` is False).
    """
    if pred_centers.shape[0] == 0 or gt_centers.shape[0] == 0:
        m = pred_centers.shape[0]
        return (
            torch.zeros(m, dtype=torch.bool, device=pred_centers.device),
            torch.zeros(m, dtype=torch.long, device=pred_centers.device),
        )
    d = torch.cdist(pred_centers, gt_centers)  # [M, N]
    nnd, gt_idx = d.min(dim=1)
    keep = nnd < thresh
    return keep, gt_idx
