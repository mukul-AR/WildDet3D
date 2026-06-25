"""Prompt-free dense 3D detection package (design-doc Track A "core surgery").

Replaces WildDet3D's text/prompt-conditioned grounding decoder with a dense
convolutional detection head over the depth-fused SAM3 FPN features. The head
predicts, per FPN cell, an objectness heatmap plus a 12-scalar 9-DoF OBB
(CenterNet-3D style: 2D sub-cell offset + metric depth + log-size + 6D
rotation). No text encoder, no point/box prompts, no per-object input.
"""

from __future__ import annotations

from wilddet3d.dense.head import DenseConvHead
from wilddet3d.dense.loss import DenseDet3DLoss
from wilddet3d.dense.model import DenseDet3D

__all__ = ["DenseConvHead", "DenseDet3DLoss", "DenseDet3D"]
