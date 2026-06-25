"""Two-stage 9-DoF box detector (Anyware warehouse cartons).

Stage 1 (visual geometry): RGB + Depth crop -> visible 9-DoF OBB.
Stage 2 (actual geometry): visual OBB + scene walls + SKU candidates ->
corrected full 9-DoF OBB (occluded depth filled, snapped to SKU dims,
consistent with the container).

This package is a self-contained PyTorch implementation that reuses
WildDet3D's 6D continuous rotation representation and cuboid symmetry-aware
rotation loss. It does not depend on vis4d so the full data -> model -> loss
-> optimizer -> checkpoint pipeline can be exercised end to end.
"""

from __future__ import annotations

from wilddet3d.twostage.losses import NineDoFLoss
from wilddet3d.twostage.model import TwoStage9DoF
from wilddet3d.twostage.stage1_visual import Stage1VisualNet
from wilddet3d.twostage.stage2_geometry import Stage2GeometryNet

__all__ = [
    "NineDoFLoss",
    "Stage1VisualNet",
    "Stage2GeometryNet",
    "TwoStage9DoF",
]
