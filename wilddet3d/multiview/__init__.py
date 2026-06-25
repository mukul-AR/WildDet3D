"""Multi-view (multi-capture scene) support for WildDet3D."""

from .late_fusion import SceneLateFusion, transform_boxes3d
from .scene_fusion import SceneFusion

__all__ = ["SceneLateFusion", "SceneFusion", "transform_boxes3d"]
