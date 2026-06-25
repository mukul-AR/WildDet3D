"""Stub for ``vis4d_cuda_ops`` (the SysCV CUDA extension).

The real extension provides 3D-IoU (``iou_box3d``), rotated NMS
(``nms_rotated``) and deformable attention (``ms_deform_attn_*``), used by
vis4d's model zoo / eval and ``wilddet3d.eval.detect3d``. None of it is
exercised by WildDet3D's training forward/backward (the loss uses the
shapely-based ``wilddet3d.ops.iou_3d_safe``). Building the real extension for
Blackwell (sm_120) needs a CUDA 12.8 toolchain; this stub lets the vis4d CLI
and eval modules *import* without it. Any symbol resolves to a callable that
raises clearly if actually invoked.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:  # PEP 562: lazy module attribute access
    def _f(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            f"vis4d_cuda_ops.{name} is a stub (CUDA ops not built). "
            "Build github.com/SysCV/vis4d_cuda_ops (needs CUDA 12.8 nvcc for "
            "sm_120) to enable 3D-IoU / rotated NMS / deformable attention."
        )

    _f.__name__ = name
    return _f
