"""Late multi-view fusion: merge per-view 3D detections in a shared frame.

The baseline path for multi-capture scenes (n = 1..3 views):
1. Run single-view WildDet3D inference per view.
2. Transform each view's 10-dim boxes3d (camera frame) into the shared
   base_link frame using the per-view T_base_cam extrinsics.
3. Greedy cross-view 3D NMS / weighted-merge of duplicate detections.

This requires no training and degrades to a no-op for n=1 scenes.
"""

from __future__ import annotations

import torch
from torch import Tensor
from vis4d.op.geometry.rotation import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)

from wilddet3d.ops.iou_3d_safe import batch_box3d_iou


def transform_boxes3d(boxes3d: Tensor, t_dst_src: Tensor) -> Tensor:
    """Transform 10-dim boxes3d [center(3), W, L, H, quat(4)] between frames.

    Args:
        boxes3d: [N, 10] boxes in the source frame.
        t_dst_src: [4, 4] rigid transform mapping source -> destination.

    Returns:
        [N, 10] boxes in the destination frame (dims unchanged).
    """
    if boxes3d.numel() == 0:
        return boxes3d
    rot = t_dst_src[:3, :3]
    trans = t_dst_src[:3, 3]
    centers = boxes3d[:, :3] @ rot.T + trans
    r_box = quaternion_to_matrix(boxes3d[:, 6:10])
    r_new = rot.unsqueeze(0) @ r_box
    quat = matrix_to_quaternion(r_new)
    return torch.cat([centers, boxes3d[:, 3:6], quat], dim=1)


class SceneLateFusion:
    """Cross-view merge of per-view detections in a shared frame.

    Args:
        iou_threshold: 3D IoU above which two detections from different
            views are considered the same physical object.
        merge_mode: "nms" keeps the highest-scoring detection;
            "weighted" score-weighted-averages centers/dims and keeps
            the best view's rotation (rotation averaging across flips
            is ambiguous for cuboids).
        score_boost: Multiplicative bonus per supporting view, rewarding
            multi-view-consistent detections (1.0 = disabled).
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        merge_mode: str = "weighted",
        score_boost: float = 1.05,
    ) -> None:
        assert merge_mode in {"nms", "weighted"}
        self.iou_threshold = iou_threshold
        self.merge_mode = merge_mode
        self.score_boost = score_boost

    @torch.no_grad()
    def __call__(
        self,
        boxes3d_per_view: list[Tensor],
        scores_per_view: list[Tensor],
        extrinsics_per_view: list[Tensor],
        class_ids_per_view: list[Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Fuse detections from multiple views of the same scene.

        Args:
            boxes3d_per_view: per view [N_i, 10] boxes in CAMERA frame.
            scores_per_view: per view [N_i] confidence scores.
            extrinsics_per_view: per view [4, 4] T_base_cam.
            class_ids_per_view: optional per view [N_i] class ids.

        Returns:
            dict with fused "boxes3d" [M, 10] (base_link frame),
            "scores" [M], "class_ids" [M], "num_views" [M] (supporting
            view count), "view_ids" [M] (winning view per detection).
        """
        device = (
            boxes3d_per_view[0].device if boxes3d_per_view else "cpu"
        )
        all_boxes, all_scores, all_views, all_cls = [], [], [], []
        for view_id, (boxes, scores, t_base_cam) in enumerate(
            zip(boxes3d_per_view, scores_per_view, extrinsics_per_view)
        ):
            if boxes.numel() == 0:
                continue
            all_boxes.append(transform_boxes3d(boxes, t_base_cam.to(boxes)))
            all_scores.append(scores)
            all_views.append(
                torch.full((len(boxes),), view_id, device=boxes.device)
            )
            if class_ids_per_view is not None:
                all_cls.append(class_ids_per_view[view_id])

        if not all_boxes:
            return {
                "boxes3d": torch.zeros(0, 10, device=device),
                "scores": torch.zeros(0, device=device),
                "class_ids": torch.zeros(0, dtype=torch.long, device=device),
                "num_views": torch.zeros(0, dtype=torch.long, device=device),
                "view_ids": torch.zeros(0, dtype=torch.long, device=device),
            }

        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        views = torch.cat(all_views)
        cls_ids = (
            torch.cat(all_cls)
            if all_cls
            else torch.zeros(len(boxes), dtype=torch.long, device=boxes.device)
        )

        order = scores.argsort(descending=True)
        keep_groups: list[list[int]] = []  # winner idx + supporters
        suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)

        for i in order.tolist():
            if suppressed[i]:
                continue
            group = [i]
            suppressed[i] = True
            remaining = (~suppressed).nonzero(as_tuple=True)[0]
            if len(remaining) > 0:
                ious = batch_box3d_iou(
                    boxes[i : i + 1].expand(len(remaining), -1),
                    boxes[remaining],
                )
                ious = torch.as_tensor(ious, device=boxes.device).reshape(-1)
                dup = remaining[ious > self.iou_threshold]
                # Only fuse across views; same-view dups were handled by
                # the per-view NMS already, but suppress them anyway.
                for j in dup.tolist():
                    suppressed[j] = True
                    group.append(j)
            keep_groups.append(group)

        fused_boxes, fused_scores, fused_cls = [], [], []
        fused_nviews, fused_view_ids = [], []
        for group in keep_groups:
            idx = torch.tensor(group, device=boxes.device)
            g_scores = scores[idx]
            winner = idx[g_scores.argmax()]
            n_views = int(views[idx].unique().numel())
            if self.merge_mode == "weighted" and len(idx) > 1:
                w = g_scores / g_scores.sum()
                center = (boxes[idx, :3] * w.unsqueeze(1)).sum(0)
                dims = (boxes[idx, 3:6] * w.unsqueeze(1)).sum(0)
                quat = boxes[winner, 6:10]  # keep best view's rotation
                fused_boxes.append(torch.cat([center, dims, quat]))
            else:
                fused_boxes.append(boxes[winner])
            score = scores[winner] * (self.score_boost ** max(n_views - 1, 0))
            fused_scores.append(score.clamp(max=1.0))
            fused_cls.append(cls_ids[winner])
            fused_nviews.append(n_views)
            fused_view_ids.append(int(views[winner].item()))

        return {
            "boxes3d": torch.stack(fused_boxes),
            "scores": torch.stack(fused_scores),
            "class_ids": torch.stack(fused_cls),
            "num_views": torch.tensor(fused_nviews, device=boxes.device),
            "view_ids": torch.tensor(fused_view_ids, device=boxes.device),
        }
