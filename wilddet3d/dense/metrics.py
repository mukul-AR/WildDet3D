"""3D OBB evaluation metrics for the JENGA detector.

Full-rotation 3D IoU has no closed form (per the design doc); we approximate it
by **Monte-Carlo sampling** — sample points in the union AABB of the two oriented
boxes and count inside-each. Pure torch, runs on any device, handles arbitrary
9-DoF rotation. Used for eval metrics (not as a training loss).
"""
from __future__ import annotations

import itertools

import torch
from torch import Tensor

from wilddet3d.dense.rotation_utils import rotation_6d_to_matrix

_UNIT_SIGNS = torch.tensor(
    list(itertools.product((-0.5, 0.5), repeat=3)), dtype=torch.float32
)  # [8, 3]


def box_corners(center: Tensor, size: Tensor, r: Tensor) -> Tensor:
    """Oriented-box corners ``[N, 8, 3]`` from center/size/rotation ``[N,...]``."""
    local = _UNIT_SIGNS.to(center)[None] * size[:, None, :]  # [N, 8, 3]
    return center[:, None, :] + torch.einsum("nij,nkj->nki", r, local)


def _inside(points: Tensor, center: Tensor, size: Tensor, r: Tensor) -> Tensor:
    """Mask ``[N, S]`` of which sampled points lie inside each oriented box."""
    rel = points - center[:, None, :]  # [N, S, 3]
    local = torch.einsum("nij,nsj->nsi", r.transpose(-1, -2), rel)  # to box frame
    half = size[:, None, :] * 0.5
    return (local.abs() <= half + 1e-9).all(dim=-1)  # [N, S]


def iou3d_mc(
    c1: Tensor, s1: Tensor, r1: Tensor,
    c2: Tensor, s2: Tensor, r2: Tensor,
    n_samples: int = 8192,
) -> Tensor:
    """Monte-Carlo 3D IoU ``[N]`` between two batches of oriented boxes.

    Samples ``n_samples`` points uniformly in each pair's union AABB and returns
    intersection/union of the inside-counts. Error ~ ``1/sqrt(n_samples)``.
    """
    corners = torch.cat(
        [box_corners(c1, s1, r1), box_corners(c2, s2, r2)], dim=1
    )  # [N, 16, 3]
    lo = corners.min(dim=1).values  # [N, 3]
    hi = corners.max(dim=1).values
    u = torch.rand(c1.shape[0], n_samples, 3, device=c1.device, dtype=c1.dtype)
    pts = lo[:, None, :] + u * (hi - lo)[:, None, :]  # [N, S, 3]
    in1 = _inside(pts, c1, s1, r1)
    in2 = _inside(pts, c2, s2, r2)
    inter = (in1 & in2).float().sum(dim=-1)
    union = (in1 | in2).float().sum(dim=-1).clamp_min(1.0)
    return inter / union


@torch.no_grad()
def stage2_eval_arrays(out: dict, batch: dict, n_samples: int = 4096) -> dict:
    """Per-query teacher-forced eval arrays for the Stage-2 actual-box outputs.

    Reconstructs the predicted actual box (rotation inherited from the visible
    box; size = argmax-selected catalog SKU laid onto axes by visible per-axis
    extent order; center = visible center + predicted delta) and the GT actual
    box, 1:1 per query (no detection matching needed under teacher forcing).

    Returns 1D tensors: ``iou`` ``center_dist`` ``size_err`` ``correct`` (all
    ``[Q_total]``) and ``overlap`` (pairwise IoU among predicted actual boxes
    within each scene, ``[pairs]``). All float32 on the input device.
    """
    device = out["assign_logits"].device
    b = out["assign_logits"].shape[0]
    pc, ps, pr, gc, gs, gr, correct = [], [], [], [], [], [], []
    overlaps = []
    for i in range(b):
        n = int(out["q_mask"][i].sum())
        if n == 0:
            continue
        cat = batch["catalog"][i].to(device).float()  # [K, 3]
        assign = out["assign_logits"][i, :n].float().argmax(-1)  # [n]
        sku = cat[assign]  # [n, 3] ascending
        order = batch["vis_size"][i].to(device).float().argsort(dim=-1)  # axes asc
        size_pred = torch.zeros_like(sku).scatter_(1, order, sku)  # per-axis
        center_pred = out["center_delta"][i, :n].float() + batch["vis_center"][i].to(device).float()
        r_pred = rotation_6d_to_matrix(batch["vis_rot6d"][i].to(device).float())  # inherited
        pc.append(center_pred)
        ps.append(size_pred)
        pr.append(r_pred)
        gc.append(batch["act_center"][i].to(device).float())
        gs.append(batch["act_size"][i].to(device).float())
        gr.append(rotation_6d_to_matrix(batch["act_rot6d"][i].to(device).float()))
        correct.append((assign == batch["assign"][i].to(device)).float())
        if n >= 2:
            ii, jj = torch.triu_indices(n, n, offset=1, device=device)
            overlaps.append(
                iou3d_mc(
                    center_pred[ii], size_pred[ii], r_pred[ii],
                    center_pred[jj], size_pred[jj], r_pred[jj],
                    max(n_samples // 2, 512),
                )
            )
    if not pc:
        z = torch.zeros(0, device=device)
        return {"iou": z, "center_dist": z, "size_err": z, "correct": z, "overlap": z}
    pc_, ps_, pr_ = torch.cat(pc), torch.cat(ps), torch.cat(pr)
    gc_, gs_, gr_ = torch.cat(gc), torch.cat(gs), torch.cat(gr)
    return {
        "iou": iou3d_mc(pc_, ps_, pr_, gc_, gs_, gr_, n_samples),
        "center_dist": (pc_ - gc_).norm(dim=-1),
        "size_err": (ps_ - gs_).abs().sum(-1),
        "correct": torch.cat(correct),
        "overlap": torch.cat(overlaps) if overlaps else torch.zeros(0, device=device),
    }


def summarize_eval(arrays: dict) -> dict:
    """Reduce concatenated :func:`stage2_eval_arrays` outputs to scalar metrics."""
    iou, ov = arrays["iou"], arrays["overlap"]
    if iou.numel() == 0:
        return {}
    return {
        "iou3d": iou.mean().item(),
        "iou_50": (iou > 0.5).float().mean().item(),
        "iou_75": (iou > 0.75).float().mean().item(),
        "center_dist": arrays["center_dist"].mean().item(),
        "size_err": arrays["size_err"].mean().item(),
        "assign_acc": arrays["correct"].mean().item(),
        "overlap_frac": (ov > 0.05).float().mean().item() if ov.numel() else 0.0,
        "n_boxes": int(iou.numel()),
    }
