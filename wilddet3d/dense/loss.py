"""Dense 3D detection loss: focal heatmap + 9-DoF regression at peak cells."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wilddet3d.dense.metrics import corner_distance
from wilddet3d.dense.rotation_utils import (
    rad2deg,
    rotation_6d_to_matrix,
    symmetry_chordal_loss,
    symmetry_min_geodesic,
)


def centernet_focal_loss(pred_sigmoid: Tensor, gt: Tensor, eps: float = 1e-4) -> Tensor:
    """Penalty-reduced focal loss (CenterNet) on a Gaussian heatmap."""
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    pred = pred_sigmoid.clamp(eps, 1 - eps)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = (
        torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    )
    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


class DenseDet3DLoss(nn.Module):
    """Weighted dense 9-DoF loss.

    Args:
        w_hm: objectness heatmap (focal) weight.
        w_off: 2D sub-cell offset (du, dv) weight.
        w_depth: log-depth weight.
        w_size: log-size weight.
        w_rot: symmetry-aware rotation weight.
    """

    def __init__(
        self,
        w_hm: float = 1.0,
        w_off: float = 1.0,
        w_depth: float = 1.0,
        w_size: float = 1.0,
        w_rot: float = 1.0,
    ) -> None:
        super().__init__()
        self.w_hm = w_hm
        self.w_off = w_off
        self.w_depth = w_depth
        self.w_size = w_size
        self.w_rot = w_rot

    def forward(
        self, pred: dict[str, Tensor], target: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Args: pred (heatmap logits, reg), target (heatmap, reg, pos_mask)."""
        hm = torch.sigmoid(pred["heatmap"])
        loss_hm = centernet_focal_loss(hm, target["heatmap"])

        pos = target["pos_mask"]  # [B, Hf, Wf]
        reg_pred = pred["reg"].permute(0, 2, 3, 1)[pos]  # [P, 12]
        reg_tgt = target["reg"].permute(0, 2, 3, 1)[pos]  # [P, 12]
        n = reg_pred.shape[0]
        device = pred["heatmap"].device

        if n == 0:
            zero = torch.zeros((), device=device)
            out = {
                "heatmap": loss_hm,
                "offset": zero,
                "depth": zero,
                "size": zero,
                "rot": zero,
                "total": self.w_hm * loss_hm,
                "rot_deg": zero,
                "num_pos": torch.tensor(0.0, device=device),
            }
            return out

        loss_off = (reg_pred[:, 0:2] - reg_tgt[:, 0:2]).abs().mean()
        loss_depth = (reg_pred[:, 2] - reg_tgt[:, 2]).abs().mean()
        loss_size = (reg_pred[:, 3:6] - reg_tgt[:, 3:6]).abs().mean()
        loss_rot = symmetry_chordal_loss(
            reg_pred[:, 6:12], reg_tgt[:, 6:12]
        ).mean()

        total = (
            self.w_hm * loss_hm
            + self.w_off * loss_off
            + self.w_depth * loss_depth
            + self.w_size * loss_size
            + self.w_rot * loss_rot
        )
        with torch.no_grad():
            rot_deg = rad2deg(
                symmetry_min_geodesic(reg_pred[:, 6:12], reg_tgt[:, 6:12])
            ).mean()
        return {
            "heatmap": loss_hm,
            "offset": loss_off,
            "depth": loss_depth,
            "size": loss_size,
            "rot": loss_rot,
            "total": total,
            "rot_deg": rot_deg,
            "num_pos": torch.tensor(float(n), device=device),
        }


class JengaStage2Loss(nn.Module):
    """Assignment CE + center L1 + per-axis size L1 + corner-distance (ADD).

    Rotation is inherited from the visible box (not predicted). Stage 2 predicts
    the per-axis log-extents (``out["log_size"]``) — i.e. which SKU dim goes on
    which inherited axis — and the actual-center residual. The **ADD** term
    (mean corner-to-corner distance, fixed correspondence) supervises the dim->
    axis assignment geometrically, where 3D IoU saturates on cube-ish boxes.

    Args:
        w_assign: catalog-assignment cross-entropy weight.
        w_center: actual-center L1 weight.
        w_size: per-axis log-size L1 weight.
        w_add: corner-distance (ADD) weight.
        vis_weight_thresh: if > 0, down-weight per-box losses by
            ``clamp(vis_frac / thresh, floor, 1)`` so capacity focuses on
            graspable (well-visible) boxes; heavily-occluded boxes keep a small
            floor weight (still emit a sane box for collision). 0 = uniform.
        vis_weight_floor: minimum per-box weight for occluded boxes.
    """

    def __init__(
        self, w_assign: float = 1.0, w_center: float = 1.0,
        w_size: float = 1.0, w_add: float = 1.0,
        vis_weight_thresh: float = 0.0, vis_weight_floor: float = 0.1,
    ) -> None:
        super().__init__()
        self.w_assign, self.w_center = w_assign, w_center
        self.w_size, self.w_add = w_size, w_add
        self.vis_weight_thresh, self.vis_weight_floor = vis_weight_thresh, vis_weight_floor

    def forward(self, out: dict, batch: dict) -> dict:
        device = out["assign_logits"].device
        logits, dc, lsz = out["assign_logits"], out["center_delta"], out["log_size"]
        b = logits.shape[0]
        pa, pc, psz, pr, ta, tc, tsz, gr, vf = [], [], [], [], [], [], [], [], []
        has_vf = "vis_frac" in batch
        for i in range(b):
            n = int(out["q_mask"][i].sum())
            if n == 0:
                continue
            pa.append(logits[i, :n])  # [n, K]
            pc.append(dc[i, :n] + batch["vis_center"][i].to(device))
            psz.append(lsz[i, :n].exp())  # per-axis extents (m)
            pr.append(batch["vis_rot6d"][i].to(device))  # inherited rotation
            ta.append(batch["assign"][i].to(device))
            tc.append(batch["act_center"][i].to(device))
            tsz.append(batch["act_size"][i].to(device))  # native per-axis
            gr.append(batch["act_rot6d"][i].to(device))
            if has_vf:
                vf.append(batch["vis_frac"][i].to(device))
        if not pa:
            z = torch.zeros((), device=device)
            return {
                "assign": z, "center": z, "size": z, "add": z, "total": z,
                "assign_acc": z, "num_q": torch.tensor(0.0, device=device),
            }
        pa_c, ta_c = torch.cat(pa), torch.cat(ta)
        pc_c, tc_c = torch.cat(pc), torch.cat(tc)
        psz_c, tsz_c = torch.cat(psz), torch.cat(tsz)
        pr_c, gr_c = torch.cat(pr), torch.cat(gr)

        # per-box weights: focus on graspable boxes (occluded ones keep a floor)
        if self.vis_weight_thresh > 0 and vf:
            w = (torch.cat(vf) / self.vis_weight_thresh).clamp(self.vis_weight_floor, 1.0)
        else:
            w = torch.ones_like(ta_c, dtype=torch.float32)
        ws = w.sum().clamp_min(1.0)

        loss_assign = (F.cross_entropy(pa_c, ta_c, reduction="none") * w).sum() / ws
        loss_center = ((pc_c - tc_c).abs().mean(-1) * w).sum() / ws
        loss_size = ((psz_c.clamp_min(1e-3).log() - tsz_c.clamp_min(1e-3).log()).abs().mean(-1) * w).sum() / ws
        # ADD-S (symmetry-tolerant): doesn't fight Stage-1's symmetric rotation
        # relabelings, so it cleanly supervises size + center.
        add_per = corner_distance(
            pc_c, psz_c, rotation_6d_to_matrix(pr_c),
            tc_c, tsz_c, rotation_6d_to_matrix(gr_c), symmetric=True,
        )
        loss_add = (add_per * w).sum() / ws
        total = (
            self.w_assign * loss_assign + self.w_center * loss_center
            + self.w_size * loss_size + self.w_add * loss_add
        )
        with torch.no_grad():
            acc = (pa_c.argmax(-1) == ta_c).float().mean()
        return {
            "assign": loss_assign, "center": loss_center, "size": loss_size,
            "add": loss_add, "total": total, "assign_acc": acc,
            "num_q": torch.tensor(float(ta_c.numel()), device=device),
        }
