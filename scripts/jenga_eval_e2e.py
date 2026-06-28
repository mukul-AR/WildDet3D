#!/usr/bin/env python3
"""End-to-end JENGA eval: Stage-1 detections -> Stage-2, scored against ALL GT.

Unlike the teacher-forced eval, this runs the real pipeline (decode_dense ->
decode_jenga) and matches predicted actual boxes to **every** GT box (nearest
center), so boxes Stage 1 *missed* count against recall. Reports:

  * recall@0.5 / @0.75 and detection recall (GT with any nearby prediction),
  * precision@0.5, mean per-GT IoU,
  * a TAIL DIAGNOSTIC: mean IoU bucketed by visible_fraction, and the mean
    center / size / rotation error for low-IoU vs high-IoU boxes — i.e. *what*
    caps the tail.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/jenga_eval_e2e.py \
        --ckpt ckpt/real1/jenga_last.pt --max-scenes 300 --n-samples 4096
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "third_party/sam3")
sys.path.insert(0, "third_party/lingbot_depth")
sys.path.insert(0, "third_party/moge")

from wilddet3d.dense.decode import decode_dense, decode_jenga  # noqa: E402
from wilddet3d.dense.metrics import iou3d_mc  # noqa: E402
from wilddet3d.dense.model import DenseDet3D  # noqa: E402
from wilddet3d.dense.rotation_utils import (  # noqa: E402
    matrix_to_rotation_6d, rad2deg, symmetry_min_geodesic,
)
from wilddet3d.dense.sim_jenga_dataset import SimJengaDataset, jenga_collate  # noqa: E402
from wilddet3d.dense.stage2 import JengaStage2  # noqa: E402


def move(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else (
            [x.to(device) for x in v] if isinstance(v, list) else v)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim-root", default=None)
    ap.add_argument("--wilddet3d-ckpt", default=None)
    ap.add_argument("--max-scenes", type=int, default=300)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--match-thresh", type=float, default=0.30, help="GT<->pred center match (m)")
    ap.add_argument("--n-samples", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    sim_root = args.sim_root or a["sim_root"]
    base = args.wilddet3d_ckpt or a.get("wilddet3d_ckpt")
    ds = SimJengaDataset(sim_root, a.get("size", 1008), args.max_scenes,
                         split="val", val_frac=a.get("val_frac", 0.1))
    print(f"val views: {len(ds)}  (ckpt epoch {ck.get('epoch','?')})", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=jenga_collate, pin_memory=True)

    model = DenseDet3D.from_wilddet3d(
        ckpt_path=base if base and os.path.exists(base) else None,
        fpn_level=a.get("fpn_level", 1), train_fusion=True,
        head_kwargs={"feat_ch": a.get("head_width", 256), "n_convs": a.get("head_convs", 4)},
        device=args.device)
    model.load_state_dict(ck["model"])
    stage2 = JengaStage2(in_ch=256, d_model=a.get("d_model", 512),
                         layers=a.get("layers", 12), heads=a.get("heads", 8)).to(args.device)
    stage2.load_state_dict(ck["stage2"])
    model.eval(); stage2.eval()

    # accumulate per-GT records (matched to nearest predicted actual box)
    iou_l, ctr_l, sz_l, rot_l, vf_l = [], [], [], [], []
    n_gt = 0
    n_pred = 0
    prec_hit = 0
    for batch in loader:
        batch = move(batch, args.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(batch["image"], batch["depth"], batch["K"], return_feat=True)
        feat, stride = pred["feat"].float(), pred["stride"]
        dets = decode_dense(pred["heatmap"].float(), pred["reg"].float(),
                            batch["K"], stride, score_thresh=args.score_thresh)
        res = decode_jenga(dets, feat, stride, stage2, batch["catalog"], batch["K"])
        for i in range(len(batch["act_center"])):
            gc = batch["act_center"][i].float()
            gs = batch["act_size"][i].float()
            from wilddet3d.dense.rotation_utils import rotation_6d_to_matrix
            gr = rotation_6d_to_matrix(batch["act_rot6d"][i].float())
            vf = batch["vis_frac"][i].float()
            ng = gc.shape[0]
            n_gt += ng
            pc, ps, pr = res[i]["center"], res[i]["size"], res[i]["R"]
            m = pc.shape[0]
            n_pred += m
            if ng == 0:
                continue
            if m == 0:
                iou_l.append(torch.zeros(ng)); ctr_l.append(torch.full((ng,), 9.9))
                sz_l.append(torch.full((ng,), 9.9)); rot_l.append(torch.full((ng,), 99.0))
                vf_l.append(vf.cpu())
                continue
            d = torch.cdist(gc, pc)            # [N, M] GT->pred center dist
            nn = d.argmin(dim=1); nnd = d.min(dim=1).values
            mp_c, mp_s, mp_r = pc[nn], ps[nn], pr[nn]   # nearest pred per GT
            iou = iou3d_mc(mp_c, mp_s, mp_r, gc, gs, gr, args.n_samples)
            iou = torch.where(nnd < args.match_thresh, iou, torch.zeros_like(iou))
            rot = rad2deg(symmetry_min_geodesic(
                matrix_to_rotation_6d(mp_r), matrix_to_rotation_6d(gr)))
            iou_l.append(iou.cpu()); ctr_l.append(nnd.cpu())
            sz_l.append((mp_s - gs).abs().sum(-1).cpu()); rot_l.append(rot.cpu())
            vf_l.append(vf.cpu())
            # precision: each pred -> nearest GT, count IoU>=0.5
            dp = torch.cdist(pc, gc); pnn = dp.argmin(dim=1)
            pio = iou3d_mc(pc, ps, pr, gc[pnn], gs[pnn], gr[pnn], args.n_samples)
            prec_hit += int((pio >= 0.5).sum())

    iou = torch.cat(iou_l); ctr = torch.cat(ctr_l); sz = torch.cat(sz_l)
    rot = torch.cat(rot_l); vf = torch.cat(vf_l)
    print("\n============== JENGA end-to-end eval (Stage 1 -> Stage 2) ==============")
    print(f"  GT boxes: {n_gt}  |  predicted boxes: {n_pred}")
    print(f"  mean per-GT 3D IoU   : {iou.mean():.4f}")
    print(f"  recall @0.50 / @0.75 : {(iou>=0.5).float().mean():.3f} / {(iou>=0.75).float().mean():.3f}")
    print(f"  detection recall     : {(ctr<args.match_thresh).float().mean():.3f}  (GT with a pred within {args.match_thresh}m)")
    print(f"  precision @0.50      : {prec_hit/max(n_pred,1):.3f}")
    print("\n  --- SYSTEM PERFORMANCE by visibility (headline = graspable boxes) ---")
    print(f"    {'subset':<22}{'n':>7}{'meanIoU':>10}{'recall@.5':>11}{'recall@.75':>12}")
    for thr, name in [(0.0, "all boxes"), (0.6, "graspable (vf>=.6)"), (0.9, "front (vf>=.9)")]:
        m = vf >= thr
        if int(m.sum()):
            print(f"    {name:<22}{int(m.sum()):>7}{iou[m].mean():>10.3f}"
                  f"{(iou[m]>=0.5).float().mean():>11.3f}{(iou[m]>=0.75).float().mean():>12.3f}")
    print("\n  --- TAIL DIAGNOSTIC ---")
    print("  mean IoU by visible_fraction:")
    for lo, hi in [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]:
        m = (vf >= lo) & (vf < hi)
        if int(m.sum()):
            print(f"    vis_frac [{lo:.1f},{hi:.1f}): n={int(m.sum()):5d}  IoU={iou[m].mean():.3f}")
    lowm = iou < 0.75
    himask = ~lowm
    def stats(mask):
        if int(mask.sum()) == 0:
            return "n=0"
        return (f"n={int(mask.sum()):5d}  center={ctr[mask].mean()*100:.1f}cm  "
                f"size={sz[mask].mean()*100:.1f}cm  rot={rot[mask].mean():.1f}deg  "
                f"vis_frac={vf[mask].mean():.2f}")
    print(f"  low-IoU (<0.75)  : {stats(lowm)}")
    print(f"  high-IoU (>=0.75): {stats(himask)}")
    print("========================================================================\n")


if __name__ == "__main__":
    main()
