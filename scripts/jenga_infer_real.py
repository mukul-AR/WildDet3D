#!/usr/bin/env python3
"""Run the JENGA chain on a REAL robot capture and build a self-contained HTML
viewer showing GT / Stage-1 / Stage-2 boxes per camera (toggleable overlays).

- GT          : the production pipeline's boxes from scene.json
                (`scene.boxes_yaml_string`; xyzwpr = position[m] + W,P,R Euler
                in DEGREES, composed Rz(R)Ry(P)Rx(W); base_link frame).
- Stage-1     : the dense head's VISIBLE 9-DoF box (`decode_dense`).
- Stage-2     : the dim-conditioned ACTUAL box (`decode_jenga`); inherits the
                visible rotation, snaps to the scene SKU, places the center.

Real format differs from sim: ROS `camera_info.k`, quaternion extrinsics,
`image.jpg`, sparse projected depth, SKU catalog in `scene.skus_yaml_string`.
No sim GT -> qualitative check, not a metric eval.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/jenga_infer_real.py \
    --ckpt ckpt/real1/jenga_last.pt \
    --data-dir data/place_1779901365_69470867_344c9b82dc7d2436 \
    --out-html viz_out/real1_place.html --score-thresh 0.3 [--densify]
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "third_party/sam3")
sys.path.insert(0, "third_party/lingbot_depth")
sys.path.insert(0, "third_party/moge")

from wilddet3d.dense.decode import decode_dense, decode_jenga  # noqa: E402
from wilddet3d.dense.model import DenseDet3D  # noqa: E402
from wilddet3d.dense.sim_dataset import (  # noqa: E402
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _resize_pad,
)
from wilddet3d.dense.stage2 import JengaStage2  # noqa: E402

_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                  dtype=np.float64)
_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def _quat_to_R(w, x, y, z):
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _euler_zyx_deg(w, p, r):
    """W,P,R degrees about X,Y,Z -> R = Rz(R)·Ry(P)·Rx(W)."""
    w, p, r = np.radians(w), np.radians(p), np.radians(r)
    cw, sw, cp, sp, cr, sr = np.cos(w), np.sin(w), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    Rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _densify(depth_mm: np.ndarray) -> np.ndarray:
    mask = (depth_mm == 0).astype(np.uint8)
    if mask.sum() == 0:
        return depth_mm
    _, lbl = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.where(depth_mm > 0)
    vals = depth_mm[ys, xs]
    order = np.argsort(ys * depth_mm.shape[1] + xs)
    lut = np.zeros(lbl.max() + 1, dtype=depth_mm.dtype)
    lut[1:len(order) + 1] = vals[order]
    out = depth_mm.copy()
    out[mask == 1] = lut[lbl[mask == 1]]
    return out


def real_catalog(scene_json: str) -> np.ndarray:
    s = json.load(open(scene_json))
    skus = yaml.safe_load(s["scene"]["skus_yaml_string"])
    if isinstance(skus, dict):
        skus = skus.get("skus", [])
    dims = {tuple(sorted(round(float(x), 4) for x in sk["geometry"])) for sk in skus}
    return np.array(sorted(dims), dtype=np.float32).reshape(-1, 3)


def gt_boxes_cam(scene_json: str, E: np.ndarray):
    """Production boxes -> camera-frame (center[3], R[3,3], size[3], conf)."""
    s = json.load(open(scene_json))
    boxes = yaml.safe_load(s["scene"]["boxes_yaml_string"])
    R_bc = E[:3, :3].T          # base_link -> cam (E is cam->base_link)
    t_bc = -R_bc @ E[:3, 3]
    out = []
    for b in boxes.values():
        p = b["xyzwpr"]
        c_bl = np.array(p[:3], dtype=np.float64)
        R_bl = _euler_zyx_deg(p[3], p[4], p[5])
        c_cam = R_bc @ c_bl + t_bc
        R_cam = R_bc @ R_bl
        out.append((c_cam, R_cam, np.array(b["geometry"], dtype=np.float64),
                    float(b.get("confidence", 1.0))))
    return out


def gt_boxes_cam_world(scene_json: str):
    """Production boxes in base_link frame: (center[3], R[3,3], size[3], conf)."""
    s = json.load(open(scene_json))
    boxes = yaml.safe_load(s["scene"]["boxes_yaml_string"])
    out = []
    for b in boxes.values():
        p = b["xyzwpr"]
        out.append((np.array(p[:3], dtype=np.float64),
                    _euler_zyx_deg(p[3], p[4], p[5]),
                    np.array(b["geometry"], dtype=np.float64),
                    float(b.get("confidence", 1.0))))
    return out


def project_box(center, R, size, K, w, h):
    """8 oriented corners -> image [u,v]; None if behind camera or off-frame."""
    corners = center[None] + (_SIGNS * (np.asarray(size) / 2.0)) @ np.asarray(R).T
    if (corners[:, 2] <= 1e-3).any():
        return None
    uv = (K @ corners.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    cu, cv_ = uv.mean(0)
    if not (-0.25 * w <= cu <= 1.25 * w and -0.25 * h <= cv_ <= 1.25 * h):
        return None
    return [[round(float(u), 1), round(float(v), 1)] for u, v in uv]


def load_view_real(cam_dir: str, size: int, densify: bool, depth_name: str = "depth.png"):
    meta = json.load(open(os.path.join(cam_dir, "metadata.json")))
    k = meta["camera_info"]["k"]
    fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    ex = meta["extrinsics"]
    R = _quat_to_R(ex["rotation"]["w"], ex["rotation"]["x"], ex["rotation"]["y"], ex["rotation"]["z"])
    t = np.array([ex["translation"]["x"], ex["translation"]["y"], ex["translation"]["z"]])
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t

    bgr = cv2.imread(os.path.join(cam_dir, "image.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(cam_dir, depth_name), cv2.IMREAD_UNCHANGED)
    if depth is None:
        depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
    if densify:
        depth = _densify(depth.astype(np.uint16))
    rgb_p, scale, px, py = _resize_pad(rgb, size, nearest=False)
    depth_p, _, _, _ = _resize_pad(depth.astype(np.uint16), size, nearest=True)
    img = (rgb_p.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    img = torch.from_numpy(img.transpose(2, 0, 1))[None]
    depth_m = torch.from_numpy((depth_p.astype(np.float32) / 1000.0)[None, None])
    k_model = torch.tensor(
        [[fx * scale, 0, cx * scale + px], [0, fy * scale, cy * scale + py], [0, 0, 1]],
        dtype=torch.float32)[None]
    k_orig = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return img, depth_m, k_model, k_orig, E, bgr


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-html", default="viz_out/real_infer.html")
    ap.add_argument("--wilddet3d-ckpt", default=None)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--depth-name", default="depth.png",
                    help="depth file per camera dir (e.g. refined_depth.png for LingBot-densified)")
    ap.add_argument("--densify", action="store_true")
    ap.add_argument("--posthoc-anchor", action="store_true",
                    help="geometric near-face center anchor instead of learned center_delta")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    base = args.wilddet3d_ckpt or a.get("wilddet3d_ckpt")
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

    scene_json = os.path.join(args.data_dir, "scene.json")
    cat_np = real_catalog(scene_json)
    catalog = torch.from_numpy(cat_np).to(args.device)
    print(f"catalog ({cat_np.shape[0]} SKU): {cat_np.tolist()}", flush=True)

    cams = []
    ce_all, dz_all, rot_all, n_gt_all = [], [], [], 0
    for cam_dir in sorted(glob.glob(os.path.join(args.data_dir, "*_camera_*"))):
        if not os.path.exists(os.path.join(cam_dir, "metadata.json")):
            continue
        cam = os.path.basename(cam_dir)
        img, depth, k_model, k_orig, E, bgr = load_view_real(cam_dir, args.size, args.densify, args.depth_name)
        H, W = bgr.shape[:2]
        img, depth, k_model = img.to(args.device), depth.to(args.device), k_model.to(args.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
            out = model(img, depth, k_model, return_feat=True)
        heatmap, reg = out["heatmap"].float(), out["reg"].float()
        feat, stride = out["feat"].float(), out["stride"]
        dets = decode_dense(heatmap, reg, k_model, stride, score_thresh=args.score_thresh)
        s2 = decode_jenga(dets, feat, stride, stage2, [catalog], k_model,
                          posthoc_anchor=args.posthoc_anchor)[0]
        det = dets[0]

        def cam_to_orig(c):  # model-frame center is metric (resize-invariant)
            return c  # decode used k_model; 3D boxes are in true camera metres

        s1_boxes, s2_boxes = [], []
        for j in range(det["center"].shape[0]):
            pb = project_box(det["center"][j].cpu().numpy(), det["R"][j].cpu().numpy(),
                             det["size"][j].cpu().numpy(), k_orig, W, H)
            if pb:
                s1_boxes.append({"c": pb, "s": round(float(det["score"][j]), 2)})
        for j in range(s2["center"].shape[0]):
            pb = project_box(s2["center"][j].cpu().numpy(), s2["R"][j].cpu().numpy(),
                             s2["size"][j].cpu().numpy(), k_orig, W, H)
            if pb:
                s2_boxes.append({"c": pb, "s": round(float(s2["score"][j]), 2)})
        gt_cam = gt_boxes_cam(scene_json, E)
        gt_boxes = []
        for (c, R, sz, conf) in gt_cam:
            pb = project_box(c, R, sz, k_orig, W, H)
            if pb:
                gt_boxes.append({"c": pb, "s": round(conf, 2)})
        # 3D center-error vs GT (camera frame): the real shallow-bias test.
        # match each GT to the nearest Stage-2 prediction; track |error| and the
        # SIGNED depth error (pred_z - gt_z) — negative = placed too shallow.
        if gt_cam and s2["center"].shape[0] > 0:
            gtc = np.stack([c for (c, _R, _s, _cf) in gt_cam])
            pcn = s2["center"].cpu().numpy()
            D = np.linalg.norm(gtc[:, None] - pcn[None], axis=-1)
            jj, dm = D.argmin(1), D.min(1)
            n_gt_all += len(gtc)
            matched = [gi for gi in range(len(gtc)) if dm[gi] < 0.5]
            for gi in matched:
                ce_all.append(float(dm[gi]))
                dz_all.append(float(pcn[jj[gi], 2] - gtc[gi, 2]))
            # symmetry-aware rotation error (deg) on matched boxes
            if matched:
                from wilddet3d.dense.rotation_utils import (
                    matrix_to_rotation_6d as _m6, symmetry_min_geodesic as _smg, rad2deg as _r2d)
                pR = s2["R"][[int(jj[gi]) for gi in matched]].float().cpu()
                gR = torch.tensor(np.stack([gt_cam[gi][1] for gi in matched]), dtype=torch.float32)
                rot_all.extend(_r2d(_smg(_m6(pR), _m6(gR))).tolist())

        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode()
        cams.append({"name": cam, "w": W, "h": H, "img": b64,
                     "gt": gt_boxes, "s1": s1_boxes, "s2": s2_boxes})
        print(f"{cam}: GT {len(gt_boxes)} | Stage-1 {len(s1_boxes)} | Stage-2 {len(s2_boxes)}", flush=True)

    if ce_all:
        ce, dz = np.array(ce_all), np.array(dz_all)
        tag = "GEOMETRIC ANCHOR" if args.posthoc_anchor else "learned center_delta"
        print(f"\n=== 3D center error vs GT [{tag}] (matched <0.5m) ===")
        print(f"  matched {len(ce_all)}/{n_gt_all} GT")
        print(f"  mean center error        : {ce.mean()*100:5.1f} cm")
        print(f"  mean |depth error|       : {np.abs(dz).mean()*100:5.1f} cm")
        print(f"  mean SIGNED depth (pred-gt): {dz.mean()*100:+5.1f} cm  (negative = placed too shallow)")
        if rot_all:
            rr = np.array(rot_all)
            print(f"  rotation error (sym-aware) : {rr.mean():5.1f} deg  (median {np.median(rr):.1f})")

    scene_id = os.path.basename(os.path.abspath(args.data_dir))
    html = build_html(scene_id, cat_np.tolist(), args.score_thresh, args.densify, cams)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_html)), exist_ok=True)
    open(args.out_html, "w").write(html)
    print(f"\nwrote {args.out_html} ({len(html) / 1e6:.1f} MB)", flush=True)


def build_html(scene_id, catalog, thresh, densify, cams):
    data = json.dumps({"edges": _EDGES, "cams": cams})
    sku = ", ".join("[" + " x ".join(f"{v:.3f}" for v in c) + "] m" for c in catalog)
    panels = "\n".join(
        f'<div class="panel"><div class="cam">{c["name"]}'
        f'<span class="ct"><b style="color:#f4d03f">{len(c["gt"])}</b> GT · '
        f'<b style="color:#4dd0e1">{len(c["s1"])}</b> S1 · '
        f'<b style="color:#5fe06b">{len(c["s2"])}</b> S2</span></div>'
        f'<canvas id="cv{i}" data-i="{i}"></canvas></div>'
        for i, c in enumerate(cams))
    return _TEMPLATE.replace("__SCENE__", scene_id).replace("__SKU__", sku) \
        .replace("__THRESH__", str(thresh)).replace("__DENSIFY__", "on" if densify else "off") \
        .replace("__PANELS__", panels).replace("__DATA__", data)


_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JENGA r1 — real capture</title>
<style>
:root{--bg:#0e1116;--card:#171c24;--ink:#e6edf3;--mut:#8b97a6;--line:#262d38;
--gt:#f4d03f;--s1:#4dd0e1;--s2:#5fe06b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;font-weight:650;letter-spacing:-.01em;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:20px}
.sub code{color:var(--ink);background:#0b0e12;padding:1px 6px;border-radius:5px}
.bar{position:sticky;top:0;z-index:5;background:rgba(14,17,22,.92);
backdrop-filter:blur(8px);border:1px solid var(--line);border-radius:12px;
padding:12px 16px;margin-bottom:22px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.tg{display:inline-flex;align-items:center;gap:8px;cursor:pointer;user-select:none;
padding:7px 13px;border-radius:9px;border:1px solid var(--line);background:#0b0e12;
font-weight:550;font-size:13px;transition:.12s}
.tg:hover{border-color:#3a4452}.tg.off{opacity:.4}
.sw{width:13px;height:13px;border-radius:3px;display:inline-block}
.hint{color:var(--mut);font-size:12px;margin-left:auto}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;
overflow:hidden;margin-bottom:18px}
.cam{display:flex;justify-content:space-between;align-items:center;padding:11px 15px;
border-bottom:1px solid var(--line);font-weight:600;letter-spacing:.01em}
.ct{font-weight:500;color:var(--mut);font-size:12.5px;font-variant-numeric:tabular-nums}
canvas{display:block;width:100%;height:auto}
.legend{color:var(--mut);font-size:12.5px;margin-top:18px;line-height:1.7}
.legend b{color:var(--ink);font-weight:600}
</style></head><body><div class="wrap">
<h1>JENGA · real container capture</h1>
<div class="sub">Model <code>real-train-1</code> (sim-trained, zero-shot) on capture
<code>__SCENE__</code> &nbsp;·&nbsp; SKU __SKU__ &nbsp;·&nbsp; score ≥ __THRESH__ &nbsp;·&nbsp; depth-densify __DENSIFY__</div>
<div class="bar">
<label class="tg" data-k="gt"><span class="sw" style="background:var(--gt)"></span>GT (production)</label>
<label class="tg" data-k="s1"><span class="sw" style="background:var(--s1)"></span>Stage-1 · visible</label>
<label class="tg" data-k="s2"><span class="sw" style="background:var(--s2)"></span>Stage-2 · actual</label>
<span class="hint">click a layer to toggle · scores shown on each box</span>
</div>
__PANELS__
<div class="legend">
<b style="color:var(--gt)">GT</b> — boxes from the existing production pipeline (<code>scene.json</code>), the reference. &nbsp;
<b style="color:var(--s1)">Stage-1</b> — the dense head's <b>visible</b> 9-DoF box (what the camera sees). &nbsp;
<b style="color:var(--s2)">Stage-2</b> — the dim-conditioned <b>actual</b> box: inherits Stage-1's rotation, snaps to the scene SKU, places the full extent.<br>
No sim ground truth on real data — this is a qualitative generalization check, not a metric eval. Depth is the sparse on-robot projection (~20% valid).
</div></div>
<script>
const D=__DATA__, ST={gt:true,s1:false,s2:true},
COL={gt:'#f4d03f',s1:'#4dd0e1',s2:'#5fe06b'};
const imgs={};
function draw(i){
 const c=D.cams[i], cv=document.getElementById('cv'+i), x=cv.getContext('2d');
 const im=imgs[i]; if(!im) return;
 const W=cv.clientWidth*devicePixelRatio, H=W*c.h/c.w;
 cv.width=W; cv.height=H; const sx=W/c.w, sy=H/c.h;
 x.clearRect(0,0,W,H); x.drawImage(im,0,0,W,H);
 x.lineWidth=Math.max(1.2,1.6*devicePixelRatio); x.font=(12*devicePixelRatio)+'px ui-monospace,monospace';
 for(const k of ['gt','s1','s2']){ if(!ST[k]) continue;
  x.strokeStyle=COL[k]; x.fillStyle=COL[k];
  for(const b of c[k]){ const p=b.c;
   x.beginPath();
   for(const [a,bb] of D.edges){ x.moveTo(p[a][0]*sx,p[a][1]*sy); x.lineTo(p[bb][0]*sx,p[bb][1]*sy); }
   x.stroke();
   let mu=0,mv=0; for(const q of p){mu+=q[0];mv+=q[1];}
   x.fillText(b.s.toFixed(2), mu/8*sx, mv/8*sy);
  }
 }
}
function redrawAll(){ D.cams.forEach((_,i)=>draw(i)); }
D.cams.forEach((c,i)=>{ const im=new Image(); im.onload=()=>{imgs[i]=im;draw(i);};
 im.src='data:image/jpeg;base64,'+c.img; });
document.querySelectorAll('.tg').forEach(t=>{
 const k=t.dataset.k; if(!ST[k]) t.classList.add('off');
 t.onclick=()=>{ST[k]=!ST[k]; t.classList.toggle('off',!ST[k]); redrawAll();};
});
addEventListener('resize',redrawAll);
</script></body></html>"""


if __name__ == "__main__":
    main()
