#!/usr/bin/env python3
"""Build a self-contained 3D colored-point-cloud viewer for a REAL capture:
fuse all cameras' RGB-D into one cloud (base_link frame) and overlay
GT / Stage-1 / Stage-2 boxes as orbitable 3D wireframes.

Reuses the real-format loaders + decode chain from jenga_infer_real.py.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/jenga_pcd_viz_real.py \
    --ckpt ckpt/real1/jenga_last.pt \
    --data-dir data/place_1779901365_69470867_344c9b82dc7d2436 \
    --out-html viz_out/real1_pcd.html --score-thresh 0.3
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "third_party/sam3")
sys.path.insert(0, "third_party/lingbot_depth")
sys.path.insert(0, "third_party/moge")

import jenga_infer_real as jir  # noqa: E402
from wilddet3d.dense.decode import decode_dense, decode_jenga  # noqa: E402
from wilddet3d.dense.model import DenseDet3D  # noqa: E402
from wilddet3d.dense.stage2 import JengaStage2  # noqa: E402

# base_link (x fwd, y left, z up) -> render axes (x right, y up, z fwd/depth)
def to_render(p):
    p = np.asarray(p)
    return np.stack([-p[..., 1], p[..., 2], p[..., 0]], axis=-1)


def corners_world(center, R, size):
    return center[None] + (jir._SIGNS * (np.asarray(size) / 2.0)) @ np.asarray(R).T


def cloud_bl(cam_dir, E, stride, max_pts, depth_name="depth.png"):
    """Unproject valid depth -> colored cloud in base_link. (xyz[N,3], rgb[N,3])."""
    meta = json.load(open(os.path.join(cam_dir, "metadata.json")))
    k = meta["camera_info"]["k"]
    fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    bgr = cv2.imread(os.path.join(cam_dir, "image.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(cam_dir, depth_name), cv2.IMREAD_UNCHANGED)
    z = depth.astype(np.float32) / 1000.0
    ys, xs = np.where(z > 0)
    if stride > 1:
        sel = (ys % stride == 0) & (xs % stride == 0)
        ys, xs = ys[sel], xs[sel]
    zz = z[ys, xs]
    x = (xs - cx) * zz / fx
    y = (ys - cy) * zz / fy
    pc = np.stack([x, y, zz], axis=1)
    pw = pc @ E[:3, :3].T + E[:3, 3]
    cols = rgb[ys, xs]
    if len(pw) > max_pts:
        idx = np.round(np.linspace(0, len(pw) - 1, max_pts)).astype(int)
        pw, cols = pw[idx], cols[idx]
    return pw, cols


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-html", default="viz_out/real_pcd.html")
    ap.add_argument("--wilddet3d-ckpt", default=None)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--stride", type=int, default=2, help="depth pixel decimation per camera")
    ap.add_argument("--max-pts", type=int, default=140000, help="global point cap")
    ap.add_argument("--depth-name", default="depth.png",
                    help="depth file per camera dir (e.g. refined_depth.png for LingBot-densified)")
    ap.add_argument("--densify", action="store_true")
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
    cat_np = jir.real_catalog(scene_json)
    catalog = torch.from_numpy(cat_np).to(args.device)
    print(f"catalog ({cat_np.shape[0]} SKU): {cat_np.tolist()}", flush=True)

    clouds_xyz, clouds_rgb = [], []
    boxes = {"gt": [], "s1": [], "s2": []}
    cams = sorted(glob.glob(os.path.join(args.data_dir, "*_camera_*")))

    # GT is shared (base_link); add once.
    for (c, R, sz, conf) in jir.gt_boxes_cam_world(scene_json):
        boxes["gt"].append({"c": corners_world(c, R, sz).tolist(), "s": round(conf, 2)})

    for cam_dir in cams:
        if not os.path.exists(os.path.join(cam_dir, "metadata.json")):
            continue
        cam = os.path.basename(cam_dir)
        img, depth, k_model, k_orig, E, bgr = jir.load_view_real(cam_dir, args.size, args.densify, args.depth_name)
        pw, cols = cloud_bl(cam_dir, E, args.stride, args.max_pts // max(len(cams), 1), args.depth_name)
        clouds_xyz.append(pw); clouds_rgb.append(cols)

        img, depth, k_model = img.to(args.device), depth.to(args.device), k_model.to(args.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
            out = model(img, depth, k_model, return_feat=True)
        feat, stride = out["feat"].float(), out["stride"]
        dets = decode_dense(out["heatmap"].float(), out["reg"].float(), k_model, stride,
                            score_thresh=args.score_thresh)
        s2 = decode_jenga(dets, feat, stride, stage2, [catalog], k_model)[0]
        det = dets[0]
        Rwc, twc = E[:3, :3], E[:3, 3]
        for j in range(det["center"].shape[0]):
            cc = corners_world(det["center"][j].cpu().numpy(), det["R"][j].cpu().numpy(),
                               det["size"][j].cpu().numpy())
            boxes["s1"].append({"c": (cc @ Rwc.T + twc).tolist(), "s": round(float(det["score"][j]), 2)})
        for j in range(s2["center"].shape[0]):
            cc = corners_world(s2["center"][j].cpu().numpy(), s2["R"][j].cpu().numpy(),
                               s2["size"][j].cpu().numpy())
            boxes["s2"].append({"c": (cc @ Rwc.T + twc).tolist(), "s": round(float(s2["score"][j]), 2)})
        print(f"{cam}: {len(pw)} pts | S1 {det['center'].shape[0]} | S2 {s2['center'].shape[0]}", flush=True)

    xyz = np.concatenate(clouds_xyz); rgb = np.concatenate(clouds_rgb)
    print(f"GT boxes: {len(boxes['gt'])} | total cloud: {len(xyz)} pts", flush=True)

    # to render frame, center on cloud centroid
    xyz_r = to_render(xyz)
    ctr = xyz_r.mean(0)
    xyz_r = xyz_r - ctr
    for k in boxes:
        for b in boxes[k]:
            b["c"] = (to_render(np.array(b["c"])) - ctr).round(4).tolist()

    # binary cloud: int16 cm + uint8 rgb
    xyz_cm = np.round(xyz_r * 100).astype(np.int16)
    b_xyz = base64.b64encode(xyz_cm.tobytes()).decode()
    b_rgb = base64.b64encode(rgb.astype(np.uint8).tobytes()).decode()

    scene_id = os.path.basename(os.path.abspath(args.data_dir))
    html = build_html(scene_id, cat_np.tolist(), args.score_thresh, len(xyz),
                      b_xyz, b_rgb, boxes)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_html)), exist_ok=True)
    open(args.out_html, "w").write(html)
    print(f"\nwrote {args.out_html} ({len(html) / 1e6:.1f} MB)", flush=True)


def build_html(scene_id, catalog, thresh, npts, b_xyz, b_rgb, boxes):
    sku = ", ".join("[" + " x ".join(f"{v:.3f}" for v in c) + "]" for c in catalog)
    data = json.dumps({"xyz": b_xyz, "rgb": b_rgb, "n": npts, "boxes": boxes})
    return _TEMPLATE.replace("__SCENE__", scene_id).replace("__SKU__", sku) \
        .replace("__THRESH__", str(thresh)).replace("__NPTS__", f"{npts:,}") \
        .replace("__DATA__", data) \
        .replace("__GT__", str(len(boxes["gt"]))) \
        .replace("__S1__", str(len(boxes["s1"]))).replace("__S2__", str(len(boxes["s2"])))


_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JENGA r1 — real capture 3D</title>
<style>
:root{--bg:#0b0e13;--card:#151a22;--ink:#e6edf3;--mut:#8b97a6;--line:#222a35;
--gt:#f4d03f;--s1:#4dd0e1;--s2:#5fe06b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 18px 50px}
h1{font-size:21px;font-weight:650;letter-spacing:-.01em;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
.sub code{color:var(--ink);background:#0b0e12;padding:1px 6px;border-radius:5px}
.bar{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
.tg{display:inline-flex;align-items:center;gap:8px;cursor:pointer;user-select:none;
padding:7px 13px;border-radius:9px;border:1px solid var(--line);background:#0b0e12;
font-weight:550;font-size:13px;transition:.12s}
.tg:hover{border-color:#3a4452}.tg.off{opacity:.38}
.sw{width:13px;height:13px;border-radius:3px;display:inline-block}
.hint{color:var(--mut);font-size:12px;margin-left:auto}
.stage{position:relative;background:var(--card);border:1px solid var(--line);
border-radius:14px;overflow:hidden}
canvas{display:block;width:100%;height:620px;cursor:grab}
canvas:active{cursor:grabbing}
.legend{color:var(--mut);font-size:12.5px;margin-top:14px;line-height:1.7}
.legend b{color:var(--ink)}
</style></head><body><div class="wrap">
<h1>JENGA · real capture — 3D point cloud</h1>
<div class="sub">Model <code>real-train-1</code> (sim-trained, zero-shot) · capture <code>__SCENE__</code>
· SKU __SKU__ m · __NPTS__ fused points · drag to orbit, scroll to zoom</div>
<div class="bar">
<label class="tg" data-k="pts"><span class="sw" style="background:#9aa">▦</span>points</label>
<label class="tg" data-k="gt"><span class="sw" style="background:var(--gt)"></span>GT (__GT__)</label>
<label class="tg" data-k="s1"><span class="sw" style="background:var(--s1)"></span>Stage-1 · visible (__S1__)</label>
<label class="tg" data-k="s2"><span class="sw" style="background:var(--s2)"></span>Stage-2 · actual (__S2__)</label>
<span class="hint">R resets view</span>
</div>
<div class="stage"><canvas id="cv"></canvas></div>
<div class="legend">
<b style="color:var(--gt)">GT</b> production-pipeline boxes ·
<b style="color:var(--s1)">Stage-1</b> visible box ·
<b style="color:var(--s2)">Stage-2</b> actual (dim-conditioned) box.
Cloud fused from all 3 cameras' sparse on-robot depth (base_link frame).
</div></div>
<script>
const D=__DATA__;
const dec=s=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;};
const XYZ=new Int16Array(dec(D.xyz).buffer), RGB=dec(D.rgb), N=D.n;
const EDG=[[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]];
const COL={gt:'#f4d03f',s1:'#4dd0e1',s2:'#5fe06b'};
const ST={pts:true,gt:false,s1:false,s2:true};
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
let az=0.5, el=0.32, dist=null, DEF=null;
function span(){let mx=0;for(let i=0;i<N*3;i++){const v=Math.abs(XYZ[i])/100;if(v>mx)mx=v;}return mx;}
dist=span()*2.2; DEF=dist;
let W,H,F;
function resize(){const r=devicePixelRatio>2?2:devicePixelRatio;W=cv.clientWidth*r;H=cv.clientHeight*r;cv.width=W;cv.height=H;F=0.62*W;draw();}
function rot(x,y,z){const ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el);
 const x1=ca*x+sa*z, z1=-sa*x+ca*z; const y2=ce*y-se*z1, z2=se*y+ce*z1; return [x1,y2,z2];}
function draw(){
 const id=ctx.createImageData(W,H), buf=new Uint32Array(id.data.buffer);
 buf.fill(0xff0b0e13>>>0===0?0xff130e0b:0xff130e0b); // ABGR little-endian: bg #0b0e13
 const zb=new Float32Array(W*H); zb.fill(1e9);
 if(ST.pts){
  for(let i=0;i<N;i++){
   const [rx,ry,rz]=rot(XYZ[i*3]/100,XYZ[i*3+1]/100,XYZ[i*3+2]/100);
   const zc=rz+dist; if(zc<=0.05)continue;
   const sx=(W/2+F*rx/zc)|0, sy=(H/2-F*ry/zc)|0;
   const sz=Math.max(2,Math.min(5,(F*0.009/zc)|0));
   const r=RGB[i*3],g=RGB[i*3+1],b=RGB[i*3+2], px=(0xff000000|(b<<16)|(g<<8)|r)>>>0;
   for(let oy=0;oy<sz;oy++)for(let ox=0;ox<sz;ox++){
    const X=sx+ox,Y=sy+oy; if(X<0||Y<0||X>=W||Y>=H)continue;
    const idx=Y*W+X; if(zc<zb[idx]){zb[idx]=zc;buf[idx]=px;}
   }
  }
 }
 ctx.putImageData(id,0,0);
 ctx.lineWidth=Math.max(1.1,1.5*(devicePixelRatio>2?2:devicePixelRatio));
 for(const k of ['gt','s1','s2']){ if(!ST[k])continue; ctx.strokeStyle=COL[k];
  for(const bx of D.boxes[k]){ const p=bx.c.map(c=>{const[rx,ry,rz]=rot(c[0],c[1],c[2]);
    const zc=rz+dist; return zc<=0.05?null:[W/2+F*rx/zc,H/2-F*ry/zc];});
   if(p.some(q=>q===null))continue;
   ctx.beginPath(); for(const[a,b]of EDG){ctx.moveTo(p[a][0],p[a][1]);ctx.lineTo(p[b][0],p[b][1]);} ctx.stroke();
  }
 }
}
let drag=false,lx=0,ly=0;
cv.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;});
addEventListener('mouseup',()=>drag=false);
addEventListener('mousemove',e=>{if(!drag)return;az+=(e.clientX-lx)*0.006;el+=(e.clientY-ly)*0.006;
 el=Math.max(-1.4,Math.min(1.4,el));lx=e.clientX;ly=e.clientY;draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.0011);draw();},{passive:false});
addEventListener('keydown',e=>{if(e.key==='r'||e.key==='R'){az=0.5;el=0.32;dist=DEF;draw();}});
document.querySelectorAll('.tg').forEach(t=>{const k=t.dataset.k;if(!ST[k])t.classList.add('off');
 t.onclick=()=>{ST[k]=!ST[k];t.classList.toggle('off',!ST[k]);draw();};});
addEventListener('resize',resize); resize();
</script></body></html>"""


if __name__ == "__main__":
    main()
