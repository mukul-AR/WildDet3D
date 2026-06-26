#!/usr/bin/env python3
"""Train the JENGA two-stage 9-DoF detector on sim data (single GPU).

Stage 1 = the existing dense CenterNet head supervised on the **visible** boxes.
Stage 2 = a dimension-conditioned transformer decoder that, given the visible
boxes (teacher-forced from GT during training) + the scene's candidate-dimension
catalog, hard-selects one catalog dim per box (its actual size) and completes the
actual center + rotation. Encoders are frozen; Stage-1 head + fusion + Stage-2
decoder train jointly.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/train_jenga.py \
        --sim-root <synth_dir> --epochs 12 --batch-size 8 \
        --d-model 512 --layers 12 --heads 8 --val-frac 0.1 \
        --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt --out ckpt/jenga
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "third_party/sam3")
sys.path.insert(0, "third_party/lingbot_depth")
sys.path.insert(0, "third_party/moge")

from wilddet3d.dense.sim_jenga_dataset import SimJengaDataset, jenga_collate  # noqa: E402
from wilddet3d.dense.loss import DenseDet3DLoss, JengaStage2Loss  # noqa: E402
from wilddet3d.dense.metrics import stage2_eval_arrays, summarize_eval  # noqa: E402
from wilddet3d.dense.model import DenseDet3D  # noqa: E402
from wilddet3d.dense.stage2 import JengaStage2  # noqa: E402
from wilddet3d.dense.targets import build_dense_targets  # noqa: E402


def move(batch: dict, device: str) -> dict:
    """Move tensors (and lists of tensors) to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list):
            out[k] = [x.to(device) for x in v]
        else:
            out[k] = v
    return out


def project_to_grid(centers: torch.Tensor, k: torch.Tensor, stride: float) -> torch.Tensor:
    """Project camera-frame centers ``[N,3]`` to FPN-grid uv ``[N,2]``."""
    z = centers[:, 2].clamp_min(1e-3)
    u = (k[0, 0] * centers[:, 0] / z + k[0, 2]) / stride
    v = (k[1, 1] * centers[:, 1] / z + k[1, 2]) / stride
    return torch.stack([u, v], dim=-1)


def make_queries(batch: dict, stride: float) -> tuple[list, list]:
    """Teacher-forced Stage-2 queries from GT visible boxes."""
    queries_uv, vis_obb = [], []
    for i in range(len(batch["vis_center"])):
        c = batch["vis_center"][i]
        queries_uv.append(project_to_grid(c, batch["K"][i], stride))
        vis_obb.append(
            torch.cat([c, batch["vis_size"][i], batch["vis_rot6d"][i]], dim=-1)
        )
    return queries_uv, vis_obb


@torch.no_grad()
def validate(model, stage2, loader, loss2_fn, size, device, amp, n_samples=2048) -> dict:
    """Teacher-forced val metrics: 3D IoU + center/size/assign/overlap + rot_deg."""
    model.eval()
    stage2.eval()
    keys = ("iou", "center_dist", "size_err", "correct", "overlap")
    acc = {k: [] for k in keys}
    rot_sum, rot_n = 0.0, 0.0
    for batch in loader:
        batch = move(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = model(batch["image"], batch["depth"], batch["K"], return_feat=True)
        feat, stride = pred["feat"].float(), pred["stride"]
        queries_uv, vis_obb = make_queries(batch, stride)
        out = stage2(feat, queries_uv, vis_obb, batch["catalog"])
        out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in out.items()}
        d = loss2_fn(out, batch)
        if d["num_q"].item() == 0:
            continue
        rot_sum += d["rot_deg"].item() * d["num_q"].item()
        rot_n += d["num_q"].item()
        arr = stage2_eval_arrays(out, batch, n_samples)
        for k in keys:
            acc[k].append(arr[k])
    arrays = {k: (torch.cat(v) if v else torch.zeros(0, device=device))
              for k, v in acc.items()}
    s = summarize_eval(arrays)
    if rot_n > 0:
        s["rot_deg"] = rot_sum / rot_n
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--wilddet3d-ckpt", default="ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fpn-level", type=int, default=1)
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--w-assign", type=float, default=1.0)
    ap.add_argument("--w-center", type=float, default=1.0)
    ap.add_argument("--w-rot2", type=float, default=1.0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default="ckpt/jenga")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction,
                    default=os.environ.get("WD3D_WANDB", "0") == "1")
    ap.add_argument("--wandb-project", default=os.environ.get("WD3D_WANDB_PROJECT", "jenga-9dof"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WD3D_WANDB_ENTITY", "mukul-ganwal"))
    ap.add_argument("--wandb-run-name", default=os.environ.get("WD3D_RUN_NAME", "jenga-2stage"))
    args = ap.parse_args()

    tr_ds = SimJengaDataset(args.sim_root, args.size, args.max_scenes,
                            split="train", val_frac=args.val_frac)
    va_ds = SimJengaDataset(args.sim_root, args.size, args.max_scenes,
                            split="val", val_frac=args.val_frac)
    print(f"sim samples (views): train={len(tr_ds)} val={len(va_ds)}", flush=True)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, collate_fn=jenga_collate,
                           drop_last=True, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, collate_fn=jenga_collate,
                           drop_last=False, pin_memory=True) if len(va_ds) else None

    ckpt = args.wilddet3d_ckpt if os.path.exists(args.wilddet3d_ckpt) else None
    if ckpt is None:
        print(f"WARN: {args.wilddet3d_ckpt} not found; encoders use base weights.")
    model = DenseDet3D.from_wilddet3d(
        ckpt_path=ckpt, fpn_level=args.fpn_level, train_fusion=True, device=args.device)
    stage2 = JengaStage2(in_ch=256, d_model=args.d_model, layers=args.layers,
                         heads=args.heads).to(args.device)

    trainable = [p for p in model.parameters() if p.requires_grad] + list(stage2.parameters())
    n_train = sum(p.numel() for p in trainable)
    n_s2 = sum(p.numel() for p in stage2.parameters())
    print(f"trainable params: {n_train/1e6:.2f}M (stage1+fusion + stage2 {n_s2/1e6:.2f}M)", flush=True)

    loss1_fn = DenseDet3DLoss()
    loss2_fn = JengaStage2Loss(args.w_assign, args.w_center, args.w_rot2)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    wb = None
    if args.wandb:
        import wandb

        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None,
                        name=args.wandb_run_name, config=vars(args),
                        tags=["jenga", "two-stage", "9dof", "dim-conditioned"])
        print(f"[wandb] logging to {args.wandb_project} as '{args.wandb_run_name}'", flush=True)

    os.makedirs(args.out, exist_ok=True)
    gstep = 0
    for epoch in range(args.epochs):
        model.train()
        stage2.train()
        t0 = time.time()
        agg = {"total": 0.0, "s1": 0.0, "assign": 0.0, "acc": 0.0, "n": 0}
        for batch in tr_loader:
            batch = move(batch, args.device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                pred = model(batch["image"], batch["depth"], batch["K"], return_feat=True)
            feat, stride = pred["feat"].float(), pred["stride"]
            dense = {"heatmap": pred["heatmap"].float(), "reg": pred["reg"].float()}
            _, _, hf, wf = dense["heatmap"].shape
            tgt = build_dense_targets(
                batch["vis_center"], batch["vis_size"], batch["vis_rot6d"],
                batch["vis_box2d"], batch["K"], (hf, wf), stride, args.device)
            l1 = loss1_fn(dense, tgt)

            queries_uv, vis_obb = make_queries(batch, stride)
            out = stage2(feat, queries_uv, vis_obb, batch["catalog"])
            l2 = loss2_fn(out, batch)
            total = l1["total"] + l2["total"]

            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(opt)
            scaler.update()

            agg["total"] += total.item()
            agg["s1"] += l1["total"].item()
            agg["assign"] += l2["assign"].item()
            agg["acc"] += l2["assign_acc"].item()
            agg["n"] += 1
            gstep += 1
            if wb is not None:
                wb.log({
                    "train/total": total.item(),
                    "train/stage1": l1["total"].item(),
                    "train/s1_heatmap": l1["heatmap"].item(),
                    "train/s1_rot_deg": l1["rot_deg"].item(),
                    "train/assign": l2["assign"].item(),
                    "train/center": l2["center"].item(),
                    "train/rot2": l2["rot"].item(),
                    "train/assign_acc": l2["assign_acc"].item(),
                    "train/s2_rot_deg": l2["rot_deg"].item(),
                    "lr": opt.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                }, step=gstep)
        sched.step()
        n = max(agg["n"], 1)
        val = validate(model, stage2, va_loader, loss2_fn, args.size,
                       args.device, args.amp) if va_loader else {}
        msg = (f"epoch {epoch+1:2d}/{args.epochs} | total {agg['total']/n:.4f} "
               f"(s1 {agg['s1']/n:.4f}) | assign {agg['assign']/n:.4f} "
               f"acc {agg['acc']/n:.3f} | {time.time()-t0:.0f}s")
        if val:
            msg += (f" || val iou {val.get('iou3d', 0):.3f} "
                    f"(@.5 {val.get('iou_50', 0):.2f}) acc {val.get('assign_acc', 0):.3f} "
                    f"rot {val.get('rot_deg', 0):.1f}deg ctr {val.get('center_dist', 0):.3f} "
                    f"ovlp {val.get('overlap_frac', 0):.3f}")
        print(msg, flush=True)
        if wb is not None:
            log = {"epoch/total": agg["total"] / n, "epoch/assign_acc": agg["acc"] / n,
                   "epoch": epoch + 1}
            log.update({f"val/{k}": v for k, v in val.items()})
            wb.log(log, step=gstep)
        torch.save({"model": model.state_dict(), "stage2": stage2.state_dict(),
                    "epoch": epoch + 1, "args": vars(args)},
                   os.path.join(args.out, "jenga_last.pt"))
    print(f"saved -> {os.path.join(args.out, 'jenga_last.pt')}", flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
