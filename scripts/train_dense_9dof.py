#!/usr/bin/env python3
"""Train the prompt-free DENSE 9-DoF detector on Anyware (single 24 GB GPU).

Reuses the warehouse-adapted SAM3 backbone + LingBot depth + EarlyDepthFusion
from a trained WildDet3D checkpoint (frozen, run under no_grad) and trains a
dense conv head over the fused FPN features. No prompts / text / per-object
input.

Usage:
    PYTHONPATH=. WD3D_SKIP_DEPTH_LOSS=1 .venv/bin/python \
        scripts/train_dense_9dof.py --epochs 6 \
        --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt
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

from wilddet3d.dense.sim_dataset import SimDenseDataset, dense_collate  # noqa: E402
from wilddet3d.dense.loss import DenseDet3DLoss  # noqa: E402
from wilddet3d.dense.model import DenseDet3D  # noqa: E402
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


def run_targets(batch: dict, pred: dict, size: int, device: str) -> dict:
    """Build dense targets matching the predicted FPN grid."""
    _, _, hf, wf = pred["heatmap"].shape
    stride = size / hf
    return build_dense_targets(
        batch["centers"], batch["sizes"], batch["rot6d"], batch["box2d"],
        batch["K"], (hf, wf), stride, device,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", required=True, help="sim scenes dir (.../anyware-sim/build/scenes/synth)")
    ap.add_argument("--sim-target", default="actual", choices=["actual", "visible"])
    ap.add_argument("--wilddet3d-ckpt", default="ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--fpn-level", type=int, default=1)
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default="ckpt/dense_9dof")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction,
                    default=os.environ.get("WD3D_WANDB", "0") == "1",
                    help="log to Weights & Biases (needs WANDB_API_KEY).")
    ap.add_argument("--wandb-project", default=os.environ.get("WD3D_WANDB_PROJECT", "jenga-9dof"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WD3D_WANDB_ENTITY", "mukul-ganwal"))
    ap.add_argument("--wandb-run-name", default=os.environ.get("WD3D_RUN_NAME", "jenga-dense"))
    args = ap.parse_args()

    ds = SimDenseDataset(args.sim_root, args.size, args.sim_target, args.max_scenes)
    print(f"sim train samples (views): {len(ds)}  (target={args.sim_target})")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=dense_collate, drop_last=True, pin_memory=True,
    )

    ckpt = args.wilddet3d_ckpt if os.path.exists(args.wilddet3d_ckpt) else None
    if ckpt is None:
        print(f"WARN: {args.wilddet3d_ckpt} not found; encoders use base weights.")
    model = DenseDet3D.from_wilddet3d(
        ckpt_path=ckpt, fpn_level=args.fpn_level, train_fusion=True,
        device=args.device,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train/1e6:.2f}M (dense head + fusion)")

    loss_fn = DenseDet3DLoss()
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    wb = None
    if args.wandb:
        import wandb

        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name,
            config=vars(args),
            tags=["jenga", "dense", "9dof", "prompt-free"],
        )
        print(f"[wandb] logging to {args.wandb_project} as '{args.wandb_run_name}'")

    os.makedirs(args.out, exist_ok=True)
    gstep = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        agg = {"total": 0.0, "hm": 0.0, "rot_deg": 0.0, "npos": 0.0, "n": 0}
        for batch in loader:
            batch = move(batch, args.device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                pred = model(batch["image"], batch["depth"], batch["K"])
            pred = {k: v.float() for k, v in pred.items()}
            tgt = run_targets(batch, pred, args.size, args.device)
            losses = loss_fn(pred, tgt)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(opt)
            scaler.update()

            agg["total"] += losses["total"].item()
            agg["hm"] += losses["heatmap"].item()
            agg["rot_deg"] += losses["rot_deg"].item()
            agg["npos"] += losses["num_pos"].item()
            agg["n"] += 1
            gstep += 1
            if wb is not None:
                wb.log(
                    {
                        "train/loss": losses["total"].item(),
                        "train/heatmap": losses["heatmap"].item(),
                        "train/offset": losses["offset"].item(),
                        "train/depth": losses["depth"].item(),
                        "train/size": losses["size"].item(),
                        "train/rot": losses["rot"].item(),
                        "train/rot_deg": losses["rot_deg"].item(),
                        "train/num_pos": losses["num_pos"].item(),
                        "lr": opt.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    },
                    step=gstep,
                )
        sched.step()
        n = max(agg["n"], 1)
        print(
            f"epoch {epoch+1:2d}/{args.epochs} | loss {agg['total']/n:.4f} "
            f"(hm {agg['hm']/n:.4f}) | rot {agg['rot_deg']/n:.1f}deg | "
            f"pos/iter {agg['npos']/n:.0f} | {time.time()-t0:.0f}s",
            flush=True,
        )
        if wb is not None:
            wb.log(
                {
                    "epoch/loss": agg["total"] / n,
                    "epoch/heatmap": agg["hm"] / n,
                    "epoch/rot_deg": agg["rot_deg"] / n,
                    "epoch": epoch + 1,
                },
                step=gstep,
            )
        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)},
            os.path.join(args.out, "dense_9dof_last.pt"),
        )
    print(f"saved -> {os.path.join(args.out, 'dense_9dof_last.pt')}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
