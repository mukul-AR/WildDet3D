#!/usr/bin/env python3
"""Train the two-stage 9-DoF Anyware box detector (pipeline verification).

Runs the full data -> Stage-1 (RGB-D -> visual OBB) -> frame transform ->
Stage-2 (walls + SKU -> actual OBB) -> loss -> optimizer -> checkpoint loop
for a small number of epochs (default 10) to prove the workflow is sound.

Prerequisite: build the cache first, e.g.
    python scripts/data_prep/anyware/build_two_stage_dataset.py --split train
    python scripts/data_prep/anyware/build_two_stage_dataset.py --split val

Usage:
    python scripts/train_two_stage.py --epochs 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wilddet3d.twostage.dataset import TwoStageAnywareDataset  # noqa: E402
from wilddet3d.twostage.losses import NineDoFLoss  # noqa: E402
from wilddet3d.twostage.model import TwoStage9DoF  # noqa: E402
from wilddet3d.twostage.rotation_utils import (  # noqa: E402
    rad2deg,
    symmetry_min_geodesic,
)


def to_device(batch: dict, device: str) -> dict:
    """Move all tensors in a batch dict to ``device``."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def stage1_targets(batch: dict) -> dict:
    """Stage-1 (camera-frame) supervision targets."""
    return {
        "center": batch["gt_cam_center"],
        "size": batch["gt_size"],
        "rot6d": batch["gt_cam_rot6d"],
        "presence": batch["presence"],
    }


def stage2_targets(batch: dict) -> dict:
    """Stage-2 (base_link-frame) supervision targets."""
    return {
        "center": batch["gt_base_center"],
        "size": batch["gt_size"],
        "rot6d": batch["gt_base_rot6d"],
    }


@torch.no_grad()
def evaluate(model: TwoStage9DoF, loader: DataLoader, device: str) -> dict:
    """Compute per-stage geometric error metrics on a loader."""
    model.eval()
    agg = {
        "s1_center": 0.0,
        "s1_size": 0.0,
        "s1_rot": 0.0,
        "s2_center": 0.0,
        "s2_size": 0.0,
        "s2_rot": 0.0,
        "n": 0,
    }
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        n = batch["crop"].shape[0]
        s1 = out["stage1"]
        agg["s1_center"] += (
            (s1["center"] - batch["gt_cam_center"]).norm(dim=-1).sum().item()
        )
        agg["s1_size"] += (
            (s1["size"] - batch["gt_size"]).abs().mean(-1).sum().item()
        )
        agg["s1_rot"] += rad2deg(
            symmetry_min_geodesic(s1["rot6d"], batch["gt_cam_rot6d"])
        ).sum().item()
        s2 = out["stage2"]
        agg["s2_center"] += (
            (s2["center"] - batch["gt_base_center"]).norm(dim=-1).sum().item()
        )
        agg["s2_size"] += (
            (s2["size"] - batch["gt_size"]).abs().mean(-1).sum().item()
        )
        agg["s2_rot"] += rad2deg(
            symmetry_min_geodesic(s2["rot6d"], batch["gt_base_rot6d"])
        ).sum().item()
        agg["n"] += n
    n = max(agg["n"], 1)
    return {k: v / n for k, v in agg.items() if k != "n"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/anyware_scenes")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--teacher-force-p", type=float, default=0.5)
    ap.add_argument(
        "--occlusion-aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="augment the Stage-2 input with simulated partial observability",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="ckpt/two_stage_9dof")
    args = ap.parse_args()

    cache = os.path.join(args.data_root, "two_stage_cache")
    train_ds = TwoStageAnywareDataset(os.path.join(cache, "train"))
    val_dir = os.path.join(cache, "val")
    val_ds = (
        TwoStageAnywareDataset(val_dir) if os.path.isdir(val_dir) else None
    )
    print(
        f"train boxes: {len(train_ds)}"
        + (f" | val boxes: {len(val_ds)}" if val_ds else " | no val split")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        if val_ds
        else None
    )

    model = TwoStage9DoF(
        stage1_kwargs={"ctx_dim": train_ds.ctx.shape[1]},
        stage2_kwargs={"max_sku": train_ds.skus.shape[1]},
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M | device: {args.device}")

    loss1 = NineDoFLoss(w_center=1.0, w_size=1.0, w_rot=1.0, w_presence=0.1)
    loss2 = NineDoFLoss(w_center=1.0, w_size=1.0, w_rot=1.0, w_presence=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run = {"total": 0.0, "s1": 0.0, "s2": 0.0, "n": 0}
        for batch in train_loader:
            batch = to_device(batch, args.device)
            teacher = {
                "center": batch["gt_base_center"],
                "size": batch["gt_size"],
                "rot6d": batch["gt_base_rot6d"],
            }
            out = model(
                batch,
                teacher_visual_base=teacher,
                teacher_force_p=args.teacher_force_p,
                occlusion_aug=args.occlusion_aug,
            )
            l1 = loss1(out["stage1"], stage1_targets(batch), batch["confidence"])
            l2 = loss2(out["stage2"], stage2_targets(batch), batch["confidence"])
            total = l1["total"] + l2["total"]

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = batch["crop"].shape[0]
            run["total"] += total.item() * bs
            run["s1"] += l1["total"].item() * bs
            run["s2"] += l2["total"].item() * bs
            run["n"] += bs
        sched.step()

        n = max(run["n"], 1)
        dt = time.time() - t0
        msg = (
            f"epoch {epoch + 1:2d}/{args.epochs} | "
            f"loss {run['total']/n:.4f} (s1 {run['s1']/n:.4f}, "
            f"s2 {run['s2']/n:.4f}) | {dt:.1f}s"
        )
        if val_loader is not None:
            m = evaluate(model, val_loader, args.device)
            msg += (
                f" | val s1[c {m['s1_center']*100:.1f}cm "
                f"sz {m['s1_size']*100:.1f}cm rot {m['s1_rot']:.1f}deg] "
                f"s2[c {m['s2_center']*100:.1f}cm sz {m['s2_size']*100:.1f}cm "
                f"rot {m['s2_rot']:.1f}deg]"
            )
        print(msg, flush=True)

    ckpt_path = os.path.join(args.out, "two_stage_9dof_last.pt")
    torch.save(
        {"model": model.state_dict(), "epochs": args.epochs, "args": vars(args)},
        ckpt_path,
    )
    print(f"saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
