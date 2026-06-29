"""Precompute & cache the frozen SAM3 RGB FPN level the dense head consumes.

Run ONCE per (wilddet3d-ckpt, fpn-level, size). Writes one fp16 ``.npy`` per
camera view (keyed by md5 of the abspath, matching ``feat_cache_path``) plus a
``meta.json`` manifest. Training with ``--feat-cache-dir <dir>`` then skips the
SAM3 forward (~1.6x faster step). Valid only while the RGB backbone stays
frozen — its FPN output is then a deterministic function of the input image,
and the dataset applies no augmentation, so the cache is exact.

Example:
  PYTHONPATH=. .venv/bin/python scripts/precompute_feat_cache.py \
    --sim-root /home/ubuntu/data_combined --cache-dir ckpt/feat_cache_fpn1 \
    --wilddet3d-ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt --fpn-level 1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wilddet3d.dense.model import DenseDet3D
from wilddet3d.dense.sim_jenga_dataset import SimJengaDataset, feat_cache_path


class _ImgOnly(Dataset):
    """Wraps SimJengaDataset to yield only (preprocessed image, cam_dir)."""

    def __init__(self, ds: SimJengaDataset):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        return self.ds[i]["image"], self.ds.samples[i][0]


def _collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--wilddet3d-ckpt", default="ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt")
    ap.add_argument("--fpn-level", type=int, default=1)
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    ckpt = args.wilddet3d_ckpt if os.path.exists(args.wilddet3d_ckpt) else None
    if ckpt is None:
        raise SystemExit(
            f"{args.wilddet3d_ckpt} not found. The cache MUST be built from the same "
            f"checkpoint training loads, or features will be wrong."
        )

    # Only the SAM3 backbone is used; build with a tiny head to save memory.
    model = DenseDet3D.from_wilddet3d(
        ckpt_path=ckpt, fpn_level=args.fpn_level, train_fusion=False,
        train_encoders=False, train_depth_encoder=False,
        head_kwargs={"feat_ch": 64, "n_convs": 1}, device=args.device,
    )
    model.eval()

    meta_p = os.path.join(args.cache_dir, "meta.json")
    n_done = n_new = 0
    t0 = time.time()
    for split in ("train", "val"):
        ds = SimJengaDataset(args.sim_root, args.size, args.max_scenes,
                             split=split, val_frac=args.val_frac)
        loader = DataLoader(_ImgOnly(ds), batch_size=args.batch_size,
                            num_workers=args.workers, collate_fn=_collate,
                            shuffle=False, pin_memory=True)
        print(f"[{split}] {len(ds)} views", flush=True)
        for imgs, cams in loader:
            paths = [feat_cache_path(args.cache_dir, c) for c in cams]
            todo = [i for i, p in enumerate(paths) if not os.path.exists(p)]
            n_done += len(cams)
            if not todo:
                continue
            sel = imgs[todo].to(args.device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.backbone.forward_image(model._to_sam3(sel))
            fpn = out["backbone_fpn"]
            if not isinstance(fpn, list):
                fpn = [fpn]
            feat = fpn[args.fpn_level].float().cpu().numpy().astype(np.float16)
            if not os.path.exists(meta_p):
                json.dump(
                    {"ckpt": os.path.basename(args.wilddet3d_ckpt),
                     "fpn_level": args.fpn_level, "size": args.size,
                     "shape": list(feat.shape[1:]), "dtype": "float16"},
                    open(meta_p, "w"), indent=2)
                print(f"  wrote {meta_p}: per-view shape {list(feat.shape[1:])}", flush=True)
            for j, i in enumerate(todo):
                np.save(paths[i], feat[j])
            n_new += len(todo)
            if n_done % (args.batch_size * 50) < args.batch_size:
                rate = n_new / max(time.time() - t0, 1e-6)
                print(f"  {n_done} seen / {n_new} written ({rate:.1f}/s)", flush=True)

    print(f"done: {n_done} views, {n_new} newly cached -> {args.cache_dir}", flush=True)


if __name__ == "__main__":
    main()
