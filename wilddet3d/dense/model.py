"""Prompt-free dense 3D detector: SAM3 backbone + depth fusion -> dense head.

Reuses WildDet3D's SAM3 image backbone, LingBot-Depth backend and
EarlyDepthFusion (initialised from a trained WildDet3D checkpoint, run frozen
under no_grad for speed/memory) and replaces the text/prompt grounding decoder
+ query 3D head with a dense convolutional 9-DoF head over the fused FPN
features. No prompts, no text, no per-object input.
"""

from __future__ import annotations

import contextlib

import torch
from torch import Tensor, nn

from wilddet3d.dense.head import DenseConvHead


class DenseDet3D(nn.Module):
    """Free-running dense 9-DoF detector.

    Args:
        backbone: SAM3 image backbone (provides ``forward_image`` -> FPN).
        geometry_backend: LingBot-Depth backend (-> depth latents).
        early_depth_fusion: fuses depth latents into the FPN features.
        fpn_level: which fused FPN level to predict on (0=finest .. 2=coarsest).
        in_ch: fused feature channels (256 for SAM3).
        train_fusion: if True, the fusion module is trainable; backbone +
            depth backend are always frozen (run under no_grad).
        head_kwargs: kwargs for :class:`DenseConvHead`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        geometry_backend: nn.Module,
        early_depth_fusion: nn.Module,
        fpn_level: int = 1,
        in_ch: int = 256,
        train_fusion: bool = True,
        train_encoders: bool = False,
        train_depth_encoder: bool = False,
        depth_unfreeze_blocks: int = 0,
        head_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone.eval()
        self.geometry_backend = geometry_backend.eval()
        self.early_depth_fusion = early_depth_fusion
        self.fpn_level = fpn_level
        self.train_fusion = train_fusion
        self.train_encoders = train_encoders  # SAM3 RGB backbone
        self.train_depth_encoder = train_depth_encoder  # LingBot depth backbone
        self.head = DenseConvHead(in_ch=in_ch, **(head_kwargs or {}))

        # Each encoder is frozen by default and can be unfrozen independently for
        # a low-LR fine-tune. They stay in eval() mode regardless (deterministic;
        # LayerNorm needs no running stats) — see train().
        for p in self.backbone.parameters():
            p.requires_grad_(train_encoders)
        # Depth (LingBot) backbone. depth_unfreeze_blocks>0 unfreezes ONLY the
        # last N encoder blocks (+ final norm) for a low-LR, memory-bounded
        # fine-tune; 0 keeps the legacy all-or-nothing behaviour. The optimizer
        # groups by requires_grad, so a partial unfreeze is picked up cleanly.
        self.depth_unfreeze_blocks = depth_unfreeze_blocks
        if train_depth_encoder and depth_unfreeze_blocks > 0:
            for p in self.geometry_backend.parameters():
                p.requires_grad_(False)
            bb = self.geometry_backend.encoder.backbone
            nb = len(bb.blocks)
            k = min(depth_unfreeze_blocks, nb)
            for i in range(nb - k, nb):
                for p in bb.blocks[i].parameters():
                    p.requires_grad_(True)
            for p in bb.norm.parameters():
                p.requires_grad_(True)
            print(f"[DenseDet3D] depth backbone: unfroze last {k}/{nb} blocks + norm")
        else:
            for p in self.geometry_backend.parameters():
                p.requires_grad_(train_depth_encoder)
        if not train_fusion:
            for p in self.early_depth_fusion.parameters():
                p.requires_grad_(False)
        self.register_buffer(
            "_imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "DenseDet3D":
        """Keep frozen encoders in eval mode regardless of training flag."""
        super().train(mode)
        self.backbone.eval()
        self.geometry_backend.eval()
        if not self.train_fusion:
            self.early_depth_fusion.eval()
        return self

    def _to_sam3(self, images: Tensor) -> Tensor:
        """ImageNet-normalised -> SAM3 [-1, 1] normalisation."""
        images_01 = images * self._imagenet_std + self._imagenet_mean
        return (images_01 - 0.5) / 0.5

    def _fused_feats(
        self, images: Tensor, depth: Tensor, k: Tensor, rgb_fpn: Tensor | None = None
    ) -> Tensor:
        """Encoders + depth fusion -> the single fused FPN level the head reads.

        SAM3 forward runs under ``no_grad`` when frozen; under grad when
        ``train_encoders`` is set. Fusion is per-level independent and the head
        consumes only ``self.fpn_level``, so we fuse just that level.

        When ``rgb_fpn`` is given (a precomputed SAM3 FPN tensor ``[B,C,Hf,Wf]``
        for ``self.fpn_level``), the frozen SAM3 forward is skipped entirely and
        the cached feature is fused with the (always recomputed) depth latents.
        Valid only while the RGB backbone stays frozen — its output is then a
        deterministic function of the input image. See
        ``scripts/precompute_feat_cache.py``.
        """
        _, _, h, w = images.shape
        if rgb_fpn is None:
            sam_ctx = contextlib.nullcontext() if self.train_encoders else torch.no_grad()
            with sam_ctx:
                backbone_out = self.backbone.forward_image(self._to_sam3(images))
            backbone_fpn = backbone_out["backbone_fpn"]
            if not isinstance(backbone_fpn, list):
                backbone_fpn = [backbone_fpn]
            visual = [backbone_fpn[self.fpn_level]]
        else:
            visual = [rgb_fpn]
        dep_ctx = contextlib.nullcontext() if self.train_depth_encoder else torch.no_grad()
        with dep_ctx:
            geom = self.geometry_backend(
                images=images,
                depth_feats=None,
                intrinsics=k,
                image_hw=(h, w),
                depth_gt=depth,
                depth_mask=None,
                padding=None,
            )
        depth_latents = geom["depth_latents"]
        depth_latents_hw = geom.get("aux", {}).get("depth_latents_hw")

        fused = self.early_depth_fusion(
            visual_feats=visual,
            depth_latents=depth_latents,
            depth_latents_hw=depth_latents_hw,
        )
        if not isinstance(fused, (list, tuple)):
            fused = [fused]
        return fused[0]

    def forward(
        self,
        images: Tensor,
        depth: Tensor,
        k: Tensor,
        return_feat: bool = False,
        rgb_fpn: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Args: images ``[B,3,H,W]`` (ImageNet-norm), depth ``[B,1,H,W]`` (m),
        k ``[B,3,3]``. ``rgb_fpn`` optionally supplies the precomputed SAM3 FPN
        level (skips the frozen RGB forward). Returns dense ``heatmap`` + ``reg``
        maps; when ``return_feat`` also the fused ``feat`` ``[B,C,Hf,Wf]`` +
        ``stride``."""
        feat = self._fused_feats(images, depth, k, rgb_fpn=rgb_fpn)
        out = self.head(feat)
        if return_feat:
            out["feat"] = feat
            out["stride"] = images.shape[-1] / feat.shape[-2]
        return out

    @classmethod
    def from_wilddet3d(
        cls,
        ckpt_path: str | None = None,
        fpn_level: int = 1,
        train_fusion: bool = True,
        train_encoders: bool = False,
        train_depth_encoder: bool = False,
        depth_unfreeze_blocks: int = 0,
        head_kwargs: dict | None = None,
        device: str = "cuda",
    ) -> "DenseDet3D":
        """Build from a trained WildDet3D checkpoint (extract + free the rest).

        Constructs the full WildDet3D (SAM3 structure-only, no gated download),
        loads ``ckpt_path`` non-strict (strips the ``model.`` PL prefix), keeps
        only backbone + depth backend + fusion, and frees the grounding
        decoder / query 3D head.
        """
        from vis4d.config import class_config
        from vis4d.config.config_dict import instantiate_classes

        from configs.base.model import (
            get_wilddet3d_cfg,
            get_wilddet3d_hyperparams_cfg,
        )

        params = get_wilddet3d_hyperparams_cfg(num_epochs=1, samples_per_gpu=1)
        from sam3.model_builder import build_sam3_image_model

        sam3_cfg = class_config(
            build_sam3_image_model,
            checkpoint_path=None,
            load_from_HF=False,
            device="cpu",
            eval_mode=False,
            enable_segmentation=False,
        )
        model_cfg, _ = get_wilddet3d_cfg(
            params,
            geometry_backend_type="lingbot_depth",
            sam3_model=sam3_cfg,
            lingbot_encoder_freeze_blocks=24,
            backbone_freeze_blocks=32,
            canonical_rotation=False,
            symmetry="cuboid",
            use_predicted_intrinsics=True,
            use_depth_input_test=True,
        )
        wd = instantiate_classes(model_cfg)
        if ckpt_path is not None:
            import re

            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck)
            sd = {re.sub(r"^model\.", "", k): v for k, v in sd.items()}
            missing, unexpected = wd.load_state_dict(sd, strict=False)
            print(
                f"[DenseDet3D] loaded {ckpt_path}: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        model = cls(
            backbone=wd.sam3.backbone,
            geometry_backend=wd.geometry_backend,
            early_depth_fusion=wd.early_depth_fusion,
            fpn_level=fpn_level,
            train_fusion=train_fusion,
            train_encoders=train_encoders,
            train_depth_encoder=train_depth_encoder,
            depth_unfreeze_blocks=depth_unfreeze_blocks,
            head_kwargs=head_kwargs,
        )
        del wd
        torch.cuda.empty_cache()
        return model.to(device)
