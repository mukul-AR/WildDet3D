"""Stage 5: full WildDet3D 9-DoF Anyware training (single 24 GB GPU).

Runnable fine-tune of the real WildDet3D (SAM3 ViT-L + LingBot-Depth DINOv2 +
EarlyDepthFusion) on Anyware warehouse scenes with full 9-DoF rotation
(symmetry-aware cuboid loss), starting from the released stage-2 checkpoint.

Key differences from stage4a:
- SAM3 is built WITHOUT the gated facebook/sam3 download (load_from_HF=False);
  all encoder weights come from the stage-2 checkpoint (vis4d loads it
  non-strict, stripping the ``model.`` prefix), so only the new 9-DoF head
  trains from scratch.
- Tuned for a single 24 GB GPU: small per-GPU batch + grad accumulation,
  activation checkpointing, bf16 (set MIXED_PRECISION=bf16).
- Validation is disabled (the 3D-IoU evaluator needs vis4d_cuda_ops, which is
  not built for sm_120); the run produces trained checkpoints as output.

Hyperparameters are overridable via env vars for sweeping:
  WD3D_EPOCHS, WD3D_LR, WD3D_BS, WD3D_ACCUM, WD3D_SHAPE,
  WD3D_BACKBONE_FREEZE, WD3D_LINGBOT_FREEZE, WD3D_LIMIT_TRAIN_BATCHES.

Usage:
    MIXED_PRECISION=bf16 .venv/bin/vis4d fit \
        --config configs/training/stage5_anyware_9dof_train.py --gpus 1 \
        --ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt
"""

from __future__ import annotations

import os

from vis4d.config import class_config
from vis4d.config.typing import DataConfig, ExperimentConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.data.io import FileBackend
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import (
    get_default_cfg,
    get_default_callbacks_cfg,
    get_inference_dataloaders_cfg,
)

from configs.base.connector import get_wilddet3d_data_connector_cfg
from configs.base.data import (
    wilddet3d_5mode_collate_fn,
    wilddet3d_test_collate_fn,
)
from configs.base.dataset.anyware import (
    get_anyware_test_cfg,
    get_anyware_train_cfg,
)
from configs.base.loss import get_wilddet3d_loss_cfg
from configs.base.model import (
    get_wilddet3d_cfg,
    get_wilddet3d_hyperparams_cfg,
)
from configs.base.optim import get_wilddet3d_optim_cfg
from configs.base.pl import get_pl_cfg
from wilddet3d.data.samplers import build_train_dataloader_with_ratios


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


EXPERIMENT_NAME = "wilddet3d_stage5_anyware_9dof"

_SHAPE = _env_int("WD3D_SHAPE", 1008)
SAM3_IMAGE_SHAPE = (_SHAPE, _SHAPE)

NUM_EPOCHS = _env_int("WD3D_EPOCHS", 10)
SAMPLES_PER_GPU = _env_int("WD3D_BS", 1)
WORKERS_PER_GPU = _env_int("WD3D_WORKERS", 4)
ACCUM = _env_int("WD3D_ACCUM", 4)
BASE_LR = _env_float("WD3D_LR", 2e-5)
BACKBONE_FREEZE_BLOCKS = _env_int("WD3D_BACKBONE_FREEZE", 28)
# Keep the LingBot depth encoder fully frozen by default (24/24 blocks): the
# depth backbone is a strong geometric prior we don't want to disturb. Override
# with WD3D_LINGBOT_FREEZE to thaw blocks if ever needed.
LINGBOT_ENCODER_FREEZE_BLOCKS = _env_int("WD3D_LINGBOT_FREEZE", 24)
LIMIT_TRAIN_BATCHES = _env_float("WD3D_LIMIT_TRAIN_BATCHES", 1.0)

ANYWARE_DATA_ROOT = "data/anyware_scenes"


def get_config() -> ExperimentConfig:
    """Full WildDet3D 9-DoF single-GPU training config."""
    config = get_default_cfg(exp_name=EXPERIMENT_NAME)
    config.use_checkpoint = True  # activation checkpointing

    params = get_wilddet3d_hyperparams_cfg(
        num_epochs=NUM_EPOCHS,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        base_lr=BASE_LR,
        accumulate_grad_batches=ACCUM,
        step_1=int(NUM_EPOCHS * 2 / 3),
        step_2=int(NUM_EPOCHS * 5 / 6),
        nms=True,
        nms_iou_threshold=0.4,
        score_threshold=0.05,
    )

    # ---- Model: prebuild SAM3 structure WITHOUT the gated HF download ----
    from sam3.model_builder import build_sam3_image_model

    sam3_model_cfg = class_config(
        build_sam3_image_model,
        checkpoint_path=None,
        load_from_HF=False,  # weights come from the stage-2 checkpoint
        device="cpu",
        eval_mode=False,
        enable_segmentation=False,
    )

    config.model, box_coder = get_wilddet3d_cfg(
        params,
        geometry_backend_type="lingbot_depth",
        sam3_model=sam3_model_cfg,
        lingbot_encoder_freeze_blocks=LINGBOT_ENCODER_FREEZE_BLOCKS,
        backbone_freeze_blocks=BACKBONE_FREEZE_BLOCKS,
        canonical_rotation=False,
        ambiguous_rotation=False,
        symmetry="cuboid",
        use_predicted_intrinsics=True,
        use_depth_input_test=True,
        eval_3d_conf_weight=0.5,
    )

    # ---- Data ----
    file_backend = class_config(FileBackend)
    anyware_train = get_anyware_train_cfg(
        data_root=ANYWARE_DATA_ROOT,
        train_datasets=("AnywareScenes_train",),
        data_backend=file_backend,
        shape=SAM3_IMAGE_SHAPE,
    )
    anyware_val = get_anyware_test_cfg(
        data_root=ANYWARE_DATA_ROOT,
        test_datasets=("AnywareScenes_val",),
        data_backend=file_backend,
        with_depth=True,
        shape=SAM3_IMAGE_SHAPE,
    )

    data = DataConfig()
    train_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )
    data.train_dataloader = class_config(
        build_train_dataloader_with_ratios,
        dataset=class_config(DataPipe, datasets=[anyware_train]),
        target_proportions=[1.0],
        epoch_dataset_idx=0,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        batchprocess_fn=train_batchprocess_cfg,
        collate_fn=wilddet3d_5mode_collate_fn,
    )
    test_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )
    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=class_config(DataPipe, datasets=[anyware_val]),
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=1,
        workers_per_gpu=WORKERS_PER_GPU,
        collate_fn=wilddet3d_test_collate_fn,
    )
    config.data = data

    # ---- Loss / Optimizer ----
    config.loss = get_wilddet3d_loss_cfg(
        params,
        box_coder=box_coder,
        use_3d_conf=True,
        use_ignore_suppress=True,
        presence_loss_weight=5.0,
        loss_rot_weight=1.0,
    )
    config.optimizers = get_wilddet3d_optim_cfg(params)

    # ---- Callbacks / Connectors ----
    # Training-only run: logging callbacks only (the 3D evaluator needs
    # vis4d_cuda_ops, not built for sm_120; validation is disabled below).
    config.callbacks = get_default_callbacks_cfg()
    (
        config.train_data_connector,
        config.test_data_connector,
    ) = get_wilddet3d_data_connector_cfg()

    # ---- PL trainer: single GPU, no validation (eval needs cuda ops) ----
    # NOTE: pl_trainer overrides only stick when applied AFTER value_mode().
    config.pl_trainer = get_pl_cfg(config, params)

    final = config.value_mode()
    final.pl_trainer.strategy = "auto"
    final.pl_trainer.find_unused_parameters = True
    final.pl_trainer.gradient_clip_val = 0.1
    final.pl_trainer.num_sanity_val_steps = 0
    final.pl_trainer.limit_train_batches = LIMIT_TRAIN_BATCHES
    # Validation disabled: the 3D evaluator needs vis4d_cuda_ops (not built
    # for sm_120) and it wastes time on a training-only run.
    final.pl_trainer.limit_val_batches = 0.0
    final.pl_trainer.check_val_every_n_epoch = NUM_EPOCHS + 1
    final.pl_trainer.val_check_interval = 1.0

    # ---- Weights & Biases (opt-in via WD3D_WANDB=1; needs WANDB_API_KEY) ----
    # vis4d routes pl_module.log(...) metrics to trainer.logger, so injecting a
    # WandbLogger here streams all losses/metrics to W&B. Project/entity from
    # env so no secrets are committed.
    if os.environ.get("WD3D_WANDB", "0") == "1":
        from lightning.pytorch.loggers import WandbLogger

        final.pl_trainer.logger = class_config(
            WandbLogger,
            project=os.environ.get("WD3D_WANDB_PROJECT", "jenga-9dof"),
            entity=os.environ.get("WD3D_WANDB_ENTITY", "mukul-ganwal") or None,
            name=os.environ.get("WD3D_RUN_NAME", "jenga-stage5"),
            save_dir="vis4d-workspace",
            tags=["jenga", "stage5", "9dof", "wilddet3d"],
        )
    return final
