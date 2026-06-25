"""Stage 4a: Anyware 9-DoF Single-View Fine-Tune.

Fine-tunes the full WildDet3D (stage2 checkpoint) on Anyware warehouse
scenes with full 9-DoF rotation (no canonical_rotation), symmetry-aware
rotation loss, and sensor-depth supervision.

Key differences from stage2:
- canonical_rotation=False, symmetry="cuboid" (full 9-DoF)
- Data: Anyware scenes ONLY (no Omni3D replay in 4a; add replay in 4b)
- use_depth_input_test=True (sensor depth always available at inference)
- backbone_freeze=28, depth_freeze=21 (same recipe as stage2)
- Small LR because we're fine-tuning a strong pretrained model

Usage:
    vis4d fit --config configs/training/stage4a_anyware_9dof.py --gpus 1 \
        --ckpt <path_to_stage2_or_stage3_checkpoint>
"""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import DataConfig, ExperimentConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.data.io import FileBackend
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import (
    get_default_cfg,
    get_inference_dataloaders_cfg,
)

from configs.base.callback import get_callback_cfg
from configs.base.connector import get_wilddet3d_data_connector_cfg
from configs.base.data import (
    wilddet3d_5mode_collate_fn,
    wilddet3d_test_collate_fn,
)
from configs.base.dataset.anyware import (
    get_anyware_train_cfg,
    get_anyware_test_cfg,
)
from configs.base.loss import get_wilddet3d_loss_cfg
from configs.base.model import (
    get_wilddet3d_cfg,
    get_wilddet3d_hyperparams_cfg,
)
from configs.base.optim import get_wilddet3d_optim_cfg
from configs.base.pl import get_pl_cfg

from wilddet3d.data.samplers import build_train_dataloader_with_ratios


# ============================================================
# Experiment parameters
# ============================================================
EXPERIMENT_NAME = "wilddet3d_stage4a_anyware_9dof"

SAM3_IMAGE_SHAPE = (1008, 1008)

# Hyperparameters — small LR for fine-tuning
NUM_EPOCHS = 20
SAMPLES_PER_GPU = 4
WORKERS_PER_GPU = 4
BASE_LR = 2e-5      # 5x lower than stage2
STEP_1 = 12
STEP_2 = 16

# Freeze settings (same as stage2)
BACKBONE_FREEZE_BLOCKS = 28
LINGBOT_ENCODER_FREEZE_BLOCKS = 21

# Data root — relative to working dir
ANYWARE_DATA_ROOT = "data/anyware_scenes"


def get_config() -> ExperimentConfig:
    """Get Stage 4a config: Anyware 9-DoF single-view fine-tune."""
    config = get_default_cfg(exp_name=EXPERIMENT_NAME)
    config.use_checkpoint = True

    # ==================== Hyperparameters ====================
    params = get_wilddet3d_hyperparams_cfg(
        num_epochs=NUM_EPOCHS,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        base_lr=BASE_LR,
        step_1=STEP_1,
        step_2=STEP_2,
        nms=True,
        nms_iou_threshold=0.4,      # tighter NMS: warehouse scenes dense
        score_threshold=0.05,
    )

    # ==================== Model ====================
    # canonical_rotation=False + symmetry="cuboid" = full 9-DoF
    config.model, box_coder = get_wilddet3d_cfg(
        params,
        geometry_backend_type="lingbot_depth",
        lingbot_encoder_freeze_blocks=LINGBOT_ENCODER_FREEZE_BLOCKS,
        backbone_freeze_blocks=BACKBONE_FREEZE_BLOCKS,
        canonical_rotation=False,   # CRITICAL: full 9-DoF
        ambiguous_rotation=False,
        symmetry="cuboid",          # symmetry-aware rotation loss
        use_predicted_intrinsics=True,
        use_depth_input_test=True,  # sensor depth at inference
        eval_3d_conf_weight=0.5,
    )

    # ==================== Data ====================
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

    # ==================== Loss ====================
    config.loss = get_wilddet3d_loss_cfg(
        params,
        box_coder=box_coder,
        use_3d_conf=True,
        use_ignore_suppress=True,
        presence_loss_weight=5.0,
        loss_rot_weight=1.0,
    )

    # ==================== Optimizer ====================
    config.optimizers = get_wilddet3d_optim_cfg(params)

    # ==================== Callbacks ====================
    config.callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        open_test_datasets=[],
    )

    # ==================== Connectors ====================
    (
        config.train_data_connector,
        config.test_data_connector,
    ) = get_wilddet3d_data_connector_cfg()

    # ==================== PL Trainer ====================
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
