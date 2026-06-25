"""Anyware capture-scene dataset config (warehouse cartons, 9-DoF)."""

from __future__ import annotations

import os
from collections.abc import Sequence

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe

from wilddet3d.data.datasets.anyware_scenes import (
    AnywareScenes,
    get_anyware_class_map,
    get_anyware_det_map,
)

from .transform import get_test_transforms_cfg, get_train_transforms_cfg

# Text prompts used for the single "box" category (open-vocab model).
ANYWARE_TEXT_PROMPT_MAPPING = {
    "box": {"prompt": "cardboard box"},
}


def get_anyware_dataset_cfg(
    data_root: str,
    datasets: Sequence[str],
    data_backend: None | ConfigDict = None,
    remove_empty: bool = True,
    with_depth: bool = False,
    cache_as_binary: bool = False,
    truncation_thres: float = 0.66,
    visibility_thres: float = 0.0,
    min_height_thres: float = 0.02,
    max_height_thres: float = 1.50,
) -> list[ConfigDict]:
    """Dataset configs for AnywareScenes.

    Relaxed truncation threshold: pole cameras see heavily cropped
    boxes at image borders that are still useful supervision.
    """
    cached_dir = os.path.join(data_root, "cache")

    dataset_cfg_list = []
    for dataset in datasets:
        det_map = get_anyware_det_map(dataset_name=dataset, data_root=data_root)
        class_map = get_anyware_class_map(
            dataset_name=dataset, data_root=data_root
        )

        dataset_cfg = class_config(
            AnywareScenes,
            class_map=class_map,
            data_backend=data_backend,
            data_root=data_root,
            dataset_name=dataset,
            det_map=det_map,
            with_depth=with_depth,
            remove_empty=remove_empty,
            data_prefix=None,
            text_prompt_mapping=ANYWARE_TEXT_PROMPT_MAPPING,
            cache_as_binary=cache_as_binary,
            cached_file_path=os.path.join(cached_dir, f"{dataset}.pkl"),
            truncation_thres=truncation_thres,
            visibility_thres=visibility_thres,
            min_height_thres=min_height_thres,
            max_height_thres=max_height_thres,
        )

        dataset_cfg_list.append(dataset_cfg)

    return dataset_cfg_list


def get_anyware_train_cfg(
    data_root: str = "data/anyware_scenes",
    train_datasets: Sequence[str] = ("AnywareScenes_train",),
    data_backend: None | ConfigDict = None,
    shape: tuple[int, int] = (1008, 1008),
    cache_as_binary: bool = False,
) -> ConfigDict:
    """Train config for AnywareScenes (with sensor depth supervision)."""
    train_dataset_cfg = get_anyware_dataset_cfg(
        data_root=data_root,
        datasets=train_datasets,
        data_backend=data_backend,
        remove_empty=True,
        with_depth=True,
        cache_as_binary=cache_as_binary,
    )

    train_preprocess_cfg = get_train_transforms_cfg(shape=shape)

    return class_config(
        DataPipe,
        datasets=train_dataset_cfg,
        preprocess_fn=train_preprocess_cfg,
    )


def get_anyware_test_cfg(
    data_root: str = "data/anyware_scenes",
    test_datasets: Sequence[str] = ("AnywareScenes_val",),
    data_backend: None | ConfigDict = None,
    with_depth: bool = True,
    shape: tuple[int, int] = (1008, 1008),
    cache_as_binary: bool = False,
) -> ConfigDict:
    """Test config for AnywareScenes."""
    test_dataset_cfg = get_anyware_dataset_cfg(
        data_root=data_root,
        datasets=test_datasets,
        data_backend=data_backend,
        remove_empty=False,
        with_depth=with_depth,
        cache_as_binary=cache_as_binary,
    )

    test_preprocess_cfg = get_test_transforms_cfg(shape=shape)

    return class_config(
        DataPipe,
        datasets=test_dataset_cfg,
        preprocess_fn=test_preprocess_cfg,
    )
