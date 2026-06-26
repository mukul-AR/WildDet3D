"""Canonicalization + scene-catalog helpers for the JENGA Stage-2 head.

Actual boxes are canonicalized so their local axes are ordered by ascending
extent: size becomes an ascending-sorted triple (matching the sorted catalog
dims used for assignment) and the rotation carries the orientation. Column
reordering can flip handedness; we restore a proper rotation by negating the
last axis (a cuboid D2 symmetry, already modded out by the rotation loss).
"""
from __future__ import annotations

import json

import numpy as np
import yaml


def canonicalize_obb(
    size_xyz: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder box axes to ascending extent; return (sorted size, proper R)."""
    perm = np.argsort(size_xyz)
    size_sorted = size_xyz[perm].astype(np.float32)
    R_canon = R[:, perm].astype(np.float32)
    if np.linalg.det(R_canon) < 0:
        R_canon[:, -1] *= -1.0
    return size_sorted, R_canon


def parse_catalog(scene_json_path: str) -> np.ndarray:
    """Distinct ascending-sorted catalog dims ``[K, 3]`` from skus_yaml_string."""
    scene = json.load(open(scene_json_path))
    skus = yaml.safe_load(scene["skus_yaml_string"])["skus"]
    dims = {tuple(sorted(round(float(x), 4) for x in s["geometry"])) for s in skus}
    return np.array(sorted(dims), dtype=np.float32).reshape(-1, 3)


def assign_index(size_sorted: np.ndarray, catalog: np.ndarray) -> int:
    """Index of the nearest (min-L1) catalog row to an ascending-sorted size."""
    return int(np.abs(catalog - size_sorted[None]).sum(axis=1).argmin())
