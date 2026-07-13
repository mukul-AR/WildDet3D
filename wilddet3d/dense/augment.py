"""Train-time appearance augmentation for the JENGA sim data.

Purely photometric (RGB) + sensor-noise (depth) — NO geometric transforms, so
boxes, intrinsics, and the 1008 RoPE grid are untouched. Targets the measured
sim->real gap: clean rendered RGB/depth vs real camera RGB + ZED stereo depth
(rot 9 deg / ctr 17 cm on real vs 0.3 deg / 1.6 cm on sim, identical intrinsics).

RGB (uint8, post-resize):  brightness/contrast, gamma, per-channel color
balance, hue/saturation, defocus blur, sensor noise, JPEG artifacts.
Depth (float32 m, 0 = invalid): low-frequency multiplicative warp (stereo
calibration/matching bias), relative white noise, speckle holes, edge dropout
(stereo edge fattening), 1 mm quantization.

Each component fires independently (p per component); ~5% of samples pass
through completely clean so the model never drifts from the clean-val regime.
"""
from __future__ import annotations

import cv2
import numpy as np


def _rand_field(rng, h: int, w: int, coarse: int, sigma: float) -> np.ndarray:
    """Low-frequency random field in [1-3s, 1+3s]: coarse gaussian grid upsampled."""
    g = rng.normal(0.0, sigma, size=(coarse, coarse)).astype(np.float32)
    return 1.0 + cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def augment_rgb(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """uint8 [H,W,3] -> uint8 [H,W,3]."""
    if rng.random() < 0.05:  # clean passthrough
        return rgb
    x = rgb.astype(np.float32)
    if rng.random() < 0.8:  # brightness / contrast
        alpha = rng.uniform(0.75, 1.3)
        beta = rng.uniform(-30.0, 30.0)
        x = alpha * (x - 128.0) + 128.0 + beta
    if rng.random() < 0.5:  # gamma
        g = rng.uniform(0.7, 1.4)
        x = np.clip(x, 0, 255)
        x = 255.0 * np.power(x / 255.0, g)
    if rng.random() < 0.5:  # color balance (white-balance error)
        x = x * rng.uniform(0.9, 1.1, size=3).astype(np.float32)
    if rng.random() < 0.4:  # hue / saturation
        hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if rng.random() < 0.3:  # defocus / motion softness
        sig = rng.uniform(0.5, 1.8)
        x = cv2.GaussianBlur(x, (0, 0), sig)
    if rng.random() < 0.5:  # sensor noise
        x = x + rng.normal(0.0, rng.uniform(2.0, 10.0), size=x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # JPEG artifacts
        q = int(rng.integers(40, 90))
        ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            x = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return x


def augment_depth(depth_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """float32 [H,W] metric depth (0 = invalid) -> same, ZED-like noise."""
    if rng.random() < 0.05:  # clean passthrough
        return depth_m
    h, w = depth_m.shape
    d = depth_m.copy()
    valid = d > 0
    if rng.random() < 0.8:  # low-freq multiplicative bias (stereo matching/calib)
        field = _rand_field(rng, h, w, int(rng.integers(4, 12)), rng.uniform(0.003, 0.010))
        d = d * field
    if rng.random() < 0.8:  # relative white noise
        d = d * (1.0 + rng.normal(0.0, rng.uniform(0.001, 0.004), size=d.shape).astype(np.float32))
    if rng.random() < 0.6:  # speckle holes (textureless / dark dropouts)
        blob = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), rng.uniform(2, 6))
        thr = np.quantile(blob, rng.uniform(0.005, 0.04))
        d[blob < thr] = 0.0
    if rng.random() < 0.5:  # edge fattening/dropout at depth discontinuities
        gx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
        edge = (np.abs(gx) + np.abs(gy)) > 0.10  # >10cm jump across ~1px
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        drop = edge & (rng.random(size=d.shape) < 0.5)
        d[drop] = 0.0
    d[~valid] = 0.0                      # never invent depth where sim had none
    d = np.round(d * 1000.0) / 1000.0    # 1mm quantization (uint16-mm parity)
    return np.clip(d, 0.0, None).astype(np.float32)
