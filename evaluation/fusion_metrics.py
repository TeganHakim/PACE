"""Reference-based robustness metrics on aligned 256x256 grayscale outputs.

All scores operate on saved fused PNGs (not transient float tensors). Reported
transfer ratio is the *observed* perturbation-transfer ratio at this resolution;
JPEG encode/decode differences can influence it and are documented by the run.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

SIZE = (256, 256)
EPS = 1e-9


def gray(path: Path) -> np.ndarray:
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise FileNotFoundError(path)
    return cv2.resize(x, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.


def mask_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, SIZE, interpolation=cv2.INTER_NEAREST) > 127


def transfer_ratio(fused_degraded, fused_clean, source_degraded, source_clean, mask):
    """L2 norm of fusion shift / L2 norm of source shift in corrupted region.

    NaN when source perturbation is numerically zero; such cases are excluded
    from aggregates instead of artificially setting ratio to zero.
    """
    denominator = np.linalg.norm((source_degraded - source_clean)[mask].astype(np.float64))
    if denominator < EPS or not np.any(mask):
        return float('nan')
    numerator = np.linalg.norm((fused_degraded - fused_clean)[mask].astype(np.float64))
    return float(numerator / denominator)


def edges(x):
    return cv2.Canny(np.uint8(np.clip(x, 0, 1) * 255), 75, 150) > 0


def edge_f1(fused_degraded, fused_clean, tolerance=1):
    """Edge-map stability against *same method's clean fusion*, 1-px tolerance."""
    pred = edges(fused_degraded)
    ref = edges(fused_clean)
    if not pred.any() and not ref.any():
        return 1.0
    if not pred.any() or not ref.any():
        return 0.0
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    pred_d = cv2.dilate(pred.astype(np.uint8), kernel).astype(bool)
    ref_d = cv2.dilate(ref.astype(np.uint8), kernel).astype(bool)
    p = (pred & ref_d).sum() / pred.sum()
    r = (ref & pred_d).sum() / ref.sum()
    return float(2*p*r / (p+r)) if p+r > 0 else 0.0


def gradient(x):
    x = np.asarray(x, dtype=np.float32)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def qabf(rgb, ir, fused):
    """Xydeas-Petrovic Q_AB/F with Sobel gradients; clean sources only.

    Edge strength and orientation similarities are weighted by source edge
    magnitudes, using common logistic constants (Xydeas & Petrovic, 2000).
    """
    g_f, a_f = gradient(fused)
    numerator, denominator = 0., 0.
    for source in (rgb, ir):
        g_s, a_s = gradient(source)
        g_ratio = np.divide(np.minimum(g_s, g_f), np.maximum(g_s, g_f),
                            out=np.zeros_like(g_s), where=np.maximum(g_s, g_f)>EPS)
        # Orientation difference is modulo pi: edge orientation is unsigned.
        delta = np.abs(a_s - a_f)
        delta = np.mod(delta, np.pi)
        delta = np.minimum(delta, np.pi - delta)
        a_similarity = 1. - 2.*delta/np.pi
        q_g = .9994/(1. + np.exp(np.clip(-15.*(g_ratio-.5), -80, 80)))
        q_a = .9879/(1. + np.exp(np.clip(-22.*(a_similarity-.8), -80, 80)))
        weight = g_s.astype(np.float64)
        numerator += float((q_g * q_a * weight).sum())
        denominator += float(weight.sum())
    return float(numerator/denominator) if denominator > EPS else float('nan')


def calculate(degraded_root: Path, clean_root: Path, method_dir: Path,
              clean_method_dir: Path, pair_id: str, modality: str):
    d = gray(method_dir/'fused'/f'{pair_id}.png')
    c = gray(clean_method_dir/'fused'/f'{pair_id}.png')
    source_d = gray(degraded_root/'test'/modality/f'{pair_id}.jpg')
    source_c = gray(clean_root/'test'/modality/f'{pair_id}.jpg')
    rgb_c = gray(clean_root/'test'/'rgb'/f'{pair_id}.jpg')
    ir_c = gray(clean_root/'test'/'ir'/f'{pair_id}.jpg')
    mask = mask_image(degraded_root/'test'/'degradation_masks'/f'{pair_id}.png')
    return {
        'degradation_transfer_ratio': transfer_ratio(d,c,source_d,source_c,mask),
        'edge_f1': edge_f1(d,c),
        'qabf': qabf(rgb_c,ir_c,d),
    }
