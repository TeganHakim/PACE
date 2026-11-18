"""Post-hoc, three-way PACE/CrossFuse/Addition fusion diagnostics.

Reads only artifacts produced by evaluation/eval_metrics.py. It does not run
models or modify the evaluation outputs.

Run from the PACE repository root:
    python evaluation/fusion_diagnostics.py \
        --run evaluation/results/<run-folder> \
        --bootstrap 1000 --examples 12

Outputs are written to <run>/fusion_diagnostics by default:
    per_image_method_metrics.csv
    summary_by_condition.csv
    paired_deltas_vs_addition.csv
    paired_deltas_vs_crossfuse.csv
    figures/*.png
    examples/*.png
    analysis_config.json

Metrics include detection (fixed-threshold, class-aware image-level matching),
sharpness/detail, source-edge preservation, patch-seam artifacts, similarity
to each clean modality, mutual information, Q_AB/F, and degradation transfer.
All image metrics use saved fused PNGs aligned to 256x256.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from fusion_metrics import qabf
except ImportError:
    from evaluation.fusion_metrics import qabf

SIZE = (256, 256)
METHODS = ("pace", "crossfuse", "addition")
NAMES = {"pace": "PACE", "crossfuse": "CrossFuse", "addition": "Addition"}
COLORS = {"pace": "#D17842", "crossfuse": "#375A7F", "addition": "#669C82"}
EPS = 1e-9


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True,
                   help="Completed evaluation/eval_metrics.py output folder")
    p.add_argument("--dataset-root", type=Path,
                   help="Override degraded dataset path from run/config.json")
    p.add_argument("--output", type=Path,
                   help="Default: <run>/fusion_diagnostics")
    p.add_argument("--confidence", type=float, default=None)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--grid", type=int, default=8,
                   help="Adaptive fusion grid dimension; default 8 for PACE")
    p.add_argument("--seam-widths", default="0,1,2,4",
                   help="Pixel radii excluded around internal grid boundaries")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--examples", type=int, default=12)
    a = p.parse_args()
    if not 0 <= a.iou <= 1:
        p.error("IoU must be in [0,1]")
    if a.confidence is not None and not 0 <= a.confidence <= 1:
        p.error("Confidence must be in [0,1]")
    if a.grid < 1 or a.bootstrap < 0 or a.examples < 0:
        p.error("grid must be positive; bootstrap/examples must be nonnegative")
    try:
        widths = sorted(set(int(x.strip()) for x in a.seam_widths.split(",")))
    except ValueError:
        p.error("--seam-widths must be comma-separated nonnegative integers")
    if not widths or widths[0] < 0:
        p.error("--seam-widths must contain nonnegative integers")
    a.widths = widths
    return a


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation artifact: {path}")
    return json.loads(path.read_text())


def records_by_id(path: Path):
    records = read_json(path)
    if isinstance(records, dict):
        records = records.get("records", records.get("samples", []))
    out = {}
    for row in records:
        sid = str(row["sample_id"])
        if sid in out:
            raise ValueError(f"Duplicate sample ID in {path}: {sid}")
        out[sid] = row
    return out


def load_gray(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[::-1] != SIZE:
        image = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def load_mask(path: Path):
    if not path.is_file():
        return np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    if image.shape[::-1] != SIZE:
        image = cv2.resize(image, SIZE, interpolation=cv2.INTER_NEAREST)
    return image > 127


def gt_boxes(record):
    result = []
    for gt in record.get("ground_truth", []):
        cx, cy, w, h = [float(gt[k]) for k in ("cx", "cy", "width", "height")]
        result.append((int(gt["class_id"]),
                       ((cx-w/2)*256, (cy-h/2)*256,
                        (cx+w/2)*256, (cy+h/2)*256)))
    return result


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    inter = max(0., min(ax2, bx2)-max(ax1, bx1)) * max(
        0., min(ay2, by2)-max(ay1, by1))
    area_a = max(0., ax2-ax1) * max(0., ay2-ay1)
    area_b = max(0., bx2-bx1) * max(0., by2-by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.


def detection_metrics(record, confidence, iou_cutoff):
    targets = gt_boxes(record)
    preds = sorted(
        (x for x in record.get("predictions", [])
         if float(x.get("score", 0.0)) >= confidence),
        key=lambda x: float(x.get("score", 0.0)), reverse=True)
    used = set()
    tp = 0
    matched_scores = []
    for pred in preds:
        best_idx, best_overlap = None, iou_cutoff
        for idx, (cls, box) in enumerate(targets):
            if idx in used or cls != int(pred["class_id"]):
                continue
            overlap = iou_xyxy(pred["box_xyxy"], box)
            if overlap >= best_overlap:
                best_idx, best_overlap = idx, overlap
        if best_idx is not None:
            used.add(best_idx)
            tp += 1
            matched_scores.append(float(pred["score"]))
    fp, fn = len(preds)-tp, len(targets)-tp
    precision = tp/(tp+fp) if tp+fp else (1. if not targets else 0.)
    recall = tp/(tp+fn) if tp+fn else (1. if not preds else 0.)
    denom = 2*tp+fp+fn
    f1 = 2*tp/denom if denom else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "gt": len(targets),
        "precision": precision, "recall": recall, "f1": f1,
        "matched_confidence": float(np.mean(matched_scores)) if matched_scores else float("nan"),
    }


def gradients(image):
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, np.hypot(gx, gy)


def edge_map(image):
    u8 = np.uint8(np.clip(image, 0, 1) * 255)
    return cv2.Canny(u8, 75, 150) > 0


def edge_scores(pred, ref, eligible=None, tolerance=2):
    if eligible is None:
        eligible = np.ones_like(pred, dtype=bool)
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    ref_d = cv2.dilate(ref.astype(np.uint8), kernel).astype(bool)
    pred_d = cv2.dilate(pred.astype(np.uint8), kernel).astype(bool)
    p = pred & eligible
    r = ref & eligible
    precision = float((p & ref_d).sum()/p.sum()) if p.any() else float("nan")
    recall = float((r & pred_d).sum()/r.sum()) if r.any() else float("nan")
    f1 = (2*precision*recall/(precision+recall)
          if np.isfinite(precision) and np.isfinite(recall)
          and precision+recall > 0 else float("nan"))
    return precision, recall, f1


def seam_mask(shape, grid, radius):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if radius <= 0:
        return mask
    for k in range(1, grid):
        y = int(round(k*h/grid))
        x = int(round(k*w/grid))
        mask[max(0, y-radius):min(h, y+radius+1), :] = True
        mask[:, max(0, x-radius):min(w, x+radius+1)] = True
    return mask


def entropy_bits(image):
    hist = np.histogram(np.clip(image, 0, 1), bins=256, range=(0, 1))[0].astype(float)
    p = hist / max(hist.sum(), 1.)
    p = p[p > 0]
    return float(-(p*np.log2(p)).sum())


def ssim_global_local(a, b):
    """Mean local SSIM map, using an 11x11 Gaussian window."""
    a = np.clip(a.astype(np.float32), 0, 1)
    b = np.clip(b.astype(np.float32), 0, 1)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    aa = cv2.GaussianBlur(a*a, (11, 11), 1.5) - mu_a*mu_a
    bb = cv2.GaussianBlur(b*b, (11, 11), 1.5) - mu_b*mu_b
    ab = cv2.GaussianBlur(a*b, (11, 11), 1.5) - mu_a*mu_b
    c1, c2 = 0.01**2, 0.03**2
    score = ((2*mu_a*mu_b+c1)*(2*ab+c2)) / (
        (mu_a*mu_a+mu_b*mu_b+c1)*(aa+bb+c2)+EPS)
    return float(np.mean(score))


def mutual_information(a, b, bins=64):
    """Histogram mutual information in bits; interpret comparatively only."""
    x = np.clip(a, 0, 1).ravel()
    y = np.clip(b, 0, 1).ravel()
    hist, _, _ = np.histogram2d(x, y, bins=bins, range=((0, 1), (0, 1)))
    pxy = hist / max(hist.sum(), 1.)
    px, py = pxy.sum(axis=1), pxy.sum(axis=0)
    denom = px[:, None]*py[None, :]
    valid = pxy > 0
    return float(np.sum(pxy[valid]*np.log2(pxy[valid]/denom[valid])))


def qabf_safe(rgb, ir, fused):
    try:
        return float(qabf(rgb, ir, fused))
    except Exception:
        return float("nan")


def image_metrics(fused, rgb_clean, ir_clean, mask, grid, widths):
    gx, gy, grad = gradients(fused)
    lap = cv2.Laplacian(fused, cv2.CV_32F, ksize=3)
    edges = edge_map(fused)
    rgb_edges, ir_edges = edge_map(rgb_clean), edge_map(ir_clean)
    union = rgb_edges | ir_edges
    precision, recall, f1 = edge_scores(edges, union)
    p_rgb, r_rgb, f_rgb = edge_scores(edges, rgb_edges)
    p_ir, r_ir, f_ir = edge_scores(edges, ir_edges)
    seam = seam_mask(fused.shape, grid, max(widths) if widths else 0)
    nonseam = ~seam
    seam_grad = float(grad[seam].mean()) if seam.any() else float("nan")
    off_grad = float(grad[nonseam].mean()) if nonseam.any() else float("nan")
    seam_edge = float(edges[seam].mean()) if seam.any() else float("nan")
    off_edge = float(edges[nonseam].mean()) if nonseam.any() else float("nan")
    unsupported = edges & ~cv2.dilate(union.astype(np.uint8),
                                      np.ones((5, 5), np.uint8)).astype(bool)
    seam_unsupported = (unsupported & seam).sum() / max((edges & seam).sum(), 1)
    values = {
        "brightness": float(fused.mean()),
        "contrast_std": float(fused.std()),
        "entropy_bits": entropy_bits(fused),
        "tenengrad_mean": float((gx*gx+gy*gy).mean()),
        "tenengrad_rms": float(np.sqrt((gx*gx+gy*gy).mean())),
        "gradient_mean": float(grad.mean()),
        "laplacian_variance": float(lap.var()),
        "edge_density": float(edges.mean()),
        "edge_precision_union": precision,
        "edge_recall_union": recall,
        "edge_f1_union": f1,
        "edge_precision_rgb": p_rgb,
        "edge_recall_rgb": r_rgb,
        "edge_f1_rgb": f_rgb,
        "edge_precision_ir": p_ir,
        "edge_recall_ir": r_ir,
        "edge_f1_ir": f_ir,
        "ssim_clean_rgb": ssim_global_local(fused, rgb_clean),
        "ssim_clean_ir": ssim_global_local(fused, ir_clean),
        "mi_clean_rgb_bits": mutual_information(fused, rgb_clean),
        "mi_clean_ir_bits": mutual_information(fused, ir_clean),
        "qabf_clean_sources": qabf_safe(rgb_clean, ir_clean, fused),
        "seam_gradient_mean": seam_grad,
        "nonseam_gradient_mean": off_grad,
        "seam_gradient_ratio": seam_grad/(off_grad+EPS),
        "seam_gradient_excess": seam_grad-off_grad,
        "seam_edge_density": seam_edge,
        "nonseam_edge_density": off_edge,
        "seam_edge_density_ratio": seam_edge/(off_edge+EPS),
        "seam_unsupported_edge_fraction": float(seam_unsupported),
        "unsupported_edge_fraction": float(unsupported.sum()/max(edges.sum(), 1)),
    }
    for width in widths:
        band = seam_mask(fused.shape, grid, width)
        eligible = ~band
        ep, er, ef = edge_scores(edges, union, eligible=eligible)
        values[f"edge_f1_outside_seams_r{width}"] = ef
        values[f"edge_precision_outside_seams_r{width}"] = ep
        values[f"edge_recall_outside_seams_r{width}"] = er
        values[f"sharpness_outside_seams_r{width}"] = (
            float(grad[eligible].mean()) if eligible.any() else float("nan"))
    return values


def degradation_transfer(fused_d, fused_c, source_d, source_c, mask):
    if not mask.any():
        return float("nan")
    denominator = np.linalg.norm((source_d-source_c)[mask].astype(np.float64))
    numerator = np.linalg.norm((fused_d-fused_c)[mask].astype(np.float64))
    return float(numerator/denominator) if denominator > EPS else float("nan")


def bootstrap_mean(values, count, seed):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    if count <= 0 or len(x) < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(count)
    for i in range(count):
        draws[i] = rng.choice(x, size=len(x), replace=True).mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return mean, float(lo), float(hi)


def group_definitions(data):
    groups = [("all_images", data)]
    for col, prefix in (("degraded_modality", "modality"),
                        ("degradation_type", "degradation")):
        for value, group in data.groupby(col, dropna=False):
            groups.append((f"{prefix}={value}", group))
    for (modality, degradation), group in data.groupby(
            ["degraded_modality", "degradation_type"], dropna=False):
        groups.append((f"modality={modality}; degradation={degradation}", group))
    return groups


def summarize(data, bootstrap, seed):
    metric_cols = [
        "f1", "precision", "recall", "tp", "fp", "fn", "matched_confidence",
        "tenengrad_mean", "tenengrad_rms", "laplacian_variance", "gradient_mean",
        "edge_density", "edge_f1_union", "edge_f1_rgb", "edge_f1_ir",
        "edge_f1_outside_seams_r0", "edge_f1_outside_seams_r1",
        "edge_f1_outside_seams_r2", "edge_f1_outside_seams_r4",
        "seam_gradient_ratio", "seam_gradient_excess",
        "seam_unsupported_edge_fraction", "unsupported_edge_fraction",
        "ssim_clean_rgb", "ssim_clean_ir", "mi_clean_rgb_bits",
        "mi_clean_ir_bits", "qabf_clean_sources", "degradation_transfer_ratio",
        "edge_stability_f1",
    ]
    rows = []
    for condition, group in group_definitions(data):
        for method in METHODS:
            part = group[group.method == method]
            row = {"condition": condition, "method": method,
                   "n_images": len(part), "n_f1_scored": int(part.f1.notna().sum())}
            for j, metric in enumerate(metric_cols):
                if metric not in part:
                    continue
                mean, lo, hi = bootstrap_mean(part[metric], bootstrap,
                                              seed + j + METHODS.index(method)*1000)
                row[metric+"_mean"] = mean
                row[metric+"_ci95_low"] = lo
                row[metric+"_ci95_high"] = hi
            row["total_tp"] = int(part.tp.sum())
            row["total_fp"] = int(part.fp.sum())
            row["total_fn"] = int(part.fn.sum())
            row["fp_per_image"] = float(part.fp.mean()) if len(part) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(data, baseline, bootstrap, seed):
    metrics = [
        "f1", "precision", "recall", "tp", "fp", "fn",
        "tenengrad_mean", "laplacian_variance", "gradient_mean",
        "edge_f1_union", "edge_f1_rgb", "edge_f1_ir",
        "edge_f1_outside_seams_r0", "edge_f1_outside_seams_r1",
        "edge_f1_outside_seams_r2", "edge_f1_outside_seams_r4",
        "seam_gradient_ratio", "seam_gradient_excess",
        "seam_unsupported_edge_fraction", "unsupported_edge_fraction",
        "ssim_clean_rgb", "ssim_clean_ir", "mi_clean_rgb_bits",
        "mi_clean_ir_bits", "qabf_clean_sources", "degradation_transfer_ratio",
        "edge_stability_f1",
    ]
    wide = data.pivot(index="sample_id", columns="method")
    rows = []
    for method in METHODS:
        if method == baseline:
            continue
        for metric in metrics:
            if (metric, method) not in wide.columns or (metric, baseline) not in wide.columns:
                continue
            delta = wide[(metric, method)] - wide[(metric, baseline)]
            mean, lo, hi = bootstrap_mean(
                delta, bootstrap, seed + 8100 + metrics.index(metric)
            )
            rows.append({
                "method": method, "baseline": baseline, "metric": metric,
                "n_paired": int(delta.notna().sum()), "delta_mean": mean,
                "delta_ci95_low": lo, "delta_ci95_high": hi,
                "direction": "higher_is_better" if metric not in {
                    "fp", "fn", "seam_gradient_ratio", "seam_gradient_excess",
                    "seam_unsupported_edge_fraction", "unsupported_edge_fraction",
                    "degradation_transfer_ratio"} else "lower_is_better",
            })
    return pd.DataFrame(rows)


def plot_overall(summary, out):
    overall = summary[summary.condition == "all_images"].set_index("method")
    plots = [
        ("f1_mean", "Fixed-threshold detection F1", True),
        ("recall_mean", "Detection recall", True),
        ("fp_per_image", "False positives per image", False),
        ("tenengrad_mean_mean", "Tenengrad (sharpness)", True),
        ("qabf_clean_sources_mean", "Q_AB/F (clean sources)", True),
        ("edge_f1_union_mean", "Edge F1 vs clean-source union", True),
        ("edge_f1_outside_seams_r2_mean", "Edge F1 outside 2px seam bands", True),
        ("seam_gradient_ratio_mean", "Seam / non-seam gradient ratio", False),
        ("ssim_clean_rgb_mean", "SSIM vs clean RGB", True),
        ("ssim_clean_ir_mean", "SSIM vs clean IR", True),
        ("mi_clean_rgb_bits_mean", "MI vs clean RGB (bits)", True),
        ("mi_clean_ir_bits_mean", "MI vs clean IR (bits)", True),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(15, 15), layout="constrained")
    for ax, (key, title, higher) in zip(axes.flat, plots):
        vals = [overall.loc[m, key] if m in overall.index and key in overall else np.nan
                for m in METHODS]
        ax.bar(range(3), vals, color=[COLORS[m] for m in METHODS], width=.62)
        ax.set_xticks(range(3), [NAMES[m] for m in METHODS])
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        if np.isfinite(vals).any():
            for i, v in enumerate(vals):
                if np.isfinite(v):
                    ax.annotate(f"{v:.3f}", (i, v), xytext=(0, 4),
                                textcoords="offset points", ha="center", fontsize=8)
        ax.set_ylabel("higher is better" if higher else "lower is better")
    for ax in axes.flat[len(plots):]:
        ax.axis("off")
    fig.suptitle("Three-way fusion diagnostics — overall test set", fontsize=15)
    fig.savefig(out/"overall_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_conditions(data, out):
    metrics = [
        ("f1", "Detection F1"),
        ("edge_f1_union", "Edge F1 vs source union"),
        ("qabf_clean_sources", "Q_AB/F"),
        ("tenengrad_mean", "Tenengrad"),
        ("seam_gradient_ratio", "Seam gradient ratio"),
        ("edge_f1_outside_seams_r2", "Edge F1 outside 2px seams"),
    ]
    conditions = []
    for modality, degradation in data[
            ["degraded_modality", "degradation_type"]].drop_duplicates().itertuples(index=False):
        conditions.append((modality, degradation))
    conditions.sort()
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(max(10, 2.0*len(conditions)), 5),
                               layout="constrained")
        x = np.arange(len(conditions))
        width = .24
        for i, method in enumerate(METHODS):
            vals = []
            for modality, degradation in conditions:
                part = data[(data.method == method) &
                            (data.degraded_modality == modality) &
                            (data.degradation_type == degradation)]
                vals.append(part[metric].mean() if len(part) else np.nan)
            ax.bar(x+(i-1)*width, vals, width=width,
                   label=NAMES[method], color=COLORS[method])
        labels = [f"{m.upper()}\\n{d}" for m, d in conditions]
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by degraded modality and corruption", loc="left")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=3)
        fig.savefig(out/f"by_condition_{metric}.png", dpi=190, bbox_inches="tight")
        plt.close(fig)


def plot_distributions(data, out):
    metrics = [
        ("f1", "Detection F1"),
        ("tenengrad_mean", "Tenengrad"),
        ("laplacian_variance", "Laplacian variance"),
        ("qabf_clean_sources", "Q_AB/F"),
        ("edge_f1_union", "Edge F1"),
        ("seam_gradient_ratio", "Seam gradient ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, (metric, title) in zip(axes.flat, metrics):
        groups = [data.loc[data.method == m, metric].dropna().values for m in METHODS]
        ax.boxplot(groups, tick_labels=[NAMES[m] for m in METHODS],
                   showfliers=False, patch_artist=False)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Per-image metric distributions (outliers hidden for readability)")
    fig.savefig(out/"metric_distributions.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_seam_sensitivity(summary, out):
    metrics = [f"edge_f1_outside_seams_r{w}_mean" for w in (0, 1, 2, 4)]
    overall = summary[summary.condition == "all_images"].set_index("method")
    fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
    x = np.arange(4)
    for method in METHODS:
        vals = [overall.loc[method, k] if method in overall.index and k in overall else np.nan
                for k in metrics]
        ax.plot(x, vals, marker="o", linewidth=2, label=NAMES[method],
                color=COLORS[method])
    ax.set_xticks(x, ["No mask", "1px", "2px", "4px"])
    ax.set_ylabel("Edge F1 outside excluded seam bands")
    ax.set_title("Sensitivity to excluding adaptive-grid boundaries", loc="left")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(out/"seam_mask_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(data, out):
    metrics = ["f1", "tenengrad_mean", "qabf_clean_sources", "edge_f1_union",
               "edge_f1_outside_seams_r2", "seam_gradient_ratio",
               "ssim_clean_rgb", "ssim_clean_ir"]
    subset = data[metrics].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3:
        return
    corr = subset.corr()
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(metrics)), metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(metrics)), metrics)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center",
                    va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Metric correlation across method-image observations")
    fig.savefig(out/"metric_correlations.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_detection_quality(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for method in METHODS:
        part = data[data.method == method]
        axes[0].scatter(part.tenengrad_mean, part.f1, s=9, alpha=.25,
                        label=NAMES[method], color=COLORS[method])
        axes[1].scatter(part.qabf_clean_sources, part.f1, s=9, alpha=.25,
                        label=NAMES[method], color=COLORS[method])
    axes[0].set_xlabel("Tenengrad")
    axes[0].set_ylabel("Per-image detection F1")
    axes[0].set_title("Sharpness vs detection")
    axes[1].set_xlabel("Q_AB/F")
    axes[1].set_ylabel("Per-image detection F1")
    axes[1].set_title("Source-edge preservation vs detection")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.savefig(out/"detection_vs_quality.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_example_panel(
    sid, run, predictions, output_dir, confidence, iou
):
    methods = ("pace", "crossfuse", "addition")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, method in zip(axes, methods):
        record = predictions[method][sid]

        # Use the same detection calculation as the main evaluation.
        det = detection_metrics(record, confidence, iou)
        score = det["f1"]

        image_path = (
            run / method / "degraded" / "fused" / f"{sid}.png"
        )

        if image_path.is_file():
            image = load_gray(image_path)
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            ax.text(
                0.5, 0.5, "Image unavailable",
                ha="center", va="center"
            )

        score_text = f"{score:.3f}" if np.isfinite(score) else "N/A"
        ax.set_title(
            f"{NAMES[method]}\n"
            f"F1={score_text} | TP={det['tp']} "
            f"FP={det['fp']} FN={det['fn']}"
        )
        ax.axis("off")

    fig.suptitle(f"Fusion comparison: {sid}")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{sid}_comparison.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.close(fig)


def main():
    a = parse_args()
    run = a.run.resolve()
    config = read_json(run/"config.json")
    dataset = (a.dataset_root or Path(config["dataset"])).resolve()
    out = (a.output or run/"fusion_diagnostics").resolve()
    out.mkdir(parents=True, exist_ok=True)
    figure_dir = out/"figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    confidence = a.confidence if a.confidence is not None else float(
        config.get("confidence", .5))

    predictions = {
        method: records_by_id(run/method/"degraded"/"predictions.json")
        for method in METHODS
    }
    id_sets = {method: set(predictions[method]) for method in METHODS}
    common_ids = set.intersection(*id_sets.values())
    if any(ids != common_ids for ids in id_sets.values()):
        raise ValueError("PACE, CrossFuse, and Addition prediction IDs differ; "
                         "refusing an unpaired comparison")
    ids = sorted(common_ids)
    if not ids:
        raise ValueError("No shared prediction records found")

    metadata = pd.read_csv(dataset/"metadata.csv", dtype={"pair_id": str}).fillna("")
    metadata["sample_id"] = metadata["pair_id"].map(lambda x: Path(str(x)).stem)
    if "split" in metadata:
        test_rows = metadata[metadata["split"].astype(str).str.lower() == "test"]
        if len(test_rows):
            metadata = test_rows
    metadata = metadata.drop_duplicates("sample_id").set_index("sample_id")
    clean_root = run/"_dataset_clean"/"test"
    have_clean = (clean_root/"rgb").is_dir() and (clean_root/"ir").is_dir()

    rows = []
    for index, sid in enumerate(ids):
        if sid not in metadata.index:
            raise KeyError(f"No test metadata row for sample {sid}")
        meta = metadata.loc[sid]
        modality = str(meta.get("modality", "")).strip().lower()
        degradation = str(meta.get("degradation", meta.get("degradation_type", "")))
        rgb_d = load_gray(dataset/"test"/"rgb"/f"{sid}.jpg")
        ir_d = load_gray(dataset/"test"/"ir"/f"{sid}.jpg")
        mask = load_mask(dataset/"test"/"degradation_masks"/f"{sid}.png")
        rgb_c_path, ir_c_path = clean_root/"rgb"/f"{sid}.jpg", clean_root/"ir"/f"{sid}.jpg"
        if have_clean and rgb_c_path.is_file() and ir_c_path.is_file():
            rgb_c, ir_c = load_gray(rgb_c_path), load_gray(ir_c_path)
        else:
            # The metrics that require clean references remain NaN rather than
            # silently treating degraded inputs as clean references.
            rgb_c, ir_c = None, None

        gt_ref = gt_boxes(predictions["pace"][sid])
        for method in METHODS:
            record = predictions[method][sid]
            if gt_boxes(record) != gt_ref:
                raise ValueError(f"Ground truth differs across methods for {sid}")
            det = detection_metrics(record, confidence, a.iou)
            fused_path = run/method/"degraded"/"fused"/f"{sid}.png"
            fused = load_gray(fused_path)
            row = {
                "sample_id": sid, "method": method,
                "degraded_modality": modality, "degradation_type": degradation,
                "degradation_coverage": float(mask.mean()),
                **det,
            }
            if rgb_c is not None and ir_c is not None:
                row.update(image_metrics(fused, rgb_c, ir_c, mask, a.grid, a.widths))
                clean_fused_path = run/method/"clean"/"fused"/f"{sid}.png"
                if clean_fused_path.is_file():
                    fused_clean = load_gray(clean_fused_path)
                    source_d = rgb_d if modality == "rgb" else ir_d
                    source_c = rgb_c if modality == "rgb" else ir_c
                    row["degradation_transfer_ratio"] = degradation_transfer(
                        fused, fused_clean, source_d, source_c, mask)
                    pred_edges, clean_edges = edge_map(fused), edge_map(fused_clean)
                    row["edge_stability_f1"] = edge_scores(
                        pred_edges, clean_edges, tolerance=1)[2]
                else:
                    row["degradation_transfer_ratio"] = float("nan")
                    row["edge_stability_f1"] = float("nan")
            else:
                row.update({key: float("nan") for key in [
                    "brightness", "contrast_std", "entropy_bits", "tenengrad_mean",
                    "tenengrad_rms", "gradient_mean", "laplacian_variance",
                    "edge_density", "edge_precision_union", "edge_recall_union",
                    "edge_f1_union", "edge_precision_rgb", "edge_recall_rgb",
                    "edge_f1_rgb", "edge_precision_ir", "edge_recall_ir",
                    "edge_f1_ir", "ssim_clean_rgb", "ssim_clean_ir",
                    "mi_clean_rgb_bits", "mi_clean_ir_bits", "qabf_clean_sources",
                    "seam_gradient_mean", "nonseam_gradient_mean",
                    "seam_gradient_ratio", "seam_gradient_excess",
                    "seam_edge_density", "nonseam_edge_density",
                    "seam_edge_density_ratio", "seam_unsupported_edge_fraction",
                    "unsupported_edge_fraction", "degradation_transfer_ratio",
                    "edge_stability_f1",
                ]})
                for width in a.widths:
                    row[f"edge_f1_outside_seams_r{width}"] = float("nan")
                    row[f"edge_precision_outside_seams_r{width}"] = float("nan")
                    row[f"edge_recall_outside_seams_r{width}"] = float("nan")
                    row[f"sharpness_outside_seams_r{width}"] = float("nan")
            rows.append(row)
        if (index+1) % 100 == 0:
            print(f"Analyzed {index+1}/{len(ids)} images", flush=True)

    data = pd.DataFrame(rows).sort_values(["sample_id", "method"])
    data.to_csv(out/"per_image_method_metrics.csv", index=False)
    summary = summarize(data, a.bootstrap, a.seed)
    summary.to_csv(out/"summary_by_condition.csv", index=False)
    paired_deltas(
        data, "addition", a.bootstrap, a.seed
    ).to_csv(out/"paired_deltas_vs_addition.csv", index=False)

    paired_deltas(
        data, "crossfuse", a.bootstrap, a.seed
    ).to_csv(out/"paired_deltas_vs_crossfuse.csv", index=False)

    plot_overall(summary, figure_dir)
    plot_conditions(data, figure_dir)
    plot_distributions(data, figure_dir)
    plot_seam_sensitivity(summary, figure_dir)
    plot_correlations(data, figure_dir)
    plot_detection_quality(data, figure_dir)

    # Select examples with the largest disagreement in detection F1, plus
    # examples with the strongest seam artifact signal.
    wide_f1 = data.pivot(index="sample_id", columns="method", values="f1")
    disagreement = (wide_f1.max(axis=1)-wide_f1.min(axis=1)).sort_values(
        ascending=False)
    sample_rows = data[data.method == "pace"].set_index("sample_id")
    example_dir = out/"examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    chosen = list(disagreement.head(a.examples).index)
    for sid in chosen:
        make_example_panel(
            sid, run, predictions, example_dir, confidence, a.iou
        )

    report = {
        "run": str(run), "dataset": str(dataset), "output": str(out),
        "sample_count": len(ids), "method_count": len(METHODS),
        "methods": list(METHODS), "confidence": confidence, "iou": a.iou,
        "grid": a.grid, "seam_widths": a.widths,
        "clean_references_available": have_clean,
        "primary_detection_metric":
            "per-image class-aware greedy F1 at fixed confidence and IoU",
        "image_metrics": [
            "Tenengrad, Laplacian variance, gradient magnitude, entropy",
            "edge precision/recall/F1 against clean RGB, clean IR, and their union",
            "edge metrics outside 0/1/2/4-pixel bands around 8x8 patch boundaries",
            "seam-to-nonseam gradient and edge-density ratios",
            "unsupported seam edge fraction",
            "SSIM and histogram mutual information against each clean modality",
            "Q_AB/F against clean RGB and IR",
            "degradation transfer ratio and clean/degraded edge stability when clean runs exist",
        ],
        "interpretation_cautions": [
            "Sharpness and entropy can reward noise; do not interpret them alone.",
            "SSIM against either modality is not a universal fused-image quality score.",
            "Histogram mutual information is intensity-statistical and does not measure spatial alignment.",
            "Q_AB/F and edge metrics depend on Canny/Sobel thresholds and tolerance.",
            "Seam masks are diagnostics; report unmasked and masked results together.",
            "Detection F1 is per-image at a fixed confidence threshold, not mAP.",
            "Bootstrap intervals resample images; group by source scene if multiple samples share a scene.",
        ],
    }
    (out/"analysis_config.json").write_text(json.dumps(report, indent=2))
    print(f"Done: {len(ids)} paired images across {len(METHODS)} methods.")
    print("Outputs:", out)
    print("Figures:", figure_dir)


if __name__ == "__main__":
    main()