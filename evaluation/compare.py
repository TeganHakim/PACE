#!/usr/bin/env python3
"""
Comprehensive CrossFuse vs. PACE evaluation comparison.

Expected inputs are evaluation run directories containing some or all of:
    summary.json
    latency.json
    per_sample.csv
    per_class.csv
    predictions.json
    visualizations/*.png

Example from the repository root:
    python evaluation/compare.py \
        baseline/CrossFuse/evaluation/baseline_epoch_28_test_100 \
        pace/evaluation/pace_test_100

Or specify the output directory:
    python evaluation/compare.py \
        baseline/CrossFuse/evaluation/baseline_epoch_28_test_100 \
        pace/evaluation/pace_test_100 \
        --output evaluation/comparisons/test_100

The script is intentionally tolerant of missing metrics and slightly
different report schemas. Missing values are shown as unavailable, not 0.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------
# General helpers
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "crossfuse",
        type=Path,
        help="CrossFuse evaluation output directory",
    )
    parser.add_argument(
        "pace",
        type=Path,
        help="PACE evaluation output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Comparison output directory (default: evaluation/comparisons/...)",
    )
    parser.add_argument(
        "--max-gallery",
        type=int,
        default=60,
        help="Maximum matched visualization pairs to create (0 = all)",
    )
    parser.add_argument(
        "--gallery-columns",
        type=int,
        default=2,
        help="Number of comparison pairs per contact-sheet row",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=170,
    )
    return parser.parse_args()


def read_json(path):
    if not path.is_file():
        return {}
    try:
        with path.open() as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        print("Warning: could not read {}: {}".format(path, exc))
        return {}


def read_csv(path):
    if not path.is_file():
        return []
    try:
        with path.open(newline="") as file:
            return list(csv.DictReader(file))
    except OSError as exc:
        print("Warning: could not read {}: {}".format(path, exc))
        return []


def as_float(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def first_value(mapping, *paths):
    """Return the first non-None value at a sequence of dotted paths."""
    for path in paths:
        current = mapping
        found = True
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found and current is not None:
            return current
    return None


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("._") or "run"


def label_from_dir(path, fallback):
    name = path.name.strip()
    return name if name else fallback


def metric_value(report, *keys):
    value = first_value(report, *keys)
    return as_float(value)


def nested_metric_map(report, *keys):
    value = first_value(report, *keys)
    return value if isinstance(value, dict) else {}


def save_figure(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def available_pairs(first, second):
    labels = []
    values_a = []
    values_b = []
    for label, a, b in zip(first[0], first[1], second[1]):
        if a is not None or b is not None:
            labels.append(label)
            values_a.append(np.nan if a is None else a)
            values_b.append(np.nan if b is None else b)
    return labels, values_a, values_b


def annotate_bars(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        height = bar.get_height()
        if np.isfinite(height):
            ax.annotate(
                fmt.format(height),
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def grouped_bar(
    labels,
    values_a,
    values_b,
    name_a,
    name_b,
    title,
    ylabel,
    path,
    dpi,
    value_format="{:.3f}",
    rotate=0,
):
    valid = [
        i for i, (a, b) in enumerate(zip(values_a, values_b))
        if np.isfinite(a) or np.isfinite(b)
    ]
    if not valid:
        return False

    labels = [labels[i] for i in valid]
    a = np.asarray([values_a[i] for i in valid], dtype=float)
    b = np.asarray([values_b[i] for i in valid], dtype=float)

    x = np.arange(len(labels))
    width = 0.36

    fig_width = max(8, min(18, 1.0 * len(labels) + 4))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    bars_a = ax.bar(
        x - width / 2,
        np.nan_to_num(a, nan=0.0),
        width,
        label=name_a,
    )
    bars_b = ax.bar(
        x + width / 2,
        np.nan_to_num(b, nan=0.0),
        width,
        label=name_b,
    )

    # Hide missing values rather than presenting them as true zeros.
    for bars, values in ((bars_a, a), (bars_b, b)):
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                bar.set_alpha(0.0)
                bar.set_edgecolor("none")
            else:
                ax.annotate(
                    value_format.format(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, path, dpi)
    return True


def single_bar(labels, values, title, ylabel, path, dpi, rotate=0):
    valid = [
        (label, value)
        for label, value in zip(labels, values)
        if value is not None
    ]
    if not valid:
        return False

    labels, values = zip(*valid)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    bars = ax.bar(x, values)
    annotate_bars(ax, bars)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, path, dpi)
    return True


def boxplot_comparison(data_a, data_b, name_a, name_b, title, ylabel, path, dpi):
    arrays = []
    labels = []
    if len(data_a):
        arrays.append(data_a)
        labels.append(name_a)
    if len(data_b):
        arrays.append(data_b)
        labels.append(name_b)
    if not arrays:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(arrays, tick_labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, path, dpi)
    return True


def scatter_comparison(x, y, labels, title, xlabel, ylabel, path, dpi):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return False

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[valid], y[valid], alpha=0.65)
    low = min(x[valid].min(), y[valid].min())
    high = max(x[valid].max(), y[valid].max())
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, path, dpi)
    return True


# ---------------------------------------------------------
# Run loading and schema normalization
# ---------------------------------------------------------

def load_run(path, label):
    if not path.is_dir():
        raise FileNotFoundError(
            "Evaluation directory does not exist: {}".format(path)
        )

    return {
        "label": label,
        "path": path,
        "summary": read_json(path / "summary.json"),
        "latency": read_json(path / "latency.json"),
        "per_sample": read_csv(path / "per_sample.csv"),
        "per_class": read_csv(path / "per_class.csv"),
        "predictions": read_json(path / "predictions.json"),
        "visualization_dir": path / "visualizations",
    }


def detection_metrics(run):
    report = run["summary"]
    detection = report.get("detection", {})
    if not isinstance(detection, dict):
        detection = {}

    aliases = {
        "precision": (
            "precision",
            "detection.precision",
            "detection.P",
        ),
        "recall": (
            "recall",
            "detection.recall",
            "detection.R",
        ),
        "f1": (
            "f1",
            "F1",
            "detection.f1",
        ),
        "map50": (
            "map50",
            "map_50",
            "mAP50",
            "detection.map50",
            "detection.map_50",
        ),
        "map50_95": (
            "map50_95",
            "map",
            "mAP50-95",
            "detection.map50_95",
            "detection.map",
        ),
    }

    result = {}
    for name, keys in aliases.items():
        result[name] = metric_value(
            {"detection": detection, **report},
            *keys,
        )

    # If precision/recall are present but F1 is not, derive F1.
    if result["f1"] is None:
        p = result["precision"]
        r = result["recall"]
        if p is not None and r is not None and p + r > 0:
            result["f1"] = 2 * p * r / (p + r)

    return result


def per_class_metrics(run):
    report = run["summary"]
    detection = report.get("detection", {})
    if not isinstance(detection, dict):
        detection = {}

    ap_map = first_value(
        detection,
        "map_per_class",
        "ap_per_class",
        "per_class_ap",
    )
    recall_map = first_value(
        detection,
        "mar_100_per_class",
        "recall_per_class",
        "per_class_recall",
    )

    result = {}
    if isinstance(ap_map, dict):
        for name, value in ap_map.items():
            result.setdefault(str(name), {})["ap"] = as_float(value)
    if isinstance(recall_map, dict):
        for name, value in recall_map.items():
            result.setdefault(str(name), {})["recall"] = as_float(value)

    # Use per_class.csv when the summary did not include class metrics.
    for row in run["per_class"]:
        name = (
            row.get("class_name")
            or row.get("class")
            or row.get("name")
        )
        if not name:
            continue
        target = result.setdefault(name, {})
        for key, aliases in {
            "ap": ("ap50_95", "map50_95", "ap", "mAP"),
            "ap50": ("ap50", "map50", "mAP50"),
            "recall": ("recall_100", "mar_100", "recall"),
            "precision": ("precision",),
            "f1": ("f1", "F1"),
        }.items():
            if target.get(key) is None:
                for alias in aliases:
                    value = as_float(row.get(alias))
                    if value is not None:
                        target[key] = value
                        break

    return result


def subset_metrics(run):
    subsets = run["summary"].get("subsets", {})
    result = {}
    if not isinstance(subsets, dict):
        return result

    for group, values in subsets.items():
        if not isinstance(values, dict):
            continue
        result[str(group)] = {
            "count": as_float(values.get("count")),
            "map50": as_float(values.get("map50", values.get("map_50"))),
            "map50_95": as_float(values.get("map50_95", values.get("map"))),
            "precision": as_float(values.get("precision")),
            "recall": as_float(values.get("recall")),
            "f1": as_float(values.get("f1", values.get("F1"))),
        }
        if result[str(group)]["f1"] is None:
            p = result[str(group)]["precision"]
            r = result[str(group)]["recall"]
            if p is not None and r is not None and p + r > 0:
                result[str(group)]["f1"] = 2 * p * r / (p + r)
    return result


def degradation_metrics(run):
    value = run["summary"].get("degradation_breakdowns", {})
    return value if isinstance(value, dict) else {}


def latency_metrics(run):
    raw = run["latency"]
    if not raw:
        raw = run["summary"].get("latency", {})
    if not isinstance(raw, dict):
        return {}

    result = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            # Aggregate objects contain mean/median/p95/min/max.
            result[key] = {
                name: as_float(value.get(name))
                for name in ("mean", "median", "p95", "min", "max")
            }
        else:
            number = as_float(value)
            if number is not None:
                result[key] = {"mean": number}
    return result


def parameter_metrics(run):
    value = run["summary"].get("parameters", {})
    if not isinstance(value, dict):
        return {}
    return {
        str(key): as_float(number)
        for key, number in value.items()
        if as_float(number) is not None
    }


def fusion_metrics_summary(run):
    value = run["summary"].get("fusion_quality", {})
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, stats in value.items():
        if isinstance(stats, dict):
            result[key] = {
                name: as_float(stats.get(name))
                for name in ("mean", "median", "p95", "min", "max")
            }
        else:
            number = as_float(stats)
            if number is not None:
                result[key] = {"mean": number}
    return result


def sample_map(run):
    result = {}
    for row in run["per_sample"]:
        sample_id = row.get("sample_id") or row.get("pair_id") or row.get("id")
        if sample_id:
            result[str(sample_id)] = row
    return result


# ---------------------------------------------------------
# Detection, latency, memory, and fusion charts
# ---------------------------------------------------------

def create_metric_charts(a, b, out, dpi):
    charts = []

    # Overall detection.
    metrics_a = detection_metrics(a)
    metrics_b = detection_metrics(b)

    for metric, title, ylabel in (
        ("precision", "Overall precision", "Precision"),
        ("recall", "Overall recall", "Recall"),
        ("f1", "Overall F1 score", "F1"),
        ("map50", "Detection mAP@0.50", "mAP@0.50"),
        ("map50_95", "Detection mAP@0.50:0.95", "mAP@0.50:0.95"),
    ):
        if grouped_bar(
            [metric],
            [np.nan if metrics_a[metric] is None else metrics_a[metric]],
            [np.nan if metrics_b[metric] is None else metrics_b[metric]],
            a["label"],
            b["label"],
            title,
            ylabel,
            out / "detection_{}.png".format(metric),
            dpi,
        ):
            charts.append("detection_{}.png".format(metric))

    det_labels = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]
    det_keys = ["precision", "recall", "f1", "map50", "map50_95"]
    va = [np.nan if metrics_a[k] is None else metrics_a[k] for k in det_keys]
    vb = [np.nan if metrics_b[k] is None else metrics_b[k] for k in det_keys]
    if grouped_bar(
        det_labels, va, vb, a["label"], b["label"],
        "Overall detection metrics", "Score",
        out / "detection_overview.png", dpi,
    ):
        charts.append("detection_overview.png")

    # Per-class metrics.
    classes_a = per_class_metrics(a)
    classes_b = per_class_metrics(b)
    class_names = sorted(set(classes_a) | set(classes_b))

    for metric in ("ap", "ap50", "recall", "precision", "f1"):
        labels = class_names
        va = [
            np.nan if classes_a.get(name, {}).get(metric) is None
            else classes_a[name][metric]
            for name in labels
        ]
        vb = [
            np.nan if classes_b.get(name, {}).get(metric) is None
            else classes_b[name][metric]
            for name in labels
        ]
        if grouped_bar(
            labels, va, vb, a["label"], b["label"],
            "Per-class {}".format(metric.upper()),
            metric.upper(),
            out / "per_class_{}.png".format(metric),
            dpi,
            rotate=25,
        ):
            charts.append("per_class_{}.png".format(metric))

    # Clean/degraded subsets.
    subsets_a = subset_metrics(a)
    subsets_b = subset_metrics(b)
    subset_names = sorted(set(subsets_a) | set(subsets_b))

    for metric in ("map50", "map50_95", "precision", "recall", "f1"):
        va = [
            np.nan if subsets_a.get(name, {}).get(metric) is None
            else subsets_a[name][metric]
            for name in subset_names
        ]
        vb = [
            np.nan if subsets_b.get(name, {}).get(metric) is None
            else subsets_b[name][metric]
            for name in subset_names
        ]
        if grouped_bar(
            subset_names, va, vb, a["label"], b["label"],
            "Clean/degraded subset {}".format(metric.upper()),
            metric.upper(),
            out / "subset_{}.png".format(metric),
            dpi,
        ):
            charts.append("subset_{}.png".format(metric))

    # Degradation breakdowns.
    deg_a = degradation_metrics(a)
    deg_b = degradation_metrics(b)
    for field in sorted(set(deg_a) | set(deg_b)):
        group_a = deg_a.get(field, {})
        group_b = deg_b.get(field, {})
        if not isinstance(group_a, dict):
            group_a = {}
        if not isinstance(group_b, dict):
            group_b = {}
        names = sorted(set(group_a) | set(group_b))

        for metric in ("precision", "recall", "f1", "map50", "map50_95"):
            vals_a = []
            vals_b = []
            for name in names:
                x = group_a.get(name, {})
                y = group_b.get(name, {})
                if not isinstance(x, dict):
                    x = {}
                if not isinstance(y, dict):
                    y = {}
                v1 = x.get(metric, x.get("map_50" if metric == "map50" else metric))
                v2 = y.get(metric, y.get("map_50" if metric == "map50" else metric))
                vals_a.append(np.nan if as_float(v1) is None else as_float(v1))
                vals_b.append(np.nan if as_float(v2) is None else as_float(v2))

            filename = "degradation_{}_{}.png".format(
                safe_name(field), metric
            )
            if grouped_bar(
                names, vals_a, vals_b, a["label"], b["label"],
                "{} breakdown: {}".format(field.replace("_", " ").title(), metric.upper()),
                metric.upper(),
                out / filename,
                dpi,
                rotate=30,
            ):
                charts.append(filename)

    return charts


def create_latency_charts(a, b, out, dpi):
    charts = []
    la = latency_metrics(a)
    lb = latency_metrics(b)
    keys = sorted(set(la) | set(lb))

    # Compare means and medians for each recorded stage.
    for stat in ("mean", "median", "p95"):
        labels = []
        va = []
        vb = []
        for key in keys:
            x = la.get(key, {}).get(stat)
            y = lb.get(key, {}).get(stat)
            if x is None and y is None:
                continue
            labels.append(key.replace("_", " "))
            va.append(np.nan if x is None else x)
            vb.append(np.nan if y is None else y)

        filename = "latency_{}.png".format(stat)
        if grouped_bar(
            labels, va, vb, a["label"], b["label"],
            "Latency by pipeline stage ({})".format(stat),
            "Milliseconds (if recorded in ms)",
            out / filename,
            dpi,
            rotate=25,
        ):
            charts.append(filename)

    # End-to-end latency distribution from per-sample records.
    def get_sample_latency(run):
        vals = []
        for row in run["per_sample"]:
            for key in (
                "end_to_end_ms",
                "total_ms",
                "latency_ms",
                "inference_time_ms",
            ):
                value = as_float(row.get(key))
                if value is not None:
                    vals.append(value)
                    break
        return vals

    vals_a = get_sample_latency(a)
    vals_b = get_sample_latency(b)
    if boxplot_comparison(
        vals_a, vals_b, a["label"], b["label"],
        "Per-sample end-to-end latency",
        "Milliseconds",
        out / "latency_distribution.png",
        dpi,
    ):
        charts.append("latency_distribution.png")

    # FPS from recorded latency.
    fps_labels = []
    fps_a = []
    fps_b = []
    for label, run, target in (
        (a["label"], a, fps_a),
        (b["label"], b, fps_b),
    ):
        latency = latency_metrics(run)
        end = (
            latency.get("end_to_end_ms", {}).get("mean")
            or latency.get("end_to_end", {}).get("mean")
        )
        fps = 1000.0 / end if end and end > 0 else None
        target.append(fps)

    if grouped_bar(
        ["FPS from mean end-to-end latency"],
        [np.nan if fps_a[0] is None else fps_a[0]],
        [np.nan if fps_b[0] is None else fps_b[0]],
        a["label"], b["label"],
        "Throughput comparison",
        "Frames per second",
        out / "throughput_fps.png",
        dpi,
    ):
        charts.append("throughput_fps.png")

    return charts


def create_resource_charts(a, b, out, dpi):
    charts = []
    summary_a = a["summary"]
    summary_b = b["summary"]

    # Memory fields can be reported either at summary root or in latency.
    memory_candidates = (
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "peak_memory_mb",
        "peak_allocated_mb",
        "gpu_memory_mb",
        "memory_mb",
        "peak_rss_mb",
    )

    def memory_values(run):
        result = {}
        for key in memory_candidates:
            value = first_value(
                run["latency"],
                key,
            )
            if value is None:
                value = first_value(
                    run["summary"],
                    "latency." + key,
                    "memory." + key,
                    "resources." + key,
                )
            number = as_float(value)
            if number is not None:
                result[key] = number
        return result

    mem_a = memory_values(a)
    mem_b = memory_values(b)
    memory_keys = sorted(set(mem_a) | set(mem_b))

    if memory_keys:
        if grouped_bar(
            memory_keys,
            [
                np.nan if mem_a.get(key) is None else mem_a[key]
                for key in memory_keys
            ],
            [
                np.nan if mem_b.get(key) is None else mem_b[key]
                for key in memory_keys
            ],
            a["label"], b["label"],
            "Recorded peak memory",
            "MB",
            out / "memory_comparison.png",
            dpi,
            rotate=25,
        ):
            charts.append("memory_comparison.png")

    # Parameter counts.
    params_a = parameter_metrics(a)
    params_b = parameter_metrics(b)
    param_keys = sorted(
        set(params_a) | set(params_b),
        key=lambda key: (
            key == "total_including_detector",
            key,
        ),
    )
    if param_keys:
        if grouped_bar(
            param_keys,
            [
                np.nan if params_a.get(key) is None else params_a[key] / 1e6
                for key in param_keys
            ],
            [
                np.nan if params_b.get(key) is None else params_b[key] / 1e6
                for key in param_keys
            ],
            a["label"], b["label"],
            "Model parameter counts",
            "Parameters (millions)",
            out / "parameter_counts.png",
            dpi,
            value_format="{:.2f}",
            rotate=30,
        ):
            charts.append("parameter_counts.png")

    # Parameter total excluding/including detector, when present.
    return charts


def create_fusion_charts(a, b, out, dpi):
    charts = []
    fa = fusion_metrics_summary(a)
    fb = fusion_metrics_summary(b)
    keys = sorted(set(fa) | set(fb))

    for stat in ("mean", "median"):
        labels = []
        va = []
        vb = []
        for key in keys:
            x = fa.get(key, {}).get(stat)
            y = fb.get(key, {}).get(stat)
            if x is None and y is None:
                continue
            labels.append(key.replace("_", " "))
            va.append(np.nan if x is None else x)
            vb.append(np.nan if y is None else y)

        filename = "fusion_quality_{}.png".format(stat)
        if grouped_bar(
            labels, va, vb, a["label"], b["label"],
            "Fusion quality metrics ({})".format(stat),
            "Metric value",
            out / filename,
            dpi,
            rotate=25,
        ):
            charts.append(filename)

    # Per-sample gate activity and fused image intensity diagnostics.
    rows_a = sample_map(a)
    rows_b = sample_map(b)
    common = sorted(set(rows_a) & set(rows_b))

    for field, title, ylabel, filename in (
        ("active_fraction", "PACE gate active fraction", "Active fraction", "gate_activity.png"),
        ("fused_mean", "Per-sample fused-image mean intensity", "Mean intensity", "fused_mean_intensity.png"),
        ("fused_max", "Per-sample fused-image maximum intensity", "Maximum intensity", "fused_max_intensity.png"),
    ):
        if field == "active_fraction":
            # Gate statistics are specific to PACE; plot PACE's distribution.
            pace_rows = b["per_sample"]
            vals = [as_float(row.get(field)) for row in pace_rows]
            vals = [value for value in vals if value is not None]
            if vals:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(vals, bins=20)
                ax.set_title(title)
                ax.set_xlabel(ylabel)
                ax.set_ylabel("Number of samples")
                ax.grid(axis="y", alpha=0.25)
                fig.tight_layout()
                save_figure(fig, out / filename, dpi)
                charts.append(filename)
            continue

        vals_a = [as_float(rows_a[s].get(field)) for s in common]
        vals_b = [as_float(rows_b[s].get(field)) for s in common]
        if boxplot_comparison(
            [x for x in vals_a if x is not None],
            [x for x in vals_b if x is not None],
            a["label"], b["label"],
            title,
            ylabel,
            out / filename,
            dpi,
        ):
            charts.append(filename)

    return charts


def create_sample_charts(a, b, out, dpi):
    charts = []
    rows_a = sample_map(a)
    rows_b = sample_map(b)
    common = sorted(set(rows_a) & set(rows_b))

    if not common:
        return charts

    # Prediction counts and confidence.
    for field, title, ylabel, filename in (
        ("num_predictions_at_confidence", "Predictions per sample", "Detection count", "sample_prediction_counts.png"),
        ("max_prediction_score", "Maximum detection confidence per sample", "Maximum confidence", "sample_max_confidence.png"),
        ("num_ground_truth", "Ground-truth objects per sample", "Object count", "sample_ground_truth_counts.png"),
    ):
        va = [as_float(rows_a[s].get(field)) for s in common]
        vb = [as_float(rows_b[s].get(field)) for s in common]
        if boxplot_comparison(
            [v for v in va if v is not None],
            [v for v in vb if v is not None],
            a["label"], b["label"],
            title,
            ylabel,
            out / filename,
            dpi,
        ):
            charts.append(filename)

    # Paired scatter for per-sample latency.
    latency_fields = (
        "end_to_end_ms",
        "total_ms",
        "latency_ms",
    )
    field = next(
        (
            key
            for key in latency_fields
            if any(rows_a[s].get(key) for s in common)
            and any(rows_b[s].get(key) for s in common)
        ),
        None,
    )
    if field:
        va = [as_float(rows_a[s].get(field)) for s in common]
        vb = [as_float(rows_b[s].get(field)) for s in common]
        if scatter_comparison(
            va, vb, common,
            "Per-sample latency: CrossFuse vs PACE",
            a["label"] + " latency (ms)",
            b["label"] + " latency (ms)",
            out / "sample_latency_scatter.png",
            dpi,
        ):
            charts.append("sample_latency_scatter.png")

    # Paired sample prediction count scatter.
    va = [
        as_float(rows_a[s].get("num_predictions_at_confidence"))
        for s in common
    ]
    vb = [
        as_float(rows_b[s].get("num_predictions_at_confidence"))
        for s in common
    ]
    if scatter_comparison(
        va, vb, common,
        "Per-sample prediction counts",
        a["label"] + " predictions",
        b["label"] + " predictions",
        out / "sample_prediction_scatter.png",
        dpi,
    ):
        charts.append("sample_prediction_scatter.png")

    # Paired per-sample fused means (if both evaluators recorded them).
    va = [as_float(rows_a[s].get("fused_mean")) for s in common]
    vb = [as_float(rows_b[s].get("fused_mean")) for s in common]
    if scatter_comparison(
        va, vb, common,
        "Per-sample fused-image mean intensity",
        a["label"] + " fused mean",
        b["label"] + " fused mean",
        out / "sample_fused_mean_scatter.png",
        dpi,
    ):
        charts.append("sample_fused_mean_scatter.png")

    return charts


# ---------------------------------------------------------
# Visualization gallery
# ---------------------------------------------------------

def image_files(folder):
    if not folder.is_dir():
        return {}
    result = {}
    for path in folder.glob("*.png"):
        # Standalone PACE fused outputs are not treated as the full panel.
        if path.stem.endswith("_fused"):
            continue
        result[path.stem] = path
    return result


def load_font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def fit_image(image, max_width, max_height):
    image = image.convert("RGB")
    scale = min(
        max_width / image.width,
        max_height / image.height,
    )
    size = (
        max(1, int(image.width * scale)),
        max(1, int(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def make_pair_image(path_a, path_b, label_a, label_b, sample_id, target):
    try:
        im_a = Image.open(path_a).convert("RGB")
        im_b = Image.open(path_b).convert("RGB")
    except (OSError, ValueError):
        return False

    target_height = 430
    im_a = fit_image(im_a, 1100, target_height)
    im_b = fit_image(im_b, 1100, target_height)

    label_height = 42
    gap = 12
    width = im_a.width + im_b.width + gap
    height = label_height + max(im_a.height, im_b.height)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    small = load_font(15)

    draw.text((5, 8), "{} — {}".format(label_a, sample_id), fill="black", font=font)
    draw.text((im_a.width + gap + 5, 8), "{} — {}".format(label_b, sample_id), fill="black", font=font)
    canvas.paste(im_a, (0, label_height))
    canvas.paste(im_b, (im_a.width + gap, label_height))
    canvas.save(target)
    return True


def create_visualization_gallery(a, b, out, max_gallery, columns):
    gallery_dir = out / "visualization_pairs"
    sheets_dir = out / "contact_sheets"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    images_a = image_files(a["visualization_dir"])
    images_b = image_files(b["visualization_dir"])
    common = sorted(set(images_a) & set(images_b))

    if max_gallery > 0:
        common = common[:max_gallery]

    pair_paths = []
    for sample_id in common:
        target = gallery_dir / "{}_comparison.png".format(
            safe_name(sample_id)
        )
        if make_pair_image(
            images_a[sample_id],
            images_b[sample_id],
            a["label"],
            b["label"],
            sample_id,
            target,
        ):
            pair_paths.append(target)

    # Create contact sheets from the side-by-side comparisons.
    per_sheet = max(1, columns) * 3
    sheet_paths = []

    for start in range(0, len(pair_paths), per_sheet):
        batch = pair_paths[start:start + per_sheet]
        thumbs = []
        for path in batch:
            try:
                image = Image.open(path).convert("RGB")
                thumbs.append(fit_image(image, 900, 300))
            except OSError:
                continue

        if not thumbs:
            continue

        thumb_width = max(image.width for image in thumbs)
        thumb_height = max(image.height for image in thumbs)
        cols = max(1, columns)
        rows = int(math.ceil(len(thumbs) / cols))
        margin = 18

        sheet = Image.new(
            "RGB",
            (
                cols * thumb_width + (cols + 1) * margin,
                rows * thumb_height + (rows + 1) * margin,
            ),
            "#e8e8e8",
        )

        for index, image in enumerate(thumbs):
            row = index // cols
            col = index % cols
            x = margin + col * (thumb_width + margin)
            y = margin + row * (thumb_height + margin)
            sheet.paste(image, (x, y))

        target = sheets_dir / "contact_sheet_{:03d}.jpg".format(
            len(sheet_paths) + 1
        )
        sheet.save(target, quality=90)
        sheet_paths.append(target)

    # Also compare standalone fused images when both runs provide them.
    fused_a = {
        path.stem[:-6]: path
        for path in a["visualization_dir"].glob("*_fused.png")
    } if a["visualization_dir"].is_dir() else {}
    fused_b = {
        path.stem[:-6]: path
        for path in b["visualization_dir"].glob("*_fused.png")
    } if b["visualization_dir"].is_dir() else {}

    fused_common = sorted(set(fused_a) & set(fused_b))
    fused_dir = out / "fused_image_pairs"
    if fused_common:
        fused_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in fused_common[:max_gallery or None]:
            try:
                im_a = Image.open(fused_a[sample_id]).convert("RGB")
                im_b = Image.open(fused_b[sample_id]).convert("RGB")
                h = 420
                im_a = fit_image(im_a, 850, h)
                im_b = fit_image(im_b, 850, h)
                gap = 10
                label_h = 42
                canvas = Image.new(
                    "RGB",
                    (im_a.width + im_b.width + gap,
                     label_h + max(im_a.height, im_b.height)),
                    "white",
                )
                draw = ImageDraw.Draw(canvas)
                font = load_font(18)
                draw.text((5, 8), "{} — {}".format(a["label"], sample_id), fill="black", font=font)
                draw.text((im_a.width + gap + 5, 8), "{} — {}".format(b["label"], sample_id), fill="black", font=font)
                canvas.paste(im_a, (0, label_h))
                canvas.paste(im_b, (im_a.width + gap, label_h))
                canvas.save(fused_dir / "{}_fused_comparison.png".format(safe_name(sample_id)))
            except OSError:
                pass

    return {
        "matched_panel_count": len(pair_paths),
        "matched_fused_count": len(fused_common),
        "pair_directory": gallery_dir,
        "contact_sheets": sheet_paths,
        "fused_directory": fused_dir if fused_common else None,
    }


# ---------------------------------------------------------
# Report generation
# ---------------------------------------------------------

def format_value(value, digits=4):
    if value is None:
        return "N/A"
    return "{:.{}f}".format(value, digits)


def write_report(a, b, out, charts, gallery):
    metrics_a = detection_metrics(a)
    metrics_b = detection_metrics(b)
    latency_a = latency_metrics(a)
    latency_b = latency_metrics(b)
    params_a = parameter_metrics(a)
    params_b = parameter_metrics(b)

    lines = [
        "# CrossFuse vs. PACE Evaluation Report",
        "",
        "## Evaluation runs",
        "",
        "- **CrossFuse:** `{}`".format(a["path"]),
        "- **PACE:** `{}`".format(b["path"]),
        "- **Output:** `{}`".format(out),
        "",
        "## Overall detection metrics",
        "",
        "| Metric | CrossFuse | PACE | PACE − CrossFuse |",
        "|---|---:|---:|---:|",
    ]

    for key in ("precision", "recall", "f1", "map50", "map50_95"):
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        delta = vb - va if va is not None and vb is not None else None
        lines.append(
            "| {} | {} | {} | {} |".format(
                key,
                format_value(va),
                format_value(vb),
                format_value(delta),
            )
        )

    lines.extend([
        "",
        "## Latency",
        "",
        "| Stage | CrossFuse mean | PACE mean |",
        "|---|---:|---:|",
    ])

    for key in sorted(set(latency_a) | set(latency_b)):
        va = latency_a.get(key, {}).get("mean")
        vb = latency_b.get(key, {}).get("mean")
        lines.append(
            "| {} | {} | {} |".format(
                key,
                format_value(va, 3),
                format_value(vb, 3),
            )
        )

    lines.extend([
        "",
        "## Parameter counts",
        "",
        "| Component | CrossFuse | PACE |",
        "|---|---:|---:|",
    ])
    for key in sorted(set(params_a) | set(params_b)):
        lines.append(
            "| {} | {} | {} |".format(
                key,
                "{:,.0f}".format(params_a[key]) if key in params_a else "N/A",
                "{:,.0f}".format(params_b[key]) if key in params_b else "N/A",
            )
        )

    lines.extend([
        "",
        "## Generated charts",
        "",
    ])
    if charts:
        for chart in charts:
            lines.append("- `{}`".format(chart))
    else:
        lines.append("- No numeric chartable metrics were found.")

    lines.extend([
        "",
        "## Visualization gallery",
        "",
        "- Matched side-by-side panels: {}".format(gallery["matched_panel_count"]),
        "- Matched standalone fused images: {}".format(gallery["matched_fused_count"]),
        "- Contact sheets: {}".format(len(gallery["contact_sheets"])),
        "- Pair images: `{}`".format(gallery["pair_directory"]),
    ])

    lines.extend([
        "",
        "## Important interpretation notes",
        "",
        "- Missing metrics are reported as N/A; they are not treated as zero.",
        "- A chart is generated only when the corresponding metric exists in at least one run.",
        "- Per-sample comparisons use the intersection of sample IDs present in both `per_sample.csv` files.",
        "- Visualization pairs are matched by PNG filename stem. If the two evaluators use different filenames, those samples will not appear in paired galleries.",
        "- CUDA peak memory is compared only if the evaluation summaries recorded it. This script cannot reconstruct peak GPU memory after the evaluation has finished.",
        "- Parameter counts are model parameter counts, not runtime memory usage.",
        "- Latency numbers should only be compared directly when both runs used the same hardware, device, input size, warmup policy, and timing methodology.",
        "",
    ])

    (out / "comparison_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def create_dashboard(a, b, out, dpi):
    metrics_a = detection_metrics(a)
    metrics_b = detection_metrics(b)
    latency_a = latency_metrics(a)
    latency_b = latency_metrics(b)
    params_a = parameter_metrics(a)
    params_b = parameter_metrics(b)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Detection panel.
    ax = axes[0, 0]
    names = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]
    keys = ["precision", "recall", "f1", "map50", "map50_95"]
    x = np.arange(len(keys))
    width = 0.36
    va = [np.nan if metrics_a[k] is None else metrics_a[k] for k in keys]
    vb = [np.nan if metrics_b[k] is None else metrics_b[k] for k in keys]
    ax.bar(x - width / 2, np.nan_to_num(va), width, label=a["label"])
    ax.bar(x + width / 2, np.nan_to_num(vb), width, label=b["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(bottom=0)
    ax.set_title("Detection performance")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    # Latency panel.
    ax = axes[0, 1]
    latency_keys = [
        key for key in ("encoder_quality_gate_ms", "selective_fusion_ms",
                        "detection_nms_ms", "end_to_end_ms")
        if key in latency_a or key in latency_b
    ]
    if latency_keys:
        x = np.arange(len(latency_keys))
        va = [
            np.nan if latency_a.get(k, {}).get("mean") is None
            else latency_a[k]["mean"]
            for k in latency_keys
        ]
        vb = [
            np.nan if latency_b.get(k, {}).get("mean") is None
            else latency_b[k]["mean"]
            for k in latency_keys
        ]
        ax.bar(x - width / 2, np.nan_to_num(va), width, label=a["label"])
        ax.bar(x + width / 2, np.nan_to_num(vb), width, label=b["label"])
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("_", "\n") for k in latency_keys], fontsize=8)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Latency unavailable", ha="center", va="center")
    ax.set_title("Mean latency")
    ax.set_ylabel("Milliseconds")
    ax.grid(axis="y", alpha=0.2)

    # Parameter panel.
    ax = axes[1, 0]
    key = "total_including_detector"
    if key in params_a or key in params_b:
        vals = [
            params_a.get(key, np.nan),
            params_b.get(key, np.nan),
        ]
        bars = ax.bar([a["label"], b["label"]], np.nan_to_num(vals) / 1e6)
        annotate_bars(ax, bars, "{:.2f}M")
        ax.set_ylabel("Parameters (millions)")
    else:
        ax.text(0.5, 0.5, "Parameter counts unavailable", ha="center", va="center")
    ax.set_title("Total parameter count")
    ax.grid(axis="y", alpha=0.2)

    # Sample coverage.
    ax = axes[1, 1]
    n_a = len(a["per_sample"])
    n_b = len(b["per_sample"])
    bars = ax.bar([a["label"], b["label"]], [n_a, n_b])
    annotate_bars(ax, bars, "{:.0f}")
    ax.set_title("Per-sample records")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("CrossFuse vs. PACE — Evaluation Dashboard", fontsize=16)
    fig.tight_layout()
    save_figure(fig, out / "00_dashboard.png", dpi)


def main():
    args = parse_args()

    crossfuse_path = args.crossfuse.expanduser().resolve()
    pace_path = args.pace.expanduser().resolve()

    label_a = label_from_dir(crossfuse_path, "CrossFuse")
    label_b = label_from_dir(pace_path, "PACE")

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (
            Path(__file__).resolve().parent
            / "comparisons"
            / "{}_vs_{}".format(safe_name(label_a), safe_name(label_b))
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    charts_dir = output / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CrossFuse run: {}".format(crossfuse_path))
    crossfuse = load_run(crossfuse_path, label_a)

    print("Loading PACE run: {}".format(pace_path))
    pace = load_run(pace_path, label_b)

    print("Generating charts...")
    charts = []
    charts.extend(
        create_metric_charts(
            crossfuse,
            pace,
            charts_dir,
            args.dpi,
        )
    )
    charts.extend(
        create_latency_charts(
            crossfuse,
            pace,
            charts_dir,
            args.dpi,
        )
    )
    charts.extend(
        create_resource_charts(
            crossfuse,
            pace,
            charts_dir,
            args.dpi,
        )
    )
    charts.extend(
        create_fusion_charts(
            crossfuse,
            pace,
            charts_dir,
            args.dpi,
        )
    )
    charts.extend(
        create_sample_charts(
            crossfuse,
            pace,
            charts_dir,
            args.dpi,
        )
    )

    create_dashboard(
        crossfuse,
        pace,
        charts_dir,
        args.dpi,
    )

    print("Matching visualizations...")
    gallery = create_visualization_gallery(
        crossfuse,
        pace,
        output,
        args.max_gallery,
        args.gallery_columns,
    )

    write_report(
        crossfuse,
        pace,
        output,
        charts,
        gallery,
    )

    # Save normalized machine-readable comparison values.
    comparison = {
        "crossfuse_directory": str(crossfuse_path),
        "pace_directory": str(pace_path),
        "crossfuse_detection": detection_metrics(crossfuse),
        "pace_detection": detection_metrics(pace),
        "crossfuse_per_class": per_class_metrics(crossfuse),
        "pace_per_class": per_class_metrics(pace),
        "crossfuse_subsets": subset_metrics(crossfuse),
        "pace_subsets": subset_metrics(pace),
        "crossfuse_latency": latency_metrics(crossfuse),
        "pace_latency": latency_metrics(pace),
        "crossfuse_parameters": parameter_metrics(crossfuse),
        "pace_parameters": parameter_metrics(pace),
        "crossfuse_fusion_quality": fusion_metrics_summary(crossfuse),
        "pace_fusion_quality": fusion_metrics_summary(pace),
        "matched_sample_ids": sorted(
            set(sample_map(crossfuse)) & set(sample_map(pace))
        ),
        "visualization_gallery": {
            "matched_panel_count": gallery["matched_panel_count"],
            "matched_fused_count": gallery["matched_fused_count"],
            "contact_sheets": [
                str(path) for path in gallery["contact_sheets"]
            ],
        },
        "charts": charts,
    }

    with (output / "comparison.json").open("w") as file:
        json.dump(comparison, file, indent=2, allow_nan=False)

    print("")
    print("Comparison complete.")
    print("Charts: {}".format(charts_dir))
    print("Side-by-side pairs: {}".format(gallery["pair_directory"]))
    print("Contact sheets: {}".format(output / "contact_sheets"))
    print("Report: {}".format(output / "comparison_report.md"))
    print("Machine-readable results: {}".format(output / "comparison.json"))
    print("Charts generated: {}".format(len(charts) + 1))
    print("Matched panel images: {}".format(gallery["matched_panel_count"]))
    print("Matched fused images: {}".format(gallery["matched_fused_count"]))


if __name__ == "__main__":
    main()
