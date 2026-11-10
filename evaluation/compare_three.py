#!/usr/bin/env python3
"""Compare saved CrossFuse, PACE, and Addition evaluations."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

NAMES = ("CrossFuse", "PACE", "Addition")
COLORS = {
    "CrossFuse": "#375A7F",
    "PACE": "#D17842",
    "Addition": "#438C78",
}
METRICS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("map50", "mAP@50"),
    ("map50_95", "mAP@50:95"),
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crossfuse", type=Path, required=True
    )
    parser.add_argument(
        "--pace", type=Path, required=True
    )
    parser.add_argument(
        "--addition", type=Path, required=True
    )
    parser.add_argument(
        "--output", type=Path, required=True
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--visualizations", type=int, default=12)
    return parser.parse_args()


def numeric(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def metric_value(data, key):
    aliases = {
        "map50": ("map50", "map_50"),
        "map50_95": ("map50_95", "map", "map_50_95"),
    }
    for name in aliases.get(key, (key,)):
        result = numeric(data.get(name))
        if result is not None:
            return result
    return None


def load_run(path):
    path = path.resolve()

    def read_json(name):
        with (path / name).open() as f:
            return json.load(f)

    with (path / "per_sample.csv").open(newline="") as f:
        raw_rows = list(csv.DictReader(f))

    rows = {
        str(row["sample_id"]): row
        for row in raw_rows
    }

    predictions = read_json("predictions.json")
    prediction_ids = {
        str(item["sample_id"]) for item in predictions
    }

    if len(rows) != len(raw_rows):
        raise ValueError(f"Duplicate sample IDs in {path}")
    if len(prediction_ids) != len(predictions):
        raise ValueError(f"Duplicate predictions in {path}")
    if set(rows) != prediction_ids:
        raise ValueError(f"Rows/predictions mismatch: {path}")

    return {
        "path": path,
        "summary": read_json("summary.json"),
        "latency": read_json("latency.json"),
        "rows": rows,
    }


def validate(runs):
    ids = None
    reference = None

    for name, run in runs.items():
        current = set(run["rows"])

        if ids is None:
            ids = current
        elif current != ids:
            raise ValueError(
                f"{name} has different sample IDs. "
                "Use matching evaluation runs."
            )

        config = run["summary"].get("configuration", {})

        for key in (
            "split", "image_size", "confidence",
            "nms_iou", "seed",
        ):
            if key in config:
                if reference is None:
                    continue
                previous = reference.get(key)
                if previous is not None and str(previous) != str(
                    config[key]
                ):
                    raise ValueError(
                        f"Incompatible {key}: {name}"
                    )

        if reference is None:
            reference = config

    # Validate device separately, including presence.
    devices = {
        str(run["summary"].get(
            "configuration", {}
        ).get("device", "unknown"))
        for run in runs.values()
    }
    if len(devices) != 1:
        raise ValueError(
            f"Different recorded devices: {devices}"
        )

    return sorted(ids)


def style():
    sns.set_theme(
        style="white",
        context="paper",
        font_scale=1.12,
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Avenir Next",
                "Helvetica Neue",
                "DejaVu Sans",
            ],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def finish(fig, path, dpi):
    fig.savefig(
        path, dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def clean(ax, axis="x"):
    ax.grid(False)
    ax.grid(axis=axis, color="#E9EDF1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def grouped_bars(ax, entries):
    """entries: [(label, {model: score}), ...]."""
    ys = np.arange(len(entries))
    height = 0.22
    offsets = (-height, 0, height)

    for name, offset in zip(NAMES, offsets):
        values = [
            item[1][name] for item in entries
        ]
        bars = ax.barh(
            ys + offset, values,
            height=height * 0.9,
            color=COLORS[name],
            label=name,
        )
        for bar, value in zip(bars, values):
            ax.text(
                value + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center", fontsize=8,
            )

    ax.set_yticks(
        ys, [item[0] for item in entries]
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    clean(ax)


def detection_figure(runs, output, dpi):
    entries = []

    for key, label in METRICS:
        values = {
            name: metric_value(
                run["summary"].get("detection", {}),
                key,
            )
            for name, run in runs.items()
        }
        if all(value is not None for value in values.values()):
            entries.append((label, values))

    if not entries:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    grouped_bars(ax, entries)

    ax.set_title("Detection performance", loc="left", pad=18)
    ax.set_xlabel("Detection score (higher is better)")
    ax.legend(
        loc="lower right",
        ncol=3, frameon=False,
    )

    fig.tight_layout()
    finish(fig, output / "01_detection.png", dpi)
    return "01_detection.png"


def degradation_figure(runs, output, dpi):
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5),
    )

    for ax, subset in zip(axes, ("clean", "degraded")):
        entries = []

        for key, label in (
            ("f1", "F1"),
            ("map50", "mAP@50"),
        ):
            values = {
                name: metric_value(
                    run["summary"].get(
                        "subsets", {}
                    ).get(subset, {}),
                    key,
                )
                for name, run in runs.items()
            }

            if all(v is not None for v in values.values()):
                entries.append((label, values))

        if entries:
            grouped_bars(ax, entries)

        counts = [
            run["summary"].get(
                "subsets", {}
            ).get(subset, {}).get("count")
            for run in runs.values()
        ]

        label = subset.title()
        if all(c == counts[0] for c in counts):
            label += f" (n={counts[0]})"

        ax.set_title(label, loc="left", pad=13)
        ax.set_xlabel("Detection score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3, frameon=False,
    )
    fig.suptitle(
        "Detection by image condition",
        fontsize=15, fontweight="bold",
        y=1.03,
    )

    fig.subplots_adjust(
        left=0.11, right=0.98,
        bottom=0.18, top=0.72,
        wspace=0.30,
    )
    finish(fig, output / "02_degradation.png", dpi)
    return "02_degradation.png"


def latency_figure(runs, ids, output, dpi):
    values = []

    for run in runs.values():
        series = [
            numeric(
                run["rows"][sid].get("end_to_end_ms")
            )
            for sid in ids
        ]
        if any(v is None for v in series):
            return None
        values.append(np.array(series, dtype=float))

    fig, ax = plt.subplots(figsize=(8, 4.7))
    bp = ax.boxplot(
        values,
        tick_labels=NAMES,
        patch_artist=True,
        showfliers=True,
        widths=0.42,
        medianprops={
            "color": "#202B36",
            "linewidth": 2,
        },
        flierprops={
            "marker": "o",
            "markersize": 3,
            "alpha": 0.3,
        },
    )

    for patch, name in zip(bp["boxes"], NAMES):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COLORS[name])

    for index, series in enumerate(values, start=1):
        mean = np.mean(series)
        median = np.median(series)

        ax.scatter(
            index, mean,
            color=COLORS[NAMES[index - 1]],
            marker="D", s=36, zorder=5,
        )
        ax.text(
            index + 0.17, median,
            f"{median:.1f} ms",
            fontsize=9, va="center",
        )

    ax.set_xlim(0.5, 3.7)
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title(
        "End-to-end inference latency",
        loc="left", pad=15,
    )
    clean(ax, "y")
    fig.tight_layout()

    finish(fig, output / "03_latency.png", dpi)
    return "03_latency.png"


def mean_stage(run, key):
    value = run["latency"].get(key)
    if isinstance(value, dict):
        return numeric(value.get("mean"))
    return numeric(value)


def stages_figure(runs, output, dpi):
    stages = {
        "CrossFuse": (
            ("Encoder", "encoder_ms"),
            ("Fusion", "fusion_ms"),
            ("Detection", "detection_nms_ms"),
        ),
        "PACE": (
            ("Encoder + quality + gate", "encoder_quality_gate_ms"),
            ("Fusion", "selective_fusion_ms"),
            ("Detection", "detection_nms_ms"),
        ),
        "Addition": (
            ("Fusion", "fusion_ms"),
            ("Detection", "detection_nms_ms"),
        ),
    }

    stage_colors = {
        "Encoder": "#A2AEB8",
        "Encoder + quality + gate": "#A2AEB8",
        "Fusion": "#5A7F9B",
        "Detection": "#C6CDD3",
    }

    fig, ax = plt.subplots(figsize=(10, 4.1))
    totals = []

    for index, name in enumerate(NAMES):
        left = 0

        for label, key in stages[name]:
            value = mean_stage(runs[name], key)
            if value is None:
                raise ValueError(
                    f"Missing latency stage: {name}/{key}"
                )

            ax.barh(
                index, value,
                left=left,
                height=0.48,
                color=stage_colors[label],
                edgecolor="white",
            )

            if value >= 5:
                ax.text(
                    left + value / 2, index,
                    f"{value:.1f}",
                    ha="center", va="center",
                    fontsize=9,
                )
            left += value

        totals.append(left)
        ax.text(
            left + 1.5, index,
            f"{left:.1f} ms",
            va="center", fontweight="bold",
        )

    ax.set_yticks(range(3), NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.27)
    ax.set_xlabel("Mean stage runtime (ms)")
    ax.set_title(
        "Where inference time is spent",
        loc="left", pad=15,
    )
    clean(ax)

    fig.tight_layout()
    finish(fig, output / "04_stage_latency.png", dpi)
    return "04_stage_latency.png"


def memory_figure(runs, output, dpi):
    values = [
        mean_stage(run, "cuda_peak_allocated_mb")
        for run in runs.values()
    ]
    if any(v is None for v in values):
        return None

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(
        NAMES, values,
        color=[COLORS[n] for n in NAMES],
        height=0.48,
    )

    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f} MB",
            va="center",
        )

    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("Peak allocated CUDA memory (MB)")
    ax.set_title("GPU memory", loc="left")
    clean(ax)
    fig.tight_layout()

    finish(fig, output / "05_memory.png", dpi)
    return "05_memory.png"


def gate_figure(runs, ids, output, dpi):
    pace = runs["PACE"]
    bins = [
        ("0%", lambda x: x == 0),
        (">0–10%", lambda x: 0 < x <= 10),
        (">10–25%", lambda x: 10 < x <= 25),
        (">25%", lambda x: x > 25),
    ]

    pairs = []
    for sid in ids:
        row = pace["rows"][sid]
        fraction = numeric(row.get("active_fraction"))
        latency = numeric(row.get("end_to_end_ms"))
        if fraction is not None and latency is not None:
            pairs.append((fraction * 100, latency))

    groups = [
        (
            label,
            np.array([
                t for fraction, t in pairs
                if predicate(fraction)
            ]),
        )
        for label, predicate in bins
    ]
    groups = [
        (label, values)
        for label, values in groups if len(values)
    ]

    if not groups:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4))
    means = [v.mean() for _, v in groups]
    bars = ax.bar(
        range(len(groups)), means,
        color=COLORS["PACE"], width=0.55,
    )

    for bar, (_, values) in zip(bars, groups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(means) * 0.025,
            f"{values.mean():.1f} ms\nn={len(values)}",
            ha="center", fontsize=9,
        )

    ax.set_xticks(
        range(len(groups)),
        [label for label, _ in groups],
    )
    ax.set_ylim(0, max(means) * 1.30)
    ax.set_ylabel("Mean PACE E2E latency (ms)")
    ax.set_xlabel("Patches selected for fusion")
    ax.set_title(
        "PACE gate activity versus latency",
        loc="left", pad=12,
    )
    clean(ax, "y")
    fig.tight_layout()

    finish(fig, output / "06_gate_latency.png", dpi)
    return "06_gate_latency.png"


def tradeoff_figure(runs, output, dpi):
    cross = runs["CrossFuse"]
    pace = runs["PACE"]

    cross_fusion = mean_stage(cross, "fusion_ms")
    pace_fusion = mean_stage(pace, "selective_fusion_ms")
    cross_encoder = mean_stage(cross, "encoder_ms")
    pace_encoder = mean_stage(
        pace, "encoder_quality_gate_ms"
    )

    if any(
        value is None
        for value in (
            cross_fusion, pace_fusion,
            cross_encoder, pace_encoder,
        )
    ):
        return None

    savings = cross_fusion - pace_fusion
    overhead = pace_encoder - cross_encoder

    labels = (
        "Fusion time saved",
        "Additional encoder / gate cost",
    )
    values = (savings, overhead)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(
        labels, values,
        color=("#438C78", "#BF6B55"),
        height=0.45,
    )

    limit = max(abs(v) for v in values) or 1
    ax.set_xlim(-limit * 0.25, limit * 1.4)

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.02 * limit,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f} ms",
            va="center", fontweight="bold",
        )

    ax.axvline(0, color="#74808B")
    ax.invert_yaxis()
    ax.set_xlabel("Time difference (ms)")
    ax.set_title(
        "PACE fusion savings versus selection overhead",
        loc="left", pad=15,
    )
    clean(ax)
    fig.tight_layout()

    finish(fig, output / "07_tradeoff.png", dpi)
    return "07_tradeoff.png"


def gallery(runs, ids, output, count):
    if count == 0:
        return 0

    directory = output / "visualization_pairs"
    directory.mkdir(parents=True, exist_ok=True)
    created = 0

    for sid in ids:
        paths = [
            runs[name]["path"]
            / "visualizations"
            / f"{sid}.png"
            for name in NAMES
        ]
        if not all(p.exists() for p in paths):
            continue

        images = []
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB")
                height = 300
                width = max(
                    1, round(image.width * height / image.height)
                )
                images.append(image.resize((width, height)))

        gap = 16
        header = 38
        width = sum(im.width for im in images) + 2 * gap
        canvas = Image.new(
            "RGB", (width, 300 + header), "white"
        )
        draw = ImageDraw.Draw(canvas)

        x = 0
        for name, image in zip(NAMES, images):
            draw.text((x + 5, 10), name, fill="black")
            canvas.paste(image, (x, header))
            x += image.width + gap

        canvas.save(
            directory / f"{sid}_comparison.png"
        )
        created += 1

        if created >= count:
            break

    return created


def main():
    args = arguments()
    style()

    runs = {
        "CrossFuse": load_run(args.crossfuse),
        "PACE": load_run(args.pace),
        "Addition": load_run(args.addition),
    }

    ids = validate(runs)

    output = args.output.resolve()
    source_paths = [
        run["path"] for run in runs.values()
    ]
    if any(
        output == source or output.is_relative_to(source)
        for source in source_paths
    ):
        raise ValueError(
            "Comparison output cannot be inside an input run."
        )

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    generators = [
        lambda: detection_figure(runs, figures, args.dpi),
        lambda: degradation_figure(runs, figures, args.dpi),
        lambda: latency_figure(runs, ids, figures, args.dpi),
        lambda: stages_figure(runs, figures, args.dpi),
        lambda: memory_figure(runs, figures, args.dpi),
        lambda: gate_figure(runs, ids, figures, args.dpi),
        lambda: tradeoff_figure(runs, figures, args.dpi),
    ]

    generated = [
        result
        for generate in generators
        if (result := generate()) is not None
    ]

    visualizations = gallery(
        runs, ids, output, args.visualizations
    )

    statistics = {
        "n_samples": len(ids),
        "matched_sample_ids": ids,
        "source_directories": {
            name: str(run["path"])
            for name, run in runs.items()
        },
        "detection": {
            name: run["summary"].get("detection", {})
            for name, run in runs.items()
        },
        "subsets": {
            name: run["summary"].get("subsets", {})
            for name, run in runs.items()
        },
        "latency": {
            name: run["latency"]
            for name, run in runs.items()
        },
        "figures": generated,
        "visualization_pairs": visualizations,
    }

    with (output / "comparison.json").open("w") as f:
        json.dump(statistics, f, indent=2)

    with (output / "statistics.json").open("w") as f:
        json.dump(statistics, f, indent=2)

    lines = [
        "# CrossFuse vs. PACE vs. Addition",
        "",
        f"Compared **{len(ids)} matched samples**.",
        "",
        "| Metric | CrossFuse | PACE | Addition |",
        "|---|---:|---:|---:|",
    ]

    for key, label in METRICS:
        values = [
            metric_value(
                runs[name]["summary"].get("detection", {}),
                key,
            )
            for name in NAMES
        ]
        lines.append(
            "| " + label + " | "
            + " | ".join(
                f"{v:.4f}" if v is not None else "N/A"
                for v in values
            )
            + " |"
        )

    lines.extend([
        "",
        "## Methodology",
        "",
        "- All metrics were loaded from existing evaluation outputs.",
        "- No CrossFuse or PACE inference was performed.",
        "- All three runs must contain identical sample IDs.",
        "- Detection metrics are dataset-level point estimates.",
        "- Latencies are single-run device measurements.",
        "- Gate and tradeoff figures diagnose PACE relative to CrossFuse.",
        "",
        "## Figures",
        "",
        *[f"- `figures/{name}`" for name in generated],
        "",
        "## Inputs",
        "",
        *[
            f"- {name}: `{runs[name]['path']}`"
            for name in NAMES
        ],
    ])

    (output / "comparison_report.md").write_text(
        "\n".join(lines)
    )

    print(f"Matched {len(ids)} samples")
    print(f"Generated {len(generated)} figures")
    print(f"Generated {visualizations} visualization comparisons")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()