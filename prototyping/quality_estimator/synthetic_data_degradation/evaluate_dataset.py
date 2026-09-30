"""
Evaluate a generated PACE synthetic degradation dataset.

Evaluates:

1. Original RGB lighting distribution
2. RGB quality-label distribution
3. IR quality-label distribution
4. Clean vs. synthetically degraded balance
5. RGB vs. IR degradation balance
6. Degradation-type distribution
7. Severity distribution
8. Representative RGB samples for manual inspection
9. Samples near the lighting decision boundary

IMPORTANT:
Lighting statistics are computed from the ORIGINAL source RGB image
after applying the same 100-pixel crop used by build_dataset.py.
This prevents synthetic degradation from contaminating lighting analysis.

Run from PACE root:

    python prototyping/quality_estimator/synthetic_data_degradation/evaluate_dataset.py
"""

import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PROCESSED_ROOT = Path("processed_dataset")
METADATA_PATH = PROCESSED_ROOT / "metadata.csv"

GRID_SIZE = 8
BORDER_PIXELS = 100

# IMPORTANT:
# These should match the thresholds currently used in lighting.py.
#
# We are evaluating whether these values are appropriate.
DARK_THRESHOLD = 0.05
GOOD_THRESHOLD = 0.15


# ------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------

def load_metadata():
    """Load generated metadata.csv."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(
            "metadata.csv contains no samples."
        )

    return rows


def apply_crop(image, border_pixels=BORDER_PIXELS):
    """
    Apply the same fixed DroneVehicle crop used during generation.
    """

    if image is None:
        raise ValueError("Cannot crop None image.")

    height, width = image.shape[:2]

    if (
        height <= 2 * border_pixels
        or width <= 2 * border_pixels
    ):
        raise ValueError(
            f"Image shape {image.shape} is too small "
            f"for a {border_pixels}px border crop."
        )

    return image[
        border_pixels:height - border_pixels,
        border_pixels:width - border_pixels
    ]


def compute_luminance(image):
    """
    Compute normalized RGB luminance in [0, 1].

    OpenCV loads color images as BGR.
    """

    image = image.astype(np.float32) / 255.0

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    luminance = (
        0.2126 * r
        + 0.7152 * g
        + 0.0722 * b
    )

    return luminance


def describe(values):
    """Return summary statistics for a sequence."""

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if values.size == 0:
        return None

    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def print_description(title, values):
    """Print summary statistics."""

    stats = describe(values)

    print()
    print(title)
    print("-" * len(title))

    if stats is None:
        print("No values.")
        return

    keys = [
        "min",
        "p05",
        "p10",
        "p25",
        "median",
        "p75",
        "p90",
        "p95",
        "max",
        "mean",
    ]

    for key in keys:
        print(
            f"{key:>7}: "
            f"{stats[key]:.4f}"
        )


# ------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------

def evaluate_dataset():

    rows = load_metadata()

    print("=" * 70)
    print("PACE SYNTHETIC DATASET EVALUATION")
    print("=" * 70)

    print(
        f"\nSamples in metadata: "
        f"{len(rows)}"
    )

    print(
        f"Lighting thresholds under evaluation: "
        f"DARK={DARK_THRESHOLD:.3f}, "
        f"GOOD={GOOD_THRESHOLD:.3f}"
    )

    # ==============================================================
    # Statistics containers
    # ==============================================================

    valid_pair_ids = []

    image_mean_luminances = []
    image_median_luminances = []

    dark_pixel_fractions = []
    good_pixel_fractions = []
    intermediate_pixel_fractions = []

    rgb_quality_values = []
    ir_quality_values = []

    rgb_image_mean_quality = []
    ir_image_mean_quality = []

    skipped_samples = []

    # ==============================================================
    # Process samples
    # ==============================================================

    for index, row in enumerate(
        rows,
        start=1
    ):

        pair_id = row["pair_id"]

        # ----------------------------------------------------------
        # IMPORTANT:
        # Evaluate lighting from ORIGINAL source RGB, not degraded RGB
        # ----------------------------------------------------------

        source_rgb_path = Path(
            row["source_rgb"]
        )

        source_rgb = cv2.imread(
            str(source_rgb_path),
            cv2.IMREAD_COLOR
        )

        if source_rgb is None:

            print(
                f"WARNING: could not read source RGB "
                f"for {pair_id}: {source_rgb_path}"
            )

            skipped_samples.append(
                pair_id
            )

            continue

        source_rgb = apply_crop(
            source_rgb
        )

        luminance = compute_luminance(
            source_rgb
        )

        # ----------------------------------------------------------
        # Lighting statistics
        # ----------------------------------------------------------

        mean_luminance = float(
            np.mean(luminance)
        )

        median_luminance = float(
            np.median(luminance)
        )

        dark_fraction = float(
            np.mean(
                luminance <= DARK_THRESHOLD
            )
        )

        good_fraction = float(
            np.mean(
                luminance >= GOOD_THRESHOLD
            )
        )

        intermediate_fraction = float(
            np.mean(
                (luminance > DARK_THRESHOLD)
                &
                (luminance < GOOD_THRESHOLD)
            )
        )

        # ----------------------------------------------------------
        # Load quality maps
        # ----------------------------------------------------------

        rgb_quality_path = Path(
            row["rgb_quality"]
        )

        ir_quality_path = Path(
            row["ir_quality"]
        )

        if not rgb_quality_path.exists():

            print(
                f"WARNING: missing RGB quality map "
                f"for {pair_id}: {rgb_quality_path}"
            )

            skipped_samples.append(
                pair_id
            )

            continue

        if not ir_quality_path.exists():

            print(
                f"WARNING: missing IR quality map "
                f"for {pair_id}: {ir_quality_path}"
            )

            skipped_samples.append(
                pair_id
            )

            continue

        rgb_quality = np.load(
            rgb_quality_path
        )

        ir_quality = np.load(
            ir_quality_path
        )

        if rgb_quality.shape != (
            GRID_SIZE,
            GRID_SIZE
        ):

            raise ValueError(
                f"{pair_id} RGB quality map "
                f"has shape {rgb_quality.shape}; "
                f"expected ({GRID_SIZE}, {GRID_SIZE})."
            )

        if ir_quality.shape != (
            GRID_SIZE,
            GRID_SIZE
        ):

            raise ValueError(
                f"{pair_id} IR quality map "
                f"has shape {ir_quality.shape}; "
                f"expected ({GRID_SIZE}, {GRID_SIZE})."
            )

        # ----------------------------------------------------------
        # Record sample statistics
        # ----------------------------------------------------------

        valid_pair_ids.append(
            pair_id
        )

        image_mean_luminances.append(
            mean_luminance
        )

        image_median_luminances.append(
            median_luminance
        )

        dark_pixel_fractions.append(
            dark_fraction
        )

        good_pixel_fractions.append(
            good_fraction
        )

        intermediate_pixel_fractions.append(
            intermediate_fraction
        )

        rgb_quality_values.extend(
            rgb_quality.flatten().tolist()
        )

        ir_quality_values.extend(
            ir_quality.flatten().tolist()
        )

        rgb_image_mean_quality.append(
            float(
                np.mean(rgb_quality)
            )
        )

        ir_image_mean_quality.append(
            float(
                np.mean(ir_quality)
            )
        )

        if index % 100 == 0:
            print(
                f"Processed {index}/{len(rows)}"
            )

    # ==============================================================
    # Basic validation
    # ==============================================================

    if not valid_pair_ids:
        raise ValueError(
            "No valid samples could be evaluated."
        )

    print(
        f"\nSuccessfully evaluated: "
        f"{len(valid_pair_ids)}"
    )

    if skipped_samples:
        print(
            f"Skipped samples: "
            f"{len(skipped_samples)}"
        )

    # ==============================================================
    # 1. RGB lighting distribution
    # ==============================================================

    print()
    print("=" * 70)
    print("1. ORIGINAL RGB LIGHTING DISTRIBUTION")
    print("=" * 70)

    print_description(
        "Mean luminance per image",
        image_mean_luminances
    )

    print_description(
        "Median luminance per image",
        image_median_luminances
    )

    print_description(
        f"Fraction of pixels <= DARK_THRESHOLD "
        f"({DARK_THRESHOLD:.2f})",
        dark_pixel_fractions
    )

    print_description(
        f"Fraction of pixels >= GOOD_THRESHOLD "
        f"({GOOD_THRESHOLD:.2f})",
        good_pixel_fractions
    )

    print_description(
        "Fraction of intermediate-light pixels",
        intermediate_pixel_fractions
    )

    # ==============================================================
    # 2. RGB quality distribution
    # ==============================================================

    print()
    print("=" * 70)
    print("2. RGB QUALITY DISTRIBUTION")
    print("=" * 70)

    rgb_values = np.asarray(
        rgb_quality_values,
        dtype=np.float32
    )

    print_description(
        "All RGB 8x8 quality values",
        rgb_values
    )

    pct_very_low = float(
        np.mean(
            rgb_values <= 0.10
        )
    )

    pct_low = float(
        np.mean(
            (rgb_values > 0.10)
            &
            (rgb_values < 0.50)
        )
    )

    pct_medium = float(
        np.mean(
            (rgb_values >= 0.50)
            &
            (rgb_values < 0.90)
        )
    )

    pct_high = float(
        np.mean(
            rgb_values >= 0.90
        )
    )

    print()
    print("RGB patch categories:")

    print(
        f"  quality <= 0.10:  "
        f"{pct_very_low:.2%}"
    )

    print(
        f"  quality 0.10-0.50: "
        f"{pct_low:.2%}"
    )

    print(
        f"  quality 0.50-0.90: "
        f"{pct_medium:.2%}"
    )

    print(
        f"  quality >= 0.90:   "
        f"{pct_high:.2%}"
    )

    print_description(
        "Mean RGB quality per image",
        rgb_image_mean_quality
    )

    # ==============================================================
    # 3. IR quality distribution
    # ==============================================================

    print()
    print("=" * 70)
    print("3. IR QUALITY DISTRIBUTION")
    print("=" * 70)

    ir_values = np.asarray(
        ir_quality_values,
        dtype=np.float32
    )

    print_description(
        "All IR 8x8 quality values",
        ir_values
    )

    ir_perfect_fraction = float(
        np.mean(
            np.isclose(
                ir_values,
                1.0
            )
        )
    )

    print(
        f"\nIR patches at exactly 1.0: "
        f"{ir_perfect_fraction:.2%}"
    )

    print_description(
        "Mean IR quality per image",
        ir_image_mean_quality
    )

    # ==============================================================
    # 4. Synthetic degradation distribution
    # ==============================================================

    print()
    print("=" * 70)
    print("4. SYNTHETIC DEGRADATION DISTRIBUTION")
    print("=" * 70)

    degraded_rows = [
        row
        for row in rows
        if row["degraded"].strip().lower()
        == "true"
    ]

    clean_rows = [
        row
        for row in rows
        if row["degraded"].strip().lower()
        != "true"
    ]

    clean_fraction = (
        len(clean_rows) /
        len(rows)
    )

    degraded_fraction = (
        len(degraded_rows) /
        len(rows)
    )

    print(
        f"\nClean examples: "
        f"{len(clean_rows)} "
        f"({clean_fraction:.2%})"
    )

    print(
        f"Actually degraded examples: "
        f"{len(degraded_rows)} "
        f"({degraded_fraction:.2%})"
    )

    # --------------------------------------------------------------
    # Modality distribution
    # --------------------------------------------------------------

    modality_counts = Counter(
        row["modality"]
        for row in degraded_rows
    )

    print()
    print("Degraded modalities:")

    num_degraded = max(
        len(degraded_rows),
        1
    )

    for modality in sorted(
        modality_counts
    ):

        count = modality_counts[
            modality
        ]

        fraction = (
            count /
            num_degraded
        )

        print(
            f"  {modality}: "
            f"{count} "
            f"({fraction:.2%})"
        )

    # --------------------------------------------------------------
    # Degradation-type distribution
    # --------------------------------------------------------------

    type_counts = Counter(
        row["degradation_type"]
        for row in degraded_rows
    )

    print()
    print("Degradation types:")

    for degradation_type in sorted(
        type_counts
    ):

        count = type_counts[
            degradation_type
        ]

        fraction = (
            count /
            num_degraded
        )

        print(
            f"  {degradation_type}: "
            f"{count} "
            f"({fraction:.2%})"
        )

    # --------------------------------------------------------------
    # Modality + type combinations
    # --------------------------------------------------------------

    combination_counts = Counter(
        (
            row["modality"],
            row["degradation_type"]
        )
        for row in degraded_rows
    )

    print()
    print("Modality / degradation combinations:")

    for combination in sorted(
        combination_counts
    ):

        modality, degradation_type = (
            combination
        )

        count = combination_counts[
            combination
        ]

        fraction = (
            count /
            num_degraded
        )

        print(
            f"  {modality:<3} / "
            f"{degradation_type:<15}: "
            f"{count:>4} "
            f"({fraction:.2%})"
        )

    # --------------------------------------------------------------
    # Severity distribution
    # --------------------------------------------------------------

    if degraded_rows:

        severities = [
            float(
                row["severity"]
            )
            for row in degraded_rows
        ]

        print_description(
            "Degradation severity",
            severities
        )

    # ==============================================================
    # 5. Representative RGB samples
    # ==============================================================

    print()
    print("=" * 70)
    print("5. REPRESENTATIVE RGB SAMPLES")
    print("=" * 70)

    ranked = list(
        zip(
            valid_pair_ids,
            rgb_image_mean_quality,
            image_mean_luminances,
            image_median_luminances,
            dark_pixel_fractions,
            good_pixel_fractions
        )
    )

    ranked.sort(
        key=lambda item: item[1]
    )

    number_to_show = min(
        15,
        len(ranked)
    )

    representative_indices = (
        np.linspace(
            0,
            len(ranked) - 1,
            number_to_show,
            dtype=int
        )
    )

    print(
        "\nSamples spanning the RGB "
        "quality distribution:\n"
    )

    print(
        f"{'pair':<10}"
        f"{'mean Q':>10}"
        f"{'mean lum':>12}"
        f"{'med lum':>12}"
        f"{'dark %':>10}"
        f"{'good %':>10}"
    )

    print(
        "-" * 64
    )

    representative_ids = []

    for representative_index in (
        representative_indices
    ):

        (
            pair_id,
            mean_quality,
            mean_luminance,
            median_luminance,
            dark_fraction,
            good_fraction
        ) = ranked[
            representative_index
        ]

        representative_ids.append(
            pair_id
        )

        print(
            f"{pair_id:<10}"
            f"{mean_quality:>10.3f}"
            f"{mean_luminance:>12.3f}"
            f"{median_luminance:>12.3f}"
            f"{dark_fraction:>9.1%}"
            f"{good_fraction:>9.1%}"
        )

    # ==============================================================
    # 6. Lighting threshold cases
    # ==============================================================

    print()
    print("=" * 70)
    print("6. LIGHTING THRESHOLD CASES")
    print("=" * 70)

    # --------------------------------------------------------------
    # Samples around 50% sufficiently-lit coverage
    # --------------------------------------------------------------

    borderline_good = list(
        zip(
            valid_pair_ids,
            good_pixel_fractions,
            dark_pixel_fractions,
            image_mean_luminances,
            rgb_image_mean_quality
        )
    )

    borderline_good.sort(
        key=lambda item:
        abs(
            item[1] - 0.50
        )
    )

    print(
        "\nImages closest to 50% "
        "sufficiently-lit pixels:\n"
    )

    print(
        f"{'pair':<10}"
        f"{'good %':>10}"
        f"{'dark %':>10}"
        f"{'mean lum':>12}"
        f"{'mean Q':>10}"
    )

    print(
        "-" * 52
    )

    borderline_ids = []

    for item in borderline_good[:10]:

        (
            pair_id,
            good_fraction,
            dark_fraction,
            mean_luminance,
            mean_quality
        ) = item

        borderline_ids.append(
            pair_id
        )

        print(
            f"{pair_id:<10}"
            f"{good_fraction:>9.1%}"
            f"{dark_fraction:>9.1%}"
            f"{mean_luminance:>12.3f}"
            f"{mean_quality:>10.3f}"
        )

    # --------------------------------------------------------------
    # Very dark samples
    # --------------------------------------------------------------

    darkest = sorted(
        zip(
            valid_pair_ids,
            image_mean_luminances,
            rgb_image_mean_quality,
            dark_pixel_fractions
        ),
        key=lambda item: item[1]
    )

    print(
        "\nDarkest RGB images:\n"
    )

    print(
        f"{'pair':<10}"
        f"{'mean lum':>12}"
        f"{'mean Q':>10}"
        f"{'dark %':>10}"
    )

    print(
        "-" * 42
    )

    darkest_ids = []

    for item in darkest[:10]:

        (
            pair_id,
            mean_luminance,
            mean_quality,
            dark_fraction
        ) = item

        darkest_ids.append(
            pair_id
        )

        print(
            f"{pair_id:<10}"
            f"{mean_luminance:>12.3f}"
            f"{mean_quality:>10.3f}"
            f"{dark_fraction:>9.1%}"
        )

    # --------------------------------------------------------------
    # Brightest samples
    # --------------------------------------------------------------

    brightest = sorted(
        zip(
            valid_pair_ids,
            image_mean_luminances,
            rgb_image_mean_quality,
            good_pixel_fractions
        ),
        key=lambda item: item[1],
        reverse=True
    )

    print(
        "\nBrightest RGB images:\n"
    )

    print(
        f"{'pair':<10}"
        f"{'mean lum':>12}"
        f"{'mean Q':>10}"
        f"{'good %':>10}"
    )

    print(
        "-" * 42
    )

    brightest_ids = []

    for item in brightest[:10]:

        (
            pair_id,
            mean_luminance,
            mean_quality,
            good_fraction
        ) = item

        brightest_ids.append(
            pair_id
        )

        print(
            f"{pair_id:<10}"
            f"{mean_luminance:>12.3f}"
            f"{mean_quality:>10.3f}"
            f"{good_fraction:>9.1%}"
        )

    # ==============================================================
    # 7. Basic integrity checks
    # ==============================================================

    print()
    print("=" * 70)
    print("7. BASIC LABEL INTEGRITY")
    print("=" * 70)

    rgb_out_of_range = int(
        np.sum(
            (rgb_values < 0.0)
            |
            (rgb_values > 1.0)
        )
    )

    ir_out_of_range = int(
        np.sum(
            (ir_values < 0.0)
            |
            (ir_values > 1.0)
        )
    )

    rgb_nan = int(
        np.sum(
            np.isnan(rgb_values)
        )
    )

    ir_nan = int(
        np.sum(
            np.isnan(ir_values)
        )
    )

    print(
        f"\nRGB values outside [0,1]: "
        f"{rgb_out_of_range}"
    )

    print(
        f"IR values outside [0,1]:  "
        f"{ir_out_of_range}"
    )

    print(
        f"RGB NaN values: "
        f"{rgb_nan}"
    )

    print(
        f"IR NaN values:  "
        f"{ir_nan}"
    )

    # ==============================================================
    # 8. Manual inspection commands
    # ==============================================================

    print()
    print("=" * 70)
    print("8. RECOMMENDED MANUAL INSPECTION")
    print("=" * 70)

    script_path = (
        "prototyping/quality_estimator/"
        "synthetic_data_degradation/"
        "visualize_quality.py"
    )

    # Avoid duplicates while preserving order.
    inspection_ids = []

    candidate_ids = (
        darkest_ids[:3]
        + borderline_ids[:5]
        + representative_ids[::4]
        + brightest_ids[:2]
    )

    for pair_id in candidate_ids:

        if pair_id not in inspection_ids:
            inspection_ids.append(
                pair_id
            )

    print(
        "\nInspect these RGB examples before "
        "changing lighting thresholds:\n"
    )

    for pair_id in inspection_ids:

        print(
            f"python {script_path} "
            f"{pair_id} rgb"
        )

    # ==============================================================
    # 9. Interpretation reminders
    # ==============================================================

    print()
    print("=" * 70)
    print("9. WHAT TO LOOK FOR")
    print("=" * 70)

    print(
        """
Use these statistics together with the visualizations.

For DARK_THRESHOLD:
  - Inspect the darkest and low-quality examples.
  - Regions at/below this threshold should contain essentially
    unusable visible information.
  - If useful object/scene information is being labeled near 0,
    DARK_THRESHOLD may be too high.
  - If effectively black regions receive meaningful quality,
    DARK_THRESHOLD may be too low.

For GOOD_THRESHOLD:
  - Inspect the borderline examples.
  - Regions at/above this threshold should contain clearly usable
    visible information.
  - If visibly poor/dim regions are already treated as fully good,
    GOOD_THRESHOLD may be too low.
  - If clearly usable regions rarely reach full quality,
    GOOD_THRESHOLD may be too high.

For degradation balance:
  - Compare the actual degraded percentage with the desired dataset
    composition.
  - Check RGB vs IR balance.
  - Check degradation-type balance.
  - Remember that RGB attempts may be rejected when too little of the
    sampled mask overlaps sufficiently illuminated image regions.

Do not choose new thresholds from numerical percentiles alone.
The final decision should combine these statistics with visual
inspection of representative images.
"""
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_dataset()