"""
we're facing some issues with the quality estimation model 
--> use this script to explore possible issues

key issues to analyze:
1) ground truth confirmation -- confirm that the ground truth labels show low scores for degraded regions as expected
2) measure label imbalance --  see how many patches are falling into different score categories (<.5, <.7, <.9) 
3) possible improvement --> measure degraded patch performance separately 

notes:
- ground truth is fine!
- becuase of the imbalance of data, we need to pay more attention to the degraded patches when computing the loss
    - one way to do this is via weighted MSE
"""
from pathlib import Path
import random

from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "processed_dataset"

SPLIT = "train"
MODALITY = "ir"
NUM_SAMPLES = 3
RANDOM_SEED = 42


def inspect_ground_truth():
    base = DATASET_ROOT / SPLIT
    quality_dir = base / "quality_maps"
    image_dir = base / MODALITY

    paths = sorted(quality_dir.glob(f"*_{MODALITY}.npy"))

    if not paths:
        raise FileNotFoundError(quality_dir)

    rng = random.Random(RANDOM_SEED)

    # 3 columns, image above quality map.
    fig, axes = plt.subplots(
        2, NUM_SAMPLES,
        figsize=(18, 9),
        gridspec_kw={"height_ratios": [1.3, 1]},
    )

    plt.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.92,
        bottom=0.12,
        hspace=0.20,
        wspace=0.12,
    )

    def display_samples(event=None):
        selected = rng.sample(paths, min(NUM_SAMPLES, len(paths)))

        for col, path in enumerate(selected):
            sample_id = path.stem.removesuffix(f"_{MODALITY}")
            image_path = image_dir / f"{sample_id}.jpg"

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE if MODALITY == "ir"
                else cv2.IMREAD_COLOR,
            )

            if image is None:
                raise FileNotFoundError(image_path)

            if MODALITY == "rgb":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            quality = np.load(path)

            ax_img = axes[0, col]
            ax_quality = axes[1, col]

            ax_img.clear()
            ax_quality.clear()

            ax_img.imshow(
                image,
                cmap="gray" if MODALITY == "ir" else None,
            )
            ax_img.set_title(f"Image {sample_id}", fontsize=13)
            ax_img.axis("off")

            ax_quality.imshow(
                quality,
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax_quality.set_title(
                f"Ground Truth | min={quality.min():.2f}",
                fontsize=12,
            )
            ax_quality.set_xticks(np.arange(-0.5, 8, 1), minor=True)
            ax_quality.set_yticks(np.arange(-0.5, 8, 1), minor=True)
            ax_quality.grid(
                which="minor",
                color="black",
                linewidth=0.5,
                alpha=0.4,
            )
            ax_quality.tick_params(
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )

        fig.canvas.draw_idle()

    button_ax = fig.add_axes([0.39, 0.025, 0.22, 0.055])
    button = Button(button_ax, "New Random 3")
    button.on_clicked(display_samples)

    display_samples()
    plt.show()


def analyze_quality_distribution(split="train", modality="ir"):
    quality_dir = DATASET_ROOT / split / "quality_maps"
    paths = sorted(quality_dir.glob(f"*_{modality}.npy"))

    if not paths:
        raise FileNotFoundError(quality_dir)

    # Collect all ground-truth patch quality scores.
    all_quality = np.concatenate([
        np.load(path).ravel()
        for path in tqdm(
            paths,
            desc=f"Analyzing {modality.upper()} quality maps",
            unit="image",
        )
    ])

    # -----------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------

    print(f"\n{'=' * 50}")
    print(f"{modality.upper()} QUALITY DISTRIBUTION")
    print(f"{'=' * 50}")

    print(f"Images:  {len(paths):,}")
    print(f"Patches: {len(all_quality):,}")
    print(f"Mean:    {all_quality.mean():.4f}")
    print(f"Median:  {np.median(all_quality):.4f}")
    print(f"Std:     {all_quality.std():.4f}")
    print(f"Min:     {all_quality.min():.4f}")
    print(f"Max:     {all_quality.max():.4f}")

    # -----------------------------------------------------
    # Distribution by quality range
    # -----------------------------------------------------

    print("\nPatch quality distribution:")

    bins = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 0.95),
        (0.95, 1.01),
    ]

    for low, high in bins:
        count = np.sum(
            (all_quality >= low) & (all_quality < high)
        )
        pct = 100 * count / len(all_quality)

        print(
            f"  [{low:.2f}, {min(high, 1.0):.2f}]: "
            f"{count:,} patches ({pct:.2f}%)"
        )

    # -----------------------------------------------------
    # Histogram
    # -----------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.hist(
        all_quality,
        bins=50,
        range=(0, 1),
        edgecolor="black",
        linewidth=0.3,
    )

    plt.xlabel("Ground-truth patch quality")
    plt.ylabel("Number of patches")
    plt.title(
        f"{modality.upper()} Patch Quality Distribution "
        f"({split})"
    )

    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # Export numerical results to TXT
    # -----------------------------------------------------

    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / f"quality_distribution_{modality}_{split}.txt"

    bins = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 0.95),
        (0.95, 1.01),
    ]

    lines = [
        "=" * 55,
        f"{modality.upper()} GROUND-TRUTH QUALITY DISTRIBUTION",
        "=" * 55,
        f"Split:             {split}",
        f"Images:            {len(paths):,}",
        f"Total patches:     {len(all_quality):,}",
        "",
        "SUMMARY STATISTICS",
        "-" * 55,
        f"Mean:              {all_quality.mean():.4f}",
        f"Median:            {np.median(all_quality):.4f}",
        f"Standard deviation:{all_quality.std():.4f}",
        f"Minimum:           {all_quality.min():.4f}",
        f"Maximum:           {all_quality.max():.4f}",
        "",
        "PATCH QUALITY DISTRIBUTION",
        "-" * 55,
    ]

    for low, high in bins:
        count = np.sum((all_quality >= low) & (all_quality < high))
        pct = 100 * count / len(all_quality)
        lines.append(
            f"[{low:.2f}, {min(high, 1.0):.2f}]: "
            f"{count:,} patches ({pct:.2f}%)"
        )

    clean_pct = 100 * np.mean(all_quality >= 0.95)
    degraded_pct = 100 * np.mean(all_quality < 0.8)
    severe_pct = 100 * np.mean(all_quality < 0.5)

    lines.extend([
        "",
        "KEY TAKEAWAYS",
        "-" * 55,
        f"Near-clean patches (>= 0.95): {clean_pct:.2f}%",
        f"Degraded patches   (< 0.80):  {degraded_pct:.2f}%",
        f"Severely degraded  (< 0.50):  {severe_pct:.2f}%",
        "",
        "INTERPRETATION",
        "-" * 55,
        (
            "Strong class imbalance: most patches have near-clean labels."
            if clean_pct >= 90
            else "Near-clean patches do not exceed 90% of the dataset."
        ),
        (
            "Unweighted MSE may favor consistently high predictions."
            if clean_pct >= 90
            else "Check estimator predictions before attributing errors to imbalance."
        ),
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nNumerical report saved to: {output_path}")

if __name__ == "__main__":
    # 1)
    # inspect_ground_truth()

    # 2)
    analyze_quality_distribution(split="train", modality="ir")
    analyze_quality_distribution(split="train", modality="rgb")

