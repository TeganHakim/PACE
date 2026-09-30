"""
Build the synthetic PACE quality-estimation dataset.

For every synchronized RGB/IR pair:

1. Load RGB, IR, and XML annotations.
2. Remove the fixed 100-pixel DroneVehicle padding.
3. Shift/crop XML annotation coordinates accordingly.
4. Compute baseline quality.
5. Randomly choose whether to attempt degradation.
6. Apply degradation only when the effective degradation region is
   sufficiently large.
7. Construct 8x8 quality labels.
8. Save cropped images, adjusted annotations, labels, and metadata.

Important:
    "degraded=True" means a meaningful degradation was actually applied,
    not merely that the sample was selected for a degradation attempt.
"""

import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from degradation import (
    generate_mask,
    apply_degradation
)

from lighting import (
    compute_rgb_baseline_quality
)

from quality_labels import (
    create_ir_baseline,
    mask_to_coverage,
    compute_final_quality
)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

RANDOM_SEED = 42

DEGRADATION_PROBABILITY = 0.50

# An attempted degradation must affect at least this fraction of the
# cropped image before it is considered an actual degradation.
MIN_DEGRADED_FRACTION = 0.01

BORDER_PIXELS = 100

DEGRADATION_TYPES = [
    "water_blur",
    "motion_blur",
    "sensor_noise",
    "saturation",
]


# ------------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------------

DATASET_ROOT = Path(
    "VisDrone-DroneVehicle"
)

OUTPUT_ROOT = Path(
    "processed_dataset"
)

RGB_DIR = (
    DATASET_ROOT /
    "train" /
    "trainimg"
)

IR_DIR = (
    DATASET_ROOT /
    "train" /
    "trainimgr"
)

RGB_XML_DIR = (
    DATASET_ROOT /
    "train" /
    "trainlabel"
)

IR_XML_DIR = (
    DATASET_ROOT /
    "train" /
    "trainlabelr"
)


# Keep small while validating.
# Set to None for the full training set.
MAX_SAMPLES = 500


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_random_seed(seed):
    """
    Seed Python and NumPy random generators.
    """

    random.seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------------------
# DroneVehicle crop
# ------------------------------------------------------------------

def apply_crop(
    image,
    border_pixels=BORDER_PIXELS
):
    """
    Remove the fixed DroneVehicle border padding.

    Standard dataset images:

        original: 840 x 712
        cropped:  640 x 512
    """

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    height, width = image.shape[:2]

    if (
        height <= 2 * border_pixels or
        width <= 2 * border_pixels
    ):
        raise ValueError(
            f"Image {image.shape} is too small "
            f"to remove {border_pixels}px "
            f"from every side."
        )

    return image[
        border_pixels:height - border_pixels,
        border_pixels:width - border_pixels
    ]


# ------------------------------------------------------------------
# XML adjustment
# ------------------------------------------------------------------

def adjust_xml_for_crop(
    xml_input_path,
    xml_output_path,
    original_width,
    original_height,
    border_pixels=BORDER_PIXELS
):
    """
    Adjust DroneVehicle polygon annotations after cropping.

    Each coordinate is shifted by:

        x_new = x_old - border_pixels
        y_new = y_old - border_pixels

    Objects completely outside the retained region are removed.

    Objects crossing the crop boundary are retained and their
    coordinates are clipped to the new image bounds.
    """

    tree = ET.parse(
        xml_input_path
    )

    root = tree.getroot()

    new_width = (
        original_width -
        2 * border_pixels
    )

    new_height = (
        original_height -
        2 * border_pixels
    )

    if (
        new_width <= 0 or
        new_height <= 0
    ):
        raise ValueError(
            f"Invalid cropped image size: "
            f"{new_width} x {new_height}"
        )

    # --------------------------------------------------------------
    # Update stored image dimensions, if present
    # --------------------------------------------------------------

    size_node = root.find(
        "size"
    )

    if size_node is not None:

        width_node = size_node.find(
            "width"
        )

        height_node = size_node.find(
            "height"
        )

        if width_node is not None:
            width_node.text = str(
                new_width
            )

        if height_node is not None:
            height_node.text = str(
                new_height
            )

    # --------------------------------------------------------------
    # Crop boundaries in ORIGINAL coordinates
    # --------------------------------------------------------------

    crop_left = border_pixels
    crop_top = border_pixels

    crop_right = (
        original_width -
        border_pixels
    )

    crop_bottom = (
        original_height -
        border_pixels
    )

    # --------------------------------------------------------------
    # Adjust every annotated object
    # --------------------------------------------------------------

    for obj in list(
        root.findall("object")
    ):

        polygon = obj.find(
            "polygon"
        )

        if polygon is None:
            continue

        points = []

        valid_polygon = True

        for i in range(1, 5):

            x_node = polygon.find(
                f"x{i}"
            )

            y_node = polygon.find(
                f"y{i}"
            )

            if (
                x_node is None or
                y_node is None or
                x_node.text is None or
                y_node.text is None
            ):
                valid_polygon = False
                break

            x = float(
                x_node.text
            )

            y = float(
                y_node.text
            )

            points.append(
                (x, y)
            )

        if not valid_polygon:
            continue

        xs = [
            point[0]
            for point in points
        ]

        ys = [
            point[1]
            for point in points
        ]

        # ----------------------------------------------------------
        # Remove objects entirely outside retained image
        # ----------------------------------------------------------

        if (
            max(xs) < crop_left or
            min(xs) >= crop_right or
            max(ys) < crop_top or
            min(ys) >= crop_bottom
        ):

            root.remove(
                obj
            )

            continue

        # ----------------------------------------------------------
        # Shift and clip coordinates
        # ----------------------------------------------------------

        for i in range(1, 5):

            x_node = polygon.find(
                f"x{i}"
            )

            y_node = polygon.find(
                f"y{i}"
            )

            x = (
                float(x_node.text) -
                border_pixels
            )

            y = (
                float(y_node.text) -
                border_pixels
            )

            x = np.clip(
                x,
                0,
                new_width - 1
            )

            y = np.clip(
                y,
                0,
                new_height - 1
            )

            x_node.text = str(
                int(round(x))
            )

            y_node.text = str(
                int(round(y))
            )

    tree.write(
        xml_output_path,
        encoding="utf-8",
        xml_declaration=True
    )


# ------------------------------------------------------------------
# Output directories
# ------------------------------------------------------------------

def create_output_directories(
    output_root
):
    """
    Create generated dataset directories.
    """

    output_root = Path(
        output_root
    )

    directories = {
        "rgb":
            output_root / "rgb",

        "ir":
            output_root / "ir",

        "rgb_annotations":
            output_root /
            "rgb_annotations",

        "ir_annotations":
            output_root /
            "ir_annotations",

        "quality_maps":
            output_root /
            "quality_maps",
    }

    for directory in directories.values():

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    return directories


# ------------------------------------------------------------------
# Degradation validation
# ------------------------------------------------------------------

def mask_fraction(mask):
    """
    Return the fraction of image pixels selected by a binary mask.
    """

    if mask.size == 0:
        return 0.0

    return (
        np.count_nonzero(mask) /
        mask.size
    )


def is_meaningful_degradation(
    mask,
    min_fraction=MIN_DEGRADED_FRACTION
):
    """
    Determine whether a degradation mask affects enough pixels to
    count as an actual degraded example.
    """

    return (
        mask_fraction(mask) >=
        min_fraction
    )


# ------------------------------------------------------------------
# Process one synchronized pair
# ------------------------------------------------------------------

def process_pair(
    pair_id,
    rgb_path,
    ir_path,
    rgb_xml_path,
    ir_xml_path,
    output_dirs,
    split
):
    """
    Process one synchronized RGB/IR pair.
    """

    # --------------------------------------------------------------
    # Load original padded images
    # --------------------------------------------------------------

    rgb = cv2.imread(
        str(rgb_path),
        cv2.IMREAD_COLOR
    )

    ir = cv2.imread(
        str(ir_path),
        cv2.IMREAD_UNCHANGED
    )

    if rgb is None:
        raise ValueError(
            f"Could not load RGB image: "
            f"{rgb_path}"
        )

    if ir is None:
        raise ValueError(
            f"Could not load IR image: "
            f"{ir_path}"
        )

    # --------------------------------------------------------------
    # Record original dimensions for XML adjustment
    # --------------------------------------------------------------

    rgb_original_height, rgb_original_width = (
        rgb.shape[:2]
    )

    ir_original_height, ir_original_width = (
        ir.shape[:2]
    )

    # --------------------------------------------------------------
    # Remove fixed padding BEFORE all quality/degradation work
    # --------------------------------------------------------------

    rgb = apply_crop(
        rgb
    )

    ir = apply_crop(
        ir
    )

    # --------------------------------------------------------------
    # Crop sanity checks
    # --------------------------------------------------------------

    expected_rgb_shape = (
        rgb_original_height -
        2 * BORDER_PIXELS,
        rgb_original_width -
        2 * BORDER_PIXELS
    )

    expected_ir_shape = (
        ir_original_height -
        2 * BORDER_PIXELS,
        ir_original_width -
        2 * BORDER_PIXELS
    )

    if rgb.shape[:2] != expected_rgb_shape:

        raise ValueError(
            f"Unexpected RGB crop for "
            f"{pair_id}: "
            f"{rgb.shape[:2]} vs "
            f"{expected_rgb_shape}"
        )

    if ir.shape[:2] != expected_ir_shape:

        raise ValueError(
            f"Unexpected IR crop for "
            f"{pair_id}: "
            f"{ir.shape[:2]} vs "
            f"{expected_ir_shape}"
        )

    # --------------------------------------------------------------
    # Baseline quality
    # --------------------------------------------------------------

    (
        rgb_quality,
        sufficient_light_mask,
        _
    ) = compute_rgb_baseline_quality(
        rgb
    )

    # Clean IR is treated as reference quality.
    ir_quality = (
        create_ir_baseline()
    )

    # --------------------------------------------------------------
    # Start with clean copies and CLEAN metadata
    # --------------------------------------------------------------

    output_rgb = rgb.copy()
    output_ir = ir.copy()

    degraded = False
    degraded_modality = "none"
    degradation_type = "none"
    severity = 0.0

    # --------------------------------------------------------------
    # Decide whether to ATTEMPT a degradation
    # --------------------------------------------------------------

    attempt_degradation = (
        random.random() <
        DEGRADATION_PROBABILITY
    )

    if attempt_degradation:

        selected_modality = (
            random.choice([
                "rgb",
                "ir"
            ])
        )

        selected_type = (
            random.choice(
                DEGRADATION_TYPES
            )
        )

        selected_severity = (
            random.uniform(
                0.2,
                1.0
            )
        )

        # ----------------------------------------------------------
        # RGB degradation attempt
        # ----------------------------------------------------------

        if selected_modality == "rgb":

            random_mask = (
                generate_mask(
                    rgb.shape[0],
                    rgb.shape[1]
                )
            )

            # RGB degradation is only meaningful where sufficient
            # visible-light information exists.
            effective_mask = (
                random_mask *
                sufficient_light_mask
            ).astype(np.uint8)

            effective_fraction = (
                mask_fraction(
                    effective_mask
                )
            )

            # ------------------------------------------------------
            # Only apply + record degradation if enough pixels survive
            # the lighting constraint.
            # ------------------------------------------------------

            if is_meaningful_degradation(
                effective_mask
            ):

                degraded = True
                degraded_modality = "rgb"
                degradation_type = selected_type
                severity = selected_severity

                output_rgb, _ = (
                    apply_degradation(
                        rgb,
                        degradation_type=(
                            degradation_type
                        ),
                        severity=severity,
                        mask=effective_mask
                    )
                )

                coverage = (
                    mask_to_coverage(
                        effective_mask
                    )
                )

                rgb_quality = (
                    compute_final_quality(
                        rgb_quality,
                        coverage,
                        severity
                    )
                )

            else:

                print(
                    f"  RGB degradation skipped for "
                    f"{pair_id}: effective mask covers "
                    f"{effective_fraction:.2%} of image "
                    f"(< {MIN_DEGRADED_FRACTION:.2%})"
                )

        # ----------------------------------------------------------
        # IR degradation attempt
        # ----------------------------------------------------------

        else:

            ir_mask = (
                generate_mask(
                    ir.shape[0],
                    ir.shape[1]
                )
            )

            ir_fraction = (
                mask_fraction(
                    ir_mask
                )
            )

            if is_meaningful_degradation(
                ir_mask
            ):

                degraded = True
                degraded_modality = "ir"
                degradation_type = selected_type
                severity = selected_severity

                output_ir, _ = (
                    apply_degradation(
                        ir,
                        degradation_type=(
                            degradation_type
                        ),
                        severity=severity,
                        mask=ir_mask
                    )
                )

                coverage = (
                    mask_to_coverage(
                        ir_mask
                    )
                )

                ir_quality = (
                    compute_final_quality(
                        ir_quality,
                        coverage,
                        severity
                    )
                )

            else:

                print(
                    f"  IR degradation skipped for "
                    f"{pair_id}: mask covers "
                    f"{ir_fraction:.2%} of image "
                    f"(< {MIN_DEGRADED_FRACTION:.2%})"
                )

    # --------------------------------------------------------------
    # Ensure dimensions never change
    # --------------------------------------------------------------

    assert (
        output_rgb.shape ==
        rgb.shape
    ), (
        f"RGB shape changed for "
        f"{pair_id}: "
        f"{rgb.shape} -> "
        f"{output_rgb.shape}"
    )

    assert (
        output_ir.shape ==
        ir.shape
    ), (
        f"IR shape changed for "
        f"{pair_id}: "
        f"{ir.shape} -> "
        f"{output_ir.shape}"
    )

    # --------------------------------------------------------------
    # Output paths
    # --------------------------------------------------------------

    rgb_output_path = (
        output_dirs["rgb"] /
        f"{pair_id}.jpg"
    )

    ir_output_path = (
        output_dirs["ir"] /
        f"{pair_id}.jpg"
    )

    rgb_xml_output = (
        output_dirs[
            "rgb_annotations"
        ] /
        f"{pair_id}.xml"
    )

    ir_xml_output = (
        output_dirs[
            "ir_annotations"
        ] /
        f"{pair_id}.xml"
    )

    rgb_quality_path = (
        output_dirs[
            "quality_maps"
        ] /
        f"{pair_id}_rgb.npy"
    )

    ir_quality_path = (
        output_dirs[
            "quality_maps"
        ] /
        f"{pair_id}_ir.npy"
    )

    # --------------------------------------------------------------
    # Save images
    # --------------------------------------------------------------

    rgb_write_success = (
        cv2.imwrite(
            str(rgb_output_path),
            output_rgb
        )
    )

    ir_write_success = (
        cv2.imwrite(
            str(ir_output_path),
            output_ir
        )
    )

    if not rgb_write_success:

        raise IOError(
            f"Failed to save RGB image: "
            f"{rgb_output_path}"
        )

    if not ir_write_success:

        raise IOError(
            f"Failed to save IR image: "
            f"{ir_output_path}"
        )

    # --------------------------------------------------------------
    # Save 8x8 quality maps
    # --------------------------------------------------------------

    np.save(
        rgb_quality_path,
        rgb_quality
    )

    np.save(
        ir_quality_path,
        ir_quality
    )

    # --------------------------------------------------------------
    # Save adjusted XML annotations
    # --------------------------------------------------------------

    adjust_xml_for_crop(
        rgb_xml_path,
        rgb_xml_output,
        rgb_original_width,
        rgb_original_height
    )

    adjust_xml_for_crop(
        ir_xml_path,
        ir_xml_output,
        ir_original_width,
        ir_original_height
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    return {
        "pair_id":
            pair_id,

        "split":
            split,

        "rgb_path":
            str(rgb_output_path),

        "ir_path":
            str(ir_output_path),

        "rgb_xml":
            str(rgb_xml_output),

        "ir_xml":
            str(ir_xml_output),

        "rgb_quality":
            str(rgb_quality_path),

        "ir_quality":
            str(ir_quality_path),

        "degraded":
            degraded,

        "modality":
            degraded_modality,

        "degradation_type":
            degradation_type,

        "severity":
            severity,

        "source_rgb":
            str(rgb_path),

        "source_ir":
            str(ir_path),

        "seed":
            RANDOM_SEED,
    }


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------

def write_metadata(
    rows,
    output_path
):
    """
    Write metadata.csv.
    """

    if not rows:

        raise ValueError(
            "No metadata rows were generated."
        )

    with open(
        output_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    set_random_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------------
    # Verify source directories
    # --------------------------------------------------------------

    source_dirs = [
        RGB_DIR,
        IR_DIR,
        RGB_XML_DIR,
        IR_XML_DIR,
    ]

    for directory in source_dirs:

        if not directory.exists():

            raise FileNotFoundError(
                f"Required dataset "
                f"directory not found: "
                f"{directory}"
            )

    # --------------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------------

    output_dirs = (
        create_output_directories(
            OUTPUT_ROOT
        )
    )

    # --------------------------------------------------------------
    # Find RGB images
    # --------------------------------------------------------------

    rgb_paths = sorted(
        RGB_DIR.glob(
            "*.jpg"
        )
    )

    if MAX_SAMPLES is not None:

        rgb_paths = (
            rgb_paths[
                :MAX_SAMPLES
            ]
        )

    print(
        f"Processing "
        f"{len(rgb_paths)} "
        f"synchronized RGB/IR pairs..."
    )

    metadata_rows = []

    # --------------------------------------------------------------
    # Process synchronized pairs
    # --------------------------------------------------------------

    for index, rgb_path in enumerate(
        rgb_paths,
        start=1
    ):

        pair_id = (
            rgb_path.stem
        )

        ir_path = (
            IR_DIR /
            f"{pair_id}.jpg"
        )

        rgb_xml_path = (
            RGB_XML_DIR /
            f"{pair_id}.xml"
        )

        ir_xml_path = (
            IR_XML_DIR /
            f"{pair_id}.xml"
        )

        required_files = [
            ir_path,
            rgb_xml_path,
            ir_xml_path,
        ]

        missing_files = [
            path
            for path in required_files
            if not path.exists()
        ]

        if missing_files:

            print(
                f"Skipping pair "
                f"{pair_id}: "
                f"missing files "
                f"{missing_files}"
            )

            continue

        print(
            f"[{index}/"
            f"{len(rgb_paths)}] "
            f"Processing pair "
            f"{pair_id}"
        )

        metadata = (
            process_pair(
                pair_id=pair_id,
                rgb_path=rgb_path,
                ir_path=ir_path,
                rgb_xml_path=(
                    rgb_xml_path
                ),
                ir_xml_path=(
                    ir_xml_path
                ),
                output_dirs=(
                    output_dirs
                ),
                split="train"
            )
        )

        metadata_rows.append(
            metadata
        )

    # --------------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------------

    metadata_path = (
        OUTPUT_ROOT /
        "metadata.csv"
    )

    write_metadata(
        metadata_rows,
        metadata_path
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    degraded_count = sum(
        row["degraded"]
        for row in metadata_rows
    )

    clean_count = (
        len(metadata_rows) -
        degraded_count
    )

    print()

    print(
        "Dataset generation complete."
    )

    print(
        f"Processed pairs: "
        f"{len(metadata_rows)}"
    )

    print(
        f"Actually degraded: "
        f"{degraded_count}"
    )

    print(
        f"Clean: "
        f"{clean_count}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_ROOT.resolve()}"
    )

    print(
        f"Metadata: "
        f"{metadata_path.resolve()}"
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()