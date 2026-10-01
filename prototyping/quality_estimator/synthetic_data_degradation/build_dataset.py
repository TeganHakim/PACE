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
import pickle
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from degradation import generate_mask, apply_degradation

from lighting import compute_rgb_baseline_quality

from quality_labels import create_ir_baseline, mask_to_coverage, compute_final_quality

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

# Save a checkpoint every N successfully processed samples.
CHECKPOINT_EVERY = 100


# ------------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------------

DATASET_ROOT = Path("VisDrone-DroneVehicle")

OUTPUT_ROOT = Path("processed_dataset")

METADATA_PATH = OUTPUT_ROOT / "metadata.csv"

CHECKPOINT_PATH = OUTPUT_ROOT / "build_checkpoint.pkl"

BUILD_COMPLETE_PATH = OUTPUT_ROOT / "BUILD_COMPLETE.txt"


SPLIT_CONFIGS = {
    "train": {
        "rgb": DATASET_ROOT / "train" / "trainimg",
        "ir": DATASET_ROOT / "train" / "trainimgr",
        "rgb_xml": DATASET_ROOT / "train" / "trainlabel",
        "ir_xml": DATASET_ROOT / "train" / "trainlabelr",
    },
    "val": {
        "rgb": DATASET_ROOT / "val" / "valimg",
        "ir": DATASET_ROOT / "val" / "valimgr",
        "rgb_xml": DATASET_ROOT / "val" / "vallabel",
        "ir_xml": DATASET_ROOT / "val" / "vallabelr",
    },
    "test": {
        "rgb": DATASET_ROOT / "test" / "testimg",
        "ir": DATASET_ROOT / "test" / "testimgr",
        "rgb_xml": DATASET_ROOT / "test" / "testlabel",
        "ir_xml": DATASET_ROOT / "test" / "testlabelr",
    },
}


# This limit is applied PER SPLIT.
#
# Example:
#     MAX_SAMPLES = 2
#
# gives:
#     2 train
#     2 val
#     2 test
#
# Use None for the complete dataset.
MAX_SAMPLES = None


# ------------------------------------------------------------------
# Metadata fields
# ------------------------------------------------------------------

METADATA_FIELDS = [
    "pair_id",
    "split",
    "rgb_path",
    "ir_path",
    "rgb_xml",
    "ir_xml",
    "rgb_quality",
    "ir_quality",
    "degraded",
    "modality",
    "degradation_type",
    "severity",
    "source_rgb",
    "source_ir",
    "seed",
]


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
# Checkpoint / resume utilities
# ------------------------------------------------------------------


def save_checkpoint(split, next_index, completed_count):
    """
    Save enough state to resume the build deterministically.

    next_index is the zero-based index of the NEXT sample that should
    be processed within the current split.

    Random states are saved so that a resumed run produces exactly the
    same future random decisions as an uninterrupted run.
    """

    checkpoint = {
        "split": split,
        "next_index": next_index,
        "completed_count": completed_count,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }

    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")

    with open(temporary_path, "wb") as checkpoint_file:

        pickle.dump(checkpoint, checkpoint_file)

    temporary_path.replace(CHECKPOINT_PATH)


def load_checkpoint():
    """
    Load an existing build checkpoint.

    Returns None when no checkpoint exists.
    """

    if not CHECKPOINT_PATH.exists():
        return None

    with open(CHECKPOINT_PATH, "rb") as checkpoint_file:

        checkpoint = pickle.load(checkpoint_file)

    return checkpoint


def restore_random_state(checkpoint):
    """
    Restore Python and NumPy RNG states from a checkpoint.
    """

    random.setstate(checkpoint["python_random_state"])

    np.random.set_state(checkpoint["numpy_random_state"])


def clear_checkpoint():
    """
    Remove active checkpoint after a successful full build.
    """

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


# ------------------------------------------------------------------
# Metadata utilities
# ------------------------------------------------------------------


def initialize_metadata_file():
    """
    Create a fresh metadata CSV and write its header.
    """

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "w", newline="") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=METADATA_FIELDS)

        writer.writeheader()


def append_metadata_row(row):
    """
    Append one successfully completed sample to metadata.csv.

    The row is written immediately so a crash does not discard all
    previously generated metadata.
    """

    with open(METADATA_PATH, "a", newline="") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=METADATA_FIELDS)

        writer.writerow(row)


def read_metadata():
    """
    Read existing metadata rows.
    """

    if not METADATA_PATH.exists():
        return []

    with open(METADATA_PATH, "r", newline="") as csv_file:

        reader = csv.DictReader(csv_file)

        return list(reader)


def metadata_key(row):
    """
    Unique sample key across train/val/test.
    """

    return (row["split"], row["pair_id"])


def trim_metadata_to_checkpoint(checkpoint):
    """
    Remove metadata rows written after the most recent checkpoint.

    A crash may occur after one or more metadata rows were written but
    before the next checkpoint was saved. Those samples will be replayed
    from the checkpoint, so their old metadata rows must first be removed
    to avoid duplicates.
    """

    if not METADATA_PATH.exists():
        return

    rows = read_metadata()

    keep_count = checkpoint["completed_count"]

    if len(rows) <= keep_count:
        return

    rows = rows[:keep_count]

    with open(METADATA_PATH, "w", newline="") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=METADATA_FIELDS)

        writer.writeheader()

        writer.writerows(rows)


# ------------------------------------------------------------------
# DroneVehicle crop
# ------------------------------------------------------------------


def apply_crop(image, border_pixels=BORDER_PIXELS):
    """
    Remove the fixed DroneVehicle border padding.

    Standard dataset images:

        original: 840 x 712
        cropped:  640 x 512
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    height, width = image.shape[:2]

    if height <= 2 * border_pixels or width <= 2 * border_pixels:
        raise ValueError(
            f"Image {image.shape} is too small "
            f"to remove {border_pixels}px "
            f"from every side."
        )

    return image[
        border_pixels : height - border_pixels, border_pixels : width - border_pixels
    ]


# ------------------------------------------------------------------
# XML adjustment
# ------------------------------------------------------------------


def adjust_xml_for_crop(
    xml_input_path,
    xml_output_path,
    original_width,
    original_height,
    border_pixels=BORDER_PIXELS,
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

    tree = ET.parse(xml_input_path)

    root = tree.getroot()

    new_width = original_width - 2 * border_pixels

    new_height = original_height - 2 * border_pixels

    if new_width <= 0 or new_height <= 0:
        raise ValueError(f"Invalid cropped image size: " f"{new_width} x {new_height}")

    # --------------------------------------------------------------
    # Update stored image dimensions, if present
    # --------------------------------------------------------------

    size_node = root.find("size")

    if size_node is not None:

        width_node = size_node.find("width")

        height_node = size_node.find("height")

        if width_node is not None:
            width_node.text = str(new_width)

        if height_node is not None:
            height_node.text = str(new_height)

    # --------------------------------------------------------------
    # Crop boundaries in ORIGINAL coordinates
    # --------------------------------------------------------------

    crop_left = border_pixels
    crop_top = border_pixels

    crop_right = original_width - border_pixels

    crop_bottom = original_height - border_pixels

    # --------------------------------------------------------------
    # Adjust every annotated object
    # --------------------------------------------------------------

    for obj in list(root.findall("object")):

        polygon = obj.find("polygon")

        if polygon is None:
            continue

        points = []
        valid_polygon = True

        for i in range(1, 5):

            x_node = polygon.find(f"x{i}")

            y_node = polygon.find(f"y{i}")

            if (
                x_node is None
                or y_node is None
                or x_node.text is None
                or y_node.text is None
            ):
                valid_polygon = False
                break

            x = float(x_node.text)

            y = float(y_node.text)

            points.append((x, y))

        if not valid_polygon:
            continue

        xs = [point[0] for point in points]

        ys = [point[1] for point in points]

        # ----------------------------------------------------------
        # Remove objects entirely outside retained image
        # ----------------------------------------------------------

        if (
            max(xs) < crop_left
            or min(xs) >= crop_right
            or max(ys) < crop_top
            or min(ys) >= crop_bottom
        ):

            root.remove(obj)

            continue

        # ----------------------------------------------------------
        # Shift and clip coordinates
        # ----------------------------------------------------------

        for i in range(1, 5):

            x_node = polygon.find(f"x{i}")

            y_node = polygon.find(f"y{i}")

            x = float(x_node.text) - border_pixels

            y = float(y_node.text) - border_pixels

            x = np.clip(x, 0, new_width - 1)

            y = np.clip(y, 0, new_height - 1)

            x_node.text = str(int(round(x)))

            y_node.text = str(int(round(y)))

    tree.write(xml_output_path, encoding="utf-8", xml_declaration=True)


# ------------------------------------------------------------------
# Output directories
# ------------------------------------------------------------------


def create_output_directories(output_root):
    """
    Create generated dataset directories.
    """

    output_root = Path(output_root)

    directories = {
        "rgb": output_root / "rgb",
        "ir": output_root / "ir",
        "rgb_annotations": output_root / "rgb_annotations",
        "ir_annotations": output_root / "ir_annotations",
        "quality_maps": output_root / "quality_maps",
    }

    for directory in directories.values():

        directory.mkdir(parents=True, exist_ok=True)

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

    return np.count_nonzero(mask) / mask.size


def is_meaningful_degradation(mask, min_fraction=MIN_DEGRADED_FRACTION):
    """
    Determine whether a degradation mask affects enough pixels to
    count as an actual degraded example.
    """

    return mask_fraction(mask) >= min_fraction


# ------------------------------------------------------------------
# Process one synchronized RGB / IR pair
# ------------------------------------------------------------------


def process_pair(
    pair_id, rgb_path, ir_path, rgb_xml_path, ir_xml_path, output_dirs, split
):
    """
    Process one synchronized RGB/IR pair.

    Returns
    -------
    dict
        One metadata row describing the generated sample.
    """

    # --------------------------------------------------------------
    # Load original images
    # --------------------------------------------------------------

    rgb_original = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    ir_original = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)

    if rgb_original is None:
        raise ValueError(f"Could not load RGB image: " f"{rgb_path}")

    if ir_original is None:
        raise ValueError(f"Could not load IR image: " f"{ir_path}")

    rgb_original_height, rgb_original_width = rgb_original.shape[:2]

    ir_original_height, ir_original_width = ir_original.shape[:2]

    # --------------------------------------------------------------
    # Crop fixed DroneVehicle padding
    # --------------------------------------------------------------

    rgb = apply_crop(rgb_original)

    ir = apply_crop(ir_original)

    # RGB and IR should remain spatially synchronized.
    if rgb.shape[:2] != ir.shape[:2]:
        raise ValueError(
            f"RGB/IR spatial mismatch for "
            f"{split}/{pair_id}: "
            f"RGB={rgb.shape[:2]}, "
            f"IR={ir.shape[:2]}"
        )

    height, width = rgb.shape[:2]

    # --------------------------------------------------------------
    # Output paths
    # --------------------------------------------------------------

    rgb_output_path = output_dirs["rgb"] / f"{pair_id}.jpg"

    ir_output_path = output_dirs["ir"] / f"{pair_id}.jpg"

    rgb_xml_output_path = output_dirs["rgb_annotations"] / f"{pair_id}.xml"

    ir_xml_output_path = output_dirs["ir_annotations"] / f"{pair_id}.xml"

    rgb_quality_output_path = output_dirs["quality_maps"] / f"{pair_id}_rgb.npy"

    ir_quality_output_path = output_dirs["quality_maps"] / f"{pair_id}_ir.npy"

    # --------------------------------------------------------------
    # Adjust XML annotations to cropped coordinate system
    # --------------------------------------------------------------

    adjust_xml_for_crop(
        xml_input_path=rgb_xml_path,
        xml_output_path=(rgb_xml_output_path),
        original_width=(rgb_original_width),
        original_height=(rgb_original_height),
    )

    adjust_xml_for_crop(
        xml_input_path=ir_xml_path,
        xml_output_path=(ir_xml_output_path),
        original_width=(ir_original_width),
        original_height=(ir_original_height),
    )

    # --------------------------------------------------------------
    # Compute baseline quality
    # --------------------------------------------------------------

    rgb_quality, sufficient_light_mask, _ = compute_rgb_baseline_quality(rgb)

    ir_quality = create_ir_baseline()

    # --------------------------------------------------------------
    # Default state = clean
    # --------------------------------------------------------------

    degraded = False
    modality = "none"
    degradation_type = "none"
    severity = 0.0

    rgb_final = rgb.copy()
    ir_final = ir.copy()

    # --------------------------------------------------------------
    # Decide whether to ATTEMPT degradation
    # --------------------------------------------------------------

    attempt_degradation = random.random() < DEGRADATION_PROBABILITY

    if attempt_degradation:

        candidate_modality = random.choice(["rgb", "ir"])

        candidate_type = random.choice(DEGRADATION_TYPES)

        candidate_severity = random.uniform(0.2, 1.0)

        # ----------------------------------------------------------
        # Generate candidate spatial mask
        # ----------------------------------------------------------

        candidate_mask = generate_mask(height, width)

        # ----------------------------------------------------------
        # RGB degradation
        #
        # RGB degradation should only be applied where RGB already
        # contains usable visible-light information.
        # ----------------------------------------------------------

        if candidate_modality == "rgb":

            # Only degrade sufficiently illuminated RGB pixels.
            effective_mask = (candidate_mask * sufficient_light_mask).astype(np.uint8)

            effective_fraction = mask_fraction(effective_mask)

            if effective_fraction >= MIN_DEGRADED_FRACTION:

                rgb_final, degradation_metadata = apply_degradation(
                    rgb,
                    degradation_type=(candidate_type),
                    severity=(candidate_severity),
                    mask=(effective_mask),
                )

                # ----------------------------------------------
                # Convert degradation mask to 8x8 coverage
                # ----------------------------------------------

                rgb_coverage = mask_to_coverage(effective_mask)

                rgb_quality = compute_final_quality(
                    rgb_quality, rgb_coverage, candidate_severity
                )

                degraded = True

                modality = "rgb"

                degradation_type = degradation_metadata["type"]

                severity = float(degradation_metadata["severity"])

            else:

                print(
                    f"  RGB degradation skipped "
                    f"for {pair_id}: "
                    f"effective mask covers "
                    f"{effective_fraction:.2%} "
                    f"of image "
                    f"(< "
                    f"{MIN_DEGRADED_FRACTION:.2%})"
                )

        # ----------------------------------------------------------
        # IR degradation
        # ----------------------------------------------------------

        else:

            effective_mask = candidate_mask

            effective_fraction = mask_fraction(effective_mask)

            if effective_fraction >= MIN_DEGRADED_FRACTION:

                ir_final, degradation_metadata = apply_degradation(
                    ir,
                    degradation_type=(candidate_type),
                    severity=(candidate_severity),
                    mask=(effective_mask),
                )

                ir_coverage = mask_to_coverage(effective_mask)

                ir_quality = compute_final_quality(
                    ir_quality, ir_coverage, candidate_severity
                )

                degraded = True

                modality = "ir"

                degradation_type = degradation_metadata["type"]

                severity = float(degradation_metadata["severity"])

            else:

                print(
                    f"  IR degradation skipped "
                    f"for {pair_id}: "
                    f"mask covers "
                    f"{effective_fraction:.2%} "
                    f"of image "
                    f"(< "
                    f"{MIN_DEGRADED_FRACTION:.2%})"
                )

    # --------------------------------------------------------------
    # Final quality-map validation
    # --------------------------------------------------------------

    rgb_quality = np.clip(rgb_quality, 0.0, 1.0).astype(np.float32)

    ir_quality = np.clip(ir_quality, 0.0, 1.0).astype(np.float32)

    if rgb_quality.shape != (8, 8):
        raise RuntimeError(
            f"RGB quality map for "
            f"{split}/{pair_id} has "
            f"unexpected shape "
            f"{rgb_quality.shape}."
        )

    if ir_quality.shape != (8, 8):
        raise RuntimeError(
            f"IR quality map for "
            f"{split}/{pair_id} has "
            f"unexpected shape "
            f"{ir_quality.shape}."
        )

    if not np.all(np.isfinite(rgb_quality)):
        raise RuntimeError(
            f"RGB quality map for " f"{split}/{pair_id} " f"contains NaN/Inf."
        )

    if not np.all(np.isfinite(ir_quality)):
        raise RuntimeError(
            f"IR quality map for " f"{split}/{pair_id} " f"contains NaN/Inf."
        )

    # --------------------------------------------------------------
    # Save processed images
    # --------------------------------------------------------------

    rgb_write_success = cv2.imwrite(str(rgb_output_path), rgb_final)

    ir_write_success = cv2.imwrite(str(ir_output_path), ir_final)

    if not rgb_write_success:
        raise IOError(f"Failed to write RGB image: " f"{rgb_output_path}")

    if not ir_write_success:
        raise IOError(f"Failed to write IR image: " f"{ir_output_path}")

    # --------------------------------------------------------------
    # Save 8x8 quality maps
    # --------------------------------------------------------------

    np.save(rgb_quality_output_path, rgb_quality)

    np.save(ir_quality_output_path, ir_quality)

    # --------------------------------------------------------------
    # Construct metadata
    # --------------------------------------------------------------

    metadata = {
        "pair_id": pair_id,
        "split": split,
        "rgb_path": str(rgb_output_path),
        "ir_path": str(ir_output_path),
        "rgb_xml": str(rgb_xml_output_path),
        "ir_xml": str(ir_xml_output_path),
        "rgb_quality": str(rgb_quality_output_path),
        "ir_quality": str(ir_quality_output_path),
        "degraded": degraded,
        "modality": modality,
        "degradation_type": degradation_type,
        "severity": severity,
        "source_rgb": str(rgb_path),
        "source_ir": str(ir_path),
        "seed": RANDOM_SEED,
    }

    return metadata


# ------------------------------------------------------------------
# Source-dataset validation
# ------------------------------------------------------------------


def validate_source_directories():
    """
    Verify all configured source directories exist before beginning
    the long dataset-generation run.
    """

    for split, config in SPLIT_CONFIGS.items():

        for source_type, path in config.items():

            if not path.exists():

                raise FileNotFoundError(
                    f"Missing {split} " f"{source_type} " f"directory: {path}"
                )


# ------------------------------------------------------------------
# Get synchronized RGB candidates for a split
# ------------------------------------------------------------------


def get_rgb_paths(split_config):
    """
    Return sorted RGB image paths for one split.
    """

    rgb_paths = sorted(split_config["rgb"].glob("*.jpg"))

    if MAX_SAMPLES is not None:

        rgb_paths = rgb_paths[:MAX_SAMPLES]

    return rgb_paths


# ------------------------------------------------------------------
# Verify a pair has all required inputs
# ------------------------------------------------------------------


def get_pair_paths(pair_id, split_config):
    """
    Construct synchronized IR/XML paths for one pair.
    """

    ir_path = split_config["ir"] / f"{pair_id}.jpg"

    rgb_xml_path = split_config["rgb_xml"] / f"{pair_id}.xml"

    ir_xml_path = split_config["ir_xml"] / f"{pair_id}.xml"

    return (ir_path, rgb_xml_path, ir_xml_path)


def find_missing_files(ir_path, rgb_xml_path, ir_xml_path):
    """
    Return any missing synchronized source files.
    """

    required_files = [
        ir_path,
        rgb_xml_path,
        ir_xml_path,
    ]

    return [path for path in required_files if not path.exists()]


# ------------------------------------------------------------------
# Build-state helpers
# ------------------------------------------------------------------


def split_index(split):
    """
    Return ordering index for train -> val -> test.
    """

    order = ["train", "val", "test"]

    return order.index(split)


def should_skip_split(split, resume_split):
    """
    Determine whether an entire split was completed before the
    checkpoint.
    """

    if resume_split is None:
        return False

    return split_index(split) < split_index(resume_split)


# ------------------------------------------------------------------
# Count existing metadata by split
# ------------------------------------------------------------------


def metadata_split_counts():
    """
    Count metadata rows currently written for each split.
    """

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for row in read_metadata():

        split = row.get("split")

        if split in counts:
            counts[split] += 1

    return counts


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    """
    Build the complete processed dataset.

    The build is resumable. On a fresh run:

        - seed Python and NumPy with RANDOM_SEED
        - create a new metadata.csv
        - begin at train sample 0

    On a resumed run:

        - load the most recent checkpoint
        - trim metadata rows written after that checkpoint
        - restore Python and NumPy RNG states
        - continue from the checkpoint's next sample
    """

    # --------------------------------------------------------------
    # Validate source dataset
    # --------------------------------------------------------------

    validate_source_directories()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Determine fresh vs resumed build
    # --------------------------------------------------------------

    checkpoint = load_checkpoint()

    if checkpoint is None:

        print("Starting fresh dataset build.")

        set_random_seed(RANDOM_SEED)

        initialize_metadata_file()

        resume_split = None
        resume_index = 0
        completed_count = 0

        if BUILD_COMPLETE_PATH.exists():
            BUILD_COMPLETE_PATH.unlink()

    else:

        print("Checkpoint found.")

        print("Resuming dataset build...")

        resume_split = checkpoint["split"]

        resume_index = checkpoint["next_index"]

        completed_count = checkpoint["completed_count"]

        # Remove metadata that may have been written between the
        # checkpoint and the crash.
        trim_metadata_to_checkpoint(checkpoint)

        # Restore exact RNG state at checkpoint.
        restore_random_state(checkpoint)

        print(f"Resume split: " f"{resume_split}")

        print(f"Resume index: " f"{resume_index}")

        print(f"Completed at checkpoint: " f"{completed_count}")

    # --------------------------------------------------------------
    # Process train / val / test
    # --------------------------------------------------------------

    for split, config in SPLIT_CONFIGS.items():

        # ----------------------------------------------------------
        # If resuming, earlier splits were already completed.
        # ----------------------------------------------------------

        if should_skip_split(split, resume_split):

            print()
            print(f"Skipping completed split: " f"{split.upper()}")

            continue

        print()
        print("=" * 70)

        print(f"PROCESSING SPLIT: " f"{split.upper()}")

        print("=" * 70)

        # ----------------------------------------------------------
        # Create split-specific output directories
        # ----------------------------------------------------------

        split_output_root = OUTPUT_ROOT / split

        output_dirs = create_output_directories(split_output_root)

        # ----------------------------------------------------------
        # Get candidate RGB paths
        # ----------------------------------------------------------

        rgb_paths = get_rgb_paths(config)

        print(
            f"Processing "
            f"{len(rgb_paths)} "
            f"synchronized RGB/IR pairs "
            f"from {split}..."
        )

        # ----------------------------------------------------------
        # Determine starting index for this split
        # ----------------------------------------------------------

        if checkpoint is not None and split == resume_split:

            start_index = resume_index

        else:

            start_index = 0

        # ----------------------------------------------------------
        # Process synchronized pairs
        # ----------------------------------------------------------

        for zero_index in range(start_index, len(rgb_paths)):

            rgb_path = rgb_paths[zero_index]

            pair_id = rgb_path.stem

            ir_path, rgb_xml_path, ir_xml_path = get_pair_paths(pair_id, config)

            missing_files = find_missing_files(ir_path, rgb_xml_path, ir_xml_path)

            # ------------------------------------------------------
            # Missing source data
            # ------------------------------------------------------

            if missing_files:

                print(
                    f"Skipping pair "
                    f"{split}/{pair_id}: "
                    f"missing files "
                    f"{missing_files}"
                )

                # IMPORTANT:
                #
                # No random numbers were consumed for this sample,
                # and no metadata row was created.
                #
                # Save the next index if this happens to coincide
                # with a checkpoint boundary.
                continue

            display_index = zero_index + 1

            print(
                f"[{split} "
                f"{display_index}/"
                f"{len(rgb_paths)}] "
                f"Processing pair "
                f"{pair_id}"
            )

            # ------------------------------------------------------
            # Generate one sample
            # ------------------------------------------------------

            metadata = process_pair(
                pair_id=pair_id,
                rgb_path=rgb_path,
                ir_path=ir_path,
                rgb_xml_path=(rgb_xml_path),
                ir_xml_path=(ir_xml_path),
                output_dirs=(output_dirs),
                split=split,
            )

            # ------------------------------------------------------
            # Persist metadata immediately
            # ------------------------------------------------------

            append_metadata_row(metadata)

            completed_count += 1

            # ------------------------------------------------------
            # Periodic checkpoint
            #
            # Save RNG state AFTER processing the current sample.
            # Therefore next_index points to the NEXT sample.
            # ------------------------------------------------------

            if completed_count % CHECKPOINT_EVERY == 0:

                save_checkpoint(
                    split=split,
                    next_index=(zero_index + 1),
                    completed_count=(completed_count),
                )

                print(
                    f"  Checkpoint saved "
                    f"after "
                    f"{completed_count} "
                    f"total pairs."
                )

        # ----------------------------------------------------------
        # Split completed
        #
        # Save a checkpoint at the boundary even if the total count
        # is not an exact multiple of CHECKPOINT_EVERY.
        #
        # next_index=len(rgb_paths) means the current split is done.
        # ----------------------------------------------------------

        save_checkpoint(
            split=split, next_index=(len(rgb_paths)), completed_count=(completed_count)
        )

        # ----------------------------------------------------------
        # Split summary from persisted metadata
        # ----------------------------------------------------------

        current_rows = read_metadata()

        split_rows = [row for row in current_rows if row["split"] == split]

        split_degraded = sum(
            str(row["degraded"]).lower() == "true" for row in split_rows
        )

        split_clean = len(split_rows) - split_degraded

        print()
        print(f"{split.upper()} complete.")

        print(f"Processed pairs: " f"{len(split_rows)}")

        print(f"Actually degraded: " f"{split_degraded}")

        print(f"Clean: " f"{split_clean}")

        # ----------------------------------------------------------
        # Important resume-boundary handling
        #
        # Once this split completes normally, subsequent splits must
        # start from index 0.
        # ----------------------------------------------------------

        if checkpoint is not None and split == resume_split:
            checkpoint = None
            resume_split = None
            resume_index = 0

        # ----------------------------------------------------------
        # Save boundary checkpoint for NEXT split
        #
        # This makes a crash immediately after a completed split
        # resumable without replaying that split.
        # ----------------------------------------------------------

        split_names = list(SPLIT_CONFIGS.keys())

        current_split_position = split_names.index(split)

        if current_split_position < len(split_names) - 1:

            next_split = split_names[current_split_position + 1]

            save_checkpoint(
                split=next_split, next_index=0, completed_count=(completed_count)
            )

    # --------------------------------------------------------------
    # Entire dataset completed successfully
    # --------------------------------------------------------------

    all_rows = read_metadata()

    degraded_count = sum(str(row["degraded"]).lower() == "true" for row in all_rows)

    clean_count = len(all_rows) - degraded_count

    # --------------------------------------------------------------
    # Count samples by split
    # --------------------------------------------------------------

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for row in all_rows:

        split = row.get("split")

        if split in split_counts:
            split_counts[split] += 1

    # --------------------------------------------------------------
    # Final integrity check
    # --------------------------------------------------------------

    metadata_keys = [(row["split"], row["pair_id"]) for row in all_rows]

    unique_metadata_keys = set(metadata_keys)

    if len(metadata_keys) != len(unique_metadata_keys):

        raise RuntimeError(
            "Duplicate (split, pair_id) " "entries detected in metadata.csv."
        )

    # --------------------------------------------------------------
    # Write completion marker
    # --------------------------------------------------------------

    with open(BUILD_COMPLETE_PATH, "w") as completion_file:

        completion_file.write(
            "PACE processed dataset " "generation completed " "successfully.\n"
        )

        completion_file.write(f"seed={RANDOM_SEED}\n")

        completion_file.write(f"total_pairs=" f"{len(all_rows)}\n")

        completion_file.write(f"train_pairs=" f"{split_counts['train']}\n")

        completion_file.write(f"val_pairs=" f"{split_counts['val']}\n")

        completion_file.write(f"test_pairs=" f"{split_counts['test']}\n")

        completion_file.write(f"degraded_pairs=" f"{degraded_count}\n")

        completion_file.write(f"clean_pairs=" f"{clean_count}\n")

    # --------------------------------------------------------------
    # Remove active checkpoint ONLY after everything above succeeds
    # --------------------------------------------------------------

    clear_checkpoint()

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print()
    print("=" * 70)

    print("DATASET GENERATION COMPLETE")

    print("=" * 70)

    print(f"Total processed pairs: " f"{len(all_rows)}")

    print(f"  train: " f"{split_counts['train']}")

    print(f"  val: " f"{split_counts['val']}")

    print(f"  test: " f"{split_counts['test']}")

    print(f"Actually degraded: " f"{degraded_count}")

    print(f"Clean: " f"{clean_count}")

    print(f"Output directory: " f"{OUTPUT_ROOT.resolve()}")

    print(f"Metadata: " f"{METADATA_PATH.resolve()}")

    print(f"Completion marker: " f"{BUILD_COMPLETE_PATH.resolve()}")

    print("Active checkpoint removed.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
