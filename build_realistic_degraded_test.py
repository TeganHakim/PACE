#!/usr/bin/env python3
"""
PACE realistic degradation stress test.

Generates a new test dataset from ORIGINAL DroneVehicle images.

Degradation types:
    1. Severe sensor noise in 3-6 horizontal or vertical bands.
    2. Severe blur in 3-6 irregular, droplet-shaped regions.

Each degradation:
    - Affects approximately 50% of the image.
    - Targets annotated vehicle locations.
    - Applies to only one randomly selected modality.

The other modality remains clean.

Original DroneVehicle paths:
    RGB: VisDrone-DroneVehicle/test/testimg
    IR:  VisDrone-DroneVehicle/test/testimgr

Cropping:
    Remove 100 pixels from every side, exactly as in
    PACE's original build_dataset.py.

Annotations:
    Use processed_dataset/test/*_annotations/*.xml.
    These annotations are already in cropped coordinates.

No quality maps are generated.
"""

import argparse
import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

BORDER_PIXELS = 100


# ============================================================
# REPRODUCIBILITY AND IMAGE LOADING
# ============================================================

def make_rng(seed, pair_id):
    digest = hashlib.sha256(
        f"{seed}:{pair_id}".encode()
    ).digest()

    return np.random.default_rng(
        int.from_bytes(digest[:8], "little")
    )


def index_images(directory):
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing original image directory: {directory}"
        )

    images = {}

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise ValueError(
                    f"Duplicate image ID: {path.stem}"
                )

            images[path.stem] = path

    return images


# ============================================================
# ORIGINAL PACE CROPPING
# ============================================================

def apply_crop(image, border_pixels=BORDER_PIXELS):
    """
    Same fixed-padding crop as original build_dataset.py.

    Original: 840 x 712
    Cropped:  640 x 512
    """

    if image is None:
        raise ValueError(
            "Cannot crop an unreadable image"
        )

    height, width = image.shape[:2]

    if (
        height <= 2 * border_pixels
        or width <= 2 * border_pixels
    ):
        raise ValueError(
            f"Image {width}x{height} too small "
            f"for {border_pixels}px crop"
        )

    return image[
        border_pixels:height - border_pixels,
        border_pixels:width - border_pixels
    ].copy()


# ============================================================
# ANNOTATION LOADING
# ============================================================

def annotation_centers(xml_path, width, height):
    """
    Read already-cropped PACE annotations.

    IMPORTANT:
    Do not subtract 100 again.
    """

    if not xml_path.is_file():
        raise FileNotFoundError(
            f"Missing processed annotation: {xml_path}"
        )

    root = ET.parse(xml_path).getroot()

    xml_width = int(
        root.findtext("size/width") or width
    )

    xml_height = int(
        root.findtext("size/height") or height
    )

    if (xml_width, xml_height) != (width, height):
        raise ValueError(
            f"{xml_path.name}: XML is "
            f"{xml_width}x{xml_height}, "
            f"but cropped image is {width}x{height}"
        )

    centers = []

    for obj in root.findall(".//object"):
        polygon = obj.find("polygon")
        bbox = obj.find("bndbox")

        try:
            if polygon is not None:
                xs = [
                    float(polygon.findtext(f"x{i}"))
                    for i in range(1, 5)
                ]

                ys = [
                    float(polygon.findtext(f"y{i}"))
                    for i in range(1, 5)
                ]

                x = (min(xs) + max(xs)) / 2
                y = (min(ys) + max(ys)) / 2

            elif bbox is not None:
                x1, y1, x2, y2 = (
                    float(bbox.findtext(k))
                    for k in (
                        "xmin", "ymin", "xmax", "ymax"
                    )
                )

                x = (x1 + x2) / 2
                y = (y1 + y2) / 2

            else:
                continue

        except (TypeError, ValueError):
            continue

        centers.append((
            np.clip(x, 0, width - 1),
            np.clip(y, 0, height - 1)
        ))

    return np.asarray(
        centers, dtype=np.float32
    ).reshape(-1, 2)


# ============================================================
# SENSOR NOISE: BAND GENERATION
# ============================================================

def random_band_intervals(dimension, count, rng):
    """
    Generate non-overlapping bands.

    Total band width = exactly 50% of image dimension.

    Bands have variable widths and small gaps.
    """

    total = round(dimension * 0.5)

    minimum_width = min(
        max(2, round(dimension * 0.025)),
        total // count
    )

    if minimum_width < 1:
        raise ValueError(
            "Image too small for requested bands"
        )

    widths = minimum_width + rng.multinomial(
        total - count * minimum_width,
        rng.dirichlet(np.full(count, 2.0))
    )

    min_gap = 2 if count > 1 else 0

    free = (
        dimension
        - total
        - (count - 1) * min_gap
    )

    if free < 0:
        raise ValueError(
            "Not enough space for non-overlapping bands"
        )

    gaps = rng.multinomial(
        free,
        rng.dirichlet(np.ones(count + 1))
    )

    if count > 1:
        gaps[1:count] += min_gap

    intervals = []

    start = int(gaps[0])

    for i, band_width in enumerate(widths):
        end = start + int(band_width)

        intervals.append((start, end))

        start = end + int(gaps[i + 1])

    assert sum(
        b - a for a, b in intervals
    ) == total

    assert intervals[-1][1] <= dimension

    return intervals


def choose_noise_bands(
    width, height, centers, count, rng
):
    """
    Try horizontal and vertical band configurations.

    Select the arrangement covering the most
    annotated vehicle centers.
    """

    best = None
    best_score = -1

    for orientation, dimension, axis in (
        ("horizontal", height, 1),
        ("vertical", width, 0),
    ):
        for _ in range(100):
            intervals = random_band_intervals(
                dimension, count, rng
            )

            coords = centers[:, axis]

            hits = sum(
                int(np.count_nonzero(
                    (coords >= a) & (coords < b)
                ))
                for a, b in intervals
            )

            if (
                hits > best_score
                or (
                    hits == best_score
                    and rng.random() < 0.1
                )
            ):
                best_score = hits
                best = (orientation, intervals)

    orientation, intervals = best

    mask = np.zeros(
        (height, width), dtype=np.uint8
    )

    rectangles = []

    for a, b in intervals:
        if orientation == "horizontal":
            mask[a:b, :] = 255

            rectangles.append([
                0, a, width, b
            ])

        else:
            mask[:, a:b] = 255

            rectangles.append([
                a, 0, b, height
            ])

    return (
        mask,
        orientation,
        rectangles,
        best_score
    )


# ============================================================
# SENSOR NOISE: APPLICATION
# ============================================================

def add_sensor_noise(image, mask, orientation, rng):
    """
    Apply severe sensor noise inside bands.

    Combines:
        - Independent Gaussian pixel noise
        - Correlated row/column readout noise

    Outside the bands, pixels are unchanged.
    """

    height, width = image.shape[:2]

    n = (
        height
        if orientation == "horizontal"
        else width
    )

    # Correlated readout offsets.
    offsets = rng.normal(
        0, 42, n
    ).astype(np.float32)

    offsets = cv2.GaussianBlur(
        offsets.reshape(-1, 1),
        (1, 5),
        sigmaX=0,
        sigmaY=1.0
    ).reshape(-1)

    if orientation == "horizontal":
        stripe = offsets[:, None]
    else:
        stripe = offsets[None, :]

    # Severe independent pixel noise.
    noise = rng.normal(
        0, 48, (height, width)
    ).astype(np.float32)

    noise = noise + stripe

    if image.ndim == 3:
        noise = noise[:, :, None]

    degraded = np.clip(
        image.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)

    result = image.copy()

    result[mask > 0] = degraded[mask > 0]

    return result


# ============================================================
# DROPLET BLUR: MASK GENERATION
# ============================================================

def make_droplet_mask(
    width,
    height,
    droplets,
    scale,
    downsample=1
):
    """
    Rasterize irregular droplet-shaped regions.

    Each droplet is based on an ellipse with
    smooth radial perturbations.
    """

    small_width = max(
        2, round(width / downsample)
    )

    small_height = max(
        2, round(height / downsample)
    )

    mask = np.zeros(
        (small_height, small_width),
        dtype=np.uint8
    )

    for (
        cx, cy, rx, ry, angles, radial
    ) in droplets:

        x = (
            cx
            + rx * scale * radial * np.cos(angles)
        ) / downsample

        y = (
            cy
            + ry * scale * radial * np.sin(angles)
        ) / downsample

        points = np.column_stack(
            [x, y]
        ).round().astype(np.int32)

        cv2.fillPoly(
            mask,
            [points],
            255
        )

    return mask


def candidate_droplets(
    width, height, centers, count, rng
):
    """
    Generate candidate droplet locations.

    Prefer positions near annotated vehicles,
    with some random spatial variation.
    """

    droplets = []

    base = np.sqrt(
        0.5 / (count * np.pi)
    )

    for _ in range(count):

        if len(centers) and rng.random() < 0.75:
            target = centers[
                int(rng.integers(len(centers)))
            ]

            cx = float(
                target[0]
                + rng.normal(0, width * 0.06)
            )

            cy = float(
                target[1]
                + rng.normal(0, height * 0.06)
            )

        else:
            cx = float(
                rng.uniform(0, width)
            )

            cy = float(
                rng.uniform(0, height)
            )

        rx = (
            width
            * base
            * rng.uniform(0.75, 1.25)
        )

        ry = (
            height
            * base
            * rng.uniform(0.75, 1.25)
        )

        angles = np.linspace(
            0,
            2 * np.pi,
            96,
            endpoint=False
        )

        phases = rng.uniform(
            0, 2 * np.pi, 3
        )

        # Smooth irregular droplet boundaries.
        radial = (
            1.0
            + 0.11 * np.sin(
                3 * angles + phases[0]
            )
            + 0.07 * np.sin(
                5 * angles + phases[1]
            )
            + 0.04 * np.sin(
                7 * angles + phases[2]
            )
        )

        droplets.append((
            cx,
            cy,
            rx,
            ry,
            angles,
            radial
        ))

    return droplets


def choose_droplets(
    width, height, centers, count, rng
):
    """
    Choose irregular droplet regions.

    Adjust region sizes to reach approximately
    50% total image coverage.

    Favor arrangements covering vehicle centers.
    """

    best = None

    for _ in range(8):
        droplets = candidate_droplets(
            width,
            height,
            centers,
            count,
            rng
        )

        # First estimate coverage at reduced resolution.
        low = 0.05
        high = 10.0

        for _ in range(14):
            mid = (low + high) / 2

            small = make_droplet_mask(
                width,
                height,
                droplets,
                mid,
                downsample=4
            )

            if np.mean(small > 0) < 0.5:
                low = mid
            else:
                high = mid

        # Refine coverage at full resolution.
        low = max(0.01, low * 0.85)
        high = high * 1.15

        for _ in range(12):
            mid = (low + high) / 2

            mask = make_droplet_mask(
                width,
                height,
                droplets,
                mid
            )

            if np.mean(mask > 0) < 0.5:
                low = mid
            else:
                high = mid

        scale = (low + high) / 2

        mask = make_droplet_mask(
            width,
            height,
            droplets,
            scale
        )

        coverage = float(
            np.mean(mask > 0)
        )

        if len(centers):
            xs = np.clip(
                centers[:, 0].astype(int),
                0,
                width - 1
            )

            ys = np.clip(
                centers[:, 1].astype(int),
                0,
                height - 1
            )

            hits = int(
                np.count_nonzero(mask[ys, xs])
            )

        else:
            hits = 0

        score = (
            hits
            - 15 * abs(coverage - 0.5)
        )

        if best is None or score > best[0]:

            locations = [
                [
                    round(cx, 1),
                    round(cy, 1),
                    round(rx * scale, 1),
                    round(ry * scale, 1)
                ]
                for (
                    cx, cy, rx, ry, _, _
                ) in droplets
            ]

            best = (
                score,
                mask,
                locations,
                hits
            )

    _, mask, locations, hits = best

    return mask, locations, hits


# ============================================================
# DROPLET BLUR: APPLICATION
# ============================================================

def add_droplet_blur(image, mask):
    """
    Apply severe Gaussian blur in droplet regions.

    Feather boundaries to avoid hard polygon edges.

    This approximates lens contamination, not
    full physical water refraction.
    """

    height, width = image.shape[:2]

    blur_sigma = max(
        8.0,
        min(height, width) * 0.025
    )

    feather_sigma = max(
        2.0,
        min(height, width) * 0.008
    )

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma
    )

    alpha = cv2.GaussianBlur(
        mask.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=feather_sigma,
        sigmaY=feather_sigma
    )

    if image.ndim == 3:
        alpha = alpha[:, :, None]

    result = (
        image.astype(np.float32) * (1 - alpha)
        + blurred.astype(np.float32) * alpha
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# OUTPUT UTILITIES
# ============================================================

def save_jpg(path, image):
    success = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 97]
    )

    if not success:
        raise IOError(
            f"Could not save {path}"
        )


# ============================================================
# DATASET GENERATION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path("VisDrone-DroneVehicle")
    )

    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--ir-dir",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("processed_dataset/test")
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "processed_dataset_realistic_degraded_v2"
        )
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Number of pairs; 0 = all test pairs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # ORIGINAL DroneVehicle directories
    # --------------------------------------------------------

    # Matches original PACE build_dataset.py:
    # testimg  = RGB
    # testimgr = IR

    rgb_dir = (
        args.rgb_dir
        or args.original_root / "test" / "testimg"
    ).resolve()

    ir_dir = (
        args.ir_dir
        or args.original_root / "test" / "testimgr"
    ).resolve()

    annotations = args.annotations_root.resolve()

    output = args.output.resolve()

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    for source in (rgb_dir, ir_dir):
        if (
            "processed_dataset" in source.parts
            or any(
                "degraded" in part.lower()
                for part in source.parts
            )
        ):
            parser.error(
                f"Image input must be ORIGINAL DroneVehicle: "
                f"{source}"
            )

    if args.max_samples < 0:
        parser.error(
            "--max-samples must be >= 0"
        )

    if (
        output in (rgb_dir, ir_dir, annotations)
        or any(
            output in source.parents
            for source in (
                rgb_dir,
                ir_dir,
                annotations
            )
        )
    ):
        parser.error(
            "Output cannot overwrite a source directory"
        )

    if any(
        source == output or output in source.parents
        for source in (rgb_dir, ir_dir)
    ):
        parser.error(
            "Output cannot contain original images"
        )

    # --------------------------------------------------------
    # Match original RGB/IR image pairs
    # --------------------------------------------------------

    rgb_files = index_images(rgb_dir)
    ir_files = index_images(ir_dir)

    ids = sorted(
        set(rgb_files) & set(ir_files)
    )

    if args.max_samples:
        ids = ids[:args.max_samples]

    if not ids:
        parser.error(
            "No matching original RGB/IR test images found"
        )

    # --------------------------------------------------------
    # Output setup
    # --------------------------------------------------------

    marker = (
        output / "realistic_degradation_config.json"
    )

    if output.exists() and any(output.iterdir()):
        if (
            not args.overwrite
            or not marker.is_file()
        ):
            parser.error(
                "Output exists. Use --overwrite only "
                "for previously generated outputs."
            )

        shutil.rmtree(output)

    folders = (
        "rgb",
        "ir",
        "rgb_annotations",
        "ir_annotations",
        "degradation_masks"
    )

    for folder in folders:
        (
            output / "test" / folder
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    config = {
        "original_rgb": str(rgb_dir),
        "original_ir": str(ir_dir),
        "processed_annotations": str(annotations),
        "crop_pixels_per_side": BORDER_PIXELS,
        "seed": args.seed,
        "samples": len(ids),
        "degradation_modes": [
            "sensor_noise_bands",
            "droplet_blur"
        ],
        "target_coverage": 0.5,
        "region_count": "uniformly sampled 3-6",
        "source_images": "original DroneVehicle only"
    }

    marker.write_text(
        json.dumps(config, indent=2)
    )

    records = []

    # --------------------------------------------------------
    # Process each original image pair
    # --------------------------------------------------------

    for index, pair_id in enumerate(ids, 1):

        rng = make_rng(
            args.seed,
            pair_id
        )

        # Randomly choose one modality.
        modality = str(
            rng.choice(["rgb", "ir"])
        )

        # Randomly choose degradation.
        degradation = str(
            rng.choice([
                "sensor_noise_bands",
                "droplet_blur"
            ])
        )

        # Randomly choose 3-6 regions.
        region_count = int(
            rng.integers(3, 7)
        )

        # ----------------------------------------------------
        # Load ORIGINAL images
        # ----------------------------------------------------

        rgb = cv2.imread(
            str(rgb_files[pair_id]),
            cv2.IMREAD_UNCHANGED
        )

        ir = cv2.imread(
            str(ir_files[pair_id]),
            cv2.IMREAD_UNCHANGED
        )

        # ----------------------------------------------------
        # Apply ORIGINAL PACE crop to BOTH modalities
        # ----------------------------------------------------

        rgb = apply_crop(rgb)
        ir = apply_crop(ir)

        if rgb.shape[:2] != ir.shape[:2]:
            raise ValueError(
                f"{pair_id}: cropped RGB/IR sizes differ"
            )

        height, width = rgb.shape[:2]

        # ----------------------------------------------------
        # Load ALREADY-CROPPED XML annotations
        # ----------------------------------------------------

        xml_path = (
            annotations
            / f"{modality}_annotations"
            / f"{pair_id}.xml"
        )

        centers = annotation_centers(
            xml_path,
            width,
            height
        )

        image = (
            rgb if modality == "rgb"
            else ir
        )

        # ----------------------------------------------------
        # Generate and apply degradation
        # ----------------------------------------------------

        if degradation == "sensor_noise_bands":

            (
                mask,
                orientation,
                regions,
                hits
            ) = choose_noise_bands(
                width,
                height,
                centers,
                region_count,
                rng
            )

            changed = add_sensor_noise(
                image,
                mask,
                orientation,
                rng
            )

        else:

            (
                mask,
                regions,
                hits
            ) = choose_droplets(
                width,
                height,
                centers,
                region_count,
                rng
            )

            orientation = "none"

            changed = add_droplet_blur(
                image,
                mask
            )

        # ----------------------------------------------------
        # Replace ONLY the selected modality
        # ----------------------------------------------------

        if modality == "rgb":
            rgb = changed
        else:
            ir = changed

        # ----------------------------------------------------
        # Save cropped RGB/IR pair
        # ----------------------------------------------------

        save_jpg(
            output / "test" / "rgb" / f"{pair_id}.jpg",
            rgb
        )

        save_jpg(
            output / "test" / "ir" / f"{pair_id}.jpg",
            ir
        )

        # ----------------------------------------------------
        # Save degradation mask
        # ----------------------------------------------------

        mask_path = (
            output
            / "test"
            / "degradation_masks"
            / f"{pair_id}.png"
        )

        if not cv2.imwrite(
            str(mask_path),
            mask
        ):
            raise IOError(
                f"Could not save mask for {pair_id}"
            )

        # ----------------------------------------------------
        # Copy processed annotations unchanged
        # ----------------------------------------------------

        for sensor in ("rgb", "ir"):

            source_xml = (
                annotations
                / f"{sensor}_annotations"
                / f"{pair_id}.xml"
            )

            if not source_xml.is_file():
                raise FileNotFoundError(
                    source_xml
                )

            destination_xml = (
                output
                / "test"
                / f"{sensor}_annotations"
                / source_xml.name
            )

            shutil.copy2(
                source_xml,
                destination_xml
            )

        # ----------------------------------------------------
        # Record metadata
        # ----------------------------------------------------

        records.append({
            "pair_id": pair_id,
            "modality": modality,
            "degradation": degradation,
            "orientation": orientation,
            "region_count": region_count,
            "coverage": round(
                float(np.mean(mask > 0)),
                5
            ),
            "vehicles_total": len(centers),
            "vehicles_covered": hits,
            "regions": json.dumps(regions),
            "seed": args.seed
        })

        if (
            index % 100 == 0
            or index == len(ids)
        ):
            print(
                f"Processed {index}/{len(ids)}"
            )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_path = (
        output / "metadata.csv"
    )

    with metadata_path.open(
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0])
        )

        writer.writeheader()
        writer.writerows(records)

    print(
        f"\nDone: {len(ids)} original pairs"
        f"\nOutput: {output}"
    )


if __name__ == "__main__":
    main()