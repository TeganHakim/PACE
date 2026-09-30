"""
How good the original RGB image is
- compute pixel luminance
- determine sufficiently-lit regions
- produce baseline RGB 8×8 quality map

RGB lighting analysis utilities.

Note: lighting thresholds should be updated after sufficient luminance stats are checked across a sample of the dataset
"""

import cv2
import numpy as np


GRID_SIZE = 8

# Initial thresholds. These should be calibrated against the
# DroneVehicle RGB luminance distribution before full generation.
DARK_THRESHOLD = 0.10
GOOD_THRESHOLD = 0.35


def compute_luminance(image):
    """
    Convert an RGB/BGR uint8 image into normalized luminance [0, 1].

    OpenCV images are assumed to use BGR channel ordering.

    Returns:
        luminance: H x W float32 array in [0, 1].
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    image = image.astype(np.float32) / 255.0

    # OpenCV ordering: B, G, R
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    luminance = (
        0.2126 * r +
        0.7152 * g +
        0.0722 * b
    )

    return np.clip(luminance, 0.0, 1.0)


def luminance_to_quality(
    luminance,
    dark_threshold=DARK_THRESHOLD,
    good_threshold=GOOD_THRESHOLD
):
    """
    Convert luminance into continuous quality values.

    luminance <= dark_threshold -> 0
    luminance >= good_threshold -> 1
    values between thresholds are linearly interpolated.
    """
    if good_threshold <= dark_threshold:
        raise ValueError(
            "good_threshold must be greater than dark_threshold."
        )

    quality = (
        (luminance - dark_threshold) /
        (good_threshold - dark_threshold)
    )

    return np.clip(quality, 0.0, 1.0).astype(np.float32)


def compute_sufficient_light_mask(
    luminance,
    threshold=GOOD_THRESHOLD
):
    """
    Return a binary H x W mask indicating pixels with sufficient
    illumination for meaningful synthetic RGB degradation.
    """
    return (luminance >= threshold).astype(np.uint8)


def downsample_to_grid(values, grid_size=GRID_SIZE):
    """
    Convert a pixel-level H x W map into a grid_size x grid_size map.

    Each grid cell contains the mean value of the corresponding
    image region.
    """
    height, width = values.shape

    row_edges = np.linspace(
        0, height, grid_size + 1, dtype=int
    )
    col_edges = np.linspace(
        0, width, grid_size + 1, dtype=int
    )

    grid = np.zeros(
        (grid_size, grid_size),
        dtype=np.float32
    )

    for row in range(grid_size):
        for col in range(grid_size):
            region = values[
                row_edges[row]:row_edges[row + 1],
                col_edges[col]:col_edges[col + 1]
            ]

            if region.size > 0:
                grid[row, col] = np.mean(region)

    return grid


def compute_rgb_baseline_quality(
    image,
    dark_threshold=DARK_THRESHOLD,
    good_threshold=GOOD_THRESHOLD
):
    """
    Compute all lighting information needed for one RGB image.

    Returns:
        baseline_quality:
            8x8 RGB baseline quality map.

        sufficient_light_mask:
            H x W binary mask specifying where RGB degradation
            can meaningfully be applied.

        luminance:
            H x W normalized luminance map.
    """
    luminance = compute_luminance(image)

    pixel_quality = luminance_to_quality(
        luminance,
        dark_threshold,
        good_threshold
    )

    baseline_quality = downsample_to_grid(
        pixel_quality
    )

    sufficient_light_mask = compute_sufficient_light_mask(
        luminance,
        threshold=good_threshold
    )

    return (
        baseline_quality,
        sufficient_light_mask,
        luminance
    )