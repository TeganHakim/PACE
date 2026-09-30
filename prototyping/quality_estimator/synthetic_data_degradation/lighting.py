"""
Baseline RGB quality estimation from lighting.

The goal is NOT to measure how bright a region is.

Instead, lighting quality estimates whether insufficient illumination
makes RGB information unreliable.

Therefore:
    - very dark pixels receive quality near 0
    - dim / transitional pixels receive intermediate quality
    - adequately illuminated pixels quickly saturate at quality 1

IR baseline quality is handled separately and is assumed to be 1.0
before synthetic degradation.
"""

import cv2
import numpy as np


# ------------------------------------------------------------------
# Lighting thresholds
# ------------------------------------------------------------------

# Below this luminance, RGB information is considered effectively
# unusable because of insufficient illumination.
DARK_THRESHOLD = 0.05

# Above this luminance, lighting is considered sufficiently good.
# Further brightness should NOT increase quality.
GOOD_THRESHOLD = 0.15


# ------------------------------------------------------------------
# Luminance
# ------------------------------------------------------------------

def compute_luminance(image):
    """
    Compute normalized luminance for a BGR OpenCV image.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.

    Returns
    -------
    np.ndarray
        2D float32 luminance map in [0, 1].
    """

    if image is None:
        raise ValueError("image cannot be None")

    image_float = (
        image.astype(np.float32)
        / 255.0
    )

    b = image_float[:, :, 0]
    g = image_float[:, :, 1]
    r = image_float[:, :, 2]

    # Standard perceptual luminance weighting.
    luminance = (
        0.2126 * r
        + 0.7152 * g
        + 0.0722 * b
    )

    return luminance.astype(
        np.float32
    )


# ------------------------------------------------------------------
# Luminance -> quality
# ------------------------------------------------------------------

def luminance_to_quality(luminance):
    """
    Convert luminance into baseline RGB lighting quality.

    This is intentionally a narrow transition.

    DARK_THRESHOLD:
        quality = 0

    GOOD_THRESHOLD:
        quality = 1

    Between the thresholds:
        quality increases linearly from 0 -> 1.

    Importantly, pixels brighter than GOOD_THRESHOLD are all assigned
    quality 1.0. We do not penalize naturally darker scene content
    once it is sufficiently illuminated.
    """

    luminance = np.asarray(
        luminance,
        dtype=np.float32
    )

    quality = (
        luminance - DARK_THRESHOLD
    ) / (
        GOOD_THRESHOLD - DARK_THRESHOLD
    )

    quality = np.clip(
        quality,
        0.0,
        1.0
    )

    return quality.astype(
        np.float32
    )


# ------------------------------------------------------------------
# Full-resolution lighting quality
# ------------------------------------------------------------------

def compute_lighting_quality(image):
    """
    Compute a full-resolution RGB lighting-quality map.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.

    Returns
    -------
    np.ndarray
        H x W quality map in [0, 1].
    """

    luminance = compute_luminance(
        image
    )

    quality = luminance_to_quality(
        luminance
    )

    return quality


# ------------------------------------------------------------------
# 8x8 baseline quality
# ------------------------------------------------------------------

def compute_rgb_quality_map(
    image,
    grid_size=8
):
    """
    Compute the 8x8 baseline RGB quality map.

    Lighting quality is first calculated at pixel resolution and then
    averaged within each spatial grid cell.

    This preserves localized low-light information rather than
    collapsing an entire patch to its mean luminance before evaluating
    lighting quality.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.

    grid_size : int
        Number of rows and columns in the output quality grid.

    Returns
    -------
    np.ndarray
        grid_size x grid_size float32 quality map.
    """

    pixel_quality = (
        compute_lighting_quality(
            image
        )
    )

    height, width = (
        pixel_quality.shape
    )

    row_edges = np.linspace(
        0,
        height,
        grid_size + 1,
        dtype=int
    )

    col_edges = np.linspace(
        0,
        width,
        grid_size + 1,
        dtype=int
    )

    quality_map = np.zeros(
        (
            grid_size,
            grid_size
        ),
        dtype=np.float32
    )

    for row in range(
        grid_size
    ):

        y0 = row_edges[row]
        y1 = row_edges[row + 1]

        for col in range(
            grid_size
        ):

            x0 = col_edges[col]
            x1 = col_edges[col + 1]

            patch = pixel_quality[
                y0:y1,
                x0:x1
            ]

            if patch.size == 0:
                quality_map[
                    row,
                    col
                ] = 1.0
            else:
                quality_map[
                    row,
                    col
                ] = float(
                    np.mean(patch)
                )

    return quality_map


# ------------------------------------------------------------------
# RGB degradation eligibility
# ------------------------------------------------------------------

def sufficiently_lit_mask(image):
    """
    Return a boolean mask identifying pixels that are sufficiently
    illuminated for meaningful synthetic RGB degradation.

    We only synthetically degrade RGB where the underlying RGB signal
    is already usable.

    This prevents synthetic degradation from being placed primarily
    over regions that are already unusable because of darkness.
    """

    luminance = compute_luminance(
        image
    )

    return (
        luminance >= GOOD_THRESHOLD
    )


# ------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------

def get_lighting_quality(
    image,
    grid_size=8
):
    """
    Convenience wrapper for generating an 8x8 RGB lighting-quality map.
    """

    return compute_rgb_quality_map(
        image,
        grid_size=grid_size
    )

def compute_rgb_baseline_quality(
    image,
    grid_size=8
):
    """
    Compute baseline RGB quality information.

    Returns
    -------
    quality_map : np.ndarray
        8x8 baseline RGB quality map.

    sufficient_light_mask : np.ndarray
        Full-resolution uint8 mask indicating pixels where RGB
        contains sufficient visible-light information for meaningful
        synthetic degradation.

    luminance : np.ndarray
        Full-resolution normalized luminance map in [0, 1].
    """

    # Full-resolution luminance
    luminance = compute_luminance(image)

    # Full-resolution lighting quality
    pixel_quality = luminance_to_quality(
        luminance
    )

    # Pixels where RGB is sufficiently illuminated for synthetic
    # degradation.
    sufficient_light_mask = (
        luminance >= GOOD_THRESHOLD
    ).astype(np.uint8)

    # --------------------------------------------------------------
    # Convert full-resolution quality into 8x8 target
    # --------------------------------------------------------------

    height, width = pixel_quality.shape

    row_edges = np.linspace(
        0,
        height,
        grid_size + 1,
        dtype=int
    )

    col_edges = np.linspace(
        0,
        width,
        grid_size + 1,
        dtype=int
    )

    quality_map = np.zeros(
        (grid_size, grid_size),
        dtype=np.float32
    )

    for row in range(grid_size):

        y0 = row_edges[row]
        y1 = row_edges[row + 1]

        for col in range(grid_size):

            x0 = col_edges[col]
            x1 = col_edges[col + 1]

            patch = pixel_quality[
                y0:y1,
                x0:x1
            ]

            if patch.size > 0:
                quality_map[row, col] = float(
                    np.mean(patch)
                )

    return (
        quality_map,
        sufficient_light_mask,
        luminance
    )