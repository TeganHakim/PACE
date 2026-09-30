"""
How to turn degradation into labels.
- overlay 8x8 grid
- calculate degradation coverage per patch
- combine baseline quality + coverage + severity
- output final 8x8 quality map
"""

import numpy as np


GRID_SIZE = 8


def mask_to_coverage(mask, grid_size=GRID_SIZE):
    """
    Convert a binary H x W degradation mask into an 8x8
    degradation coverage map.

    Each cell represents the fraction of pixels within that
    spatial region that were degraded.

    Returns:
        coverage: grid_size x grid_size float32 array in [0, 1].
    """
    if mask is None:
        raise ValueError("Mask cannot be None.")

    if mask.ndim != 2:
        raise ValueError(
            "Degradation mask must be a 2D array."
        )

    mask = (mask > 0).astype(np.float32)

    height, width = mask.shape

    row_edges = np.linspace(
        0, height, grid_size + 1, dtype=int
    )
    col_edges = np.linspace(
        0, width, grid_size + 1, dtype=int
    )

    coverage = np.zeros(
        (grid_size, grid_size),
        dtype=np.float32
    )

    for row in range(grid_size):
        for col in range(grid_size):
            region = mask[
                row_edges[row]:row_edges[row + 1],
                col_edges[col]:col_edges[col + 1]
            ]

            if region.size > 0:
                coverage[row, col] = np.mean(region)

    return coverage


def compute_final_quality(
    baseline_quality,
    coverage,
    severity
):
    """
    Combine baseline sensor quality with synthetic degradation.

    Formula:

        Q_final = Q_base * (1 - coverage * severity)

    Args:
        baseline_quality:
            8x8 baseline quality map.

        coverage:
            8x8 degradation coverage map.

        severity:
            Scalar degradation severity in [0, 1].

    Returns:
        final_quality:
            8x8 quality map in [0, 1].
    """
    baseline_quality = np.asarray(
        baseline_quality,
        dtype=np.float32
    )

    coverage = np.asarray(
        coverage,
        dtype=np.float32
    )

    if baseline_quality.shape != coverage.shape:
        raise ValueError(
            "Baseline quality and coverage must have "
            "the same shape."
        )

    if not 0.0 <= severity <= 1.0:
        raise ValueError(
            "Severity must be between 0 and 1."
        )

    quality_loss = coverage * severity

    final_quality = (
        baseline_quality *
        (1.0 - quality_loss)
    )

    return np.clip(
        final_quality,
        0.0,
        1.0
    ).astype(np.float32)


def create_ir_baseline(grid_size=GRID_SIZE):
    """
    Create the reference-quality IR baseline.
    """
    return np.ones(
        (grid_size, grid_size),
        dtype=np.float32
    )