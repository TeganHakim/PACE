"""
HOW to degrade an image.

Synthetic camera and sensor degradation utilities.

Goal: take an image and return a realistically degraded image plus
metadata describing exactly what was done.

All degradation functions support both:
    - grayscale images: H x W
    - multi-channel images: H x W x C
"""

import cv2
import numpy as np
import random


# ------------------------------------------------------------------
# Mask utilities
# ------------------------------------------------------------------

def generate_mask(height, width, num_regions=None):
    """
    Generate a binary mask containing multiple independent degradation
    regions.

    Returns
    -------
    np.ndarray
        H x W uint8 binary mask containing values 0 or 1.
    """

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if num_regions is None:
        num_regions = random.randint(
            1,
            4
        )

    for _ in range(num_regions):

        cx = random.randint(
            0,
            width - 1
        )

        cy = random.randint(
            0,
            height - 1
        )

        rx = random.randint(
            max(2, width // 30),
            max(3, width // 5)
        )

        ry = random.randint(
            max(2, height // 30),
            max(3, height // 5)
        )

        points = []

        num_points = random.randint(
            8,
            14
        )

        for i in range(num_points):

            angle = (
                2
                * np.pi
                * i
                / num_points
            )

            radius = random.uniform(
                0.7,
                1.3
            )

            x = int(
                cx
                + rx
                * radius
                * np.cos(angle)
            )

            y = int(
                cy
                + ry
                * radius
                * np.sin(angle)
            )

            points.append(
                [x, y]
            )

        points = np.array(
            points,
            dtype=np.int32
        )

        cv2.fillPoly(
            mask,
            [points],
            1
        )

    return mask


def prepare_mask_for_image(mask, image):
    """
    Convert a 2D binary mask into a shape that broadcasts correctly
    with the supplied image.

    Grayscale image:
        image -> H x W
        mask  -> H x W

    Multi-channel image:
        image -> H x W x C
        mask  -> H x W x 1
    """

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2D or 3D image, "
            f"got shape {image.shape}."
        )

    # Remove a singleton channel dimension if one was supplied.
    if mask.ndim == 3:

        if mask.shape[2] != 1:
            raise ValueError(
                f"Expected a 2D mask or HxWx1 mask, "
                f"got shape {mask.shape}."
            )

        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, "
            f"got shape {mask.shape}."
        )

    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match "
            f"image spatial shape {image.shape[:2]}."
        )

    mask = (
        mask > 0
    ).astype(np.float32)

    if image.ndim == 3:
        mask = mask[..., None]

    return mask


def blend_with_mask(
    original,
    degraded,
    mask
):
    """
    Blend a degraded image into the original image using a binary mask.

    Handles both grayscale and multi-channel images.
    """

    if original.shape != degraded.shape:
        raise ValueError(
            f"Original image shape {original.shape} "
            f"does not match degraded image shape "
            f"{degraded.shape}."
        )

    blend_mask = prepare_mask_for_image(
        mask,
        original
    )

    original_float = (
        original.astype(np.float32)
    )

    degraded_float = (
        degraded.astype(np.float32)
    )

    output = (
        original_float
        * (1.0 - blend_mask)
        +
        degraded_float
        * blend_mask
    )

    return np.clip(
        output,
        0,
        255
    ).astype(np.uint8)


# ------------------------------------------------------------------
# Water blur
# ------------------------------------------------------------------

def apply_water_blur(
    image,
    mask,
    severity
):
    """
    Simulate water droplets, condensation, or a dirty camera lens.

    Supports both grayscale and multi-channel images.
    """

    sigma = (
        0.5
        + severity * 7.0
    )

    blurred = cv2.GaussianBlur(
        image,
        ksize=(0, 0),
        sigmaX=sigma
    )

    return blend_with_mask(
        image,
        blurred,
        mask
    )


# ------------------------------------------------------------------
# Motion blur
# ------------------------------------------------------------------

def apply_motion_blur(
    image,
    mask,
    severity
):
    """
    Simulate camera motion during image capture.

    Supports both grayscale and multi-channel images.
    """

    size = int(
        3
        + severity * 18
    )

    if size % 2 == 0:
        size += 1

    kernel = np.zeros(
        (size, size),
        dtype=np.float32
    )

    kernel[
        size // 2,
        :
    ] = (
        1.0 / size
    )

    blurred = cv2.filter2D(
        image,
        -1,
        kernel
    )

    return blend_with_mask(
        image,
        blurred,
        mask
    )


# ------------------------------------------------------------------
# Sensor noise
# ------------------------------------------------------------------

def apply_sensor_noise(
    image,
    mask,
    severity
):
    """
    Simulate highly disruptive electronic sensor noise.

    Includes:
        - high-variance Gaussian noise
        - line striping
        - salt-and-pepper / dead-stuck pixels

    Supports both grayscale and multi-channel images.
    """

    sigma = (
        severity * 60.0
    )

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    ).astype(np.float32)

    noisy = (
        image.astype(np.float32)
        + noise
    )

    # --------------------------------------------------------------
    # Corrupted horizontal sensor lines
    # --------------------------------------------------------------

    height = image.shape[0]

    num_corrupted_lines = int(
        severity
        * (height // 10)
    )

    if num_corrupted_lines > 0:

        num_corrupted_lines = min(
            num_corrupted_lines,
            height
        )

        corrupted_rows = (
            np.random.choice(
                height,
                num_corrupted_lines,
                replace=False
            )
        )

        if image.ndim == 2:

            line_offsets = (
                np.random.uniform(
                    -100,
                    100,
                    size=(
                        num_corrupted_lines,
                        1
                    )
                )
            )

        else:

            line_offsets = (
                np.random.uniform(
                    -100,
                    100,
                    size=(
                        num_corrupted_lines,
                        1,
                        1
                    )
                )
            )

        noisy[
            corrupted_rows,
            ...
        ] += line_offsets

    # --------------------------------------------------------------
    # Salt-and-pepper / dead-stuck pixels
    # --------------------------------------------------------------

    sp_ratio = (
        severity * 0.08
    )

    random_matrix = (
        np.random.random(
            image.shape[:2]
        )
    )

    salt_mask = (
        random_matrix
        < (sp_ratio / 2.0)
    )

    pepper_mask = (
        (
            random_matrix
            >= (sp_ratio / 2.0)
        )
        &
        (
            random_matrix
            < sp_ratio
        )
    )

    # NumPy applies the H x W boolean mask to all channels
    # automatically for H x W x C images.
    noisy[salt_mask] = 255.0
    noisy[pepper_mask] = 0.0

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    return blend_with_mask(
        image,
        noisy,
        mask
    )


# ------------------------------------------------------------------
# Saturation
# ------------------------------------------------------------------

def apply_saturation(
    image,
    mask,
    severity
):
    """
    Simulate localized sensor saturation or overexposure.

    Supports both grayscale and multi-channel images.
    """

    output = (
        image.astype(np.float32)
    )

    amount = (
        0.3
        + 0.7 * severity
    )

    saturated = (
        output
        +
        (
            255.0 - output
        )
        * amount
    )

    saturated = np.clip(
        saturated,
        0,
        255
    ).astype(np.uint8)

    return blend_with_mask(
        image,
        saturated,
        mask
    )


# ------------------------------------------------------------------
# Main degradation interface
# ------------------------------------------------------------------

def apply_degradation(
    image,
    degradation_type=None,
    severity=None,
    mask=None
):
    """
    Apply one synthetic degradation to an image.

    Parameters
    ----------
    image : np.ndarray
        Input RGB/IR image.

        Supported shapes:
            H x W
            H x W x C

    degradation_type : str or None
        One of:
            water_blur
            motion_blur
            sensor_noise
            saturation

        If None, one is selected randomly.

    severity : float or None
        Degradation severity in [0, 1].

        If None, a random value in [0.2, 1.0] is selected.

    mask : np.ndarray or None
        Optional externally generated H x W binary degradation mask.

        If None, a random mask is generated.

    Returns
    -------
    degraded_image : np.ndarray
        Degraded uint8 image with the same shape as the input.

    metadata : dict
        Dictionary containing:
            type
            severity
            mask
    """

    # --------------------------------------------------------------
    # Validate image
    # --------------------------------------------------------------

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected image shape HxW or HxWxC, "
            f"got {image.shape}."
        )

    if image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255
        ).astype(np.uint8)

    height, width = (
        image.shape[:2]
    )

    # --------------------------------------------------------------
    # Degradation type
    # --------------------------------------------------------------

    degradation_types = [
        "water_blur",
        "motion_blur",
        "sensor_noise",
        "saturation"
    ]

    if degradation_type is None:

        degradation_type = (
            random.choice(
                degradation_types
            )
        )

    if (
        degradation_type
        not in degradation_types
    ):
        raise ValueError(
            f"Unknown degradation: "
            f"{degradation_type}"
        )

    # --------------------------------------------------------------
    # Severity
    # --------------------------------------------------------------

    if severity is None:

        severity = (
            random.uniform(
                0.2,
                1.0
            )
        )

    severity = float(
        severity
    )

    if not (
        0.0
        <= severity
        <= 1.0
    ):
        raise ValueError(
            f"Severity must be in [0, 1], "
            f"got {severity}."
        )

    # --------------------------------------------------------------
    # Mask
    # --------------------------------------------------------------

    if mask is None:

        mask = generate_mask(
            height,
            width
        )

    else:

        # Allow H x W x 1 input, but canonicalize all metadata masks
        # to H x W.
        if (
            mask.ndim == 3
            and mask.shape[2] == 1
        ):
            mask = mask[:, :, 0]

        if mask.shape != (
            height,
            width
        ):
            raise ValueError(
                f"Mask shape {mask.shape} "
                f"does not match image shape "
                f"{(height, width)}."
            )

        mask = (
            mask > 0
        ).astype(np.uint8)

    # --------------------------------------------------------------
    # Apply degradation
    # --------------------------------------------------------------

    if (
        degradation_type
        == "water_blur"
    ):

        output = (
            apply_water_blur(
                image,
                mask,
                severity
            )
        )

    elif (
        degradation_type
        == "motion_blur"
    ):

        output = (
            apply_motion_blur(
                image,
                mask,
                severity
            )
        )

    elif (
        degradation_type
        == "sensor_noise"
    ):

        output = (
            apply_sensor_noise(
                image,
                mask,
                severity
            )
        )

    elif (
        degradation_type
        == "saturation"
    ):

        output = (
            apply_saturation(
                image,
                mask,
                severity
            )
        )

    # --------------------------------------------------------------
    # Final validation
    # --------------------------------------------------------------

    output = np.clip(
        output,
        0,
        255
    ).astype(np.uint8)

    if output.shape != image.shape:
        raise RuntimeError(
            f"Degradation changed image shape "
            f"from {image.shape} "
            f"to {output.shape}."
        )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    metadata = {
        "type": degradation_type,
        "severity": severity,
        "mask": mask
    }

    return (
        output,
        metadata
    )