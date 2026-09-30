"""

HOW to degrade an image.

Synthetic camera and sensor degradation utilities.

Goal: take an image and return a realistically degraded image plus metadata describing exactly what was done.
"""

import cv2
import numpy as np
import random


def generate_mask(height, width, num_regions=None):
    """
    Generate a binary mask containing multiple independent degradation
    regions.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if num_regions is None:
        num_regions = random.randint(1, 4)

    for _ in range(num_regions):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)

        rx = random.randint(
            max(2, width // 30),
            max(3, width // 5)
        )
        ry = random.randint(
            max(2, height // 30),
            max(3, height // 5)
        )

        points = []
        num_points = random.randint(8, 14)

        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            radius = random.uniform(0.7, 1.3)

            x = int(cx + rx * radius * np.cos(angle))
            y = int(cy + ry * radius * np.sin(angle))

            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)

    return mask


def apply_water_blur(image, mask, severity):
    """
    Simulate water droplets, condensation, or a dirty camera lens.
    """
    sigma = 0.5 + severity * 7.0

    blurred = cv2.GaussianBlur(
        image,
        ksize=(0, 0),
        sigmaX=sigma
    )

    mask = mask[..., None]

    return image * (1 - mask) + blurred * mask


def apply_motion_blur(image, mask, severity):
    """
    Simulate camera motion during image capture.
    """
    size = int(3 + severity * 18)
    size = size if size % 2 == 1 else size + 1

    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size

    blurred = cv2.filter2D(image, -1, kernel)

    mask = mask[..., None]

    return image * (1 - mask) + blurred * mask

def apply_sensor_noise(image, mask, severity):
    """
    Simulate highly disruptive electronic sensor noise.
    Includes high-variance Gaussian noise, line striping, and dead/stuck pixels.
    Works with RGB and single-channel IR images.
    """
    
    sigma = severity * 60.0
    noise = np.random.normal(0, sigma, image.shape)
    
    noisy = image.astype(np.float32) + noise
    
    h, _ = image.shape[:2]
    num_corrupted_lines = int(severity * (h // 10))
    if num_corrupted_lines > 0:
        corrupted_rows = np.random.choice(h, num_corrupted_lines, replace=False)
        
        line_offsets = np.random.uniform(-100, 100, size=(num_corrupted_lines, 1))
        if len(image.shape) == 3:
            line_offsets = np.expand_dims(line_offsets, axis=-1)
        noisy[corrupted_rows, :] += line_offsets

    # Salt and Pepper noise
    sp_ratio = severity * 0.08
    random_matrix = np.random.random(image.shape[:2])
    
    salt_mask = random_matrix < (sp_ratio / 2)
    noisy[salt_mask] = 255

    pepper_mask = (random_matrix >= (sp_ratio / 2)) & (random_matrix < sp_ratio)
    noisy[pepper_mask] = 0

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    if len(image.shape) == 3 and len(mask.shape) == 2:
        mask = mask[..., None]

    return (image * (1 - mask) + noisy * mask).astype(np.uint8)


def apply_saturation(image, mask, severity):
    """
    Simulate localized sensor saturation or overexposure.
    """
    output = image.astype(np.float32)

    amount = 0.3 + 0.7 * severity

    saturated = output + (
        (255.0 - output) * amount
    )

    saturated = np.clip(
        saturated,
        0,
        255
    )

    mask = mask[..., None]

    return output * (1 - mask) + saturated * mask


def apply_degradation(
    image,
    degradation_type=None,
    severity=None,
    mask=None
):
    """
    Main entry point.

    Args:
        image: RGB or IR image as a NumPy array.

        degradation_type:
            Type of degradation to apply. If None,
            one is selected randomly.

        severity:
            Degradation severity in the range [0, 1].
            If None, a random severity is selected.

        mask:
            Optional externally generated binary degradation mask.
            If None, a random mask is generated.

    Returns:
        degraded_image:
            Degraded image as uint8.

        metadata:
            Dictionary describing the degradation.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    if image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255
        ).astype(np.uint8)

    height, width = image.shape[:2]

    degradation_types = [
        "water_blur",
        "motion_blur",
        "sensor_noise",
        "saturation"
    ]

    # Randomly choose degradation type if not provided.
    if degradation_type is None:
        degradation_type = random.choice(
            degradation_types
        )

    # Randomly choose severity if not provided.
    if severity is None:
        severity = random.uniform(
            0.2,
            1.0
        )

    # Generate a random degradation mask unless
    # an external mask was provided.
    if mask is None:
        mask = generate_mask(
            height,
            width
        )
    else:
        # Ensure supplied mask matches image dimensions.
        if mask.shape != (height, width):
            raise ValueError(
                f"Mask shape {mask.shape} does not match "
                f"image shape {(height, width)}."
            )

        # Ensure binary uint8 mask.
        mask = (
            mask > 0
        ).astype(np.uint8)

    # Apply selected degradation.
    if degradation_type == "water_blur":
        output = apply_water_blur(
            image,
            mask,
            severity
        )

    elif degradation_type == "motion_blur":
        output = apply_motion_blur(
            image,
            mask,
            severity
        )

    elif degradation_type == "sensor_noise":
        output = apply_sensor_noise(
            image,
            mask,
            severity
        )

    elif degradation_type == "saturation":
        output = apply_saturation(
            image,
            mask,
            severity
        )

    else:
        raise ValueError(
            f"Unknown degradation: {degradation_type}"
        )

    metadata = {
        "type": degradation_type,
        "severity": severity,
        "mask": mask
    }

    return (
        np.clip(
            output,
            0,
            255
        ).astype(np.uint8),
        metadata
    )