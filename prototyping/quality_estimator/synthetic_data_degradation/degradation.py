"""
Key degradation utility script. 

Goal: take an image and return a degraded image plus metadata describing exactly what was done.
"""

import cv2
import numpy as np
import random


def generate_mask(height, width):
    """
    Generate a random spatial region independent of the 8x8 feature grid.
    """
    mask = np.zeros((height, width), dtype=np.float32)

    # Random ellipse
    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)

    rx = random.randint(width // 20, width // 4)
    ry = random.randint(height // 20, height // 4)

    cv2.ellipse(
        mask,
        (cx, cy),
        (rx, ry),
        angle=random.uniform(0, 180),
        startAngle=0,
        endAngle=360,
        color=1.0,
        thickness=-1
    )

    return mask


def apply_blur(image, mask, severity):
    """
    severity: [0, 1]
    """
    sigma = 0.5 + severity * 5.0

    blurred = cv2.GaussianBlur(
        image,
        ksize=(0, 0),
        sigmaX=sigma
    )

    mask = mask[..., None]
    return image * (1 - mask) + blurred * mask


def apply_noise(image, mask, severity):
    sigma = severity * 40

    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image + noise, 0, 255)

    mask = mask[..., None]
    return image * (1 - mask) + noisy * mask


def apply_contrast_loss(image, mask, severity):
    # severity 0 -> original contrast
    # severity 1 -> very low contrast
    alpha = 1.0 - 0.8 * severity

    mean = image.mean(axis=(0, 1), keepdims=True)
    low_contrast = mean + alpha * (image - mean)
    low_contrast = np.clip(low_contrast, 0, 255)

    mask = mask[..., None]
    return image * (1 - mask) + low_contrast * mask


def apply_degradation(image, degradation_type=None):
    """
    Main entry point.

    Returns:
        degraded_image
        metadata
    """

    h, w = image.shape[:2]

    if degradation_type is None:
        degradation_type = random.choice([
            "blur",
            "noise",
            "contrast"
        ])

    severity = random.uniform(0.2, 1.0)
    mask = generate_mask(h, w)

    if degradation_type == "blur":
        output = apply_blur(image, mask, severity)

    elif degradation_type == "noise":
        output = apply_noise(image, mask, severity)

    elif degradation_type == "contrast":
        output = apply_contrast_loss(image, mask, severity)

    else:
        raise ValueError(f"Unknown degradation: {degradation_type}")

    metadata = {
        "type": degradation_type,
        "severity": severity,
        "mask": mask
    }

    return output.astype(np.uint8), metadata