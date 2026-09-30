"""
Visualize a processed PACE sample in three panels:

1. Processed image
2. Processed image with 8x8 quality map overlay
3. Processed image with XML bounding boxes overlay

Run from PACE root:

    python prototyping/quality_estimator/synthetic_data_degradation/visualize_quality.py 00004 ir
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PROCESSED_ROOT = Path("processed_dataset")
GRID_SIZE = 8


# ------------------------------------------------------------------
# XML loading
# ------------------------------------------------------------------

def load_boxes(xml_path):
    """
    Load DroneVehicle polygon annotations and convert each polygon
    into an enclosing axis-aligned bounding box.

    Returns:
        List of dictionaries containing:
            x1, y1, x2, y2, class_name
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []

    for obj in root.findall("object"):

        name_node = obj.find("name")

        class_name = (
            name_node.text
            if name_node is not None
            else "object"
        )

        polygon = obj.find("polygon")

        if polygon is None:
            continue

        points = []

        for i in range(1, 5):

            x_node = polygon.find(f"x{i}")
            y_node = polygon.find(f"y{i}")

            if (
                x_node is None or
                y_node is None or
                x_node.text is None or
                y_node.text is None
            ):
                continue

            x = float(x_node.text)
            y = float(y_node.text)

            points.append((x, y))

        if len(points) != 4:
            continue

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        boxes.append({
            "x1": min(xs),
            "y1": min(ys),
            "x2": max(xs),
            "y2": max(ys),
            "class_name": class_name,
        })

    return boxes


# ------------------------------------------------------------------
# Quality overlay
# ------------------------------------------------------------------

def draw_quality_overlay(
    ax,
    image,
    quality
):
    """
    Draw image with 8x8 quality grid and values.
    """

    height, width = image.shape[:2]

    ax.imshow(image)

    row_edges = np.linspace(
        0,
        height,
        GRID_SIZE + 1,
        dtype=int
    )

    col_edges = np.linspace(
        0,
        width,
        GRID_SIZE + 1,
        dtype=int
    )

    # --------------------------------------------------------------
    # Grid
    # --------------------------------------------------------------

    for y in row_edges:

        ax.plot(
            [-0.5, width - 0.5],
            [y - 0.5, y - 0.5],
            linewidth=1
        )

    for x in col_edges:

        ax.plot(
            [x - 0.5, x - 0.5],
            [-0.5, height - 0.5],
            linewidth=1
        )

    # --------------------------------------------------------------
    # Quality values
    # --------------------------------------------------------------

    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            x_center = (
                col_edges[col] +
                col_edges[col + 1]
            ) / 2

            y_center = (
                row_edges[row] +
                row_edges[row + 1]
            ) / 2

            ax.text(
                x_center,
                y_center,
                f"{quality[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                    pad=1.5
                )
            )

    ax.set_xlim(
        -0.5,
        width - 0.5
    )

    ax.set_ylim(
        height - 0.5,
        -0.5
    )

    ax.set_aspect("equal")


# ------------------------------------------------------------------
# Bounding-box overlay
# ------------------------------------------------------------------

def draw_boxes(
    ax,
    image,
    boxes
):
    """
    Draw adjusted XML bounding boxes over image.
    """

    height, width = image.shape[:2]

    ax.imshow(image)

    for box in boxes:

        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        box_width = x2 - x1
        box_height = y2 - y1

        rect = Rectangle(
            (x1, y1),
            box_width,
            box_height,
            fill=False,
            linewidth=1.5
        )

        ax.add_patch(rect)

        ax.text(
            x1,
            max(y1 - 3, 5),
            box["class_name"],
            fontsize=7,
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                pad=1
            )
        )

    ax.set_xlim(
        -0.5,
        width - 0.5
    )

    ax.set_ylim(
        height - 0.5,
        -0.5
    )

    ax.set_aspect("equal")


# ------------------------------------------------------------------
# Main visualization
# ------------------------------------------------------------------

def visualize_quality(
    pair_id,
    modality
):

    modality = modality.lower()

    if modality not in (
        "rgb",
        "ir"
    ):
        raise ValueError(
            "modality must be 'rgb' or 'ir'"
        )

    # --------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------

    image_path = (
        PROCESSED_ROOT /
        modality /
        f"{pair_id}.jpg"
    )

    quality_path = (
        PROCESSED_ROOT /
        "quality_maps" /
        f"{pair_id}_{modality}.npy"
    )

    if modality == "rgb":

        xml_path = (
            PROCESSED_ROOT /
            "rgb_annotations" /
            f"{pair_id}.xml"
        )

    else:

        xml_path = (
            PROCESSED_ROOT /
            "ir_annotations" /
            f"{pair_id}.xml"
        )

    # --------------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------------

    if not image_path.exists():
        raise FileNotFoundError(
            image_path
        )

    if not quality_path.exists():
        raise FileNotFoundError(
            quality_path
        )

    if not xml_path.exists():
        raise FileNotFoundError(
            xml_path
        )

    # --------------------------------------------------------------
    # Load image
    # --------------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Could not load "
            f"{image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------------
    # Load quality map
    # --------------------------------------------------------------

    quality = np.load(
        quality_path
    )

    if quality.shape != (
        GRID_SIZE,
        GRID_SIZE
    ):

        raise ValueError(
            f"Expected "
            f"{GRID_SIZE}x{GRID_SIZE} "
            f"quality map, "
            f"got {quality.shape}"
        )

    # --------------------------------------------------------------
    # Load XML annotations
    # --------------------------------------------------------------

    boxes = load_boxes(
        xml_path
    )

    height, width = (
        image.shape[:2]
    )

    # --------------------------------------------------------------
    # Print diagnostics
    # --------------------------------------------------------------

    print(
        f"Image: {image_path}"
    )

    print(
        f"Image shape: "
        f"{image.shape}"
    )

    print(
        f"Quality map: "
        f"{quality_path}"
    )

    print(
        f"Quality shape: "
        f"{quality.shape}"
    )

    print(
        f"Quality range: "
        f"{quality.min():.3f} - "
        f"{quality.max():.3f}"
    )

    print(
        f"XML: {xml_path}"
    )

    print(
        f"Bounding boxes: "
        f"{len(boxes)}"
    )

    # --------------------------------------------------------------
    # Three-panel display
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(21, 7)
    )

    # --------------------------------------------------------------
    # Panel 1: image only
    # --------------------------------------------------------------

    axes[0].imshow(
        image
    )

    axes[0].set_title(
        "Processed Image"
    )

    axes[0].set_xlim(
        -0.5,
        width - 0.5
    )

    axes[0].set_ylim(
        height - 0.5,
        -0.5
    )

    axes[0].set_aspect(
        "equal"
    )

    axes[0].axis(
        "off"
    )

    # --------------------------------------------------------------
    # Panel 2: quality map
    # --------------------------------------------------------------

    draw_quality_overlay(
        axes[1],
        image,
        quality
    )

    axes[1].set_title(
        "Quality Map"
    )

    axes[1].axis(
        "off"
    )

    # --------------------------------------------------------------
    # Panel 3: bounding boxes
    # --------------------------------------------------------------

    draw_boxes(
        axes[2],
        image,
        boxes
    )

    axes[2].set_title(
        f"XML Bounding Boxes "
        f"({len(boxes)} objects)"
    )

    axes[2].axis(
        "off"
    )

    # --------------------------------------------------------------
    # Overall title
    # --------------------------------------------------------------

    fig.suptitle(
        f"Pair {pair_id} — "
        f"{modality.upper()}",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize processed image, "
            "quality labels, and XML annotations."
        )
    )

    parser.add_argument(
        "pair_id",
        help=(
            "Pair ID, e.g. 00004"
        )
    )

    parser.add_argument(
        "modality",
        choices=[
            "rgb",
            "ir"
        ]
    )

    args = parser.parse_args()

    visualize_quality(
        args.pair_id,
        args.modality
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":

    main()