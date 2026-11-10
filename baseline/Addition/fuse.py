#!/usr/bin/env python3
"""Generate a standalone 50/50 RGB–IR fused dataset."""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CROSSFUSE = ROOT / "baseline" / "CrossFuse"

sys.path.insert(0, str(CROSSFUSE))
from tools import utils


def sample_paths(dataset_root, split, maximum=0):
    paths = [
        Path(p)
        for p in utils.list_dronevehicle_images(
            str(dataset_root), "ir", (split,), None
        )
    ]
    if maximum > 0:
        paths = paths[:maximum]

    if not paths:
        raise RuntimeError(f"No IR images found for split {split}")

    pairs = []
    for ir_path in paths:
        rgb_path = dataset_root / split / "rgb" / ir_path.name
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        pairs.append((ir_path.stem, rgb_path, ir_path))

    return pairs


def read_gray(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)

    return cv2.resize(
        image, (size, size), interpolation=cv2.INTER_LINEAR
    )


def fuse_arrays(rgb, ir):
    """Round half-up to uint8, avoiding uint8 addition overflow."""
    if rgb.shape != ir.shape:
        raise ValueError("RGB and IR image dimensions must match")

    return (
        (rgb.astype(np.uint16) + ir.astype(np.uint16) + 1) // 2
    ).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "processed_dataset",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="0 = all images",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.max_samples < 0 or args.image_size <= 0:
        parser.error("Invalid max-samples or image-size")

    output = args.output or (
        HERE / f"fused_{args.split}"
    )
    output.mkdir(parents=True, exist_ok=True)

    pairs = sample_paths(
        args.dataset_root, args.split, args.max_samples
    )

    rows = []

    for index, (sample_id, rgb_path, ir_path) in enumerate(pairs):
        rgb = read_gray(rgb_path, args.image_size)
        ir = read_gray(ir_path, args.image_size)

        fused = fuse_arrays(rgb, ir)
        fused_path = output / f"{sample_id}.png"

        if not cv2.imwrite(str(fused_path), fused):
            raise IOError(f"Could not write {fused_path}")

        rows.append({
            "sample_id": sample_id,
            "rgb_path": str(rgb_path),
            "ir_path": str(ir_path),
            "fused_path": str(fused_path),
        })

        if (index + 1) % 100 == 0:
            print(f"Fused {index + 1}/{len(pairs)}")

    with (output / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(pairs)} fused images in {output}")


if __name__ == "__main__":
    main()