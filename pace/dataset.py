# pace/dataset.py

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DroneVehicleDataset(Dataset):
    """
    Dataset for paired RGB/IR images and patch-quality maps.

    Expected layout:

        root/
            train/
                rgb/
                ir/
                rgb_annotations/
                ir_annotations/
                quality_maps/
            val/
                ...
            test/
                ...
    """

    def __init__(self, root, split="train"):
        self.root = Path(root) / split

        self.rgb_dir = self.root / "rgb"
        self.ir_dir = self.root / "ir"
        self.quality_dir = self.root / "quality_maps"

        self.ids = sorted(
            path.stem for path in self.rgb_dir.glob("*.jpg")
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]

        rgb_path = self.rgb_dir / f"{image_id}.jpg"
        ir_path = self.ir_dir / f"{image_id}.jpg"

        rgb_quality_path = (
            self.quality_dir / f"{image_id}_rgb.npy"
        )
        ir_quality_path = (
            self.quality_dir / f"{image_id}_ir.npy"
        )

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)

        if rgb is None:
            raise FileNotFoundError(rgb_path)

        if ir is None:
            raise FileNotFoundError(ir_path)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        rgb = torch.from_numpy(rgb).float() / 255.0
        ir = torch.from_numpy(ir).float() / 255.0

        rgb = rgb.permute(2, 0, 1)
        ir = ir.unsqueeze(0)

        rgb_quality = np.load(rgb_quality_path)
        ir_quality = np.load(ir_quality_path)

        rgb_quality = torch.from_numpy(
            rgb_quality
        ).float()

        ir_quality = torch.from_numpy(
            ir_quality
        ).float()

        return {
            "id": image_id,
            "rgb": rgb,
            "ir": ir,
            "rgb_quality": rgb_quality,
            "ir_quality": ir_quality,
        }