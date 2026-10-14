"""
load the frozen trained CrossFuse encoder, run the RGB/IR images through it, 
and save the relevant encoder feature maps to features/rgb/ and features/ir/. 
This is run once after encoder training.

(run once)
- Take X training images.
- Pass each through its respective frozen encoder.
- Obtain X grids of feature vectors.
- Save those grids to disk.
"""

"""
Extract and cache deep feature maps from frozen CrossFuse encoders.

Run from the PACE repository root:

    python prototyping/quality_estimator/extract_features.py --max-samples 10

For the complete dataset:

    python prototyping/quality_estimator/extract_features.py

Features are saved as:
    prototyping/quality_estimator/features/
        rgb/{train,val,test}/{id}.pt
        ir/{train,val,test}/{id}.pt

The existing ground-truth quality maps remain in processed_dataset.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALITY_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "prototyping" / "encoder"))

from pace.dataset import DroneVehicleDataset
from encoder import load_encoder, extract_features


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFAULT_DATASET_ROOT = REPO_ROOT / "processed_dataset"
DEFAULT_OUTPUT_ROOT = QUALITY_ROOT / "features"

SPLITS = ("train", "val", "test")
MODALITIES = ("rgb", "ir")


def get_device():
    """Choose an available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def preprocess_image(image, modality):
    """
    Convert dataset images into CrossFuse encoder inputs.

    DroneVehicleDataset returns:
        RGB: [B, 3, H, W], normalized to [0, 1]
        IR:  [B, 1, H, W], normalized to [0, 1]

    CrossFuse expects:
        [B, 1, H, W], pixel values [0, 255]

    CrossFuse internally divides the input by 255.
    """
    if modality == "rgb":
        # Standard RGB-to-grayscale luminance conversion.
        image = (
            0.299 * image[:, 0:1]
            + 0.587 * image[:, 1:2]
            + 0.114 * image[:, 2:3]
        )

    return image * 255.0


def save_features(features, output_path):
    """Save a feature map in half precision to reduce disk usage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove batch dimension: [1, C, H, W] -> [C, H, W]
    features = features.squeeze(0)

    # Save on CPU so files are portable across devices.
    features = features.detach().cpu().half().contiguous()

    torch.save(features, output_path)


def process_split(
    split,
    dataset_root,
    output_root,
    encoders,
    device,
    max_samples=None,
    overwrite=False,
):
    """Extract features for one dataset split."""

    dataset = DroneVehicleDataset(
        root=dataset_root,
        split=split,
    )

    if max_samples is not None:
        indices = range(min(max_samples, len(dataset)))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"\nProcessing {split}: {len(dataset)} samples")

    counts = {modality: 0 for modality in MODALITIES}

    for index, batch in enumerate(loader, start=1):
        sample_id = batch["id"][0]

        for modality in MODALITIES:
            output_path = (
                output_root / modality / split / f"{sample_id}.pt"
            )

            # Skip already-extracted features.
            if output_path.exists() and not overwrite:
                continue

            image = batch[modality]
            image = preprocess_image(image, modality)
            image = image.to(device)

            encoder = encoders[modality]

            # Frozen encoder: inference only.
            with torch.inference_mode():
                _, deep_features = extract_features(
                    encoder, image
                )

            if deep_features.ndim != 4:
                raise ValueError(
                    f"Unexpected feature shape: "
                    f"{deep_features.shape}"
                )

            if index == 1:
                print(
                    f"  {modality.upper()} feature shape: "
                    f"{tuple(deep_features.shape)}"
                )

            save_features(deep_features, output_path)
            counts[modality] += 1

        if index % 100 == 0 or index == len(dataset):
            print(f"  Completed {index}/{len(dataset)}")

    print(f"Finished {split}: {counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract CrossFuse deep features."
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum samples per split. Default: 10.",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    dataset_root = DEFAULT_DATASET_ROOT
    output_root = DEFAULT_OUTPUT_ROOT

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_root}"
        )

    print("Loading frozen CrossFuse encoders...")

    encoders = {
        modality: load_encoder(modality, device)
        for modality in MODALITIES
    }

    for split in SPLITS:
        process_split(
            split=split,
            dataset_root=dataset_root,
            output_root=output_root,
            encoders=encoders,
            device=device,
            max_samples=args.max_samples,
            overwrite=False,
        )

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
