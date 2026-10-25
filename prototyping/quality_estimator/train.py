
"""
Train RGB and IR spatial quality estimators with online feature extraction.

For each batch:
    1. Extract frozen CrossFuse deep features (128 channels).
    2. Compute four fixed degradation-aware feature maps.
    3. Resize, scale, and concatenate (132 channels).
    4. Train the spatial quality estimator.

Includes:
    - Weighted MSE for degraded patches
    - Balanced sampling of clean/degraded images
    - Validation metrics by patch-quality category
    - ReduceLROnPlateau learning-rate scheduler
    - Best-validation checkpoint saving

Run from PACE repository root:
    python prototyping/quality_estimator/train.py
"""

from pathlib import Path
import random
import sys
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
)

from model import SpatialQualityEstimator
from degradation_features import degradation_features


# ---------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALITY_ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0, str(REPO_ROOT / "prototyping" / "encoder")
)

from encoder import (
    load_encoder,
    extract_features as extract_crossfuse_features,
)

DATASET_ROOT = REPO_ROOT / "processed_dataset"
CHECKPOINT_ROOT = QUALITY_ROOT / "checkpoints"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODALITIES = ("rgb", "ir")

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-6
LR_PATIENCE = 3
LR_FACTOR = 0.5

RANDOM_SEED = 42

LOSS_WEIGHT_STRENGTH = 4.0
DEGRADED_THRESHOLD = 0.8
DEGRADED_SAMPLE_FRACTION = 0.5

EXTRA_FEATURE_SCALE = 10.0

# None = use all available images.
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class QualityImageDataset(Dataset):
    """
    Load original images and ground-truth quality maps.

    Images:
        processed_dataset/{split}/{modality}/{id}.jpg

    Labels:
        processed_dataset/{split}/quality_maps/
            {id}_{modality}.npy

    Returns:
        image:   [1, H, W], float32 in [0, 1]
        quality: [1, 8, 8]
    """

    def __init__(self, modality, split, max_samples=None):
        self.modality = modality
        self.split = split

        image_dir = DATASET_ROOT / split / modality
        quality_dir = DATASET_ROOT / split / "quality_maps"

        if not image_dir.exists():
            raise FileNotFoundError(image_dir)

        if not quality_dir.exists():
            raise FileNotFoundError(quality_dir)

        self.samples = []
        self.is_degraded = []

        for label_path in sorted(
            quality_dir.glob(f"*_{modality}.npy")
        ):
            sample_id = label_path.stem.removesuffix(
                f"_{modality}"
            )
            image_path = image_dir / f"{sample_id}.jpg"

            if not image_path.exists():
                continue

            quality = np.load(label_path)

            if quality.shape != (8, 8):
                raise ValueError(
                    f"Unexpected label shape: {label_path}"
                )

            self.samples.append((image_path, label_path))
            self.is_degraded.append(
                bool(np.any(quality < DEGRADED_THRESHOLD))
            )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
            self.is_degraded = self.is_degraded[:max_samples]

        if not self.samples:
            raise ValueError(
                f"No paired samples for {modality}/{split}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        # Match the successful experiment's preprocessing.
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
            if self.modality == "ir"
            else cv2.IMREAD_COLOR,
        )

        if image is None:
            raise FileNotFoundError(image_path)

        if self.modality == "rgb":
            image = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

        image = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).unsqueeze(0)

        quality = torch.from_numpy(
            np.load(label_path).astype(np.float32)
        ).unsqueeze(0)

        return image, quality


# ---------------------------------------------------------
# Online feature extraction
# ---------------------------------------------------------

def extract_quality_features(encoder, images):
    """
    Combine CrossFuse and degradation-aware features.

    Args:
        encoder: frozen CrossFuse encoder
        images: [B, 1, H, W], values [0, 1]

    Returns:
        [B, 132, 64, 80] for current image dimensions.
    """

    with torch.no_grad():
        # CrossFuse expects values in [0, 255].
        _, deep = extract_crossfuse_features(
            encoder, images * 255.0
        )

        # Four fixed image-statistic feature maps.
        extra = degradation_features(images)

        # Match the CrossFuse feature resolution.
        extra = F.adaptive_avg_pool2d(
            extra, deep.shape[-2:]
        )

        # Preserve scaling from the successful experiment.
        extra = extra * EXTRA_FEATURE_SCALE

        return torch.cat(
            [deep.float(), extra.float()],
            dim=1,
        )


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_mse(predictions, targets):
    weights = (
        1.0 + LOSS_WEIGHT_STRENGTH * (1.0 - targets)
    )

    squared_error = (predictions - targets) ** 2

    return (
        (weights * squared_error).sum()
        / weights.sum()
    )


def create_balanced_sampler(dataset):
    degraded = np.asarray(
        dataset.is_degraded, dtype=bool
    )

    n_degraded = int(degraded.sum())
    n_clean = len(degraded) - n_degraded

    print(f"Clean images:    {n_clean:,}")
    print(f"Degraded images: {n_degraded:,}")

    if n_degraded == 0 or n_clean == 0:
        print(
            "Only one image category found; "
            "using uniform sampling."
        )
        weights = np.ones(
            len(dataset), dtype=np.float64
        )
    else:
        weights = np.where(
            degraded,
            DEGRADED_SAMPLE_FRACTION / n_degraded,
            (1.0 - DEGRADED_SAMPLE_FRACTION) / n_clean,
        )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    return WeightedRandomSampler(
        weights=torch.as_tensor(
            weights, dtype=torch.double
        ),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


# ---------------------------------------------------------
# Training / validation
# ---------------------------------------------------------

def run_epoch(
    model,
    encoder,
    loader,
    device,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    encoder.eval()

    total_weighted_error = 0.0
    total_weight = 0.0
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    category_error = {
        "clean": 0.0,
        "degraded": 0.0,
        "severe": 0.0,
    }
    category_count = {
        "clean": 0,
        "degraded": 0,
        "severe": 0,
    }

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            # Recompute features for every batch.
            features = extract_quality_features(
                encoder, images
            )

            predictions = model(features)
            loss = weighted_mse(predictions, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                weights = (
                    1.0
                    + LOSS_WEIGHT_STRENGTH
                    * (1.0 - targets)
                )

                squared_error = (
                    predictions - targets
                ) ** 2

                absolute_error = (
                    predictions - targets
                ).abs()

                total_weighted_error += (
                    (weights * squared_error).sum().item()
                )
                total_weight += weights.sum().item()

                total_squared_error += (
                    squared_error.sum().item()
                )
                total_absolute_error += (
                    absolute_error.sum().item()
                )
                total_elements += targets.numel()

                masks = {
                    "clean": targets >= 0.95,
                    "degraded": targets < 0.8,
                    "severe": targets < 0.5,
                }

                for category, mask in masks.items():
                    category_error[category] += (
                        absolute_error[mask].sum().item()
                    )
                    category_count[category] += (
                        mask.sum().item()
                    )

    metrics = {
        "weighted_mse": (
            total_weighted_error / total_weight
        ),
        "mse": (
            total_squared_error / total_elements
        ),
        "mae": (
            total_absolute_error / total_elements
        ),
    }

    for category in category_error:
        count = category_count[category]

        metrics[f"{category}_mae"] = (
            category_error[category] / count
            if count > 0
            else float("nan")
        )
        metrics[f"{category}_count"] = count

    return metrics


# ---------------------------------------------------------
# Train one modality
# ---------------------------------------------------------

def train_modality(modality, device):
    print(f"\n{'=' * 55}")
    print(f"Training {modality.upper()} quality estimator")
    print(f"{'=' * 55}")

    train_dataset = QualityImageDataset(
        modality,
        "train",
        MAX_TRAIN_SAMPLES,
    )
    val_dataset = QualityImageDataset(
        modality,
        "val",
        MAX_VAL_SAMPLES,
    )

    sampler = create_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Frozen pretrained CrossFuse encoder.
    encoder = load_encoder(modality, device)
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    # Determine channel count dynamically.
    sample_image, _ = train_dataset[0]

    with torch.no_grad():
        sample_features = extract_quality_features(
            encoder,
            sample_image.unsqueeze(0).to(device),
        )

    in_channels = sample_features.shape[1]

    model = SpatialQualityEstimator(
        in_channels=in_channels
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_ROOT / f"quality_{modality}.pt"
    )

    best_val_loss = float("inf")

    print(f"Train samples:  {len(train_dataset):,}")
    print(f"Val samples:    {len(val_dataset):,}")
    print(f"Input channels: {in_channels}")
    print(f"Device:         {device}")
    print(f"Initial LR:     {LEARNING_RATE:.2e}")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            model,
            encoder,
            train_loader,
            device,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model,
            encoder,
            val_loader,
            device,
        )

        # Reduce LR when validation MSE plateaus.
        scheduler.step(val_metrics["mse"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Weighted MSE: "
            f"{train_metrics['weighted_mse']:.5f} | "
            f"Val MSE: {val_metrics['mse']:.5f} | "
            f"Val MAE: {val_metrics['mae']:.5f} | "
            f"Degraded MAE: "
            f"{val_metrics['degraded_mae']:.5f} | "
            f"Severe MAE: "
            f"{val_metrics['severe_mae']:.5f}"
        )

        # Save checkpoint with best validation MSE.
        if val_metrics["mse"] < best_val_loss:
            best_val_loss = val_metrics["mse"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "modality": modality,
                    "in_channels": in_channels,
                    "epoch": epoch,
                    "val_mse": best_val_loss,
                    "val_mae": val_metrics["mae"],
                    "val_degraded_mae": (
                        val_metrics["degraded_mae"]
                    ),
                    "loss_weight_strength": (
                        LOSS_WEIGHT_STRENGTH
                    ),
                    "degraded_sample_fraction": (
                        DEGRADED_SAMPLE_FRACTION
                    ),
                    "extra_feature_scale": (
                        EXTRA_FEATURE_SCALE
                    ),
                    "learning_rate": current_lr,
                },
                checkpoint_path,
            )

    print(f"\nBest validation MSE: {best_val_loss:.5f}")
    print(f"Saved checkpoint: {checkpoint_path}")

    del encoder, model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    global MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES, EPOCHS
    parser = argparse.ArgumentParser(
        description="Train RGB and IR quality estimators."
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum training images per modality (default: all).",
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Maximum validation images per modality (default: all).",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )

    args = parser.parse_args()

    MAX_TRAIN_SAMPLES = args.max_train_samples
    MAX_VAL_SAMPLES = args.max_val_samples
    EPOCHS = args.epochs

    set_seed(RANDOM_SEED)

    device = get_device()
    print(f"Using device: {device}")

    for modality in MODALITIES:
        train_modality(modality, device)

    print("\nQuality estimator training complete.")


if __name__ == "__main__":
    main()
