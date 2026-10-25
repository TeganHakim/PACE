"""
Train the PACE quality estimator from original RGB/IR images.

Pipeline:
    image (0-255 grayscale)
        -> pretrained CrossFuse Auto_Encoder_single
        -> shallow + deep feature maps
        -> SpatialQualityEstimator
        -> predicted 8x8 quality map

Expected dataset layout:
    processed_dataset/
        train/
            rgb/
            rgb_annotations/
            ir/
            ir_annotations/
            quality_maps/
                00001_rgb.npy
                00001_ir.npy
        val/
            ... same layout ...

Run from the PACE repository root:
    python prototyping/quality_estimator/train.py
"""

from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from model_v2 import SpatialQualityEstimator


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALITY_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "processed_dataset"
CROSSFUSE_ROOT = REPO_ROOT / "baseline" / "CrossFuse"
CHECKPOINT_ROOT = QUALITY_ROOT / "checkpoints"

AUTOENCODER_PATHS = {
    "ir": (
        CROSSFUSE_ROOT / "models" / "autoencoder"
        / "auto_encoder_epoch_4_ir.model"
    ),
    "rgb": (
        CROSSFUSE_ROOT / "models" / "autoencoder"
        / "auto_encoder_epoch_4_rgb.model"
    ),
}

MODALITIES = ("rgb", "ir")
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

LOSS_WEIGHT_STRENGTH = 4.0
DEGRADED_THRESHOLD = 0.7
DEGRADED_SAMPLE_FRACTION = 0.5

IMAGE_SIZE = 256
NUM_WORKERS = 0

# Must match the configuration used to train the CrossFuse
# autoencoder checkpoints.
AUTOENCODER_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "en_out_channels1": 32,
    "en_out_channels": 64,
    "num_layers": 3,
    "dense_out": 128,
    "part_out": 128,
    "train_flag": True,
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


# ---------------------------------------------------------
# Import the original CrossFuse encoder
# ---------------------------------------------------------

if not CROSSFUSE_ROOT.exists():
    raise FileNotFoundError(
        f"CrossFuse repository not found: {CROSSFUSE_ROOT}"
    )

sys.path.insert(0, str(CROSSFUSE_ROOT))

try:
    from network.net_autoencoder import Auto_Encoder_single
except ImportError as exc:
    raise ImportError(
        "Could not import Auto_Encoder_single from "
        "baseline/CrossFuse/network/net_autoencoder.py. "
        "Check that the CrossFuse repository and its dependencies exist."
    ) from exc


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class QualityImageDataset(Dataset):
    """Load original images and their matching 8x8 quality maps."""

    def __init__(self, modality, split):
        self.modality = modality
        self.split = split
        self.image_dir = DATASET_ROOT / split / modality
        self.quality_dir = DATASET_ROOT / split / "quality_maps"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )
        if not self.quality_dir.is_dir():
            raise FileNotFoundError(
                f"Quality-map directory not found: {self.quality_dir}"
            )

        image_paths = sorted(
            path for path in self.image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        self.samples = []
        self.is_degraded = []

        for image_path in image_paths:
            image_id = image_path.stem
            modality_suffix = f"_{modality}"

            if image_id.lower().endswith(modality_suffix):
                image_id = image_id[:-len(modality_suffix)]

            label_path = (
                self.quality_dir / f"{image_id}_{modality}.npy"
            )

            if not label_path.exists():
                continue

            quality = np.load(label_path)
            if quality.shape != (8, 8):
                raise ValueError(
                    f"Expected an 8x8 quality map, got "
                    f"{quality.shape}: {label_path}"
                )

            self.samples.append((image_path, label_path))
            self.is_degraded.append(
                bool(np.any(quality < DEGRADED_THRESHOLD))
            )

        if not self.samples:
            raise ValueError(
                f"No matching image/quality-map pairs found for "
                f"{modality}/{split}.\n"
                f"Images: {self.image_dir}\n"
                f"Quality maps: {self.quality_dir}\n"
                "Expected names such as image 00001.png and label "
                "00001_rgb.npy or 00001_ir.npy."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        # CrossFuse's Auto_Encoder_single divides input by 255
        # internally, so preserve the image's 0-255 scale here.
        with Image.open(image_path) as image:
            image = image.convert("L")
            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR,
            )
            image_array = np.asarray(image, dtype=np.float32).copy()

        image_tensor = torch.from_numpy(image_array).unsqueeze(0)

        quality_array = np.load(label_path).astype(np.float32)
        quality_tensor = torch.from_numpy(
            quality_array
        ).unsqueeze(0)

        return image_tensor, quality_tensor


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class ShallowDeepQualityModel(torch.nn.Module):
    """
    Run an image through its frozen CrossFuse encoder, combine
    shallow and deep features, then predict its 8x8 quality map.
    """

    def __init__(self, modality):
        super().__init__()

        self.modality = modality
        self.encoder = Auto_Encoder_single(**AUTOENCODER_CONFIG)

        checkpoint_path = AUTOENCODER_PATHS[modality]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Encoder checkpoint not found: {checkpoint_path}"
            )

        # CrossFuse saves the model state_dict directly.
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.encoder.load_state_dict(state_dict, strict=True)
        self.encoder.eval()

        # The CrossFuse encoder returns shallow features with 32
        # channels and deep features with 128 channels.
        # Align their spatial sizes before concatenating them.
        self.quality_estimator = SpatialQualityEstimator(
            in_channels=32 + 128
        )

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        # Keep the pretrained encoder in eval mode while allowing
        # the quality estimator to train.
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            shallow_features, deep_features = self.encoder(images)

        if shallow_features.shape[-2:] != deep_features.shape[-2:]:
            shallow_features = F.adaptive_avg_pool2d(
                shallow_features,
                deep_features.shape[-2:],
            )

        combined_features = torch.cat(
            (shallow_features, deep_features),
            dim=1,
        )

        return self.quality_estimator(combined_features)


# ---------------------------------------------------------
# Device, reproducibility, loss, and sampling
# ---------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def weighted_mse(predictions, targets):
    """Prioritize degraded and severely degraded patches."""

    weights = torch.ones_like(targets)

    weights = torch.where(
        targets < 0.8,
        torch.full_like(targets, 5.0),
        weights,
    )

    weights = torch.where(
        targets < 0.5,
        torch.full_like(targets, 10.0),
        weights,
    )

    squared_error = (predictions - targets) ** 2

    return (weights * squared_error).sum() / weights.sum()


def create_balanced_sampler(dataset):
    """Sample images containing degraded patches more often."""
    degraded = np.asarray(dataset.is_degraded, dtype=bool)
    n_degraded = int(degraded.sum())
    n_clean = len(degraded) - n_degraded

    print(f"Clean images:    {n_clean:,}")
    print(f"Degraded images: {n_degraded:,}")

    if n_degraded == 0 or n_clean == 0:
        weights = np.ones(len(dataset), dtype=np.float64)
    else:
        weights = np.where(
            degraded,
            DEGRADED_SAMPLE_FRACTION / n_degraded,
            (1.0 - DEGRADED_SAMPLE_FRACTION) / n_clean,
        )

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


# ---------------------------------------------------------
# Training and validation
# ---------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0
    total_weighted_error = 0.0
    total_weights = 0.0

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

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            loss = weighted_mse(predictions, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            weights = 1.0 + LOSS_WEIGHT_STRENGTH * (1.0 - targets)
            squared_error = (predictions - targets) ** 2
            absolute_error = (predictions - targets).abs()

            total_weighted_error += (
                weights * squared_error
            ).sum().item()
            total_weights += weights.sum().item()
            total_squared_error += squared_error.sum().item()
            total_absolute_error += absolute_error.sum().item()
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
                category_count[category] += mask.sum().item()

    metrics = {
        "weighted_mse": (
            total_weighted_error / max(total_weights, 1.0)
        ),
        "mse": total_squared_error / max(total_elements, 1),
        "mae": total_absolute_error / max(total_elements, 1),
    }

    for category in category_error:
        count = category_count[category]
        metrics[f"{category}_mae"] = (
            category_error[category] / count
            if count
            else float("nan")
        )
        metrics[f"{category}_count"] = count

    return metrics


def train_modality(modality, device):
    print(f"\n{'=' * 58}")
    print(f"Training {modality.upper()} quality estimator")
    print(f"{'=' * 58}")

    train_dataset = QualityImageDataset(modality, TRAIN_SPLIT)
    val_dataset = QualityImageDataset(modality, VAL_SPLIT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=create_balanced_sampler(train_dataset),
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = ShallowDeepQualityModel(modality).to(device)

    optimizer = torch.optim.Adam(
        model.quality_estimator.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.97,
    )

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        CHECKPOINT_ROOT / f"quality_{modality}_shallow_deep.pt"
    )
    best_val_loss = float("inf")

    print(f"Encoder:       {AUTOENCODER_PATHS[modality]}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")
    print("Features:      shallow (32) + deep (128) = 160 channels")
    print(f"Device:        {device}\n")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Weighted MSE: {train_metrics['weighted_mse']:.5f} | "
            f"Val MSE: {val_metrics['mse']:.5f} | "
            f"Val MAE: {val_metrics['mae']:.5f}"
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Learning rate: {current_lr:.2e}")

        print(
            f"  Clean MAE: {val_metrics['clean_mae']:.5f} "
            f"(n={val_metrics['clean_count']:,}) | "
            f"Degraded MAE: {val_metrics['degraded_mae']:.5f} "
            f"(n={val_metrics['degraded_count']:,}) | "
            f"Severe MAE: {val_metrics['severe_mae']:.5f} "
            f"(n={val_metrics['severe_count']:,})"
        )

        if val_metrics["mse"] < best_val_loss:
            best_val_loss = val_metrics["mse"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "modality": modality,
                    "epoch": epoch,
                    "val_mse": best_val_loss,
                    "val_mae": val_metrics["mae"],
                    "val_degraded_mae": val_metrics["degraded_mae"],
                    "encoder_checkpoint": str(
                        AUTOENCODER_PATHS[modality]
                    ),
                    "input_channels": 160,
                    "image_size": IMAGE_SIZE,
                },
                checkpoint_path,
            )

    print(f"\nBest validation MSE: {best_val_loss:.5f}")
    print(f"Saved checkpoint: {checkpoint_path}")


def main():
    set_seed(RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")

    # for modality in MODALITIES:
    train_modality("ir", device)

    print("\nQuality estimator training complete.")


if __name__ == "__main__":
    main()
