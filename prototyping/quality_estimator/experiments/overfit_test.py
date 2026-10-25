
"""
Overfitting test using BOTH shallow and deep CrossFuse features.

- Select 10 heavily degraded training images.
- Extract shallow and deep features using the frozen CrossFuse encoder.
- Resize shallow features to match deep spatial dimensions.
- Concatenate features along the channel dimension.
- Train a fresh spatial quality estimator using weighted MSE.
- Evaluate on the SAME 10 images.

Run from repository root:

    python prototyping/quality_estimator/overfit_shallow_deep.py
"""

from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import SpatialQualityEstimator


# ---------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALITY_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "processed_dataset"

sys.path.insert(
    0,
    str(REPO_ROOT / "prototyping" / "encoder")
)

from encoder import load_encoder, extract_features


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODALITY = "ir"
SPLIT = "train"

NUM_IMAGES = 500
QUALITY_THRESHOLD = 0.5

EPOCHS = 100
LEARNING_RATE = 1e-3
LOSS_WEIGHT_STRENGTH = 4.0

RANDOM_SEED = 42


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------
# Weighted MSE
# ---------------------------------------------------------

def weighted_mse(predictions, targets):
    weights = 1.0 + LOSS_WEIGHT_STRENGTH * (1.0 - targets)

    squared_error = (predictions - targets) ** 2

    return (weights * squared_error).sum() / weights.sum()


# ---------------------------------------------------------
# Select degraded images
# ---------------------------------------------------------

def select_images():
    quality_dir = DATASET_ROOT / SPLIT / "quality_maps"
    image_dir = DATASET_ROOT / SPLIT / MODALITY

    candidates = []

    for quality_path in sorted(
        quality_dir.glob(f"*_{MODALITY}.npy")
    ):
        sample_id = quality_path.stem.removesuffix(
            f"_{MODALITY}"
        )

        image_path = image_dir / f"{sample_id}.jpg"

        if not image_path.exists():
            continue

        quality = np.load(quality_path)

        if np.any(quality < QUALITY_THRESHOLD):
            candidates.append(
                (image_path, quality_path)
            )

    if len(candidates) < NUM_IMAGES:
        raise ValueError(
            f"Only found {len(candidates)} eligible images."
        )

    rng = random.Random(RANDOM_SEED)

    selected = rng.sample(
        candidates,
        NUM_IMAGES
    )

    print(f"Found {len(candidates)} eligible images.")
    print(f"Selected {len(selected)} images.")

    return selected


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def load_image(image_path, device):
    """
    Match the preprocessing used by extract_features.py.

    CrossFuse expects a single-channel image in [0, 255].
    """

    if MODALITY == "ir":
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )
    else:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is not None:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            image = (
                0.299 * image[:, :, 0]
                + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]
            )

    if image is None:
        raise FileNotFoundError(image_path)

    image = np.asarray(image, dtype=np.float32)

    return (
        torch.from_numpy(image)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )


# ---------------------------------------------------------
# Extract shallow + deep features
# ---------------------------------------------------------

def load_data(selected, encoder, device):
    features = []
    targets = []
    sample_ids = []

    encoder.eval()

    with torch.no_grad():
        for image_path, quality_path in selected:
            image = load_image(image_path, device)

            shallow, deep = extract_features(
                encoder, image
            )

            # Match spatial resolution before concatenation.
            shallow = F.interpolate(
                shallow,
                size=deep.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            combined = torch.cat(
                [shallow, deep],
                dim=1
            )

            features.append(
                combined.squeeze(0).detach().cpu()
            )

            quality = np.load(
                quality_path
            ).astype(np.float32)

            targets.append(
                torch.from_numpy(quality).unsqueeze(0)
            )

            sample_ids.append(image_path.stem)

    features = torch.stack(features).to(device)
    targets = torch.stack(targets).to(device)

    return features, targets, sample_ids


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(predictions, targets):
    absolute_error = torch.abs(
        predictions - targets
    )

    degraded_mask = targets < QUALITY_THRESHOLD
    clean_mask = targets >= 0.95

    return {
        "mse": torch.mean(
            (predictions - targets) ** 2
        ).item(),

        "weighted_mse": weighted_mse(
            predictions, targets
        ).item(),

        "degraded_mae": (
            absolute_error[degraded_mask].mean().item()
            if degraded_mask.any().item()
            else float("nan")
        ),

        "clean_mae": (
            absolute_error[clean_mask].mean().item()
            if clean_mask.any().item()
            else float("nan")
        ),
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = get_device()

    print(f"Device: {device}")
    print(f"Modality: {MODALITY}")

    selected = select_images()

    print("Loading frozen CrossFuse encoder...")

    encoder = load_encoder(
        MODALITY,
        device
    )

    print("Extracting shallow + deep features...")

    features, targets, sample_ids = load_data(
        selected,
        encoder,
        device
    )

    print(f"Combined feature shape: {tuple(features.shape)}")
    print(f"Target shape: {tuple(targets.shape)}")
    print(f"Minimum GT quality: {targets.min().item():.4f}")

    model = SpatialQualityEstimator(
        in_channels=features.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    for epoch in range(1, EPOCHS + 1):
        model.train()

        predictions = model(features)
        loss = weighted_mse(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            model.eval()

            with torch.no_grad():
                updated_predictions = model(features)

                metrics = calculate_metrics(
                    updated_predictions,
                    targets
                )

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Weighted MSE: {metrics['weighted_mse']:.6f} | "
                f"Degraded MAE: {metrics['degraded_mae']:.6f} | "
                f"Clean MAE: {metrics['clean_mae']:.6f}"
            )

    # -----------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------

    model.eval()

    with torch.no_grad():
        predictions = model(features)

        metrics = calculate_metrics(
            predictions,
            targets
        )

    predictions = predictions.cpu().numpy()[:, 0]
    targets = targets.cpu().numpy()[:, 0]

    print("\n" + "=" * 55)
    print("FINAL RESULTS — SHALLOW + DEEP FEATURES")
    print("=" * 55)

    print(f"Weighted MSE: {metrics['weighted_mse']:.6f}")
    print(f"MSE:          {metrics['mse']:.6f}")
    print(f"Degraded MAE: {metrics['degraded_mae']:.6f}")
    print(f"Clean MAE:    {metrics['clean_mae']:.6f}")

    print("\nPer-image results:")

    for i, sample_id in enumerate(sample_ids):
        mask = targets[i] < QUALITY_THRESHOLD

        degraded_mae = np.mean(
            np.abs(
                predictions[i][mask] - targets[i][mask]
            )
        )

        print(
            f"{sample_id}: "
            f"GT min={targets[i].min():.3f}, "
            f"Pred min={predictions[i].min():.3f}, "
            f"Degraded MAE={degraded_mae:.4f}"
        )

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    num_display = min(5, NUM_IMAGES)

    fig, axes = plt.subplots(
        num_display,
        2,
        figsize=(8, 3 * num_display),
        squeeze=False,
    )

    for i in range(num_display):
        axes[i, 0].imshow(
            targets[i],
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        axes[i, 0].set_title(
            f"{sample_ids[i]} | Ground Truth"
        )

        axes[i, 1].imshow(
            predictions[i],
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        axes[i, 1].set_title(
            f"Prediction | min={predictions[i].min():.3f}"
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
