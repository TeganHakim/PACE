
from pathlib import Path
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import SpatialQualityEstimator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "processed_dataset"

sys.path.insert(0, str(REPO_ROOT / "prototyping" / "encoder"))
from encoder import load_encoder, extract_features


# ---------------- Configuration ----------------

MODALITY = "ir"             # "ir" or "rgb"
SPLIT = "train"

NUM_IMAGES = 500             # Try 10 first, then 500
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LOSS_WEIGHT_STRENGTH = 4.0
QUALITY_THRESHOLD = 0.5
SEED = 42

# True: train/test on same images (memorization test).
# False: hold out 20% for validation.
MEMORIZATION_TEST = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_images():
    quality_dir = DATASET_ROOT / SPLIT / "quality_maps"
    image_dir = DATASET_ROOT / SPLIT / MODALITY

    candidates = []

    for label_path in sorted(
        quality_dir.glob(f"*_{MODALITY}.npy")
    ):
        sample_id = label_path.stem.removesuffix(
            f"_{MODALITY}"
        )
        image_path = image_dir / f"{sample_id}.jpg"

        if not image_path.exists():
            continue

        quality = np.load(label_path)

        if np.any(quality < QUALITY_THRESHOLD):
            candidates.append((image_path, label_path))

    if len(candidates) < NUM_IMAGES:
        raise ValueError("Not enough degraded examples.")

    return random.Random(SEED).sample(
        candidates, NUM_IMAGES
    )


def load_image(path):
    if MODALITY == "ir":
        image = cv2.imread(
            str(path), cv2.IMREAD_GRAYSCALE
        )
    else:
        image = cv2.imread(
            str(path), cv2.IMREAD_COLOR
        )
        if image is not None:
            image = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

    if image is None:
        raise FileNotFoundError(path)

    return image.astype(np.float32) / 255.0


def degradation_features(image):
    """
    Compute four local image statistics.

    Input:  grayscale image [H, W], values [0, 1]
    Output: feature maps [1, 4, H, W]

    These are descriptive statistics, not degradation
    labels or ground-truth quality scores.
    """
    x = torch.from_numpy(image)[None, None]

    # Sobel gradient magnitude
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32
    ).reshape(1, 1, 3, 3) / 8.0

    sobel_y = sobel_x.transpose(-1, -2)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)

    gradient = torch.sqrt(gx**2 + gy**2 + 1e-8)

    # Laplacian magnitude: high-frequency content
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32
    ).reshape(1, 1, 3, 3)

    laplacian = F.conv2d(
        x, lap_kernel, padding=1
    ).abs()

    # Local intensity variance
    mean = F.avg_pool2d(
        x, kernel_size=7, stride=1, padding=3
    )
    mean_sq = F.avg_pool2d(
        x * x, kernel_size=7, stride=1, padding=3
    )
    variance = (mean_sq - mean**2).clamp_min(0)

    # High-frequency residual magnitude
    blurred = F.avg_pool2d(
        x, kernel_size=5, stride=1, padding=2
    )
    residual = (x - blurred).abs()

    return torch.cat(
        [gradient, laplacian, variance, residual],
        dim=1
    )


def build_dataset(selected, encoder, device):
    all_features = []
    all_targets = []
    sample_ids = []

    encoder.eval()

    for image_path, label_path in selected:
        image = load_image(image_path)

        # CrossFuse expects [0, 255].
        encoder_input = (
            torch.from_numpy(image)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device) * 255.0
        )

        with torch.no_grad():
            _, deep = extract_features(
                encoder, encoder_input
            )

        # Compute image statistics on CPU.
        extra = degradation_features(image)

        # Match CrossFuse deep-feature resolution.
        extra = F.adaptive_avg_pool2d(
            extra, deep.shape[-2:]
        ).to(device)

        # Scale feature channels using fixed factors.
        # Avoid image-specific min-max normalization,
        # which can erase absolute degradation cues.
        extra = extra * 10.0

        combined = torch.cat(
            [deep.float(), extra], dim=1
        )

        quality = np.load(
            label_path
        ).astype(np.float32).reshape(1, 8, 8)

        all_features.append(
            combined.squeeze(0).cpu()
        )
        all_targets.append(
            torch.from_numpy(quality)
        )
        sample_ids.append(image_path.stem)

    return (
        torch.stack(all_features),
        torch.stack(all_targets),
        sample_ids
    )


def weighted_mse(predictions, targets):
    weights = (
        1.0 + LOSS_WEIGHT_STRENGTH * (1.0 - targets)
    )
    return (
        weights * (predictions - targets)**2
    ).sum() / weights.sum()


def evaluate(model, loader, device):
    model.eval()

    total_error = 0.0
    total_cells = 0
    degraded_error = 0.0
    degraded_cells = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            error = (pred - y).abs()

            total_error += error.sum().item()
            total_cells += y.numel()

            mask = y < QUALITY_THRESHOLD

            degraded_error += error[mask].sum().item()
            degraded_cells += mask.sum().item()

    return (
        total_error / max(total_cells, 1),
        degraded_error / max(degraded_cells, 1)
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print(f"Device: {device}")

    selected = select_images()
    print(f"Selected {len(selected)} images")

    encoder = load_encoder(MODALITY, device)

    print("Extracting features in memory...")
    features, targets, sample_ids = build_dataset(
        selected, encoder, device
    )

    print("Feature shape:", tuple(features.shape))
    print("Target shape:", tuple(targets.shape))

    # Free encoder memory before training.
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()

    if MEMORIZATION_TEST:
        train_indices = list(range(len(features)))
        val_indices = train_indices
    else:
        indices = list(range(len(features)))
        random.Random(SEED).shuffle(indices)

        split = int(0.8 * len(indices))
        train_indices = indices[:split]
        val_indices = indices[split:]

    train_loader = DataLoader(
        TensorDataset(
            features[train_indices],
            targets[train_indices]
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            features[val_indices],
            targets[val_indices]
        ),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = SpatialQualityEstimator(
        in_channels=features.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = weighted_mse(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 5 == 0:
            mae, degraded_mae = evaluate(
                model, val_loader, device
            )

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"MAE: {mae:.4f} | "
                f"Degraded MAE: {degraded_mae:.4f}"
            )

    # Visualize up to five evaluation examples.
    model.eval()

    n = min(5, len(val_indices))
    fig, axes = plt.subplots(
        n, 2, figsize=(7, 2.7 * n),
        squeeze=False
    )

    with torch.no_grad():
        for row, idx in enumerate(val_indices[:n]):
            pred = model(
                features[idx:idx+1].to(device)
            )[0, 0].cpu().numpy()

            gt = targets[idx, 0].numpy()

            axes[row, 0].imshow(
                gt, cmap="RdYlGn", vmin=0, vmax=1
            )
            axes[row, 0].set_title(
                f"{sample_ids[idx]} | Ground truth"
            )

            axes[row, 1].imshow(
                pred, cmap="RdYlGn", vmin=0, vmax=1
            )
            axes[row, 1].set_title(
                f"Prediction | min={pred.min():.3f}"
            )

            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
