
"""
PACE: End-to-end selective local CrossFuse inference.

Quality estimation uses original-resolution images, matching
the quality estimator's training and visualization pipeline.

Selective fusion continues to use 256x256 images.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
CF = ROOT / "baseline" / "CrossFuse"

for folder in (
    CF,
    ROOT / "prototyping" / "encoder",
    ROOT / "prototyping" / "quality_estimator",
    ROOT / "prototyping" / "reliability_gate",
):
    sys.path.insert(0, str(folder))

from encoder import load_encoder, extract_features
from model import SpatialQualityEstimator
from degradation_features import degradation_features
from reliability_gate import ReliabilityGate
from network.net_conv_trans import Trans_FuseNet
from local_crossfuse import LocalCrossFuse


FUSION_CONFIG = dict(
    img_size=32,
    patch_size=2,
    en_out_channels1=32,
    out_channels=1,
    part_out=128,
    train_flag=False,
    depth_self=1,
    depth_cross=1,
    n_heads=16,
    mlp_ratio=4,
    qkv_bias=True,
    p=0.0,
    attn_p=0.0,
    num_classes=5,
)


def choose_device(request):
    if request != "auto":
        device = torch.device(request)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def state(path, device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}"
        )

    obj = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"], obj

    return obj, {}


def load_models(
    device,
    quality_source="predicted",
    gate_mode="trained",
    context_features=8,
    batch_size=16,
):
    """
    Returns:
        encoders
        selective
        estimators
        scales
        gate_model
    """

    encoders = {
        modality: load_encoder(modality, device).eval()
        for modality in ("rgb", "ir")
    }

    for encoder in encoders.values():
        encoder.requires_grad_(False)

    baseline = Trans_FuseNet(**FUSION_CONFIG)

    checkpoint = (
        CF
        / "models"
        / "transfuse"
        / "fusetrans_epoch_28.model"
    )

    weights, _ = state(checkpoint, device)

    baseline.load_state_dict(
        weights,
        strict=True,
    )

    baseline = (
        baseline.to(device)
        .eval()
        .requires_grad_(False)
    )

    local_model = LocalCrossFuse(
        rgb_encoder=encoders["rgb"],
        ir_encoder=encoders["ir"],
        baseline=baseline,
        grid_size=8,
        context_features=context_features,
        batch_size=batch_size,
    ).to(device).eval()

    selective = local_model

    estimators = {}
    scales = {}

    if quality_source == "predicted":
        for modality in ("rgb", "ir"):
            path = (
                ROOT
                / "prototyping"
                / "quality_estimator"
                / "checkpoints"
                / f"quality_{modality}.pt"
            )

            weights, metadata = state(path, device)

            estimator = SpatialQualityEstimator(
                in_channels=int(
                    metadata.get("in_channels", 132)
                )
            )

            estimator.load_state_dict(
                weights,
                strict=True,
            )

            estimators[modality] = (
                estimator.to(device)
                .eval()
                .requires_grad_(False)
            )

            scales[modality] = float(
                metadata.get("extra_feature_scale", 10.0)
            )

            print(
                f"{modality.upper()} estimator: {path} "
                f"| epoch={metadata.get('epoch', '?')} "
                f"| scale={scales[modality]}"
            )

    gate_model = None

    if gate_mode == "trained":
        path = (
            ROOT
            / "prototyping"
            / "reliability_gate"
            / "checkpoints"
            / "reliability_gate_best.pt"
        )

        weights, metadata = state(path, device)

        gate_model = ReliabilityGate(
            hidden_dim=int(
                metadata.get("hidden_dim", 16)
            )
        )

        gate_model.load_state_dict(
            weights,
            strict=True,
        )

        gate_model = (
            gate_model.to(device)
            .eval()
            .requires_grad_(False)
        )

    return (
        encoders,
        selective,
        estimators,
        scales,
        gate_model,
    )


def load_quality_image(path, modality, device):
    """
    Match QualityImageDataset and the working notebook.

    RGB: read color, then BGR -> grayscale.
    IR: read grayscale directly.

    IMPORTANT: no resizing.

    Returns [1, 1, H, W] in [0, 1].
    """
    read_mode = (
        cv2.IMREAD_GRAYSCALE
        if modality == "ir"
        else cv2.IMREAD_COLOR
    )

    image = cv2.imread(str(path), read_mode)

    if image is None:
        raise FileNotFoundError(path)

    if modality == "rgb":
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    image = image.astype(np.float32) / 255.0

    return (
        torch.from_numpy(image)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )


def load_image(path, device):
    """
    Load an image for local CrossFuse fusion.

    Returns [1, 1, 256, 256] in [0, 1].
    """
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(path)

    image = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_AREA,
    )

    return (
        torch.from_numpy(image.copy())
        .to(device)
        .float()[None, None]
        / 255.0
    )


def load_quality(path, device):
    q = np.load(path).astype(np.float32)

    if q.shape != (8, 8):
        raise ValueError(
            f"Invalid quality map shape: {q.shape}"
        )

    if (
        not np.isfinite(q).all()
        or q.min() < 0
        or q.max() > 1
    ):
        raise ValueError(
            f"Invalid quality values: {path}"
        )

    return (
        torch.from_numpy(q.copy())
        .to(device)[None, None]
    )


def predict_quality(
    estimator,
    scale,
    image_01,
    deep_features,
):
    """
    Match quality_estimator/train.py:

    128-channel CrossFuse deep features
    + four degradation-aware features
    = 132 channels.
    """
    extra = degradation_features(image_01)

    extra = F.adaptive_avg_pool2d(
        extra,
        deep_features.shape[-2:],
    ) * scale

    features = torch.cat(
        (
            deep_features.float(),
            extra.float(),
        ),
        dim=1,
    )

    return estimator(features)


@torch.inference_mode()
def predict_quality_from_deep_features(
    deep_features,
    estimator, 
    scale, 
    image_01,
):
    """Predict quality using precomputed encoder features."""
    return predict_quality(
        estimator, 
        scale, 
        image_01, 
        deep_features
    )

@torch.inference_mode()
def predict_quality_from_original(
    encoder,
    estimator,
    scale,
    image_01,
):
    """
    Predict using original-resolution image features,
    exactly as in the working quality notebook.
    """
    _, deep = extract_features(
        encoder,
        image_01 * 255.0,
    )

    return predict_quality(
        estimator,
        scale,
        image_01,
        deep,
    )


def samples(split, limit):
    root = ROOT / "processed_dataset" / split
    quality_dir = root / "quality_maps"

    found = []

    for rgb_qpath in sorted(
        quality_dir.glob("*_rgb.npy")
    ):
        stem = rgb_qpath.stem.removesuffix("_rgb")

        rgb_path = root / "rgb" / f"{stem}.jpg"
        ir_path = root / "ir" / f"{stem}.jpg"
        ir_qpath = quality_dir / f"{stem}_ir.npy"

        if all(
            p.is_file()
            for p in (
                rgb_path,
                ir_path,
                ir_qpath,
            )
        ):
            found.append(
                (
                    stem,
                    rgb_path,
                    ir_path,
                    rgb_qpath,
                    ir_qpath,
                )
            )

            if (
                limit is not None
                and len(found) >= limit
            ):
                break

    if not found:
        raise RuntimeError(
            f"No complete image pairs under {root}"
        )

    return found


def save_image(tensor, path):
    image = (
        tensor[0, 0]
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .cpu()
        .numpy()
    )

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(
            f"Failed to save {path}"
        )


@torch.inference_mode()
def run_pipeline(
    encoders,
    local_model,
    estimators,
    scales,
    gate_model,
    rgb,
    ir,
    quality_source="predicted",
    gate_mode="trained",
    rgb_qpath=None,
    ir_qpath=None,
    rgb_original=None,
    ir_original=None,
):
    """
    rgb, ir:
        256x256 tensors used for fusion.

    rgb_original, ir_original:
        Native-resolution tensors used for quality estimation.
    """

    # Fusion-resolution encoder features.
    rgb_sh, rgb_de = extract_features(
        encoders["rgb"],
        rgb * 255.0,
    )

    ir_sh, ir_de = extract_features(
        encoders["ir"],
        ir * 255.0,
    )

    if quality_source == "ground_truth":
        if rgb_qpath is None or ir_qpath is None:
            raise ValueError(
                "Ground-truth quality paths are required"
            )

        rgb_q = load_quality(
            rgb_qpath,
            rgb.device,
        )

        ir_q = load_quality(
            ir_qpath,
            ir.device,
        )

    elif quality_source == "predicted":
        if rgb_original is None or ir_original is None:
            raise ValueError(
                "Predicted quality requires original-resolution "
                "RGB and IR images."
            )

        # Native-resolution feature extraction.
        rgb_q = predict_quality_from_deep_features(
            rgb_de,
            estimators["rgb"],
            scales["rgb"],
            rgb
        )

        ir_q = predict_quality_from_deep_features(
            ir_de,
            estimators["ir"],
            scales["ir"],
            ir
        )

    else:
        raise ValueError(
            f"Unknown quality source: {quality_source}"
        )

    if gate_mode == "all_on":
        gate = torch.ones_like(rgb_q)

    elif gate_mode == "all_off":
        gate = torch.zeros_like(rgb_q)

    elif gate_mode == "trained":
        if gate_model is None:
            raise ValueError(
                "Trained gate model was not loaded"
            )

        gate = gate_model(rgb_q, ir_q)

    else:
        raise ValueError(
            f"Unknown gate mode: {gate_mode}"
        )

    output = local_model(
        rgb=rgb,
        ir=ir,
        rgb_quality=rgb_q,
        ir_quality=ir_q,
        gate=gate,
        encoded_features=(
            rgb_sh,
            rgb_de,
            ir_sh,
            ir_de,
        ),
    )

    return output, rgb_q, ir_q, gate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--quality-source",
        choices=("predicted", "ground_truth"),
        default="predicted",
    )

    parser.add_argument(
        "--gate-mode",
        choices=("trained", "all_on", "all_off"),
        default="trained",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--context-features",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "prototyping"
            / "selective_fusion"
            / "outputs"
        ),
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    device = choose_device(args.device)

    print(f"Device: {device}")
    print(f"Quality source: {args.quality_source}")
    print(f"Gate mode: {args.gate_mode}")

    (
        encoders,
        local_model,
        estimators,
        scales,
        gate_model,
    ) = load_models(
        device,
        quality_source=args.quality_source,
        gate_mode=args.gate_mode,
        context_features=args.context_features,
        batch_size=args.batch_size,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    with torch.inference_mode():
        for (
            stem,
            rgb_path,
            ir_path,
            rgb_qpath,
            ir_qpath,
        ) in samples(args.split, args.limit):

            # 256x256 fusion inputs.
            rgb = load_image(rgb_path, device)
            ir = load_image(ir_path, device)

            # Original-resolution quality inputs.
            rgb_original = None
            ir_original = None

            if args.quality_source == "predicted":
                rgb_original = load_quality_image(
                    rgb_path,
                    "rgb",
                    device,
                )

                ir_original = load_quality_image(
                    ir_path,
                    "ir",
                    device,
                )

            output, rgb_q, ir_q, gate = run_pipeline(
                encoders=encoders,
                local_model=local_model,
                estimators=estimators,
                scales=scales,
                gate_model=gate_model,
                rgb=rgb,
                ir=ir,
                quality_source=args.quality_source,
                gate_mode=args.gate_mode,
                rgb_qpath=rgb_qpath,
                ir_qpath=ir_qpath,
                rgb_original=rgb_original,
                ir_original=ir_original,
            )

            prefix = f"{args.split}_{stem}"

            image_path = (
                args.output
                / f"{prefix}_selective.png"
            )

            save_image(
                output["out"],
                image_path,
            )

            np.save(
                args.output
                / f"{prefix}_gate.npy",
                gate[0, 0].cpu().numpy(),
            )

            np.save(
                args.output
                / f"{prefix}_rgb_quality.npy",
                rgb_q[0, 0].cpu().numpy(),
            )

            np.save(
                args.output
                / f"{prefix}_ir_quality.npy",
                ir_q[0, 0].cpu().numpy(),
            )

            fusion_rate = float(
                gate.mean().item()
            )

            results.append({
                "sample": prefix,
                "fusion_rate": fusion_rate,
                "quality_source": args.quality_source,
                "gate_mode": args.gate_mode,
            })

            print(
                f"{prefix}: "
                f"{int(gate.sum().item())}/64 "
                f"patches fused "
                f"({fusion_rate:.1%})"
            )

            print(
                f"  RGB quality: "
                f"min={rgb_q.min().item():.3f}, "
                f"mean={rgb_q.mean().item():.3f}, "
                f"max={rgb_q.max().item():.3f}"
            )

            print(
                f"  IR quality:  "
                f"min={ir_q.min().item():.3f}, "
                f"mean={ir_q.mean().item():.3f}, "
                f"max={ir_q.max().item():.3f}"
            )

    summary_path = args.output / "summary.csv"

    with summary_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample",
                "fusion_rate",
                "quality_source",
                "gate_mode",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"Saved {len(results)} images to {args.output}"
    )


if __name__ == "__main__":
    main()
