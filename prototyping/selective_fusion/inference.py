
"""
PACE: End-to-end selective local CrossFuse inference.

Pipeline:
    RGB + IR images
        -> Pretrained CrossFuse encoders
        -> Trained RGB/IR quality estimators
        -> Trained reliability gate
        -> Selective local CrossFuse
        -> Fused image

Run from repository root:

python prototyping/selective_fusion/inference.py \
    --split val \
    --limit 5 \
    --quality-source predicted \
    --gate-mode trained \
    --device cpu
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

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
    Load pretrained encoders, CrossFuse, quality estimators,
    reliability gate, and local fusion.

    Returns:
        encoders
        selective
        estimators
        scales
        gate_model

    selective.baseline is retained for compatibility with
    the validation scripts.
    """

    # --------------------------------------------------
    # Pretrained encoders
    # --------------------------------------------------

    encoders = {
        modality: load_encoder(modality, device).eval()
        for modality in ("rgb", "ir")
    }

    for encoder in encoders.values():
        encoder.requires_grad_(False)

    # --------------------------------------------------
    # Original pretrained CrossFuse
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Local CrossFuse
    # --------------------------------------------------

    local_model = LocalCrossFuse(
        rgb_encoder=encoders["rgb"],
        ir_encoder=encoders["ir"],
        baseline=baseline,
        grid_size=8,
        context_features=context_features,
        batch_size=batch_size,
    ).to(device).eval()

    # Keep .baseline accessible to existing validation code.
    # LocalCrossFuse already exposes self.baseline.
    selective = local_model

    # --------------------------------------------------
    # Trained quality estimators
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Trained reliability gate
    # --------------------------------------------------

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


def load_image(path, device):
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
            for p in (rgb_path, ir_path, ir_qpath)
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
    """
    Save an image normalized to [0,1].
    """
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
        raise RuntimeError(f"Failed to save {path}")


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
):
    """
    Execute the complete selective fusion pipeline.

    Encoder features are computed once and reused by
    quality estimation and local fusion.
    """

    # --------------------------------------------------
    # Encode both modalities once
    # --------------------------------------------------

    rgb_sh, rgb_de = extract_features(
        encoders["rgb"],
        rgb * 255.0,
    )

    ir_sh, ir_de = extract_features(
        encoders["ir"],
        ir * 255.0,
    )

    # --------------------------------------------------
    # Estimate quality
    # --------------------------------------------------

    if quality_source == "ground_truth":

        if rgb_qpath is None or ir_qpath is None:
            raise ValueError(
                "Ground-truth quality paths are required"
            )

        rgb_q = load_quality(rgb_qpath, rgb.device)
        ir_q = load_quality(ir_qpath, ir.device)

    else:

        rgb_q = predict_quality(
            estimators["rgb"],
            scales["rgb"],
            rgb,
            rgb_de,
        )

        ir_q = predict_quality(
            estimators["ir"],
            scales["ir"],
            ir,
            ir_de,
        )

    # --------------------------------------------------
    # Reliability gate
    # --------------------------------------------------

    if gate_mode == "all_on":
        gate = torch.ones_like(rgb_q)

    elif gate_mode == "all_off":
        gate = torch.zeros_like(rgb_q)

    else:
        if gate_model is None:
            raise ValueError(
                "Trained gate model was not loaded"
            )

        gate = gate_model(rgb_q, ir_q)

    # --------------------------------------------------
    # Selective local CrossFuse
    # --------------------------------------------------

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

            rgb = load_image(rgb_path, device)
            ir = load_image(ir_path, device)

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
                args.output / f"{prefix}_gate.npy",
                gate[0, 0].cpu().numpy(),
            )

            np.save(
                args.output / f"{prefix}_rgb_quality.npy",
                rgb_q[0, 0].cpu().numpy(),
            )

            np.save(
                args.output / f"{prefix}_ir_quality.npy",
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
