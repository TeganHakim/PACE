#!/usr/bin/env python3
"""
Run PACE on one paired RGB/IR sample from
processed_dataset_realistic_degraded_v2 and save visualizations.

Dataset selection matches evaluation/common.py selected_ids():
    - enumerate sorted test/ir/*.jpg
    - retain IDs with a matching test/rgb/*.jpg
    - use metadata.csv for degradation metadata

Run from the repository root:
    python pace/visualize_one_image_pace_realistic.py
    python pace/visualize_one_image_pace_realistic.py --sample-id 00315
    python pace/visualize_one_image_pace_realistic.py --sample-index 25

The quality estimators use the 256x256 pipeline and checkpoints:
    quality_rgb_256.pt
    quality_ir_256.pt

Outputs:
    pace/visualize_one_image_pace_realistic_degraded_v2/<sample-id>/

The model and detector operate on grayscale images. For visualization,
PACE's fused luminance is combined with the original RGB chrominance.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression


# ---------------------------------------------------------------------
# Repository paths and imports
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATASET_DEFAULT = REPO / "processed_dataset_realistic_degraded_v2"
CROSSFUSE_ROOT = REPO / "baseline" / "CrossFuse"
SELECTIVE_ROOT = REPO / "prototyping" / "selective_fusion"
ENCODER_ROOT = REPO / "prototyping" / "encoder"
QUALITY_ROOT = REPO / "prototyping" / "quality_estimator"
GATE_ROOT = REPO / "prototyping" / "reliability_gate"

for folder in (
    CROSSFUSE_ROOT,
    SELECTIVE_ROOT,
    ENCODER_ROOT,
    QUALITY_ROOT,
    GATE_ROOT,
    HERE,
):
    sys.path.insert(0, str(folder))

from inference import (
    FUSION_CONFIG,
    LocalCrossFuse,
    ReliabilityGate,
    SpatialQualityEstimator,
    Trans_FuseNet,
    choose_device,
    load_encoder,
    load_image,
    predict_quality_from_deep_features,
    predict_quality_from_original,
    load_quality_image,
)
from train_conv_trans import CLASS_TO_ID, ID_TO_CLASS, load_targets


# ---------------------------------------------------------------------
# Plot defaults
# ---------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
})


# ---------------------------------------------------------------------
# Arguments and sample selection
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_DEFAULT,
        help="Dataset root; defaults to processed_dataset_realistic_degraded_v2.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-features", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=(
            CROSSFUSE_ROOT
            / "models"
            / "transfuse"
            / "fusetrans_epoch_28.model"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            HERE
            / "visualize_one_image_pace_realistic_degraded_v2"
        ),
    )

    args = parser.parse_args()

    if args.sample_id is not None and args.sample_index is not None:
        parser.error("Use --sample-id or --sample-index, not both.")
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1.")
    if not 0 <= args.nms_confidence <= 1:
        parser.error("--nms-confidence must be between 0 and 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_metadata(dataset_root):
    path = dataset_root / "metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Expected metadata.csv, matching evaluation/eval_metrics.py: "
            + str(path)
        )

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    return {
        Path(row.get("pair_id", "")).stem: row
        for row in rows
        if row.get("pair_id")
    }


def paired_sample_ids(dataset_root, split):
    """
    Match evaluation/common.py:selected_ids exactly:
    sorted IR JPG stems with a corresponding RGB JPG.
    """
    split_root = dataset_root / split
    rgb_dir = split_root / "rgb"
    ir_dir = split_root / "ir"

    if not rgb_dir.is_dir() or not ir_dir.is_dir():
        raise FileNotFoundError(
            "Expected paired image directories: {} and {}".format(
                rgb_dir, ir_dir
            )
        )

    ids = sorted(
        path.stem
        for path in ir_dir.glob("*.jpg")
        if (rgb_dir / path.name).is_file()
    )

    if not ids:
        raise RuntimeError(
            "No paired RGB/IR JPG images found under {}".format(
                split_root
            )
        )

    return ids


def select_sample(args, ids):
    if args.sample_id is not None:
        requested = Path(args.sample_id).stem
        if requested not in ids:
            raise ValueError(
                "Sample ID {} not found in {}. Paired sample count: {}.".format(
                    requested, args.split, len(ids)
                )
            )
        return requested

    if args.sample_index is not None:
        if not 0 <= args.sample_index < len(ids):
            raise IndexError(
                "Sample index {} is out of range; {} contains {} paired samples.".format(
                    args.sample_index, args.split, len(ids)
                )
            )
        return ids[args.sample_index]

    return random.Random(args.seed).choice(ids)


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------

def safe_torch_load(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def unwrap_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_pace_models(device, context_features, batch_size):
    """
    Load the same frozen PACE components as inference.load_models(),
    but explicitly load the 256-trained quality checkpoints.
    """
    encoders = {
        modality: load_encoder(modality, device).eval()
        for modality in ("rgb", "ir")
    }

    for encoder in encoders.values():
        encoder.requires_grad_(False)

    # Load the trained Trans_FuseNet fusion baseline.
    baseline = Trans_FuseNet(**FUSION_CONFIG)
    baseline_path = (
        CROSSFUSE_ROOT
        / "models"
        / "transfuse"
        / "fusetrans_epoch_28.model"
    )
    baseline_checkpoint = safe_torch_load(baseline_path, device)
    baseline_state = unwrap_state(baseline_checkpoint)
    baseline.load_state_dict(baseline_state, strict=True)
    baseline = baseline.to(device).eval().requires_grad_(False)

    local_model = LocalCrossFuse(
        rgb_encoder=encoders["rgb"],
        ir_encoder=encoders["ir"],
        baseline=baseline,
        grid_size=8,
        context_features=context_features,
        batch_size=batch_size,
    ).to(device).eval()

    # Load both quality estimators trained with 256x256 inputs.
    estimators = {}
    scales = {}

    for modality in ("rgb", "ir"):
        checkpoint_path = (
            QUALITY_ROOT
            / "checkpoints"
            / "quality_{}.pt".format(modality)
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Missing {} quality checkpoint: {}".format(
                    modality.upper(), checkpoint_path
                )
            )

        checkpoint = safe_torch_load(checkpoint_path, device)
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Expected a checkpoint dictionary at {}".format(
                    checkpoint_path
                )
            )

        in_channels = int(checkpoint["in_channels"])
        estimator = SpatialQualityEstimator(
            in_channels=in_channels
        )
        estimator.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        estimators[modality] = (
            estimator.to(device).eval().requires_grad_(False)
        )
        scales[modality] = float(
            checkpoint.get("extra_feature_scale", 10.0)
        )

        print(
            "{} estimator: {} | epoch={} | input_channels={} | scale={}".format(
                modality.upper(),
                checkpoint_path,
                checkpoint.get("epoch", "?"),
                in_channels,
                scales[modality],
            )
        )

    # Load the trained reliability gate.
    gate_path = (
        GATE_ROOT / "checkpoints" / "reliability_gate_best.pt"
    )
    gate_checkpoint = safe_torch_load(gate_path, device)
    gate_state = unwrap_state(gate_checkpoint)
    hidden_dim = int(
        gate_checkpoint.get("hidden_dim", 16)
        if isinstance(gate_checkpoint, dict)
        else 16
    )

    gate_model = ReliabilityGate(hidden_dim=hidden_dim)
    gate_model.load_state_dict(gate_state, strict=True)
    gate_model = (
        gate_model.to(device).eval().requires_grad_(False)
    )

    return encoders, local_model, estimators, scales, gate_model


def load_detector(path, device):
    if not path.is_file():
        raise FileNotFoundError(
            "Detector checkpoint not found: {}".format(path)
        )

    checkpoint = safe_torch_load(path, device)
    state = unwrap_state(checkpoint)

    if not isinstance(state, dict):
        raise TypeError(
            "Expected a detector state_dict, got {}".format(type(state))
        )

    detector_state = {
        key[len("detector."):]: value
        for key, value in state.items()
        if key.startswith("detector.")
    }

    if not detector_state:
        raise RuntimeError(
            "No detector.* parameters found in {}".format(path)
        )

    detector = DetectionModel(
        "yolo11n.yaml",
        ch=1,
        nc=len(CLASS_TO_ID),
        verbose=False,
    ).to(device)

    detector.load_state_dict(detector_state, strict=True)
    return detector.eval().requires_grad_(False)


# ---------------------------------------------------------------------
# PACE inference
# ---------------------------------------------------------------------

@torch.inference_mode()
def prepare_sample(
    encoders,
    estimators,
    scales,
    gate_model,
    rgb,
    ir,
    rgb_original,
    ir_original
):
    """
    Both quality estimators were trained on 256x256 images.
    Encode the resized RGB/IR tensors and use those same deep features
    for quality prediction and fusion.
    """
    # rgb_shallow, rgb_deep = encoders["rgb"](rgb * 255.0)
    # ir_shallow, ir_deep = encoders["ir"](ir * 255.0)

    # rgb_quality = predict_quality_from_deep_features(
    #     rgb_deep,
    #     estimators["rgb"],
    #     scales["rgb"],
    #     rgb,
    # )
    # ir_quality = predict_quality_from_deep_features(
    #     ir_deep,
    #     estimators["ir"],
    #     scales["ir"],
    #     ir,
    # )

    # gate = gate_model(rgb_quality, ir_quality)
    rgb_shallow, rgb_deep = encoders['rgb'](rgb * 255.0)
    ir_shallow, ir_deep = encoders['ir'](ir * 255.0)
    rgb_quality = predict_quality_from_original(encoders['rgb'], estimators['rgb'], scales['rgb'], rgb_original)
    ir_quality = predict_quality_from_original(encoders['ir'], estimators['ir'], scales['ir'], ir_original)
    gate = gate_model(rgb_quality, ir_quality)

    expected_shape = (1, 1, 8, 8)
    for name, value in (
        ("RGB quality", rgb_quality),
        ("IR quality", ir_quality),
        ("Reliability gate", gate),
    ):
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                "{} has shape {}; expected {}.".format(
                    name, tuple(value.shape), expected_shape
                )
            )
        if not torch.isfinite(value).all():
            raise ValueError("{} contains NaN or Inf.".format(name))

    return {
        "rgb_quality": rgb_quality,
        "ir_quality": ir_quality,
        "gate": gate,
        "encoded_features": (
            rgb_shallow,
            rgb_deep,
            ir_shallow,
            ir_deep,
        ),
    }


@torch.inference_mode()
def run_pace(local_model, rgb, ir, prepared):
    output = local_model(
        rgb=rgb,
        ir=ir,
        rgb_quality=prepared["rgb_quality"],
        ir_quality=prepared["ir_quality"],
        gate=prepared["gate"],
        encoded_features=prepared["encoded_features"],
    )

    if not isinstance(output, dict) or "out" not in output:
        raise RuntimeError(
            "LocalCrossFuse must return a dictionary containing 'out'."
        )

    fused = output["out"]

    if not torch.is_tensor(fused):
        raise TypeError("LocalCrossFuse output['out'] must be a tensor.")
    if fused.ndim != 4 or fused.shape[1] != 1:
        raise ValueError(
            "Expected fused output [B,1,H,W], got {}.".format(
                tuple(fused.shape)
            )
        )
    if not torch.isfinite(fused).all():
        raise ValueError("Fused image contains NaN or Inf.")
    if fused.min().item() < -1e-4 or fused.max().item() > 1.0001:
        raise ValueError(
            "Expected fused output in [0,1], got [{:.4f}, {:.4f}].".format(
                fused.min().item(), fused.max().item()
            )
        )

    return fused.float()


@torch.inference_mode()
def detect(detector, image, nms_confidence, nms_iou):
    raw = detector(image)
    if isinstance(raw, (tuple, list)):
        raw = raw[0]

    result = non_max_suppression(
        raw,
        conf_thres=nms_confidence,
        iou_thres=nms_iou,
        nc=len(CLASS_TO_ID),
        max_det=300,
    )[0]

    return {
        "boxes": result[:, :4].detach().cpu().numpy(),
        "scores": result[:, 4].detach().cpu().numpy(),
        "labels": result[:, 5].long().detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------
# Conversion and drawing helpers
# ---------------------------------------------------------------------

def tensor_image(tensor):
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(
            "Expected [1,1,H,W], got {}".format(tuple(tensor.shape))
        )
    return (
        tensor.detach().float().cpu()[0, 0]
        .clamp(0, 1).numpy()
    )


def load_rgb_display(path, width, height):
    """
    Load the visible RGB image for visualization only.

    The PACE network still receives the existing grayscale tensor.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("Could not read RGB image: {}".format(path))

    bgr = cv2.resize(
        bgr, (width, height), interpolation=cv2.INTER_AREA
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def restore_rgb_chroma(rgb_image, fused_luminance):
    """
    Create a color visualization from PACE's grayscale fused output.

    PACE supplies the Y (luminance) channel. Cb/Cr chrominance comes
    from the original visible RGB image, following the color-output
    approach used by the original CrossFuse implementation.

    This does not change the tensor used by the detector.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(
            "Expected RGB display image [H,W,3], got {}".format(
                rgb_image.shape
            )
        )
    if fused_luminance.ndim != 2:
        raise ValueError(
            "Expected fused luminance [H,W], got {}".format(
                fused_luminance.shape
            )
        )
    if rgb_image.shape[:2] != fused_luminance.shape:
        raise ValueError(
            "RGB and fused dimensions differ: {} vs {}".format(
                rgb_image.shape[:2], fused_luminance.shape
            )
        )

    rgb_u8 = np.clip(
        np.rint(rgb_image * 255.0), 0, 255
    ).astype(np.uint8)
    ycrcb = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)

    ycrcb[:, :, 0] = np.clip(
        np.rint(fused_luminance * 255.0), 0, 255
    ).astype(np.uint8)

    fused_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return fused_rgb.astype(np.float32) / 255.0


def tensor_map(tensor, name):
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(
            "{} must be [1,1,h,w], got {}".format(
                name, tuple(tensor.shape)
            )
        )
    result = tensor.detach().float().cpu()[0, 0].numpy()
    if not np.isfinite(result).all():
        raise ValueError("{} contains NaN or Inf.".format(name))
    return result


def class_name(class_id):
    return ID_TO_CLASS.get(int(class_id), str(int(class_id)))


def draw_ground_truth(ax, targets, width, height):
    for target in targets:
        class_id, cx, cy, box_width, box_height = target[:5]
        x1 = (float(cx) - float(box_width) / 2) * width
        y1 = (float(cy) - float(box_height) / 2) * height
        rect_width = float(box_width) * width
        rect_height = float(box_height) * height

        ax.add_patch(Rectangle(
            (x1, y1),
            rect_width,
            rect_height,
            fill=False,
            edgecolor="#ffb000",
            linewidth=1.6,
            linestyle="--",
        ))
        ax.text(
            x1,
            max(2, y1 - 3),
            "GT: {}".format(class_name(class_id)),
            fontsize=6,
            color="black",
            bbox={
                "facecolor": "#ffb000",
                "alpha": 0.88,
                "edgecolor": "none",
                "pad": 1,
            },
        )


def draw_predictions(ax, prediction, confidence):
    for box, score, label in zip(
        prediction["boxes"],
        prediction["scores"],
        prediction["labels"],
    ):
        if float(score) < confidence:
            continue

        x1, y1, x2, y2 = [float(value) for value in box]
        ax.add_patch(Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="#00e676",
            linewidth=1.6,
        ))
        ax.text(
            x1,
            max(2, y1 - 3),
            "{} {:.2f}".format(class_name(label), float(score)),
            fontsize=6,
            color="black",
            bbox={
                "facecolor": "#00e676",
                "alpha": 0.88,
                "edgecolor": "none",
                "pad": 1,
            },
        )


def save_figure(fig, path):
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------

def plot_inputs_quality(rgb, ir, rgb_q, ir_q, sample_id, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].imshow(rgb, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("(a) RGB input")
    axes[0, 1].imshow(ir, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("(b) IR input")

    for ax, quality, title in (
        (axes[1, 0], rgb_q, "(c) RGB predicted quality"),
        (axes[1, 1], ir_q, "(d) IR predicted quality"),
    ):
        image = ax.imshow(
            quality, cmap="viridis", vmin=0, vmax=1
        )
        ax.set_title(title)
        ax.set_xlabel("Patch column")
        ax.set_ylabel("Patch row")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[0]:
        ax.set_axis_off()

    fig.suptitle(
        "PACE Inputs and Predicted Quality — {}".format(sample_id),
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout()
    save_figure(fig, out / "01_inputs_and_quality.png")


def plot_quality_overlays(rgb, ir, rgb_q, ir_q, sample_id, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, image, quality, title in (
        (axes[0], rgb, rgb_q, "RGB quality overlay"),
        (axes[1], ir, ir_q, "IR quality overlay"),
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        overlay = ax.imshow(
            quality,
            cmap="viridis",
            vmin=0,
            vmax=1,
            alpha=0.48,
            interpolation="nearest",
            extent=(-0.5, image.shape[1] - 0.5,
                    image.shape[0] - 0.5, -0.5),
        )
        ax.set_title(title)
        ax.set_axis_off()

    fig.colorbar(
        overlay, ax=axes, fraction=0.035, pad=0.025,
        label="Predicted quality",
    )
    fig.suptitle(
        "Spatial Modality Quality — {}".format(sample_id),
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout()
    save_figure(fig, out / "02_quality_overlays.png")


def plot_gate(gate, rgb, ir, sample_id, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    image = axes[0].imshow(
        gate, cmap="RdYlBu_r", vmin=0, vmax=1
    )
    axes[0].set_title("Predicted reliability gate")
    axes[0].set_xlabel("Patch column")
    axes[0].set_ylabel("Patch row")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    for ax, source, title in (
        (axes[1], rgb, "Gate over RGB"),
        (axes[2], ir, "Gate over IR"),
    ):
        ax.imshow(source, cmap="gray", vmin=0, vmax=1)
        ax.imshow(
            gate,
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
            alpha=0.42,
            interpolation="nearest",
            extent=(-0.5, source.shape[1] - 0.5,
                    source.shape[0] - 0.5, -0.5),
        )
        ax.set_title(title)
        ax.set_axis_off()

    fig.suptitle(
        "PACE Patch-Level Reliability Routing — {}".format(sample_id),
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout()
    save_figure(fig, out / "03_reliability_gate.png")


def plot_fusion(rgb, ir, fused, sample_id, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, image, title in (
        (axes[0], rgb, "(a) RGB input"),
        (axes[1], ir, "(b) IR input"),
        (axes[2], fused, "(c) PACE fused output"),
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_axis_off()

    fig.suptitle(
        "PACE Selective Multimodal Fusion — {}".format(sample_id),
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout()
    save_figure(fig, out / "04_fusion_output.png")


def plot_detection_comparison(
    rgb, ir, fused, predictions, targets, confidence, sample_id, out
):
    panels = (
        ("RGB-only", rgb, predictions["rgb"]),
        ("IR-only", ir, predictions["ir"]),
        ("PACE fused", fused, predictions["pace"]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    for ax, (title, image, prediction) in zip(axes, panels):
        height, width = image.shape[:2]
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(image, 0, 1))
        draw_ground_truth(ax, targets, width, height)
        draw_predictions(ax, prediction, confidence)
        count = int(
            np.count_nonzero(prediction["scores"] >= confidence)
        )
        ax.set_title("{} — {} detections".format(title, count))
        ax.set_axis_off()

    fig.legend(
        handles=[
            Patch(
                facecolor="none", edgecolor="#ffb000",
                linestyle="--", label="Ground truth",
            ),
            Patch(
                facecolor="none", edgecolor="#00e676",
                label="Predicted detection",
            ),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "RGB vs. IR vs. PACE Detection — {}\n"
        "Same YOLO11n detector; displayed confidence ≥ {:.2f}".format(
            sample_id, confidence
        ),
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout()
    save_figure(
        fig,
        out / "05_rgb_ir_pace_detection_comparison.png",
    )


def plot_pace_ground_truth(fused, prediction, targets, confidence, sample_id, out):
    height, width = fused.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    axes[0].imshow(np.clip(fused, 0, 1))
    draw_ground_truth(axes[0], targets, width, height)
    axes[0].set_title("Ground-truth annotations")

    axes[1].imshow(np.clip(fused, 0, 1))
    draw_predictions(axes[1], prediction, confidence)
    axes[1].set_title("PACE predicted detections")

    axes[2].imshow(np.clip(fused, 0, 1))
    draw_ground_truth(axes[2], targets, width, height)
    draw_predictions(axes[2], prediction, confidence)
    axes[2].set_title("Ground truth + PACE predictions")

    for ax in axes:
        ax.set_axis_off()

    fig.legend(
        handles=[
            Patch(
                facecolor="none", edgecolor="#ffb000",
                linestyle="--", label="Ground truth",
            ),
            Patch(
                facecolor="none", edgecolor="#00e676",
                label="PACE prediction",
            ),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "PACE Detection vs. Ground Truth — {}".format(sample_id),
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout()
    save_figure(fig, out / "06_ground_truth_vs_pace.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_root = args.dataset_root.resolve()
    device = choose_device(args.device)

    ids = paired_sample_ids(dataset_root, args.split)
    metadata = read_metadata(dataset_root)
    sample_id = select_sample(args, ids)
    record = metadata.get(sample_id, {})

    split_root = dataset_root / args.split
    rgb_path = split_root / "rgb" / "{}.jpg".format(sample_id)
    ir_path = split_root / "ir" / "{}.jpg".format(sample_id)
    rgb_xml = (
        split_root / "rgb_annotations" / "{}.xml".format(sample_id)
    )
    ir_xml = (
        split_root / "ir_annotations" / "{}.xml".format(sample_id)
    )

    for path in (rgb_path, ir_path, rgb_xml, ir_xml):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset: {}".format(dataset_root))
    print("Split:   {}".format(args.split))
    print("Pairs:   {}".format(len(ids)))
    print("Sample:  {}".format(sample_id))
    print("RGB:     {}".format(rgb_path))
    print("IR:      {}".format(ir_path))
    print("Device:  {}".format(device))
    print("Output:  {}".format(output_dir))
    if record:
        print(
            "Metadata: modality={} | degradation={} | severity={}".format(
                record.get("modality", ""),
                record.get("degradation", record.get("degradation_type", "")),
                record.get("severity", ""),
            )
        )

    encoders, local_model, estimators, scales, gate_model = (
        load_pace_models(
            device,
            args.context_features,
            args.batch_size,
        )
    )
    detector = load_detector(args.detector_weights, device)

    # Both modalities are loaded as grayscale 256x256 tensors in [0,1].
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
    rgb = load_image(rgb_path, device)
    ir = load_image(ir_path, device)

    if tuple(rgb.shape) != tuple(ir.shape):
        raise ValueError(
            "RGB and IR tensor shapes differ: {} vs {}".format(
                tuple(rgb.shape), tuple(ir.shape)
            )
        )

    targets = load_targets(
        str(ir_path),
        str(rgb_path),
        256,
        256,
    )

    with torch.inference_mode():
        prepared = prepare_sample(
            encoders,
            estimators,
            scales,
            gate_model,
            rgb,
            ir,
            rgb_original,
            ir_original

        )

        fused = run_pace(
            local_model,
            rgb,
            ir,
            prepared,
        )

        # Same detector for RGB-only, IR-only, and PACE.
        rgb_prediction = detect(
            detector, rgb.float(),
            args.nms_confidence, args.nms_iou,
        )
        ir_prediction = detect(
            detector, ir.float(),
            args.nms_confidence, args.nms_iou,
        )
        pace_prediction = detect(
            detector, fused.float(),
            args.nms_confidence, args.nms_iou,
        )

    # Keep the model/detector inputs grayscale, but retain the visible
    # RGB chrominance for color visualization of the fused result.
    ir_image = tensor_image(ir)
    fused_image = tensor_image(fused)
    rgb_image = load_rgb_display(
        rgb_path, width=rgb.shape[-1], height=rgb.shape[-2]
    )
    fused_color_image = restore_rgb_chroma(
        rgb_image, fused_image
    )
    rgb_quality = tensor_map(prepared["rgb_quality"], "RGB quality")
    ir_quality = tensor_map(prepared["ir_quality"], "IR quality")
    gate = tensor_map(prepared["gate"], "Reliability gate")

    np.save(output_dir / "rgb_quality_map.npy", rgb_quality)
    np.save(output_dir / "ir_quality_map.npy", ir_quality)
    np.save(output_dir / "reliability_gate.npy", gate)
    # Preserve the grayscale model output for reproducibility and
    # detector evaluation; save the color-restored visualization too.
    np.save(output_dir / "fused_image.npy", fused_image)
    np.save(
        output_dir / "fused_color_image.npy", fused_color_image
    )
    plt.imsave(
        output_dir / "fused_color_image.png",
        np.clip(fused_color_image, 0, 1),
    )

    plot_inputs_quality(
        rgb_image, ir_image, rgb_quality, ir_quality,
        sample_id, output_dir,
    )
    plot_quality_overlays(
        rgb_image, ir_image, rgb_quality, ir_quality,
        sample_id, output_dir,
    )
    plot_gate(
        gate, rgb_image, ir_image, sample_id, output_dir,
    )
    plot_fusion(
        rgb_image, ir_image, fused_color_image, sample_id, output_dir,
    )
    plot_detection_comparison(
        rgb_image,
        ir_image,
        fused_color_image,
        {
            "rgb": rgb_prediction,
            "ir": ir_prediction,
            "pace": pace_prediction,
        },
        targets,
        args.confidence,
        sample_id,
        output_dir,
    )
    plot_pace_ground_truth(
        fused_color_image,
        pace_prediction,
        targets,
        args.confidence,
        sample_id,
        output_dir,
    )

    print("\nDetection comparison (confidence >= {:.2f}):".format(
        args.confidence
    ))
    for name, prediction in (
        ("RGB-only", rgb_prediction),
        ("IR-only", ir_prediction),
        ("PACE", pace_prediction),
    ):
        count = int(
            np.count_nonzero(prediction["scores"] >= args.confidence)
        )
        print("  {:10s}: {} detections".format(name, count))

    print("\nQuality statistics:")
    for name, quality in (
        ("RGB", rgb_quality),
        ("IR", ir_quality),
        ("Gate", gate),
    ):
        print(
            "  {}: min={:.4f}, mean={:.4f}, max={:.4f}".format(
                name, quality.min(), quality.mean(), quality.max()
            )
        )

    print("\nSaved visualizations to: {}".format(output_dir))


if __name__ == "__main__":
    main()