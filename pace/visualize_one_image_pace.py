#!/usr/bin/env python3
"""
Run the complete PACE pipeline on one paired RGB/IR test image and save
publication-quality visualizations of its intermediate outputs.

Run from the repository root:
    python pace/visualize_one_image_pace.py

Choose a sample by ID or sorted sample index:
    python pace/visualize_one_image_pace.py --sample-id 00315
    python pace/visualize_one_image_pace.py --sample-index 25

If neither is provided, a test pair is selected randomly.

The script follows evaluate.py's inference path:
    resized RGB/IR images
        -> frozen CrossFuse encoders
        -> predicted RGB/IR quality maps
        -> trained reliability gate
        -> LocalCrossFuse selective fusion
        -> CrossFuse YOLO11n detector

The same detector is also run independently on the RGB-only and IR-only
images to provide a qualitative comparison with PACE fusion.

Outputs are written to:
    pace/visualize_one_image_pace/<sample-id>/
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import seaborn as sns
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Repository paths and project imports
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CROSSFUSE_ROOT = REPO / "baseline" / "CrossFuse"
SELECTIVE_ROOT = REPO / "prototyping" / "selective_fusion"

sys.path.insert(0, str(CROSSFUSE_ROOT))
sys.path.insert(0, str(SELECTIVE_ROOT))
sys.path.insert(0, str(HERE))

from inference import (
    choose_device,
    load_image,
    load_models,
    load_quality_image, 
    predict_quality,
    predict_quality_from_original,
    predict_quality_from_deep_features,
    samples,
)
from local_crossfuse import LocalCrossFuse
from train_conv_trans import CLASS_TO_ID, ID_TO_CLASS, load_targets


# ---------------------------------------------------------------------
# Visualization defaults
# ---------------------------------------------------------------------

sns.set_theme(
    context="paper",
    style="white",
    font="DejaVu Sans",
    font_scale=1.15,
)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "font.size": 10,
})


# ---------------------------------------------------------------------
# Arguments and reproducibility
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split from which to select the paired image.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Exact pair ID/stem, e.g. 00367.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Zero-based index in the sorted sample list.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when selecting a sample automatically.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--context-features",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence displayed for detections.",
    )
    parser.add_argument(
        "--nms-confidence",
        type=float,
        default=0.001,
        help="Low NMS threshold to retain candidates before display filtering.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
    )
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
        default=HERE / "visualize_one_image_pace",
        help="Output root directory.",
    )

    args = parser.parse_args()

    if args.sample_id is not None and args.sample_index is not None:
        parser.error("Use --sample-id or --sample-index, not both.")

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1.")

    if not 0.0 <= args.nms_confidence <= 1.0:
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


def load_detector(path, device):
    """
    Extract the YOLO11n detector from the trained Trans_FuseNet
    checkpoint, matching evaluate.py.
    """
    if not path.is_file():
        raise FileNotFoundError(
            "Detector checkpoint not found: {}".format(path)
        )

    checkpoint = safe_torch_load(path, device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected a state_dict checkpoint, got {}".format(
                type(checkpoint)
            )
        )

    detector_state = {
        key[len("detector."):]: value
        for key, value in checkpoint.items()
        if key.startswith("detector.")
    }

    if not detector_state:
        raise RuntimeError(
            "No detector.* parameters found in {}. Expected the full "
            "CrossFuse Trans_FuseNet checkpoint.".format(path)
        )

    detector = DetectionModel(
        "yolo11n.yaml",
        ch=1,
        nc=len(CLASS_TO_ID),
        verbose=False,
    ).to(device)

    detector.load_state_dict(detector_state, strict=True)
    detector.eval().requires_grad_(False)
    return detector


# ---------------------------------------------------------------------
# Sample selection and image conversion
# ---------------------------------------------------------------------

def select_sample(args):
    sample_list = samples(args.split, None)

    if not sample_list:
        raise RuntimeError(
            "No paired RGB/IR samples found for split '{}'.".format(
                args.split
            )
        )

    if args.sample_id is not None:
        requested = Path(args.sample_id).stem
        matches = [
            sample for sample in sample_list
            if Path(sample[0]).stem == requested
        ]

        if not matches:
            raise ValueError(
                "Sample ID '{}' was not found in split '{}'. "
                "Available sample count: {}.".format(
                    requested,
                    args.split,
                    len(sample_list),
                )
            )

        return matches[0]

    if args.sample_index is not None:
        if not 0 <= args.sample_index < len(sample_list):
            raise IndexError(
                "Sample index {} is out of range; split '{}' has {} "
                "samples.".format(
                    args.sample_index,
                    args.split,
                    len(sample_list),
                )
            )
        return sample_list[args.sample_index]

    return random.Random(args.seed).choice(sample_list)


def tensor_image(tensor):
    """Convert a normalized [1,1,H,W] tensor to a 2-D NumPy image."""
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(
            "Expected a [1,1,H,W] image tensor; got {}".format(
                tuple(tensor.shape)
            )
        )

    return (
        tensor.detach()
        .float()
        .cpu()[0, 0]
        .clamp(0.0, 1.0)
        .numpy()
    )


def tensor_map(tensor, name):
    """Convert a [1,1,h,w] model output into a 2-D NumPy map."""
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(
            "{} must have shape [1,1,h,w]; got {}".format(
                name,
                tuple(tensor.shape),
            )
        )

    result = tensor.detach().float().cpu()[0, 0].numpy()

    if not np.isfinite(result).all():
        raise ValueError("{} contains NaN or Inf values.".format(name))

    return result


# ---------------------------------------------------------------------
# PACE preparation and fusion
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
    Encode the resized inputs once and reuse those deep features for
    quality estimation. This matches the current 256x256 inference path.
    """
    # rgb_shallow_original, rgb_deep_original = encoders["rgb"](rgb_original * 255.0)
    # ir_shallow_original, ir_deep_original = encoders["ir"](ir_original * 255.0)

    # rgb_quality = predict_quality(
    #     estimators["rgb"],
    #     scales["rgb"],
    #     rgb_original,
    #     rgb_deep_original,
    # )

    # # Resize features for the existing fusion architecture.
    # rgb_shallow = F.interpolate(
    #     rgb_shallow_original,
    #     size=(256, 256),
    #     mode="bilinear",
    #     align_corners=False,
    # )

    # rgb_deep = F.interpolate(
    #     rgb_deep_original,
    #     size=(32, 32),
    #     mode="bilinear",
    #     align_corners=False,
    # )

    # ir_quality = predict_quality(
    #     estimators["ir"],
    #     scales["ir"],
    #     ir_original,
    #     ir_deep_original,
    # )

    # ir_shallow = F.interpolate(
    #     ir_shallow_original,
    #     size=(256, 256),
    #     mode="bilinear",
    #     align_corners=False,
    # )

    # ir_deep = F.interpolate(
    #     ir_deep_original,
    #     size=(32, 32),
    #     mode="bilinear",
    #     align_corners=False,
    # )

    # rgb_quality = predict_quality_from_original(
    #     encoders["rgb"],
    #     estimators["rgb"],
    #     scales["rgb"],
    #     rgb_original,
    # )

    # ir_quality = predict_quality_from_original(
    #     encoders["ir"],
    #     estimators["ir"],
    #     scales["ir"],
    #     ir_original,
    # )

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

    rgb_shallow, rgb_deep = encoders['rgb'](rgb * 255.0)
    ir_shallow, ir_deep = encoders['ir'](ir * 255.0)
    rgb_quality = predict_quality_from_original(encoders['rgb'], estimators['rgb'], scales['rgb'], rgb_original)
    ir_quality = predict_quality_from_original(encoders['ir'], estimators['ir'], scales['ir'], ir_original)

    gate = gate_model(rgb_quality, ir_quality)

    expected_quality_shape = (1, 1, 8, 8)
    for name, quality in (
        ("RGB quality", rgb_quality),
        ("IR quality", ir_quality),
        ("Reliability gate", gate),
    ):
        if tuple(quality.shape) != expected_quality_shape:
            raise ValueError(
                "{} has shape {}; expected {}.".format(
                    name,
                    tuple(quality.shape),
                    expected_quality_shape,
                )
            )

        if not torch.isfinite(quality).all():
            raise ValueError("{} contains NaN or Inf.".format(name))

    encoded_features = (
        rgb_shallow,
        rgb_deep,
        ir_shallow,
        ir_deep,
    )

    return {
        "rgb_quality": rgb_quality,
        "ir_quality": ir_quality,
        "gate": gate,
        "encoded_features": encoded_features,
    }


@torch.inference_mode()
def run_pace(local_model, rgb, ir, prepared):
    """
    Use the same LocalCrossFuse invocation as evaluate.py.

    output['out'] is already normalized to [0,1] and is passed directly
    to the detector without additional scaling.
    """
    output = local_model(
        rgb,
        ir,
        prepared["rgb_quality"],
        prepared["ir_quality"],
        prepared["gate"],
        encoded_features=prepared["encoded_features"],
    )

    if not isinstance(output, dict) or "out" not in output:
        raise RuntimeError(
            "LocalCrossFuse must return a dictionary containing 'out'."
        )

    fused = output["out"]

    if not torch.is_tensor(fused):
        raise TypeError(
            "LocalCrossFuse output['out'] must be a tensor; got {}".format(
                type(fused)
            )
        )

    if fused.ndim != 4 or fused.shape[1] != 1:
        raise ValueError(
            "Expected fused output [B,1,H,W], got {}.".format(
                tuple(fused.shape)
            )
        )

    if not torch.isfinite(fused).all():
        raise ValueError("Fused image contains NaN or Inf values.")

    if fused.min().item() < -1e-4 or fused.max().item() > 1.0001:
        raise ValueError(
            "Expected LocalCrossFuse output in [0,1], got range [{:.4f}, "
            "{:.4f}].".format(
                fused.min().item(),
                fused.max().item(),
            )
        )

    return fused.float(), output


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------

@torch.inference_mode()
def detect(detector, image, confidence, nms_iou):
    """
    Run the same YOLO detector and NMS implementation for RGB, IR,
    and PACE. All inputs are normalized [1,1,256,256] tensors.
    """
    raw = detector(image)

    if isinstance(raw, (tuple, list)):
        raw = raw[0]

    result = non_max_suppression(
        raw,
        conf_thres=confidence,
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
# Plotting helpers
# ---------------------------------------------------------------------

def save_figure(fig, path):
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_inputs_and_quality(
    rgb_image,
    ir_image,
    rgb_quality,
    ir_quality,
    sample_id,
    output_dir,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        constrained_layout=True,
    )

    axes[0, 0].imshow(rgb_image, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("(a) Visible-spectrum RGB input")
    axes[0, 1].imshow(ir_image, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("(b) Infrared input")

    sns.heatmap(
        rgb_quality,
        ax=axes[1, 0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Predicted quality"},
        square=True,
    )
    axes[1, 0].set_title("(c) RGB patch-level quality")
    axes[1, 0].set_xlabel("Patch column")
    axes[1, 0].set_ylabel("Patch row")

    sns.heatmap(
        ir_quality,
        ax=axes[1, 1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Predicted quality"},
        square=True,
    )
    axes[1, 1].set_title("(d) IR patch-level quality")
    axes[1, 1].set_xlabel("Patch column")
    axes[1, 1].set_ylabel("Patch row")

    for ax in axes[0]:
        ax.set_axis_off()

    fig.suptitle(
        "PACE Input Modalities and Predicted Quality Maps\n"
        "Test pair: {}".format(sample_id),
        fontsize=16,
        fontweight="semibold",
    )

    save_figure(fig, output_dir / "01_inputs_and_quality.png")


def plot_quality_overlays(
    rgb_image,
    ir_image,
    rgb_quality,
    ir_quality,
    sample_id,
    output_dir,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        constrained_layout=True,
    )

    for ax, image, quality, title in (
        (
            axes[0],
            rgb_image,
            rgb_quality,
            "RGB quality overlay",
        ),
        (
            axes[1],
            ir_image,
            ir_quality,
            "IR quality overlay",
        ),
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        overlay = ax.imshow(
            quality,
            cmap="viridis",
            vmin=0,
            vmax=1,
            alpha=0.48,
            interpolation="nearest",
            extent=(
                -0.5,
                image.shape[1] - 0.5,
                image.shape[0] - 0.5,
                -0.5,
            ),
        )
        ax.set_title(title)
        ax.set_axis_off()

    fig.colorbar(
        overlay,
        ax=axes,
        fraction=0.035,
        pad=0.025,
        label="Predicted quality",
    )
    fig.suptitle(
        "Spatial Distribution of Modality Quality — {}".format(sample_id),
        fontsize=15,
        fontweight="semibold",
    )

    save_figure(fig, output_dir / "02_quality_overlays.png")


def plot_gate(gate, rgb_image, ir_image, sample_id, output_dir):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        constrained_layout=True,
    )

    sns.heatmap(
        gate,
        ax=axes[0],
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Gate decision"},
        square=True,
    )
    axes[0].set_title("Predicted reliability gate")
    axes[0].set_xlabel("Patch column")
    axes[0].set_ylabel("Patch row")

    for ax, image, title in (
        (axes[1], rgb_image, "Gate over RGB input"),
        (axes[2], ir_image, "Gate over IR input"),
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.imshow(
            gate,
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
            alpha=0.42,
            interpolation="nearest",
            extent=(
                -0.5,
                image.shape[1] - 0.5,
                image.shape[0] - 0.5,
                -0.5,
            ),
        )
        ax.set_title(title)
        ax.set_axis_off()

    fig.suptitle(
        "PACE Patch-Level Reliability Routing — {}".format(sample_id),
        fontsize=15,
        fontweight="semibold",
    )

    save_figure(fig, output_dir / "03_reliability_gate.png")


def plot_fusion(rgb_image, ir_image, fused_image, sample_id, output_dir):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        constrained_layout=True,
    )

    panels = (
        (rgb_image, "(a) RGB input"),
        (ir_image, "(b) IR input"),
        (fused_image, "(c) PACE fused output"),
    )

    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_axis_off()

    fig.suptitle(
        "PACE Selective Multimodal Fusion — {}".format(sample_id),
        fontsize=15,
        fontweight="semibold",
    )

    save_figure(fig, output_dir / "04_fusion_output.png")


def class_name(class_id):
    return ID_TO_CLASS.get(
        int(class_id),
        str(int(class_id)),
    )


def draw_ground_truth(ax, targets, width, height):
    for target in targets:
        class_id, cx, cy, box_width, box_height = target[:5]

        x1 = (float(cx) - float(box_width) / 2.0) * width
        y1 = (float(cy) - float(box_height) / 2.0) * height
        rect_width = float(box_width) * width
        rect_height = float(box_height) * height

        ax.add_patch(Rectangle(
            (x1, y1),
            rect_width,
            rect_height,
            fill=False,
            edgecolor="#ffb000",
            linewidth=1.8,
            linestyle="--",
        ))

        ax.text(
            x1,
            max(2, y1 - 3),
            "GT: {}".format(class_name(class_id)),
            color="black",
            fontsize=7,
            fontweight="bold",
            bbox={
                "facecolor": "#ffb000",
                "alpha": 0.88,
                "edgecolor": "none",
                "pad": 1.5,
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
            linewidth=1.8,
        ))

        ax.text(
            x1,
            min(y2 - 2, max(10, y1 + 10)),
            "{} {:.2f}".format(class_name(label), float(score)),
            color="black",
            fontsize=7,
            fontweight="bold",
            bbox={
                "facecolor": "#00e676",
                "alpha": 0.88,
                "edgecolor": "none",
                "pad": 1.5,
            },
        )


def plot_detection_comparison(
    images,
    predictions,
    targets,
    confidence,
    sample_id,
    output_dir,
):
    """
    Compare the exact same detector on RGB-only, IR-only, and PACE.
    Ground-truth boxes are shown in orange; predictions are green.
    """
    names = (
        ("RGB-only", images["rgb"], predictions["rgb"]),
        ("IR-only", images["ir"], predictions["ir"]),
        ("PACE fused", images["pace"], predictions["pace"]),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 6),
        constrained_layout=True,
    )

    for ax, (name, image, prediction) in zip(axes, names):
        height, width = image.shape

        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        draw_ground_truth(ax, targets, width, height)
        draw_predictions(ax, prediction, confidence)

        visible_count = int(
            np.count_nonzero(prediction["scores"] >= confidence)
        )

        ax.set_title(
            "{}\n{} detections".format(name, visible_count)
        )
        ax.set_axis_off()

    legend_handles = [
        Patch(
            facecolor="none",
            edgecolor="#ffb000",
            linestyle="--",
            label="Ground truth",
        ),
        Patch(
            facecolor="none",
            edgecolor="#00e676",
            label="Predicted detection",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, -0.015),
    )

    fig.suptitle(
        "Single-Modality vs. PACE Detection Comparison — {}\n"
        "Same YOLO11n detector; display confidence ≥ {:.2f}".format(
            sample_id,
            confidence,
        ),
        fontsize=15,
        fontweight="semibold",
    )

    save_figure(
        fig,
        output_dir / "05_rgb_ir_pace_detection_comparison.png",
    )


def plot_ground_truth_vs_pace(
    fused_image,
    pace_prediction,
    targets,
    confidence,
    sample_id,
    output_dir,
):
    height, width = fused_image.shape

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        constrained_layout=True,
    )

    axes[0].imshow(fused_image, cmap="gray", vmin=0, vmax=1)
    draw_ground_truth(axes[0], targets, width, height)
    axes[0].set_title("Ground-truth annotations")

    axes[1].imshow(fused_image, cmap="gray", vmin=0, vmax=1)
    draw_predictions(axes[1], pace_prediction, confidence)
    axes[1].set_title("PACE predicted detections")

    axes[2].imshow(fused_image, cmap="gray", vmin=0, vmax=1)
    draw_ground_truth(axes[2], targets, width, height)
    draw_predictions(axes[2], pace_prediction, confidence)
    axes[2].set_title("Overlay: ground truth and predictions")

    for ax in axes:
        ax.set_axis_off()

    legend_handles = [
        Patch(
            facecolor="none",
            edgecolor="#ffb000",
            linestyle="--",
            label="Ground truth",
        ),
        Patch(
            facecolor="none",
            edgecolor="#00e676",
            label="PACE prediction",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, -0.015),
    )

    fig.suptitle(
        "PACE Detection Output vs. Ground Truth — {}".format(sample_id),
        fontsize=15,
        fontweight="semibold",
    )

    save_figure(
        fig,
        output_dir / "06_ground_truth_vs_pace.png",
    )


def plot_pipeline_summary(
    rgb_image,
    ir_image,
    rgb_quality,
    ir_quality,
    gate,
    fused_image,
    pace_prediction,
    targets,
    confidence,
    sample_id,
    output_dir,
):
    """Create a compact overview figure for presentations and papers."""
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(1, 1, 1.15))

    ax_rgb = fig.add_subplot(grid[0, 0])
    ax_ir = fig.add_subplot(grid[0, 1])
    ax_rgb_q = fig.add_subplot(grid[0, 2])
    ax_ir_q = fig.add_subplot(grid[0, 3])

    ax_gate = fig.add_subplot(grid[1, 0])
    ax_gate_overlay = fig.add_subplot(grid[1, 1])
    ax_fused = fig.add_subplot(grid[1, 2])
    ax_detection = fig.add_subplot(grid[1, 3])

    panels = (
        (ax_rgb, rgb_image, "RGB input"),
        (ax_ir, ir_image, "IR input"),
        (ax_fused, fused_image, "PACE fused output"),
    )

    for ax, image, title in panels:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_axis_off()

    sns.heatmap(
        rgb_quality,
        ax=ax_rgb_q,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar=False,
        square=True,
    )
    ax_rgb_q.set_title("RGB quality")

    sns.heatmap(
        ir_quality,
        ax=ax_ir_q,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar=False,
        square=True,
    )
    ax_ir_q.set_title("IR quality")

    sns.heatmap(
        gate,
        ax=ax_gate,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        cbar=False,
        annot=True,
        fmt=".0f",
        square=True,
    )
    ax_gate.set_title("Reliability gate")

    ax_gate_overlay.imshow(rgb_image, cmap="gray", vmin=0, vmax=1)
    ax_gate_overlay.imshow(
        gate,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        alpha=0.42,
        interpolation="nearest",
        extent=(
            -0.5,
            rgb_image.shape[1] - 0.5,
            rgb_image.shape[0] - 0.5,
            -0.5,
        ),
    )
    ax_gate_overlay.set_title("Gate over RGB")
    ax_gate_overlay.set_axis_off()

    ax_detection.imshow(fused_image, cmap="gray", vmin=0, vmax=1)
    draw_ground_truth(
        ax_detection,
        targets,
        fused_image.shape[1],
        fused_image.shape[0],
    )
    draw_predictions(ax_detection, pace_prediction, confidence)
    ax_detection.set_title("PACE detections + ground truth")
    ax_detection.set_axis_off()

    ax_comparison = fig.add_subplot(grid[2, :])
    ax_comparison.imshow(fused_image, cmap="gray", vmin=0, vmax=1)
    draw_ground_truth(
        ax_comparison,
        targets,
        fused_image.shape[1],
        fused_image.shape[0],
    )
    draw_predictions(ax_comparison, pace_prediction, confidence)
    ax_comparison.set_title(
        "Final PACE detection result — orange dashed: ground truth; "
        "green: predicted detections"
    )
    ax_comparison.set_axis_off()

    fig.suptitle(
        "PACE End-to-End Pipeline Visualization | {}".format(sample_id),
        fontsize=17,
        fontweight="semibold",
    )

    save_figure(fig, output_dir / "00_pipeline_overview.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = choose_device(args.device)
    sample = select_sample(args)

    sample_id, rgb_path, ir_path, _, _ = sample
    sample_id = Path(sample_id).stem

    output_dir = args.output / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Selected pair: {}".format(sample_id))
    print("RGB: {}".format(rgb_path))
    print("IR:  {}".format(ir_path))
    print("Device: {}".format(device))
    print("Output: {}".format(output_dir))

    # Load PACE components through the same loader as evaluate.py.
    (
        encoders,
        selective,
        estimators,
        scales,
        gate_model,
    ) = load_models(
        device,
        quality_source="predicted",
        gate_mode="trained",
        context_features=args.context_features,
        batch_size=args.batch_size,
    )

    local_model = LocalCrossFuse(
        rgb_encoder=encoders["rgb"],
        ir_encoder=encoders["ir"],
        baseline=selective.baseline,
        context_features=args.context_features,
        batch_size=args.batch_size,
    ).to(device).eval()

    detector = load_detector(args.detector_weights, device)

    # Load the paired, grayscale, 256x256 inputs in [0,1].
    rgb = load_image(rgb_path, device)
    ir = load_image(ir_path, device)

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

    if tuple(rgb.shape) != tuple(ir.shape):
        raise ValueError(
            "RGB and IR input shapes do not match: {} vs {}".format(
                tuple(rgb.shape),
                tuple(ir.shape),
            )
        )

    # Ground truth is loaded using the exact helper used by evaluate.py.
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

        fused, _fusion_output = run_pace(
            local_model,
            rgb,
            ir,
            prepared,
        )

        # Use the same detector weights and NMS settings for all three
        # inputs. No additional scaling is applied to PACE's fused tensor.
        rgb_prediction = detect(
            detector,
            rgb.float(),
            args.nms_confidence,
            args.nms_iou,
        )
        ir_prediction = detect(
            detector,
            ir.float(),
            args.nms_confidence,
            args.nms_iou,
        )
        pace_prediction = detect(
            detector,
            fused.float(),
            args.nms_confidence,
            args.nms_iou,
        )

    rgb_image = tensor_image(rgb)
    ir_image = tensor_image(ir)
    fused_image = tensor_image(fused)

    rgb_quality = tensor_map(
        prepared["rgb_quality"],
        "RGB quality",
    )
    ir_quality = tensor_map(
        prepared["ir_quality"],
        "IR quality",
    )
    gate = tensor_map(
        prepared["gate"],
        "Reliability gate",
    )

    # Save raw maps as arrays as well as publication-ready figures.
    np.save(output_dir / "rgb_quality_map.npy", rgb_quality)
    np.save(output_dir / "ir_quality_map.npy", ir_quality)
    np.save(output_dir / "reliability_gate.npy", gate)

    # Save individual visualizations.
    plot_inputs_and_quality(
        rgb_image,
        ir_image,
        rgb_quality,
        ir_quality,
        sample_id,
        output_dir,
    )

    plot_quality_overlays(
        rgb_image,
        ir_image,
        rgb_quality,
        ir_quality,
        sample_id,
        output_dir,
    )

    plot_gate(
        gate,
        rgb_image,
        ir_image,
        sample_id,
        output_dir,
    )

    plot_fusion(
        rgb_image,
        ir_image,
        fused_image,
        sample_id,
        output_dir,
    )

    images = {
        "rgb": rgb_image,
        "ir": ir_image,
        "pace": fused_image,
    }
    predictions = {
        "rgb": rgb_prediction,
        "ir": ir_prediction,
        "pace": pace_prediction,
    }

    plot_detection_comparison(
        images,
        predictions,
        targets,
        args.confidence,
        sample_id,
        output_dir,
    )

    plot_ground_truth_vs_pace(
        fused_image,
        pace_prediction,
        targets,
        args.confidence,
        sample_id,
        output_dir,
    )

    plot_pipeline_summary(
        rgb_image,
        ir_image,
        rgb_quality,
        ir_quality,
        gate,
        fused_image,
        pace_prediction,
        targets,
        args.confidence,
        sample_id,
        output_dir,
    )

    print("\nSaved visualizations:")
    for path in sorted(output_dir.iterdir()):
        print("  {}".format(path))

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

    print("\nCompleted single-image PACE visualization.")


if __name__ == "__main__":
    main()