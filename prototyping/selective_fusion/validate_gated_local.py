
import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import torch

from inference import (
    load_models,
    load_image,
    load_quality_image,
    predict_quality_from_original,
    samples,
    choose_device,
)
from local_crossfuse import LocalCrossFuse


ROOT = Path(__file__).resolve().parents[2]


def synchronize(device):
    device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elif device.type == "mps":
        torch.mps.synchronize()


def timed(device, fn):
    synchronize(device)
    start = time.perf_counter()

    result = fn()

    synchronize(device)
    return result, time.perf_counter() - start


@torch.inference_mode()
def full_crossfuse(baseline, encoders, rgb, ir):
    rgb_sh, rgb_de = encoders["rgb"](
        rgb * 255.0
    )

    ir_sh, ir_de = encoders["ir"](
        ir * 255.0
    )

    ir_windows, padding = (
        baseline.patch_embed_tensor(ir_de)
    )

    rgb_windows, _ = (
        baseline.patch_embed_tensor(rgb_de)
    )

    fused_windows = []

    for i in range(ir_windows.shape[2]):
        fused = baseline.cross_atten_block(
            ir_windows[:, :, i],
            rgb_windows[:, :, i],
            True,
        )[0]

        fused_windows.append(fused)

    fused_features = baseline.recons_tensor(
        torch.stack(fused_windows, dim=2),
        padding,
    )

    raw = baseline.decoder_fusion(
        ir_sh,
        rgb_sh,
        ir_de,
        rgb_de,
        fused_features,
    )

    from tools import utils

    return utils.normalize_tensor(raw)


def image_np(x):
    return (
        x[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def overlay_quality_scores(ax, quality):
    """
    Overlay 8x8 predicted quality scores on an image.
    """
    scores = (
        quality[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    rows, cols = scores.shape

    image_h, image_w = (
        ax.images[0].get_array().shape
    )

    cell_h = image_h / rows
    cell_w = image_w / cols

    for r in range(1, rows):
        ax.axhline(
            r * cell_h - 0.5,
            color="white",
            alpha=0.35,
            linewidth=0.5,
        )

    for c in range(1, cols):
        ax.axvline(
            c * cell_w - 0.5,
            color="white",
            alpha=0.35,
            linewidth=0.5,
        )

    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) * cell_w - 0.5
            y = (r + 0.5) * cell_h - 0.5

            label = ax.text(
                x,
                y,
                f"{scores[r, c]:.2f}",
                color="white",
                fontsize=6,
                fontweight="bold",
                ha="center",
                va="center",
            )

            label.set_path_effects([
                patheffects.withStroke(
                    linewidth=1.5,
                    foreground="black",
                )
            ])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=25,
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
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "prototyping"
            / "selective_fusion"
            / "outputs"
            / "gated_local_validation.png"
        ),
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error(
            "--repeats must be >= 1"
        )

    device = choose_device(args.device)

    # Load the trained quality estimators and gate
    # through the main inference pipeline.
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

    baseline = selective.baseline

    sample_list = samples(
        args.split,
        args.sample_index + 1,
    )

    if not 0 <= args.sample_index < len(sample_list):
        raise ValueError(
            "Sample index out of range"
        )

    (
        stem,
        rgb_path,
        ir_path,
        _,
        _,
    ) = sample_list[args.sample_index]

    # --------------------------------------------------
    # Load both resolutions
    # --------------------------------------------------

    # 256x256 images for CrossFuse fusion.
    rgb = load_image(rgb_path, device)
    ir = load_image(ir_path, device)

    # Original-resolution images for quality estimation.
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

    print(f"\nSample: {stem}")
    print(f"RGB original: {tuple(rgb_original.shape)}")
    print(f"IR original:  {tuple(ir_original.shape)}")
    print(f"RGB fusion:   {tuple(rgb.shape)}")
    print(f"IR fusion:    {tuple(ir.shape)}")

    local_model = LocalCrossFuse(
        rgb_encoder=encoders["rgb"],
        ir_encoder=encoders["ir"],
        baseline=baseline,
        context_features=args.context_features,
        batch_size=args.batch_size,
    ).to(device).eval()

    all_on = torch.ones(
        (1, 1, 8, 8),
        device=device,
    )

    # --------------------------------------------------
    # Preparation
    # --------------------------------------------------

    def prepare():
        # Encode resized images for fusion.
        rgb_sh, rgb_de = encoders["rgb"](
            rgb * 255.0
        )

        ir_sh, ir_de = encoders["ir"](
            ir * 255.0
        )

        # Predict quality using original-resolution images.
        # This matches the working notebook.
        rgb_q = predict_quality_from_original(
            encoders["rgb"],
            estimators["rgb"],
            scales["rgb"],
            rgb_original,
        )

        ir_q = predict_quality_from_original(
            encoders["ir"],
            estimators["ir"],
            scales["ir"],
            ir_original,
        )

        gate = gate_model(
            rgb_q,
            ir_q,
        )

        features = (
            rgb_sh,
            rgb_de,
            ir_sh,
            ir_de,
        )

        return rgb_q, ir_q, gate, features

    def run_local(prepared, mask):
        rgb_q, ir_q, _, features = prepared

        return local_model(
            rgb,
            ir,
            rgb_q,
            ir_q,
            mask,
            encoded_features=features,
        )

    # --------------------------------------------------
    # Warmup and benchmarking
    # --------------------------------------------------

    with torch.inference_mode():
        warm = prepare()

        full_crossfuse(
            baseline,
            encoders,
            rgb,
            ir,
        )

        run_local(warm, all_on)
        run_local(warm, warm[2])

        full_times = []
        all_times = []
        gate_times = []
        prepare_times = []
        end_to_end_times = []

        for _ in range(args.repeats):
            _, t = timed(
                device,
                lambda: full_crossfuse(
                    baseline,
                    encoders,
                    rgb,
                    ir,
                ),
            )
            full_times.append(t)

            prepared, t = timed(
                device,
                prepare,
            )
            prepare_times.append(t)

            _, t = timed(
                device,
                lambda: run_local(
                    prepared,
                    all_on,
                ),
            )
            all_times.append(t)

            _, t = timed(
                device,
                lambda: run_local(
                    prepared,
                    prepared[2],
                ),
            )
            gate_times.append(t)

            def complete_pipeline():
                p = prepare()
                return run_local(p, p[2])

            _, t = timed(
                device,
                complete_pipeline,
            )
            end_to_end_times.append(t)

        prepared = prepare()

        rgb_q, ir_q, gate, _ = prepared

        gated_result = run_local(
            prepared,
            gate,
        )

        all_result = run_local(
            prepared,
            all_on,
        )

        full = full_crossfuse(
            baseline,
            encoders,
            rgb,
            ir,
        )

    # --------------------------------------------------
    # Timing results
    # --------------------------------------------------

    count = int(gate.sum().item())
    rate = count / 64.0

    median = lambda values: float(
        np.median(values)
    )

    full_t = median(full_times)
    all_t = median(all_times)
    gated_t = median(gate_times)
    prep_t = median(prepare_times)
    total_t = median(end_to_end_times)

    print("\n" + "=" * 58)
    print(f"Sample: {stem}")
    print(
        f"Selected patches: "
        f"{count}/64 ({rate:.1%})"
    )
    print(
        f"Skipped patches: {64 - count}/64"
    )
    print(f"Batch size: {args.batch_size}")
    print(
        f"Context features: "
        f"{args.context_features}x"
        f"{args.context_features}"
    )
    print(
        f"Median of {args.repeats} timed runs"
    )
    print("=" * 58)

    print(
        f"Full CrossFuse:            "
        f"{full_t:.4f} s"
    )

    print(
        f"Quality + gate + encoding: "
        f"{prep_t:.4f} s"
    )

    print(
        f"Local all-on (fusion):     "
        f"{all_t:.4f} s"
    )

    print(
        f"Local gated (fusion):      "
        f"{gated_t:.4f} s"
    )

    print(
        f"PACE end-to-end:           "
        f"{total_t:.4f} s"
    )

    if gated_t > 0:
        print(
            f"Fusion-only speedup:       "
            f"{all_t / gated_t:.2f}x"
        )

    if total_t > 0:
        print(
            f"Vs full CrossFuse:         "
            f"{full_t / total_t:.2f}x"
        )

    # --------------------------------------------------
    # Quality diagnostics
    # --------------------------------------------------

    print("\n" + "=" * 58)
    print("PREDICTED QUALITY ESTIMATOR OUTPUTS")
    print("=" * 58)

    for name, quality in (
        ("RGB", rgb_q),
        ("IR", ir_q),
    ):
        print(f"\n{name} quality:")
        print(f"  Min:  {quality.min().item():.4f}")
        print(f"  Mean: {quality.mean().item():.4f}")
        print(f"  Max:  {quality.max().item():.4f}")

        print(
            np.array2string(
                image_np(quality),
                precision=3,
                suppress_small=True,
            )
        )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        constrained_layout=True,
    )

    panels = [
        (
            "Original RGB + Predicted Quality",
            image_np(rgb),
        ),
        (
            "Original IR + Predicted Quality",
            image_np(ir),
        ),
        (
            f"Trained Gate ({count}/64)",
            image_np(gate),
        ),
        (
            "Full CrossFuse",
            image_np(full),
        ),
        (
            "Local All-On",
            image_np(all_result["out"]),
        ),
        (
            "Local Trained Gate",
            image_np(gated_result["out"]),
        ),
    ]

    for idx, (ax, (title, data)) in enumerate(
        zip(axes.flat, panels)
    ):
        if title.startswith("Trained Gate"):
            ax.imshow(
                data,
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
        else:
            ax.imshow(
                data,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

        if idx == 0:
            overlay_quality_scores(
                ax,
                rgb_q,
            )

        elif idx == 1:
            overlay_quality_scores(
                ax,
                ir_q,
            )

        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"{stem} | {count}/64 fused | "
        f"Full {full_t:.3f}s | "
        f"PACE end-to-end {total_t:.3f}s"
    )

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output,
        dpi=160,
        bbox_inches="tight",
    )

    from PIL import Image

    fused = image_np(
        gated_result["out"]
    )

    Image.fromarray(
        (
            np.clip(fused, 0, 1) * 255
        ).astype(np.uint8)
    ).save(
        args.output.parent
        / "gated_local_fused.png"
    )

    print(
        f"\nSaved visualization: "
        f"{args.output}"
    )

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
