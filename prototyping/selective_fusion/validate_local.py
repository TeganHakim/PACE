
"""
Validate local feature-space CrossFuse.

Examples:

python prototyping/selective_fusion/validate_local.py \
    --device cpu --sample-index 25 --gate-mode all_on

python prototyping/selective_fusion/validate_local.py \
    --device cpu --sample-index 25 \
    --gate-mode file --gate-npy path/to/gate.npy
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from inference import (
    load_models,
    load_image,
    load_quality,
    samples,
    choose_device,
)

from local_crossfuse import LocalCrossFuse
from tools import utils


HERE = Path(__file__).resolve().parent


def as_image(tensor):
    return tensor.detach().cpu().float()[0, 0].numpy()


@torch.no_grad()
def full_crossfuse(baseline, rgb_encoder, ir_encoder, rgb, ir):
    rgb_sh, rgb_de = rgb_encoder(rgb * 255)
    ir_sh, ir_de = ir_encoder(ir * 255)

    ir_windows, padding = baseline.patch_embed_tensor(ir_de)
    rgb_windows, _ = baseline.patch_embed_tensor(rgb_de)

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

    return utils.normalize_tensor(raw)


def load_gate(args, device):
    if args.gate_mode == "all_on":
        return torch.ones(
            (1, 1, 8, 8),
            device=device,
            dtype=torch.float32,
        )

    if args.gate_npy is None:
        raise ValueError("--gate-npy required with --gate-mode file")

    array = np.load(args.gate_npy).astype(np.float32)
    array = np.squeeze(array)

    if array.shape != (8, 8):
        raise ValueError(
            f"Expected 8x8 gate, got {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError("Gate contains non-finite values")

    if not np.isin(array, [0, 1]).all():
        raise ValueError("Gate must contain only 0 and 1")

    return torch.from_numpy(
        array[None, None]
    ).to(device)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-index", type=int, default=25)

    parser.add_argument(
        "--gate-mode",
        choices=["all_on", "file"],
        default="all_on",
    )
    parser.add_argument("--gate-npy", type=Path)

    parser.add_argument("--context-features", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "outputs" / "local_validation.png",
    )

    args = parser.parse_args()

    device = choose_device(args.device)

    encoders, selective, _, _, _ = load_models(
        device,
        quality_source="ground_truth",
        gate_mode="all_on",
    )

    baseline = selective.baseline

    sample_list = samples(
        args.split,
        args.sample_index + 1,
    )

    if args.sample_index < 0 or args.sample_index >= len(sample_list):
        raise ValueError("Sample index out of range")

    (
        stem,
        rgb_path,
        ir_path,
        rgb_qpath,
        ir_qpath,
    ) = sample_list[args.sample_index]

    rgb = load_image(rgb_path, device)
    ir = load_image(ir_path, device)

    rgb_quality = load_quality(rgb_qpath, device)
    ir_quality = load_quality(ir_qpath, device)

    gate = load_gate(args, device)

    model = LocalCrossFuse(
        rgb_encoder=encoders["rgb"],
        ir_encoder=encoders["ir"],
        baseline=baseline,
        context_features=args.context_features,
        batch_size=args.batch_size,
    ).to(device)

    print(f"Sample: {stem}")
    print(f"Selected patches: {int(gate.sum().item())}/64")

    with torch.inference_mode():
        start = time.perf_counter()
        full = full_crossfuse(
            baseline,
            encoders["rgb"],
            encoders["ir"],
            rgb,
            ir,
        )
        full_time = time.perf_counter() - start

        start = time.perf_counter()
        result = model(
            rgb,
            ir,
            rgb_quality,
            ir_quality,
            gate,
        )
        local_time = time.perf_counter() - start

    local = result["out"]

    if not torch.isfinite(local).all():
        raise RuntimeError("Local fusion produced non-finite values")

    print(f"Full CrossFuse: {full_time:.3f}s")
    print(f"Local CrossFuse: {local_time:.3f}s")
    print(f"Local fused patches: {result['fused_count']}")

    full_img = as_image(full)
    local_img = as_image(local)
    difference = np.abs(full_img - local_img)

    fig, axes = plt.subplots(
        2, 3, figsize=(15, 10), constrained_layout=True
    )

    panels = [
        ("Original RGB", as_image(rgb)),
        ("Original IR", as_image(ir)),
        ("8x8 Fusion Gate", as_image(gate)),
        ("Full CrossFuse", full_img),
        ("Local Feature CrossFuse", local_img),
        ("Absolute Difference", difference),
    ]

    for ax, (title, image) in zip(axes.flat, panels):
        if title == "8x8 Fusion Gate":
            ax.imshow(
                image,
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                interpolation="nearest",
                extent=(0, 256, 256, 0),
            )
        elif title == "Absolute Difference":
            ax.imshow(image, cmap="inferno")
        else:
            ax.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"{stem} | Local context: "
        f"{args.context_features}x{args.context_features} features | "
        f"{result['fused_count']}/64 patches | "
        f"Full: {full_time:.2f}s, Local: {local_time:.2f}s"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")

    from PIL import Image

    fused_path = args.output.parent / "local_fused.png"
    Image.fromarray(
        (local_img.clip(0, 1) * 255).astype(np.uint8)
    ).save(fused_path)

    print(f"Visualization: {args.output}")
    print(f"Fused image: {fused_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
