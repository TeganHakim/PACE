
import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from inference import (
    load_models,
    load_image,
    samples,
    choose_device,
)
from local_crossfuse import LocalCrossFuse

ROOT = Path(__file__).resolve().parents[2]
QE_DIR = ROOT / "prototyping" / "quality_estimator"
GATE_DIR = ROOT / "prototyping" / "reliability_gate"

sys.path.insert(0, str(QE_DIR))
sys.path.insert(0, str(GATE_DIR))

from model import SpatialQualityEstimator
from degradation_features import degradation_features
from reliability_gate import ReliabilityGate


def checkpoint(path, device):
    data = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" not in data:
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return data


def load_quality_model(modality, device):
    path = QE_DIR / "checkpoints" / f"quality_{modality}.pt"
    data = checkpoint(path, device)

    model = SpatialQualityEstimator(
        in_channels=int(data["in_channels"])
    ).to(device)

    model.load_state_dict(data["model_state_dict"])
    model.eval().requires_grad_(False)

    scale = float(data.get("extra_feature_scale", 10.0))
    return model, scale


def load_gate_model(device):
    path = GATE_DIR / "checkpoints" / "reliability_gate_best.pt"
    data = checkpoint(path, device)

    model = ReliabilityGate(
        hidden_dim=int(data.get("hidden_dim", 16))
    ).to(device)

    model.load_state_dict(data["model_state_dict"])
    model.eval().requires_grad_(False)
    return model


def predict_quality(model, image, deep, scale):
    extra = degradation_features(image)
    extra = F.adaptive_avg_pool2d(
        extra, deep.shape[-2:]
    )
    features = torch.cat(
        [deep.float(), extra.float() * scale],
        dim=1,
    )
    return model(features)


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
    rgb_sh, rgb_de = encoders["rgb"](rgb * 255.0)
    ir_sh, ir_de = encoders["ir"](ir * 255.0)

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
        ir_sh, rgb_sh, ir_de, rgb_de, fused_features
    )

    from tools import utils
    return utils.normalize_tensor(raw)


def image_np(x):
    return x[0, 0].detach().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-index", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-features", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "prototyping" / "selective_fusion"
        / "outputs" / "gated_local_validation.png",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    device = choose_device(args.device)

    encoders, selective, _, _, _ = load_models(
        device,
        quality_source="ground_truth",
        gate_mode="all_on",
    )
    baseline = selective.baseline

    rgb_q_model, rgb_scale = load_quality_model("rgb", device)
    ir_q_model, ir_scale = load_quality_model("ir", device)
    gate_model = load_gate_model(device)

    sample_list = samples(args.split, args.sample_index + 1)
    if not 0 <= args.sample_index < len(sample_list):
        raise ValueError("Sample index out of range")

    stem, rgb_path, ir_path, _, _ = sample_list[args.sample_index]
    rgb = load_image(rgb_path, device)
    ir = load_image(ir_path, device)

    local_model = LocalCrossFuse(
        rgb_encoder=encoders["rgb"],
        ir_encoder=encoders["ir"],
        baseline=baseline,
        context_features=args.context_features,
        batch_size=args.batch_size,
    ).to(device)

    all_on = torch.ones((1, 1, 8, 8), device=device)

    def prepare():
        rgb_sh, rgb_de = encoders["rgb"](rgb * 255.0)
        ir_sh, ir_de = encoders["ir"](ir * 255.0)

        rgb_q = predict_quality(
            rgb_q_model, rgb, rgb_de, rgb_scale
        )
        ir_q = predict_quality(
            ir_q_model, ir, ir_de, ir_scale
        )

        gate = gate_model(rgb_q, ir_q)

        features = (rgb_sh, rgb_de, ir_sh, ir_de)
        return rgb_q, ir_q, gate, features

    def run_local(prepared, mask):
        rgb_q, ir_q, _, features = prepared
        return local_model(
            rgb, ir, rgb_q, ir_q, mask,
            encoded_features=features,
        )

    # Warm up the exact operations we intend to benchmark.
    with torch.inference_mode():
        warm = prepare()
        full_crossfuse(baseline, encoders, rgb, ir)
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
                    baseline, encoders, rgb, ir
                ),
            )
            full_times.append(t)

            prepared, t = timed(device, prepare)
            prepare_times.append(t)

            _, t = timed(
                device,
                lambda: run_local(prepared, all_on),
            )
            all_times.append(t)

            _, t = timed(
                device,
                lambda: run_local(prepared, prepared[2]),
            )
            gate_times.append(t)

            def complete_pipeline():
                p = prepare()
                return run_local(p, p[2])

            _, t = timed(device, complete_pipeline)
            end_to_end_times.append(t)

        prepared = prepare()
        rgb_q, ir_q, gate, _ = prepared

        gated_result = run_local(prepared, gate)
        all_result = run_local(prepared, all_on)
        full = full_crossfuse(baseline, encoders, rgb, ir)

    count = int(gate.sum().item())
    rate = count / 64.0

    median = lambda values: float(np.median(values))

    full_t = median(full_times)
    all_t = median(all_times)
    gated_t = median(gate_times)
    prep_t = median(prepare_times)
    total_t = median(end_to_end_times)

    print("\n" + "=" * 58)
    print(f"Sample: {stem}")
    print(f"Selected patches: {count}/64 ({rate:.1%})")
    print(f"Skipped patches: {64-count}/64")
    print(f"Batch size: {args.batch_size}")
    print(f"Context features: {args.context_features}x"
          f"{args.context_features}")
    print(f"Median of {args.repeats} timed runs")
    print("=" * 58)

    print(f"Full CrossFuse:            {full_t:.4f} s")
    print(f"Quality + gate + encoding: {prep_t:.4f} s")
    print(f"Local all-on (fusion):     {all_t:.4f} s")
    print(f"Local gated (fusion):      {gated_t:.4f} s")
    print(f"PACE end-to-end:           {total_t:.4f} s")

    if gated_t > 0:
        print(f"Fusion-only speedup:       {all_t/gated_t:.2f}x")
    if total_t > 0:
        print(f"Vs full CrossFuse:         {full_t/total_t:.2f}x")

    # Visualize the trained gate and actual fusion result.
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 10), constrained_layout=True
    )

    panels = [
        ("Original RGB", image_np(rgb)),
        ("Original IR", image_np(ir)),
        (f"Trained Gate ({count}/64)", image_np(gate)),
        ("Full CrossFuse", image_np(full)),
        ("Local All-On", image_np(all_result["out"])),
        ("Local Trained Gate", image_np(gated_result["out"])),
    ]

    for ax, (title, data) in zip(axes.flat, panels):
        if title.startswith("Trained Gate"):
            ax.imshow(
                data, cmap="RdYlGn", vmin=0, vmax=1,
                interpolation="nearest",
            )
        else:
            ax.imshow(data, cmap="gray", vmin=0, vmax=1)

        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"{stem} | {count}/64 fused | "
        f"Full {full_t:.3f}s | "
        f"PACE end-to-end {total_t:.3f}s"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")

    from PIL import Image

    fused = image_np(gated_result["out"])
    Image.fromarray(
        (np.clip(fused, 0, 1) * 255).astype(np.uint8)
    ).save(args.output.parent / "gated_local_fused.png")

    print(f"\nSaved visualization: {args.output}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
