
"""Diagnose PACE reliability-gate labels, predictions, and loss weighting."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reliability_gate import ReliabilityGate


def load_cache(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing label cache: {path}")

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    x = data["inputs"].float()
    y = data["targets"].float()

    assert x.ndim == 4 and x.shape[1:] == (2, 8, 8), x.shape
    assert y.shape == (x.shape[0], 1, 8, 8), y.shape
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert torch.all((y == 0) | (y == 1))

    return x, y, data


def stats(name, x, y, gate, pos_weight, batch_size, threshold):
    n = y.numel()
    positives = int(y.sum().item())

    qrgb = x[:, :1]
    qir = x[:, 1:2]
    best_quality = torch.maximum(qrgb, qir)

    print(f"\n=== {name.upper()} ({len(x)} pairs; {n:,} patches) ===")
    print(
        f"Label 1: {positives:,}/{n:,} "
        f"= {positives / n:.2%}"
    )
    print(f"Label 0: {1 - positives / n:.2%}")
    print(
        f"Always-0 baseline accuracy: "
        f"{1 - positives / n:.2%}"
    )

    print(
        "RGB quality: mean {:.3f}, min {:.3f}, max {:.3f}".format(
            qrgb.mean().item(),
            qrgb.min().item(),
            qrgb.max().item(),
        )
    )
    print(
        "IR quality:  mean {:.3f}, min {:.3f}, max {:.3f}".format(
            qir.mean().item(),
            qir.min().item(),
            qir.max().item(),
        )
    )

    print("Positive-label rate by BEST original modality quality:")

    for lo, hi in [
        (0.0, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 0.95),
        (0.95, 1.001),
    ]:
        mask = (best_quality >= lo) & (best_quality < hi)

        if mask.any():
            count = int(mask.sum().item())
            fuse_rate = y[mask].mean().item()

            print(
                f"  [{lo:.2f}, {min(hi, 1.0):.2f}): "
                f"{count:5d} patches, "
                f"fuse={fuse_rate:.2%}"
            )

    with torch.inference_mode():
        logits = torch.cat(
            [
                gate.network(x[i:i + batch_size])
                for i in range(0, len(x), batch_size)
            ],
            dim=0,
        )

        probs = torch.sigmoid(logits)

        if threshold is None:
            # Use the actual threshold in ReliabilityGate.forward().
            preds = torch.cat(
                [
                    gate(
                        x[i:i + batch_size, 0:1],
                        x[i:i + batch_size, 1:2],
                    )
                    for i in range(0, len(x), batch_size)
                ],
                dim=0,
            ) > 0.5

            decision_rule = (
                "ReliabilityGate.forward() (current code)"
            )
        else:
            preds = probs > threshold
            decision_rule = (
                f"p(fuse) > {threshold:.3f} "
                "(--threshold override)"
            )

        truth = y.bool()

        def evaluate_subset(name, mask):
            pred = preds[mask]
            target = truth[mask]

            tp = int((pred & target).sum())
            fp = int((pred & ~target).sum())
            fn = int((~pred & target).sum())
            tn = int((~pred & ~target).sum())

            total = target.numel()
            if total == 0:
                print(f"\n{name}: No matching patches")
                return

            accuracy = (tp + tn) / total
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)

            print(f"\n=== {name} ===")
            print(f"Patches: {total:,}")
            print(f"Positive labels: {target.float().mean():.2%}")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
            print(f"F1: {f1:.4f}")
            print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")


        evaluate_subset(
            "ALL PATCHES",
            torch.ones_like(truth, dtype=torch.bool),
        )

        evaluate_subset(
            "DEGRADED PATCHES",
            (qrgb < 0.95) | (qir < 0.95),
        )

        tp = int((preds & truth).sum().item())
        fp = int((preds & ~truth).sum().item())
        fn = int((~preds & truth).sum().item())
        tn = int((~preds & ~truth).sum().item())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        accuracy = (tp + tn) / n
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        print(f"\nDecision rule: {decision_rule}")
        print(
            f"Predicted 1: {int(preds.sum().item()):,}/{n:,} "
            f"= {preds.float().mean().item():.2%}"
        )
        print(
            f"Probability: mean={probs.mean().item():.4f}, "
            f"min={probs.min().item():.4f}, "
            f"max={probs.max().item():.4f}"
        )

        for label in (0, 1):
            selected = truth == bool(label)

            if selected.any():
                values = probs[selected]

                mean = values.mean().item()
                p10 = torch.quantile(values, 0.1).item()
                p90 = torch.quantile(values, 0.9).item()

                print(
                    f"  p(fuse) for target={label}: "
                    f"mean={mean:.4f}, "
                    f"10th={p10:.4f}, "
                    f"90th={p90:.4f}"
                )

        print(
            f"Confusion matrix: "
            f"TP={tp:,} FP={fp:,} "
            f"FN={fn:,} TN={tn:,}"
        )
        print(
            f"Accuracy={accuracy:.2%} | "
            f"Precision={precision:.2%} | "
            f"Recall={recall:.2%} | "
            f"Specificity={specificity:.2%} | "
            f"F1={f1:.4f}"
        )

        unweighted_bce = F.binary_cross_entropy_with_logits(
            logits,
            y,
        ).item()

        weighted_bce = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=torch.tensor(pos_weight),
        ).item()

        print(f"BCE (unweighted)={unweighted_bce:.4f}")
        print(
            f"BCE (training pos_weight={pos_weight:.2f})="
            f"{weighted_bce:.4f}"
        )

        print("Prediction rates at alternative probability thresholds:")

        for cutoff in (0.20, 0.24, 0.26, 0.28, 0.30, 0.50):
            alternative = probs > cutoff

            fusion_rate = alternative.float().mean().item()
            alternative_accuracy = (
                (alternative == truth).float().mean().item()
            )

            print(
                f"  {cutoff:.2f}: "
                f"predicted-1={fusion_rate:.2%}, "
                f"accuracy={alternative_accuracy:.2%}"
            )

        first_ten = [
            int(v.item())
            for v in y[:10].sum(dim=(1, 2, 3))
        ]

        print(
            "Per-example fusion positives (first 10):",
            first_ten,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-train",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=HERE / "checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override the threshold in "
            "ReliabilityGate.forward()"
        ),
    )

    args = parser.parse_args()

    if args.threshold is not None:
        if not 0.0 <= args.threshold <= 1.0:
            parser.error("--threshold must be between 0 and 1")

    directory = args.checkpoint_dir

    train_x, train_y, train_meta = load_cache(
        directory
        / f"gate_labels_train_first{args.max_train}.pt"
    )

    val_x, val_y, _ = load_cache(
        directory
        / f"gate_labels_val_first{args.max_val}.pt"
    )

    positives = train_y.sum().item()
    total = train_y.numel()

    natural_weight = (
        (total - positives) / max(positives, 1)
    )
    weight = min(natural_weight, 20.0)

    effective_positive_fraction = (
        weight * positives
        / (weight * positives + total - positives)
    )

    print(
        f"Training positive fraction: "
        f"{positives / total:.2%}"
    )
    print(
        f"Natural inverse-frequency weight: "
        f"{natural_weight:.2f}"
    )
    print(
        f"Weight ACTUALLY USED in train.py: "
        f"{weight:.2f}"
    )
    print(
        "Effective positive fraction under weighted BCE: "
        f"{effective_positive_fraction:.2%}"
    )

    if "margin" in train_meta:
        print(
            f"Cached pseudo-label margin: "
            f"{train_meta['margin']}"
        )

    checkpoint_path = (
        directory / "reliability_gate_best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    gate = ReliabilityGate(
        hidden_dim=int(checkpoint.get("hidden_dim", 16))
    )

    gate.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    gate.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    stats(
        "train",
        train_x,
        train_y,
        gate,
        weight,
        args.batch_size,
        args.threshold,
    )

    stats(
        "validation",
        val_x,
        val_y,
        gate,
        weight,
        args.batch_size,
        args.threshold,
    )


if __name__ == "__main__":
    main()
