
import argparse
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support

from reliability_gate import ReliabilityGate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-val", type=int, default=200)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent / "checkpoints"

    data = torch.load(
        root / f"gate_labels_val_first{args.max_val}.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint = torch.load(
        root / "reliability_gate_best.pt",
        map_location="cpu",
        weights_only=False,
    )

    gate = ReliabilityGate()
    gate.load_state_dict(checkpoint["model_state_dict"])
    gate.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(
            gate.network(data["inputs"].float())
        ).flatten().numpy()

    targets = data["targets"].flatten().numpy().astype(int)

    print(f"{'Threshold':>10} {'Precision':>11} {'Recall':>9} "
          f"{'F1':>9} {'Fusion %':>10}")

    best = (0, None)

    for threshold in [i / 100 for i in range(10, 51, 2)]:
        predictions = (probabilities >= threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="binary", zero_division=0
        )

        fusion_rate = predictions.mean() * 100

        print(f"{threshold:10.2f} {precision:11.3f} "
              f"{recall:9.3f} {f1:9.3f} {fusion_rate:9.2f}%")

        if f1 > best[0]:
            best = (f1, threshold)

    print(f"\nBest validation F1: {best[0]:.4f}")
    print(f"Best threshold: {best[1]}")


if __name__ == "__main__":
    main()
