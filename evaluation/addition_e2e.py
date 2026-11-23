"""Exact existing integer 50/50 addition, followed by existing detector evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import ROOT, command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--visualizations', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--nms-iou', type=float, default=.7)
    args = parser.parse_args()
    fused = args.output/'fused'
    shared = ('--dataset-root', args.dataset_root, '--split', 'test', '--max-samples', args.max_samples)
    command(ROOT/'baseline'/'Addition'/'fuse_color.py', *shared, '--output', fused)
    command(ROOT/'baseline'/'Addition'/'evaluate_color.py', *shared, '--fused-dir', fused,
            '--output', args.output, '--device', args.device, '--visualizations', args.visualizations,
            '--warmup', args.warmup, '--seed', args.seed, '--confidence', args.confidence,
            '--nms-iou', args.nms_iou)


if __name__ == '__main__':
    main()
