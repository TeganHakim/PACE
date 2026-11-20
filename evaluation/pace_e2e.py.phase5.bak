"""Execute the current PACE evaluator, patching only sample enumeration and fused saving.

No model layers, checkpoint loading, normalization, inference, or detector logic is changed.
The compatibility edits are applied to an in-memory copy; the repository file is untouched.
Fail fast if upstream source changes rather than silently running a different pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path
from common import ROOT

SOURCE = ROOT/'pace'/'evaluate.py'


def inject(source: str) -> str:
    original = 'sample_list = load_samples(args.split, args.max_samples)'
    replacement = '''sample_list = []
    root = args.dataset_root / args.split
    for ir_path in sorted((root / 'ir').glob('*.jpg')):
        rgb_path = root / 'rgb' / ir_path.name
        if rgb_path.is_file():
            sample_list.append((ir_path.stem, rgb_path, ir_path, None, None))
            if args.max_samples and len(sample_list) >= args.max_samples:
                break
    if not sample_list:
        raise RuntimeError('No RGB/IR JPG pairs in ' + str(root))'''
    if source.count(original) != 1:
        raise RuntimeError('PACE evaluator sample-selection anchor changed; review pace/evaluate.py')
    source = source.replace(original, replacement)
    original = '            rows.append(row)'
    replacement = '''            (visualization_dir.parent / 'fused').mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(visualization_dir.parent / 'fused' / f'{sample_id}.png'), fused_image):
                raise IOError(f'Failed to save PACE fused image {sample_id}')
            rows.append(row)'''
    if source.count(original) != 1:
        raise RuntimeError('PACE evaluator per-sample anchor changed; review pace/evaluate.py')
    return source.replace(original, replacement)


def main() -> None:
    source = inject(SOURCE.read_text())
    sys.path.insert(0, str(SOURCE.parent))
    namespace = {'__name__': '__main__', '__file__': str(SOURCE), '__package__': None}
    exec(compile(source, str(SOURCE), 'exec'), namespace)


if __name__ == '__main__':
    main()
