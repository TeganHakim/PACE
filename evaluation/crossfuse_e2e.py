"""Current CrossFuse evaluator with an in-memory output-only fused-image hook."""
from __future__ import annotations
import sys
from common import ROOT

SOURCE = ROOT/'baseline'/'CrossFuse'/'evaluate.py'


def inject(source: str) -> str:
    original = '                rows.append(row)'
    replacement = '''                (out/'fused').mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(out/'fused'/f'{sample_id}.png'), fused_numpy(output['out'])):
                    raise IOError(f'Failed to save CrossFuse fused image {sample_id}')
                rows.append(row)'''
    if source.count(original) != 1:
        raise RuntimeError('CrossFuse per-sample anchor changed; review baseline/CrossFuse/evaluate.py')
    return source.replace(original, replacement)


def main() -> None:
    source = inject(SOURCE.read_text())
    sys.path.insert(0, str(SOURCE.parent))
    namespace = {'__name__': '__main__', '__file__': str(SOURCE), '__package__': None}
    exec(compile(source, str(SOURCE), 'exec'), namespace)


if __name__ == '__main__':
    main()
