"""Small shared helpers; no model code lives here."""
from __future__ import annotations
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = Path(__file__).resolve().parent
METHODS = ('pace', 'crossfuse', 'addition')


def command(script: Path, *args: object) -> None:
    cmd = [sys.executable, str(script), *(str(v) for v in args)]
    print('\n$ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def selected_ids(dataset: Path, maximum: int) -> list[str]:
    rgb = dataset/'test'/'rgb'
    ir = dataset/'test'/'ir'
    if not rgb.is_dir() or not ir.is_dir():
        raise FileNotFoundError(f'Missing test/rgb or test/ir inside {dataset}')
    ids = sorted(p.stem for p in ir.glob('*.jpg') if (rgb/p.name).is_file())
    if not ids:
        raise ValueError('No paired JPEG test images found')
    return ids[:maximum] if maximum else ids


def check_run(path: Path, ids: list[str]) -> None:
    required = ('summary.json', 'latency.json', 'per_sample.csv', 'predictions.json')
    for name in required:
        if not (path/name).is_file():
            raise FileNotFoundError(f'Incomplete evaluation: {path/name}')
    rows = read_csv(path/'per_sample.csv')
    found = [row['sample_id'] for row in rows]
    if found != ids:
        raise ValueError(f'IDs differ in {path}: expected {len(ids)} ordered IDs, got {len(found)}')
