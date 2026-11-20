"""Read-only, deterministic inventory of PACE experiment scripts and artifacts."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

SCRIPT_SUFFIXES = {'.py', '.sh', '.ipynb'}
ARTIFACT_SUFFIXES = {'.json', '.csv', '.png', '.jpg', '.jpeg', '.pdf', '.npy', '.npz', '.pt', '.pth', '.model', '.yaml', '.yml', '.toml'}
EXCLUDE_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache', 'node_modules', 'processed_dataset', 'processed_dataset_realistic_degraded_v2', 'VisDrone-DroneVehicle', 'datasets', 'data'}
EXCLUDE_FILES = {'.DS_Store'}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def inventory(root, *, hash_limit_mb=32, include_large=False):
    root = Path(root).resolve()
    records = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.'))
        for filename in sorted(files):
            if filename in EXCLUDE_FILES or filename.endswith(('.bak', '.pyc')):
                continue
            path = Path(base) / filename
            rel = path.relative_to(root).as_posix()
            if rel.startswith(('reports/refactor/phase6_', 'evaluation/results/', 'evaluation/comparisons/')):
                # Include research result directories, but never inventory our own output.
                if rel.startswith('reports/refactor/phase6_'):
                    continue
            suffix = path.suffix.lower()
            if suffix not in SCRIPT_SUFFIXES | ARTIFACT_SUFFIXES:
                continue
            stat = path.stat()
            kind = 'script' if suffix in SCRIPT_SUFFIXES else 'artifact'
            size = stat.st_size
            digest = sha256(path) if (include_large or size <= hash_limit_mb * 1024 * 1024) else None
            records.append({'path': rel, 'kind': kind, 'extension': suffix, 'bytes': size, 'sha256': digest, 'hash_status': 'hashed' if digest else 'skipped_size_limit'})
    return sorted(records, key=lambda r: r['path'])

def write_inventory(records, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'experiment_inventory.json').open('w') as f:
        json.dump({'schema_version': 1, 'files': records}, f, indent=2)
        f.write('\n')
    with (output / 'experiment_inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path','kind','extension','bytes','sha256','hash_status'])
        writer.writeheader()
        writer.writerows(records)
    counts = {'scripts': sum(r['kind']=='script' for r in records), 'artifacts': sum(r['kind']=='artifact' for r in records), 'unhashed_large_files': sum(r['sha256'] is None for r in records)}
    (output / 'summary.json').write_text(json.dumps(counts, indent=2) + '\n')
    return counts

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, default=Path('reports/refactor/phase6_experiment_inventory'))
    p.add_argument('--hash-limit-mb', type=int, default=32)
    p.add_argument('--include-large', action='store_true', help='Hash large model/artifact files too (slower)')
    a = p.parse_args()
    if a.hash_limit_mb < 0: p.error('--hash-limit-mb must be >= 0')
    records = inventory(a.root, hash_limit_mb=a.hash_limit_mb, include_large=a.include_large)
    print('PASS: read-only experiment inventory', write_inventory(records, a.output))
    print('Saved:', a.output)

if __name__ == '__main__':
    main()
