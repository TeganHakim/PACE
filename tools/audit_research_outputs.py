"""Read-only audit of PACE research evidence and cleanup candidates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = {
    'full_test_comparison': 'evaluation/comparisons/both_baselines_vs_pace_test_set/comparison_report.md',
    'condition_analysis_script': 'evaluation/condition_analysis.py',
    'condition_analysis_instructions': 'evaluation/READ_CONDITION_ANALYSIS.md',
    'artifact_manifest': 'reports/refactor/phase6_artifact_manifest.json',
    'experiment_inventory': 'reports/refactor/phase6_experiment_inventory',
}
CONDITION_HINTS = ('condition', 'sensor_noise', 'sensor-noise', 'subgroup')
TEMP_SUFFIXES = ('.bak', '.zip', '.pyc')
TEMP_NAMES = ('apply_phase', 'README-phase', 'README-step', 'inventory_training.py')
EXCLUDED = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'node_modules', 'processed_dataset', 'processed_dataset_realistic_degraded_v2', 'VisDrone-DroneVehicle'}

def scan(root):
    evidence = {name: {'path': rel, 'exists': (root / rel).exists()} for name, rel in EVIDENCE.items()}
    candidates, condition_files = [], []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDED for part in rel.parts):
            continue
        name = p.name.lower()
        if any(hint in name for hint in CONDITION_HINTS) and p.suffix.lower() in {'.csv', '.json', '.md', '.png', '.pdf', '.svg'}:
            condition_files.append(str(rel))
        if p.suffix.lower() in TEMP_SUFFIXES or p.name.startswith(TEMP_NAMES) or p.name in TEMP_NAMES:
            candidates.append({'path': str(rel), 'bytes': p.stat().st_size, 'reason': 'possible temporary file; review manually'})
    return {'schema_version': 1, 'evidence': evidence,
            'condition_related_files': sorted(condition_files),
            'cleanup_candidates': sorted(candidates, key=lambda x:x['path']),
            'warning': 'Candidates are informational only. Do not delete without reviewing references and Git tracking.'}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', default='reports/refactor/phase6_research_audit')
    args=parser.parse_args(argv)
    root=args.root.resolve()
    report=scan(root)
    output=root/args.output
    output.mkdir(parents=True,exist_ok=True)
    (output/'research_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'cleanup_candidates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','reason'])
        writer.writeheader();writer.writerows(report['cleanup_candidates'])
    missing=[key for key, val in report['evidence'].items() if not val['exists']]
    print('PASS: read-only research artifact audit')
    print('Condition-related files:',len(report['condition_related_files']))
    print('Cleanup candidates (nothing deleted):',len(report['cleanup_candidates']))
    print('Missing evidence:', ', '.join(missing) if missing else 'none')
    print('Saved:',output)
    return 0

if __name__=='__main__':raise SystemExit(main())
