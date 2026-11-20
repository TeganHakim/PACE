import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inventory_experiments', ROOT / 'tools/inventory_experiments.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_deterministic_order(tmp_path):
    (tmp_path/'z.py').write_text('print(1)')
    (tmp_path/'a.csv').write_text('x,y\n')
    rows = mod.inventory(tmp_path)
    assert [r['path'] for r in rows] == ['a.csv', 'z.py']
    assert rows == mod.inventory(tmp_path)

def test_excludes_datasets_and_backups(tmp_path):
    (tmp_path/'processed_dataset').mkdir()
    (tmp_path/'processed_dataset'/'large.npy').write_bytes(b'hi')
    (tmp_path/'x.py.bak').write_text('backup')
    (tmp_path/'x.py').write_text('code')
    assert [r['path'] for r in mod.inventory(tmp_path)] == ['x.py']

def test_hash_matches_content(tmp_path):
    (tmp_path/'x.json').write_text('{}')
    row = mod.inventory(tmp_path)[0]
    assert row['sha256'] == mod.sha256(tmp_path/'x.json')

def test_size_limit_marks_unhashed(tmp_path):
    (tmp_path/'x.pt').write_bytes(b'checkpoint')
    row = mod.inventory(tmp_path, hash_limit_mb=0)[0]
    assert row['sha256'] is None and row['hash_status'] == 'skipped_size_limit'

def test_reports_roundtrip(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    (source/'script.py').write_text('pass')
    dest = tmp_path/'reports'
    counts = mod.write_inventory(mod.inventory(source), dest)
    assert counts['scripts'] == 1
    assert json.loads((dest/'experiment_inventory.json').read_text())['schema_version'] == 1
    assert (dest/'experiment_inventory.csv').exists()

def test_output_does_not_self_inventory(tmp_path):
    (tmp_path/'a.py').write_text('pass')
    dest = tmp_path/'reports/refactor/phase6_experiment_inventory'
    mod.write_inventory(mod.inventory(tmp_path), dest)
    assert [r['path'] for r in mod.inventory(tmp_path)] == ['a.py']
