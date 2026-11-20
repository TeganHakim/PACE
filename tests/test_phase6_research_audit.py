import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit_research',ROOT/'tools/audit_research_outputs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_missing_evidence_reported(tmp_path):
    r=m.scan(tmp_path)
    assert not r['evidence']['full_test_comparison']['exists']

def test_existing_evidence(tmp_path):
    p=tmp_path/m.EVIDENCE['full_test_comparison'];p.parent.mkdir(parents=True);p.write_text('result')
    assert m.scan(tmp_path)['evidence']['full_test_comparison']['exists']

def test_condition_csv_discovered(tmp_path):
    (tmp_path/'rgb_sensor_noise_condition.csv').write_text('x')
    assert 'rgb_sensor_noise_condition.csv' in m.scan(tmp_path)['condition_related_files']

def test_temp_candidate_detected(tmp_path):
    (tmp_path/'file.phase5.bak').write_text('x')
    assert any(x['path']=='file.phase5.bak' for x in m.scan(tmp_path)['cleanup_candidates'])

def test_non_temp_not_candidate(tmp_path):
    (tmp_path/'model.py').write_text('x')
    assert not m.scan(tmp_path)['cleanup_candidates']

def test_excluded_dataset_not_scanned(tmp_path):
    p=tmp_path/'processed_dataset'/'sensor_noise_condition.csv';p.parent.mkdir();p.write_text('x')
    assert not m.scan(tmp_path)['condition_related_files']

def test_report_output(tmp_path):
    assert m.main(['--root',str(tmp_path)])==0
    p=tmp_path/'reports/refactor/phase6_research_audit/research_evidence.json'
    assert json.loads(p.read_text())['schema_version']==1

def test_csv_header(tmp_path):
    m.main(['--root',str(tmp_path)])
    assert (tmp_path/'reports/refactor/phase6_research_audit/cleanup_candidates.csv').read_text().startswith('path,bytes,reason')
