import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("audit8",ROOT/"tools/phase8_scientific_audit.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_missing_report(tmp_path):
 r=m.audit(tmp_path)
 assert not r["exists"] and r["sample_count"] is None
def test_parses_report(tmp_path):
 p=tmp_path/m.REPORT;p.parent.mkdir(parents=True)
 p.write_text("Compared **8980 matched samples**.\n| Metric | CrossFuse | PACE | Addition |\n|---|---:|---:|---:|\n| mAP@50 | 0.4712 | 0.4544 | 0.4417 |\n")
 r=m.audit(tmp_path)
 assert r["sample_count"]==8980
 assert r["pace_beats_addition_on_full_test_map50"]
 assert r["crossfuse_beats_pace_on_full_test_map50"]
 assert not r["condition_specific_rgb_noise_reproduced"]
def test_real_report_exists():
 assert (ROOT/m.REPORT).is_file()
