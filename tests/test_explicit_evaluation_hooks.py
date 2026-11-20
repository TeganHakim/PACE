from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
def test_pace_explicit_hooks():
    s=(ROOT/"pace/evaluate.py").read_text()
    assert "'--paired-jpg-samples'" in s
    assert "'--save-fused'" in s
    assert "if args.paired_jpg_samples:" in s
    assert "if args.save_fused:" in s

def test_crossfuse_explicit_hooks():
    s=(ROOT/"baseline/CrossFuse/evaluate.py").read_text()
    assert "'--save-fused'" in s and "if a.save_fused:" in s

def test_wrappers_no_source_exec():
    for name in ("pace_e2e.py","crossfuse_e2e.py"):
        s=(ROOT/"evaluation"/name).read_text()
        assert "runpy.run_path" in s and "exec(compile(" not in s and "def inject(" not in s

def test_source_changes_exactly_match_migration():
    spec=importlib.util.spec_from_file_location("migration",ROOT/"apply_phase5_batch2.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for path,transform in ((module.p,module.pace),(module.c,module.cross)):
        backup=path.with_name(path.name+".phase5.bak")
        assert backup.is_file()
        assert transform(backup.read_text())==path.read_text()
