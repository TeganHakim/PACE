"""No-inference evaluation regression and compatibility checks."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("pace_eval_verify",ROOT/"evaluation/verify_regression.py")
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

def test_all_methods_accounted_for():
    assert mod.METHODS==("pace","crossfuse","addition")

def test_numeric_regression_detects_change():
    assert mod.compare_values({"map50":0.5},{"map50":0.6})

def test_numeric_regression_tolerance():
    assert not mod.compare_values({"map50":0.5},{"map50":0.5+1e-12})

def test_regression_detects_missing_method(tmp_path):
    checked,failures=mod.verify(tmp_path/"reference",tmp_path/"candidate")
    assert not checked and failures

def test_duplicate_sample_ids_rejected(tmp_path):
    path=tmp_path/"per_sample.csv"
    path.write_text("sample_id,value\n00001,1\n00001,2\n")
    with pytest.raises(ValueError,match="Duplicate"):
        mod.ids(path)

def test_evaluation_entrypoints_and_condition_analysis_preserved():
    for path in (
        "evaluation/pace_e2e.py",
        "evaluation/crossfuse_e2e.py",
        "evaluation/addition_e2e.py",
        "evaluation/eval_metrics.py",
        "evaluation/condition_analysis.py",
        "evaluation/READ_CONDITION_ANALYSIS.md",
    ):
        assert (ROOT/path).is_file(),path

def test_evaluator_explicit_hooks_present():
    assert "if args.paired_jpg_samples:" in (ROOT/"pace/evaluate.py").read_text()
    assert "if args.save_fused:" in (ROOT/"pace/evaluate.py").read_text()
    assert "if a.save_fused:" in (ROOT/"baseline/CrossFuse/evaluate.py").read_text()
