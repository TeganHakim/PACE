"""Condition-analysis scientific contracts, no inference or large datasets."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("condition_analysis",ROOT/"evaluation/condition_analysis.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def example(predictions, ground_truth):
    return {"sample_id":"00001","predictions":predictions,"ground_truth":ground_truth}

GT={"class_id":0,"cx":.5,"cy":.5,"width":.25,"height":.25}
BOX=[96.,96.,160.,160.]
def pred(cls=0,score=.9,box=BOX):
    return {"class_id":cls,"score":score,"box_xyxy":box}

def test_box_iou_identity_and_disjoint():
    assert mod.box_iou(BOX,BOX)==1.
    assert mod.box_iou(BOX,[0,0,10,10])==0.

def test_matching_true_positive():
    result=mod.evaluate_boxes(example([pred()],[GT]),.5,.5)
    assert (result["tp"],result["fp"],result["fn"])==(1,0,0)
    assert result["f1"]==1.

def test_class_mismatch_counts_fp_and_fn():
    result=mod.evaluate_boxes(example([pred(cls=1)],[GT]),.5,.5)
    assert (result["tp"],result["fp"],result["fn"])==(0,1,1)

def test_duplicate_predictions_are_one_to_one():
    result=mod.evaluate_boxes(example([pred(),pred(score=.8)],[GT]),.5,.5)
    assert (result["tp"],result["fp"],result["fn"])==(1,1,0)

def test_confidence_filter():
    result=mod.evaluate_boxes(example([pred(score=.49)],[GT]),.5,.5)
    assert (result["tp"],result["fp"],result["fn"])==(0,0,1)

def test_empty_empty_f1_is_nan_not_perfect():
    result=mod.evaluate_boxes(example([],[]),.5,.5)
    assert math.isnan(result["f1"])

def test_duplicate_ids_rejected(tmp_path):
    p=tmp_path/"predictions.json"
    p.write_text(json.dumps([example([],[]),example([],[])]))
    with pytest.raises(ValueError,match="Duplicate"):
        mod.records_by_id(p)

def test_bootstrap_deterministic():
    a=mod.bootstrap_ci([0.2,0.1,-0.1,0.3],100,42)
    b=mod.bootstrap_ci([0.2,0.1,-0.1,0.3],100,42)
    assert a==b

def test_bootstrap_excludes_nonfinite():
    a=mod.bootstrap_ci([0.25,float("nan")],50,42)
    assert a==(0.25,0.25)

def test_bootstrap_zero_returns_nan():
    a=mod.bootstrap_ci([1.,2.],0,42)
    assert all(math.isnan(v) for v in a)
