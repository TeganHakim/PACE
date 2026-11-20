"""Phase 4 final compatibility tests: no training is performed."""
import ast
import subprocess
import sys
from pathlib import Path

from pace.training.contract_audit import ROOT, REQUIRED, audit_scripts, audit_checkpoints

def test_quality_training_checkpoint_schema():
    result=audit_scripts()["quality_estimator"]
    assert REQUIRED["quality_estimator"].issubset(result["checkpoint_fields"])

def test_gate_training_checkpoint_schema():
    result=audit_scripts()["reliability_gate"]
    assert REQUIRED["reliability_gate"].issubset(result["checkpoint_fields"])

def test_existing_pretrained_checkpoints_remain_readable():
    checkpoints=audit_checkpoints()
    assert all(item["state_dict_tensors"]>0 for item in checkpoints.values())

def test_original_training_sources_match_phase4_inventory():
    import json
    inventory=ROOT/"reports/refactor/phase4_training_inventory.json"
    if not inventory.is_file():
        raise AssertionError("Run python inventory_training.py first")
    data=json.loads(inventory.read_text())
    originals={x["path"]:x["sha256"] for x in data["scripts"]}
    for item in audit_scripts().values():
        assert originals[item["path"]]==item["sha256"],f"Training source changed since inventory: {item['path']}"

def test_training_entrypoint_help_is_safe():
    for module in ("pace.training.quality_estimator","pace.training.reliability_gate"):
        result=subprocess.run([sys.executable,"-m",module,"--help"],cwd=ROOT,capture_output=True,text=True,timeout=40)
        assert result.returncode==0,(module,result.stderr)
        assert "usage:" in result.stdout.lower()
