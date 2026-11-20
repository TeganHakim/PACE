"""No-training compatibility checks for reorganized training entry points."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,legacy", [
    ("quality_estimator", "prototyping/quality_estimator/train.py"),
    ("reliability_gate", "prototyping/reliability_gate/train.py"),
])
def test_entrypoint_targets_unchanged_training_script(name, legacy):
    source = (ROOT / "pace" / "training" / f"{name}.py").read_text()
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_legacy"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == legacy
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("name", ["quality_estimator", "reliability_gate"])
def test_entrypoint_help_exits_without_training(name):
    result = subprocess.run(
        [sys.executable, "-m", f"pace.training.{name}", "--help"],
        cwd=ROOT, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "usage:" in result.stdout.lower()
