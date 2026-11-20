import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("readme_updater",ROOT/"tools/apply_phase7_readme.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_index_preserves_existing_content():
 t="# Custom local README\n\nUser content"
 assert t in m.update(t)
def test_idempotent():
 t=m.update("# PACE\n")
 assert m.update(t)==t
def test_docs_exist():
 for name in ("ARCHITECTURE.md","WORKFLOWS.md","ENVIRONMENTS.md"):
  assert (ROOT/"docs/refactor"/name).is_file()
def test_readme_links_resolve():
 t=m.update("# PACE\n")
 for name in ("ARCHITECTURE.md","WORKFLOWS.md","ENVIRONMENTS.md"):
  assert "docs/refactor/"+name in t
def test_mac_jetson_distinction():
 t=(ROOT/"docs/refactor/ENVIRONMENTS.md").read_text()
 assert "MPS" in t and "JetPack" in t and "--require-cuda" in t
def test_evaluation_commands_present():
 t=(ROOT/"docs/refactor/WORKFLOWS.md").read_text()
 assert "evaluation/eval_metrics.py" in t and "evaluation/verify_regression.py" in t
def test_no_fake_jetson_validation():
 t=(ROOT/"docs/refactor/ENVIRONMENTS.md").read_text()
 assert "not yet certified" in t
