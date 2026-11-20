import importlib.util,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("cleanup8",ROOT/"tools/phase8_cleanup_audit.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_inventory_is_read_only():
 before=subprocess.run(["git","status","--porcelain"],cwd=ROOT,capture_output=True,text=True,check=True).stdout
 r=m.inventory(ROOT)
 after=subprocess.run(["git","status","--porcelain"],cwd=ROOT,capture_output=True,text=True,check=True).stdout
 assert before==after and r["nothing_deleted"]
def test_inventory_shape():
 assert isinstance(m.inventory(ROOT)["working_tree"],list)
