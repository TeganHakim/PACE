import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("p6",ROOT/"tools/verify_research_artifacts.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_stream_hash(tmp_path):
 p=tmp_path/"x";p.write_bytes(b"abc"*100)
 assert m.sha(p,2)==m.sha(p,1024)
def test_present(tmp_path):
 (tmp_path/"x").write_text("a")
 assert m.collect(tmp_path,{"sources":{"x":"x"}})["artifacts"]["sources"]["x"]["exists"]
def test_missing(tmp_path):
 assert not m.collect(tmp_path,{"sources":{"x":"x"}})["artifacts"]["sources"]["x"]["exists"]
def test_same(tmp_path):
 a=m.collect(tmp_path,{"sources":{"x":"x"}})
 assert m.differences(a,a)==[]
def test_change(tmp_path):
 p=tmp_path/"x";p.write_text("a")
 a=m.collect(tmp_path,{"sources":{"x":"x"}})
 p.write_text("b")
 assert m.differences(a,m.collect(tmp_path,{"sources":{"x":"x"}}))==["sources/x"]
def test_deleted(tmp_path):
 p=tmp_path/"x";p.write_text("a")
 a=m.collect(tmp_path,{"sources":{"x":"x"}})
 p.unlink()
 assert m.differences(a,m.collect(tmp_path,{"sources":{"x":"x"}}))==["sources/x"]
def test_cli(tmp_path):
 assert m.main(["--root",str(tmp_path)])==0
 assert (tmp_path/"reports/refactor/phase6_artifact_manifest.json").exists()
def test_cli_verify(tmp_path):
 m.main(["--root",str(tmp_path)])
 assert m.main(["--root",str(tmp_path),"--verify",str(tmp_path/"reports/refactor/phase6_artifact_manifest.json")])==0
