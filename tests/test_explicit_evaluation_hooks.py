from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_pace_explicit_hooks():
    s = (ROOT / "pace/evaluate.py").read_text()
    assert "'--paired-jpg-samples'" in s
    assert "'--save-fused'" in s
    assert "if args.paired_jpg_samples:" in s
    assert "if args.save_fused:" in s


def test_crossfuse_explicit_hooks():
    s = (ROOT / "baseline/CrossFuse/evaluate.py").read_text()
    assert "'--save-fused'" in s
    assert "if a.save_fused:" in s


def test_wrappers_no_source_exec():
    for name in ("pace_e2e.py", "crossfuse_e2e.py"):
        s = (ROOT / "evaluation" / name).read_text()
        assert "runpy.run_path" in s
        assert "exec(compile(" not in s
        assert "def inject(" not in s


def test_evaluation_entrypoints_parse():
    """Evaluation entrypoints must remain valid Python without migration scripts."""
    paths = (
        "pace/evaluate.py",
        "baseline/CrossFuse/evaluate.py",
        "evaluation/pace_e2e.py",
        "evaluation/crossfuse_e2e.py",
    )
    for name in paths:
        path = ROOT / name
        ast.parse(path.read_text(), filename=str(path))
