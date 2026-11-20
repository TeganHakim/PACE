"""Preflight is importable and does not require CUDA on Mac."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("jetson_preflight",ROOT/"evaluation/jetson_preflight.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
def test_checkpoint_manifest():
    assert set(mod.CHECKPOINTS)=={"crossfuse","quality_rgb","quality_ir","reliability_gate"}
def test_preflight_reports_device_capabilities():
    report=mod.inspect(require_cuda=False)
    assert "cuda_available" in report and "mps_available" in report
    assert isinstance(report["checkpoints"],dict)
def test_cuda_requirement_is_explicit():
    report=mod.inspect(require_cuda=True)
    if not report["cuda_available"]:
        assert "CUDA unavailable" in report["errors"]
