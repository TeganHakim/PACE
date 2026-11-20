"""Pretrained checkpoint parity across legacy and pace.models import paths.

Run from the repository root. Tests deliberately require real checkpoints.
No checkpoints are written or modified.
"""
from pathlib import Path

import pytest
import torch

from prototyping.quality_estimator.model import SpatialQualityEstimator as LegacyQuality
from prototyping.reliability_gate.reliability_gate import ReliabilityGate as LegacyGate
from pace.models import SpatialQualityEstimator, ReliabilityGate

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "prototyping"

def _checkpoint(path):
    assert path.is_file(), f"Required pretrained checkpoint missing: {path}"
    obj = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(obj, dict), f"Unexpected checkpoint type: {path}"
    return obj.get("model_state_dict", obj), obj

@pytest.mark.parametrize("modality", ["rgb", "ir"])
def test_quality_estimator_pretrained_checkpoint_parity(modality):
    path = CHECKPOINTS / "quality_estimator" / "checkpoints" / f"quality_{modality}.pt"
    weights, metadata = _checkpoint(path)
    channels = int(metadata.get("in_channels", 132))
    legacy = LegacyQuality(in_channels=channels).eval()
    organized = SpatialQualityEstimator(in_channels=channels).eval()
    legacy.load_state_dict(weights, strict=True)
    organized.load_state_dict(weights, strict=True)
    assert list(legacy.state_dict()) == list(organized.state_dict())
    torch.manual_seed(130 + (modality == "ir"))
    x = torch.randn(2, channels, 16, 16)
    with torch.inference_mode():
        expected = legacy(x)
        actual = organized(x)
    assert expected.shape == (2, 1, 8, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

def test_reliability_gate_pretrained_checkpoint_parity():
    path = CHECKPOINTS / "reliability_gate" / "checkpoints" / "reliability_gate_best.pt"
    weights, metadata = _checkpoint(path)
    hidden_dim = int(metadata.get("hidden_dim", 16))
    legacy = LegacyGate(hidden_dim=hidden_dim).eval()
    organized = ReliabilityGate(hidden_dim=hidden_dim).eval()
    legacy.load_state_dict(weights, strict=True)
    organized.load_state_dict(weights, strict=True)
    assert list(legacy.state_dict()) == list(organized.state_dict())
    torch.manual_seed(132)
    rgb, ir = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    with torch.inference_mode():
        expected = legacy(rgb, ir)
        actual = organized(rgb, ir)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all((actual == 0) | (actual == 1))

def test_fusion_class_import_is_identical():
    from prototyping.selective_fusion.local_crossfuse import LocalCrossFuse as LegacyFusion
    from pace.models import LocalCrossFuse
    assert LocalCrossFuse is LegacyFusion
