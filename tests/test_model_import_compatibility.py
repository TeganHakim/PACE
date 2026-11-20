"""Architecture compatibility checks; no pretrained assets required."""
import torch

from pace.models import SpatialQualityEstimator, ReliabilityGate, LocalCrossFuse
from prototyping.quality_estimator.model import SpatialQualityEstimator as OriginalQuality
from prototyping.reliability_gate.reliability_gate import ReliabilityGate as OriginalGate
from prototyping.selective_fusion.local_crossfuse import LocalCrossFuse as OriginalFusion


def test_public_imports_reuse_original_classes():
    assert SpatialQualityEstimator is OriginalQuality
    assert ReliabilityGate is OriginalGate
    assert LocalCrossFuse is OriginalFusion


def test_quality_state_dict_strict_round_trip():
    original = OriginalQuality(in_channels=132).eval()
    migrated = SpatialQualityEstimator(in_channels=132).eval()
    migrated.load_state_dict(original.state_dict(), strict=True)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(migrated.state_dict()[key], value, rtol=0, atol=0)


def test_gate_state_dict_strict_round_trip():
    original = OriginalGate(hidden_dim=16).eval()
    migrated = ReliabilityGate(hidden_dim=16).eval()
    migrated.load_state_dict(original.state_dict(), strict=True)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(migrated.state_dict()[key], value, rtol=0, atol=0)
