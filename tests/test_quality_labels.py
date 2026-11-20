"""Behavior-locking tests for the ORIGINAL quality-label methodology."""
import numpy as np
import pytest

MODULE = 'prototyping/synthetic_data_degradation/quality_labels.py'

@pytest.fixture
def labels(load_legacy):
    return load_legacy(MODULE)

def test_empty_mask_has_zero_coverage(labels):
    result = labels.mask_to_coverage(np.zeros((16, 16), dtype=np.uint8))
    assert result.shape == (8, 8)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.zeros((8, 8)))

def test_full_mask_has_unit_coverage(labels):
    result = labels.mask_to_coverage(np.ones((16, 16), dtype=np.uint8))
    np.testing.assert_array_equal(result, np.ones((8, 8)))

def test_single_patch_coverage(labels):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:2, :2] = 1
    result = labels.mask_to_coverage(mask)
    expected = np.zeros((8, 8), dtype=np.float32)
    expected[0, 0] = 1
    np.testing.assert_array_equal(result, expected)

def test_fractional_coverage(labels):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0, 0] = 255
    result = labels.mask_to_coverage(mask)
    assert result[0, 0] == pytest.approx(0.25)
    assert result.sum() == pytest.approx(0.25)

def test_mask_validation(labels):
    with pytest.raises(ValueError):
        labels.mask_to_coverage(None)
    with pytest.raises(ValueError):
        labels.mask_to_coverage(np.zeros((2, 2, 1)))

def test_quality_formula(labels):
    base = np.full((8, 8), 0.8, dtype=np.float32)
    coverage = np.full((8, 8), 0.5, dtype=np.float32)
    result = labels.compute_final_quality(base, coverage, 0.75)
    np.testing.assert_allclose(result, 0.5, rtol=1e-6)
    assert result.dtype == np.float32

def test_zero_severity_preserves_baseline(labels):
    base = np.full((8, 8), 0.7, dtype=np.float32)
    result = labels.compute_final_quality(base, np.ones((8, 8)), 0)
    np.testing.assert_allclose(result, base)

def test_full_severity_full_coverage_zeroes_quality(labels):
    result = labels.compute_final_quality(np.ones((8, 8)), np.ones((8, 8)), 1)
    np.testing.assert_array_equal(result, np.zeros((8, 8)))

def test_quality_input_validation(labels):
    with pytest.raises(ValueError):
        labels.compute_final_quality(np.ones((8, 8)), np.ones((7, 8)), 0.5)
    with pytest.raises(ValueError):
        labels.compute_final_quality(np.ones((8, 8)), np.ones((8, 8)), -0.1)
    with pytest.raises(ValueError):
        labels.compute_final_quality(np.ones((8, 8)), np.ones((8, 8)), 1.1)
