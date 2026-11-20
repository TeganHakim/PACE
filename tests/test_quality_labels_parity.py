"""Check that the new package matches the original research implementation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pace.data import quality_labels as packaged

LEGACY = (Path(__file__).resolve().parents[1] /
          'prototyping/synthetic_data_degradation/quality_labels.py')


@pytest.fixture(scope='module')
def legacy():
    spec = importlib.util.spec_from_file_location('legacy_quality_labels_parity', LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_source_matches_packaged_source():
    """Guard against accidentally editing the copied research implementation."""
    assert LEGACY.read_bytes() == Path(packaged.__file__).read_bytes()


@pytest.mark.parametrize('shape', [(512, 640), (17, 23), (8, 8)])
@pytest.mark.parametrize('grid', [4, 8])
def test_mask_coverage_parity(legacy, shape, grid):
    mask = np.random.default_rng(42).integers(0, 2, size=shape, dtype=np.uint8)
    np.testing.assert_array_equal(
        packaged.mask_to_coverage(mask, grid), legacy.mask_to_coverage(mask, grid)
    )


@pytest.mark.parametrize('severity', [0.0, 0.25, 0.8, 1.0])
def test_quality_parity(legacy, severity):
    rng = np.random.default_rng(42)
    baseline = rng.random((8, 8), dtype=np.float32)
    coverage = rng.random((8, 8), dtype=np.float32)
    np.testing.assert_array_equal(
        packaged.compute_final_quality(baseline, coverage, severity),
        legacy.compute_final_quality(baseline, coverage, severity),
    )


@pytest.mark.parametrize('grid', [4, 8, 16])
def test_ir_baseline_parity(legacy, grid):
    np.testing.assert_array_equal(
        packaged.create_ir_baseline(grid), legacy.create_ir_baseline(grid)
    )
