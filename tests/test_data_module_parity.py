"""Behavior and seeded-randomness parity for migrated data modules."""
import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def original(name):
    path = ROOT / 'prototyping' / 'synthetic_data_degradation' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'_legacy_{name}_step3', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def modules():
    from pace.data import degradation, lighting
    return (original('degradation'), degradation, original('lighting'), lighting)


def seed():
    random.seed(2731)
    np.random.seed(2731)


@pytest.mark.parametrize('shape', [(32, 40), (32, 40, 3)])
@pytest.mark.parametrize('kind', ['water_blur', 'motion_blur', 'sensor_noise', 'saturation'])
def test_degradation_parity(modules, shape, kind):
    old, new, _, _ = modules
    image = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape).astype(np.uint8)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[3:24, 5:31] = 1
    seed()
    a, meta_a = old.apply_degradation(image, kind, 0.67, mask)
    state_a = (random.getstate(), np.random.get_state())
    seed()
    b, meta_b = new.apply_degradation(image, kind, 0.67, mask)
    state_b = (random.getstate(), np.random.get_state())
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(meta_a['mask'], meta_b['mask'])
    assert meta_a['type'] == meta_b['type']
    assert meta_a['severity'] == meta_b['severity']
    assert state_a[0] == state_b[0]
    assert state_a[1][0] == state_b[1][0]
    np.testing.assert_array_equal(state_a[1][1], state_b[1][1])
    assert state_a[1][2:] == state_b[1][2:]


@pytest.mark.parametrize('shape', [(32, 40), (64, 80)])
def test_mask_generation_parity(modules, shape):
    old, new, _, _ = modules
    seed()
    a = old.generate_mask(*shape)
    seed()
    b = new.generate_mask(*shape)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('value', [0, 10, 60, 200])
def test_lighting_parity(modules, value):
    _, _, old, new = modules
    image = np.full((32, 40, 3), value, dtype=np.uint8)
    for function in ('compute_luminance', 'compute_lighting_quality', 'compute_rgb_quality_map', 'sufficiently_lit_mask'):
        np.testing.assert_array_equal(getattr(old, function)(image), getattr(new, function)(image))
    a = old.compute_rgb_baseline_quality(image)
    b = new.compute_rgb_baseline_quality(image)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_module_source_identical():
    for name in ('degradation', 'lighting'):
        old = ROOT / 'prototyping' / 'synthetic_data_degradation' / f'{name}.py'
        new = ROOT / 'pace' / 'data' / f'{name}.py'
        assert old.read_bytes() == new.read_bytes()
