"""Regression tests for the original PACE dataset generation code."""
import importlib.util
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "prototyping" / "synthetic_data_degradation"


def load_source(name):
    """Load original standalone scripts without modifying repository source."""
    spec = importlib.util.spec_from_file_location(name, SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def degradation():
    return load_source("degradation")


@pytest.fixture
def dataset(monkeypatch):
    # build_dataset.py uses sibling imports from its original script directory.
    monkeypatch.syspath_prepend(str(SOURCE))
    return load_source("build_dataset")


def test_crop_standard_dimensions(dataset):
    image = np.zeros((712, 840, 3), dtype=np.uint8)
    assert dataset.apply_crop(image).shape == (512, 640, 3)


def test_crop_retains_correct_pixels(dataset):
    image = np.arange(712 * 840, dtype=np.int32).reshape(712, 840)
    cropped = dataset.apply_crop(image)
    np.testing.assert_array_equal(cropped, image[100:612, 100:740])
    assert cropped[0, 0] == image[100, 100]


def test_crop_rejects_invalid_image(dataset):
    with pytest.raises(ValueError):
        dataset.apply_crop(None)
    with pytest.raises(ValueError):
        dataset.apply_crop(np.zeros((200, 840), dtype=np.uint8))


def write_xml(path):
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "840"
    ET.SubElement(size, "height").text = "712"
    polygons = [
        [(110, 120), (150, 120), (150, 160), (110, 160)],  # fully inside
        [(10, 10), (30, 10), (30, 30), (10, 30)],       # outside
        [(90, 95), (120, 95), (120, 120), (90, 120)],   # crosses top/left
    ]
    for points in polygons:
        obj = ET.SubElement(root, "object")
        poly = ET.SubElement(obj, "polygon")
        for i, (x, y) in enumerate(points, 1):
            ET.SubElement(poly, f"x{i}").text = str(x)
            ET.SubElement(poly, f"y{i}").text = str(y)
    ET.ElementTree(root).write(path)


def test_xml_annotation_crop_and_clipping(dataset, tmp_path):
    src, dst = tmp_path / "source.xml", tmp_path / "cropped.xml"
    write_xml(src)
    dataset.adjust_xml_for_crop(src, dst, 840, 712)
    root = ET.parse(dst).getroot()
    assert root.findtext("size/width") == "640"
    assert root.findtext("size/height") == "512"
    objects = root.findall("object")
    assert len(objects) == 2
    p0 = objects[0].find("polygon")
    assert [(int(p0.findtext(f"x{i}")), int(p0.findtext(f"y{i}"))) for i in range(1, 5)] == [
        (10, 20), (50, 20), (50, 60), (10, 60)
    ]
    p1 = objects[1].find("polygon")
    assert [(int(p1.findtext(f"x{i}")), int(p1.findtext(f"y{i}"))) for i in range(1, 5)] == [
        (0, 0), (20, 0), (20, 20), (0, 20)
    ]


def test_xml_crop_rejects_invalid_dimensions(dataset, tmp_path):
    src, dst = tmp_path / "source.xml", tmp_path / "cropped.xml"
    write_xml(src)
    with pytest.raises(ValueError):
        dataset.adjust_xml_for_crop(src, dst, 200, 712)


def test_mask_fraction_and_threshold(dataset):
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert dataset.mask_fraction(mask) == 0
    assert not dataset.is_meaningful_degradation(mask)
    mask[0, 0] = 1
    assert dataset.mask_fraction(mask) == pytest.approx(0.01)
    assert dataset.is_meaningful_degradation(mask)
    assert dataset.mask_fraction(np.zeros((0, 0))) == 0


@pytest.mark.parametrize("shape", [(24, 32), (24, 32, 3)])
def test_mask_broadcasting(degradation, shape):
    image = np.zeros(shape, dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[:4, :6] = 1
    prepared = degradation.prepare_mask_for_image(mask, image)
    assert prepared.shape == ((24, 32, 1) if len(shape) == 3 else (24, 32))
    assert prepared.dtype == np.float32
    assert prepared.max() == 1


def test_blending_preserves_unmasked_pixels(degradation):
    image = np.full((16, 16, 3), 25, dtype=np.uint8)
    changed = np.full_like(image, 200)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:6, 3:8] = 1
    out = degradation.blend_with_mask(image, changed, mask)
    assert np.all(out[2:6, 3:8] == 200)
    assert np.all(out[0, :] == 25)
    assert np.all(out[8:, :] == 25)


def test_mask_shape_validation(degradation):
    with pytest.raises(ValueError):
        degradation.prepare_mask_for_image(np.ones((5, 5)), np.zeros((6, 6)))
    with pytest.raises(ValueError):
        degradation.blend_with_mask(np.zeros((6, 6)), np.zeros((7, 6)), np.zeros((6, 6)))


@pytest.mark.parametrize("kind", ["water_blur", "motion_blur", "sensor_noise", "saturation"])
def test_degradation_preserves_shape_dtype_and_unmasked_area(degradation, kind):
    image = np.full((64, 64, 3), 110, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:40, 10:40] = 1
    random.seed(42)
    np.random.seed(42)
    out, metadata = degradation.apply_degradation(image, kind, 0.8, mask)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[mask == 0], image[mask == 0])
    assert metadata["type"] == kind
    assert metadata["severity"] == 0.8
    np.testing.assert_array_equal(metadata["mask"], mask)


def test_sensor_noise_is_reproducible_with_both_seeds(degradation):
    image = np.full((64, 64), 110, dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.uint8)
    random.seed(123)
    np.random.seed(123)
    first, _ = degradation.apply_degradation(image, "sensor_noise", 0.8, mask)
    random.seed(123)
    np.random.seed(123)
    second, _ = degradation.apply_degradation(image, "sensor_noise", 0.8, mask)
    np.testing.assert_array_equal(first, second)


def test_random_mask_is_reproducible(degradation):
    random.seed(123)
    first = degradation.generate_mask(64, 64)
    random.seed(123)
    second = degradation.generate_mask(64, 64)
    np.testing.assert_array_equal(first, second)


def test_invalid_degradation_parameters(degradation):
    image = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError):
        degradation.apply_degradation(image, "not_a_type", 0.5)
    with pytest.raises(ValueError):
        degradation.apply_degradation(image, "sensor_noise", 1.5)
    with pytest.raises(ValueError):
        degradation.apply_degradation(image, "sensor_noise", 0.5, np.zeros((3, 3)))
