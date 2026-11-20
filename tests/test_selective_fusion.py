"""Regression tests for the original selective local fusion implementation.

These tests use lightweight stand-ins for pretrained CrossFuse components.
They do not modify or load actual checkpoints.
"""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "prototyping/selective_fusion/local_crossfuse.py"
)


@pytest.fixture(scope="module")
def local_class():
    spec = importlib.util.spec_from_file_location("pace_legacy_local_crossfuse", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LocalCrossFuse


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x, F.avg_pool2d(x, kernel_size=4)


class FakeBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_calls = 0
        self.decoder_calls = 0
        self.attention_batch_sizes = []
        self.decoder_batch_sizes = []

    def cross_atten_block(self, ir_de, rgb_de, shifted):
        assert shifted is False
        self.attention_calls += 1
        self.attention_batch_sizes.append(ir_de.shape[0])
        return (ir_de + rgb_de) / 2, None

    def decoder_fusion(self, ir_sh, rgb_sh, ir_de, rgb_de, fused):
        self.decoder_calls += 1
        self.decoder_batch_sizes.append(ir_sh.shape[0])
        n, _, h, w = ir_sh.shape
        ramp = torch.linspace(0, 1, h * w, device=ir_sh.device).reshape(1, 1, h, w)
        return ramp.expand(n, 1, h, w).clone()


@pytest.fixture
def setup_model(local_class):
    rgb_encoder = FakeEncoder()
    ir_encoder = FakeEncoder()
    baseline = FakeBaseline()
    model = local_class(
        rgb_encoder, ir_encoder, baseline,
        grid_size=8, context_features=4, batch_size=3,
    )
    return model, rgb_encoder, ir_encoder, baseline


def inputs(batch=1):
    rgb = torch.full((batch, 1, 64, 64), 0.2)
    ir = torch.full((batch, 1, 64, 64), 0.8)
    rgb_q = torch.zeros(batch, 1, 8, 8)
    ir_q = torch.ones(batch, 1, 8, 8)
    gate = torch.zeros(batch, 1, 8, 8)
    return rgb, ir, rgb_q, ir_q, gate


def test_zero_gate_skips_encoders_and_decoder(setup_model):
    model, re, ie, baseline = setup_model
    result = model(*inputs())
    torch.testing.assert_close(result["out"], torch.full((1, 1, 64, 64), 0.8))
    assert result["fused_count"] == 0
    assert result["fusion_rate"] == 0.0
    assert re.calls == ie.calls == 0
    assert baseline.attention_calls == baseline.decoder_calls == 0


def test_preferred_modality_is_chosen_patchwise(setup_model):
    model, *_ = setup_model
    rgb, ir, rgb_q, ir_q, gate = inputs()
    rgb_q[:, :, 2, 3] = 1.0
    ir_q[:, :, 2, 3] = 0.0
    result = model(rgb, ir, rgb_q, ir_q, gate)
    assert torch.all(result["out"][:, :, 16:24, 24:32] == 0.2)
    assert torch.all(result["out"][:, :, :8, :8] == 0.8)
    assert result["prefer_rgb"][0, 0, 2, 3].item() is True


def test_quality_ties_prefer_rgb(setup_model):
    model, *_ = setup_model
    rgb, ir, rgb_q, ir_q, gate = inputs()
    ir_q.zero_()
    result = model(rgb, ir, rgb_q, ir_q, gate)
    torch.testing.assert_close(result["out"], rgb)
    assert bool(result["prefer_rgb"].all())


def test_one_selected_patch_only_changes_that_patch(setup_model):
    model, re, ie, baseline = setup_model
    rgb, ir, rgb_q, ir_q, gate = inputs()
    gate[:, :, 0, 0] = 1
    result = model(rgb, ir, rgb_q, ir_q, gate)
    assert result["fused_count"] == 1
    assert result["fusion_rate"] == pytest.approx(1 / 64)
    assert re.calls == ie.calls == 1
    assert baseline.attention_calls == baseline.decoder_calls == 1
    torch.testing.assert_close(result["out"][:, :, 8:, :], ir[:, :, 8:, :])
    torch.testing.assert_close(result["out"][:, :, :8, 8:], ir[:, :, :8, 8:])
    assert not torch.allclose(result["out"][:, :, :8, :8], ir[:, :, :8, :8])


def test_selected_patches_are_batched(setup_model):
    model, _, _, baseline = setup_model
    rgb, ir, rgb_q, ir_q, gate = inputs()
    gate[:, :, 0, :7] = 1
    result = model(rgb, ir, rgb_q, ir_q, gate)
    assert result["fused_count"] == 7
    assert result["fusion_rate"] == pytest.approx(7 / 64)
    assert baseline.attention_batch_sizes == [3, 3, 1]
    assert baseline.decoder_batch_sizes == [3, 3, 1]


def test_precomputed_features_bypass_encoders(setup_model):
    model, re, ie, baseline = setup_model
    rgb, ir, rgb_q, ir_q, gate = inputs()
    gate[:, :, 7, 7] = 1
    features = (rgb, F.avg_pool2d(rgb, 4), ir, F.avg_pool2d(ir, 4))
    result = model(rgb, ir, rgb_q, ir_q, gate, encoded_features=features)
    assert result["fused_count"] == 1
    assert re.calls == ie.calls == 0
    assert baseline.decoder_calls == 1


def test_crop_padded_replication(local_class):
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    crop = local_class.crop_padded(x, -1, -1, 4)
    assert crop.shape == (1, 1, 4, 4)
    assert crop[0, 0, 0, 0].item() == 0
    assert crop[0, 0, 0, 1].item() == 0
    assert crop[0, 0, 1, 0].item() == 0
    assert crop[0, 0, 2, 2].item() == 5


def test_normalization_is_per_sample(local_class):
    x = torch.stack([
        torch.tensor([[[2., 4.], [6., 8.]]]),
        torch.tensor([[[10., 20.], [30., 40.]]]),
    ])
    result = local_class.normalize_individual(x)
    assert result.shape == x.shape
    torch.testing.assert_close(result.amin(dim=(1, 2, 3)), torch.zeros(2))
    torch.testing.assert_close(result.amax(dim=(1, 2, 3)), torch.ones(2))


@pytest.mark.parametrize("invalid_gate", [0.5, float("nan"), float("inf")])
def test_invalid_gate_values_rejected(setup_model, invalid_gate):
    model, *_ = setup_model
    rgb, ir, rgb_q, ir_q, gate = inputs()
    gate[:, :, 0, 0] = invalid_gate
    with pytest.raises(ValueError):
        model(rgb, ir, rgb_q, ir_q, gate)


def test_invalid_quality_shape_rejected(setup_model):
    model, *_ = setup_model
    rgb, ir, rgb_q, ir_q, gate = inputs()
    with pytest.raises(ValueError):
        model(rgb, ir, rgb_q[:, :, :7, :], ir_q, gate)


def test_invalid_image_dimensions_rejected(setup_model):
    model, *_ = setup_model
    rgb, ir, rgb_q, ir_q, gate = inputs()
    with pytest.raises(ValueError):
        model(rgb[:, :, :-1, :], ir[:, :, :-1, :], rgb_q, ir_q, gate)


def test_invalid_precomputed_features_rejected(setup_model):
    model, *_ = setup_model
    rgb, ir, rgb_q, ir_q, gate = inputs()
    gate[:, :, 0, 0] = 1
    with pytest.raises(ValueError):
        model(rgb, ir, rgb_q, ir_q, gate, encoded_features=(rgb, ir))
