"""Preserve gate shape, binary routing, empirical threshold, and STE gradients."""
import pytest
import torch

MODULE = 'prototyping/reliability_gate/reliability_gate.py'

@pytest.fixture
def gate_class(load_legacy):
    return load_legacy(MODULE).ReliabilityGate

def test_gate_shape_and_binary_values(gate_class):
    model = gate_class().eval()
    with torch.no_grad():
        out = model(torch.ones(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))
    assert out.shape == (2, 1, 8, 8)
    assert torch.all((out == 0) | (out == 1))

def test_gate_empirical_threshold_is_026(gate_class):
    model = gate_class().eval()
    with torch.no_grad():
        for layer in model.network:
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.zero_()
                layer.bias.zero_()
        # sigmoid(-1) = 0.26894, above 0.26 but below 0.5.
        model.network[-1].bias.fill_(-1.0)
        out = model(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    assert torch.all(out == 1), 'Original PACE routing uses threshold 0.26, not 0.5'

def test_gate_straight_through_gradient(gate_class):
    model = gate_class()
    out = model(torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    out.sum().backward()
    assert model.network[-1].weight.grad is not None
    assert torch.isfinite(model.network[-1].weight.grad).all()

def test_invalid_grid_is_rejected(gate_class):
    model = gate_class()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 1, 7, 8), torch.zeros(1, 1, 8, 8))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 7))
