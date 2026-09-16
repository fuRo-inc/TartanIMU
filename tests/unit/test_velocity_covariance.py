import torch

from tartan_imu.model.common.velocity_covariance import (
    build_velocity_cholesky,
    gaussian_velocity_nll,
    velocity_covariance_from_raw,
)
from tartan_imu.model.lstm.heads import OutputHead


def test_softplus_cholesky_shapes_symmetry_pd_and_gradients():
    raw = torch.randn(2, 5, 6, requires_grad=True)
    L = build_velocity_cholesky(raw, 1e-4)
    sigma = velocity_covariance_from_raw(raw, 1e-4)
    assert L.shape == (2, 5, 3, 3)
    assert sigma.shape == (2, 5, 3, 3)
    assert torch.allclose(sigma, sigma.transpose(-1, -2))
    assert torch.all(torch.diagonal(L, dim1=-2, dim2=-1) > 0)
    torch.linalg.cholesky(sigma)
    nll, _, _ = gaussian_velocity_nll(torch.randn(2, 5, 3), torch.randn(2, 5, 3), raw)
    nll.mean().backward()
    assert torch.isfinite(nll).all() and torch.isfinite(raw.grad).all()


def test_covariance_head_initialization_and_legacy_forward():
    base = {"model_param": {"lstm_size": 8, "lstm_dropout": 0.0, "output_dim": 3, "drop_ratio": 0.0, "split_z": False}, "model": {"pred_velocity": True}}
    legacy = OutputHead(base, "dog")
    legacy_output = legacy(torch.randn(4, 8), 2, 2)
    assert legacy_output.shape == (2, 2, 3)
    cfg = {**base, "model": {"pred_velocity": True, "velocity_covariance": {"enabled": True, "eps": 1e-4, "initial_std": 0.1}}}
    head = OutputHead(cfg, "dog")
    _, raw = head(torch.randn(4, 8), 2, 2, predict_cov=True)
    sigma = velocity_covariance_from_raw(raw, 1e-4)
    assert raw.shape == (2, 2, 6)
    assert torch.allclose(torch.diagonal(sigma, dim1=-2, dim2=-1), torch.full((2, 2, 3), 0.01), atol=1e-5)
    assert torch.allclose(sigma[..., [0, 0, 1], [1, 2, 2]], torch.zeros(2, 2, 3), atol=1e-7)
