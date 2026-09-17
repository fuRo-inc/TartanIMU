"""Full body-frame velocity covariance utilities.

The raw parameter order is fixed for exported models:
``[Lxx_raw, Lyy_raw, Lzz_raw, Lyx, Lzx, Lzy]``.
"""
import math

import torch
import torch.nn.functional as F


RAW_VELOCITY_COVARIANCE_ORDER = (
    "Lxx_raw", "Lyy_raw", "Lzz_raw", "Lyx", "Lzx", "Lzy",
)


def build_velocity_cholesky(raw_cov: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    """Build ``L`` from raw [..., 6] Softplus-Cholesky parameters.

    The order is ``[Lxx_raw, Lyy_raw, Lzz_raw, Lyx, Lzx, Lzy]`` and all
    resulting Cholesky diagonals are strictly positive.
    """
    if raw_cov.shape[-1] != 6:
        raise ValueError(f"velocity covariance raw dimension must be 6, got {raw_cov.shape[-1]}")
    if eps <= 0:
        raise ValueError("velocity covariance eps must be positive")
    diagonal = F.softplus(raw_cov[..., :3]) + eps
    L = raw_cov.new_zeros(*raw_cov.shape[:-1], 3, 3)
    L[..., 0, 0] = diagonal[..., 0]
    L[..., 1, 1] = diagonal[..., 1]
    L[..., 2, 2] = diagonal[..., 2]
    L[..., 1, 0] = raw_cov[..., 3]
    L[..., 2, 0] = raw_cov[..., 4]
    L[..., 2, 1] = raw_cov[..., 5]
    return L


def velocity_covariance_from_raw(raw_cov: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    """Return positive-definite body-frame covariance ``Sigma = L @ L.T``."""
    L = build_velocity_cholesky(raw_cov, eps)
    return L @ L.transpose(-1, -2)


def gaussian_velocity_nll(
    predicted_velocity: torch.Tensor,
    target_body_velocity: torch.Tensor,
    raw_cov: torch.Tensor,
    eps: float = 1.0e-4,
    detach_velocity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multivariate Gaussian NLL (without constant) in the body frame."""
    # Covariance linear algebra is evaluated in FP32.
    # torch.linalg.solve_triangular on CUDA does not support FP16.
    predicted_velocity_fp32 = predicted_velocity.float()
    target_body_velocity_fp32 = target_body_velocity.float()
    raw_cov_fp32 = raw_cov.float()
    error = target_body_velocity_fp32 - predicted_velocity_fp32
    if detach_velocity:
        error = error.detach()
    L = build_velocity_cholesky(raw_cov_fp32, eps)
    y = torch.linalg.solve_triangular(L, error.unsqueeze(-1), upper=False)
    mahalanobis = y.squeeze(-1).square().sum(dim=-1)
    diagonal = torch.diagonal(L, dim1=-2, dim2=-1)
    log_det = 2.0 * torch.log(diagonal).sum(dim=-1)
    nll = 0.5 * (mahalanobis + log_det)
    if not (torch.isfinite(nll).all() and torch.isfinite(L).all()):
        raise FloatingPointError("Non-finite full velocity covariance NLL or Cholesky factor")
    return nll, mahalanobis, L


def inverse_softplus(value: float) -> float:
    """Numerically stable inverse softplus for a positive scalar."""
    if value <= 0:
        raise ValueError("inverse_softplus input must be positive")
    return value + math.log(-math.expm1(-value))
