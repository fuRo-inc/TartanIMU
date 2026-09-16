"""Unit coverage for bounded GT-speed weighting of current velocity loss."""
import torch

from tartan_imu.model.common.losses import (
    single_head_velocity_loss,
    speed_dependent_velocity_weight,
)


CFG = {"enabled": True, "alpha": 2.0, "v_ref": 0.25, "power": 2.0,
       "normalize_mean": False}


def test_speed_weights_are_bounded_ordered_and_axis_broadcastable():
    targ = torch.tensor([[[0., 0., 0.], [0.1, 0., 0.], [0.5, 0., 0.], [1., 0., 0.]]])
    weights = speed_dependent_velocity_weight(targ, CFG)
    assert weights.shape == (1, 4, 1)
    assert torch.isfinite(weights).all()
    assert weights[0, 0] > weights[0, 1] > weights[0, 2] > weights[0, 3]
    # At sufficiently high speed this bounded weighting tends to one.
    high = speed_dependent_velocity_weight(torch.tensor([[[100., 0., 0.]]]), CFG)
    assert torch.allclose(high, torch.ones_like(high), atol=1e-4)


def test_normalization_disabled_is_identity_and_enabled_means_one():
    targ = torch.tensor([[[0., 0., 0.], [1., 0., 0.]], [[0.1, 0., 0.], [0.5, 0., 0.]]])
    normalized = speed_dependent_velocity_weight(targ, {**CFG, "normalize_mean": True})
    assert torch.allclose(normalized.mean(), torch.tensor(1.0), atol=1e-6)
    disabled = speed_dependent_velocity_weight(targ, {"enabled": False})
    assert torch.equal(disabled, torch.ones_like(disabled))


def test_weighted_loss_backward_zero_and_disabled_baseline():
    torch.manual_seed(4)
    targ = torch.randn(2, 3, 3)
    pred = (targ + 0.1).detach().requires_grad_(True)
    cov = torch.zeros_like(pred)
    weighted = single_head_velocity_loss(pred, cov, targ, CFG)["loss"]
    assert torch.isfinite(weighted).all()
    weighted.mean().backward()
    assert torch.isfinite(pred.grad).all()

    baseline = single_head_velocity_loss(pred.detach(), cov, targ)["loss"]
    disabled = single_head_velocity_loss(pred.detach(), cov, targ, {"enabled": False})["loss"]
    assert torch.equal(baseline, disabled)
    exact = single_head_velocity_loss(targ, torch.zeros_like(targ), targ, CFG)["loss"]
    assert torch.equal(exact, torch.zeros_like(exact))


def test_one_vector_weight_is_applied_to_all_velocity_axes():
    targ = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    pred = targ + 1.0  # equal L1 residual on every vx/vy/vz component
    loss = single_head_velocity_loss(pred, torch.zeros_like(pred), targ, CFG)["loss"]
    # Every axis at a given timestep must have the exact same scalar loss.
    assert torch.equal(loss[..., 0], loss[..., 1])
    assert torch.equal(loss[..., 1], loss[..., 2])
