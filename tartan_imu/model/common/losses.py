# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Loss functions for velocity prediction (L1 / Huber / Gaussian NLL variants).

The training loss path is ``single_head_velocity_loss`` (L1, weight=20). Helpers
here cover the multi-head masked-loss reduction and the sequence loss used
during evaluation.
"""
from typing import Callable, Mapping, Optional

import torch

EPSILON = 1e-7


def speed_dependent_velocity_weight(
    targ: torch.Tensor, speed_weighting: Optional[Mapping] = None
) -> torch.Tensor:
    """Return one bounded, GT-only weight per velocity vector.

    The returned shape is ``targ.shape[:-1] + (1,)`` so a single weight is
    broadcast across vx/vy/vz.  This intentionally does not filter samples:
    low-speed fallen/stationary data can also be upweighted, so any future
    upright/fallen filtering must remain a separate dataset-level policy.
    """
    cfg = speed_weighting or {}
    if not cfg.get("enabled", False):
        return torch.ones_like(targ[..., :1])

    alpha = float(cfg.get("alpha", 2.0))
    v_ref = float(cfg.get("v_ref", 0.25))
    power = float(cfg.get("power", 2.0))
    if alpha < 0 or v_ref <= 0 or power <= 0:
        raise ValueError("speed_weighting requires alpha >= 0, v_ref > 0, power > 0")

    # Targets do not require gradients in normal training; detach makes the
    # GT-only contract explicit and also protects callers that pass one that does.
    speed = torch.linalg.vector_norm(targ.detach(), dim=-1, keepdim=True)
    raw = 1.0 + alpha / (1.0 + (speed / v_ref).pow(power))
    if cfg.get("normalize_mean", True):
        raw = raw / raw.mean().clamp_min(EPSILON)
    return raw


def loss_distribution_diag(
    pred: torch.Tensor, pred_cov: torch.Tensor, targ: torch.Tensor
) -> torch.Tensor:
    """Bounded covariance-based diagonal Gaussian NLL (up to a constant)."""
    pred_cov_clamped = torch.clamp(pred_cov, min=-10.0, max=10.0)
    exp_term = torch.exp(2 * pred_cov_clamped)
    exp_term = torch.clamp(exp_term, min=1e-6, max=1e6)

    squared_error = (pred - targ).pow(2)
    loss = squared_error / (2 * exp_term) + pred_cov_clamped
    loss = torch.clamp(loss, min=1e-8)
    return loss


def loss_cum_distribution_diag(
    pred: torch.Tensor, pred_cov: torch.Tensor, targ: torch.Tensor
) -> torch.Tensor:
    """Diagonal Gaussian NLL variant. Kept for compatibility (currently unused)."""
    pred_cov_clamped = torch.clamp(pred_cov, min=1e-6, max=1e6)
    loss = ((pred - targ).pow(2)) / (2 * pred_cov_clamped) + 0.5 * torch.log(
        pred_cov_clamped
    )
    loss = torch.clamp(loss, min=1e-8)
    return loss


def get_sequence_smooth_loss(
    pred: torch.Tensor,
    pred_cov: torch.Tensor,
    targ: torch.Tensor,
    epoch: int,
    start_cov_epoch: int,
    velocity_loss_config: Optional[Mapping] = None,
) -> torch.Tensor:
    """
    Sequence loss for evaluation.
    - Before start_cov_epoch: same velocity loss as training
      (single_head_velocity_loss: L1 * weight) so test_loss is directly
      comparable to train_loss. Previously this used plain MSE, making the two
      curves incomparable (R1).
    - After start_cov_epoch: bounded covariance loss (diagonal Gaussian NLL style).
    """
    if epoch <= start_cov_epoch:
        loss = single_head_velocity_loss(
            pred, pred_cov, targ, velocity_loss_config=velocity_loss_config
        )["loss"]
    else:
        loss = loss_distribution_diag(pred, pred_cov, targ)
    return torch.mean(loss)


def multi_head_smooth_loss(
    multi_head_pred: dict,
    multi_head_cov: dict,
    targ: torch.Tensor,
    epoch: int,
    multi_head_mask: dict,
    start_cov_epoch: int,
    use_local_coord: bool,
    velocity_loss_config: Optional[Mapping] = None,
) -> torch.Tensor:
    """Sum the per-head masked smooth loss across all motion-type heads."""
    multi_head_loss = {}
    total_loss = 0
    for key in multi_head_pred:
        multi_head_loss[key] = single_head_mask_loss(
            multi_head_pred[key],
            multi_head_cov[key],
            targ,
            epoch,
            multi_head_mask[key],
            start_cov_epoch,
            use_local_coord,
            velocity_loss_config,
        )
        total_loss = multi_head_loss[key] + total_loss
    return total_loss


def efficient_multi_head_smooth_loss(
    multi_head_pred: dict,
    multi_head_cov: dict,
    targ: torch.Tensor,
    epoch: int,
    multi_head_mask: dict,
    start_cov_epochs: int,
    use_local_coord: bool,
    velocity_loss_config: Optional[Mapping] = None,
) -> torch.Tensor:
    """Efficient multi-head smooth loss without hot-path debug I/O."""
    total_loss = 0
    num_active_heads = 0

    for key, pred in multi_head_pred.items():
        if key in multi_head_mask:
            mask = multi_head_mask[key]
            if mask.sum() > 0:
                head_loss = single_head_mask_loss(
                    pred,
                    multi_head_cov.get(key, torch.zeros_like(pred)),
                    targ,
                    epoch,
                    mask,
                    start_cov_epochs,
                    use_local_coord,
                    velocity_loss_config,
                )

                total_loss += head_loss
                num_active_heads += 1

    if num_active_heads > 0:
        total_loss = total_loss / num_active_heads

    return total_loss


def smooth_transition_weight(
    epoch: int, start_cov_epoch: int, transition_epochs: int = 10
) -> float:
    """
    Smooth transition weight for covariance training.
    Returns a value between 0 and 1 that gradually increases over transition_epochs.
    """
    if epoch <= start_cov_epoch:
        return 0.0
    elif epoch >= start_cov_epoch + transition_epochs:
        return 1.0
    else:
        # Linear interpolation
        progress = (epoch - start_cov_epoch) / transition_epochs
        return progress


def _masked_mean(loss: torch.Tensor, mask_view: torch.Tensor) -> torch.Tensor:
    """Mean of ``loss`` over the elements selected by ``mask_view``.

    ``loss`` is already zeroed outside the mask; this divides the sum by the
    number of active elements (mask count times the per-sample feature size)
    rather than the full tensor size, so a head's loss does not depend on how
    many of the batch's samples belong to it. An all-True mask reduces exactly
    to ``torch.mean(loss)`` (single-head training baseline is unchanged).
    """
    per_sample = loss[0].numel()  # feature elements per masked sample (e.g. T*3)
    active = mask_view.sum() * per_sample
    if active == 0:
        return loss.sum() * 0.0  # keep graph connected, zero contribution
    return loss.sum() / active


def single_head_mask_loss(
    pred: torch.Tensor,
    pred_cov: torch.Tensor,
    targ: torch.Tensor,
    epoch: int,
    mask: torch.Tensor,
    start_cov_epoch: int,
    use_local_coord: bool = False,
    velocity_loss_config: Optional[Mapping] = None,
) -> torch.Tensor:
    """
    Simplified masked loss per head:
    - MSE before covariance training
    - Covariance-based loss after, with smooth transition
    Absolute/cumulative terms are removed for simplicity.
    """
    # Smooth transition for covariance
    cov_weight = smooth_transition_weight(epoch, start_cov_epoch, transition_epochs=5)

    if use_local_coord:
        # Local-coordinate velocity path (used by Go2 current-velocity training).
        loss_covariance = single_head_velocity_loss(
            pred, pred_cov, targ, velocity_loss_config=velocity_loss_config
        )
        loss = loss_covariance["loss"]
        # Mask to this head's samples, then normalize by the head's OWN active
        # element count (not the whole batch) so the per-head loss is
        # independent of how many of the batch's samples belong to this head.
        # An all-True mask reduces to torch.mean(loss) (single-head baseline).
        mask_view = mask.bool().view(-1, 1, 1)
        loss = loss * mask_view
        return _masked_mean(loss, mask_view)

    # Per-step MSE
    mse_loss = (pred - targ).pow(2)

    # Add scale-aware loss for velocity prediction (when use_local_coord=True)
    if use_local_coord:
        # Compute scale error (ratio of predicted to target magnitudes)
        pred_mag = torch.norm(pred, dim=-1, keepdim=True)
        targ_mag = torch.norm(targ, dim=-1, keepdim=True)

        # Avoid division by zero
        targ_mag = torch.clamp(targ_mag, min=1e-6)
        scale_ratio = pred_mag / targ_mag

        # Penalize scale errors (log-scale loss)
        scale_loss = torch.log(scale_ratio + 1e-6).pow(2)

        # Combine MSE and scale loss
        mse_loss = mse_loss + 0.1 * scale_loss

    # Apply mask to MSE
    mask_view = mask.bool().view(-1, 1, 1)
    mse_loss = mse_loss * mask_view

    if cov_weight > 0:
        cov_loss = loss_distribution_diag(pred, pred_cov, targ)
        cov_loss = cov_loss * mask_view
        # Weighted combination
        loss = (1 - cov_weight) * mse_loss + cov_weight * cov_loss
    else:
        loss = mse_loss

    return _masked_mean(loss, mask_view)


def L2(dist: torch.Tensor) -> torch.Tensor:
    """Element-wise squared error."""
    error = dist.pow(2)
    return error


def L1(dist: torch.Tensor) -> torch.Tensor:
    """Element-wise absolute error."""
    error = (dist).abs()
    return error


def Huber(dist: torch.Tensor, delta: float = 0.005) -> torch.Tensor:
    """Element-wise Huber loss against zero (no reduction)."""
    error = torch.nn.functional.huber_loss(
        dist, torch.zeros_like(dist, device=dist.device), delta=delta, reduction="none"
    )
    return error


def motion_loss_(
    fc: Callable, pred: torch.Tensor, targ: torch.Tensor
) -> tuple:
    """Apply loss function ``fc`` to the prediction-target residual.

    Returns the per-element loss and the raw residual ``pred - targ``.
    """
    dist = pred - targ
    loss = fc(dist)
    return loss, dist


def diag_ln_cov_loss(
    dist: torch.Tensor, pred_cov: torch.Tensor, use_epsilon: bool = False
) -> torch.Tensor:
    """Diagonal covariance log-likelihood term: error/cov + log(cov)."""
    error = (dist).pow(2)
    if use_epsilon:
        nll = (error / pred_cov) + torch.log(pred_cov + EPSILON)
    else:
        nll = (error / pred_cov) + torch.log(pred_cov)
    return nll


def single_head_velocity_loss(
    pred: torch.Tensor,
    pred_cov: torch.Tensor,
    targ: torch.Tensor,
    velocity_loss_config: Optional[Mapping] = None,
) -> dict:
    """Primary velocity loss: weighted L1.

    Returns a dict with ``loss`` (weight * per-element L1) and ``cov_loss``
    (zero unless covariance propagation is enabled).
    """
    # TODO: Need to put the following parameters in the yaml file
    confs = {
        "loss": "L1",
        "propcov": False,
        "cov_weight": 1e-4,
        "covaug": False,
        "weight": 20,  # 20
    }
    # Decouple velocity prediction for xyz three axes
    ## The state loss for evaluation
    loss, cov_loss = torch.zeros_like(pred, device=pred.device), torch.zeros_like(
        pred, device=pred.device
    )
    loss_fc = loss_fc_list[confs["loss"]]
    vel_loss, vel_dist = motion_loss_(loss_fc, pred, targ)

    # Apply the covariance loss
    if confs["propcov"]:
        # velocity covariance.
        cov = pred_cov
        cov_loss = cov.mean()

        if "covaug" in confs and confs["covaug"] is True:
            vel_loss += confs["cov_weight"] * diag_ln_cov_loss(vel_dist, cov)
        else:
            vel_loss += confs["cov_weight"] * diag_ln_cov_loss(vel_dist.detach(), cov)
    speed_weight = speed_dependent_velocity_weight(targ, velocity_loss_config)
    loss += confs["weight"] * vel_loss * speed_weight
    return {"loss": loss, "cov_loss": cov_loss, "speed_weight": speed_weight}


loss_fc_list = {
    "L2": L2,
    "L1": L1,
    "diag_cov_ln": diag_ln_cov_loss,
    "Huber_loss005": lambda dist: Huber(dist, delta=0.005),
    "Huber_loss05": lambda dist: Huber(dist, delta=0.05),
}


def smooth_velocity_predictions(
    velocities: torch.Tensor, window_size: int = 3
) -> torch.Tensor:
    """
    Simple post-processing function to smooth velocity predictions and reduce jittering.

    Args:
        velocities: [B, T, 3] or [T, 3] velocity predictions
        window_size: Size of smoothing window (odd number recommended)

    Returns:
        Smoothed velocities with same shape as input
    """
    import torch.nn.functional as F

    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Handle different input shapes
    if velocities.dim() == 2:
        velocities = velocities.unsqueeze(0)  # [T, 3] -> [1, T, 3]
        single_batch = True
    else:
        single_batch = False

    # Simple moving average smoothing
    pad_size = window_size // 2
    padded = F.pad(velocities, (0, 0, pad_size, pad_size), mode="replicate")
    smoothed = F.avg_pool1d(
        padded.transpose(1, 2),  # [B, C, T+2*pad]
        kernel_size=window_size,
        stride=1,
        padding=0,
    ).transpose(
        1, 2
    )  # [B, T, C]

    if single_batch:
        smoothed = smoothed.squeeze(0)  # [1, T, 3] -> [T, 3]

    return smoothed
