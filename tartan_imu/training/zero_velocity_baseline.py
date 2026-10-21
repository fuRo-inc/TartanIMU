"""Detached current body-velocity diagnostics; never part of the objective."""
import csv
import math
import os
import tempfile
from pathlib import Path

import torch

from tartan_imu.model.common.losses import speed_dependent_velocity_weight


@torch.no_grad()
def metric_rows(pred, target, speed_config):
    """One row per endpoint window, with batch-local loss weight normalization."""
    if pred.shape != target.shape or target.ndim != 2 or target.shape[-1] != 3:
        raise ValueError("Expected matching endpoint velocities of shape (N, 3)")
    weights = speed_dependent_velocity_weight(target.detach(), speed_config).double()
    residual = pred.detach().double() - target.detach().double()
    target = target.detach().double()
    return torch.stack((residual.square().sum(-1), target.square().sum(-1),
                        (20 * weights * residual.abs()).sum(-1),
                        (20 * weights * target.abs()).sum(-1)), dim=-1)


def summarize(sums, count):
    values = [float(x) / (3 * count) if count else None for x in sums]
    model, zero, model_l1, zero_l1 = values
    return dict(model_mse=model, zero_mse=zero,
                model_rmse_3d=math.sqrt(3 * model) if count else None,
                zero_rmse_3d=math.sqrt(3 * zero) if count else None,
                skill_score_vs_zero=1 - model / zero if zero else None,
                model_velocity_l1_monitor=model_l1, zero_velocity_l1=zero_l1,
                sample_count=int(count))


def save_metrics(out_dir, epoch, split, metrics):
    """Atomically upsert (epoch, split), including on resumed/replayed epochs.

    Undefined values are empty CSV cells (and Python None for JSON/W&B).
    Only the primary distributed rank may call this writer.
    """
    path = Path(out_dir) / 'zero_velocity_baseline_metrics.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if path.exists():
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                rows[(int(row['epoch']), row['split'])] = row
    row = dict(epoch=epoch, split=split, **metrics)
    rows[(epoch, split)] = row
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, newline='', delete=False) as stream:
            temporary = stream.name
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows[key] for key in sorted(rows))
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
