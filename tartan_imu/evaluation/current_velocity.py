"""Shared, causal current body-velocity inference helpers.

The target is deliberately read as ``target[:, -1, :3]``: this is the
endpoint target constructed by ``dataset_AirLab`` for ``velocity_target=current``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from tartan_imu.dataloader import dataset_AirLab as dataset_utils

MOTION_TYPE_ID = {"car": 1, "dog": 2, "drone": 3, "human": 4}


def infer_current_velocity(model, config, npz_path, motion_type, device, batch_size=None):
    """Return predicted and GT body-frame endpoint velocities for one NPZ.

    This intentionally creates a dataset per trajectory while reusing the
    already-loaded model.  ``BasicSequenceData`` owns all endpoint validity
    filtering, so no evaluator-specific filtering is applied here.
    """
    if motion_type not in MOTION_TYPE_ID:
        raise ValueError("unsupported motion type: %s" % motion_type)
    basic = dataset_utils.BasicSequenceData(config, [str(npz_path)], mode="test")
    dataset = dataset_utils.ResNetLSTMSeqToSeqDataset(
        config, basic, basic.get_merged_index_map(), mode="test"
    )
    loader = DataLoader(dataset, batch_size=batch_size or config.get("test", {}).get("batch_size", 256))
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            imu = batch[0].to(device)
            targets.append(batch[1][:, -1, :3].numpy())
            label = torch.full((imu.shape[0],), MOTION_TYPE_ID[motion_type], dtype=torch.long, device=device)
            prediction = model(imu, motion_type=label, predict_cov=False, compute_all_heads=False)[motion_type]
            predictions.append(prediction[:, -1, :].cpu().numpy())
    if not predictions:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.concatenate(predictions), np.concatenate(targets)
