# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Shared dataloader helpers extracted from the per-dataset modules.

These functions were previously duplicated verbatim across the
``dataset_*.py`` loaders. They are kept here as the single canonical
implementation and re-exported from each loader module.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def calculate_velocity_from_poses(
    ts: np.ndarray, pos: np.ndarray, quat: np.ndarray
) -> tuple:
    """
    Calculate velocity from pose data with safety checks.

    Args:
        ts (np.ndarray): Timestamps
        pos (np.ndarray): Positions
        quat (np.ndarray): Quaternions (xyzw format)

    Returns:
        tuple: (velocity_global, velocity_body)
    """
    if len(ts) < 2 or len(pos) < 2 or len(quat) < 2:
        raise ValueError("Input arrays must have at least 2 elements")

    if len(ts) != len(pos) or len(ts) != len(quat):
        raise ValueError("All input arrays must have the same length")

    # Calculate relative positions
    pos_gt = pos - pos[0]
    quat_gt = Rotation.from_quat(quat)

    # Calculate time differences
    dt = ts[1:] - ts[:-1]
    dt = np.where(dt == 0, 1e-6, dt)
    dt = np.where(dt < 1e-9, 1e-6, dt)

    if dt.shape == (dt.shape[0],):
        dt = dt[:, np.newaxis]

    # Calculate velocities
    velocity_global = (pos_gt[1:] - pos_gt[:-1]) / dt
    velocity_body = quat_gt[:-1].inv().apply(velocity_global)

    return velocity_global, velocity_body


def current_velocity_from_positions(ts: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Per-timestamp world velocity using central differences where possible.

    Unlike the legacy helper this returns one velocity for every input timestamp;
    it is deliberately not a long-window average.
    """
    ts = np.asarray(ts, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    if ts.ndim != 1 or pos.ndim != 2 or pos.shape[0] != len(ts) or len(ts) < 2:
        raise ValueError("ts and pos must be aligned arrays with at least two samples")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    velocity = np.empty_like(pos, dtype=np.float64)
    velocity[0] = (pos[1] - pos[0]) / (ts[1] - ts[0])
    velocity[-1] = (pos[-1] - pos[-2]) / (ts[-1] - ts[-2])
    velocity[1:-1] = (pos[2:] - pos[:-2]) / (ts[2:, None] - ts[:-2, None])
    return velocity


def world_to_body_velocity(velocity_world: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert current world velocity to current body/base velocity (xyzw quat)."""
    return Rotation.from_quat(quat_xyzw).inv().apply(velocity_world)


def clip_velocity_outliers(
    velocity: np.ndarray, max_speed: float = 15.0
) -> np.ndarray:
    """Clip physically-impossible velocity spikes to a speed ceiling.

    TartanIMU car/dog GT positions have occasional jumps that differentiate
    into speeds up to ~210 m/s (vs a p50 of ~0.5 m/s). A single such frame
    integrates into a multi-metre position jump and destroys long-horizon ATE.
    Frames whose speed exceeds ``max_speed`` are rescaled to ``max_speed`` while
    preserving their direction; all other frames are returned unchanged.

    Args:
        velocity (np.ndarray): (N, 3) velocity vectors.
        max_speed (float): Speed ceiling in m/s (default 15.0, far above any
            legitimate car/dog/human motion in this dataset).

    Returns:
        np.ndarray: (N, 3) velocity with outlier frames clipped in magnitude.
    """
    speed = np.linalg.norm(velocity, axis=1)
    over = speed > max_speed
    if not np.any(over):
        return velocity
    out = velocity.copy()
    scale = max_speed / speed[over]
    out[over] = velocity[over] * scale[:, np.newaxis]
    return out


def partition_data(
    index_map: list,
    valid_samples: np.ndarray,
    valid_all_samples: float,
    training_rate: float = 0.9,
    valuation_rate: float = 0.1,
    data_rate: float = 1.0,
    shuffle: bool = True,
    **kwargs,
) -> tuple:
    """Partition data into train/val sets."""
    if shuffle:
        np.random.shuffle(index_map)

    all_size = 0
    sum_valid_samples = valid_all_samples * data_rate

    accum_samples = 0.0
    for i in range(len(index_map)):
        accum_samples += valid_samples[index_map[i][0][0]]
        all_size = i
        if accum_samples > sum_valid_samples:
            break

    valuation_samples = sum_valid_samples * valuation_rate
    train_index_map, valuation_index_map = [], []
    accum_valuation_samples = 0

    for i in range(all_size):
        if accum_valuation_samples < valuation_samples:
            valuation_index_map += index_map[i]
            accum_valuation_samples += valid_samples[index_map[i][0][0]]
        else:
            train_index_map += index_map[i]

    return train_index_map, valuation_index_map
