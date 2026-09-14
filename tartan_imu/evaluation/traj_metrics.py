# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Trajectory metric aggregation.

Aggregates per-segment metrics into per-trajectory averages, computes
full-trajectory ATE/RTE/accuracy metrics, and rolls everything up into overall
statistics across all trajectories and segments.

Migrated from the monolithic top-level ``test.py``.
"""

import numpy as np
from tartan_imu.evaluation.metrics import compute_accruacy_metrics, compute_ate_rte


def compute_aggregated_metrics(segment_metrics: list) -> dict:
    """Average per-segment metrics into per-trajectory aggregates.

    Compute aggregated metrics from segment metrics.

    Args:
        segment_metrics: List of segment metrics

    Returns:
        Dictionary of aggregated metrics
    """
    if not segment_metrics:
        return {}

    metrics_keys = [
        "ate",
        "t_rte",
        "d_rte",
        "ATE",
        "AVE",
        "P_RMSE",
        "V_RMSE",
        "X_ATE",
        "Y_ATE",
        "Z_ATE",
        "X_AVE",
        "Y_AVE",
        "Z_AVE",
    ]

    aggregated = {}
    for key in metrics_keys:
        values = [seg[key] for seg in segment_metrics]
        aggregated[f"avg_{key}"] = np.mean(values)

    return aggregated


def compute_full_trajectory_metrics(traj_attr_dict: dict, cfg: dict) -> dict:
    """Compute ATE/RTE and accuracy metrics for the full trajectory.

    Compute metrics for the full trajectory.

    Args:
        traj_attr_dict: Full trajectory data
        cfg: Configuration dictionary

    Returns:
        Dictionary of full trajectory metrics
    """
    inference_rate = cfg["data"].get("sample_freq", cfg["data"]["imu_freq"])
    ate, t_rte, d_rte = compute_ate_rte(
        traj_attr_dict["pos_pred"],
        traj_attr_dict["pos_gt"],
        int(inference_rate * 1),
    )

    ATE, AVE, P_RMSE, V_RMSE, X_ATE, Y_ATE, Z_ATE, X_AVE, Y_AVE, Z_AVE = (
        compute_accruacy_metrics(
            traj_attr_dict,
            int(inference_rate * 1),
            cfg["data"]["use_local_coord"],
        )
    )

    result = {
        "ate": ate,
        "t_rte": t_rte,
        "d_rte": d_rte,
        "ATE": ATE,
        "AVE": AVE,
        "P_RMSE": P_RMSE,
        "V_RMSE": V_RMSE,
        "X_ATE": X_ATE,
        "Y_ATE": Y_ATE,
        "Z_ATE": Z_ATE,
        "X_AVE": X_AVE,
        "Y_AVE": Y_AVE,
        "Z_AVE": Z_AVE,
    }
    if "velocity_metrics" in traj_attr_dict:
        result.update(traj_attr_dict["velocity_metrics"])
        result["final_displacement_error"] = float(
            np.linalg.norm(traj_attr_dict["pos_pred"][-1] - traj_attr_dict["pos_gt"][-1])
        )
    return result


def compute_overall_statistics(
    all_trajectory_results: list, segment_metrics_all: list
) -> dict:
    """Roll up per-trajectory and per-segment metrics into overall stats.

    Compute overall statistics across all trajectories and segments.

    Args:
        all_trajectory_results: List of all trajectory results
        segment_metrics_all: List of all segment metrics

    Returns:
        Dictionary of overall statistics
    """
    # Extract aggregated metrics from all trajectories
    metrics_keys = [
        "avg_ate",
        "avg_t_rte",
        "avg_d_rte",
        "avg_ATE",
        "avg_AVE",
        "avg_P_RMSE",
        "avg_V_RMSE",
        "avg_X_ATE",
        "avg_Y_ATE",
        "avg_Z_ATE",
        "avg_X_AVE",
        "avg_Y_AVE",
        "avg_Z_AVE",
    ]

    overall_stats = {"total_segments": len(segment_metrics_all)}

    for key in metrics_keys:
        values = [traj[key] for traj in all_trajectory_results if key in traj]
        if values:
            overall_stats[key] = float(np.mean(values))

    # Add segment-wise statistics
    if segment_metrics_all:
        overall_stats["segment_statistics"] = {
            "total_segments": len(segment_metrics_all),
            "avg_segment_length": float(
                np.mean([seg["segment_length"] for seg in segment_metrics_all])
            ),
            "segment_ate_stats": {
                "mean": float(np.mean([seg["ate"] for seg in segment_metrics_all])),
                "std": float(np.std([seg["ate"] for seg in segment_metrics_all])),
                "min": float(np.min([seg["ate"] for seg in segment_metrics_all])),
                "max": float(np.max([seg["ate"] for seg in segment_metrics_all])),
            },
            "segment_rmse_stats": {
                "mean": float(np.mean([seg["P_RMSE"] for seg in segment_metrics_all])),
                "std": float(np.std([seg["P_RMSE"] for seg in segment_metrics_all])),
                "min": float(np.min([seg["P_RMSE"] for seg in segment_metrics_all])),
                "max": float(np.max([seg["P_RMSE"] for seg in segment_metrics_all])),
            },
        }

    return overall_stats
