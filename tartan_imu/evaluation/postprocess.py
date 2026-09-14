# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Post-processing for trajectory evaluation.

Integrates predicted (body- or global-frame) velocities into positions,
builds plotting dictionaries, and renders trajectory/velocity figures.
"""

import logging
import os
from os import path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation


def integrate_current_body_velocity(ts, velocity_body, quat_xyzw, initial_pos=None):
    """Causally recover world positions from current base velocities.

    ``quat_xyzw[k]`` and ``velocity_body[k]`` are deliberately paired at the
    same timestamp.  Variable step sizes and trapezoidal integration are used.
    """
    ts = np.asarray(ts, dtype=np.float64)
    velocity_body = np.asarray(velocity_body, dtype=np.float64)
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    if len(ts) != len(velocity_body) or len(ts) != len(quat_xyzw) or len(ts) < 1:
        raise ValueError("ts, velocity_body, and quat_xyzw must be aligned non-empty arrays")
    if len(ts) > 1 and np.any(np.diff(ts) <= 0):
        raise ValueError("integration timestamps must be strictly increasing")
    velocity_world = Rotation.from_quat(quat_xyzw).apply(velocity_body)
    positions = np.empty_like(velocity_world)
    positions[0] = np.zeros(3) if initial_pos is None else initial_pos
    for k, dt in enumerate(np.diff(ts)):
        positions[k + 1] = positions[k] + 0.5 * (velocity_world[k] + velocity_world[k + 1]) * dt
    return positions, velocity_world


def current_velocity_metrics(predicted_body, target_body):
    """Per-axis/current 3-D velocity errors used by Go2 evaluation."""
    error = np.asarray(predicted_body) - np.asarray(target_body)
    return {
        "vx_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "vy_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "vz_rmse": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "velocity_3d_rmse": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
        "velocity_bias": np.mean(error, axis=0),
    }


def recover_global_pose_from_local_velocity(
    ts, velocity_local, initial_pos, initial_quat
):

    dt = ts[1:] - ts[:-1]
    dt = np.concatenate([dt, [dt[-1]]])
    n = len(dt)
    # Initialize global pose arrays
    pos_global = np.zeros((n, 3))
    # Set initial pose
    pos_global[0] = initial_pos

    current_rotation = Rotation.from_quat(initial_quat)

    print("n", n)
    for i in range(1, n):
        # Transform local velocity to global frame

        global_velocity = current_rotation[i - 1].apply(velocity_local[i - 1])

        # Integrate position
        pos_global[i] = pos_global[i - 1] + global_velocity * dt[i - 1]

    return pos_global


def align_trajectory_to_ground_truth(pred_positions, gt_positions):
    """
    Align predicted trajectory to ground truth trajectory.
    Ensures the predicted trajectory starts from the same initial position as ground truth.

    Args:
        pred_positions: Predicted positions [N, 3]
        gt_positions: Ground truth positions [N, 3]

    Returns:
        Aligned predicted positions [N, 3]
    """
    if len(pred_positions) != len(gt_positions):
        raise ValueError(
            f"Position arrays must have same length: {len(pred_positions)} vs {len(gt_positions)}"
        )

    # Get initial positions
    gt_initial = gt_positions[0]
    pred_initial = pred_positions[0]

    # Calculate the translation needed to align initial positions
    translation = gt_initial - pred_initial

    # Apply translation to align initial positions
    aligned_pred = pred_positions + translation

    return aligned_pred


def align_trajectory_with_scale_and_rotation(pred_positions, gt_positions):
    """
    Align predicted trajectory to ground truth using Umeyama algorithm.
    This provides more sophisticated alignment including scale and rotation.

    Args:
        pred_positions: Predicted positions [N, 3]
        gt_positions: Ground truth positions [N, 3]

    Returns:
        Aligned predicted positions [N, 3]
    """
    if len(pred_positions) != len(gt_positions):
        raise ValueError(
            f"Position arrays must have same length: {len(pred_positions)} vs {len(gt_positions)}"
        )

    # Center the trajectories
    gt_centered = gt_positions - np.mean(gt_positions, axis=0)
    pred_centered = pred_positions - np.mean(pred_positions, axis=0)

    # Compute covariance matrix
    H = pred_centered.T @ gt_centered

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    scale = np.sum(S) / np.sum(pred_centered**2)

    # Apply transformation
    aligned_pred_centered = scale * (pred_centered @ R.T)

    # Translate back to original coordinate system
    aligned_pred = aligned_pred_centered + np.mean(gt_positions, axis=0)

    return aligned_pred


def pose_integrate(cfg, dataset, preds_dict, use_local_coordinate=False):
    if cfg.get("data", {}).get("velocity_target", cfg.get("model", {}).get("velocity_target", "window_mean")) == "current":
        # Dataset examples are ordered at sample_freq; target/orientation are at
        # the causal-history endpoint.  Never use the legacy midpoint offset.
        endpoints = np.asarray([entry[1] + dataset.seq_len * dataset.window_size - 1 for entry in dataset.index_map])
        ts = np.asarray(dataset.ts[0])[endpoints]
        quat = np.asarray(preds_dict["orien"])
        if quat.ndim == 3:
            quat = quat[:, -1, :]
        gt_pos = np.asarray(dataset.gt_pos[0])[endpoints]
        pos_pred, pred_world = integrate_current_body_velocity(ts, preds_dict["preds"], quat, gt_pos[0])
        pos_oracle, oracle_world = integrate_current_body_velocity(ts, preds_dict["targets"], quat, gt_pos[0])
        return {
            "ts": ts, "pred_ts": ts, "pos_pred": pos_pred, "pos_gt": gt_pos,
            "pos_oracle": pos_oracle, "preds": preds_dict["preds"],
            "targets": preds_dict["targets"], "pred_world_velocity": pred_world,
            "oracle_world_velocity": oracle_world,
            "vel_body_pred": preds_dict["preds"], "vel_body_gt": preds_dict["targets"],
            "velocity_metrics": current_velocity_metrics(preds_dict["preds"], preds_dict["targets"]),
            "pred_sigmas": np.exp(preds_dict["preds_cov"]),
        }
    use_local_coordinate = cfg["data"]["use_local_coord"]
    preds = preds_dict["preds"]  # pose prediction
    preds_cov = preds_dict["preds_cov"]  # cov prediction
    window_time = cfg["model_param"]["window_time"]  # 1.0
    imu_freq = cfg["data"]["imu_freq"]  # 100.0 or 200.0
    seq_len = cfg["train"]["seq_len"]  # 10
    dp_t = window_time
    pred_vels = preds / dp_t  # 1204*3
    ind = np.array(
        [i[1] for i in dataset.index_map], dtype=np.int32
    )  # array([   0,    5,   10, ..., 6005, 6010, 6015])
    delta_int = int(window_time * imu_freq / 2.0)  # 50.0
    delta_int += int((seq_len - 1) * window_time * imu_freq)  # 50+900
    if not (window_time * imu_freq / 2.0).is_integer():
        logging.info("Trajectory integration point is not centered.")
    ind_intg = (
        ind + delta_int
    )  # array([  1900,   1905,   1910, ..., 123885, 123890, 123895])
    ts = dataset.ts[0]  
    # Use variable time steps instead of mean for more accurate integration
    dts_array = ts[ind_intg[1:]] - ts[ind_intg[:-1]]  # Variable time steps
    dts = np.mean(dts_array)  # Mean for reference
    # Check if time steps are consistent
    if np.std(dts_array) / dts > 0.01:
        logging.warning(f"⚠️  Variable time steps detected! std={np.std(dts_array):.6f}s, mean={dts:.6f}s")
        logging.warning("   This may cause integration errors. Consider using variable time steps.")
    # Equivalent to obtaining the time difference between every 5 frames delta_t
    pos_intg = np.zeros([pred_vels.shape[0] + 1, pred_vels.shape[1]])  # 1205*3
    gt_vel_intg = np.zeros([pred_vels.shape[0] + 1, pred_vels.shape[1]])  # 1205*3
    pos_intg[0] = dataset.gt_pos[0][
        ind_intg[0], 0 : pos_intg.shape[1]
    ]  # Take the GT position of frame 1900 of the current trajectory as the first frame
    gt_vel_intg[0] = dataset.gt_pos[0][ind_intg[0], 0 : pos_intg.shape[1]]
    preds_cov = np.concatenate([np.zeros([1, 3]), preds_cov])
    pos_gt = dataset.gt_pos[0][
        ind_intg[0] : ind_intg[-1], 0 : pos_intg.shape[1]
    ]  # Ground truth pose data from frame 950 to 6965, 6015*3
    # dataset.gt_pos[0] is a list with length 7016, the first pose read is the ground truth pose at timestamp 950
    if use_local_coordinate:
        # FIX 1: Use ground truth orientation at EACH integration point (ind_intg[i])
        # This ensures orientation is always correct and eliminates yaw drift accumulation
        # CRITICAL: Use GT orientation at the actual integration frame, not window start
        integration_oris = dataset.gt_ori[0][ind_intg, :]  # [num_windows, 4] - GT orientation at each integration point
        
        # FIX FOR INITIAL YAW DRIFT: Ensure first orientation exactly matches initial position
        # The initial position is at ind_intg[0], so use that exact orientation
        initial_gt_ori = dataset.gt_ori[0][ind_intg[0], :]
        # Verify the first integration orientation matches
        if not np.allclose(integration_oris[0], initial_gt_ori, atol=1e-6):
            logging.warning("⚠️  First integration orientation doesn't match initial GT orientation!")
            logging.warning("   Using exact initial GT orientation to prevent yaw drift")
            integration_oris[0] = initial_gt_ori.copy()
        
        # Additional check: Verify initial position and orientation are aligned
        initial_pos = pos_intg[0]
        initial_pos_gt = dataset.gt_pos[0][ind_intg[0], :]
        if not np.allclose(initial_pos, initial_pos_gt, atol=1e-6):
            logging.warning("⚠️  Initial position mismatch! Using GT initial position")
            pos_intg[0] = initial_pos_gt.copy()
        
        ## Transform local velocity to global frame using orientation at each integration point
        pred_global_vel = np.zeros_like(pred_vels)
        
        # CRITICAL: Use GT targets directly - these are body-frame velocities (mean over windows)
        # This is what the model was trained on, so it's the ground truth for comparison
        gt_targets_body = preds_dict["targets"]  # [num_windows, 3] - body-frame velocities
        
        # Transform GT targets from body frame to global frame
        gt_global_vel = np.zeros_like(pred_vels)
        for i in range(len(pred_vels)):
            current_rotation = Rotation.from_quat(integration_oris[i])
            gt_global_vel[i] = current_rotation.apply(gt_targets_body[i])
            pred_global_vel[i] = current_rotation.apply(pred_vels[i])
        
        # SANITY CHECK: Also compute GT global velocities from positions for verification
        # (This helps detect if targets are incorrect, but we use targets as primary GT)
        gt_pos_at_intg = dataset.gt_pos[0][ind_intg, :]  # [num_windows, 3]
        if len(ind_intg) > 1:
            avg_spacing = int(np.mean(np.diff(ind_intg)))  # Average frame spacing
        else:
            avg_spacing = int(dts * imu_freq)  # Fallback to time-based spacing
        
        next_pos_idx = ind_intg[-1] + avg_spacing
        if next_pos_idx < len(dataset.gt_pos[0]):
            gt_pos_at_intg_extended = np.vstack([gt_pos_at_intg, dataset.gt_pos[0][next_pos_idx:next_pos_idx+1, :]])
        else:
            gt_pos_at_intg_extended = np.vstack([gt_pos_at_intg, gt_pos_at_intg[-1:]])
        
        gt_global_vel_from_pos = np.diff(gt_pos_at_intg_extended, axis=0) / dts  # [num_windows, 3]
        
        # Verify shapes match
        assert len(gt_global_vel) == len(pred_vels), f"GT vel shape {gt_global_vel.shape} != pred_vels shape {pred_vels.shape}"
        assert len(gt_targets_body) == len(pred_vels), f"GT targets shape {gt_targets_body.shape} != pred_vels shape {pred_vels.shape}"
        
        logging.info(f"\n{'='*80}")
        logging.info("TRAINING VS TEST TRAJECTORY ANALYSIS")
        logging.info(f"{'='*80}")
        logging.info("Hypothesis: If coordinate frame mismatch was the ONLY issue,")
        logging.info("            it would affect training and test equally.")
        logging.info("            The fact that training works suggests:")
        logging.info("            1. Model learned trajectory-specific patterns (overfitting)")
        logging.info("            2. Training/test have different motion characteristics")
        logging.info("            3. Model predictions are worse on test (not just integration)")
        logging.info(f"{'='*80}\n")
        
        # 1. Analyze MODEL PREDICTION QUALITY (before coordinate transformation)
        # This tells us if the issue is in predictions or integration
        pred_local_errors = np.linalg.norm(pred_vels - gt_targets_body, axis=1)
        pred_local_errors_relative = pred_local_errors / (np.linalg.norm(gt_targets_body, axis=1) + 1e-6)
        
        logging.info("1. MODEL PREDICTION QUALITY (Body Frame - Before Transformation):")
        logging.info("   This measures if the model is predicting correctly in body frame")
        logging.info("   (independent of coordinate frame transformation issues)")
        logging.info(f"   Mean absolute error: {np.mean(pred_local_errors):.4f} m/s")
        logging.info(f"   Std absolute error: {np.std(pred_local_errors):.4f} m/s")
        logging.info(f"   Max absolute error: {np.max(pred_local_errors):.4f} m/s")
        logging.info(f"   Mean relative error: {np.mean(pred_local_errors_relative):.2%}")
        logging.info(f"   % of steps with >50% relative error: {100*np.mean(pred_local_errors_relative > 0.5):.1f}%")
        
        # Check if errors are systematic (bias) or random (variance)
        pred_bias = np.mean(pred_vels - gt_targets_body, axis=0)
        pred_variance = np.var(pred_vels - gt_targets_body, axis=0)
        logging.info(f"   Prediction bias (mean error per axis): {pred_bias}")
        logging.info(f"   Prediction variance (error variance per axis): {pred_variance}")
        if np.any(np.abs(pred_bias) > 0.1):
            logging.warning("   ⚠️  Systematic bias detected! Model consistently over/under-predicts")
            logging.warning("      This suggests the model learned training-specific patterns")
        
        # 2. Analyze INITIAL CONDITIONS
        initial_ori = integration_oris[0]
        initial_yaw = Rotation.from_quat(initial_ori).as_euler('xyz')[2]
        final_yaw = Rotation.from_quat(integration_oris[-1]).as_euler('xyz')[2]
        total_yaw_change_gt = final_yaw - initial_yaw
        
        logging.info("\n2. INITIAL CONDITIONS & MOTION CHARACTERISTICS:")
        logging.info(f"   Initial orientation (quat): {initial_ori}")
        logging.info(f"   Initial yaw: {np.degrees(initial_yaw):.2f}°")
        logging.info(f"   Final yaw: {np.degrees(final_yaw):.2f}°")
        logging.info(f"   Total GT yaw change: {np.degrees(total_yaw_change_gt):.2f}°")
        logging.info(f"   Initial position: {pos_intg[0]}")
        logging.info(f"   Trajectory length: {len(pred_vels)} steps")
        logging.info(f"   Total time: {len(pred_vels) * dts:.2f} s")
        
        # Analyze motion patterns
        gt_vel_magnitudes = np.linalg.norm(gt_targets_body, axis=1)
        pred_vel_magnitudes = np.linalg.norm(pred_vels, axis=1)
        logging.info(f"   Mean GT velocity magnitude: {np.mean(gt_vel_magnitudes):.4f} m/s")
        logging.info(f"   Mean pred velocity magnitude: {np.mean(pred_vel_magnitudes):.4f} m/s")
        logging.info(f"   Velocity magnitude ratio: {np.mean(pred_vel_magnitudes) / (np.mean(gt_vel_magnitudes) + 1e-6):.3f}")
        
        # 3. SEPARATE PREDICTION ERRORS FROM INTEGRATION ERRORS
        # If predictions are perfect in body frame, errors come from integration/transformation
        # If predictions are bad in body frame, errors come from model generalization
        logging.info("\n3. ERROR SOURCE ANALYSIS:")
        if np.mean(pred_local_errors) < 0.1:
            logging.info("   ✓ Model predictions are GOOD in body frame (mean error < 0.1 m/s)")
            logging.info("     → Errors likely come from:")
            logging.info("       1. Coordinate frame transformation (orientation mismatch)")
            logging.info("       2. Integration accumulation")
            logging.info("       3. Initial condition misalignment")
        else:
            logging.warning("   ⚠️  Model predictions are POOR in body frame (mean error > 0.1 m/s)")
            logging.warning("      → Primary issue: Model generalization failure")
            logging.warning("        The model is not predicting velocities correctly on this trajectory")
            logging.warning("        This suggests overfitting to training data patterns")
            logging.warning("        Coordinate frame mismatch is secondary")
        
        # 4. COORDINATE FRAME MISMATCH IMPACT (same as before, but contextualized)
        window_start_oris = dataset.gt_ori[0][ind, :]
        ori_diffs = []
        for i in range(min(100, len(pred_vels))):
            ori_ws = Rotation.from_quat(window_start_oris[i])
            ori_intg = Rotation.from_quat(integration_oris[i])
            ori_diff = ori_ws * ori_intg.inv()
            angle_diff = np.linalg.norm(ori_diff.as_rotvec())
            ori_diffs.append(np.degrees(angle_diff))
        
        ori_diffs = np.array(ori_diffs)
        logging.info("\n4. COORDINATE FRAME MISMATCH (Context for Training vs Test):")
        logging.info(f"   Mean orientation difference: {np.mean(ori_diffs):.2f}°")
        logging.info("   This mismatch exists for BOTH training and test trajectories")
        logging.info("   If training works but test doesn't, the mismatch is NOT the primary cause")
        logging.info("   → The model likely learned to compensate for this mismatch on training data")
        logging.info("   → On test data, the model makes different prediction errors that compound")
        
        # 5. CUMULATIVE ERROR ANALYSIS
        # Simulate what happens if we had perfect predictions but coordinate frame mismatch
        # vs what happens with actual predictions
        logging.info("\n5. CUMULATIVE ERROR SIMULATION:")
        
        # Scenario A: Perfect predictions, coordinate frame mismatch only
        perfect_pred_body = gt_targets_body.copy()
        perfect_pred_global = np.zeros_like(pred_global_vel)
        for i in range(len(perfect_pred_body)):
            perfect_pred_global[i] = Rotation.from_quat(integration_oris[i]).apply(perfect_pred_body[i])
        
        # Integrate perfect predictions
        if len(perfect_pred_global) == len(dts_array):
            perfect_pos = pos_intg[0] + np.cumsum(perfect_pred_global * dts_array.reshape(-1, 1), axis=0)
        else:
            perfect_pos = pos_intg[0] + np.cumsum(perfect_pred_global[:-1, :] * dts_array.reshape(-1, 1), axis=0)
            perfect_pos = np.vstack([perfect_pos, perfect_pos[-1] + perfect_pred_global[-1, :] * dts_array[-1]])
        
        perfect_pos = np.vstack([pos_intg[0:1], perfect_pos])
        perfect_final_error = np.linalg.norm(perfect_pos[-1] - dataset.gt_pos[0][ind_intg[-1], :])
        
        # Scenario B: Actual predictions
        actual_final_error = np.linalg.norm(pos_intg[-1] - dataset.gt_pos[0][ind_intg[-1], :])
        
        logging.info("   Scenario A (Perfect predictions + coordinate mismatch):")
        logging.info(f"     Final position error: {perfect_final_error:.4f} m")
        logging.info("   Scenario B (Actual predictions + coordinate mismatch):")
        logging.info(f"     Final position error: {actual_final_error:.4f} m")
        logging.info(f"   Difference (B - A): {actual_final_error - perfect_final_error:.4f} m")
        
        if actual_final_error > perfect_final_error * 2:
            logging.warning("   ⚠️  Actual predictions cause MUCH larger errors than coordinate mismatch alone")
            logging.warning("      → Primary issue: Model prediction quality, not coordinate frame")
        else:
            logging.info("   → Errors are similar, coordinate frame mismatch is significant contributor")
        
        # 6. RECOMMENDATIONS
        logging.info("\n6. ROOT CAUSE & RECOMMENDATIONS:")
        if np.mean(pred_local_errors) > 0.15:
            logging.warning("   PRIMARY ISSUE: Model generalization failure")
            logging.warning("   - Model predictions are poor on this trajectory")
            logging.warning(f"   - Mean body-frame error: {np.mean(pred_local_errors):.4f} m/s")
            logging.warning("   - This suggests overfitting to training data")
            logging.warning("   SOLUTIONS:")
            logging.warning("   1. Improve model generalization (more diverse training, regularization)")
            logging.warning("   2. Use domain adaptation or fine-tuning on test-like data")
            logging.warning("   3. Check if test trajectories have different motion patterns")
        else:
            logging.info("   PRIMARY ISSUE: Coordinate frame mismatch + integration accumulation")
            logging.info(f"   - Model predictions are reasonable (mean error: {np.mean(pred_local_errors):.4f} m/s)")
            logging.info("   - But coordinate frame mismatch causes integration errors")
            logging.info("   SOLUTIONS:")
            logging.info("   1. Fix coordinate frame alignment (use consistent orientation convention)")
            logging.info("   2. Apply periodic drift correction")
            logging.info("   3. Use better integration methods (e.g., quaternion-based)")
        
        logging.info(f"{'='*80}\n")
        
        # ========================================================================
        # END OF COORDINATE FRAME MISMATCH DIAGNOSTIC
        # ========================================================================
        # DEBUG: Check first few velocities before and after transformation
        debug_first_n = min(5, len(pred_vels))
        logging.info(f"\n{'='*80}")
        logging.info(f"TRANSFORMATION DEBUG (first {debug_first_n} steps)")
        logging.info(f"{'='*80}")
        
        for i in range(len(pred_vels)):
            # Debug output for first few steps
            if i < debug_first_n:
                logging.info(f"\nStep {i} (Frame {ind_intg[i]}):")
                logging.info(f"  Local pred vel (body): {pred_vels[i]}")
                logging.info(f"  Local GT vel (body, from targets): {gt_targets_body[i]}")
                logging.info(f"  Global pred vel: {pred_global_vel[i]}")
                logging.info(f"  Global GT vel (from transformed targets): {gt_global_vel[i]}")
                logging.info(f"  Global GT vel (from positions, sanity check): {gt_global_vel_from_pos[i] if i < len(gt_global_vel_from_pos) else 'N/A'}")
                logging.info(f"  Orientation (quat): {integration_oris[i]}")
                # Check if transformation preserves magnitude
                local_pred_mag = np.linalg.norm(pred_vels[i])
                global_pred_mag = np.linalg.norm(pred_global_vel[i])
                local_gt_mag = np.linalg.norm(gt_targets_body[i])
                global_gt_mag = np.linalg.norm(gt_global_vel[i])
                logging.info(f"  Local pred mag: {local_pred_mag:.4f}, Global pred mag: {global_pred_mag:.4f} (should be equal)")
                logging.info(f"  Local GT mag: {local_gt_mag:.4f}, Global GT mag: {global_gt_mag:.4f} (should be equal)")
                if abs(local_pred_mag - global_pred_mag) > 0.001:
                    logging.warning("  ⚠️  Pred magnitude mismatch! Rotation may not be preserving magnitude")
                if abs(local_gt_mag - global_gt_mag) > 0.001:
                    logging.warning("  ⚠️  GT magnitude mismatch! Rotation may not be preserving magnitude")
        

        # FIX 2: Comprehensive debugging for velocity scale and yaw drift
        if len(pred_global_vel) > 0 and len(gt_global_vel) > 0:
            # First, check LOCAL velocities (before transformation) to see if model prediction is the issue
            # Compare predicted body-frame velocities with GT targets (which are already body-frame)
            pred_local_mag = np.linalg.norm(pred_vels, axis=1)
            gt_local_mag_from_targets = np.linalg.norm(gt_targets_body, axis=1)
            local_valid_mask = gt_local_mag_from_targets > 0.01
            
            if np.any(local_valid_mask):
                # Compare predicted body-frame velocities with GT targets (both in body frame)
                local_scale_ratio = np.mean(pred_local_mag[local_valid_mask] / gt_local_mag_from_targets[local_valid_mask])
                
                logging.info(f"\n{'='*80}")
                logging.info("LOCAL VELOCITY (BODY FRAME) ANALYSIS")
                logging.info(f"{'='*80}")
                logging.info(f"Local velocity scale ratio (pred vs GT targets): {local_scale_ratio:.3f} (1.0 = perfect)")
                logging.info(f"Mean pred local velocity: {np.mean(pred_local_mag[local_valid_mask]):.4f} m/s")
                logging.info(f"Mean GT local velocity (from targets): {np.mean(gt_local_mag_from_targets[local_valid_mask]):.4f} m/s")
                if abs(local_scale_ratio - 1.0) > 0.1:
                    logging.warning("⚠️  LOCAL velocity scale mismatch! Model may be predicting wrong scale")
                    logging.warning(f"   Predicted velocities are {local_scale_ratio:.1%} of GT targets")
            
            # Velocity scale analysis (after transformation to global frame)
            pred_vel_mag = np.linalg.norm(pred_global_vel, axis=1)
            gt_vel_mag = np.linalg.norm(gt_global_vel, axis=1)
            valid_mask = gt_vel_mag > 0.01  # Only compare when there's actual motion
            
            if np.any(valid_mask):
                scale_ratio = np.mean(pred_vel_mag[valid_mask] / gt_vel_mag[valid_mask])
                
                # Per-axis scale: use proper ratio (not absolute) to handle sign issues
                # Only compute where GT is significant to avoid division by tiny numbers
                x_mask = valid_mask & (np.abs(gt_global_vel[:, 0]) > 0.01)
                y_mask = valid_mask & (np.abs(gt_global_vel[:, 1]) > 0.01)
                z_mask = valid_mask & (np.abs(gt_global_vel[:, 2]) > 0.01)
                
                if np.any(x_mask):
                    scale_ratio_x = np.mean(pred_global_vel[x_mask, 0] / gt_global_vel[x_mask, 0])
                else:
                    scale_ratio_x = 1.0
                    
                if np.any(y_mask):
                    scale_ratio_y = np.mean(pred_global_vel[y_mask, 1] / gt_global_vel[y_mask, 1])
                else:
                    scale_ratio_y = 1.0
                    
                if np.any(z_mask):
                    scale_ratio_z = np.mean(pred_global_vel[z_mask, 2] / gt_global_vel[z_mask, 2])
                else:
                    scale_ratio_z = 1.0
                
                logging.info(f"\n{'='*80}")
                logging.info("VELOCITY SCALE ANALYSIS")
                logging.info(f"{'='*80}")
                logging.info(f"Overall scale ratio (pred/gt): {scale_ratio:.3f} (1.0 = perfect)")
                logging.info(f"  X-axis scale: {scale_ratio_x:.3f}")
                logging.info(f"  Y-axis scale: {scale_ratio_y:.3f}")
                logging.info(f"  Z-axis scale: {scale_ratio_z:.3f}")
                logging.info(f"Mean pred velocity magnitude: {np.mean(pred_vel_mag[valid_mask]):.4f} m/s")
                logging.info(f"Mean GT velocity magnitude: {np.mean(gt_vel_mag[valid_mask]):.4f} m/s")
                logging.info(f"Integration time step (dts): {dts:.6f} s (mean)")
                logging.info(f"Integration time step std: {np.std(dts_array):.6f} s")
                logging.info(f"Window time: {window_time:.2f} s")
                
                # Additional diagnostics for Z-axis under-prediction
                if abs(scale_ratio_z - 1.0) > 0.2:
                    z_valid = valid_mask & (np.abs(gt_global_vel[:, 2]) > 0.01)
                    if np.any(z_valid):
                        logging.warning(f"\n⚠️  Z-axis under-prediction detected ({scale_ratio_z:.3f} scale)")
                        logging.warning(f"   Mean pred Z velocity: {np.mean(pred_global_vel[z_valid, 2]):.4f} m/s")
                        logging.warning(f"   Mean GT Z velocity: {np.mean(gt_global_vel[z_valid, 2]):.4f} m/s")
                        logging.warning(f"   Z-axis correction factor needed: {1.0/scale_ratio_z:.3f}")
                        logging.warning("   Possible causes:")
                        logging.warning("     - Training data has less Z-axis motion (gravity compensation issue?)")
                        logging.warning("     - Model architecture bias toward horizontal motion")
                        logging.warning("     - Normalization issue (Z-axis may be normalized differently)")
                logging.info("Using GT targets (transformed to global frame) as ground truth")
                
                # Sanity check: Compare position-derived velocities with transformed targets
                if len(gt_global_vel_from_pos) == len(gt_global_vel):
                    pos_derived_valid = np.linalg.norm(gt_global_vel_from_pos, axis=1) > 0.01
                    if np.any(pos_derived_valid):
                        pos_derived_mag = np.linalg.norm(gt_global_vel_from_pos[pos_derived_valid], axis=1)
                        target_transformed_mag = np.linalg.norm(gt_global_vel[pos_derived_valid], axis=1)
                        pos_vs_target_ratio = np.mean(pos_derived_mag / target_transformed_mag) if np.any(target_transformed_mag > 0.01) else 1.0
                        logging.info("\nSanity Check (position-derived vs transformed targets):")
                        logging.info(f"  Ratio: {pos_vs_target_ratio:.3f} (1.0 = perfect match)")
                        if abs(pos_vs_target_ratio - 1.0) > 0.2:
                            logging.warning("  ⚠️  Position-derived velocities differ from targets! This may indicate:")
                            logging.warning("      - Integration points don't align with window boundaries")
                            logging.warning("      - Targets are mean velocities over windows, not instantaneous")
                
                if abs(scale_ratio - 1.0) > 0.1:
                    logging.warning(f"⚠️  Velocity scale mismatch! Predicted velocities are {scale_ratio:.1%} of GT")
                    logging.warning("   This suggests the model may be predicting at wrong scale or units mismatch")
                    logging.warning("   Possible causes:")
                    logging.warning("     1. Model training target computation differs from inference")
                    logging.warning("     2. window_time scaling issue (check function.py line 240)")
                    logging.warning("     3. Normalization/denormalization mismatch")
                    
                    # Suggest per-axis scale correction
                    logging.info(f"\n{'='*80}")
                    logging.info("SUGGESTED PER-AXIS SCALE CORRECTION (for debugging):")
                    logging.info(f"{'='*80}")
                    logging.info(f"  X-axis correction factor: {1.0/scale_ratio_x:.3f}")
                    logging.info(f"  Y-axis correction factor: {1.0/scale_ratio_y:.3f}")
                    logging.info(f"  Z-axis correction factor: {1.0/scale_ratio_z:.3f}")
                    logging.info(f"  Overall correction factor: {1.0/scale_ratio:.3f}")
                    logging.info("  Note: Apply these corrections to predicted velocities before integration")
                    logging.info("        if you want to test if scale mismatch is the main issue")
                    logging.info(f"{'='*80}\n")
            
            # Yaw drift analysis - check orientation changes and trajectory-based yaw
            if len(integration_oris) > 1:
                # Extract yaw angles from quaternions
                rots = Rotation.from_quat(integration_oris)
                yaw_angles = np.array([rot.as_euler('xyz')[2] for rot in rots])  # Extract yaw (z-axis rotation)
                
                # Check if we're using correct orientations
                yaw_diff = np.diff(yaw_angles)
                total_yaw_change = yaw_angles[-1] - yaw_angles[0]
                
                # Compute trajectory-based yaw (from velocity direction in XY plane)
                # This tells us if velocity predictions have systematic yaw errors
                pred_xy_vel = pred_global_vel[:, :2]  # [N, 2] - X and Y components
                gt_xy_vel = gt_global_vel[:, :2]  # [N, 2]
                
                # Compute yaw from velocity direction (atan2(y, x))
                pred_vel_yaw = np.arctan2(pred_xy_vel[:, 1], pred_xy_vel[:, 0])  # [N]
                gt_vel_yaw = np.arctan2(gt_xy_vel[:, 1], gt_xy_vel[:, 0])  # [N]
                
                # Only compute for significant motion
                vel_mag_mask = np.linalg.norm(pred_xy_vel, axis=1) > 0.1
                if np.any(vel_mag_mask):
                    vel_yaw_error = pred_vel_yaw[vel_mag_mask] - gt_vel_yaw[vel_mag_mask]
                    # Wrap to [-pi, pi]
                    vel_yaw_error = np.arctan2(np.sin(vel_yaw_error), np.cos(vel_yaw_error))
                    mean_vel_yaw_error = np.degrees(np.mean(vel_yaw_error))
                    std_vel_yaw_error = np.degrees(np.std(vel_yaw_error))
                else:
                    mean_vel_yaw_error = 0.0
                    std_vel_yaw_error = 0.0
                
                logging.info(f"\n{'='*80}")
                logging.info("YAW DRIFT ANALYSIS")
                logging.info(f"{'='*80}")
                logging.info("GT Orientation Yaw (from quaternions):")
                logging.info(f"  Initial yaw: {np.degrees(yaw_angles[0]):.2f}°")
                logging.info(f"  Final yaw: {np.degrees(yaw_angles[-1]):.2f}°")
                logging.info(f"  Total yaw change: {np.degrees(total_yaw_change):.2f}°")
                logging.info(f"  Mean yaw change per step: {np.degrees(np.mean(np.abs(yaw_diff))):.2f}°")
                logging.info(f"  Max yaw change per step: {np.degrees(np.max(np.abs(yaw_diff))):.2f}°")
                
                logging.info("\nVelocity Direction Yaw (from predicted velocities):")
                if np.any(vel_mag_mask):
                    logging.info(f"  Mean velocity yaw error: {mean_vel_yaw_error:.2f}°")
                    logging.info(f"  Std velocity yaw error: {std_vel_yaw_error:.2f}°")
                    logging.info("  → This indicates if velocity predictions have systematic yaw bias")
                    if abs(mean_vel_yaw_error) > 5.0:
                        logging.warning(f"  ⚠️  Systematic velocity yaw error detected! ({mean_vel_yaw_error:.2f}°)")
                        logging.warning("     This suggests velocity predictions are rotated relative to GT")
                else:
                    logging.info("  Insufficient motion for yaw analysis")
                
                # Check orientation alignment at key points
                logging.info("\nOrientation check at first 5 integration points:")
                for i in range(min(5, len(integration_oris))):
                    frame_idx = ind_intg[i]
                    ori_used = integration_oris[i]
                    ori_gt_at_frame = dataset.gt_ori[0][frame_idx, :]
                    ori_diff = Rotation.from_quat(ori_used) * Rotation.from_quat(ori_gt_at_frame).inv()
                    angle_diff = np.linalg.norm(ori_diff.as_rotvec())
                    logging.info(f"  Step {i}: Frame {frame_idx}, Orientation error: {np.degrees(angle_diff):.4f}°")
                    
                    # Also check velocity direction at this point
                    if i < len(pred_global_vel) and np.linalg.norm(pred_xy_vel[i]) > 0.1:
                        pred_yaw_i = np.degrees(np.arctan2(pred_xy_vel[i, 1], pred_xy_vel[i, 0]))
                        gt_yaw_i = np.degrees(np.arctan2(gt_xy_vel[i, 1], gt_xy_vel[i, 0]))
                        yaw_from_ori = np.degrees(yaw_angles[i])
                        logging.info(f"    Pred vel yaw: {pred_yaw_i:.2f}°, GT vel yaw: {gt_yaw_i:.2f}°, Ori yaw: {yaw_from_ori:.2f}°")
            
            logging.info(f"{'='*80}\n")
        
        # Integrate position with proper orientation at each integration point
        # Use variable time steps for more accurate integration
        # pred_global_vel has shape [num_windows, 3], dts_array has shape [num_windows-1]
        # We need to integrate: pos[i+1] = pos[i] + vel[i] * dt[i]
        if len(pred_global_vel) == len(dts_array):
            # Same length: integrate each velocity with its corresponding time step
            pos_intg[1:] = pos_intg[0] + np.cumsum(pred_global_vel * dts_array.reshape(-1, 1), axis=0)
        elif len(pred_global_vel) == len(dts_array) + 1:
            # One more velocity than time steps: use last time step for last velocity
            pos_intg[1:-1] = pos_intg[0] + np.cumsum(pred_global_vel[:-1, :] * dts_array.reshape(-1, 1), axis=0)
            pos_intg[-1] = pos_intg[-2] + pred_global_vel[-1, :] * dts_array[-1]
        else:
            # Fallback to mean time step if shapes don't match
            logging.warning(f"⚠️  Shape mismatch: pred_global_vel={pred_global_vel.shape}, dts_array={dts_array.shape}")
            logging.warning("   Using mean time step for integration")
            pos_intg[1:] = pos_intg[0] + np.cumsum(pred_global_vel * dts, axis=0)
        
        # GT velocity integration: use transformed targets
        if len(gt_global_vel) == len(dts_array):
            gt_vel_intg[1:] = pos_intg[0] + np.cumsum(gt_global_vel * dts_array.reshape(-1, 1), axis=0)
        elif len(gt_global_vel) == len(dts_array) + 1:
            gt_vel_intg[1:-1] = pos_intg[0] + np.cumsum(gt_global_vel[:-1, :] * dts_array.reshape(-1, 1), axis=0)
            gt_vel_intg[-1] = gt_vel_intg[-2] + gt_global_vel[-1, :] * dts_array[-1]
        else:
            gt_vel_intg[1:] = pos_intg[0] + np.cumsum(gt_global_vel * dts, axis=0)
    else:
        pos_intg[1:] = np.cumsum(pred_vels[:, :] * dts, axis=0) + pos_intg[0]  # 1205*3
        pred_global_vel = pred_vels
        gt_global_vel = None

    ts_intg = np.append(
        ts[ind_intg], ts[ind_intg[-1]] + dts
    )  # Timestamp corresponding to frame 950 and frame 6965 plus additional 5-frame time interval
    ts_in_range = ts[
        ind_intg[0] : ind_intg[-1]
    ]  # In this time range, there are 6015 frames of data corresponding timestamps
    pos_pred = interp1d(ts_intg, pos_intg, axis=0)(
        ts_in_range
    )  # Interpolate predicted timestamps and pose information sampled at 20Hz into 6015 frames of data 6015*3
    # Interpolate velocity predictions to match position timestamps (Global Frame)
    vel_pred_interp = interp1d(ts[ind_intg], pred_global_vel, axis=0)(ts_in_range)
    if use_local_coordinate:
        # Interpolate body-frame velocities for more accurate network evaluation
        vel_gt_interp = interp1d(ts[ind_intg], gt_global_vel, axis=0)(ts_in_range)
        vel_body_pred_interp = interp1d(ts[ind_intg], pred_vels, axis=0)(ts_in_range)
        vel_body_gt_interp = interp1d(ts[ind_intg], gt_targets_body, axis=0)(ts_in_range)
    else:
        vel_gt_interp = None
        vel_body_pred_interp = None
        vel_body_gt_interp = None
    cov_pred = interp1d(ts_intg, preds_cov, axis=0)(ts_in_range)
    logging.info(
        f"pos_pred.shape:{pos_pred.shape} pos_gt.shape {pos_gt.shape} cov_pred.shape: {cov_pred.shape}"
    )

    # Debug: Check if initial positions are actually the same
    initial_diff = np.linalg.norm(pos_pred[0] - pos_gt[0])
    logging.info(f"Initial position difference: {initial_diff:.6f}")
    logging.info(f"GT initial position: {pos_gt[0]}")
    logging.info(f"Pred initial position: {pos_pred[0]}")

    traj_attr_dict = {
        "ts": ts_in_range,
        "pos_pred": pos_pred,  # Use original predicted positions
        "pos_gt": pos_gt,
        # "pos_gtvel_intg": gt_vel_in_pos,
        "cov_pred": cov_pred,
        "vel_pred": vel_pred_interp,  # Use interpolated velocity predictions (Global)
        "vel_gt": vel_gt_interp,      # Use interpolated velocity targets (Global)
        "vel_body_pred": vel_body_pred_interp, # Body-frame predictions
        "vel_body_gt": vel_body_gt_interp,     # Body-frame targets
    }

    return traj_attr_dict


def compute_plot_dict(cfg, net_attr_dict, traj_attr_dict):
    ts = traj_attr_dict["ts"]  # 6015
    pos_pred = traj_attr_dict["pos_pred"]  # 6015*3 (already aligned)
    pos_gt = traj_attr_dict["pos_gt"]
    sample_freq = cfg["data"]["sample_freq"]
    window_time = cfg["model_param"]["window_time"]
    pred_velocity = cfg.get("model", {}).get("pred_velocity", False)

    total_pred = net_attr_dict["preds"].shape[0]  # 1204
    pred_ts = (1.0 / sample_freq) * np.arange(total_pred)  # (0-1203)/20.0
    
    # Scale predictions and covariances for visualization if they are displacement-based 
    # but the ground truth targets are velocity-based.
    # Based on our analysis, the model outputs displacements (m) to match the integration 
    # logic, but the targets in the plot are in m/s.
    preds = net_attr_dict["preds"].copy()
    preds_cov = net_attr_dict["preds_cov"].copy()
    
    # Check if we need to scale from displacement to velocity for the plot
    # If ATE is good, it means preds are displacements. If targets are velocities, we scale.
    # In Humanoid dataset, pred_velocity=True means targets are velocities.
    if pred_velocity:
        preds = preds / window_time
        # Adjust log-sigma: log(sigma/w) = log(sigma) - log(w)
        preds_cov = preds_cov - np.log(window_time)

    pred_sigmas = np.exp(preds_cov)  # (1204, 3)
    plot_dict = {
        "ts": ts,
        "pos_pred": pos_pred,  # Already aligned in pose_integrate
        "pos_gt": pos_gt,
        "pred_ts": pred_ts,
        "preds": preds,
        "targets": net_attr_dict["targets"],
        "pred_sigmas": pred_sigmas,
    }

    return plot_dict


def plot_imus(feat, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    x = len(feat[:, 0])
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(x, feat[:, i])
        plt.plot(x, feat[:, i])
    plt.ylabel("gyr")
    plt.legend()
    plt.grid(True)
    plt.xlabel("t(s)")

    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(x, feat[:, 3 + i])
        plt.plot(x, feat[:, 3 + i])
    plt.ylabel("acc")
    plt.legend()
    plt.grid(True)
    plt.xlabel("t(s)")
    return fig


def rotate_ax(step, ax):
    ax.view_init(30, step * 2)


def make_3d_trj_plots(plot_dict, outdir):
    pos_pred = plot_dict["pos_pred"]  # 6015*3
    pos_gt = plot_dict["pos_gt"]  # 6015*3
    mpl.rcParams["legend.fontsize"] = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], label="network_pred")
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], label="Ground_truth")
    plt.legend(["network_pred", "Ground_truth"])
    if outdir is not None:
        import matplotlib.animation as animation

        anim = animation.FuncAnimation(
            fig, rotate_ax, 180, fargs=(ax,), interval=100, blit=False
        )
        anim.save(osp.join(outdir, "3D_traj.gif"), writer="imagemagick", fps=20)


def make_plots(
    plot_dict, outdir, epoch_num, ave_ate, t_rte, d_rte, use_local=False, ratio=None, full_metrics=None
):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    pos_pred = plot_dict["pos_pred"]  # 6015*3
    pos_gt = plot_dict["pos_gt"]  # 6015*3

    # Calculate trajectory length
    traj_length = np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1))

    # Debug: Check initial positions in visualization
    initial_diff = np.linalg.norm(pos_pred[0] - pos_gt[0])
    print(f"[DEBUG] Visualization - Initial position difference: {initial_diff:.6f}")
    print(f"[DEBUG] GT initial: {pos_gt[0]}")
    print(f"[DEBUG] Pred initial: {pos_pred[0]}")
    print(f"[DEBUG] Are initial positions equal? {np.allclose(pos_pred[0], pos_gt[0])}")
    
    preds = plot_dict["preds"]  # 1204*3
    targets = plot_dict["targets"]  # 1204*3
    dpi = 90
    figsize = (18, 10) # Slightly larger to accommodate text

    fig1 = plt.figure(num="ins_traj", dpi=dpi, figsize=figsize)
    targ_names = ["dx", "dy", "dz"]
    
    # Grid: 3 rows, 4 columns
    # Left side (2 cols): Trajectory and Metrics
    # Right side (2 cols): Velocity components
    
    # --- 1. Trajectory Plot (Top Left) ---
    plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=2)
    plt.plot(pos_gt[:, 0], pos_gt[:, 1], 'r-', linewidth=1.5, label="Ground Truth", alpha=0.8)
    plt.plot(pos_pred[:, 0], pos_pred[:, 1], 'b--', linewidth=1.5, label="Network Prediction")
    plt.axis("equal")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    
    # Extract experiment path for title
    path_parts = outdir.split(os.sep)
    try:
        exp_idx = path_parts.index("exp_result")
        exp_path_display = os.sep.join(path_parts[exp_idx:exp_idx+3])
    except (ValueError, IndexError):
        exp_path_display = outdir

    ratio_str = f"{ratio:.2%}" if ratio is not None else "N/A"
    plt.title(f"Experiment: {exp_path_display}\nEpoch: {epoch_num} | Ratio: {ratio_str}", 
              fontsize=12, fontweight='bold', pad=10)

    # --- 2. Metrics over Epochs (Bottom Left - Left half) ---
    plt.subplot2grid((3, 4), (2, 0), rowspan=1, colspan=1)
    if ave_ate:
        plt.plot(range(1, len(ave_ate) + 1), ave_ate, "-o", markersize=4, label="ATE", color="#d62728")
    if t_rte:
        plt.plot(range(1, len(t_rte) + 1), t_rte, "-s", label="T_RTE", color="#1f77b4")
    if d_rte:
        plt.plot(
            range(1, len(d_rte) + 1), d_rte, "-^", label="D_RTE", color="#9467bd"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Drift Trends", fontsize=10, fontweight='bold')
    plt.legend(prop={'size': 7}, loc='upper right')
    plt.grid(True, alpha=0.3)

    # --- 3. Detailed Metrics Text (Bottom Left - Right half) ---
    plt.subplot2grid((3, 4), (2, 1), rowspan=1, colspan=1)
    plt.axis("off")
    
    ate_val = ave_ate[-1] if ave_ate else 0.0
    t_rte_val = t_rte[-1] if t_rte else 0.0
    d_rte_val = d_rte[-1] if d_rte else 0.0

    metrics_info = (
        f"$\mathbf{{Trajectory\ Stats}}$\n"
        f"Length: {traj_length:.2f} m\n\n"
        f"$\mathbf{{Drift\ Metrics}}$\n"
        f"ATE: {ate_val:.4f} m\n"
        f"T_RTE: {t_rte_val:.4f} m/s\n"
        f"D_RTE: {d_rte_val:.4f} m/1m\n\n"
        f"X_ATE: {full_metrics.get('X_ATE', 0):.4f} m\n"
        f"Y_ATE: {full_metrics.get('Y_ATE', 0):.4f} m\n"
        f"Z_ATE: {full_metrics.get('Z_ATE', 0):.4f} m\n"
        f"$\mathbf{{Detailed\ Errors}}$\n"
        f"AVE: {full_metrics.get('AVE', 0):.4f} m/s\n"
        f"V_RMSE: {full_metrics.get('V_RMSE', 0):.4f} m/s\n"
        f"X_AVE: {full_metrics.get('X_AVE', 0):.4f} m/s\n"
        f"Y_AVE: {full_metrics.get('Y_AVE', 0):.4f} m/s\n"
        f"Z_AVE: {full_metrics.get('Z_AVE', 0):.4f} m/s\n"
    )
    
    plt.text(0.05, 0.5, metrics_info, transform=plt.gca().transAxes, 
             verticalalignment='center', fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f9f9f9', edgecolor='#cccccc', alpha=0.9))

    # --- 4. Velocity Components (Right Side) ---
    for i in range(preds.shape[1]):
        plt.subplot2grid((3, 4), (i, 2), rowspan=1, colspan=2)
        plt.plot(targets[:, i], color='#ff7f0e', linewidth=1.2, alpha=0.7, label="Ground Truth")
        plt.plot(preds[:, i], color='#1f77b4', linewidth=1.0, linestyle='--', label="Prediction")
        
        plt.legend(prop={'size': 7}, loc='upper right')
        
        frame_type = "Body Frame Velocity" if use_local else "Displacement"
        unit = "m/s" if use_local else "m"
        plt.title(f"{targ_names[i].upper()} ({frame_type})", fontsize=10, fontweight='bold')
        plt.ylabel(f"Value ({unit})", fontsize=8)
        plt.grid(True, linestyle=':', alpha=0.5)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(osp.join(outdir, f"2D_traj_{epoch_num}.png"))

    plt.close("all")
