# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Train/test forward-pass helpers that wire models to the multi-head losses.

These functions handle motion-type head selection, optional covariance
prediction, KV-cache plumbing for the transformer, and the prediction/target
reshaping the training loop expects.
"""
import logging
from collections import Counter

import torch
from tartan_imu.model.common.losses import (
    efficient_multi_head_smooth_loss,
    get_sequence_smooth_loss,
    multi_head_smooth_loss,
)


def _current_velocity_loss_config(cfg):
    """Enable speed weighting only for causal endpoint velocity targets."""
    if cfg.get("data", {}).get("velocity_target", "window_mean") != "current":
        return None
    return cfg.get("train", {}).get("velocity_loss", {}).get("speed_weighting")


def fun_train_forward(cfg, model, batch, start_cov_epochs, epoch):
    """
    Training forward pass with proper motion type handling.

    Args:
        cfg (dict): Configuration dictionary
        model (nn.Module): Neural network model
        batch (tuple): (features, targets, aux_data, motion_type)
        start_cov_epochs (int): Epoch to start covariance prediction
        epoch (int): Current epoch

    Returns:
        tuple: (predictions, covariances, targets, loss)
    """
    feat, targ, _, motion_type = (
        batch  # feat: [batch_size, seq_len, 6, frames], targ: [batch_size, seq_len, 3]
    )

    if epoch <= start_cov_epochs:  # Before covariance prediction
        pred_multi_head = model(feat, motion_type)  # Use motion_type for optimization
        pred_multi_cov = {}
        for key, value in pred_multi_head.items():
            if isinstance(value, torch.Tensor):
                pred_multi_cov[key] = torch.zeros_like(value)
    else:  # Predict covariance
        logging.info("Starting covariance prediction training")
        pred_multi_head, pred_multi_cov = model(feat, motion_type, predict_cov=True)

    motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}
    multi_head_mask = {}

    for motion_id, motion_name in motion_types.items():
        multi_head_mask[motion_name] = motion_type == motion_id

    for key in list(pred_multi_head.keys()):
        # Only process prediction heads, ignore auxiliary outputs like present_kv
        if not isinstance(pred_multi_head[key], torch.Tensor):
            continue
            
        output_dim = cfg["model_param"]["output_dim"]  # 3
        pred_multi_head[key] = pred_multi_head[key][:, cfg["train"]["predict_start"] :]
        pred_multi_cov[key] = pred_multi_cov[key][
            :,
            cfg["train"]["predict_start"] :,
        ]  # pred_cov: [1, 10, 3]
        targ = targ[:, cfg["train"]["predict_start"] :, 0:output_dim]  # [1, 10, 3]
        if cfg.get("data", {}).get("velocity_target", "window_mean") == "current":
            pred_multi_head[key] = pred_multi_head[key][:, -1:]
            pred_multi_cov[key] = pred_multi_cov[key][:, -1:]
            targ = targ[:, -1:]

    # Note: Velocity scaling is now handled by learnable parameters in OutputHead

    loss = multi_head_smooth_loss(
        pred_multi_head,
        pred_multi_cov,
        targ,
        epoch,
        multi_head_mask,
        start_cov_epochs,
        cfg["data"]["use_local_coord"],
        _current_velocity_loss_config(cfg),
    )

    last_key = None
    for key in list(pred_multi_head.keys()):
        if not isinstance(pred_multi_head[key], torch.Tensor):
            continue
        all_b = pred_multi_head[key].size(0) * pred_multi_head[key].size(1)  # 1*10
        pred_multi_head[key] = pred_multi_head[key].contiguous().view(all_b, -1)  # 10*3
        pred_multi_cov[key] = pred_multi_cov[key].contiguous().view(all_b, -1)  # 10*3
        targ = targ.contiguous().view(all_b, -1)  # 10*3
        last_key = key

    return pred_multi_head[last_key], pred_multi_cov[last_key], targ, loss


def fun_train_forward_efficient(cfg, model, batch, start_cov_epochs, epoch):
    """
    Efficient training forward pass that only computes needed heads.
    """
    feat, targ, _, motion_type = batch

    needed_heads = get_active_heads(cfg, motion_type)

    if epoch <= start_cov_epochs:
        # Only compute predictions for needed heads
        outputs = model(feat, motion_type, compute_all_heads=False)
        if isinstance(outputs, dict):
            pred_multi_head = outputs
        else:
            pred_multi_head = {"human": outputs}  # Fallback
            
        pred_multi_cov = {}
        for key in list(pred_multi_head.keys()):
            value = pred_multi_head[key]
            if isinstance(value, torch.Tensor):
                pred_multi_cov[key] = torch.zeros_like(value)
            else:
                # Remove non-tensor auxiliary data from the head dict to avoid keeping it in graph
                pred_multi_head.pop(key)
    else:
        outputs, pred_multi_cov = model(
            feat, motion_type, predict_cov=True, compute_all_heads=False
        )
        if isinstance(outputs, dict):
            pred_multi_head = outputs
        else:
            pred_multi_head = {"human": outputs}
            
        # Clean up non-tensor auxiliary data
        for key in list(pred_multi_head.keys()):
            if not isinstance(pred_multi_head[key], torch.Tensor):
                pred_multi_head.pop(key)

    # Create masks only for needed heads
    multi_head_mask = {}
    motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}

    for motion_id, motion_name in motion_types.items():
        if motion_name in needed_heads and motion_name in pred_multi_head:
            multi_head_mask[motion_name] = motion_type == motion_id

    # Process predictions
    for key in list(pred_multi_head.keys()):
        # Only process prediction heads, ignore auxiliary outputs like present_kv
        if not isinstance(pred_multi_head[key], torch.Tensor):
            continue
            
        output_dim = cfg["model_param"]["output_dim"]
        pred_multi_head[key] = pred_multi_head[key][
            :,
            cfg["train"]["predict_start"] :,
        ]
        pred_multi_cov[key] = pred_multi_cov[key][
            :,
            cfg["train"]["predict_start"] :,
        ]
        targ = targ[:, cfg["train"]["predict_start"] :, 0:output_dim]
        if cfg.get("data", {}).get("velocity_target", "window_mean") == "current":
            pred_multi_head[key] = pred_multi_head[key][:, -1:]
            pred_multi_cov[key] = pred_multi_cov[key][:, -1:]
            targ = targ[:, -1:]

    # Note: Velocity scaling is now handled by learnable parameters in OutputHead

    loss = efficient_multi_head_smooth_loss(
        pred_multi_head,
        pred_multi_cov,
        targ,
        epoch,
        multi_head_mask,
        start_cov_epochs,
        cfg["data"]["use_local_coord"],
        _current_velocity_loss_config(cfg),
    )

    last_key = None
    for key in list(pred_multi_head.keys()):
        if not isinstance(pred_multi_head[key], torch.Tensor):
            continue
        all_b = pred_multi_head[key].size(0) * pred_multi_head[key].size(1)  # 1*10
        pred_multi_head[key] = pred_multi_head[key].contiguous().view(all_b, -1)  # 10*3
        pred_multi_cov[key] = pred_multi_cov[key].contiguous().view(all_b, -1)  # 10*3
        targ = targ.contiguous().view(all_b, -1)  # 10*3
        last_key = key

    return pred_multi_head[last_key], pred_multi_cov[last_key], targ, loss


def get_needed_heads_from_motion_type(motion_type):
    """Extract needed heads from motion type tensor."""
    if isinstance(motion_type, torch.Tensor):
        unique_types = torch.unique(motion_type).tolist()
    else:
        unique_types = [motion_type]

    motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}
    needed_heads = []

    for motion_id in unique_types:
        if motion_id in motion_types:
            needed_heads.append(motion_types[motion_id])

    return needed_heads


def get_active_heads(cfg, motion_type):
    """Return heads to train; prefer config over batch inference."""
    ah = cfg.get("train", {}).get("active_heads", cfg.get("active_heads", None))
    if ah is not None and len(ah) > 0:
        return list(ah)
    # fallback (your existing behavior)
    return get_needed_heads_from_motion_type(motion_type)


def fun_test_forward(cfg, model, batch, start_cov_epochs, epoch, past_kv=None, time_offset=0):
    """
    Testing forward pass with proper motion type handling and KV-cache support.

    Args:
        cfg (dict): Configuration dictionary
        model (nn.Module): Neural network model
        batch (tuple): (features, targets, orientations, motion_type)
        start_cov_epochs (int): Epoch to start covariance prediction
        epoch (int): Current epoch
        past_kv (list, optional): Previous KV cache
        time_offset (int, optional): Current time offset for positional embeddings

    Returns:
        tuple: (predictions, covariances, targets, orientations, loss, present_kv)
    """
    feat, targ, orien, motion_type = batch

    # Determine the most common motion type in the batch
    frequency = Counter(motion_type.tolist())
    most_motion_pattern, count = frequency.most_common(1)[0]
    motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}

    if most_motion_pattern not in motion_types:
        raise ValueError(f"Unknown motion type: {most_motion_pattern}")

    motion_dynamic = motion_types[most_motion_pattern]

    # Check if model supports KV cache
    use_kv_cache = cfg.get("model_param", {}).get("use_kv_cache", False)
    
    if epoch <= start_cov_epochs:
        if use_kv_cache:
            outputs = model(feat, motion_type, past_kv=past_kv, time_offset=time_offset)
            pred_multi_head = outputs
            present_kv = outputs.get("present_kv", None)
        else:
            pred_multi_head = model(feat, motion_type)
            present_kv = None
            
        pred = pred_multi_head[motion_dynamic]
        pred_cov = torch.zeros_like(pred)
    else:
        if use_kv_cache:
            outputs, pred_multi_cov = model(
                feat, motion_type, predict_cov=True, past_kv=past_kv, time_offset=time_offset
            )
            pred_multi_head = outputs
            present_kv = outputs.get("present_kv", None)
        else:
            pred_multi_head, pred_multi_cov = model(
                feat, motion_type, predict_cov=True
            )
            present_kv = None
            
        pred = pred_multi_head[motion_dynamic]
        pred_cov = pred_multi_cov[motion_dynamic]
        
    output_dim = cfg["model_param"]["output_dim"]
    pred = pred[
        :,
        -1,
    ]  # Test output: last prediction result of the sequence (B,3)
    pred_cov = pred_cov[
        :,
        -1,
    ]  # Variance of the last prediction (B,3)
    targ = targ[:, -1, 0:output_dim]  # (B,3)

    imu_freq = int(cfg["data"]["imu_freq"])
    if cfg.get("data", {}).get("velocity_target", "window_mean") == "current":
        # Same timestamp as target: last raw IMU sample in the causal history.
        orien = orien[:, -1, :]
    else:
        orien = orien[:, -imu_freq:]
    
    if cfg["model"]["pred_velocity"] and cfg.get("data", {}).get("velocity_target", "window_mean") != "current":
        pred = pred * cfg["model_param"]["window_time"]
        pred_cov = (
            torch.log(cfg["model_param"]["window_time"] * torch.ones_like(pred_cov))
            + pred_cov
        )

        # Apply velocity smoothing to reduce jittering
        # from tartan_imu.model.common.losses import smooth_velocity_predictions
        # pred = smooth_velocity_predictions(pred, window_size=3)

    loss = get_sequence_smooth_loss(
        pred,
        pred_cov,
        targ,
        epoch,
        start_cov_epochs,
        velocity_loss_config=_current_velocity_loss_config(cfg),
    )

    return pred, pred_cov, targ, orien, loss, present_kv
