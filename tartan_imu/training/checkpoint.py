# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Checkpoint saving/rotation and metric-logging helpers.

Module-level helpers extracted from the top-level ``train.py`` training
pipeline. These handle wandb/tensorboard metric logging plus checkpoint
serialization (with best-model variants) and epoch-checkpoint rotation.

Behaviour is intentionally identical to the previous in-``train.py`` versions;
this module only relocates the code. ``train.py`` re-exports these names so
existing call sites (including ``train.save_model``) keep working.
"""

import logging
import os
from os import path as osp

import torch
import wandb

# Keep at most this many epoch checkpoints (plus best-model checkpoints).
# Used only by ``save_model`` below (via its ``cleanup_old_checkpoints`` call),
# so it lives here rather than in train.py.
_KEEP_LAST_CHECKPOINTS = 10


def write_wandb(header, objs, epoch_i, local_rank=0):
    """Write metrics to wandb.

    Args:
        header: Metric name/key to log under.
        objs: Scalar value to log.
        epoch_i: Step index passed to ``wandb.log``.
        local_rank: Process rank; only rank 0 logs to avoid duplicate entries.
    """
    # Only log from rank 0 to avoid duplicate entries
    if wandb.run is not None and local_rank == 0:
        wandb.log({header: objs}, step=epoch_i)


def log_training_metrics(summary_writer, mode, ml_loss, epoch, optimizer):
    """Log training metrics to tensorboard.

    Args:
        summary_writer: TensorBoard ``SummaryWriter`` instance.
        mode: Metric-group label (e.g. ``"train"``/``"val"``).
        ml_loss: Scalar loss value to record.
        epoch: Current epoch index.
        optimizer: Optimizer whose learning rate is logged.
    """
    summary_writer.add_scalar(f"{mode}_dist/loss_full", ml_loss, epoch)
    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
        )
    logging.info(f"{mode}: average ml loss: {ml_loss}")


def save_model(
    path,
    epoch,
    network,
    optimizer,
    use_multi_gpu=False,
    local_rank=0,
    save_reason="",
    trainer_state=None,
    scheduler_state=None,
    scaler_state=None,
):
    """Save model checkpoint with enhanced error handling, multiple checkpoint support, and rotation.

    Args:
        path: Output directory; a ``checkpoints`` subdirectory is created under it.
        epoch: Current epoch, used in the epoch-checkpoint filename.
        network: Model to serialize (unwrapped if wrapped in ``DistributedDataParallel``).
        optimizer: Optimizer whose state is serialized.
        use_multi_gpu: Multi-GPU flag (logged on error only).
        local_rank: Process rank (logged on error only).
        save_reason: When ``"improved_val_loss"``/``"improved_train_loss"``/
            ``"validation_better_than_train"``, also writes a best-model checkpoint.
        trainer_state: Optional trainer bookkeeping payload.
        scheduler_state: Optional LR-scheduler state dict.
        scaler_state: Optional AMP scaler state dict.
    """
    try:
        # Validate output directory
        if not os.path.isdir(path):
            logging.error(f"Invalid output directory: {path}")
            raise ValueError(f"Invalid output directory: {path}")

        # Create checkpoints directory
        checkpoints_dir = osp.join(path, "checkpoints")
        if not osp.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir, exist_ok=True)
            logging.info(f"Created checkpoints directory: {checkpoints_dir}")

        # Always save epoch-specific checkpoint for history
        epoch_checkpoint = osp.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pt")

        # Additional best model checkpoints (these get overwritten)
        best_checkpoints = []
        if save_reason == "improved_val_loss":
            best_checkpoints.append(
                osp.join(checkpoints_dir, "checkpoint_best_val_loss.pt")
            )
        elif save_reason == "improved_train_loss":
            best_checkpoints.append(
                osp.join(checkpoints_dir, "checkpoint_best_train_loss.pt")
            )
        elif save_reason == "validation_better_than_train":
            best_checkpoints.append(
                osp.join(checkpoints_dir, "checkpoint_best_val_loss.pt")
            )

        # Use epoch checkpoint as primary path
        model_path = epoch_checkpoint

        metadata = (trainer_state or {}).get("checkpoint_metadata", {})
        # Prepare state dict (handle both wrapped and unwrapped models)
        if hasattr(network, "module"):
            # Model is wrapped in DistributedDataParallel
            state_dict = {
                "model_state_dict": network.module.state_dict() if hasattr(network, "module") else network.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "trainer_state": trainer_state or {},
                "scheduler_state_dict": scheduler_state or {},
                "scaler_state_dict": scaler_state or {},
                "metadata": metadata,
            }
        else:
            # Model is not wrapped
            state_dict = {
                "model_state_dict": network.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "trainer_state": trainer_state or {},
                "scheduler_state_dict": scheduler_state or {},
                "scaler_state_dict": scaler_state or {},
                "metadata": metadata,
            }

        # Save epoch-specific checkpoint
        torch.save(state_dict, model_path)
        logging.info(f"Epoch checkpoint saved to {model_path}")

        # Save best model checkpoints (if this is a best model)
        for best_path in best_checkpoints:
            torch.save(state_dict, best_path)
            logging.info(f"Best model checkpoint saved to {best_path}")

        # Implement checkpoint rotation (keep last N epochs + best models)
        cleanup_old_checkpoints(checkpoints_dir, keep_last=_KEEP_LAST_CHECKPOINTS)

        # Verify files were created
        all_paths = [model_path] + best_checkpoints
        for path_to_check in all_paths:
            if os.path.exists(path_to_check):
                file_size = os.path.getsize(path_to_check)
                logging.info(
                    f"Checkpoint verified. Size: {file_size} bytes - {path_to_check}"
                )
            else:
                logging.error(f"Checkpoint file was not created: {path_to_check}")

    except Exception as e:
        logging.error(f"Error saving model checkpoint: {e}")
        logging.error(
            f"Path: {path}, Epoch: {epoch}, Use multi-GPU: {use_multi_gpu}, Rank: {local_rank}"
        )
        raise


def cleanup_old_checkpoints(checkpoints_dir, keep_last=10):
    """Clean up old epoch checkpoints, keeping only the most recent ones and best models.

    Args:
        checkpoints_dir: Directory holding ``checkpoint_epoch_*.pt`` files.
        keep_last: Number of most-recent epoch checkpoints to retain.
    """
    try:
        import glob
        import re

        # Find all epoch checkpoints
        epoch_pattern = osp.join(checkpoints_dir, "checkpoint_epoch_*.pt")
        epoch_files = glob.glob(epoch_pattern)

        if len(epoch_files) <= keep_last:
            return  # No cleanup needed

        # Extract epoch numbers and sort
        epoch_info = []
        for filepath in epoch_files:
            filename = osp.basename(filepath)
            match = re.search(r"checkpoint_epoch_(\d+)\.pt", filename)
            if match:
                epoch_num = int(match.group(1))
                epoch_info.append((epoch_num, filepath))

        # Sort by epoch number and keep only the most recent
        epoch_info.sort(key=lambda x: x[0])
        files_to_remove = epoch_info[:-keep_last]  # Remove all but last N

        for epoch_num, filepath in files_to_remove:
            try:
                os.remove(filepath)
                logging.info(f"Removed old checkpoint: {filepath}")
            except OSError as e:
                logging.warning(f"Could not remove old checkpoint {filepath}: {e}")

    except Exception as e:
        logging.warning(f"Error during checkpoint cleanup: {e}")
