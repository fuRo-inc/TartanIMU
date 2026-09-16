# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Training loop and checkpointing for neural inertial tracking.

Defines the Trainer class (standard training, online adaptation, offline
finetune) plus checkpoint save/rotation helpers.
"""

import logging
import os
import time

import numpy as np
import torch
import wandb
from tartan_imu.model.common import function
from tartan_imu.model.common.losses import speed_dependent_velocity_weight

from tools import distributed_eval

# Try to import lora (may not be available in all environments)
try:
    from tartan_imu.model.adapters import lora
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    logging.warning("LoRA not available - online adaptation may not work properly")

from tartan_imu.model.model_factory import ModelFactory
# Re-exported for backward compatibility: these live in
# tartan_imu.training.checkpoint but must stay accessible as train.<name>
# (e.g. tests call train.save_model, train.cleanup_old_checkpoints).
from tartan_imu.training.checkpoint import (  # noqa: F401
    cleanup_old_checkpoints,
    log_training_metrics,
    save_model,
    write_wandb,
)
from tartan_imu.utils.constants import torch_to_numpy
from tartan_imu.utils.logging_config import get_logger
from tartan_imu.utils.rich_logging import (
    debug,
    info,
    print_training_stats,
    rich_logger,
    success,
    warning,
)

logger = get_logger(__name__)

# Log train loss only every Nth optimizer step to keep stdout readable.
_TRAIN_LOG_EVERY_N_STEPS = 1000
# Save a checkpoint at least this often (epochs) regardless of improvement.
_REGULAR_SAVE_EVERY_N_EPOCHS = 10
# Force an emergency save if this many epochs pass without saving.
_EMERGENCY_SAVE_AFTER_N_EPOCHS = 5


def freeze_backbone_parameters(model):
    """Freeze backbone parameters for finetuning."""
    for name, param in model.named_parameters():
        # Freeze backbone parameters (typically the feature extraction layers)
        # This is a simple implementation - you may need to adjust based on your model structure
        if "backbone" in name or "trunk" in name or "encoder" in name:
            param.requires_grad = False
            logging.info(f"Frozen parameter: {name}")
        else:
            param.requires_grad = True
            logging.info(f"Trainable parameter: {name}")

    return model  # Return the modified model


class Trainer:
    """Simplified trainer class for neural inertial tracking."""

    def __init__(self, args, cfg, model, optimizer, start_epoch=0, resume_state=None):
        # Basic setup
        self.local_rank = args.local_rank
        self.cfg = cfg
        self.device = torch.device(
            f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.optimizer = optimizer
        self.start_epoch = start_epoch

        # Configuration
        self.use_multi_gpu = cfg["train"]["use_multi_gpu"]
        self.use_amp = cfg["train"]["use_amp"]
        self.epochs = cfg["train"]["epochs"]
        self.start_cov_epochs = cfg["train"]["start_cov_epochs"]
        self.out_dir = cfg["train"]["out_dir"]
        self.speed_weighting_cfg = (
            cfg.get("train", {}).get("velocity_loss", {}).get("speed_weighting", {})
            if cfg.get("data", {}).get("velocity_target") == "current"
            else {}
        )

        # Debug output directory
        logging.info(f"Trainer initialization - Output directory: {self.out_dir}")
        logging.info(f"Trainer initialization - Local rank: {self.local_rank}")
        logging.info(f"Trainer initialization - Use multi-GPU: {self.use_multi_gpu}")

        # Validate output directory
        if not os.path.exists(self.out_dir):
            logging.warning(f"Output directory does not exist: {self.out_dir}")
            try:
                os.makedirs(self.out_dir, exist_ok=True)
                logging.info(f"Created output directory: {self.out_dir}")
            except Exception as e:
                logging.error(f"Failed to create output directory: {e}")
        elif not os.path.isdir(self.out_dir):
            logging.error(f"Output path exists but is not a directory: {self.out_dir}")
        else:
            logging.info(f"Output directory is valid: {self.out_dir}")

        # Setup scheduler
        self._setup_scheduler()

        # Setup mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Loss tracking for checkpointing
        self.best_val_loss = float("inf")
        self.best_train_loss = float("inf")
        self.last_save_epoch = 0
        self.significant_improvement_threshold = 0.005  # 0.5% improvement threshold
        self.val_every_n_epochs = int(cfg["train"].get("val_every_n_epochs", 1))
        self.test_every_n_epochs = int(cfg["train"].get("test_every_n_epochs", 1))
        self.min_lr_stop = float(cfg["train"].get("min_lr_stop", 1.1e-6))
        self.resume_state = resume_state or {}
        self._apply_resume_state()

        # Logging setup
        self.log = args.log
        logger.info(f"Trainer initialized on device: {self.device}")
        if self.speed_weighting_cfg.get("enabled", False):
            logger.info(
                "Current-velocity speed weighting enabled: alpha=%s v_ref=%s power=%s normalize_mean=%s",
                self.speed_weighting_cfg.get("alpha", 2.0),
                self.speed_weighting_cfg.get("v_ref", 0.25),
                self.speed_weighting_cfg.get("power", 2.0),
                self.speed_weighting_cfg.get("normalize_mean", True),
            )

    @staticmethod
    def _safe_cuda_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _apply_resume_state(self):
        """Restore trainer bookkeeping state from checkpoint if present."""
        if not self.resume_state:
            return
        self.best_val_loss = self.resume_state.get("best_val_loss", self.best_val_loss)
        self.best_train_loss = self.resume_state.get(
            "best_train_loss", self.best_train_loss
        )
        self.last_save_epoch = self.resume_state.get(
            "last_save_epoch", self.last_save_epoch
        )
        logger.info(
            "Restored trainer state: best_train_loss=%s best_val_loss=%s last_save_epoch=%s",
            self.best_train_loss,
            self.best_val_loss,
            self.last_save_epoch,
        )
        # _apply_resume_state() runs in __init__ AFTER _setup_scheduler() and
        # self.scaler construction, so both objects exist here and can be
        # restored directly.
        sched_state = self.resume_state.get("scheduler_state_dict")
        if sched_state and getattr(self, "scheduler", None) is not None:
            self.scheduler.load_state_dict(sched_state)
            logger.info("Restored LR scheduler state from checkpoint")
        scaler_state = self.resume_state.get("scaler_state_dict")
        if scaler_state and getattr(self, "scaler", None) is not None:
            self.scaler.load_state_dict(scaler_state)
            logger.info("Restored AMP scaler state from checkpoint")

    def get_checkpoint_state(self):
        """Build trainer state payload for checkpoint serialization."""
        return {
            "best_val_loss": self.best_val_loss,
            "best_train_loss": self.best_train_loss,
            "last_save_epoch": self.last_save_epoch,
            "checkpoint_metadata": {
                "velocity_target": self.cfg.get("data", {}).get("velocity_target", "window_mean"),
                "history_duration_s": self.cfg["train"]["seq_len"] * self.cfg["model_param"]["window_time"],
                "seq_len": self.cfg["train"]["seq_len"],
                "imu_frequency_hz": self.cfg["data"]["imu_freq"],
                "sample_frequency_hz": self.cfg["data"]["sample_freq"],
                "output_frame": "base" if self.cfg.get("data", {}).get("velocity_target") == "current" else "legacy",
                "training_source_checkpoint": self.cfg["train"].get("pretrained_source"),
                "dog_head_initialization": self.cfg["train"].get("dog_head_init", "pretrained"),
            },
        }

    def _setup_scheduler(self):
        """Setup learning rate scheduler (ReduceLROnPlateau)."""
        scheduler_cfg = self.cfg["train"]["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=scheduler_cfg["factor"],
            patience=scheduler_cfg["patience"],
            eps=1e-6,
        )

    def inference_step(self, data_loader, epoch):
        """Perform inference on validation/test data."""
        self.model.eval()
        all_results = {"targets": [], "preds": [], "preds_cov": [], "losses": []}

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass
                pred, pred_cov, target, orien, loss, _ = function.fun_test_forward(
                    self.cfg, self.model, batch, self.start_cov_epochs, epoch
                )

                # Collect results
                all_results["targets"].append(target)
                all_results["preds"].append(pred)
                all_results["preds_cov"].append(pred_cov)
                all_results["losses"].append(loss.unsqueeze(0))

        # Concatenate results
        results = self._concatenate_results(all_results, len(data_loader.dataset))

        # Convert to numpy for evaluation
        return {key: torch_to_numpy(value) for key, value in results.items()}

    def _move_batch_to_device(self, batch):
        """Move batch tensors to appropriate device."""
        if self.use_multi_gpu:
            return [t.cuda(self.local_rank, non_blocking=True) for t in batch]
        else:
            return [t.to(self.device) for t in batch]

    def _concatenate_results(self, all_results, dataset_size):
        """Concatenate results from all batches."""
        concatenated = {}
        for key, values in all_results.items():
            concatenated_tensor = torch.concat(values, dim=0)
            if self.use_multi_gpu:
                concatenated[key] = distributed_eval.distributed_concat(
                    concatenated_tensor, dataset_size
                )
            else:
                concatenated[key] = concatenated_tensor
        return concatenated

    def train_step(self, data_loader, epoch, fix_backbone=False, current_frame=None):
        """Run one training epoch and return averaged loss/MSE/timing stats."""
        total_loss = 0.0
        total_mse = 0.0
        total_steps = 0
        total_data_time = 0.0
        total_compute_time = 0.0
        weight_sum = weight_count = 0
        weight_min, weight_max = float("inf"), float("-inf")
        self.model.train()
        # Print trainable params once per training session.
        if not hasattr(self, "_params_printed"):
            ModelFactory.print_trainable_parameters(self.model)
            self._params_printed = True
        data_start = time.time()
        iteration = 0
        if current_frame is not None:
            logging.info(
                f"-------------- Training, current trajectory length time {int(current_frame/200)} s---------------"
            )
        else:
            logging.info("-------------- Training ---------------")
        for bid, batch in enumerate(data_loader):
            iteration = iteration + 1
            if self.use_multi_gpu:
                batch = [t.cuda(self.local_rank, non_blocking=True) for t in batch]
            else:
                batch = [t.to(self.device) for t in batch]
            data_end = time.time()
            data_time = data_end - data_start  # data load time
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred, pred_cov, targ, loss = function.fun_train_forward_efficient(
                    self.cfg, self.model, batch, self.start_cov_epochs, epoch
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Log train loss periodically to reduce output frequency.
            if hasattr(self, "train_log_counter"):
                self.train_log_counter += 1
            else:
                self.train_log_counter = 0

            if self.train_log_counter % _TRAIN_LOG_EVERY_N_STEPS == 0:
                logging.info(f"train loss: {loss}")
            back_end = time.time()
            inferback_time = back_end - data_end  # training and backward time
            batch_loss = float(loss.detach().item())
            batch_mse = float(torch.mean((targ.detach() - pred.detach()) ** 2).item())
            if self.speed_weighting_cfg.get("enabled", False):
                with torch.no_grad():
                    weights = speed_dependent_velocity_weight(targ, self.speed_weighting_cfg)
                    weight_sum += float(weights.sum().item())
                    weight_count += weights.numel()
                    weight_min = min(weight_min, float(weights.min().item()))
                    weight_max = max(weight_max, float(weights.max().item()))
            total_loss += batch_loss
            total_mse += batch_mse
            total_steps += 1
            total_data_time += data_time
            total_compute_time += inferback_time
            data_start = time.time()

        if total_steps == 0:
            return {
                "avg_loss": float("inf"),
                "avg_mse": float("inf"),
                "num_steps": 0,
                "avg_data_time": float("inf"),
                "avg_compute_time": float("inf"),
            }
        result = {
            "avg_loss": total_loss / total_steps,
            "avg_mse": total_mse / total_steps,
            "num_steps": total_steps,
            "avg_data_time": total_data_time / total_steps,
            "avg_compute_time": total_compute_time / total_steps,
        }
        if weight_count:
            result.update(
                mean_speed_weight=weight_sum / weight_count,
                min_speed_weight=weight_min,
                max_speed_weight=weight_max,
            )
            logger.info(
                "Epoch speed weights: mean=%.6f min=%.6f max=%.6f",
                result["mean_speed_weight"], result["min_speed_weight"], result["max_speed_weight"],
            )
        return result

    def train(self, train_loader, val_loader=None, test_loader=None):
        """Run the standard training loop over all epochs with optional val/test."""
        # Use existing tracking variables if resuming, otherwise initialize
        if self.start_epoch == 0:
            # Fresh training - initialize tracking variables
            self.best_val_loss = float("inf")
            self.best_train_loss = float("inf")
            self.last_save_epoch = 0
            logging.info("Initialized tracking variables for fresh training")
        else:
            # Resuming training - keep existing tracking variables
            logging.info(
                f"Resuming training from epoch {self.start_epoch} with existing tracking variables"
            )

        epoch_val_loss, epoch_val_mse, epoch_test_loss = [], [], []
        epoch_train_loss, epoch_train_mse, epoch_test_mse = [], [], []

        # main training loop
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)
            rich_logger.training_header(epoch, self.epochs, "Training")
            validation_loss, validation_mse, validation_time, test_loss_val = (
                [],
                [],
                [],
                [],
            )
            test_mse_val, test_time = [], []

            start_t = time.time()
            train_attr_dict = self.train_step(train_loader, epoch)
            end_t = time.time()
            epoch_time = end_t - start_t
            info(f"Whole epoch time: {epoch_time:.2f}s")
            train_loss = float(train_attr_dict["avg_loss"])
            train_mse_val = float(train_attr_dict["avg_mse"])
            avg_data_time = float(train_attr_dict.get("avg_data_time", 0.0))
            avg_compute_time = float(train_attr_dict.get("avg_compute_time", 0.0))
            instance_per_second = len(train_loader.dataset) / epoch_time
            logging.info(
                "Batch timing breakdown - avg data: %.4fs, avg compute: %.4fs, ratio data/compute: %.2f",
                avg_data_time,
                avg_compute_time,
                (avg_data_time / avg_compute_time) if avg_compute_time > 0 else float("inf"),
            )

            if self.log:
                write_wandb("train", train_loss, epoch, self.local_rank)
                write_wandb("run_time", epoch_time, epoch, self.local_rank)
                write_wandb(
                    "instance_per_second", instance_per_second, epoch, self.local_rank
                )
                write_wandb("train_mse_val", train_mse_val, epoch, self.local_rank)
                write_wandb(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch, self.local_rank
                )

            epoch_train_loss.append(train_loss)  # The loss for each epoch is the mean of all batch losses
            epoch_train_mse.append(train_mse_val)

            # Run validation if available
            validation_loss = None
            validation_mse = None
            should_run_val = (
                val_loader is not None and (epoch % self.val_every_n_epochs == 0)
            )
            if should_run_val:
                start_t = time.time()
                val_attr_dict = self.inference_step(val_loader, epoch)
                end_t = time.time()

                validation_loss = np.average(val_attr_dict["losses"])
                validation_mse = np.mean(
                    (val_attr_dict["targets"] - val_attr_dict["preds"]) ** 2
                )
                validation_time = end_t - start_t

                if self.log:
                    write_wandb(
                        "validation_loss", validation_loss, epoch, self.local_rank
                    )
                    write_wandb(
                        "validation_mse", validation_mse, epoch, self.local_rank
                    )
                    write_wandb(
                        "validation_time", validation_time, epoch, self.local_rank
                    )

                epoch_val_loss.append(validation_loss)
                epoch_val_mse.append(validation_mse)

            # Unified scheduler step (always use training loss)
            current_train_loss = train_loss
            scheduler_metric = (
                validation_loss if validation_loss is not None else current_train_loss
            )
            if hasattr(self.scheduler, "step") and callable(
                getattr(self.scheduler, "step", None)
            ):
                self.scheduler.step(scheduler_metric)
            logging.info(
                "LR scheduler stepped with metric %.6f", scheduler_metric
            )

            # Run test set if available
            test_loss_val = None
            should_run_test = (
                test_loader is not None and (epoch % self.test_every_n_epochs == 0)
            )
            if should_run_test:
                start_t = time.time()
                test_attr_dict = self.inference_step(test_loader, epoch)
                end_t = time.time()
                test_loss_val = np.average(test_attr_dict["losses"])
                test_mse_val = np.mean(
                    (test_attr_dict["targets"] - test_attr_dict["preds"]) ** 2
                )
                test_time = end_t - start_t

                if self.log:
                    write_wandb("test_loss", test_loss_val, epoch, self.local_rank)
                    write_wandb("test_mse", test_mse_val, epoch, self.local_rank)
                    write_wandb("test_time", test_time, epoch, self.local_rank)

                epoch_test_loss.append(test_loss_val)
                epoch_test_mse.append(test_mse_val)

                # Display test statistics
                print_training_stats(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=validation_loss if validation_loss is not None else None,
                    test_loss=test_loss_val,
                    lr=self.optimizer.param_groups[0]["lr"],
                    epoch_time=epoch_time,
                )

            if self.optimizer.param_groups[0]["lr"] < self.min_lr_stop:
                logging.info(
                    "Stopping early due to min_lr_stop threshold (lr=%s threshold=%s)",
                    self.optimizer.param_groups[0]["lr"],
                    self.min_lr_stop,
                )
                break

            # Unified checkpoint saving strategy
            self.save_checkpoint_strategy(
                epoch, current_train_loss, validation_loss, test_loss_val
            )

        # Only finish wandb from rank 0
        if self.local_rank == 0 and wandb.run is not None:
            wandb.finish()

    def _plot_epoch_loss_results(
        self,
        test_loader,
        val_loader,
        epoch_train_loss,
        epoch_train_mse,
        epoch_val_loss,
        epoch_val_mse,
        epoch_test_loss,
        epoch_test_mse,
    ):
        """Save and plot per-epoch train/val/test loss and MSE curves."""
        from tartan_imu.training.plots import plot_epoch_loss_results

        return plot_epoch_loss_results(
            self.cfg,
            self.optimizer,
            self.out_dir,
            test_loader,
            val_loader,
            epoch_train_loss,
            epoch_train_mse,
            epoch_val_loss,
            epoch_val_mse,
            epoch_test_loss,
            epoch_test_mse,
        )

    def _plot_online_adaptation_results(
        self, epoch_train_loss, epoch_train_mse, total_epochs
    ):
        """Enhanced plotting for online adaptation results."""
        from tartan_imu.training.plots import plot_online_adaptation_results

        return plot_online_adaptation_results(
            self.cfg,
            self.optimizer,
            self.out_dir,
            epoch_train_loss,
            epoch_train_mse,
            total_epochs,
        )

    def _plot_offline_finetune_results(
        self, epoch_train_loss, epoch_val_loss, total_epochs
    ):
        """Enhanced plotting for offline finetuning results."""
        from tartan_imu.training.plots import plot_offline_finetune_results

        return plot_offline_finetune_results(
            self.cfg,
            self.optimizer,
            self.out_dir,
            epoch_train_loss,
            epoch_val_loss,
            total_epochs,
        )

    def online_adaptation(
        self, train_loader, tester, resume_model, test_path_list, ate_thres
    ):
        """Run the online-adaptation loop: repeatedly re-run training (LoRA-adapted
        if available) until the LR floor is reached. ``current_frame`` and
        ``total_trajectory_frames`` only feed a progress-ratio log and a training
        log line; they do not slice or limit the training data."""
        # will always loop until the ave_ate meet requirements
        epoch_num = 0
        epoch_train_loss, epoch_train_mse = [], []
        ave_ate = float("inf")

        # Fix 5: Initialize tracking variables for finetune stage
        if not hasattr(self, "best_train_loss") or self.best_train_loss == float("inf"):
            self.best_train_loss = float("inf")
            self.last_save_epoch = 0  # Add missing initialization
            info("Initialized tracking variables for finetune stage")

        # freeze subt base model params
        # self.model = freeze_backbone_parameters(self.model)

        # add lora params to base model (if available)
        if LORA_AVAILABLE:
            if hasattr(self.model, "module"):
                lora.replace_layers(self.model.module)
            else:
                lora.replace_layers(self.model)
            # add lora params to optimizer
            new_parameters = []
            for param in self.model.parameters():
                if param.requires_grad:
                    new_parameters.append(param)
            for param_group in self.optimizer.param_groups:
                param_group["params"].extend(new_parameters)
        else:
            logging.warning("LoRA not available - using standard finetuning approach")
        # Dataset-specific frame counts used only for the progress-ratio log in
        # online adaptation (3000 frames @ 200 Hz == 20 s). Values are
        # intentionally unchanged.
        total_trajectory_frames = 23997
        current_frame = 3000  # 20s @ 200 Hz
        time_buffer = []
        while True:
            epoch_num = epoch_num + 1
            rich_logger.training_header(
                epoch_num, 100, "Online Adaptation"
            )  # Use 100 as max epochs for display
            torch.cuda.synchronize()
            start_t = time.time()
            train_attr_dict = self.train_step(
                train_loader, epoch_num, fix_backbone=True, current_frame=current_frame
            )
            torch.cuda.synchronize()
            end_t = time.time()
            epoch_time = end_t - start_t
            time_buffer.append(epoch_time)
            ratio = current_frame / total_trajectory_frames
            if sum(time_buffer) > int((current_frame + 200 - 4000) / 200):
                current_frame += 200
            # train_loss = np.average(train_attr_dict["losses"])
            # train_loss_mse = np.mean(
            #     (train_attr_dict["targets"] - train_attr_dict["preds"]) ** 2
            # )
            train_loss = float(train_attr_dict["avg_loss"])
            train_loss_mse = float(train_attr_dict["avg_mse"])
                        
            instance_per_second = len(train_loader.dataset) / epoch_time

            if self.log:
                write_wandb("online_adapt/time", epoch_time, epoch_num, self.local_rank)
                write_wandb(
                    "online_adapt/instance_time",
                    instance_per_second,
                    epoch_num,
                    self.local_rank,
                )
                write_wandb(
                    "online_adapt/epoch_time", epoch_time, epoch_num, self.local_rank
                )
                write_wandb(
                    "online_adapt/train_loss", train_loss, epoch_num, self.local_rank
                )
                write_wandb(
                    "online_adapt/train_loss_mse",
                    train_loss_mse,
                    epoch_num,
                    self.local_rank,
                )

            epoch_train_loss.append(train_loss)  # The loss for each epoch is the mean of all batch losses
            epoch_train_mse.append(train_loss_mse)
            info(f'Average loss: {np.average(train_attr_dict["losses"]):.6f}')
            self.scheduler.step(np.average(train_attr_dict["losses"]))
            logging.info(
                f"current learning rate: {self.optimizer.param_groups[0]['lr']}"
            )
            if self.optimizer.param_groups[0]["lr"] < 1.1e-6:
                break

            self.save_checkpoint_strategy(epoch_num, train_loss, None, None)

            resume_model = self.model
            resume_model.eval()

            # Safety check for tester
            if tester is None or not hasattr(tester, "test"):
                warning("Tester is not available or invalid. Skipping test evaluation.")
                ave_ate = float("inf")  # Default value
            else:
                try:
                    all_metrics = tester.test(
                        test_path_list, epoch_num, resume_model, ratio=ratio
                    )
                    ave_ate = all_metrics["all_traj"]["avg_ate"]
                except Exception as e:
                    warning(f"Error during test evaluation: {e}")
                    ave_ate = float("inf")  # Default value

            # Log training statistics (replaced log_setting with standard logging)
            logging.info("Training Statistics:")
            training_statics = {
                "Epoch_num": epoch_num,
                "epoch_time": epoch_time,
                "Learning_rate": self.optimizer.param_groups[0]["lr"],
                "Train_loss": np.average(train_attr_dict["losses"]),
                "avg_atr": ave_ate,
            }
            logging.info(f"Training Statistics: {training_statics}")

        # Enhanced plotting for online adaptation
        if len(epoch_train_loss) > 0:
            self._plot_online_adaptation_results(
                epoch_train_loss, epoch_train_mse, epoch_num
            )

        return resume_model

    def offline_finetune(
        self, train_loader, val_loader=None, test_loader=None, max_epochs=20
    ):
        """
        Simple offline finetuning - much more stable than online adaptation

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
            max_epochs: Maximum number of finetuning epochs
        """
        logging.info("Starting offline finetuning...")

        # Freeze backbone parameters for finetuning (only once)
        if not hasattr(self, "_backbone_frozen"):
            if self.cfg["model"].get("adapter") == "lora":
                lora.mark_only_lora_as_trainable(self.model)
                lora.print_lora_parameters(self.model)
            else:
                self.model = freeze_backbone_parameters(self.model)
            self._backbone_frozen = True

        # Track finetuning progress
        epoch_train_loss, epoch_val_loss = [], []

        for epoch in range(1, max_epochs + 1):
            logging.info(
                f"-------------- Offline Finetune Epoch {epoch}/{max_epochs} ---------------"
            )

            # Training step
            train_attr_dict = self.train_step(train_loader, epoch, fix_backbone=True)
            # train_loss = np.average(train_attr_dict["losses"])
            # train_mse = np.mean(
            #     (train_attr_dict["targets"] - train_attr_dict["preds"]) ** 2
            # )
            train_loss = float(train_attr_dict["avg_loss"])
            train_mse = float(train_attr_dict["avg_mse"])
            epoch_train_loss.append(train_loss)

            # Validation step (if available)
            val_loss = None
            if val_loader:
                val_attr_dict = self.inference_step(val_loader, epoch)
                val_loss = np.average(val_attr_dict["losses"])
                val_mse = np.mean(
                    (val_attr_dict["targets"] - val_attr_dict["preds"]) ** 2
                )
                epoch_val_loss.append(val_loss)

                if self.log:
                    write_wandb("finetune/val_loss", val_loss, epoch, self.local_rank)
                    write_wandb("finetune/val_mse", val_mse, epoch, self.local_rank)

            # Learning rate scheduling
            if val_loader and val_loss is not None:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            if self.log:
                write_wandb("finetune/train_loss", train_loss, epoch, self.local_rank)
                write_wandb("finetune/train_mse", train_mse, epoch, self.local_rank)
                write_wandb(
                    "finetune/learning_rate", current_lr, epoch, self.local_rank
                )

            logging.info(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, lr={current_lr:.6f}"
            )
            if val_loss:
                logging.info(f"Epoch {epoch}: val_loss={val_loss:.6f}")

            # Use unified checkpoint saving strategy
            self.save_checkpoint_strategy(epoch, train_loss, val_loss, None)

            # Early stopping conditions
            if current_lr < 1e-6:
                logging.info(
                    f"Learning rate too small ({current_lr:.6f}), stopping finetuning"
                )
                break

            if val_loader and val_loss and epoch > 10:
                # Check if validation loss hasn't improved for 5 epochs
                if len(epoch_val_loss) >= 5:
                    recent_val_losses = epoch_val_loss[-5:]
                    if all(
                        recent_val_losses[i] >= recent_val_losses[i - 1]
                        for i in range(1, 5)
                    ):
                        logging.info(
                            "Validation loss not improving for 5 epochs, stopping finetuning"
                        )
                        break

        # Always save final model (only from rank 0)
        if self.local_rank == 0:
            save_model(
                self.out_dir,
                max_epochs,
                self.model,
                self.optimizer,
                self.use_multi_gpu,
                self.local_rank,
                trainer_state=self.get_checkpoint_state(),
                scheduler_state=self.scheduler.state_dict() if getattr(self, "scheduler", None) is not None else {},
                scaler_state=self.scaler.state_dict() if getattr(self, "scaler", None) is not None else {},
            )
            logging.info(f"Final finetune model saved at epoch {max_epochs}")
        else:
            logging.info(f"Final model save skipped (rank {self.local_rank})")

        logging.info("Offline finetuning completed!")

        # Enhanced plotting for offline finetuning
        if len(epoch_train_loss) > 0:
            self._plot_offline_finetune_results(
                epoch_train_loss, epoch_val_loss, max_epochs
            )

        return self.model

    def save_checkpoint_strategy(self, epoch, train_loss, val_loss, test_loss):
        """
        Unified checkpoint saving strategy with clear logic and proper rank handling.

        Args:
            epoch: Current epoch number
            train_loss: Current training loss
            val_loss: Current validation loss (None if no validation)
            test_loss: Current test loss (None if no test)
        """
        # Only save from rank 0 to avoid multiple checkpoints
        if self.local_rank != 0:
            return

        should_save = False
        save_reason = ""

        # Always save the first epoch
        if self.best_train_loss == float("inf"):
            self.best_train_loss = train_loss
            should_save = True
            save_reason = "first_epoch"
            success(f"First epoch checkpoint saved - loss: {train_loss:.6f}")

        # Save on validation loss improvement
        elif val_loss is not None and val_loss < self.best_val_loss:
            if self.best_val_loss == float("inf"):
                improvement = 1.0
            else:
                improvement = (self.best_val_loss - val_loss) / self.best_val_loss
            if improvement > self.significant_improvement_threshold:
                self.best_val_loss = val_loss
                should_save = True
                save_reason = "improved_val_loss"
                success(
                    f"Validation loss improved by {improvement:.3f} - saving checkpoint"
                )

        # Save on training loss improvement (with lower threshold)
        elif train_loss < self.best_train_loss:
            improvement = (self.best_train_loss - train_loss) / self.best_train_loss
            if improvement > 0.005:  # Lower threshold to 0.5% for more frequent saves
                self.best_train_loss = train_loss
                should_save = True
                save_reason = "improved_train_loss"
                success(
                    f"Training loss improved by {improvement:.3f} - saving checkpoint"
                )

        # Save if validation is better than best training loss
        elif val_loss is not None and val_loss < self.best_train_loss:
            prev_best_train = self.best_train_loss
            self.best_train_loss = val_loss
            should_save = True
            save_reason = "validation_better_than_train"
            success(
                f"Validation loss {val_loss:.6f} better than best train loss {prev_best_train:.6f} - saving checkpoint"
            )

        # Regular saves on a fixed epoch cadence (moved out of else block)
        if epoch % _REGULAR_SAVE_EVERY_N_EPOCHS == 0 and not should_save:
            should_save = True
            save_reason = "regular_save_every_10_epochs"
            info(f"Regular checkpoint save every 10 epochs - epoch {epoch}")

        # Emergency save if no checkpoint for several epochs
        elif (
            epoch - self.last_save_epoch >= _EMERGENCY_SAVE_AFTER_N_EPOCHS
            and not should_save
        ):
            should_save = True
            save_reason = "emergency_save_after_5_epochs"
            warning(
                f"Emergency checkpoint save after 5 epochs without saving - epoch {epoch}"
            )

        # Save the checkpoint
        if should_save:
            save_model(
                self.out_dir,
                epoch,
                self.model,
                self.optimizer,
                self.use_multi_gpu,
                self.local_rank,
                save_reason,
                trainer_state=self.get_checkpoint_state(),
                scheduler_state=self.scheduler.state_dict() if getattr(self, "scheduler", None) is not None else {},
                scaler_state=self.scaler.state_dict() if getattr(self, "scaler", None) is not None else {},
            )
            self.last_save_epoch = epoch
            success(f"Checkpoint saved: {save_reason} at epoch {epoch}")
        else:
            debug(
                f"Checkpoint NOT saved at epoch {epoch}. should_save={should_save}, best_train_loss={self.best_train_loss:.6f}, current_train_loss={train_loss:.6f}, epoch_diff={epoch - self.last_save_epoch}"
            )
