# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Configuration and model/trainer/tester assembly helpers.

Loads YAML configs (with simple ``{var}`` interpolation), and builds the
runtime objects from them: the network (:func:`build_model`, which dispatches
through :mod:`model.registry`), the trainer (:func:`build_trainer`), and the
tester (:func:`build_tester`).
"""

import logging
import os
import re
import test

import torch
import torch.distributed as dist
import torch.nn as nn
import train
import yaml
from tartan_imu.model.lstm import model_lstm
from tartan_imu.utils.rich_logging import (
    banner,
    success_highlight,
)


def interpolate_config(cfg, context=None):
    """
    Recursively interpolate strings in the config using values from the same config.
    e.g., out_dir: ../exp_result/{experiment_name}
    """
    if context is None:
        context = cfg

    if isinstance(cfg, dict):
        # First pass: collect all top-level keys for the current level's context
        # In this specific codebase, we often want to reach into 'train' or 'data'
        flat_context = {}
        def collect_flat(d, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    collect_flat(v, prefix + k + ".")
                else:
                    flat_context[prefix + k] = v
                    flat_context[k] = v # Also add without prefix for convenience
        collect_flat(context)

        for k, v in cfg.items():
            if isinstance(v, (dict, list)):
                interpolate_config(v, context)
            elif isinstance(v, str) and "{" in v and "}" in v:
                try:
                    cfg[k] = v.format(**flat_context)
                except KeyError:
                    pass # Ignore if key not found
    elif isinstance(cfg, list):
        for i in range(len(cfg)):
            if isinstance(cfg[i], (dict, list)):
                interpolate_config(cfg[i], context)
            elif isinstance(cfg[i], str) and "{" in cfg[i] and "}" in cfg[i]:
                # List items don't have easy context access but we use flat_context
                pass 

def load_config(path):
    """Load a YAML config file and interpolate its ``{var}`` placeholders.

    Args:
        path (str): path to config file
    """
    with open(path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)
    
    # Interpolate variables like {experiment_name}
    interpolate_config(cfg_special)

    return cfg_special


def update_recursive(dict1, dict2):
    """Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def build_model(args, cfg):
    """Build the network described by ``cfg`` and place it on the right device.

    Merges the model-specific YAML (``cfg["model"]["model_yaml"]``) into ``cfg``,
    dispatches on ``cfg["model"]["model_name"]`` to construct the backbone, and
    wraps it for multi-GPU training when ``cfg["train"]["use_multi_gpu"]`` is set.

    Args:
        args: parsed CLI args; ``args.local_rank`` selects the CUDA device.
        cfg (dict): the (already loaded) experiment config.

    Returns:
        The network moved to the target device (DDP-wrapped if multi-GPU).
    """
    device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    with open(cfg["model"]["model_yaml"], "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)
    update_recursive(cfg, cfg_special)
    model_name = cfg["model"]["model_name"]
    if model_name == "resnet_lstm":
        model = model_lstm.ResNetLSTMSeqNet(cfg)
    elif model_name == "resnet_lstm_light":
        model = model_lstm.ResNetLSTMSeqNet_Light(cfg)
    elif model_name == "IMU_transformer":
        raise NotImplementedError("IMU_transformer model is not available")
    else:
        # Transformer / Foundation_Model (and any future backbone) are built via
        # the model registry. Characterization-tested to be identical to the
        # former in-line construction (tests/unit/test_model_registry.py).
        from tartan_imu.model.registry import build_backbone

        model = build_backbone(model_name, cfg)

    ## use the multi GPU to train the network
    if cfg["train"]["use_multi_gpu"]:
        logging.info(f"torch.cuda.device_count() {torch.cuda.device_count()}")

        # Check if distributed process group is initialized
        if not dist.is_initialized():
            logging.warning(
                "Distributed process group not initialized. Falling back to single GPU mode."
            )
            network = model.to(device)
        else:
            # Note: torch.distributed.init_process_group() is already called in main_net.py
            torch.cuda.set_device(args.local_rank)
            
            # Check if distributed training is properly initialized before calling barrier
            if dist.is_initialized():
                dist.barrier()
            else:
                logging.warning("Distributed training not initialized, skipping barrier")
                
            # SyncBN
            model = (
                nn.SyncBatchNorm.convert_sync_batchnorm(model)
                .to(device)
                .cuda(args.local_rank)
            )
            network = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            # Check if distributed training is properly initialized before calling barrier
            if dist.is_initialized():
                dist.barrier()
            else:
                logging.warning("Distributed training not initialized, skipping barrier")
    else:
        # Single GPU mode - don't use distributed training
        logging.info("Running in single GPU mode - skipping distributed setup")
        network = model.to(device)

    return network


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split("([0-9]+)", v_str)]


def GetBestModel(path):
    """Return the path to the latest checkpoint file in ``path``.

    Lists ``path``, sorts the names with a natural (numeric-aware) sort, and
    returns the absolute path of the last regular file (the highest epoch).
    """
    names = sorted(os.listdir(path + "/"), key=str2int)
    files = []
    for name in names:
        if os.path.isfile(os.path.join(os.path.abspath(path), name)):
            files.append(name)
    model = os.path.join(os.path.abspath(path), files[-1])  # Select the last model
    logging.info(f"load model: {model}")
    return model


def build_trainer(args, cfg, model, pretrained_path=None, resume_path=None, **kwargs):
    """Build a trainer with explicit warm-start versus resume semantics.

    ``pretrained_path`` loads only model weights and starts a new run.  In
    contrast, ``resume_path`` restores all serialized training state.
    """
    if pretrained_path and resume_path:
        raise ValueError("--checkpoint and --resume_from cannot be used together")
    full_covariance = bool(
        cfg.get("model", {}).get("velocity_covariance", {}).get("enabled", False)
        and cfg.get("train", {}).get("covariance", {}).get("enabled", False)
    )
    covariance_train_cfg = cfg.get("train", {}).get("covariance", {})
    if full_covariance:
        # This is applied before optimizer construction so Stage 1 contains
        # only the new dog covariance head in its trainable parameter group.
        freeze_backbone = covariance_train_cfg.get("freeze_backbone", False)
        freeze_velocity_head = covariance_train_cfg.get("freeze_velocity_head", False)
        for name, parameter in model.named_parameters():
            is_dog_covariance = name.startswith("heads.dog.velocity_covariance_head.")
            is_dog_velocity = name.startswith("heads.dog.") and not is_dog_covariance
            parameter.requires_grad = is_dog_covariance or (is_dog_velocity and not freeze_velocity_head) or (not freeze_backbone and not name.startswith("heads.dog."))
        logging.info("Full covariance freeze policy: backbone=%s velocity_head=%s; covariance head always trainable", freeze_backbone, freeze_velocity_head)
    start_epoch = 0
    optim = (
        torch.optim.Adam
        if cfg["train"]["optimizer"]["method"] == "Adam"
        else torch.optim.SGD
    )  # Adam
    opt_cfg = cfg["train"]["optimizer"]
    # Separate parameter groups make Phase 2's low-LR trunk tuning explicit.
    if cfg["data"].get("velocity_target") == "current":
        backbone = [p for n, p in model.named_parameters() if not n.startswith("heads.dog.")]
        dog_velocity_head = [p for n, p in model.named_parameters() if n.startswith("heads.dog.") and ".velocity_covariance_head." not in n]
        dog_covariance_head = [p for n, p in model.named_parameters() if n.startswith("heads.dog.velocity_covariance_head.")]
        groups = [
            {"params": [p for p in backbone if p.requires_grad], "lr": opt_cfg.get("backbone_learning_rate", opt_cfg["learning_rate"])},
            {"params": [p for p in dog_velocity_head if p.requires_grad], "lr": opt_cfg.get("head_learning_rate", opt_cfg["learning_rate"])},
        ]
        if full_covariance:
            groups.append({"params": [p for p in dog_covariance_head if p.requires_grad], "lr": covariance_train_cfg.get("head_learning_rate", opt_cfg.get("head_learning_rate", opt_cfg["learning_rate"]))})
        optimizer = optim(groups, weight_decay=opt_cfg["weight_decay"])
    else:
        optimizer = optim(model.parameters(), opt_cfg["learning_rate"], weight_decay=opt_cfg["weight_decay"])
    resume_state = {}
    source_path = resume_path or pretrained_path
    # Preserve the established legacy config-only continuation behavior.  New
    # CLI paths never infer one another: --checkpoint is always warm start and
    # --resume_from is always resume.
    legacy_resume = False
    if not source_path and cfg["train"].get("use_pretrain_model", False):
        source_path = GetBestModel(os.path.join(cfg["train"]["out_dir"], "checkpoints"))
        legacy_resume = True
    if source_path:
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        is_resume = bool(resume_path) or legacy_resume
        # Reinitializing a head is a warm-start option only. A resume must
        # faithfully restore the already-running model.
        reinit_head = (not is_resume and cfg["train"].get("dog_head_init", "pretrained") == "reinitialize")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_state = model.state_dict()
        incompatible = [k for k, v in state_dict.items() if k in model_state and v.shape != model_state[k].shape]
        disallowed = [k for k in incompatible if not (reinit_head and k.startswith("heads.dog."))]
        if disallowed:
            raise RuntimeError(f"Incompatible checkpoint architecture tensors: {disallowed}")
        allowed_new_covariance = lambda key: full_covariance and key.startswith("heads.") and ".velocity_covariance_head." in key
        missing = [k for k in model_state if k not in state_dict and not (reinit_head and k.startswith("heads.dog.")) and not allowed_new_covariance(k)]
        if missing:
            raise RuntimeError(f"Checkpoint is missing required model tensors: {missing}")
        if reinit_head:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("heads.dog.")}
        target_model = model.module if cfg["train"].get("use_multi_gpu") and hasattr(model, "module") else model
        if full_covariance and not is_resume:
            # Strict=False is deliberately constrained by the checks above:
            # only newly introduced covariance-head tensors may be missing.
            incompatible_keys = target_model.load_state_dict(state_dict, strict=False)
            unexpected = [k for k in incompatible_keys.unexpected_keys]
            unexpected_non_cov = [k for k in unexpected if not allowed_new_covariance(k)]
            missing_non_cov = [k for k in incompatible_keys.missing_keys if not allowed_new_covariance(k)]
            if unexpected_non_cov or missing_non_cov:
                raise RuntimeError(f"Unexpected checkpoint mismatch: missing={missing_non_cov}, unexpected={unexpected_non_cov}")
        else:
            target_model.load_state_dict(state_dict, strict=not reinit_head)

        if is_resume:
            start_epoch = checkpoint.get("epoch", 0)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            resume_state = dict(checkpoint.get("trainer_state", {}))
            resume_state["scheduler_state_dict"] = checkpoint.get("scheduler_state_dict", {})
            resume_state["scaler_state_dict"] = checkpoint.get("scaler_state_dict", {})
            logging.info("Training initialization mode: resume\nSource checkpoint: %s\nStarting epoch: %s\nOptimizer: restored\nScheduler: restored\nAMP scaler: restored", source_path, start_epoch)
        else:
            logging.info("Training initialization mode: pretrained warm start\nSource checkpoint: %s\nStarting epoch: 0\nOptimizer: fresh\nScheduler: fresh\nAMP scaler: fresh", source_path)
    else:
        logging.info("Training initialization mode: fresh\nStarting epoch: 0\nOptimizer: fresh\nScheduler: fresh\nAMP scaler: fresh")

    if (not full_covariance and cfg["data"].get("velocity_target") == "current" and cfg["train"].get("freeze_backbone", False)):
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("heads.dog.")
    return train.Trainer(
        args,
        cfg,
        model=model,
        optimizer=optimizer,
        start_epoch=start_epoch,
        resume_state=resume_state,
    )


def build_tester(args, cfg, model, check_path, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Highlight test model loading process
    banner("LOADING MODEL FOR TESTING")

    if check_path:
        checkpoint = torch.load(check_path, map_location=device, weights_only=False)
        success_highlight(f"✅ LOADING TEST MODEL FROM SPECIFIED PATH: {check_path}")
    else:  # default resume form test out_dir
        checkpoint_path = GetBestModel(
            os.path.join(cfg["train"]["out_dir"], "checkpoints")
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        success_highlight(f"✅ LOADING BEST MODEL FOR TESTING: {checkpoint_path}")

    # Get model information
    model_epoch = checkpoint.get("epoch", "Unknown")

    # Create test model information table
    from rich import box
    from rich.table import Table

    test_table = Table(title="🧪 Test Model Loading Information", box=box.ROUNDED)
    test_table.add_column("Property", style="cyan", no_wrap=True)
    test_table.add_column("Value", style="green")
    test_table.add_column("Status", style="yellow")

    test_table.add_row("Model Epoch", str(model_epoch), "✅")
    test_table.add_row(
        "Checkpoint Path", str(check_path if check_path else checkpoint_path), "✅"
    )
    test_table.add_row(
        "GPU Mode", "Multi-GPU" if cfg["train"]["use_multi_gpu"] else "Single-GPU", "✅"
    )

    # Load model weights
    if cfg["train"]["use_multi_gpu"]:
        # Check if model is wrapped in DistributedDataParallel
        if hasattr(model, "module"):
            model.module.load_state_dict(
                checkpoint.get("model_state_dict"), strict=False
            )
            test_table.add_row("Model Weights", "Loaded (Multi-GPU)", "✅")
        else:
            model.load_state_dict(checkpoint.get("model_state_dict"), strict=False)
            test_table.add_row("Model Weights", "Loaded (Single-GPU)", "✅")
    else:
        model.load_state_dict(checkpoint.get("model_state_dict"), strict=False)
        test_table.add_row("Model Weights", "Loaded (Single-GPU)", "✅")

    model.eval()
    test_table.add_row("Model State", "Evaluation Mode", "✅")

    # Display the table
    from rich.console import Console

    console = Console()
    console.print(test_table)

    success_highlight("🎯 MODEL READY FOR TESTING")

    tester = test.tester(args, cfg, model)

    return tester
