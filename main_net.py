# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Entry point for neural inertial tracking.

Parses args, loads/validates config, sets up (multi-)GPU, builds data loaders
and the model, then dispatches to training/online-adaptation/testing.
"""

import argparse
import datetime
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from tartan_imu.config import configer
from tartan_imu.dataloader.paths import (
    GetDataPath,
    GetTartanAirDataPath,
    GetTartanAirDataPath_test,
    GetTartanAirDataPath_val,
    get_category_path,
    get_list_of_combined_dir,
    get_list_of_dir,
)
from tools import distributed_eval
from torch.utils.data import DataLoader
from tartan_imu.utils.registry import load_dataset_module
from tartan_imu.utils.device import get_device
from tartan_imu.utils.rich_logging import (
    info,
    rich_logger,
    warning,
)

logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)

def set_seeds(seed):
    """Seed Python/NumPy/Torch RNGs and force deterministic cuDNN."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_output_dir(out_dir):
    """Create the output dir plus its checkpoints/ and logs/ subdirectories."""
    try:
        if out_dir is not None:
            # Create directories with exist_ok=True to handle existing directories
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
            logging.info(f"Training output writes to {out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
    except ValueError as e:
        logging.error(e)
        return


class WorkerInit:
    """DataLoader worker_init_fn that seeds NumPy per worker for reproducibility."""

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed(self.seed + worker_id)


def train_load_data(cfg, args, paths, world_size):
    """Build train/val/test DataLoaders from the resolved data paths."""
    train_data_path_combined, val_data_path_combined, test_data_path_combined = (
        [],
        [],
        [],
    )

    for dir in paths:
        if dir == "train":
            for category in paths[dir]:
                train_data_path_combined.append(paths[dir][category])
                info(f"  Item: {category}, train Path: {train_data_path_combined}")
        if dir == "val":
            for category in paths[dir]:
                val_data_path_combined.append(paths[dir][category])
                info(f"  Item: {category}, val Path: {val_data_path_combined}")
        if dir == "test":
            for category in paths[dir]:
                test_data_path_combined.append(paths[dir][category])
                info(f"  Item: {category}, test Path: {test_data_path_combined}")

    dataset_name = cfg["data"]["dataset"]
    dataset_utils = load_dataset_module(dataset_name)
    info(f"Current Dataset is {dataset_name} dataset", style="cyan")
    train_loader, val_loader, test_loader = None, None, None

    worker_init_fn = WorkerInit(int(cfg["seeds"]["id"]))
    train_prefetch_factor = int(cfg["train"].get("prefetch_factor", 2))
    val_prefetch_factor = int(cfg.get("val", {}).get("prefetch_factor", 2))
    test_prefetch_factor = int(cfg.get("test", {}).get("prefetch_factor", 2))

    if cfg["data"]["random_partition"]:  # FALSE
        ## random select train, val, test data
        logging.info("random select train, val, test data")
        train_data_path_list = get_list_of_combined_dir(train_data_path_combined)
        basic_data = dataset_utils.BasicSequenceData(
            cfg, train_data_path_list, mode="train"
        )

        train_index_map, valid_index_map = dataset_utils.partition_data(
            basic_data.get_index_map(),
            valid_samples=basic_data.valid_samples,
            valid_all_samples=basic_data.valid_all_samples,
            data_paths=basic_data.data_paths,
            out_path=cfg["data"]["validation_dir"],
            training_rate=cfg["data"]["train_rate"],
            valuation_rate=cfg["data"]["valid_rate"],
            data_rate=cfg["data"]["data_rate"],
        )
        train_dataset = dataset_utils.SeqToSeqDataset(
            cfg, basic_data, train_index_map, mode="train"
        )

        val_dataset = dataset_utils.SeqToSeqDataset(
            cfg, basic_data, valid_index_map, mode="val"
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg["val"]["batch_size"], shuffle=False
        )
    else:
        ## train data
        logging.info("process train data")
        train_data_path_list = get_list_of_combined_dir(train_data_path_combined)
        logging.info("train_data_path_list count: %d", len(train_data_path_list))
        train_basic_data = dataset_utils.BasicSequenceData(
            cfg, train_data_path_list, mode="train"
        )  # Load all ground truth files.

        train_dataset = dataset_utils.SeqToSeqDataset(
            cfg, train_basic_data, train_basic_data.get_merged_index_map(), mode="train"
        )
        ## val data
        if (
            "validation_dir" in cfg["data"]
            and cfg["data"]["validation_dir"] is not None
        ) or (
            "val_scene_list" in cfg["data"]
            and cfg["data"]["val_scene_list"] is not None
        ):
            logging.info("process val data")
            val_data_path_list = get_list_of_combined_dir(val_data_path_combined)
            logging.info("val_data_path_list count: %d", len(val_data_path_list))
            valid_basic_data = dataset_utils.BasicSequenceData(
                cfg, val_data_path_list, mode="val"
            )
            val_dataset = dataset_utils.SeqToSeqDataset(
                cfg,
                valid_basic_data,
                valid_basic_data.get_merged_index_map(),
                mode="val",
            )

            if cfg["train"]["use_multi_gpu"] and dist.is_initialized():
                val_sampler = distributed_eval.SequentialDistributedSampler(
                    val_dataset, batch_size=cfg["val"]["batch_size"]
                )
                val_loader_kwargs = dict(
                    batch_size=cfg["val"]["batch_size"],
                    shuffle=False,
                    pin_memory=True,
                    sampler=val_sampler,
                    num_workers=cfg["val"]["n_workers"],
                )
                if cfg["val"]["n_workers"] > 0:
                    val_loader_kwargs.update(
                        multiprocessing_context="fork",
                        persistent_workers=True,
                        prefetch_factor=val_prefetch_factor,
                    )
                val_loader = DataLoader(
                    val_dataset,
                    **val_loader_kwargs,
                )
            else:
                val_loader_kwargs = dict(
                    batch_size=cfg["val"]["batch_size"],
                    shuffle=False,
                    pin_memory=True,
                    num_workers=cfg["val"]["n_workers"],
                )
                if cfg["val"]["n_workers"] > 0:
                    val_loader_kwargs.update(
                        multiprocessing_context="fork",
                        persistent_workers=True,
                        prefetch_factor=val_prefetch_factor,
                    )
                val_loader = DataLoader(
                    val_dataset,
                    **val_loader_kwargs,
                )

    ## test data Load test data
    if ("test_dir" in cfg["data"] and cfg["data"]["test_dir"] is not None) or (
        "test_scene_list" in cfg["data"] and cfg["data"]["test_scene_list"] is not None
    ):
        logging.info("process test data")
        test_data_path_list = get_list_of_combined_dir(test_data_path_combined)
        logging.info("test_data_path_list count: %d", len(test_data_path_list))
        test_basic_data = dataset_utils.BasicSequenceData(
            cfg, test_data_path_list, mode="test"
        )
        test_dataset = dataset_utils.SeqToSeqDataset(
            cfg, test_basic_data, test_basic_data.get_merged_index_map(), mode="test"
        )

        # Check if test configuration exists
        if "test" not in cfg or "batch_size" not in cfg["test"]:
            logging.error(
                "Test configuration missing 'batch_size'. Using train batch_size as fallback."
            )
            test_batch_size = cfg["train"]["batch_size"]
            test_n_workers = cfg["train"].get("n_workers", 0)
        else:
            test_batch_size = cfg["test"]["batch_size"]
            test_n_workers = cfg["test"]["n_workers"]

        if cfg["train"]["use_multi_gpu"] and dist.is_initialized():
            test_sampler = distributed_eval.SequentialDistributedSampler(
                test_dataset, batch_size=test_batch_size
            )
            test_loader_kwargs = dict(
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=True,
                sampler=test_sampler,
                num_workers=test_n_workers,
            )
            if test_n_workers > 0:
                test_loader_kwargs.update(
                    multiprocessing_context="fork",
                    persistent_workers=True,
                    prefetch_factor=test_prefetch_factor,
                )
            test_loader = DataLoader(test_dataset, **test_loader_kwargs)
        else:
            test_loader_kwargs = dict(
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=test_n_workers,
            )
            if test_n_workers > 0:
                test_loader_kwargs.update(
                    multiprocessing_context="fork",
                    persistent_workers=True,
                    prefetch_factor=test_prefetch_factor,
                )
            test_loader = DataLoader(test_dataset, **test_loader_kwargs)

    if cfg["train"]["use_multi_gpu"] and dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        batch_size = cfg["train"]["batch_size"] // world_size

        if cfg["train"]["n_workers"] == 0:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                pin_memory=True,
                num_workers=cfg["train"]["n_workers"],
                sampler=train_sampler,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                pin_memory=True,
                num_workers=cfg["train"]["n_workers"],
                sampler=train_sampler,
                worker_init_fn=worker_init_fn,
                multiprocessing_context="fork",
                persistent_workers=True,
                prefetch_factor=train_prefetch_factor,
                drop_last=True,
            )
    else:
        if cfg["train"]["n_workers"] == 0:
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg["train"]["batch_size"],
                shuffle=True,
                pin_memory=True,
                num_workers=cfg["train"]["n_workers"],
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg["train"]["batch_size"],
                shuffle=True,
                pin_memory=True,
                num_workers=cfg["train"]["n_workers"],
                worker_init_fn=worker_init_fn,
                multiprocessing_context="fork",
                persistent_workers=True,
                prefetch_factor=train_prefetch_factor,
            )

    return train_loader, val_loader, test_loader


def main(rank: int, world_size: int, args, resume_path, model_path, cfg):
    """Per-process entry: set up device/distributed, build data + model, run schemes."""
    torch.backends.cudnn.benchmark = True

    # Fix 21: Handle single GPU mode properly
    if world_size == 1:
        # Single GPU mode - no distributed setup needed
        logging.info("Running in single GPU mode")
        device = get_device(0)
    else:
        # Multi-GPU mode - setup distributed training with better error handling
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(cfg["port"])

        # Set NCCL timeout and retry settings
        os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes timeout
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

        try:
            # Initialize process group with timeout
            torch.distributed.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=1800),
            )
            torch.cuda.set_device(rank)
            device = torch.device(
                f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
            )
            logging.info(
                f"Successfully initialized distributed training on rank {rank}"
            )
        except Exception as e:
            logging.error(
                f"Failed to initialize distributed training on rank {rank}: {e}"
            )
            # Fallback to single GPU mode
            logging.info("Falling back to single GPU mode")
            torch.cuda.set_device(0)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            world_size = 1
            cfg["train"]["use_multi_gpu"] = False  # Disable multi-GPU for this run

    args.local_rank = rank
    args.world_size = world_size
    logging.info(f"args.local_rank: {rank}")
    logging.info(f"loaded to device {device}")
    logging.getLogger("matplotlib.font_manager").disabled = True
    if cfg["seeds"]["use_seeds"]:
        set_seeds(cfg["seeds"]["id"])

    # Fix 7: Add error handling for model creation
    try:
        model = configer.build_model(args, cfg)
        if model is None:
            raise ValueError("Model creation failed - returned None")
        logging.info(
            f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters"
        )
    except Exception as e:
        logging.error(f"Failed to create model: {e}")
        raise
    if cfg["train"]["use_multi_gpu"]:
        cfg["train"]["optimizer"]["learning_rate"] = cfg["train"]["optimizer"][
            "learning_rate"
        ]  # * dist.get_world_size()

    train_data_path = {}
    test_data_path = {}
    categories = ["car", "drone", "dog", "human"]

    logging.info(f"args.log setting: {args.log}")

    if cfg["schemes"]["train"]:

        output_dir = cfg["train"]["out_dir"]  # +"/"+args.exp_name
        create_output_dir(output_dir)
        # Copy configuration files safely
        try:
            shutil.copy2(args.yaml, os.path.join(output_dir, "default.yaml"))
            shutil.copy2(
                cfg["model"]["model_yaml"], os.path.join(output_dir, "model.yaml")
            )
            logging.info(f"Configuration files copied to {output_dir}")
        except FileNotFoundError as e:
            logging.error(f"Configuration file not found: {e}")
            raise
        except PermissionError as e:
            logging.error(f"Permission denied copying config files: {e}")
            raise
        except Exception as e:
            logging.error(f"Failed to copy config files: {e}")
            raise
        # Only initialize wandb on the main process (rank 0) to avoid multiple experiments
        if args.log and rank == 0:
            # Use experiment_name from config if available, otherwise fallback to yaml filename
            exp_name = cfg["train"].get("experiment_name")
            if not exp_name:
                exp_name = os.path.split(args.yaml)[-1].split(".")[0]
            
            wandb.init(
                project="Neural_Inertial_Tracking_" + cfg["data"]["dataset"],
                config=cfg["train"],
                name=exp_name,
            )
            wandb.config.update(cfg)
            logging.info(
                f"Wandb initialized on rank {rank} - Project: Neural_Inertial_Tracking_{cfg['data']['dataset']}"
            )
        elif args.log and rank != 0:
            # Disable wandb for non-main processes
            wandb.disabled = True
            logging.info(f"Wandb disabled on rank {rank} (only rank 0 logs)")
        else:
            wandb.disabled = True
            if rank == 0:
                warning("wandb is disabled")
        # Initialize data path variables
        train_data_path = {}
        valid_data_path = {}
        test_data_path = {}

        if cfg["data"]["dataset"] == "TarTanAir":
            train_data_path = GetTartanAirDataPath(
                cfg["data"]["data_path"], cfg["data"]["test_scene_list"]
            )
            valid_data_path = GetTartanAirDataPath_val(
                cfg["data"]["data_path"], cfg["data"]["val_scene_list"]
            )
            test_data_path = GetTartanAirDataPath_test(
                cfg["data"]["data_path"], cfg["data"]["test_scene_list"]
            )
        elif cfg["data"]["dataset"] == "AirLab":
            for category in categories:
                category_path = cfg["data"]["data_path"].get(category)
                if category_path is not None:
                    train_path, val_path, test_path = get_category_path(
                        category_path, cfg
                    )
                    train_data_path[category] = train_path
                    valid_data_path[category] = val_path
                    test_data_path[category] = test_path
        elif cfg["data"]["dataset"] in [
            "Humanoid",
            "HumanoidPostProcessed",
            "HumanoidPostProcessedCached",
        ]:
            for category in categories:
                category_path = cfg["data"]["data_path"].get(category)
                if category_path is not None:
                    train_path, val_path, test_path = get_category_path(
                        category_path, cfg
                    )
                    train_data_path[category] = train_path
                    valid_data_path[category] = val_path
                    test_data_path[category] = test_path

        elif cfg["data"]["dataset"] == "TRO":
            train_data_path = []
            valid_data_path = []
            test_data_path = []
            train_path = os.path.join(
                cfg["data"]["data_path"], cfg["data"]["train_dir"]
            )
            train_traj_path = [
                os.path.join(os.path.abspath(train_path), name)
                for name in (os.listdir(train_path + "/"))
            ]
            for traj_path in train_traj_path:
                for name in os.listdir(traj_path + "/"):
                    train_data_path.append(os.path.join(traj_path, name))
            val_path = os.path.join(
                cfg["data"]["data_path"], cfg["data"]["validation_dir"]
            )
            valid_traj_path = [
                os.path.join(os.path.abspath(val_path), name)
                for name in (os.listdir(val_path + "/"))
            ]
            for traj_path in valid_traj_path:
                for name in os.listdir(traj_path + "/"):
                    valid_data_path.append(os.path.join(traj_path, name))
            test_path = os.path.join(cfg["data"]["data_path"], cfg["data"]["test_dir"])
            test_traj_path = [
                os.path.join(os.path.abspath(test_path), name)
                for name in (os.listdir(test_path + "/"))
            ]
            for traj_path in test_traj_path:
                for name in os.listdir(traj_path + "/"):
                    test_data_path.append(os.path.join(traj_path, name))
        else:
            # Handle datasets with flat structure (like debug_dataset)
            train_data_path = GetDataPath(
                os.path.join(cfg["data"]["data_path"], cfg["data"]["train_dir"])
            )  # List, each element represents the path of each trajectory folder
            valid_data_path = GetDataPath(
                os.path.join(cfg["data"]["data_path"], cfg["data"]["validation_dir"])
            )
            test_data_path = GetDataPath(
                os.path.join(cfg["data"]["data_path"], cfg["data"]["test_dir"])
            )

            # Check if data_path is a directory with subfolders (debug_dataset structure)
            if os.path.isdir(cfg["data"]["data_path"]):
                for subfolder in os.listdir(cfg["data"]["data_path"]):
                    subfolder_path = os.path.join(cfg["data"]["data_path"], subfolder)
                    if os.path.isdir(subfolder_path):
                        # Check if this subfolder has train/val/test directories
                        train_subfolder = os.path.join(
                            subfolder_path, cfg["data"]["train_dir"]
                        )
                        val_subfolder = os.path.join(
                            subfolder_path, cfg["data"]["validation_dir"]
                        )
                        test_subfolder = os.path.join(
                            subfolder_path, cfg["data"]["test_dir"]
                        )

                        if os.path.exists(train_subfolder):
                            train_data_path.append(train_subfolder)
                        if os.path.exists(val_subfolder):
                            valid_data_path.append(val_subfolder)
                        if os.path.exists(test_subfolder):
                            test_data_path.append(test_subfolder)

        # load data Load training set, validation set, and test set
        train_loader, val_loader, test_loader = train_load_data(
            cfg,
            args,
            {
                "train": train_data_path,
                "val": valid_data_path,
                "test": test_data_path,
            },
            world_size=world_size,
        )

        if cfg["train"]["use_multi_gpu"] and dist.is_initialized():
            dist.barrier()

        # Training mode
        if cfg["schemes"]["train"]:
            trainer = configer.build_trainer(
                args, cfg, model, pretrained_path=model_path, resume_path=resume_path
            )

            if cfg["schemes"]["online_adaption"]:
                # Online adaptation mode
                logging.info("Starting online adaptation")
                # Create a proper tester for online adaptation
                tester = configer.build_tester(args, cfg, model, resume_path)
                trainer.online_adaptation(
                    train_loader=train_loader,
                    tester=tester,  # Use proper tester object
                    resume_model=model,
                    test_path_list=test_data_path,  # Use test_data_path from training data loading
                    ate_thres=0.1,  # ATE threshold for online adaptation
                )
            else:
                # Standard training mode
                logging.info("Starting standard training")
                trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                )

    # Testing mode (separate from training)
    if cfg["schemes"]["test"]:
        logging.info("Starting testing mode")
        train_category_path, val_category_path, test_category_path = {}, {}, {}
        train_path_list, val_path_list, test_path_list = {}, {}, {}
        test_data_path = []

        if cfg["data"]["data_path"] is None:
            raise ValueError("data_path must be specified.")

        if cfg["data"]["dataset"] == "TarTanAir":
            test_data_path = GetTartanAirDataPath_test(
                cfg["data"]["data_path"], cfg["data"]["test_scene_list"]
            )
        elif cfg["data"]["dataset"] == "TRO":
            test_path = os.path.join(cfg["data"]["data_path"], cfg["data"]["test_dir"])
            test_traj_path = [
                os.path.join(os.path.abspath(test_path), name)
                for name in (os.listdir(test_path + "/"))
            ]
            for traj_path in test_traj_path:
                for name in os.listdir(traj_path + "/"):
                    test_data_path.append(os.path.join(traj_path, name))
        elif cfg["data"]["dataset"] == "AirLab":
            for category in categories:
                category_path = cfg["data"]["data_path"].get(category)
                if category_path is not None:
                    train_path, val_path, test_path = get_category_path(
                        category_path, cfg
                    )
                    train_category_path[category] = train_path
                    val_category_path[category] = val_path
                    test_category_path[category] = test_path
                    train_path_list[category] = get_list_of_dir(train_path)
                    val_path_list[category] = get_list_of_dir(val_path)
                    test_path_list[category] = get_list_of_dir(test_path)
        elif cfg["data"]["dataset"] in [
            "Humanoid",
            "HumanoidPostProcessed",
            "HumanoidPostProcessedCached",
        ]:
            for category in categories:
                category_path = cfg["data"]["data_path"].get(category)
                if category_path is not None:
                    train_path, val_path, test_path = get_category_path(
                        category_path, cfg
                    )
                    train_category_path[category] = train_path
                    val_category_path[category] = val_path
                    test_category_path[category] = test_path
                    train_path_list[category] = get_list_of_dir(train_path)
                    val_path_list[category] = get_list_of_dir(val_path)
                    test_path_list[category] = get_list_of_dir(test_path)
        else:
            test_data_path = GetDataPath(
                os.path.join(cfg["data"]["data_path"], cfg["data"]["test_dir"])
            )

        if cfg["test"]["out_dir"] is None:
            raise ValueError("out_dir must be specified.")
        if rank == 0:
            if os.path.exists(cfg["test"]["out_dir"]) is False:
                os.mkdir(cfg["test"]["out_dir"])

        # Run testing
        tester = configer.build_tester(args, cfg, model, resume_path)
        tester.test(test_path_list, 1000, model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--yaml", type=str, default="./config/default.yaml")
    parser.add_argument(
        "--config", type=str, help="Path to configuration YAML file (alias for --yaml)"
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--resume_from", type=str, default=""
    )  
    parser.add_argument(
        "--log",
        action="store_true",
        default=True,
        help="if true, save the meta data to wandb",
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="if true, use pdb and switch to single process mode",
    )
    parser.add_argument(
        "--exp_name",
        default=False,
        help="please specifiy the experiment name for wandb",
    )

    args = parser.parse_args()
    info(f"Arguments: {args}")

    # Handle --config as alias for --yaml
    if args.config:
        args.yaml = args.config

    cfg = configer.load_config(args.yaml)

    # Fix 14: Add configuration validation
    try:
        from tartan_imu.utils.config_validator import (
            validate_config,
            validate_data_paths,
            validate_split_dirs,
            validate_model_dims,
        )

        if not validate_config(cfg):
            raise ValueError("Configuration validation failed")
        if not validate_data_paths(cfg):
            raise ValueError("Data path validation failed")
        # Startup guardrails (Unit 3b): fail fast on missing split dirs or a
        # stage/dim mismatch instead of crashing deep in data loading / forward.
        validate_split_dirs(cfg)
        validate_model_dims(cfg)
    except ImportError:
        logging.warning("Config validator not available, skipping validation")
    except Exception as e:
        logging.error(f"Configuration validation error: {e}")
        raise
    resume_path = args.resume_from
    model_path = args.checkpoint
    if resume_path and model_path:
        parser.error("--checkpoint and --resume_from are mutually exclusive")
    gpu_num = torch.cuda.device_count()

    # Single GPU mode support
    if not cfg["train"]["use_multi_gpu"]:
        rich_logger.gpu_info(1, "Single-GPU")
        logging.info("jump into single GPU mode")
        main(0, 1, args, resume_path, model_path, cfg)
    elif args.pdb:
        logging.info("jump into the pdb mode")
        args.log = False  # Turn off wandb logging when using pdb
        main(args.local_rank, gpu_num, args, resume_path, model_path, cfg)
    else:
        rich_logger.gpu_info(gpu_num, "Multi-GPU")
        logging.info("jump into multi training mode")

        # NCCL environment variables are already set in the main function

        # Use torch.distributed.launch instead of mp.spawn for better stability
        try:
            mp.spawn(
                main,
                args=(gpu_num, args, resume_path, model_path, cfg),
                nprocs=gpu_num,
                join=True,
            )
        except Exception as e:
            logging.error(f"Multi-GPU spawn failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            logging.info("Falling back to single GPU mode")
            main(0, 1, args, resume_path, model_path, cfg)
