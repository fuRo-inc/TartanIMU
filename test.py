# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Trajectory testing/evaluation entry point.

Runs model inference over test trajectories, integrates predicted velocities
into positions (via ``evaluation.postprocess``), applies per-segment drift
correction, computes ATE/RTE and accuracy metrics, and writes plots, CSVs, and
JSON metric files. The ``tester`` class is the top-level driver.
"""

import json
import logging

import numpy as np
import torch
from tartan_imu.evaluation import postprocess
from tartan_imu.model.common import function
from torch.utils.data import DataLoader
from tqdm import tqdm
from tartan_imu.utils import logging_config
from tartan_imu.utils.device import sync_if_cuda
from tartan_imu.utils.error_handling import is_empty_trajectory_error
from tartan_imu.utils.registry import load_dataset_module as _load_dataset_module
from tartan_imu.utils.rich_logging import info

# Plotting/reporting helpers were migrated to tartan_imu.evaluation.plots.
# Re-exported here so existing call sites (internal + external, e.g.
# example/minimal_example.py's ``test.display_rich_metrics_tables``) keep working.
from tartan_imu.evaluation.plots import (  # noqa: E402,F401
    create_3d_segments_summary,
    create_detailed_segment_plots,
    create_segment_plot,
    create_segments_summary_plot,
    create_segments_summary_statistics,
    display_rich_metrics_tables,
)

# Segmentation / metric / IO helpers were migrated to tartan_imu.evaluation.*.
# Re-exported here so existing call sites (internal + external, e.g.
# example/minimal_example.py's ``test.compute_overall_statistics``) keep working.
from tartan_imu.evaluation.segments import (  # noqa: E402,F401
    compute_segment_metrics,
    process_trajectory_segments,
    sanity_check_segment_alignment,
    segment_trajectory_5m,
    segment_trajectory_by_distance_gt,
)
from tartan_imu.evaluation.traj_metrics import (  # noqa: E402,F401
    compute_aggregated_metrics,
    compute_full_trajectory_metrics,
    compute_overall_statistics,
)
from tartan_imu.evaluation.io import (  # noqa: E402,F401
    create_output_directories,
    save_comprehensive_csv_files,
    save_full_trajectory_data,
    save_metrics_files,
    save_segment_data,
)

console = logging_config.Console()


def torch_to_numpy(torch_arr):
    """Detach a tensor and move it to CPU as a numpy array."""
    return torch_arr.cpu().detach().numpy()


def load_dataset_module(dataset_name):
    """Load the dataset reader registered under ``dataset_name``.

    Thin wrapper over :func:`tartan_imu.utils.registry.load_dataset_module` so
    the reader table lives in exactly one place and `main_net.py` and `test.py`
    can never drift apart.

    Args:
        dataset_name: Value of ``data.dataset`` in the experiment YAML.

    Returns:
        The imported reader module.
    """
    dataset_utils = _load_dataset_module(dataset_name)
    info(f"Current Dataset is {dataset_name}", style="cyan")
    return dataset_utils


def create_test_dataloader(cfg, data_path):
    """
    Create test dataloader for a given data path.

    Args:
        cfg: Configuration dictionary
        data_path: Path to test data

    Returns:
        test_loader: DataLoader for testing
        test_dataset: Dataset object
    """
    dataset_utils = load_dataset_module(cfg["data"]["dataset"])

    test_basic_data = dataset_utils.BasicSequenceData(cfg, [data_path], mode="test")
    test_dataset = dataset_utils.SeqToSeqDataset(
        cfg, test_basic_data, test_basic_data.get_merged_index_map(), mode="test"
    )

    # Use train batch_size as fallback if test batch_size is not configured
    if "test" not in cfg or "batch_size" not in cfg["test"]:
        # CRITICAL: For KV-cache streaming, we must process one window at a time 
        # to ensure temporal continuity in the cache.
        if cfg.get("model_param", {}).get("use_kv_cache", False):
            test_batch_size = 1
            logging.info("KV-cache enabled: setting test batch_size to 1 for streaming")
        else:
            test_batch_size = cfg["train"]["batch_size"]
    else:
        test_batch_size = cfg["test"]["batch_size"]

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return test_loader, test_dataset


class tester(object):
    """Drives trajectory inference and metric/plot generation for a model.

    Holds the config, device, and model, then runs per-trajectory inference,
    velocity integration, drift correction, and metric/plot output via
    ``test`` / ``process_single_trajectory``.
    """

    def __init__(self, args, cfg, model):
        super(tester, self).__init__()
        self.cfg = cfg
        self.device = torch.device(
            f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.out_dir = cfg["test"]["out_dir"]

        self.pred_velocity = cfg["model"]["pred_velocity"]
        self.window_time = cfg["model_param"]["window_time"]
        self.start_cov_epochs = cfg["train"]["start_cov_epochs"]
        self.plot_cnt = 0
        self._processed_trajectories = (
            []
        )  # Track processed trajectories for ID assignment

    @torch.no_grad()
    def inference_step(self, data_loader, epoch, resume_model=None):
        """Run forward inference over a trajectory's data loader.

        Iterates the loader (optionally maintaining a KV-cache across windows),
        collects per-window predictions, covariances, targets, orientations, and
        losses, and returns them concatenated in an attribute dict.
        """
        (
            targets_all,
            preds_all,
            preds_cov_all,
            losses_all,
            orien_all,
        ) = ([], [], [], [], [])

        # Inference must not build an autograd graph: without this the per-batch
        # forward activations are retained for a backward that never happens,
        # accumulating across batches and OOMing on memory-heavy stages (e.g.
        # Stage 3's [B*T, 29, 2] spatial encoder). eval() also disables dropout
        # and freezes BatchNorm running stats so predictions are deterministic.
        if resume_model is not None:
            resume_model.eval()

        # Initialize KV-cache for this trajectory
        past_kv = None
        time_offset = 0
        use_kv_cache = self.cfg.get("model_param", {}).get("use_kv_cache", False)

        for bid, batch in tqdm(enumerate(data_loader)):
            batch = [t.to(self.device) for t in batch]
            sync_if_cuda()

            # Forward pass with KV-cache support
            pred, pred_cov, targ, ori, loss, present_kv = function.fun_test_forward(
                self.cfg, resume_model, batch, self.start_cov_epochs, epoch, 
                past_kv=past_kv, time_offset=time_offset
            )

            sync_if_cuda()

            # Update cache and offset for next step
            if use_kv_cache:
                past_kv = present_kv
                # time_offset should be the total number of temporal tokens processed so far.
                if len(batch[0].shape) == 4:
                    num_tokens = batch[0].shape[1] * batch[0].shape[3]
                else:
                    num_tokens = batch[0].shape[1]
                
                time_offset += num_tokens

                # OPTIONAL: Truncate cache to keep only the last N tokens (e.g., 5 seconds of history)
                # This prevents memory from growing indefinitely during very long runs.
                # imu_freq * 5s = 200 * 5 = 1000 tokens
                max_cache_len = self.cfg.get("model_param", {}).get("max_cache_len", 2000)
                if past_kv[0][0].size(2) > max_cache_len:
                    new_past_kv = []
                    for layer_kv in past_kv:
                        k, v = layer_kv
                        new_past_kv.append((k[:, :, -max_cache_len:, :], v[:, :, -max_cache_len:, :]))
                    past_kv = new_past_kv
            
            targets_all.append(torch_to_numpy(targ))
            orien_all.append(torch_to_numpy(ori))
            preds_all.append(torch_to_numpy(pred))
            preds_cov_all.append(torch_to_numpy(pred_cov))
            losses_all.append(np.mean(torch_to_numpy(loss)))
            
        targets_all = np.concatenate(targets_all, axis=0)
        orien_all = np.concatenate(orien_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)
        preds_cov_all = np.concatenate(preds_cov_all, axis=0)
        
        attr_dict = {
            "targets": targets_all,
            "orien": orien_all,
            "preds": preds_all,
            "preds_cov": preds_cov_all,
            "losses": losses_all,
        }

        return attr_dict

    def process_single_trajectory(
        self, data, key, epoch_num, resume_model, segment_length=5.0
    ):
        """
        Process a single trajectory with drift correction every 5 meters.
        Uses post-inference segmentation with drift correction at segment boundaries.

        Args:
            data: Data path
            key: Motion modality key
            epoch_num: Epoch number
            resume_model: Model to test
            segment_length: Length of segments in meters

        Returns:
            Dictionary containing trajectory metrics and segment metrics
        """
        logging.info(f"Processing Path: {data}, Motion Modality: {key}")

        # Create dataloader for the full trajectory
        test_loader, test_dataset = create_test_dataloader(self.cfg, data)

        # Setup output directories
        subfolder_name = key + data.split("/")[-2]
        data_name = data.split("/")[-1]
        outdir = create_output_directories(self.out_dir, subfolder_name, data_name)

        # DRIFT CORRECTION APPROACH: Post-inference segmentation with drift correction

        # Run inference on full trajectory first
        sync_if_cuda()
        net_attr_dict = self.inference_step(test_loader, epoch_num, resume_model)
        sync_if_cuda()

        # Generate full trajectory
        traj_attr_dict = postprocess.pose_integrate(
            self.cfg, test_dataset, net_attr_dict, self.cfg["data"]["use_local_coord"]
        )

        print("\n=== AFTER pose_integrate ===")
        print("traj_attr_dict keys:", list(traj_attr_dict.keys()))
        print("cov_pred exists:", "cov_pred" in traj_attr_dict)

        if "cov_pred" in traj_attr_dict:
            print("cov_pred type :", type(traj_attr_dict["cov_pred"]))
            print("cov_pred shape:", traj_attr_dict["cov_pred"].shape)

        # Segment trajectory after inference
        trajectory_segments = segment_trajectory_5m(traj_attr_dict, segment_length)

        print("\n=== AFTER segment_trajectory_5m ===")
        print("num segments:", len(trajectory_segments))

        for i, segment in enumerate(trajectory_segments):
            print(f"segment[{i}] keys:", list(segment.keys()))
            print(f"segment[{i}] cov_pred exists:", "cov_pred" in segment)
            if "cov_pred" in segment:
                print(f"segment[{i}] cov_pred shape:", segment["cov_pred"].shape)

        # Apply drift correction to each segment
        corrected_segments = []
        for seg_idx, segment in enumerate(trajectory_segments):
            # Get ground truth initial position for this segment
            gt_initial_pos = segment["pos_gt"][0]
            pred_initial_pos = segment["pos_pred"][0]

            # Apply drift correction: align to ground truth initial position
            pos_pred_corrected = segment["pos_pred"] - pred_initial_pos + gt_initial_pos

            # Create corrected segment
            corrected_segment = segment.copy()
            corrected_segment["pos_pred"] = pos_pred_corrected

            print("\n=== CORRECTED SEGMENT ===")
            print("corrected_segment keys:", list(corrected_segment.keys()))
            print("cov_pred exists:", "cov_pred" in corrected_segment)
            corrected_segments.append(corrected_segment)

        # Apply drift correction to the FULL trajectory for visualization
        full_gt_initial = traj_attr_dict["pos_gt"][0]
        full_pred_initial = traj_attr_dict["pos_pred"][0]

        # Apply drift correction to full trajectory: align to ground truth initial position
        traj_attr_dict_corrected = traj_attr_dict.copy()
        traj_attr_dict_corrected["pos_pred"] = (
            traj_attr_dict["pos_pred"] - full_pred_initial + full_gt_initial
        )

        # Process corrected segments with trajectory information
        trajectory_info = {
            "data_name": data_name,
            "robot_type": key,
            "trajectory_id": len(self._processed_trajectories) + 1,
        }
        segment_metrics = process_trajectory_segments(
            corrected_segments, outdir, self.cfg, trajectory_info
        )

        # Compute aggregated metrics
        aggregated_metrics = compute_aggregated_metrics(segment_metrics)

        # Save full trajectory data (using corrected trajectory)
        plot_dict = postprocess.compute_plot_dict(
            self.cfg, net_attr_dict, traj_attr_dict_corrected
        )
        save_full_trajectory_data(
            traj_attr_dict_corrected, outdir, epoch_num, plot_dict
        )

        # Compute full trajectory metrics (using corrected trajectory)
        full_metrics = compute_full_trajectory_metrics(
            traj_attr_dict_corrected, self.cfg
        )

        # Prepare results
        trajectory_results = {
            "data": data_name,
            "num_segments": len(segment_metrics),
            "robot_type": key,
            "processing_path": data,
            "processing_method": "post_inference_with_drift_correction",
            "segment_metrics": segment_metrics,
            "full_trajectory": full_metrics,
            **aggregated_metrics,
        }

        # Save metrics
        save_metrics_files(trajectory_results, outdir, data_name, segment_metrics)

        # Generate plots
        # Pass the computed metrics instead of empty lists
        ave_ate_list = [full_metrics["ate"]] if "ate" in full_metrics else []
        t_rte_list = [full_metrics["t_rte"]] if "t_rte" in full_metrics else []
        d_rte_list = [full_metrics["d_rte"]] if "d_rte" in full_metrics else []

        postprocess.make_plots(
            plot_dict,
            outdir,
            epoch_num,
            ave_ate_list,
            t_rte_list,
            d_rte_list,
            use_local=self.cfg["data"]["use_local_coord"],
            full_metrics=full_metrics,
        )

        return trajectory_results, aggregated_metrics

    def test(
        self,
        test_data_path_list,
        epoch_num,
        resume_model,
        ratio=None,
        segment_length=20.0,
    ):
        """Evaluate the model over all test trajectories.

        Processes every trajectory in ``test_data_path_list``, aggregates
        per-trajectory and overall statistics, writes metric JSON/CSV files and
        summary plots, and returns the overall metrics dict.
        """
        all_trajectory_results = []
        segment_metrics_all = []
        all_metrics = {}

        logging_config.print_columns(test_data_path_list)

        # Process each trajectory
        for key in test_data_path_list:
            for data in test_data_path_list[key]:
                try:
                    trajectory_results, aggregated_metrics = (
                        self.process_single_trajectory(
                            data, key, epoch_num, resume_model, segment_length
                        )
                    )
                except ValueError as exc:
                    if not is_empty_trajectory_error(exc):
                        raise
                    logging.warning(
                        "Skipping trajectory with no valid windows: %s (platform=%s)",
                        data,
                        key,
                    )
                    continue

                all_trajectory_results.append(trajectory_results)
                segment_metrics_all.extend(trajectory_results["segment_metrics"])
                self._processed_trajectories.append(
                    trajectory_results
                )  # Track processed trajectories

                # Log trajectory statistics
                console.log("Traj Testing Statistics:", style="bold")
                test_statistics = {
                    "num_segments": trajectory_results["num_segments"],
                    **aggregated_metrics,
                }
                logging_config.console_log(test_statistics)

                # Append to main metrics file
                with open(self.out_dir + "/metrics.json", "a") as f:
                    json.dump({"cur_traj_data": trajectory_results}, f, indent=1)

        # Compute overall statistics
        overall_stats = compute_overall_statistics(
            all_trajectory_results, segment_metrics_all
        )
        all_metrics["all_traj"] = overall_stats

        # Display comprehensive rich tables
        from tartan_imu.utils.rich_logging import banner

        banner("📊 COMPREHENSIVE TEST RESULTS")
        display_rich_metrics_tables(
            all_trajectory_results, overall_stats, segment_metrics_all
        )

        # Save comprehensive CSV files
        save_comprehensive_csv_files(
            all_trajectory_results, overall_stats, segment_metrics_all, self.out_dir
        )

        # Create summary plot of all segments
        create_segments_summary_plot(all_trajectory_results, self.out_dir)

        return all_metrics
