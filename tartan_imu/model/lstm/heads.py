# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Per-motion-type output heads and the multi-head FoundationModel.

``OutputHead`` maps shared trunk features to a per-motion-type velocity (and
optional covariance) prediction. ``FoundationModel`` runs a trunk once and
dispatches its features to the relevant heads (all heads, or only those present
in the batch's motion types).
"""
import torch
import torch.nn as nn

from tartan_imu.model.common.blocks import FcBlock
from tartan_imu.model.common.velocity_covariance import inverse_softplus


class OutputHead(nn.Module):
    """
    General purpose output head for a neural network model.

    Attributes:
        lstm_size (int): Size of the LSTM layer.
        output_dim (int): Dimension of the output.
        drop_ratio (float): Dropout ratio.
        compute_type (str): Type of computation to perform.
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
    """

    def __init__(self, cfg, motion_type: str):
        super(OutputHead, self).__init__()

        self.lstm_size = cfg["model_param"]["lstm_size"]  # 256
        self.lstm_dropout = cfg["model_param"]["lstm_dropout"]  # 0.0
        self.output_dim = cfg["model_param"]["output_dim"]  # 3
        self.drop_ratio = cfg["model_param"]["drop_ratio"]  # 0.5 original:0.2
        self.split_z = cfg["model_param"]["split_z"]  # True/False
        # Output module
        self.output_block1 = FcBlock(
            self.lstm_size, self.output_dim, dropout=self.drop_ratio, cfg=cfg
        )
        self.output_block2 = FcBlock(
            self.lstm_size, self.output_dim, dropout=self.drop_ratio, cfg=cfg
        )
        covariance_cfg = cfg.get("model", {}).get("velocity_covariance", {})
        self.use_velocity_covariance = covariance_cfg.get("enabled", False)
        if self.use_velocity_covariance:
            if covariance_cfg.get("parameterization", "softplus_cholesky") != "softplus_cholesky":
                raise ValueError("Only softplus_cholesky velocity covariance is supported")
            if covariance_cfg.get("raw_dim", 6) != 6:
                raise ValueError("Softplus-Cholesky velocity covariance requires raw_dim: 6")
            # Kept separate from output_block2 so legacy 3-D covariance
            # checkpoints and their state_dict keys remain unchanged.
            self.velocity_covariance_head = FcBlock(
                self.lstm_size, 6, dropout=self.drop_ratio, cfg=cfg
            )
            self._initialize_velocity_covariance_head(covariance_cfg)
        if self.split_z:
            self.output_block1 = FcBlock(
                self.lstm_size, self.output_dim - 1, dropout=self.drop_ratio, cfg=cfg
            )  # dp mean
            self.output_block2 = FcBlock(
                self.lstm_size, self.output_dim, dropout=self.drop_ratio, cfg=cfg
            )  # dp cov
            self.output_block1_z = FcBlock(
                self.lstm_size, 1, dropout=self.drop_ratio, cfg=cfg
            )  # dp cov
        self.motion_type = motion_type

        # Add learnable scale factor for velocity prediction
        self.velocity_scale = nn.Parameter(torch.ones(1, self.output_dim))
        self.use_velocity_scale = cfg.get("model", {}).get("pred_velocity", False)

    def _initialize_velocity_covariance_head(self, covariance_cfg):
        """Initialize Sigma approximately diag(initial_std**2), off-diagonal zero."""
        eps = float(covariance_cfg.get("eps", 1.0e-4))
        initial_std = float(covariance_cfg.get("initial_std", 0.1))
        raw_diagonal_bias = inverse_softplus(initial_std - eps)
        final_layer = self.velocity_covariance_head.fcs[-1]
        nn.init.zeros_(final_layer.weight)
        with torch.no_grad():
            final_layer.bias.zero_()
            final_layer.bias[:3].fill_(raw_diagonal_bias)

    def forward(self, out, batch_size, seq_len, predict_cov=False):
        """
        Forward pass of the module.

        Args:
            out: Input tensor.

        Returns:
            Tensor: Output after forward pass.
        """
        if self.split_z:
            x = self.output_block1(out)  # mean  10*3 Linear layer converts 10*256->10*3
            z = self.output_block1_z(out)
            x1 = torch.cat((x, z), dim=1)
            x1 = x1.view(batch_size, seq_len, -1)
        else:
            x1 = self.output_block1(out)
            x1 = x1.view(batch_size, seq_len, -1)

        # Apply learnable scale factor if velocity prediction is enabled
        if self.use_velocity_scale:
            x1 = x1 * self.velocity_scale

        if predict_cov:
            # Raw order: [Lxx_raw, Lyy_raw, Lzz_raw, Lyx, Lzx, Lzy].
            x2 = (self.velocity_covariance_head if self.use_velocity_covariance else self.output_block2)(out)
            x2 = x2.view(batch_size, seq_len, -1)
            return x1, x2
        else:
            return x1


class FoundationModel(nn.Module):
    """Shared trunk + per-motion-type output heads.

    Runs ``trunk`` once and dispatches its features to the relevant ``heads``
    (all heads, or only those present in the batch's motion types).
    """

    def __init__(self, cfg, trunk: nn.Module, heads: nn.ModuleDict):
        super(FoundationModel, self).__init__()
        self.model = trunk
        self.heads = heads
        self.types = ["car", "dog", "drone", "human"]
        self.lstm_size = cfg["model_param"]["lstm_size"]  # 256
        self.lstm_dropout = cfg["model_param"]["lstm_dropout"]  # 0.0
        self.output_dim = cfg["model_param"]["output_dim"]  # 3
        self.drop_ratio = cfg["model_param"]["drop_ratio"]  # 0.5 original:0.2

    def forward(self, x, motion_type=None, predict_cov=False, compute_all_heads=True):
        """
        Forward pass with optional efficient computation.

        Args:
            x: Input tensor
            motion_type: Motion type tensor or None
            predict_cov: Whether to predict covariance
            compute_all_heads: If True, compute all heads. If False, compute only needed heads.
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Forward pass through shared backbone
        model_output = self.model(x)

        # Handle case where trunk model returns a tuple (output, hidden_states)
        if isinstance(model_output, tuple):
            model_output = model_output[0]  # Take the first element (output tensor)

        if compute_all_heads:
            # Original behavior - compute all heads
            return self._compute_all_heads(
                model_output, batch_size, seq_len, predict_cov
            )
        else:
            # Efficient behavior - compute only needed heads
            return self._compute_needed_heads(
                model_output, batch_size, seq_len, predict_cov, motion_type
            )

    def _compute_all_heads(self, model_output, batch_size, seq_len, predict_cov):
        """Original implementation - compute all heads."""
        multi_head = {}
        multi_head_cov = {}

        if predict_cov:
            car_head, car_cov = self.heads["car"](
                model_output, batch_size, seq_len, predict_cov
            )
            dog_head, dog_cov = self.heads["dog"](
                model_output, batch_size, seq_len, predict_cov
            )
            drone_head, drone_cov = self.heads["drone"](
                model_output, batch_size, seq_len, predict_cov
            )
            human_head, human_cov = self.heads["human"](
                model_output, batch_size, seq_len, predict_cov
            )

            multi_head["car"] = car_head
            multi_head["dog"] = dog_head
            multi_head["drone"] = drone_head
            multi_head["human"] = human_head

            multi_head_cov["car"] = car_cov
            multi_head_cov["dog"] = dog_cov
            multi_head_cov["drone"] = drone_cov
            multi_head_cov["human"] = human_cov

            return multi_head, multi_head_cov
        else:
            car_head = self.heads["car"](model_output, batch_size, seq_len)
            dog_head = self.heads["dog"](model_output, batch_size, seq_len)
            drone_head = self.heads["drone"](model_output, batch_size, seq_len)
            human_head = self.heads["human"](model_output, batch_size, seq_len)

            multi_head["car"] = car_head
            multi_head["dog"] = dog_head
            multi_head["drone"] = drone_head
            multi_head["human"] = human_head

            return multi_head

    def _compute_needed_heads(
        self, model_output, batch_size, seq_len, predict_cov, motion_type
    ):
        """Efficient computation - only compute heads that have data."""
        if motion_type is None:
            # Fallback to all heads if no motion type specified
            return self._compute_all_heads(
                model_output, batch_size, seq_len, predict_cov
            )

        # Determine which heads are needed based on motion type
        needed_heads = self._get_needed_heads(motion_type)

        multi_head = {}
        multi_head_cov = {}

        for motion_type_name in needed_heads:
            if predict_cov:
                pred, cov = self.heads[motion_type_name](
                    model_output, batch_size, seq_len, predict_cov
                )
                multi_head[motion_type_name] = pred
                multi_head_cov[motion_type_name] = cov
            else:
                pred = self.heads[motion_type_name](model_output, batch_size, seq_len)
                multi_head[motion_type_name] = pred

        if predict_cov:
            return multi_head, multi_head_cov
        else:
            return multi_head

    def _get_needed_heads(self, motion_type):
        """Determine which heads are needed based on motion type in batch."""
        if isinstance(motion_type, torch.Tensor):
            # Get unique motion types in batch
            unique_types = torch.unique(motion_type).tolist()
        else:
            unique_types = [motion_type]

        # Map motion type IDs to head names
        motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}
        needed_heads = []

        for motion_id in unique_types:
            if motion_id in motion_types:
                needed_heads.append(motion_types[motion_id])

        return needed_heads

    def get_num_params(self):
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
