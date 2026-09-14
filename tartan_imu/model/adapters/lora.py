"""LoRA utilities for TartanIMU.

The initial implementation adapts only the hidden Linear layers of the
dog velocity head. The shared ResNet/LSTM trunk and final output layer
remain frozen.

Default target modules:

    heads.dog.output_block1.fcs.0   # 256 -> 256
    heads.dog.output_block1.fcs.3   # 256 -> 256

The final 256 -> 3 projection is intentionally not LoRA-adapted.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import loralib as lora_lib


logger = logging.getLogger(__name__)


DEFAULT_TARGET_MODULES = (
    "heads.dog.output_block1.fcs.0",
    "heads.dog.output_block1.fcs.3",
)


def _make_lora_linear(
    module: nn.Linear,
    r: int,
    alpha: float,
    dropout: float,
) -> lora_lib.Linear:
    """Create a LoRA Linear layer from an existing nn.Linear."""

    new_layer = lora_lib.Linear(
        in_features=module.in_features,
        out_features=module.out_features,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=module.bias is not None,
        merge_weights=False,
    )

    # Move the new layer to the same device/dtype as the original layer.
    new_layer = new_layer.to(
        device=module.weight.device,
        dtype=module.weight.dtype,
    )

    # Preserve pretrained weights exactly.
    with torch.no_grad():
        new_layer.weight.copy_(module.weight)

        if module.bias is not None:
            new_layer.bias.copy_(module.bias)

    # Base weight is frozen. Only LoRA parameters are trained.
    new_layer.weight.requires_grad = False

    if new_layer.bias is not None:
        new_layer.bias.requires_grad = False

    # Preserve train/eval state.
    new_layer.train(module.training)

    return new_layer


def _replace_recursive(
    module: nn.Module,
    target_modules: set[str],
    r: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
    replaced: list[str] | None = None,
) -> list[str]:
    """Recursively replace selected nn.Linear layers with LoRA layers."""

    if replaced is None:
        replaced = []

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if full_name in target_modules:
            if not isinstance(child, nn.Linear):
                raise TypeError(
                    f"LoRA target '{full_name}' is "
                    f"{type(child).__name__}, expected nn.Linear."
                )

            new_layer = _make_lora_linear(
                module=child,
                r=r,
                alpha=alpha,
                dropout=dropout,
            )

            setattr(module, child_name, new_layer)

            replaced.append(full_name)

            logger.info(
                "Applied LoRA to %s: %d -> %d, r=%d, alpha=%s",
                full_name,
                child.in_features,
                child.out_features,
                r,
                alpha,
            )

            continue

        _replace_recursive(
            child,
            target_modules=target_modules,
            r=r,
            alpha=alpha,
            dropout=dropout,
            prefix=full_name,
            replaced=replaced,
        )

    return replaced


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze every parameter except LoRA A/B matrices."""

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def replace_layers(
    model: nn.Module,
    r: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: tuple[str, ...] | list[str] | None = None,
) -> nn.Module:
    """Apply LoRA to selected TartanIMU Linear layers.

    By default, only the first two Linear layers of the dog velocity
    prediction head are adapted.

    Args:
        model:
            TartanIMU FoundationModel.
        r:
            LoRA rank.
        alpha:
            LoRA scaling parameter.
        dropout:
            Dropout used inside the LoRA branch.
        target_modules:
            Exact module names to replace. If None, uses
            DEFAULT_TARGET_MODULES.

    Returns:
        The modified model.
    """

    if r <= 0:
        raise ValueError(f"LoRA rank must be positive, got r={r}")

    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    target_modules = set(target_modules)

    replaced = _replace_recursive(
        model,
        target_modules=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )

    missing = target_modules - set(replaced)

    if missing:
        available_linear = [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]

        raise RuntimeError(
            "Some LoRA target modules were not found.\n"
            f"Missing: {sorted(missing)}\n"
            f"Available Linear modules: {available_linear}"
        )

    # Important:
    # freeze the entire pretrained network, including the trunk and
    # non-LoRA parameters in the dog head.
    mark_only_lora_as_trainable(model)

    logger.info(
        "LoRA setup complete: replaced=%s",
        replaced,
    )

    return model


def get_trainable_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return only trainable LoRA parameters."""

    return [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" in name
    ]


def print_lora_parameters(model: nn.Module) -> None:
    """Print LoRA trainable parameter statistics."""

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logger.info(
        "LoRA parameters: trainable=%d / total=%d (%.4f%%)",
        trainable_params,
        total_params,
        100.0 * trainable_params / total_params,
    )

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                "LoRA trainable: %-60s %s",
                name,
                tuple(param.shape),
            )