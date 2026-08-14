#!/usr/bin/env python3
"""Shared distributed-training utilities for RGB diffusion-policy scripts.

The helpers in this module intentionally use explicit gradient synchronization
instead of ``DistributedDataParallel``. This matches the IDQL trainers, which
coordinate several independently optimized networks in one training step.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

import robomimic.utils.torch_utils as TorchUtils


@dataclass(frozen=True)
class DistributedContext:
    """Rank metadata and the device owned by the current process."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed(args: argparse.Namespace) -> DistributedContext:
    """Initialize one torchrun process per GPU, while preserving serial use."""
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launched_by_torchrun = all(
        name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    requested = bool(getattr(args, "distributed", False) or env_world_size > 1)
    if not requested:
        device = TorchUtils.get_torch_device(
            try_to_use_cuda=args.device == "cuda"
        )
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            backend="none",
            device=device,
        )
    if not launched_by_torchrun:
        raise RuntimeError(
            "distributed training must be launched with torchrun (or "
            "python -m torch.distributed.run)"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    cli_local_rank = getattr(args, "local_rank", None)
    if cli_local_rank is not None and int(cli_local_rank) != local_rank:
        raise RuntimeError(
            f"--local-rank={cli_local_rank} disagrees with "
            f"LOCAL_RANK={local_rank}"
        )
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("distributed CUDA training requested without CUDA")
        device_count = int(torch.cuda.device_count())
        if local_rank < 0 or local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside the {device_count} visible "
                "CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", local_rank)
        default_backend = "nccl"
    else:
        device = torch.device("cpu")
        default_backend = "gloo"

    requested_backend = str(getattr(args, "distributed_backend", "auto"))
    backend = default_backend if requested_backend == "auto" else requested_backend
    if backend == "nccl" and device.type != "cuda":
        raise ValueError("the NCCL distributed backend requires --device cuda")
    if backend != "nccl" and device.type == "cuda":
        raise ValueError("distributed CUDA training requires the NCCL backend")

    dist.init_process_group(backend=backend, init_method="env://")
    context = DistributedContext(
        enabled=True,
        rank=int(dist.get_rank()),
        local_rank=local_rank,
        world_size=int(dist.get_world_size()),
        backend=backend,
        device=device,
    )
    if context.world_size != env_world_size:
        raise RuntimeError(
            f"initialized world_size={context.world_size}, expected {env_world_size}"
        )
    return context


def modules_have_mutable_batch_norm(modules: Iterable[nn.Module]) -> bool:
    """Return whether any module updates BatchNorm running statistics."""
    return any(
        isinstance(layer, nn.modules.batchnorm._BatchNorm)
        and bool(layer.track_running_stats)
        for module in modules
        for layer in module.modules()
    )


def seed_process(seed: int, device: torch.device) -> None:
    """Seed CPU RNGs and only the CUDA generator owned by this process."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    # torch.manual_seed seeds every visible CUDA generator, which can make each
    # torchrun process touch devices owned by other local ranks.
    torch.random.default_generator.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed))


def capture_process_rng_state(device: torch.device) -> dict[str, Any]:
    """Capture process RNG state without querying CUDA devices from other ranks."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda_local"] = torch.cuda.get_rng_state(device).cpu()
    return state


def restore_process_rng_state(
    state: dict[str, Any] | None,
    device: torch.device,
) -> None:
    """Restore state produced by :func:`capture_process_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and "cuda_local" in state:
        torch.cuda.set_rng_state(state["cuda_local"].cpu(), device=device)


@torch.no_grad()
def broadcast_module_state(
    modules: Iterable[nn.Module],
    context: DistributedContext,
) -> None:
    """Broadcast parameters and buffers from rank zero in a fixed order."""
    if not context.enabled:
        return
    for module in modules:
        for parameter in module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in module.buffers():
            dist.broadcast(buffer.data, src=0)


@torch.no_grad()
def broadcast_module_buffers(
    modules: Iterable[nn.Module],
    context: DistributedContext,
) -> None:
    """Broadcast mutable module buffers from rank zero in a fixed order."""
    if not context.enabled:
        return
    for module in modules:
        for buffer in module.buffers():
            dist.broadcast(buffer.data, src=0)


@torch.no_grad()
def all_reduce_gradients(
    parameters: Iterable[torch.nn.Parameter],
    context: DistributedContext,
    bucket_cap_mb: float = 25.0,
    preserve_unused_parameters: bool = True,
    max_in_flight_buckets: int = 2,
) -> None:
    """Average dense gradients with fixed-order asynchronous buckets.

    ``bucket_cap_mb`` is a target cap; a single tensor larger than the cap is
    reduced in its own bucket. At most ``max_in_flight_buckets`` flattened
    buffers are retained, so temporary memory is bounded by that small window
    instead of the total size of all gradients.

    Every rank must pass parameters in the same order. Locally unused
    parameters contribute zeros, and parameters unused on every rank are reset
    to ``grad=None`` after synchronization.
    """
    if not context.enabled:
        return
    max_in_flight_buckets = int(max_in_flight_buckets)
    if max_in_flight_buckets <= 0:
        raise ValueError("max_in_flight_buckets must be positive")

    cap_bytes = max(1, int(float(bucket_cap_mb) * 1024 * 1024))
    trainable_parameters = [
        parameter for parameter in parameters if parameter.requires_grad
    ]
    if not trainable_parameters:
        return

    globally_used = None
    usage_work = None
    if preserve_unused_parameters:
        globally_used = torch.tensor(
            [parameter.grad is not None for parameter in trainable_parameters],
            dtype=torch.int32,
            device=context.device,
        )
        usage_work = dist.all_reduce(
            globally_used,
            op=dist.ReduceOp.MAX,
            async_op=True,
        )

    bucket: list[torch.Tensor] = []
    bucket_bytes = 0
    bucket_key = None
    pending = deque()
    scale = 1.0 / float(context.world_size)

    def finish_oldest() -> None:
        work, flat, gradients = pending.popleft()
        work.wait()
        flat.mul_(scale)
        offset = 0
        for gradient in gradients:
            count = int(gradient.numel())
            gradient.copy_(flat[offset : offset + count].view_as(gradient))
            offset += count

    def launch() -> None:
        nonlocal bucket, bucket_bytes, bucket_key
        if not bucket:
            return
        # Drain before allocating the next flattened buffer. This is important:
        # retaining all asynchronous buckets until the end defeats the memory
        # bound implied by bucket_cap_mb.
        if len(pending) >= max_in_flight_buckets:
            finish_oldest()
        gradients = tuple(bucket)
        flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
        work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        pending.append((work, flat, gradients))
        bucket = []
        bucket_bytes = 0
        bucket_key = None

    for parameter in trainable_parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
        gradient = parameter.grad
        if gradient.is_sparse:
            raise RuntimeError("distributed gradients must be dense")
        key = (gradient.device, gradient.dtype)
        gradient_bytes = int(gradient.numel() * gradient.element_size())
        if bucket and (
            key != bucket_key or bucket_bytes + gradient_bytes > cap_bytes
        ):
            launch()
        bucket_key = key
        bucket.append(gradient)
        bucket_bytes += gradient_bytes
    launch()

    while pending:
        finish_oldest()
    if usage_work is not None:
        usage_work.wait()
        globally_used_host = globally_used.cpu().tolist()
        for parameter, parameter_is_used in zip(
            trainable_parameters,
            globally_used_host,
        ):
            if not parameter_is_used:
                parameter.grad = None


def reduce_distributed_scalars(
    metrics: Mapping[str, Any],
    context: DistributedContext,
    *,
    reductions: Mapping[str, str] | None = None,
    default_reduction: str = "mean",
) -> dict[str, float]:
    """Reduce scalar metrics, with an optional operation selected per key.

    Supported operations are ``mean``, ``sum``, ``min``, and ``max``. All
    ranks must provide the same metric keys and reduction choices.
    """
    if not metrics:
        return {}
    reductions = {} if reductions is None else dict(reductions)
    unknown_keys = set(reductions).difference(metrics)
    if unknown_keys:
        raise KeyError(
            "reductions specified for unknown metrics: "
            + ", ".join(sorted(unknown_keys))
        )

    valid_reductions = {"mean", "sum", "min", "max"}
    default_reduction = str(default_reduction).lower()
    if default_reduction not in valid_reductions:
        raise ValueError(f"unsupported scalar reduction: {default_reduction!r}")

    keys = sorted(metrics)
    scalar_tensors = []
    key_reductions = []
    for key in keys:
        value = metrics[key]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"metric {key!r} is not scalar: {tuple(value.shape)}")
            scalar = value.detach().reshape(()).to(
                device=context.device,
                dtype=torch.float32,
                non_blocking=True,
            )
        else:
            scalar = torch.as_tensor(
                value,
                device=context.device,
                dtype=torch.float32,
            )
            if scalar.numel() != 1:
                raise ValueError(f"metric {key!r} is not scalar: {tuple(scalar.shape)}")
            scalar = scalar.reshape(())
        reduction = str(reductions.get(key, default_reduction)).lower()
        if reduction not in valid_reductions:
            raise ValueError(
                f"unsupported scalar reduction {reduction!r} for metric {key!r}"
            )
        scalar_tensors.append(scalar)
        key_reductions.append(reduction)

    values = torch.stack(scalar_tensors)
    if context.enabled:
        for reduction in ("mean", "sum", "min", "max"):
            indices = [
                index
                for index, key_reduction in enumerate(key_reductions)
                if key_reduction == reduction
            ]
            if not indices:
                continue
            index_tensor = torch.tensor(
                indices,
                dtype=torch.long,
                device=context.device,
            )
            reduced = values.index_select(0, index_tensor)
            if reduction in ("mean", "sum"):
                op = dist.ReduceOp.SUM
            elif reduction == "min":
                op = dist.ReduceOp.MIN
            else:
                op = dist.ReduceOp.MAX
            dist.all_reduce(reduced, op=op)
            if reduction == "mean":
                reduced.mul_(1.0 / float(context.world_size))
            values.index_copy_(0, index_tensor, reduced)

    host_values = values.cpu().tolist()
    return {key: float(value) for key, value in zip(keys, host_values)}


def mean_distributed_scalars(
    metrics: Mapping[str, Any],
    context: DistributedContext,
    reductions: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Reduce scalar metrics, using a mean unless overridden per key.

    ``reductions`` accepts ``mean``, ``min``, ``max``, or ``sum`` values. The
    optional argument keeps ordinary callers concise while allowing extrema
    such as action bounds to retain their correct distributed semantics.
    """
    return reduce_distributed_scalars(
        metrics,
        context,
        reductions=reductions,
    )


def gather_rank_runtime_states(
    loader_generator: torch.Generator,
    context: DistributedContext,
) -> list[dict[str, Any]]:
    """Gather each rank's process and data-loader RNG state on every rank."""
    local_state = {
        "rank": int(context.rank),
        "rng_state": capture_process_rng_state(context.device),
        "loader_generator_state": loader_generator.get_state(),
    }
    if not context.enabled:
        return [local_state]
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local_state)
    if any(state is None for state in gathered):
        raise RuntimeError("failed to gather all distributed RNG states")
    return [state for state in gathered if state is not None]


__all__ = [
    "DistributedContext",
    "all_reduce_gradients",
    "broadcast_module_buffers",
    "broadcast_module_state",
    "capture_process_rng_state",
    "gather_rank_runtime_states",
    "initialize_distributed",
    "mean_distributed_scalars",
    "modules_have_mutable_batch_norm",
    "reduce_distributed_scalars",
    "restore_process_rng_state",
    "seed_process",
]
