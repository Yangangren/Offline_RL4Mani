#!/usr/bin/env python3
"""Pure data helpers shared by the RGB chunk-RECAP training stages.

This module deliberately contains no command-line or robomimic dependencies.
It owns the reward / return convention used by the Monte-Carlo value model,
the terminal-safe chunk advantage calculation, binary RECAP labeling, and the
small amount of index plumbing needed for immutable label sidecars.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


SUCCESS_SOURCES = frozenset(
    (
        "expert",
        "human",
        "human_demo",
        "non_expert_success",
        "success_rollout",
    )
)
FAILURE_SOURCES = frozenset(
    (
        "non_expert_failure",
        "failure_rollout",
    )
)


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _validate_gamma(gamma: Any) -> float:
    result = _finite_scalar(gamma, "gamma")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {result}")
    return result


def _numpy_vector(values: Any, name: str, *, finite: bool = True) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    result = np.asarray(values)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1:
        raise ValueError(
            f"{name} must have shape [N] or [N,1], got {result.shape}"
        )
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def build_canonical_episode_targets(
    raw_rewards: Any,
    source: str,
    gamma: float,
    failure_penalty: float,
    return_scale: float,
    chunk_horizon: int,
) -> dict[str, np.ndarray]:
    """Build MC-value and chunk targets for one complete source episode.

    The canonical unscaled reward is ``-1`` before termination, ``0`` on a
    successful terminal transition, and ``-failure_penalty`` on a failed
    terminal transition. Rewards are divided by ``return_scale`` before any
    discounted sums are formed, so rewards, values, and advantages share one
    scale without an advantage-breaking additive normalization shift.

    Successful human / rollout episodes terminate at their first strictly
    positive raw task reward. Later rows are retained in the arrays for stable
    HDF5 index alignment but have ``value_valid=0`` and zero-valued targets.
    Failure episodes terminate on their final row and must not contain a
    positive raw task reward.

    Args:
        raw_rewards: Complete episode rewards, shaped ``[T]`` or ``[T,1]``.
        source: One of the human/success/failure source labels in
            :data:`SUCCESS_SOURCES` or :data:`FAILURE_SOURCES`.
        gamma: Discount in ``[0, 1]``.
        failure_penalty: Positive terminal failure cost before scaling.
        return_scale: Positive divisor applied to every canonical reward.
        chunk_horizon: Maximum number of actions in a chunk.

    Returns:
        A dict of float32 arrays, each shaped ``[T]``:

        - ``canonical_reward``: scaled per-transition canonical reward;
        - ``mc_return``: discounted return through the effective terminal;
        - ``value_valid``: 1 through the terminal and 0 after success;
        - ``chunk_return``: terminal-truncated ``chunk_horizon``-step return;
        - ``terminal``: whether that starting row's chunk reaches terminal;
        - ``valid_length``: number of valid actions represented by the chunk.
    """

    rewards = _numpy_vector(raw_rewards, "raw_rewards").astype(
        np.float64, copy=False
    )
    if rewards.size == 0:
        raise ValueError("raw_rewards must contain at least one transition")

    source = str(source)
    if source not in SUCCESS_SOURCES and source not in FAILURE_SOURCES:
        supported = sorted(SUCCESS_SOURCES | FAILURE_SOURCES)
        raise ValueError(
            f"unsupported episode source={source!r}; expected one of {supported}"
        )
    gamma = _validate_gamma(gamma)
    failure_penalty = _finite_scalar(failure_penalty, "failure_penalty")
    return_scale = _finite_scalar(return_scale, "return_scale")
    if failure_penalty <= 0.0:
        raise ValueError(
            f"failure_penalty must be positive, got {failure_penalty}"
        )
    if return_scale <= 0.0:
        raise ValueError(f"return_scale must be positive, got {return_scale}")
    if isinstance(chunk_horizon, (bool, np.bool_)):
        raise ValueError("chunk_horizon must be a positive integer")
    try:
        chunk_horizon = int(chunk_horizon)
    except (TypeError, ValueError) as error:
        raise ValueError("chunk_horizon must be a positive integer") from error
    if chunk_horizon <= 0:
        raise ValueError(
            f"chunk_horizon must be positive, got {chunk_horizon}"
        )

    positive_indices = np.flatnonzero(rewards > 0.0)
    successful = source in SUCCESS_SOURCES
    if successful:
        if positive_indices.size == 0:
            raise ValueError(
                f"successful source={source!r} has no positive raw reward"
            )
        terminal_index = int(positive_indices[0])
    else:
        if positive_indices.size:
            raise ValueError(
                f"failure source={source!r} contains a positive raw reward at "
                f"index {int(positive_indices[0])}"
            )
        terminal_index = int(rewards.size - 1)

    count = int(rewards.size)
    canonical_reward = np.zeros(count, dtype=np.float64)
    if terminal_index > 0:
        canonical_reward[:terminal_index] = -1.0
    canonical_reward[terminal_index] = 0.0 if successful else -failure_penalty
    canonical_reward /= return_scale

    value_valid = np.zeros(count, dtype=np.float64)
    value_valid[: terminal_index + 1] = 1.0

    mc_return = np.zeros(count, dtype=np.float64)
    running = 0.0
    for index in range(terminal_index, -1, -1):
        running = canonical_reward[index] + gamma * running
        mc_return[index] = running

    chunk_return = np.zeros(count, dtype=np.float64)
    terminal = np.zeros(count, dtype=np.float64)
    valid_length = np.zeros(count, dtype=np.float64)
    for start in range(terminal_index + 1):
        last = min(start + chunk_horizon - 1, terminal_index)
        length = last - start + 1
        discounts = np.power(
            gamma,
            np.arange(length, dtype=np.float64),
        )
        chunk_return[start] = np.dot(
            canonical_reward[start : last + 1],
            discounts,
        )
        valid_length[start] = float(length)
        terminal[start] = float(last == terminal_index)

    result = {
        "canonical_reward": canonical_reward.astype(np.float32),
        "mc_return": mc_return.astype(np.float32),
        "value_valid": value_valid.astype(np.float32),
        "chunk_return": chunk_return.astype(np.float32),
        "terminal": terminal.astype(np.float32),
        "valid_length": valid_length.astype(np.float32),
    }
    if any(array.shape != (count,) for array in result.values()):
        raise RuntimeError("canonical episode targets lost index alignment")
    if any(not np.isfinite(array).all() for array in result.values()):
        raise RuntimeError("canonical episode target construction was non-finite")
    return result


def _torch_column(
    values: Any,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    result = torch.as_tensor(values, device=device)
    if result.ndim == 1:
        result = result.unsqueeze(1)
    if result.ndim != 2 or result.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape [B] or [B,1], got {tuple(result.shape)}"
        )
    result = result.to(dtype=dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _numpy_column(values: Any, name: str) -> np.ndarray:
    return _numpy_vector(values, name).astype(np.float32, copy=False)[:, None]


def chunk_advantage(
    chunk_return: Any,
    terminal: Any,
    valid_length: Any,
    value: Any,
    next_value: Any,
    gamma: float,
) -> torch.Tensor | np.ndarray:
    """Calculate a terminal-safe semi-MDP advantage as a column vector.

    The result is

    ``R_t^L + (1 - terminal) * gamma**L * V(s_{t+L}) - V(s_t)``.

    Every input may be shaped ``[B]`` or ``[B,1]``; mixed shapes never
    broadcast across the batch. If any input is a torch tensor, the result is
    a torch tensor on the first tensor input's device. Otherwise a float32
    numpy array is returned. ``valid_length`` must contain positive integers.
    """

    gamma = _validate_gamma(gamma)
    tensor_reference = next(
        (
            item
            for item in (
                value,
                next_value,
                chunk_return,
                terminal,
                valid_length,
            )
            if isinstance(item, torch.Tensor)
        ),
        None,
    )
    if tensor_reference is not None:
        dtype = (
            tensor_reference.dtype
            if tensor_reference.is_floating_point()
            else torch.float32
        )
        device = tensor_reference.device
        columns = [
            _torch_column(item, name, device=device, dtype=dtype)
            for item, name in (
                (chunk_return, "chunk_return"),
                (terminal, "terminal"),
                (valid_length, "valid_length"),
                (value, "value"),
                (next_value, "next_value"),
            )
        ]
        returns, terminals, lengths, values, next_values = columns
        batch_sizes = {int(column.shape[0]) for column in columns}
        if len(batch_sizes) != 1:
            raise ValueError(
                "chunk advantage inputs have different batch sizes: "
                f"{[tuple(column.shape) for column in columns]}"
            )
        if not torch.allclose(
            terminals,
            terminals.round(),
            atol=1e-6,
            rtol=0.0,
        ) or bool(((terminals < 0.0) | (terminals > 1.0)).any()):
            raise ValueError("terminal must contain only 0 or 1")
        if bool((lengths <= 0.0).any()) or not torch.allclose(
            lengths,
            lengths.round(),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("valid_length must contain positive integers")
        bootstrap = torch.pow(
            lengths.new_tensor(gamma),
            lengths,
        )
        result = (
            returns
            + (1.0 - terminals) * bootstrap * next_values
            - values
        )
        if not torch.isfinite(result).all():
            raise ValueError("chunk advantage is non-finite")
        return result

    columns_np = [
        _numpy_column(item, name)
        for item, name in (
            (chunk_return, "chunk_return"),
            (terminal, "terminal"),
            (valid_length, "valid_length"),
            (value, "value"),
            (next_value, "next_value"),
        )
    ]
    returns_np, terminals_np, lengths_np, values_np, next_values_np = columns_np
    batch_sizes = {int(column.shape[0]) for column in columns_np}
    if len(batch_sizes) != 1:
        raise ValueError(
            "chunk advantage inputs have different batch sizes: "
            f"{[column.shape for column in columns_np]}"
        )
    if not np.allclose(terminals_np, np.round(terminals_np), atol=1e-6, rtol=0.0):
        raise ValueError("terminal must contain only 0 or 1")
    if np.any((terminals_np < 0.0) | (terminals_np > 1.0)):
        raise ValueError("terminal must contain only 0 or 1")
    if np.any(lengths_np <= 0.0) or not np.allclose(
        lengths_np,
        np.round(lengths_np),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("valid_length must contain positive integers")
    bootstrap_np = np.power(np.float32(gamma), lengths_np)
    result_np = (
        returns_np
        + (1.0 - terminals_np) * bootstrap_np * next_values_np
        - values_np
    ).astype(np.float32, copy=False)
    if not np.isfinite(result_np).all():
        raise ValueError("chunk advantage is non-finite")
    return result_np


def masked_chunk_advantage(
    chunk_return: Any,
    terminal: Any,
    valid_length: Any,
    value: Any,
    next_value: Any,
    eligible: Any,
    gamma: float,
) -> torch.Tensor | np.ndarray:
    """Compute chunk advantage only where a row represents a valid chunk.

    Post-success rows are retained for stable actor / HDF5 index alignment and
    intentionally carry ``valid_length=0``. They are not value examples. This
    helper substitutes a terminal one-step placeholder solely for validation,
    then returns exactly zero on every ineligible row.
    """
    tensor_reference = next(
        (
            item
            for item in (value, next_value, chunk_return, terminal, valid_length)
            if isinstance(item, torch.Tensor)
        ),
        None,
    )
    if tensor_reference is not None:
        dtype = (
            tensor_reference.dtype
            if tensor_reference.is_floating_point()
            else torch.float32
        )
        device = tensor_reference.device
        returns = _torch_column(chunk_return, "chunk_return", device=device, dtype=dtype)
        terminals = _torch_column(terminal, "terminal", device=device, dtype=dtype)
        lengths = _torch_column(valid_length, "valid_length", device=device, dtype=dtype)
        eligible_column = _torch_column(
            eligible, "eligible", device=device, dtype=dtype
        )
        if eligible_column.shape != returns.shape:
            raise ValueError(
                "eligible shape does not match chunk_return: "
                f"{tuple(eligible_column.shape)} != {tuple(returns.shape)}"
            )
        if not torch.allclose(
            eligible_column, eligible_column.round(), atol=1e-6, rtol=0.0
        ) or bool(((eligible_column < 0.0) | (eligible_column > 1.0)).any()):
            raise ValueError("eligible must contain only 0 or 1")
        mask = eligible_column > 0.5
        result = chunk_advantage(
            torch.where(mask, returns, torch.zeros_like(returns)),
            torch.where(mask, terminals, torch.ones_like(terminals)),
            torch.where(mask, lengths, torch.ones_like(lengths)),
            value,
            next_value,
            gamma,
        )
        return torch.where(mask, result, torch.zeros_like(result))

    returns_np = _numpy_column(chunk_return, "chunk_return")
    terminals_np = _numpy_column(terminal, "terminal")
    lengths_np = _numpy_column(valid_length, "valid_length")
    eligible_np = _numpy_column(eligible, "eligible")
    if eligible_np.shape != returns_np.shape:
        raise ValueError(
            "eligible shape does not match chunk_return: "
            f"{eligible_np.shape} != {returns_np.shape}"
        )
    if not np.allclose(eligible_np, np.round(eligible_np), atol=1e-6, rtol=0.0):
        raise ValueError("eligible must contain only 0 or 1")
    if np.any((eligible_np < 0.0) | (eligible_np > 1.0)):
        raise ValueError("eligible must contain only 0 or 1")
    mask_np = eligible_np > 0.5
    result_np = chunk_advantage(
        np.where(mask_np, returns_np, 0.0),
        np.where(mask_np, terminals_np, 1.0),
        np.where(mask_np, lengths_np, 1.0),
        value,
        next_value,
        gamma,
    )
    return np.where(mask_np, result_np, 0.0).astype(np.float32, copy=False)


def make_recap_conditions(
    advantage: Any,
    source_is_expert: Any,
    *,
    fixed_threshold: float | None = None,
    rollout_quantile: float | None = None,
    target_positive_fraction: float | None = None,
    eligible: Any | None = None,
) -> dict[str, Any]:
    """Threshold rollout advantages while forcing every human label to one.

    Exactly one threshold mode must be supplied:

    - ``fixed_threshold`` uses the provided raw advantage threshold;
    - ``rollout_quantile=q`` uses quantile ``q`` of eligible rollout rows;
    - ``target_positive_fraction=p`` uses rollout quantile ``1-p``.

    Conditions use the strict rule ``advantage > threshold``. Consequently,
    tied advantages can make the achieved positive fraction differ from the
    requested fraction. Human rows are always condition 1, including rows
    excluded from rollout calibration by ``eligible``.
    """

    advantages = _numpy_vector(advantage, "advantage").astype(
        np.float64, copy=False
    )
    expert_raw = _numpy_vector(
        source_is_expert,
        "source_is_expert",
    ).astype(np.float64, copy=False)
    if advantages.size == 0:
        raise ValueError("advantage must contain at least one row")
    if expert_raw.shape != advantages.shape:
        raise ValueError(
            "source_is_expert shape does not match advantage: "
            f"{expert_raw.shape} != {advantages.shape}"
        )
    if not np.allclose(expert_raw, np.round(expert_raw), atol=1e-6, rtol=0.0):
        raise ValueError("source_is_expert must contain only 0 or 1")
    if np.any((expert_raw < 0.0) | (expert_raw > 1.0)):
        raise ValueError("source_is_expert must contain only 0 or 1")
    expert = expert_raw > 0.5

    if eligible is None:
        eligible_mask = np.ones(advantages.shape, dtype=bool)
    else:
        eligible_raw = _numpy_vector(eligible, "eligible").astype(
            np.float64, copy=False
        )
        if eligible_raw.shape != advantages.shape:
            raise ValueError(
                f"eligible shape {eligible_raw.shape} does not match "
                f"advantage shape {advantages.shape}"
            )
        if not np.allclose(
            eligible_raw,
            np.round(eligible_raw),
            atol=1e-6,
            rtol=0.0,
        ) or np.any((eligible_raw < 0.0) | (eligible_raw > 1.0)):
            raise ValueError("eligible must contain only 0 or 1")
        eligible_mask = eligible_raw > 0.5

    supplied_modes = sum(
        value is not None
        for value in (
            fixed_threshold,
            rollout_quantile,
            target_positive_fraction,
        )
    )
    if supplied_modes != 1:
        raise ValueError(
            "exactly one of fixed_threshold, rollout_quantile, or "
            "target_positive_fraction must be supplied"
        )

    rollout = (~expert) & eligible_mask
    rollout_advantages = advantages[rollout]
    if rollout_advantages.size == 0:
        raise ValueError("no eligible rollout rows are available for RECAP labeling")

    if fixed_threshold is not None:
        threshold = _finite_scalar(fixed_threshold, "fixed_threshold")
        threshold_mode = "fixed"
    elif rollout_quantile is not None:
        quantile = _finite_scalar(rollout_quantile, "rollout_quantile")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(
                f"rollout_quantile must be in [0, 1], got {quantile}"
            )
        threshold = float(np.quantile(rollout_advantages, quantile))
        threshold_mode = "rollout_quantile"
    else:
        fraction = _finite_scalar(
            target_positive_fraction,
            "target_positive_fraction",
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "target_positive_fraction must be in [0, 1], got "
                f"{fraction}"
            )
        threshold = float(np.quantile(rollout_advantages, 1.0 - fraction))
        threshold_mode = "target_positive_fraction"
    if not np.isfinite(threshold):
        raise ValueError("computed RECAP threshold is non-finite")

    rollout_positive = rollout & (advantages > threshold)
    actor_condition = (expert | rollout_positive).astype(np.uint8)
    positive_count = int(rollout_positive.sum())
    rollout_count = int(rollout.sum())
    return {
        "actor_condition": actor_condition,
        "threshold": float(threshold),
        "threshold_mode": threshold_mode,
        "human_count": int(expert.sum()),
        "eligible_rollout_count": rollout_count,
        "rollout_positive_count": positive_count,
        "rollout_positive_fraction": float(positive_count / rollout_count),
    }


def _integer_indices(indices: Any, name: str = "indices") -> np.ndarray:
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    result = np.asarray(indices)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1:
        raise ValueError(f"{name} must have shape [N], got {result.shape}")
    if result.size == 0:
        return result.astype(np.int64)
    if not np.issubdtype(result.dtype, np.integer):
        if not np.isfinite(result).all() or not np.allclose(
            result,
            np.round(result),
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(f"{name} must contain integer indices")
    return result.astype(np.int64, copy=False)


def validate_sidecar_indices(indices: Any, expected_size: int) -> None:
    """Require indices to cover ``range(expected_size)`` exactly once."""

    if isinstance(expected_size, (bool, np.bool_)):
        raise ValueError("expected_size must be a non-negative integer")
    try:
        expected_size = int(expected_size)
    except (TypeError, ValueError) as error:
        raise ValueError("expected_size must be a non-negative integer") from error
    if expected_size < 0:
        raise ValueError(f"expected_size must be non-negative, got {expected_size}")
    normalized = _integer_indices(indices)
    if normalized.size != expected_size:
        raise ValueError(
            f"sidecar index count={normalized.size} does not match "
            f"expected_size={expected_size}"
        )
    if expected_size == 0:
        return
    if normalized.min() < 0 or normalized.max() >= expected_size:
        raise ValueError(
            "sidecar indices are outside the expected range "
            f"[0, {expected_size})"
        )
    counts = np.bincount(normalized, minlength=expected_size)
    duplicate = np.flatnonzero(counts > 1)
    missing = np.flatnonzero(counts == 0)
    if duplicate.size or missing.size:
        raise ValueError(
            "sidecar indices must cover every dataset row exactly once; "
            f"duplicates={duplicate[:8].tolist()}, missing={missing[:8].tolist()}"
        )


def _sidecar_fields(
    sidecar: Mapping[str, Any],
    fields: Sequence[str] | None,
) -> tuple[str, ...]:
    if fields is None:
        selected = tuple(
            key
            for key, value in sidecar.items()
            if (
                isinstance(value, torch.Tensor) and value.ndim >= 1
            )
            or (isinstance(value, np.ndarray) and value.ndim >= 1)
        )
    else:
        selected = tuple(str(field) for field in fields)
    if not selected:
        raise ValueError("no indexable sidecar fields were selected")
    missing = [field for field in selected if field not in sidecar]
    if missing:
        raise KeyError(f"sidecar is missing requested fields: {missing}")
    return selected


def sidecar_take(
    sidecar: Mapping[str, Any],
    indices: Any,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Gather sidecar arrays at stable dataset indices.

    All selected fields must have the same leading dataset dimension. When
    ``indices`` is a torch tensor, gathered values are returned as tensors on
    that index tensor's device; this makes the helper directly usable on a
    DataLoader batch. Otherwise each field retains its numpy/torch backend.
    """

    if not isinstance(sidecar, Mapping):
        raise TypeError("sidecar must be a mapping")
    selected = _sidecar_fields(sidecar, fields)
    normalized = _integer_indices(indices)
    sizes = set()
    for field in selected:
        value = sidecar[field]
        if not isinstance(value, (np.ndarray, torch.Tensor)) or value.ndim < 1:
            raise ValueError(
                f"sidecar field {field!r} must be a numpy array or tensor "
                "with a leading dataset dimension"
            )
        sizes.add(int(value.shape[0]))
    if len(sizes) != 1:
        raise ValueError(
            f"selected sidecar fields have different leading sizes: {sizes}"
        )
    dataset_size = sizes.pop()
    if normalized.size and (
        int(normalized.min()) < 0 or int(normalized.max()) >= dataset_size
    ):
        raise IndexError(
            f"sidecar batch indices are outside [0, {dataset_size})"
        )

    torch_indices = isinstance(indices, torch.Tensor)
    result: dict[str, Any] = {}
    for field in selected:
        value = sidecar[field]
        if torch_indices:
            if isinstance(value, torch.Tensor):
                field_indices = torch.as_tensor(
                    normalized,
                    dtype=torch.long,
                    device=value.device,
                )
                gathered = value.index_select(0, field_indices)
                result[field] = gathered.to(device=indices.device)
            else:
                gathered = value[normalized]
                result[field] = torch.as_tensor(
                    gathered,
                    device=indices.device,
                )
        elif isinstance(value, torch.Tensor):
            field_indices = torch.as_tensor(
                normalized,
                dtype=torch.long,
                device=value.device,
            )
            result[field] = value.index_select(0, field_indices)
        else:
            result[field] = value[normalized]
    return result


def attach_sidecar_fields(
    batch: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    fields: Sequence[str] | Mapping[str, str],
    *,
    index_key: str = "index",
) -> dict[str, Any]:
    """Return a shallow batch copy with fields gathered from a sidecar.

    A sequence overlays fields under their existing names. A mapping is
    interpreted as ``{batch_field: sidecar_field}``, allowing, for example,
    ``{"success_condition": "actor_condition"}``.
    """

    if index_key not in batch:
        raise KeyError(f"batch is missing stable index key {index_key!r}")
    if isinstance(fields, Mapping):
        destination_to_source = {
            str(destination): str(source)
            for destination, source in fields.items()
        }
    else:
        destination_to_source = {
            str(field): str(field) for field in fields
        }
    gathered = sidecar_take(
        sidecar,
        batch[index_key],
        tuple(destination_to_source.values()),
    )
    output = dict(batch)
    for destination, source in destination_to_source.items():
        output[destination] = gathered[source]
    return output


class SidecarOverlayDataset(torch.utils.data.Dataset):
    """Dataset view that overlays immutable sidecar labels by sample index."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        sidecar: Mapping[str, Any],
        fields: Sequence[str] | Mapping[str, str],
        *,
        index_key: str = "index",
    ):
        self.dataset = dataset
        self.sidecar = sidecar
        self.index_key = str(index_key)
        if isinstance(fields, Mapping):
            self.destination_to_source = {
                str(destination): str(source)
                for destination, source in fields.items()
            }
        else:
            self.destination_to_source = {
                str(field): str(field) for field in fields
            }
        selected = _sidecar_fields(
            sidecar,
            tuple(self.destination_to_source.values()),
        )
        sizes = {int(sidecar[field].shape[0]) for field in selected}
        if sizes != {len(dataset)}:
            raise ValueError(
                f"sidecar size(s)={sorted(sizes)} do not match dataset "
                f"size={len(dataset)}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        if name in {"dataset", "sidecar", "destination_to_source"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("SidecarOverlayDataset requires mapping samples")
        if self.index_key not in sample:
            raise KeyError(
                f"dataset sample is missing stable index key {self.index_key!r}"
            )
        stable_index = _integer_indices(
            np.asarray([sample[self.index_key]]),
            name=self.index_key,
        )
        gathered = sidecar_take(
            self.sidecar,
            stable_index,
            tuple(self.destination_to_source.values()),
        )
        output = dict(sample)
        for destination, source in self.destination_to_source.items():
            output[destination] = gathered[source][0]
        return output


# Descriptive aliases retained for callers that prefer verb-oriented names.
compute_chunk_advantage = chunk_advantage
threshold_rollout_advantages = make_recap_conditions


__all__ = [
    "FAILURE_SOURCES",
    "SUCCESS_SOURCES",
    "SidecarOverlayDataset",
    "attach_sidecar_fields",
    "build_canonical_episode_targets",
    "chunk_advantage",
    "compute_chunk_advantage",
    "make_recap_conditions",
    "masked_chunk_advantage",
    "sidecar_take",
    "threshold_rollout_advantages",
    "validate_sidecar_indices",
]
