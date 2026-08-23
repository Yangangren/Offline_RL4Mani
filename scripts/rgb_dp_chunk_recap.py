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


def _categorical_vector(values: Any, name: str) -> np.ndarray:
    """Normalize a row-aligned categorical vector without dtype coercion."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    result = np.asarray(values)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1:
        raise ValueError(f"{name} must have shape [N] or [N,1], got {result.shape}")
    return result


def _python_scalar(value: Any, name: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        raise ValueError(f"{name} contains an invalid category")
    try:
        hash(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain scalar categories") from error
    return value


def _scalar_token(value: Any, name: str) -> tuple[str, str]:
    value = _python_scalar(value, name)
    return type(value).__qualname__, repr(value)


def _categorical_key(value: Any) -> str:
    value = _python_scalar(value, "category")
    return str(value)


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = int(value)
        exact = float(value) == float(result)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if not exact or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _episode_table(
    episode_index: Any,
    source: Any,
    outcome: Any | None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Build one record per episode and enforce episode-constant strata."""
    episodes = _categorical_vector(episode_index, "episode_index")
    sources = _categorical_vector(source, "source")
    if sources.shape != episodes.shape:
        raise ValueError(
            f"source shape {sources.shape} does not match episode_index "
            f"shape {episodes.shape}"
        )
    outcomes = None if outcome is None else _categorical_vector(outcome, "outcome")
    if outcomes is not None and outcomes.shape != episodes.shape:
        raise ValueError(
            f"outcome shape {outcomes.shape} does not match episode_index "
            f"shape {episodes.shape}"
        )
    if episodes.size == 0:
        raise ValueError("episode_index must contain at least one row")

    by_episode: dict[tuple[str, str], dict[str, Any]] = {}
    row_tokens: list[tuple[str, str]] = []
    for row in range(int(episodes.size)):
        episode = _python_scalar(episodes[row], "episode_index")
        source_value = _python_scalar(sources[row], "source")
        outcome_value = (
            None if outcomes is None else _python_scalar(outcomes[row], "outcome")
        )
        episode_token = _scalar_token(episode, "episode_index")
        source_token = _scalar_token(source_value, "source")
        outcome_token = (
            ("__missing_outcome__", "")
            if outcomes is None
            else _scalar_token(outcome_value, "outcome")
        )
        row_tokens.append(episode_token)
        record = by_episode.get(episode_token)
        if record is None:
            by_episode[episode_token] = {
                "episode": episode,
                "token": episode_token,
                "source": source_value,
                "source_token": source_token,
                "outcome": outcome_value,
                "outcome_token": outcome_token,
                "rows": 1,
            }
        else:
            if record["source_token"] != source_token:
                raise ValueError(f"episode {episode!r} has multiple source values")
            if record["outcome_token"] != outcome_token:
                raise ValueError(f"episode {episode!r} has multiple outcome values")
            record["rows"] += 1
    return sorted(by_episode.values(), key=lambda item: item["token"]), row_tokens


def assign_stratified_episode_folds(
    episode_index: Any,
    source: Any,
    *,
    num_folds: int,
    seed: int,
    outcome: Any | None = None,
) -> np.ndarray:
    """Return deterministic row-aligned, episode-level stratified OOF folds.

    Stratification uses ``(source, outcome)`` when outcome is supplied and
    source alone otherwise. An episode is never split between folds. Episode
    counts in every stratum differ by at most one; variable-length episodes
    are distributed largest-first to keep row counts reasonably balanced.
    The result is invariant to input row order.
    """
    num_folds = _positive_integer(num_folds, "num_folds", minimum=2)
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ValueError("seed must be an integer") from error
    records, row_tokens = _episode_table(episode_index, source, outcome)
    if len(records) < num_folds:
        raise ValueError(
            f"num_folds={num_folds} exceeds episode count={len(records)}"
        )

    strata: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for record in records:
        key = record["source_token"], record["outcome_token"]
        strata.setdefault(key, []).append(record)
    rng = np.random.default_rng(seed % (2**64))
    fold_rows = np.zeros(num_folds, dtype=np.int64)
    fold_episodes = np.zeros(num_folds, dtype=np.int64)
    episode_fold: dict[tuple[str, str], int] = {}
    for stratum_key in sorted(strata):
        members = sorted(strata[stratum_key], key=lambda item: item["token"])
        member_ties = rng.random(len(members))
        members = [
            member
            for _, member in sorted(
                zip(member_ties, members),
                key=lambda pair: (-pair[1]["rows"], pair[0], pair[1]["token"]),
            )
        ]
        fold_ties = rng.random(num_folds)
        fold_order = sorted(
            range(num_folds),
            key=lambda fold: (
                int(fold_episodes[fold]),
                int(fold_rows[fold]),
                float(fold_ties[fold]),
                fold,
            ),
        )
        for position, member in enumerate(members):
            fold = int(fold_order[position % num_folds])
            episode_fold[member["token"]] = fold
            fold_episodes[fold] += 1
            fold_rows[fold] += int(member["rows"])

    result = np.asarray([episode_fold[token] for token in row_tokens], dtype=np.int64)
    validate_episode_fold_assignment(
        episode_index,
        result,
        num_folds=num_folds,
        source=source,
        outcome=outcome,
        require_all_folds=True,
        require_stratum_balance=True,
    )
    return result


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


def validate_episode_fold_assignment(
    episode_index: Any,
    fold_index: Any,
    *,
    num_folds: int,
    source: Any | None = None,
    outcome: Any | None = None,
    require_all_folds: bool = True,
    require_stratum_balance: bool = True,
) -> dict[str, Any]:
    """Validate episode isolation and source/outcome fold balance.

    The returned JSON-safe summary is suitable for target-sidecar provenance.
    A ``ValueError`` is raised for any episode split across folds, a missing
    fold (when requested), or a per-stratum episode-count difference above one.
    """
    num_folds = _positive_integer(num_folds, "num_folds", minimum=2)
    episodes = _categorical_vector(episode_index, "episode_index")
    folds = _integer_indices(fold_index, "fold_index")
    if folds.shape != episodes.shape:
        raise ValueError(
            f"fold_index shape {folds.shape} does not match episode_index "
            f"shape {episodes.shape}"
        )
    if np.any((folds < 0) | (folds >= num_folds)):
        raise ValueError(f"fold_index must be in [0, {num_folds})")
    if source is None:
        source = np.zeros(episodes.shape, dtype=np.int8)
    records, row_tokens = _episode_table(episodes, source, outcome)

    episode_folds: dict[tuple[str, str], int] = {}
    episode_rows: dict[tuple[str, str], int] = {}
    for token, fold in zip(row_tokens, folds):
        fold = int(fold)
        previous = episode_folds.setdefault(token, fold)
        if previous != fold:
            raise ValueError(
                "episode leakage: one episode has rows assigned to multiple folds"
            )
        episode_rows[token] = episode_rows.get(token, 0) + 1
    used = set(episode_folds.values())
    missing = sorted(set(range(num_folds)) - used)
    if bool(require_all_folds) and missing:
        raise ValueError(f"fold assignment is missing folds {missing}")

    fold_rows = [int(np.sum(folds == fold)) for fold in range(num_folds)]
    fold_episodes = [
        sum(value == fold for value in episode_folds.values())
        for fold in range(num_folds)
    ]
    strata: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for record in records:
        key = record["source_token"], record["outcome_token"]
        strata.setdefault(key, []).append(record)
    stratum_summaries = []
    max_imbalance = 0
    for key in sorted(strata):
        members = strata[key]
        episode_counts = [0] * num_folds
        row_counts = [0] * num_folds
        for member in members:
            fold = episode_folds[member["token"]]
            episode_counts[fold] += 1
            row_counts[fold] += episode_rows[member["token"]]
        imbalance = max(episode_counts) - min(episode_counts)
        max_imbalance = max(max_imbalance, imbalance)
        if bool(require_stratum_balance) and imbalance > 1:
            raise ValueError(
                "stratified fold assignment is imbalanced: "
                f"source={members[0]['source']!r}, "
                f"outcome={members[0]['outcome']!r}, counts={episode_counts}"
            )
        stratum_summaries.append(
            {
                "source": _categorical_key(members[0]["source"]),
                "outcome": (
                    None
                    if members[0]["outcome"] is None
                    else _categorical_key(members[0]["outcome"])
                ),
                "episodes": len(members),
                "rows": int(sum(row_counts)),
                "fold_episode_counts": episode_counts,
                "fold_row_counts": row_counts,
                "episode_count_imbalance": int(imbalance),
            }
        )
    return {
        "rows": int(episodes.size),
        "episodes": len(records),
        "num_folds": num_folds,
        "fold_rows": fold_rows,
        "fold_episodes": fold_episodes,
        "missing_folds": missing,
        "all_folds_present": not missing,
        "no_episode_leakage": True,
        "max_stratum_episode_imbalance": int(max_imbalance),
        "strata": stratum_summaries,
    }


def validate_oof_predictions(
    episode_index: Any,
    fold_index: Any,
    prediction_fold: Any,
    predictions: Any,
    *,
    num_folds: int,
    prediction_count: Any | None = None,
    training_episode_indices_by_fold: Mapping[int, Any] | None = None,
    require_complete_training_complement: bool = False,
) -> dict[str, Any]:
    """Validate complete, finite OOF predictions and optional train provenance.

    ``prediction_fold`` identifies the held-out model that produced each row
    and must exactly equal the row's assigned ``fold_index``. When training
    episode lists are supplied, this helper proves that no held-out episode was
    seen by its producer model. Exact complement coverage can also be required.
    """
    num_folds = _positive_integer(num_folds, "num_folds", minimum=2)
    episodes = _categorical_vector(episode_index, "episode_index")
    folds = _integer_indices(fold_index, "fold_index")
    producers = _integer_indices(prediction_fold, "prediction_fold")
    if folds.shape != episodes.shape or producers.shape != episodes.shape:
        raise ValueError("OOF row-aligned fields have different shapes")
    validate_episode_fold_assignment(
        episodes,
        folds,
        num_folds=num_folds,
        require_all_folds=True,
        require_stratum_balance=False,
    )
    if np.any((producers < 0) | (producers >= num_folds)):
        raise ValueError(f"prediction_fold must be in [0, {num_folds})")
    mismatch = np.flatnonzero(producers != folds)
    if mismatch.size:
        raise ValueError(
            "OOF provenance mismatch: prediction model did not hold out rows "
            f"{mismatch[:8].tolist()}"
        )
    if prediction_count is not None:
        counts = _integer_indices(prediction_count, "prediction_count")
        if counts.shape != episodes.shape or np.any(counts != 1):
            raise ValueError("every row must receive exactly one OOF prediction")

    named_predictions = (
        dict(predictions)
        if isinstance(predictions, Mapping)
        else {"prediction": predictions}
    )
    if not named_predictions:
        raise ValueError("predictions must contain at least one field")
    prediction_fields: dict[str, dict[str, Any]] = {}
    for name, values in named_predictions.items():
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        array = np.asarray(values)
        if array.ndim == 0 or int(array.shape[0]) != int(episodes.size):
            raise ValueError(
                f"prediction field {name!r} has no row-aligned leading dimension"
            )
        finite = np.isfinite(array)
        if not bool(finite.all()):
            raise ValueError(f"prediction field {name!r} contains non-finite values")
        prediction_fields[str(name)] = {
            "shape": [int(value) for value in array.shape],
            "finite": True,
        }

    episode_tokens = [_scalar_token(value, "episode_index") for value in episodes]
    all_episodes = set(episode_tokens)
    training_summaries = []
    if training_episode_indices_by_fold is not None:
        normalized_training: dict[int, Any] = {}
        for raw_fold, values in training_episode_indices_by_fold.items():
            fold = _positive_integer(int(raw_fold) + 1, "training fold") - 1
            if fold >= num_folds or fold in normalized_training:
                raise ValueError("training provenance contains an invalid fold key")
            normalized_training[fold] = values
        if set(normalized_training) != set(range(num_folds)):
            raise ValueError("training provenance must provide every OOF fold")
        for fold in range(num_folds):
            values = _categorical_vector(
                normalized_training[fold], f"training_episode_indices_by_fold[{fold}]"
            )
            train_tokens = [
                _scalar_token(value, "training_episode_index") for value in values
            ]
            if len(set(train_tokens)) != len(train_tokens):
                raise ValueError(f"training provenance for fold {fold} has duplicates")
            train_set = set(train_tokens)
            held_out = {
                token for token, row_fold in zip(episode_tokens, folds)
                if int(row_fold) == fold
            }
            leakage = held_out & train_set
            unknown = train_set - all_episodes
            if leakage:
                raise ValueError(f"OOF training leakage detected for fold {fold}")
            if unknown:
                raise ValueError(f"OOF training provenance has unknown episodes in fold {fold}")
            expected = all_episodes - held_out
            missing_training = expected - train_set
            if bool(require_complete_training_complement) and missing_training:
                raise ValueError(f"OOF training complement is incomplete for fold {fold}")
            training_summaries.append(
                {
                    "fold": fold,
                    "held_out_episodes": len(held_out),
                    "training_episodes": len(train_set),
                    "missing_complement_episodes": len(missing_training),
                    "no_training_leakage": True,
                }
            )
    return {
        "rows": int(episodes.size),
        "episodes": len(all_episodes),
        "num_folds": num_folds,
        "fold_rows": [int(np.sum(folds == fold)) for fold in range(num_folds)],
        "prediction_fields": prediction_fields,
        "each_row_predicted_once": True,
        "producer_matches_held_out_fold": True,
        "training_provenance_checked": training_episode_indices_by_fold is not None,
        "fold_training_provenance": training_summaries,
    }


def summarize_recap_conditions(
    advantage: Any,
    actor_condition: Any,
    source: Any,
    fold_index: Any,
    *,
    eligible: Any | None = None,
    episode_index: Any | None = None,
    quantiles: Sequence[float] = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0),
) -> dict[str, Any]:
    """Build finite, JSON-safe condition diagnostics by source and OOF fold."""
    advantages = _numpy_vector(advantage, "advantage", finite=False).astype(
        np.float64, copy=False
    )
    conditions_raw = _numpy_vector(
        actor_condition, "actor_condition", finite=False
    ).astype(np.float64, copy=False)
    sources = _categorical_vector(source, "source")
    folds = _integer_indices(fold_index, "fold_index")
    count = int(advantages.size)
    for name, values in (
        ("actor_condition", conditions_raw),
        ("source", sources),
        ("fold_index", folds),
    ):
        if values.shape != advantages.shape:
            raise ValueError(
                f"{name} shape {values.shape} does not match advantage "
                f"shape {advantages.shape}"
            )
    if count == 0:
        raise ValueError("advantage must contain at least one row")
    finite = np.isfinite(advantages)
    if not bool(finite.all()):
        bad = np.flatnonzero(~finite)
        raise ValueError(f"advantage contains non-finite rows {bad[:8].tolist()}")
    if not np.allclose(conditions_raw, np.round(conditions_raw), atol=0.0, rtol=0.0):
        raise ValueError("actor_condition must contain only 0 or 1")
    if np.any((conditions_raw < 0.0) | (conditions_raw > 1.0)):
        raise ValueError("actor_condition must contain only 0 or 1")
    conditions = conditions_raw > 0.5
    if np.any(folds < 0):
        raise ValueError("fold_index must be non-negative")

    if eligible is None:
        eligible_mask = np.ones(count, dtype=bool)
    else:
        eligible_raw = _numpy_vector(eligible, "eligible", finite=False).astype(
            np.float64, copy=False
        )
        if eligible_raw.shape != advantages.shape:
            raise ValueError("eligible shape does not match advantage")
        if not np.allclose(eligible_raw, np.round(eligible_raw), atol=0.0, rtol=0.0):
            raise ValueError("eligible must contain only 0 or 1")
        if np.any((eligible_raw < 0.0) | (eligible_raw > 1.0)):
            raise ValueError("eligible must contain only 0 or 1")
        eligible_mask = eligible_raw > 0.5

    quantile_values = sorted({_finite_scalar(value, "quantile") for value in quantiles})
    if not quantile_values or quantile_values[0] < 0.0 or quantile_values[-1] > 1.0:
        raise ValueError("quantiles must be a non-empty sequence in [0, 1]")
    episodes = None
    episode_tokens = None
    if episode_index is not None:
        episodes = _categorical_vector(episode_index, "episode_index")
        if episodes.shape != advantages.shape:
            raise ValueError("episode_index shape does not match advantage")
        validate_episode_fold_assignment(
            episodes,
            folds,
            num_folds=int(folds.max()) + 1,
            source=sources,
            require_all_folds=False,
            require_stratum_balance=False,
        )
        episode_tokens = np.asarray(
            [_scalar_token(value, "episode_index") for value in episodes],
            dtype=object,
        )

    def group_stats(mask: np.ndarray) -> dict[str, Any]:
        rows = int(mask.sum())
        selected = mask & eligible_mask
        values = advantages[selected]
        positive_rows = int((conditions & mask).sum())
        eligible_positive = int((conditions & selected).sum())
        episode_count = None
        if episode_tokens is not None:
            episode_count = len({tuple(value) for value in episode_tokens[mask]})
        advantage_stats = {
            "count": int(values.size),
            "finite": True,
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "quantiles": {
                format(value, ".6g"): float(np.quantile(values, value))
                for value in quantile_values
            } if values.size else {},
        }
        return {
            "rows": rows,
            "episodes": episode_count,
            "eligible_rows": int(selected.sum()),
            "positive_rows": positive_rows,
            "positive_fraction": float(positive_rows / rows) if rows else None,
            "eligible_positive_rows": eligible_positive,
            "eligible_positive_fraction": (
                float(eligible_positive / selected.sum()) if selected.any() else None
            ),
            "advantage": advantage_stats,
        }

    source_tokens = [_scalar_token(value, "source") for value in sources]
    source_groups: dict[tuple[str, str], tuple[Any, np.ndarray]] = {}
    for row, token in enumerate(source_tokens):
        if token not in source_groups:
            source_groups[token] = (sources[row], np.zeros(count, dtype=bool))
        source_groups[token][1][row] = True
    by_source = {}
    by_source_and_fold = {}
    for token in sorted(source_groups):
        value, mask = source_groups[token]
        key = _categorical_key(value)
        if key in by_source:
            key = f"{token[0]}:{key}"
        by_source[key] = group_stats(mask)
        by_source_and_fold[key] = {
            str(fold): group_stats(mask & (folds == fold))
            for fold in sorted(set(int(value) for value in folds))
        }
    return {
        "rows": count,
        "finite_checks": {"advantage": True, "condition": True},
        "quantiles": quantile_values,
        "overall": group_stats(np.ones(count, dtype=bool)),
        "by_source": by_source,
        "by_fold": {
            str(fold): group_stats(folds == fold)
            for fold in sorted(set(int(value) for value in folds))
        },
        "by_source_and_fold": by_source_and_fold,
    }


# Descriptive aliases retained for callers that prefer verb-oriented names.
compute_chunk_advantage = chunk_advantage
threshold_rollout_advantages = make_recap_conditions
summarize_recap_labels = summarize_recap_conditions
validate_oof_fold_provenance = validate_oof_predictions


__all__ = [
    "FAILURE_SOURCES",
    "SUCCESS_SOURCES",
    "SidecarOverlayDataset",
    "assign_stratified_episode_folds",
    "attach_sidecar_fields",
    "build_canonical_episode_targets",
    "chunk_advantage",
    "compute_chunk_advantage",
    "make_recap_conditions",
    "masked_chunk_advantage",
    "sidecar_take",
    "summarize_recap_conditions",
    "summarize_recap_labels",
    "threshold_rollout_advantages",
    "validate_episode_fold_assignment",
    "validate_oof_fold_provenance",
    "validate_oof_predictions",
    "validate_sidecar_indices",
]
