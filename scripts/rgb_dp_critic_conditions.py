"""Pure utilities for constructing critic-conditioned actor labels.

The learned-critic condition modes share a three-way labeling rule for rollout
rows:

* scores strictly above the high threshold are positive;
* scores strictly below the low threshold are negative;
* scores on either threshold, or between them, are null / dropped out.

Human demonstration rows are always positive.  A null condition is encoded as
``actor_condition=0`` and ``actor_condition_mask=0`` so it remains distinct
from an explicit negative condition, which is encoded as ``(0, 1)``.

This module deliberately has no dataset or model dependencies.  Critic scoring,
sidecar persistence, and training-time lookup belong to their respective
callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SUPPORTED_CRITIC_CONDITION_MODES = (
    "critic_advantage",
    "critic_q",
)
SUPPORTED_THRESHOLD_MODES = (
    "fixed",
    "quantile",
    "outcome_purity",
)


def validate_critic_condition_mode(mode: str) -> str:
    """Return a normalized learned-critic mode or raise ``ValueError``."""

    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_CRITIC_CONDITION_MODES:
        choices = ", ".join(SUPPORTED_CRITIC_CONDITION_MODES)
        raise ValueError(
            f"unsupported critic condition mode {mode!r}; expected one of: "
            f"{choices}"
        )
    return normalized


def _numpy_vector(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1:
        raise ValueError(f"{name} must have shape [N], got {result.shape}")
    return result


def _finite_scores(scores: Any) -> np.ndarray:
    try:
        result = _numpy_vector(scores, "scores").astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be a numeric vector") from exc
    if result.size == 0:
        raise ValueError("scores must contain at least one row")
    if not np.isfinite(result).all():
        bad_count = int((~np.isfinite(result)).sum())
        raise ValueError(f"scores must be finite; found {bad_count} non-finite rows")
    return result


def _binary_mask(value: Any, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    raw = _numpy_vector(value, name)
    if raw.shape != expected_shape:
        raise ValueError(
            f"{name} shape does not match scores: {raw.shape} != {expected_shape}"
        )
    try:
        numeric = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only 0 or 1") from exc
    if not np.isfinite(numeric).all() or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError(f"{name} must contain only 0 or 1")
    return numeric.astype(bool, copy=False)


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    return result


def _validate_thresholds(low_threshold: Any, high_threshold: Any) -> tuple[float, float]:
    low = _finite_scalar(low_threshold, "low_threshold")
    high = _finite_scalar(high_threshold, "high_threshold")
    if not low < high:
        raise ValueError(
            "low_threshold must be strictly less than high_threshold, "
            f"got {low} >= {high}"
        )
    return low, high


def _fraction(count: int, total: int) -> float | None:
    return float(count / total) if total else None


def _score_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "quantiles": {},
        }
    quantile_levels = (0.05, 0.25, 0.5, 0.75, 0.95)
    quantiles = np.quantile(values, quantile_levels)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            f"p{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
    }


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric != result or result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _calibrate_outcome_purity_thresholds(
    scores: np.ndarray,
    success: np.ndarray,
    *,
    target_purity: float,
    min_tail_count: int,
) -> tuple[float, float, dict[str, Any]]:
    """Find deterministic observed-score cuts for pure outcome tails."""

    # Sort once, then collapse ties into groups. Stable sorting is not needed
    # for the aggregate counts, but makes the grouping order deterministic if
    # this code later records row-level diagnostics within a tied group.
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_success = success[order].astype(np.int64, copy=False)
    group_starts = np.flatnonzero(
        np.concatenate(
            (np.ones(1, dtype=bool), sorted_scores[1:] != sorted_scores[:-1])
        )
    )
    unique_scores = sorted_scores[group_starts]
    group_ends = np.concatenate(
        (group_starts[1:], np.asarray([scores.size], dtype=np.int64))
    )
    group_counts = group_ends - group_starts
    group_success_counts = np.add.reduceat(sorted_success, group_starts)

    # A candidate cut at group i excludes that entire group from both strict
    # tails. Exclusive prefix and suffix counts therefore exactly implement
    # scores < candidate and scores > candidate without rescanning all rows.
    cumulative_counts = np.cumsum(group_counts, dtype=np.int64)
    cumulative_success_counts = np.cumsum(
        group_success_counts, dtype=np.int64
    )
    low_counts = np.concatenate(
        (np.zeros(1, dtype=np.int64), cumulative_counts[:-1])
    )
    low_success_counts = np.concatenate(
        (np.zeros(1, dtype=np.int64), cumulative_success_counts[:-1])
    )
    low_failure_counts = low_counts - low_success_counts
    high_counts = scores.size - cumulative_counts
    high_success_counts = int(sorted_success.sum()) - cumulative_success_counts
    high_failure_counts = high_counts - high_success_counts

    low_purities = np.zeros(unique_scores.size, dtype=np.float64)
    high_purities = np.zeros(unique_scores.size, dtype=np.float64)
    np.divide(
        low_failure_counts,
        low_counts,
        out=low_purities,
        where=low_counts != 0,
    )
    np.divide(
        high_success_counts,
        high_counts,
        out=high_purities,
        where=high_counts != 0,
    )
    passing_low = (low_counts >= min_tail_count) & (
        low_purities >= target_purity
    )
    passing_high = (high_counts >= min_tail_count) & (
        high_purities >= target_purity
    )

    passing_high_indices = np.flatnonzero(passing_high)
    if passing_high_indices.size == 0:
        raise ValueError(
            "no high-score tail satisfies target_purity="
            f"{target_purity} with min_tail_count={min_tail_count}"
        )
    passing_low_indices = np.flatnonzero(passing_low)
    if passing_low_indices.size == 0:
        raise ValueError(
            "no low-score tail satisfies target_purity="
            f"{target_purity} with min_tail_count={min_tail_count}"
        )

    def tail_record(index: int, *, high: bool) -> dict[str, Any]:
        if high:
            count = int(high_counts[index])
            success_count = int(high_success_counts[index])
            failure_count = int(high_failure_counts[index])
            purity = float(high_purities[index])
        else:
            count = int(low_counts[index])
            success_count = int(low_success_counts[index])
            failure_count = int(low_failure_counts[index])
            purity = float(low_purities[index])
        return {
            "threshold": float(unique_scores[index]),
            "count": count,
            "success_count": success_count,
            "failure_count": failure_count,
            "purity": purity,
        }

    # Selecting the first passing high cut and last passing low cut implements
    # the requested lowest-high / highest-low rules. Because candidate cuts
    # are tied-score groups, a boundary can never split equal scores.
    selected_high_index = int(passing_high_indices[0])
    selected_low_index = int(passing_low_indices[-1])
    selected_high = tail_record(selected_high_index, high=True)
    selected_low = tail_record(selected_low_index, high=False)
    raw_low = float(selected_low["threshold"])
    raw_high = float(selected_high["threshold"])
    threshold_adjustment = None
    if raw_low < raw_high:
        low, high = raw_low, raw_high
    else:
        # With cleanly separated outcomes, the broadest pure strict tails can
        # produce crossed *observed-score* cuts even though their selected row
        # sets do not overlap. Place two cuts inside the empty score gap. This
        # preserves both tail memberships while creating a real null band.
        low_tail = scores < raw_low
        high_tail = scores > raw_high
        low_tail_max = float(scores[low_tail].max())
        high_tail_min = float(scores[high_tail].min())
        if not low_tail_max < high_tail_min:
            raise ValueError(
                "outcome-purity tails overlap and cannot produce "
                "low_threshold < high_threshold; increase target_purity or "
                "min_tail_count"
            )
        gap = high_tail_min - low_tail_max
        low = low_tail_max + gap / 3.0
        high = low_tail_max + 2.0 * gap / 3.0
        low, high = _validate_thresholds(low, high)
        if not (
            np.array_equal(scores < low, low_tail)
            and np.array_equal(scores > high, high_tail)
        ):
            raise RuntimeError(
                "internal outcome-purity gap adjustment changed tail membership"
            )
        threshold_adjustment = {
            "reason": "crossed_observed_cuts_with_disjoint_score_tails",
            "raw_low_cut": raw_low,
            "raw_high_cut": raw_high,
            "low_tail_max": low_tail_max,
            "high_tail_min": high_tail_min,
        }
    audit = {
        "target_purity": target_purity,
        "min_tail_count": min_tail_count,
        "unique_score_count": int(unique_scores.size),
        "passing_low_candidate_count": int(passing_low_indices.size),
        "passing_high_candidate_count": int(passing_high_indices.size),
        "selected_low_tail": selected_low,
        "selected_high_tail": selected_high,
        "threshold_adjustment": threshold_adjustment,
    }
    return low, high, audit


def calibrate_critic_condition_thresholds(
    scores: Any,
    is_human: Any,
    *,
    mode: str,
    threshold_mode: str,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    low_quantile: float | None = None,
    high_quantile: float | None = None,
    rollout_success: Any | None = None,
    target_purity: float | None = None,
    min_tail_count: int | None = None,
    eligible: Any | None = None,
) -> dict[str, Any]:
    """Resolve fixed or rollout-only quantile thresholds.

    ``threshold_mode="fixed"`` requires ``low_threshold`` and
    ``high_threshold``.  ``threshold_mode="quantile"`` instead requires
    ``low_quantile`` and ``high_quantile``; their values are computed only
    from rows where ``is_human == 0`` and ``eligible == 1``.

    ``threshold_mode="outcome_purity"`` requires peer ``rollout_success``
    labels, ``target_purity``, and ``min_tail_count``.  It selects the lowest
    observed cut whose strict upper tail reaches the requested success purity,
    and the highest observed cut whose strict lower tail reaches the requested
    failure purity.  Tied scores therefore cannot be split across a boundary.
    Callers choose all defaults explicitly, which keeps task-specific
    calibration policy out of this generic utility.

    All score rows must be finite, including human or ineligible rows.  This
    catches incomplete critic sidecars before they reach actor training.
    """

    normalized_mode = validate_critic_condition_mode(mode)
    score_array = _finite_scores(scores)
    human = _binary_mask(is_human, "is_human", score_array.shape)
    if eligible is None:
        eligible_mask = np.ones(score_array.shape, dtype=bool)
    else:
        eligible_mask = _binary_mask(eligible, "eligible", score_array.shape)

    normalized_threshold_mode = str(threshold_mode).strip().lower()
    if normalized_threshold_mode not in SUPPORTED_THRESHOLD_MODES:
        choices = ", ".join(SUPPORTED_THRESHOLD_MODES)
        raise ValueError(
            f"unsupported threshold_mode {threshold_mode!r}; expected one of: "
            f"{choices}"
        )

    rollout = ~human
    calibration_mask = rollout & eligible_mask
    calibration_scores = score_array[calibration_mask]

    outcome_audit = None
    if normalized_threshold_mode == "fixed":
        if any(
            value is not None
            for value in (
                low_quantile,
                high_quantile,
                rollout_success,
                target_purity,
                min_tail_count,
            )
        ):
            raise ValueError(
                "quantile and outcome-purity arguments are not valid for "
                "threshold_mode='fixed'"
            )
        if low_threshold is None or high_threshold is None:
            raise ValueError(
                "threshold_mode='fixed' requires low_threshold and "
                "high_threshold"
            )
        low, high = _validate_thresholds(low_threshold, high_threshold)
        resolved_low_quantile = None
        resolved_high_quantile = None
    elif normalized_threshold_mode == "quantile":
        if any(
            value is not None
            for value in (
                low_threshold,
                high_threshold,
                rollout_success,
                target_purity,
                min_tail_count,
            )
        ):
            raise ValueError(
                "fixed-threshold and outcome-purity arguments are not valid "
                "for threshold_mode='quantile'"
            )
        if low_quantile is None or high_quantile is None:
            raise ValueError(
                "threshold_mode='quantile' requires low_quantile and "
                "high_quantile"
            )
        resolved_low_quantile = _finite_scalar(low_quantile, "low_quantile")
        resolved_high_quantile = _finite_scalar(high_quantile, "high_quantile")
        if not (
            0.0 <= resolved_low_quantile < resolved_high_quantile <= 1.0
        ):
            raise ValueError(
                "quantiles must satisfy 0 <= low_quantile < high_quantile "
                f"<= 1, got {resolved_low_quantile}, "
                f"{resolved_high_quantile}"
            )
        if calibration_scores.size == 0:
            raise ValueError(
                "no eligible rollout rows are available for quantile "
                "threshold calibration"
            )
        low, high = _validate_thresholds(
            np.quantile(calibration_scores, resolved_low_quantile),
            np.quantile(calibration_scores, resolved_high_quantile),
        )
    else:
        if any(
            value is not None
            for value in (
                low_threshold,
                high_threshold,
                low_quantile,
                high_quantile,
            )
        ):
            raise ValueError(
                "fixed-threshold and quantile arguments are not valid for "
                "threshold_mode='outcome_purity'"
            )
        if (
            rollout_success is None
            or target_purity is None
            or min_tail_count is None
        ):
            raise ValueError(
                "threshold_mode='outcome_purity' requires rollout_success, "
                "target_purity, and min_tail_count"
            )
        success = _binary_mask(
            rollout_success,
            "rollout_success",
            score_array.shape,
        )
        resolved_target_purity = _finite_scalar(target_purity, "target_purity")
        if not 0.0 <= resolved_target_purity <= 1.0:
            raise ValueError(
                "target_purity must be in [0, 1], got "
                f"{resolved_target_purity}"
            )
        resolved_min_tail_count = _positive_integer(
            min_tail_count, "min_tail_count"
        )
        if calibration_scores.size == 0:
            raise ValueError(
                "no eligible rollout rows are available for outcome-purity "
                "threshold calibration"
            )
        low, high, outcome_audit = _calibrate_outcome_purity_thresholds(
            calibration_scores,
            success[calibration_mask],
            target_purity=resolved_target_purity,
            min_tail_count=resolved_min_tail_count,
        )
        resolved_low_quantile = None
        resolved_high_quantile = None

    return {
        "mode": normalized_mode,
        "threshold_mode": normalized_threshold_mode,
        "low_threshold": low,
        "high_threshold": high,
        "low_quantile": resolved_low_quantile,
        "high_quantile": resolved_high_quantile,
        "audit": {
            "counts": {
                "total": int(score_array.size),
                "human": int(human.sum()),
                "rollout": int(rollout.sum()),
                "eligible_rollout": int(calibration_mask.sum()),
                "ineligible_rollout": int((rollout & ~eligible_mask).sum()),
            },
            "score_stats": {
                "all": _score_stats(score_array),
                "human": _score_stats(score_array[human]),
                "rollout": _score_stats(score_array[rollout]),
                "eligible_rollout": _score_stats(calibration_scores),
            },
            "outcome_purity": outcome_audit,
        },
    }


def construct_critic_conditions(
    scores: Any,
    is_human: Any,
    *,
    mode: str,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    """Construct positive, negative, and null actor conditions.

    Human rows are unconditionally ``(label=1, mask=1)``.  Rollout scores use
    strict comparisons, so equality with either threshold belongs to the null
    band along with every score between the thresholds.
    """

    normalized_mode = validate_critic_condition_mode(mode)
    score_array = _finite_scores(scores)
    human = _binary_mask(is_human, "is_human", score_array.shape)
    low, high = _validate_thresholds(low_threshold, high_threshold)

    rollout = ~human
    rollout_positive = rollout & (score_array > high)
    rollout_negative = rollout & (score_array < low)
    rollout_null = rollout & ~(rollout_positive | rollout_negative)

    positive = human | rollout_positive
    negative = rollout_negative
    null = rollout_null

    actor_condition = positive.astype(np.uint8)
    actor_condition_mask = (positive | negative).astype(np.uint8)

    total_count = int(score_array.size)
    rollout_count = int(rollout.sum())
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    null_count = int(null.sum())
    rollout_positive_count = int(rollout_positive.sum())
    rollout_negative_count = int(rollout_negative.sum())
    rollout_null_count = int(rollout_null.sum())

    audit = {
        "mode": normalized_mode,
        "thresholds": {
            "low": low,
            "high": high,
            "comparison": "score < low: 0; score > high: 1; otherwise: null",
        },
        "counts": {
            "total": total_count,
            "human": int(human.sum()),
            "rollout": rollout_count,
            "positive": positive_count,
            "negative": negative_count,
            "null": null_count,
            "explicit": positive_count + negative_count,
            "human_positive": int(human.sum()),
            "rollout_positive": rollout_positive_count,
            "rollout_negative": rollout_negative_count,
            "rollout_null": rollout_null_count,
            "rollout_equal_low": int((rollout & (score_array == low)).sum()),
            "rollout_equal_high": int((rollout & (score_array == high)).sum()),
        },
        "fractions": {
            "positive": _fraction(positive_count, total_count),
            "negative": _fraction(negative_count, total_count),
            "null": _fraction(null_count, total_count),
            "rollout_positive": _fraction(rollout_positive_count, rollout_count),
            "rollout_negative": _fraction(rollout_negative_count, rollout_count),
            "rollout_null": _fraction(rollout_null_count, rollout_count),
        },
        "score_stats": {
            "all": _score_stats(score_array),
            "human": _score_stats(score_array[human]),
            "rollout": _score_stats(score_array[rollout]),
            "rollout_positive": _score_stats(score_array[rollout_positive]),
            "rollout_negative": _score_stats(score_array[rollout_negative]),
            "rollout_null": _score_stats(score_array[rollout_null]),
        },
    }
    return {
        "actor_condition": actor_condition,
        "actor_condition_mask": actor_condition_mask,
        "mode": normalized_mode,
        "low_threshold": low,
        "high_threshold": high,
        "audit": audit,
    }


def make_critic_conditions(
    scores: Any,
    is_human: Any,
    *,
    mode: str,
    threshold_mode: str,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    low_quantile: float | None = None,
    high_quantile: float | None = None,
    rollout_success: Any | None = None,
    target_purity: float | None = None,
    min_tail_count: int | None = None,
    eligible: Any | None = None,
) -> dict[str, Any]:
    """Calibrate thresholds and construct conditions in one call."""

    calibration = calibrate_critic_condition_thresholds(
        scores,
        is_human,
        mode=mode,
        threshold_mode=threshold_mode,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        rollout_success=rollout_success,
        target_purity=target_purity,
        min_tail_count=min_tail_count,
        eligible=eligible,
    )
    result = construct_critic_conditions(
        scores,
        is_human,
        mode=mode,
        low_threshold=calibration["low_threshold"],
        high_threshold=calibration["high_threshold"],
    )
    result["threshold_mode"] = calibration["threshold_mode"]
    result["calibration"] = calibration
    return result


__all__ = [
    "SUPPORTED_CRITIC_CONDITION_MODES",
    "SUPPORTED_THRESHOLD_MODES",
    "calibrate_critic_condition_thresholds",
    "construct_critic_conditions",
    "make_critic_conditions",
    "validate_critic_condition_mode",
]
