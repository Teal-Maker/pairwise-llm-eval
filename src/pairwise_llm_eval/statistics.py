"""Statistical analysis for pairwise evaluation results.

Provides bootstrap confidence intervals, Wilcoxon signed-rank tests,
Benjamini-Hochberg FDR correction, and per-category/per-area breakdowns.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from .assessor import QuestionResult


def bootstrap_ci(
    data: np.ndarray,
    func: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 10_000,
    ci: float = 0.95,
    *,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for a summary statistic.

    Args:
        data: 1-D numeric array of observations.
        func: Statistic to estimate (default: ``numpy.median``).
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (0–1).
        seed: Seed for the bootstrap RNG (for reproducibility).

    Returns:
        A ``(point_estimate, ci_low, ci_high)`` triple.

    Example::

        diffs = np.array([1, -1, 2, 0, 1])
        point, lo, hi = bootstrap_ci(diffs)
    """
    if len(data) == 0:
        return 0.0, 0.0, 0.0

    point = float(func(data))
    rng = np.random.default_rng(seed)
    boot = np.array(
        [func(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    low = float(np.percentile(boot, alpha * 100))
    high = float(np.percentile(boot, (1 - alpha) * 100))
    return point, low, high


def _bh_fdr_correction(p_values: list[float]) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: Raw p-values in the same order as the hypotheses.

    Returns:
        FDR-adjusted p-values (same order as input), clipped to [0, 1].
    """
    m = len(p_values)
    if m == 0:
        return []

    p_arr = np.array(p_values, dtype=float)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    # Compute adjusted p-values in sorted order
    adjusted_sorted = np.zeros(m)
    for i in range(m):
        adjusted_sorted[i] = sorted_p[i] * m / (i + 1)

    # Enforce monotonicity (step-up) in sorted order
    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    # Clip and map back to original order
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    result = np.empty(m)
    for i, orig_i in enumerate(sorted_indices):
        result[orig_i] = adjusted_sorted[i]

    return result.tolist()


def _wilcoxon_pair(
    a: np.ndarray, b: np.ndarray
) -> tuple[float, float]:
    """Run a two-sided Wilcoxon signed-rank test on paired arrays.

    Returns ``(statistic, p_value)``.  Returns ``(0.0, 1.0)`` when there
    are no non-zero differences (all ties).
    """
    diff = b - a
    if len(diff) > 0 and np.any(diff != 0):
        stat, p_val = stats.wilcoxon(b, a, alternative="two-sided")
        return float(stat), float(p_val)
    return 0.0, 1.0


def compute_metrics(
    domain_results: list[QuestionResult],
    general_results: list[QuestionResult],
    *,
    min_cell_size: int = 20,
    domain_label: str = "domain",
) -> dict[str, Any]:
    """Compute all evaluation metrics with statistical tests.

    Args:
        domain_results: Results from the domain question assessment.
        general_results: Results from the general knowledge assessment.
        min_cell_size: Minimum number of valid pairs needed to report a
                       per-category breakdown row.  Smaller groups are
                       aggregated into an ``"OTHER"`` bucket.
        domain_label: Key prefix used for domain metrics in the output dict
                      (default ``"domain"``).

    Returns:
        A nested dict suitable for JSON serialisation, containing aggregate
        metrics, per-area/per-category breakdowns, and position-bias data.
    """
    metrics: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domain_count": len(domain_results),
        "general_count": len(general_results),
    }

    # ------------------------------------------------------------------ #
    # Domain aggregate                                                     #
    # ------------------------------------------------------------------ #
    if domain_results:
        base_scores_all = np.array(
            [r.model_a_score for r in domain_results if r.model_a_score > 0]
        )
        b_scores_all = np.array(
            [r.model_b_score for r in domain_results if r.model_b_score > 0]
        )

        # Paired valid scores only
        paired_a, paired_b = zip(
            *[
                (r.model_a_score, r.model_b_score)
                for r in domain_results
                if r.model_a_score > 0 and r.model_b_score > 0
            ]
        ) if any(r.model_a_score > 0 and r.model_b_score > 0 for r in domain_results) else ([], [])

        paired_a_arr = np.array(paired_a)
        paired_b_arr = np.array(paired_b)
        diff = paired_b_arr - paired_a_arr

        stat, p_val = _wilcoxon_pair(paired_a_arr, paired_b_arr)

        b_wins = int(np.sum(diff > 0))
        a_wins = int(np.sum(diff < 0))
        ties = int(np.sum(diff == 0))

        med_diff, med_lo, med_hi = bootstrap_ci(diff, func=np.median)
        win_arr = (diff > 0).astype(float)
        wr_point, wr_lo, wr_hi = bootstrap_ci(win_arr, func=np.mean)

        lats_a = [r.model_a_response["latency_ms"] for r in domain_results]
        lats_b = [r.model_b_response["latency_ms"] for r in domain_results]
        ref_a = sum(1 for r in domain_results if r.model_a_response["is_refusal"])
        ref_b = sum(1 for r in domain_results if r.model_b_response["is_refusal"])
        tok_a = [r.model_a_response["token_count"] for r in domain_results]
        tok_b = [r.model_b_response["token_count"] for r in domain_results]

        n_dom = len(domain_results)
        metrics[domain_label] = {
            "model_a_mean_score": float(np.mean(base_scores_all)) if len(base_scores_all) else 0.0,
            "model_b_mean_score": float(np.mean(b_scores_all)) if len(b_scores_all) else 0.0,
            "model_a_median_score": float(np.median(base_scores_all)) if len(base_scores_all) else 0.0,
            "model_b_median_score": float(np.median(b_scores_all)) if len(b_scores_all) else 0.0,
            "median_diff": med_diff,
            "median_diff_ci_95": [med_lo, med_hi],
            "wilcoxon_statistic": stat,
            "wilcoxon_p_value": p_val,
            "model_b_wins": b_wins,
            "model_a_wins": a_wins,
            "ties": ties,
            "model_b_win_rate": wr_point,
            "model_b_win_rate_ci_95": [wr_lo, wr_hi],
            "model_a_refusal_rate": ref_a / n_dom,
            "model_b_refusal_rate": ref_b / n_dom,
            "model_a_latency_p50": float(np.median(lats_a)),
            "model_a_latency_p95": float(np.percentile(lats_a, 95)),
            "model_b_latency_p50": float(np.median(lats_b)),
            "model_b_latency_p95": float(np.percentile(lats_b, 95)),
            "model_a_mean_tokens": float(np.mean(tok_a)),
            "model_b_mean_tokens": float(np.mean(tok_b)),
        }

        # Per-area breakdown with BH-FDR correction
        area_results: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for r in domain_results:
            if r.model_a_score > 0 and r.model_b_score > 0:
                area_results[r.category].append((r.model_a_score, r.model_b_score))

        area_metrics: dict[str, Any] = {}
        p_values: list[float] = []
        area_keys: list[str] = []

        for area, pairs in sorted(area_results.items()):
            if len(pairs) < min_cell_size:
                continue

            ab = np.array(pairs)
            a_base, a_b = ab[:, 0], ab[:, 1]
            a_diff = a_b - a_base

            _, a_p = _wilcoxon_pair(a_base, a_b)
            a_med, a_lo, a_hi = bootstrap_ci(a_diff, func=np.median)

            area_metrics[area] = {
                "n": len(pairs),
                "model_a_mean": float(np.mean(a_base)),
                "model_b_mean": float(np.mean(a_b)),
                "median_diff": a_med,
                "median_diff_ci_95": [a_lo, a_hi],
                "model_b_wins": int(np.sum(a_diff > 0)),
                "model_a_wins": int(np.sum(a_diff < 0)),
                "ties": int(np.sum(a_diff == 0)),
                "p_value_raw": float(a_p),
            }
            p_values.append(a_p)
            area_keys.append(area)

        if p_values:
            adjusted = _bh_fdr_correction(p_values)
            for idx, area in enumerate(area_keys):
                area_metrics[area]["p_value_fdr"] = adjusted[idx]

        # Aggregate small areas into "OTHER"
        small_areas = [a for a in area_results if len(area_results[a]) < min_cell_size]
        if small_areas:
            other_pairs: list[tuple[int, int]] = []
            for a in small_areas:
                other_pairs.extend(area_results[a])
            if len(other_pairs) >= min_cell_size:
                ob = np.array(other_pairs)
                o_diff = ob[:, 1] - ob[:, 0]
                o_med, o_lo, o_hi = bootstrap_ci(o_diff, func=np.median)
                area_metrics["OTHER"] = {
                    "n": len(other_pairs),
                    "areas_included": small_areas,
                    "model_a_mean": float(np.mean(ob[:, 0])),
                    "model_b_mean": float(np.mean(ob[:, 1])),
                    "median_diff": o_med,
                    "median_diff_ci_95": [o_lo, o_hi],
                    "model_b_wins": int(np.sum(o_diff > 0)),
                    "model_a_wins": int(np.sum(o_diff < 0)),
                    "ties": int(np.sum(o_diff == 0)),
                }

        metrics[f"{domain_label}_by_category"] = area_metrics

    # ------------------------------------------------------------------ #
    # General knowledge aggregate                                          #
    # ------------------------------------------------------------------ #
    if general_results:
        base_scores_g = np.array(
            [r.model_a_score for r in general_results if r.model_a_score > 0]
        )
        b_scores_g = np.array(
            [r.model_b_score for r in general_results if r.model_b_score > 0]
        )

        paired_ga, paired_gb = zip(
            *[
                (r.model_a_score, r.model_b_score)
                for r in general_results
                if r.model_a_score > 0 and r.model_b_score > 0
            ]
        ) if any(r.model_a_score > 0 and r.model_b_score > 0 for r in general_results) else ([], [])

        paired_ga_arr = np.array(paired_ga)
        paired_gb_arr = np.array(paired_gb)
        g_diff = paired_gb_arr - paired_ga_arr

        _, g_p = _wilcoxon_pair(paired_ga_arr, paired_gb_arr)
        g_med, g_lo, g_hi = bootstrap_ci(g_diff, func=np.median)

        metrics["general"] = {
            "model_a_mean_score": float(np.mean(base_scores_g)) if len(base_scores_g) else 0.0,
            "model_b_mean_score": float(np.mean(b_scores_g)) if len(b_scores_g) else 0.0,
            "median_diff": g_med,
            "median_diff_ci_95": [g_lo, g_hi],
            "wilcoxon_p_value": g_p,
            "model_b_wins": int(np.sum(g_diff > 0)),
            "model_a_wins": int(np.sum(g_diff < 0)),
            "ties": int(np.sum(g_diff == 0)),
            "forgetting_flag": bool(
                len(b_scores_g) > 0
                and len(base_scores_g) > 0
                and float(np.mean(b_scores_g)) < float(np.mean(base_scores_g)) - 0.5
            ),
        }

        cat_results: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for r in general_results:
            if r.model_a_score > 0 and r.model_b_score > 0:
                cat_results[r.category].append((r.model_a_score, r.model_b_score))

        cat_metrics: dict[str, Any] = {}
        for cat, pairs in sorted(cat_results.items()):
            cb = np.array(pairs)
            cat_metrics[cat] = {
                "n": len(pairs),
                "model_a_mean": float(np.mean(cb[:, 0])),
                "model_b_mean": float(np.mean(cb[:, 1])),
                "diff_mean": float(np.mean(cb[:, 1] - cb[:, 0])),
            }
        metrics["general_by_category"] = cat_metrics

    # ------------------------------------------------------------------ #
    # Position bias                                                        #
    # ------------------------------------------------------------------ #
    bias_records = []
    for r in domain_results:
        swap = r.metadata.get("swap_adjudication") if r.metadata else None
        if swap:
            bias_records.append({
                "question_id": r.question_id,
                "primary_model_a": r.model_a_score,
                "primary_model_b": r.model_b_score,
                "swap_model_a": swap["swap_model_a_score"],
                "swap_model_b": swap["swap_model_b_score"],
                "model_a_disagree": swap["model_a_disagree"],
                "model_b_disagree": swap["model_b_disagree"],
            })

    if bias_records:
        disagrees = [
            1
            for b in bias_records
            if b["model_a_disagree"] > 1 or b["model_b_disagree"] > 1
        ]
        metrics["position_bias"] = {
            "n_checked": len(bias_records),
            "n_disagree_gt1": len(disagrees),
            "disagree_rate": len(disagrees) / len(bias_records),
            "details": bias_records,
        }

    return metrics
