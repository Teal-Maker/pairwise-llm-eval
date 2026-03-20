"""Position bias detection for pairwise evaluation.

Position bias occurs when an LLM judge's scores are influenced by whether
a response appears as "Answer 1" or "Answer 2" in the prompt, independently
of actual quality.  This module provides utilities to detect and report that
bias by re-judging a subset of questions with the presentation order swapped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .assessor import QuestionResult


def compute_position_bias(results: list[QuestionResult]) -> dict[str, Any]:
    """Compute position bias statistics from swap-adjudicated results.

    Questions that received swap-order adjudication have a
    ``"swap_adjudication"`` key in their ``metadata`` dict.  For those
    questions this function compares the primary-order scores against the
    swapped-order scores and flags cases where the scores changed by more
    than 1 point (indicating the judge is sensitive to presentation order).

    Args:
        results: All domain question results, including those with
                 ``swap_adjudication`` metadata.

    Returns:
        A dict with:

        - ``n_checked`` — number of questions that were swap-adjudicated
        - ``n_disagree_gt1`` — count with disagreement > 1 point on either model
        - ``disagree_rate`` — ``n_disagree_gt1 / n_checked`` (0.0 if none checked)
        - ``details`` — per-question breakdown list
        - ``bias_flag`` — ``True`` if ``disagree_rate > 0.10``

    Example::

        stats = compute_position_bias(domain_results)
        if stats["bias_flag"]:
            print(f"Position bias detected: {stats['disagree_rate']:.1%}")
    """
    records: list[dict[str, Any]] = []

    for r in results:
        swap = r.metadata.get("swap_adjudication") if r.metadata else None
        if not swap:
            continue
        records.append({
            "question_id": r.question_id,
            "primary_model_a": r.model_a_score,
            "primary_model_b": r.model_b_score,
            "swap_model_a": swap["swap_model_a_score"],
            "swap_model_b": swap["swap_model_b_score"],
            "model_a_disagree": swap["model_a_disagree"],
            "model_b_disagree": swap["model_b_disagree"],
        })

    if not records:
        return {
            "n_checked": 0,
            "n_disagree_gt1": 0,
            "disagree_rate": 0.0,
            "details": [],
            "bias_flag": False,
        }

    n_disagree = sum(
        1
        for b in records
        if b["model_a_disagree"] > 1 or b["model_b_disagree"] > 1
    )
    rate = n_disagree / len(records)

    return {
        "n_checked": len(records),
        "n_disagree_gt1": n_disagree,
        "disagree_rate": rate,
        "details": records,
        "bias_flag": rate > 0.10,
    }


def select_bias_subset(n_total: int, subset_size: int, rng_seed: int = 42) -> set[int]:
    """Select indices for swap-order adjudication.

    Args:
        n_total: Total number of questions in the evaluation run.
        subset_size: How many questions to select for swap adjudication.
        rng_seed: Seed for reproducible selection.

    Returns:
        A set of integer indices (0-based) that should receive swap
        adjudication.
    """
    import random

    rng = random.Random(rng_seed)
    population = list(range(n_total))
    k = min(subset_size, n_total)
    return set(rng.sample(population, k))
