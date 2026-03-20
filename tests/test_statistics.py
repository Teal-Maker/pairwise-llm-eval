"""Tests for statistical analysis functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from pairwise_llm_eval.statistics import (
    _bh_fdr_correction,
    _wilcoxon_pair,
    bootstrap_ci,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Minimal QuestionResult stub (no real model calls)
# ---------------------------------------------------------------------------

@dataclass
class _FakeResult:
    """Minimal stub that satisfies the fields accessed by compute_metrics."""

    question_id: str
    question_type: str
    question_text: str
    gold_answer: str
    category: str
    model_a_response: dict[str, Any]
    model_b_response: dict[str, Any]
    judge: dict[str, Any]
    model_a_score: int
    model_b_score: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_result(
    a_score: int, b_score: int, category: str = "cat", q_id: str = "q1"
) -> _FakeResult:
    resp = {"latency_ms": 100.0, "token_count": 50, "is_refusal": False, "text": "ok"}
    return _FakeResult(
        question_id=q_id,
        question_type="domain",
        question_text="Q?",
        gold_answer="A",
        category=category,
        model_a_response=resp,
        model_b_response=resp,
        judge={},
        model_a_score=a_score,
        model_b_score=b_score,
    )


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_empty_data(self) -> None:
        point, lo, hi = bootstrap_ci(np.array([]))
        assert (point, lo, hi) == (0.0, 0.0, 0.0)

    def test_single_value(self) -> None:
        point, lo, hi = bootstrap_ci(np.array([3.0]))
        assert point == 3.0
        assert lo <= point <= hi

    def test_ci_contains_true_median(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.integers(1, 6, size=100).astype(float)
        point, lo, hi = bootstrap_ci(data, func=np.median, seed=42)
        true_median = float(np.median(data))
        assert lo <= true_median <= hi

    def test_ci_width_positive(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0])
        _, lo, hi = bootstrap_ci(data)
        assert hi > lo

    def test_mean_function(self) -> None:
        data = np.array([2.0, 4.0, 6.0])
        point, _, _ = bootstrap_ci(data, func=np.mean)
        assert abs(point - 4.0) < 1e-9

    def test_seed_reproducibility(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r1 = bootstrap_ci(data, seed=7)
        r2 = bootstrap_ci(data, seed=7)
        assert r1 == r2


# ---------------------------------------------------------------------------
# BH FDR correction
# ---------------------------------------------------------------------------

class TestBHFDR:
    def test_empty(self) -> None:
        assert _bh_fdr_correction([]) == []

    def test_single(self) -> None:
        result = _bh_fdr_correction([0.05])
        assert len(result) == 1
        assert 0.0 <= result[0] <= 1.0

    def test_adjusted_ge_raw(self) -> None:
        # BH can only increase p-values, never decrease them
        raw = [0.001, 0.01, 0.05, 0.1, 0.5]
        adjusted = _bh_fdr_correction(raw)
        for r, a in zip(raw, adjusted):
            assert a >= r - 1e-12  # small float tolerance

    def test_all_ones_remain_ones(self) -> None:
        result = _bh_fdr_correction([1.0, 1.0, 1.0])
        assert all(v == pytest.approx(1.0) for v in result)

    def test_clipped_at_one(self) -> None:
        result = _bh_fdr_correction([0.9, 0.9, 0.9])
        assert all(v <= 1.0 for v in result)


# ---------------------------------------------------------------------------
# Wilcoxon helper
# ---------------------------------------------------------------------------

class TestWilcoxonPair:
    def test_all_ties_returns_p1(self) -> None:
        a = np.array([3, 3, 3])
        stat, p = _wilcoxon_pair(a, a)
        assert stat == 0.0
        assert p == 1.0

    def test_clear_difference(self) -> None:
        a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        b = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=float)
        _, p = _wilcoxon_pair(a, b)
        assert p < 0.01

    def test_returns_floats(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        stat, p = _wilcoxon_pair(a, b)
        assert isinstance(stat, float)
        assert isinstance(p, float)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _make_domain_results(self, n: int = 30) -> list[_FakeResult]:
        rng = np.random.default_rng(42)
        results = []
        cats = ["cat_A", "cat_B", "cat_C"]
        for i in range(n):
            a = int(rng.integers(1, 6))
            b = int(rng.integers(1, 6))
            results.append(_make_result(a, b, category=cats[i % 3], q_id=f"d{i}"))
        return results

    def _make_general_results(self, n: int = 10) -> list[_FakeResult]:
        rng = np.random.default_rng(0)
        results = []
        for i in range(n):
            a = int(rng.integers(1, 6))
            b = int(rng.integers(1, 6))
            r = _make_result(a, b, category="science", q_id=f"g{i}")
            r.question_type = "general"
            results.append(r)
        return results

    def test_returns_timestamp(self) -> None:
        m = compute_metrics([], [])
        assert "timestamp" in m

    def test_counts(self) -> None:
        dom = self._make_domain_results(20)
        gen = self._make_general_results(5)
        m = compute_metrics(dom, gen)
        assert m["domain_count"] == 20
        assert m["general_count"] == 5

    def test_domain_keys_present(self) -> None:
        dom = self._make_domain_results(30)
        m = compute_metrics(dom, [])
        assert "domain" in m
        d = m["domain"]
        for key in (
            "model_a_mean_score", "model_b_mean_score",
            "wilcoxon_p_value", "model_b_wins", "model_a_wins", "ties",
            "model_b_win_rate", "model_b_win_rate_ci_95",
            "model_a_refusal_rate", "model_b_refusal_rate",
        ):
            assert key in d, f"Missing key: {key}"

    def test_general_keys_present(self) -> None:
        gen = self._make_general_results(10)
        m = compute_metrics([], gen)
        assert "general" in m
        assert "forgetting_flag" in m["general"]

    def test_forgetting_flag_triggered(self) -> None:
        # model_b consistently 2 pts below model_a — forgetting_flag should be True
        results = [_make_result(5, 3, q_id=f"g{i}") for i in range(10)]
        for r in results:
            r.question_type = "general"
        m = compute_metrics([], results)
        assert m["general"]["forgetting_flag"] is True

    def test_forgetting_flag_not_triggered(self) -> None:
        results = [_make_result(3, 3, q_id=f"g{i}") for i in range(10)]
        for r in results:
            r.question_type = "general"
        m = compute_metrics([], results)
        assert m["general"]["forgetting_flag"] is False

    def test_by_category_present(self) -> None:
        # 25 samples in each of 2 categories
        cats = ["alpha"] * 25 + ["beta"] * 25
        dom = [_make_result(3, 4, category=c, q_id=f"d{i}") for i, c in enumerate(cats)]
        m = compute_metrics(dom, [], min_cell_size=20)
        assert "domain_by_category" in m
        by_cat = m["domain_by_category"]
        assert "alpha" in by_cat
        assert "beta" in by_cat

    def test_small_categories_go_to_other(self) -> None:
        # 25 in main + 3 in tiny — tiny should end up in OTHER
        dom = (
            [_make_result(3, 3, category="big", q_id=f"d{i}") for i in range(25)]
            + [_make_result(3, 3, category="tiny", q_id=f"t{i}") for i in range(3)]
        )
        m = compute_metrics(dom, [], min_cell_size=20)
        by_cat = m["domain_by_category"]
        assert "tiny" not in by_cat
        # OTHER only appears if the combined small group >= min_cell_size; 3 < 20 so no OTHER
        assert "OTHER" not in by_cat

    def test_position_bias_from_metadata(self) -> None:
        r = _make_result(4, 3, q_id="pb1")
        r.metadata = {
            "swap_adjudication": {
                "swap_model_a_score": 3,
                "swap_model_b_score": 4,
                "model_a_disagree": 1,
                "model_b_disagree": 1,
            }
        }
        dom = [r] + [_make_result(3, 3, q_id=f"d{i}") for i in range(9)]
        m = compute_metrics(dom, [])
        assert "position_bias" in m
        pb = m["position_bias"]
        assert pb["n_checked"] == 1

    def test_empty_inputs(self) -> None:
        m = compute_metrics([], [])
        assert m["domain_count"] == 0
        assert m["general_count"] == 0
        assert "domain" not in m
        assert "general" not in m

    def test_scores_clamped_zero_excluded(self) -> None:
        # Score of 0 means parse failure — excluded from mean calculations
        dom = [_make_result(0, 0, q_id="bad")] + [
            _make_result(3, 4, q_id=f"ok{i}") for i in range(10)
        ]
        m = compute_metrics(dom, [])
        # Mean should only reflect the 10 valid pairs, not the 0-score one
        assert m["domain"]["model_a_mean_score"] == pytest.approx(3.0)
        assert m["domain"]["model_b_mean_score"] == pytest.approx(4.0)
