"""Tests for position bias detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pairwise_llm_eval.bias import compute_position_bias, select_bias_subset


@dataclass
class _FakeResult:
    question_id: str
    model_a_score: int
    model_b_score: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TestSelectBiasSubset:
    def test_returns_set(self) -> None:
        result = select_bias_subset(100, 50, rng_seed=0)
        assert isinstance(result, set)

    def test_size_capped_at_n_total(self) -> None:
        result = select_bias_subset(10, 50, rng_seed=0)
        assert len(result) == 10

    def test_size_matches_subset(self) -> None:
        result = select_bias_subset(100, 30, rng_seed=0)
        assert len(result) == 30

    def test_indices_in_range(self) -> None:
        n = 100
        result = select_bias_subset(n, 50, rng_seed=7)
        assert all(0 <= i < n for i in result)

    def test_reproducible(self) -> None:
        r1 = select_bias_subset(100, 50, rng_seed=42)
        r2 = select_bias_subset(100, 50, rng_seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self) -> None:
        r1 = select_bias_subset(100, 50, rng_seed=1)
        r2 = select_bias_subset(100, 50, rng_seed=2)
        assert r1 != r2

    def test_zero_total(self) -> None:
        result = select_bias_subset(0, 50, rng_seed=0)
        assert result == set()


class TestComputePositionBias:
    def _make_swap(
        self, a_disagree: int = 0, b_disagree: int = 0
    ) -> dict[str, Any]:
        return {
            "swap_model_a_score": 3,
            "swap_model_b_score": 3,
            "model_a_disagree": a_disagree,
            "model_b_disagree": b_disagree,
        }

    def test_no_swap_results(self) -> None:
        results = [_FakeResult(question_id="q1", model_a_score=3, model_b_score=4)]
        out = compute_position_bias(results)
        assert out["n_checked"] == 0
        assert out["disagree_rate"] == 0.0
        assert out["bias_flag"] is False

    def test_empty_results(self) -> None:
        out = compute_position_bias([])
        assert out == {
            "n_checked": 0,
            "n_disagree_gt1": 0,
            "disagree_rate": 0.0,
            "details": [],
            "bias_flag": False,
        }

    def test_low_disagree_rate_no_flag(self) -> None:
        # 1 disagree out of 20 = 5%, below 10% threshold
        results = [
            _FakeResult(
                question_id=f"q{i}",
                model_a_score=3,
                model_b_score=4,
                metadata={"swap_adjudication": self._make_swap(0, 0)},
            )
            for i in range(19)
        ] + [
            _FakeResult(
                question_id="q_dis",
                model_a_score=3,
                model_b_score=4,
                metadata={"swap_adjudication": self._make_swap(2, 0)},
            )
        ]
        out = compute_position_bias(results)
        assert out["n_checked"] == 20
        assert out["n_disagree_gt1"] == 1
        assert out["disagree_rate"] == pytest.approx(0.05)
        assert out["bias_flag"] is False

    def test_high_disagree_rate_sets_flag(self) -> None:
        # All 10 have large disagreement
        results = [
            _FakeResult(
                question_id=f"q{i}",
                model_a_score=3,
                model_b_score=4,
                metadata={"swap_adjudication": self._make_swap(2, 0)},
            )
            for i in range(10)
        ]
        out = compute_position_bias(results)
        assert out["bias_flag"] is True
        assert out["disagree_rate"] == pytest.approx(1.0)

    def test_details_list_populated(self) -> None:
        results = [
            _FakeResult(
                question_id="q1",
                model_a_score=4,
                model_b_score=3,
                metadata={"swap_adjudication": self._make_swap(1, 0)},
            )
        ]
        out = compute_position_bias(results)
        assert len(out["details"]) == 1
        detail = out["details"][0]
        assert detail["question_id"] == "q1"
        assert "model_a_disagree" in detail
        assert "model_b_disagree" in detail

    def test_threshold_boundary_exactly_10_pct(self) -> None:
        # Exactly 1/10 = 10%, not strictly > 10%, so no flag
        results = [
            _FakeResult(
                question_id=f"q{i}",
                model_a_score=3,
                model_b_score=3,
                metadata={"swap_adjudication": self._make_swap(0, 0)},
            )
            for i in range(9)
        ] + [
            _FakeResult(
                question_id="q_dis",
                model_a_score=3,
                model_b_score=3,
                metadata={"swap_adjudication": self._make_swap(2, 0)},
            )
        ]
        out = compute_position_bias(results)
        assert out["disagree_rate"] == pytest.approx(0.10)
        assert out["bias_flag"] is False  # strictly > 0.10 required
