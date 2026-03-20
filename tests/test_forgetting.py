"""Tests for the general knowledge question loader and sampler."""

from __future__ import annotations

import pytest

from pairwise_llm_eval.forgetting import (
    load_default_general_questions,
    sample_general_questions,
)


class TestLoadDefaultGeneralQuestions:
    def test_returns_100_questions(self) -> None:
        qs = load_default_general_questions()
        assert len(qs) == 100

    def test_all_have_required_keys(self) -> None:
        qs = load_default_general_questions()
        for q in qs:
            assert "category" in q
            assert "question" in q
            assert "answer" in q

    def test_has_10_categories(self) -> None:
        qs = load_default_general_questions()
        categories = {q["category"] for q in qs}
        assert len(categories) == 10

    def test_10_questions_per_category(self) -> None:
        from collections import Counter
        qs = load_default_general_questions()
        counts = Counter(q["category"] for q in qs)
        for cat, count in counts.items():
            assert count == 10, f"Category {cat!r} has {count} questions, expected 10"

    def test_no_empty_questions(self) -> None:
        qs = load_default_general_questions()
        for q in qs:
            assert q["question"].strip()
            assert q["answer"].strip()

    def test_expected_categories_present(self) -> None:
        qs = load_default_general_questions()
        cats = {q["category"] for q in qs}
        for expected in (
            "science", "history", "geography", "math_logic", "common_sense",
            "language", "technology", "economics", "arts_literature", "health_biology",
        ):
            assert expected in cats


class TestSampleGeneralQuestions:
    def test_returns_requested_count(self) -> None:
        result = sample_general_questions(30)
        assert len(result) == 30

    def test_returns_all_when_n_gte_pool(self) -> None:
        result = sample_general_questions(200)
        assert len(result) == 100

    def test_reproducible_with_seed(self) -> None:
        r1 = sample_general_questions(20, seed=0)
        r2 = sample_general_questions(20, seed=0)
        assert [q["question"] for q in r1] == [q["question"] for q in r2]

    def test_different_seeds_differ(self) -> None:
        r1 = sample_general_questions(50, seed=0)
        r2 = sample_general_questions(50, seed=99)
        assert [q["question"] for q in r1] != [q["question"] for q in r2]

    def test_stratified_covers_categories(self) -> None:
        result = sample_general_questions(10)  # 1 per category
        cats = {q["category"] for q in result}
        assert len(cats) == 10

    def test_stratified_30_questions(self) -> None:
        result = sample_general_questions(30)
        from collections import Counter
        counts = Counter(q["category"] for q in result)
        # 10 categories, 30 questions: each category gets 3
        for cat, count in counts.items():
            assert count == 3, f"Category {cat!r} count {count}, expected 3"

    def test_zero_returns_empty(self) -> None:
        # The function handles n >= pool size, but n=0 should also work cleanly
        result = sample_general_questions(0)
        assert result == []
