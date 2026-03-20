"""Shared pytest fixtures for pairwise-llm-eval tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pairwise_llm_eval.providers.base import Question


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_questions() -> list[Question]:
    """Ten sample domain questions for testing."""
    return [
        Question(
            id=f"q{i:03d}",
            question=f"Sample question {i}?",
            gold_answer=f"Answer {i}",
            category=["science", "technology", "math"][i % 3],
        )
        for i in range(10)
    ]


@pytest.fixture()
def sample_jsonl_path() -> Path:
    return FIXTURES_DIR / "sample_questions.jsonl"


@pytest.fixture()
def general_questions_subset() -> list[dict[str, str]]:
    """Small subset of general questions for fast tests."""
    return [
        {"category": "science", "question": "What is H2O?", "answer": "Water"},
        {"category": "math", "question": "What is 2+2?", "answer": "4"},
        {"category": "history", "question": "When did WWII end?", "answer": "1945"},
    ]
