"""Tests for question provider implementations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pairwise_llm_eval.providers.base import Question, QuestionProvider
from pairwise_llm_eval.providers.jsonl import JSONLProvider
from pairwise_llm_eval.providers.memory import MemoryProvider


class TestQuestion:
    def test_construction_minimal(self) -> None:
        q = Question(id="1", question="Q?", gold_answer="A", category="cat")
        assert q.id == "1"
        assert q.metadata is None

    def test_construction_with_metadata(self) -> None:
        q = Question(id="2", question="Q?", gold_answer="A", category="cat", metadata={"k": "v"})
        assert q.metadata == {"k": "v"}


class TestQuestionProviderProtocol:
    def test_memory_provider_is_protocol_instance(self, sample_questions) -> None:
        provider = MemoryProvider(sample_questions)
        assert isinstance(provider, QuestionProvider)

    def test_jsonl_provider_is_protocol_instance(self, sample_jsonl_path) -> None:
        provider = JSONLProvider(sample_jsonl_path)
        assert isinstance(provider, QuestionProvider)


class TestMemoryProvider:
    def test_sample_returns_requested_count(self, sample_questions) -> None:
        provider = MemoryProvider(sample_questions, seed=0)
        result = provider.sample(5)
        assert len(result) == 5

    def test_sample_does_not_exceed_pool(self, sample_questions) -> None:
        provider = MemoryProvider(sample_questions, seed=0)
        result = provider.sample(1000)
        assert len(result) == len(sample_questions)

    def test_sample_returns_question_instances(self, sample_questions) -> None:
        provider = MemoryProvider(sample_questions, seed=0)
        for q in provider.sample(3):
            assert isinstance(q, Question)

    def test_seed_reproducibility(self, sample_questions) -> None:
        p1 = MemoryProvider(sample_questions, seed=7)
        p2 = MemoryProvider(sample_questions, seed=7)
        assert [q.id for q in p1.sample(5)] == [q.id for q in p2.sample(5)]

    def test_different_seeds_differ(self, sample_questions) -> None:
        p1 = MemoryProvider(sample_questions, seed=1)
        p2 = MemoryProvider(sample_questions, seed=99)
        # With 10 items it's vanishingly unlikely they produce the same order
        assert [q.id for q in p1.sample(10)] != [q.id for q in p2.sample(10)]

    def test_empty_pool(self) -> None:
        provider = MemoryProvider([], seed=0)
        assert provider.sample(5) == []


class TestJSONLProvider:
    def test_loads_fixture_file(self, sample_jsonl_path) -> None:
        provider = JSONLProvider(sample_jsonl_path, stratified=False)
        assert len(provider) == 10

    def test_sample_count(self, sample_jsonl_path) -> None:
        provider = JSONLProvider(sample_jsonl_path, seed=0)
        result = provider.sample(5)
        assert len(result) == 5

    def test_sample_all(self, sample_jsonl_path) -> None:
        provider = JSONLProvider(sample_jsonl_path, seed=0)
        result = provider.sample(100)
        assert len(result) == 10  # pool is 10

    def test_question_fields(self, sample_jsonl_path) -> None:
        provider = JSONLProvider(sample_jsonl_path)
        qs = provider.sample(1)
        q = qs[0]
        assert q.id
        assert q.question
        assert q.gold_answer
        assert q.category

    def test_extra_fields_in_metadata(self) -> None:
        data = [
            {"id": "1", "question": "Q?", "gold_answer": "A", "category": "c", "difficulty": "hard"}
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for obj in data:
                f.write(json.dumps(obj) + "\n")
            tmp_path = Path(f.name)
        try:
            provider = JSONLProvider(tmp_path)
            qs = provider.sample(1)
            assert qs[0].metadata == {"difficulty": "hard"}
        finally:
            tmp_path.unlink()

    def test_missing_required_field_raises(self) -> None:
        data = [{"id": "1", "question": "Q?", "category": "c"}]  # no gold_answer
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(data[0]) + "\n")
            tmp_path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="missing required field"):
                JSONLProvider(tmp_path)
        finally:
            tmp_path.unlink()

    def test_invalid_json_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not valid json\n")
            tmp_path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="invalid JSON"):
                JSONLProvider(tmp_path)
        finally:
            tmp_path.unlink()

    def test_stratified_sampling_covers_categories(self, sample_jsonl_path) -> None:
        provider = JSONLProvider(sample_jsonl_path, stratified=True, seed=0)
        result = provider.sample(6)
        cats = {q.category for q in result}
        # fixture has 5 categories; with 6 samples we expect >= 3
        assert len(cats) >= 3

    def test_uniform_sampling(self, sample_jsonl_path) -> None:
        provider = JSONLProvider(sample_jsonl_path, stratified=False, seed=42)
        result = provider.sample(5)
        assert len(result) == 5

    def test_all_questions_property(self, sample_jsonl_path) -> None:
        provider = JSONLProvider(sample_jsonl_path)
        assert len(provider.all_questions) == 10

    def test_blank_lines_ignored(self) -> None:
        data = [
            {"id": "1", "question": "Q?", "gold_answer": "A", "category": "c"},
            {"id": "2", "question": "Q2?", "gold_answer": "B", "category": "c"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(data[0]) + "\n")
            f.write("\n")  # blank line
            f.write(json.dumps(data[1]) + "\n")
            tmp_path = Path(f.name)
        try:
            provider = JSONLProvider(tmp_path)
            assert len(provider) == 2
        finally:
            tmp_path.unlink()
