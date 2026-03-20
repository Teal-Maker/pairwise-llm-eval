"""Tests for the PairwiseAssessor and low-level model helpers.

All network calls are intercepted with the ``responses`` library so no
real inference server is needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pairwise_llm_eval.assessor import (
    REFUSAL_PATTERNS,
    JudgeResult,
    ModelResponse,
    PairwiseAssessor,
    QuestionResult,
    _is_refusal,
    _parse_judge_json,
    query_model,
    warmup_model,
)
from pairwise_llm_eval.providers.base import Question


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_response(text: str, completion_tokens: int = 20) -> dict:
    """Build a minimal OpenAI-compatible /v1/chat/completions response."""
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"completion_tokens": completion_tokens},
    }


# ---------------------------------------------------------------------------
# _is_refusal
# ---------------------------------------------------------------------------

class TestIsRefusal:
    def test_empty_string(self) -> None:
        assert _is_refusal("") is True

    def test_whitespace_only(self) -> None:
        assert _is_refusal("   ") is True

    def test_normal_answer(self) -> None:
        assert _is_refusal("The answer is 42.") is False

    @pytest.mark.parametrize("phrase", [
        "I don't know the answer",
        "I cannot help with that",
        "I can't provide that",
        "As an AI, I",
        "I'm unable to answer",
        "I don't have enough information",
    ])
    def test_refusal_phrases(self, phrase: str) -> None:
        assert _is_refusal(phrase) is True

    def test_case_insensitive(self) -> None:
        assert _is_refusal("AS AN AI I must say") is True

    def test_partial_word_no_match(self) -> None:
        # "cannot" must be a whole word; "canonical" should not match
        assert _is_refusal("The canonical form is X.") is False


# ---------------------------------------------------------------------------
# _parse_judge_json
# ---------------------------------------------------------------------------

class TestParseJudgeJson:
    def test_plain_json(self) -> None:
        result = _parse_judge_json('{"score_1": 4, "score_2": 3, "reasoning": "ok"}')
        assert result == {"score_1": 4, "score_2": 3, "reasoning": "ok"}

    def test_markdown_fence(self) -> None:
        text = '```json\n{"score_1": 5, "score_2": 4, "reasoning": "good"}\n```'
        result = _parse_judge_json(text)
        assert result["score_1"] == 5

    def test_plain_backtick_fence(self) -> None:
        text = '```\n{"score": 3}\n```'
        result = _parse_judge_json(text)
        assert result["score"] == 3

    def test_embedded_in_prose(self) -> None:
        text = 'Here is my assessment: {"score_1": 2, "score_2": 3, "reasoning": "ok"} Done.'
        result = _parse_judge_json(text)
        assert result["score_1"] == 2

    def test_invalid_returns_empty(self) -> None:
        result = _parse_judge_json("not json at all")
        assert result == {}

    def test_empty_string(self) -> None:
        assert _parse_judge_json("") == {}


# ---------------------------------------------------------------------------
# query_model
# ---------------------------------------------------------------------------

class TestQueryModel:
    def test_successful_request(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _chat_response("Paris is the capital of France.", 12)
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = query_model("http://localhost:8080", "What is the capital of France?")

        assert result.text == "Paris is the capital of France."
        assert result.token_count == 12
        assert result.is_refusal is False
        assert result.latency_ms >= 0

        # Verify URL path was appended
        called_url = mock_post.call_args[0][0]
        assert called_url.endswith("/v1/chat/completions")

    def test_url_already_has_path(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _chat_response("ok")
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            query_model("http://localhost:8080/v1/chat/completions", "Q?")

        called_url = mock_post.call_args[0][0]
        # Should not double-append the path
        assert called_url.count("/v1/chat/completions") == 1

    def test_network_error_returns_refusal(self) -> None:
        import requests as req
        with patch("requests.post", side_effect=req.RequestException("timeout")):
            result = query_model("http://localhost:8080", "Q?")

        assert result.text == ""
        assert result.is_refusal is True
        assert result.token_count == 0

    def test_fallback_token_count(self) -> None:
        """When usage is absent, token count falls back to word count."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "one two three four five"}}]
        }
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp):
            result = query_model("http://localhost:8080", "Q?")

        assert result.token_count == 5  # word count fallback

    def test_refusal_detected_in_response(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _chat_response("I cannot answer that question.")
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp):
            result = query_model("http://localhost:8080", "Q?")

        assert result.is_refusal is True


# ---------------------------------------------------------------------------
# warmup_model
# ---------------------------------------------------------------------------

class TestWarmupModel:
    def test_sends_n_requests(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _chat_response("ok")
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            warmup_model("http://localhost:8080", n=3)

        assert mock_post.call_count == 3

    def test_zero_warmup(self) -> None:
        with patch("requests.post") as mock_post:
            warmup_model("http://localhost:8080", n=0)
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# PairwiseAssessor.judge_pair
# ---------------------------------------------------------------------------

class TestJudgePair:
    def _make_assessor(self) -> PairwiseAssessor:
        return PairwiseAssessor(
            "http://a:8080", "http://b:8080", "http://judge:8080",
            warmup_requests=0,
        )

    def test_model_a_first_maps_scores(self) -> None:
        assessor = self._make_assessor()
        judge_json = '{"score_1": 4, "score_2": 2, "reasoning": "A is better"}'
        mock_resp = MagicMock()
        mock_resp.json.return_value = _chat_response(judge_json)
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp):
            result = assessor.judge_pair("Q?", "Gold", "A answer", "B answer", swap_order=False)

        assert result.order == "model_a_first"
        assert result.score_1 == 4  # Answer 1 = model_a
        assert result.score_2 == 2  # Answer 2 = model_b

    def test_swap_order_reverses(self) -> None:
        assessor = self._make_assessor()
        judge_json = '{"score_1": 3, "score_2": 5, "reasoning": "B is better"}'
        mock_resp = MagicMock()
        mock_resp.json.return_value = _chat_response(judge_json)
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp):
            result = assessor.judge_pair("Q?", "Gold", "A answer", "B answer", swap_order=True)

        assert result.order == "model_b_first"
        # score_1 corresponds to model_b (presented first), score_2 to model_a
        assert result.score_1 == 3
        assert result.score_2 == 5

    def test_scores_clamped(self) -> None:
        assessor = self._make_assessor()
        judge_json = '{"score_1": 99, "score_2": -3, "reasoning": "wild"}'
        mock_resp = MagicMock()
        mock_resp.json.return_value = _chat_response(judge_json)
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp):
            result = assessor.judge_pair("Q?", "Gold", "A", "B")

        assert result.score_1 == 5
        assert result.score_2 == 0

    def test_parse_failure_returns_zero_scores(self) -> None:
        assessor = self._make_assessor()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _chat_response("cannot parse this")
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp):
            result = assessor.judge_pair("Q?", "Gold", "A", "B")

        assert result.score_1 == 0
        assert result.score_2 == 0


# ---------------------------------------------------------------------------
# PairwiseAssessor.judge_general
# ---------------------------------------------------------------------------

class TestJudgeGeneral:
    def _make_assessor(self) -> PairwiseAssessor:
        return PairwiseAssessor(
            "http://a:8080", "http://b:8080", "http://judge:8080",
            warmup_requests=0,
        )

    def test_returns_score_and_reasoning(self) -> None:
        assessor = self._make_assessor()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _chat_response('{"score": 4, "reasoning": "mostly right"}')
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp):
            score, reasoning = assessor.judge_general("Q?", "Gold", "Answer")

        assert score == 4
        assert "mostly right" in reasoning

    def test_score_clamped(self) -> None:
        assessor = self._make_assessor()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _chat_response('{"score": 10}')
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp):
            score, _ = assessor.judge_general("Q?", "Gold", "Answer")

        assert score == 5


# ---------------------------------------------------------------------------
# PairwiseAssessor.run_domain_assessment (integration-style, all mocked)
# ---------------------------------------------------------------------------

class TestRunDomainAssessment:
    def _make_assessor(self, seed: int = 0) -> PairwiseAssessor:
        return PairwiseAssessor(
            "http://a:8080", "http://b:8080", "http://judge:8080",
            warmup_requests=0,
            position_bias_subset=2,
            seed=seed,
        )

    def _patch_post(self, model_text: str = "A fine answer.", judge_text: str | None = None):
        judge_resp_text = judge_text or '{"score_1": 4, "score_2": 3, "reasoning": "ok"}'

        call_count: list[int] = [0]

        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if "judge" in url:
                resp.json.return_value = _chat_response(judge_resp_text)
            else:
                resp.json.return_value = _chat_response(model_text)
            call_count[0] += 1
            return resp

        return patch("requests.post", side_effect=side_effect), call_count

    def _make_questions(self, n: int = 5) -> list[Question]:
        return [
            Question(id=f"q{i}", question=f"Q{i}?", gold_answer=f"A{i}", category="cat")
            for i in range(n)
        ]

    def test_returns_correct_count(self) -> None:
        assessor = self._make_assessor()
        questions = self._make_questions(4)
        patcher, _ = self._patch_post()

        with patcher:
            results = assessor.run_domain_assessment(questions)

        assert len(results) == 4

    def test_result_type(self) -> None:
        assessor = self._make_assessor()
        questions = self._make_questions(3)
        patcher, _ = self._patch_post()

        with patcher:
            results = assessor.run_domain_assessment(questions)

        for r in results:
            assert isinstance(r, QuestionResult)
            assert r.question_type == "domain"

    def test_scores_mapped_correctly(self) -> None:
        """Verify score mapping: regardless of presentation order, model_a_score
        and model_b_score always refer to the correct model."""
        assessor = self._make_assessor()
        questions = self._make_questions(1)

        # Judge always returns score_1=4, score_2=3.
        # When order="model_a_first": model_a=4, model_b=3.
        # When order="model_b_first": model_a=3, model_b=4.
        # Either way, score_1+score_2 == model_a_score+model_b_score == 7.
        patcher, _ = self._patch_post(
            judge_text='{"score_1": 4, "score_2": 3, "reasoning": "A wins"}'
        )

        with patcher:
            results = assessor.run_domain_assessment(questions)

        r = results[0]
        # Score mapping must be consistent: raw scores sum and model scores sum match
        assert r.model_a_score + r.model_b_score == 7
        assert r.judge["score_1"] + r.judge["score_2"] == 7
        # Order must be recorded
        assert r.judge["order"] in ("model_a_first", "model_b_first")
        # Verify the mapping is correct based on the recorded order
        if r.judge["order"] == "model_a_first":
            assert r.model_a_score == r.judge["score_1"]
            assert r.model_b_score == r.judge["score_2"]
        else:
            assert r.model_b_score == r.judge["score_1"]
            assert r.model_a_score == r.judge["score_2"]

    def test_swap_adjudication_present_in_subset(self) -> None:
        assessor = self._make_assessor(seed=0)
        questions = self._make_questions(10)
        patcher, _ = self._patch_post()

        with patcher:
            results = assessor.run_domain_assessment(questions)

        swap_count = sum(
            1 for r in results if "swap_adjudication" in (r.metadata or {})
        )
        assert swap_count == 2  # position_bias_subset=2

    def test_question_fields_preserved(self) -> None:
        assessor = self._make_assessor()
        questions = [Question(id="unique_id", question="My Q?", gold_answer="My A", category="mycat")]
        patcher, _ = self._patch_post()

        with patcher:
            results = assessor.run_domain_assessment(questions)

        r = results[0]
        assert r.question_id == "unique_id"
        assert r.question_text == "My Q?"
        assert r.category == "mycat"


# ---------------------------------------------------------------------------
# PairwiseAssessor.run_general_assessment
# ---------------------------------------------------------------------------

class TestRunGeneralAssessment:
    def _make_assessor(self) -> PairwiseAssessor:
        return PairwiseAssessor(
            "http://a:8080", "http://b:8080", "http://judge:8080",
            warmup_requests=0,
        )

    def test_returns_correct_count(self, general_questions_subset) -> None:
        assessor = self._make_assessor()

        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if "judge" in url:
                resp.json.return_value = _chat_response('{"score": 4, "reasoning": "ok"}')
            else:
                resp.json.return_value = _chat_response("An answer.")
            return resp

        with patch("requests.post", side_effect=side_effect):
            results = assessor.run_general_assessment(general_questions_subset)

        assert len(results) == len(general_questions_subset)

    def test_result_type_is_general(self, general_questions_subset) -> None:
        assessor = self._make_assessor()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _chat_response('{"score": 3, "reasoning": "ok"}')

        with patch("requests.post", return_value=mock_resp):
            results = assessor.run_general_assessment(general_questions_subset)

        for r in results:
            assert r.question_type == "general"

    def test_scores_present(self, general_questions_subset) -> None:
        assessor = self._make_assessor()

        call_n: list[int] = [0]

        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if "judge" in url:
                # Alternate scores for model_a and model_b calls
                score = 4 if call_n[0] % 2 == 0 else 3
                resp.json.return_value = _chat_response(f'{{"score": {score}, "reasoning": "ok"}}')
                call_n[0] += 1
            else:
                resp.json.return_value = _chat_response("An answer.")
            return resp

        with patch("requests.post", side_effect=side_effect):
            results = assessor.run_general_assessment(general_questions_subset)

        for r in results:
            assert r.model_a_score >= 0
            assert r.model_b_score >= 0
