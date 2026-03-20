"""Core pairwise evaluation engine.

Provides :class:`PairwiseAssessor`, the main orchestration class, plus the
low-level data classes and model interaction helpers.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

from .bias import select_bias_subset
from .providers.base import Question, QuestionProvider

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

REFUSAL_PATTERNS: list[str] = [
    r"\bi don'?t know\b",
    r"\bi cannot\b",
    r"\bi can'?t\b",
    r"\bas an ai\b",
    r"\bi'?m unable to\b",
    r"\bi don'?t have enough\b",
]

# ---------------------------------------------------------------------------
# Default judge prompt templates
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE_PROMPT = """\
You are assessing two AI answers to a question.

Question: {question}
Reference answer: {gold_answer}

Answer 1: {answer_a}
Answer 2: {answer_b}

Rate EACH answer on a scale of 1-5:
- 5: Excellent — accurate, complete, well-explained
- 4: Good — mostly accurate with minor gaps
- 3: Adequate — partially correct but missing key details
- 2: Poor — significant errors or mostly irrelevant
- 1: Wrong — factually incorrect or completely off-topic

Respond with JSON only: {{"score_1": N, "score_2": N, "reasoning": "brief comparison"}}"""

_DEFAULT_GENERAL_JUDGE_PROMPT = """\
You are assessing an AI answer to a general knowledge question.

Question: {question}
Expected answer: {expected}
AI answer: {answer}

Rate the answer on a scale of 1-5:
- 5: Fully correct and well-stated
- 4: Correct with minor imprecision
- 3: Partially correct
- 2: Mostly incorrect
- 1: Completely wrong

Respond with JSON only: {{"score": N, "reasoning": "brief assessment"}}"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelResponse:
    """Raw response from a model endpoint."""

    text: str
    latency_ms: float
    token_count: int
    is_refusal: bool


@dataclass
class JudgeResult:
    """Parsed result from the judge for a single pairwise comparison."""

    score_1: int
    score_2: int
    reasoning: str
    order: str  # "model_a_first" or "model_b_first"


@dataclass
class QuestionResult:
    """Full result for a single question: both model responses and judge scores."""

    question_id: str
    question_type: str  # "domain" or "general"
    question_text: str
    gold_answer: str
    category: str
    model_a_response: dict[str, Any]
    model_b_response: dict[str, Any]
    judge: dict[str, Any]
    model_a_score: int
    model_b_score: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssessmentResult:
    """Top-level container returned by :meth:`PairwiseAssessor.run_full`."""

    domain_results: list[QuestionResult]
    general_results: list[QuestionResult]
    metrics: dict[str, Any]
    model_a_label: str
    model_b_label: str


# ---------------------------------------------------------------------------
# Low-level model helpers (module-level, fully generic)
# ---------------------------------------------------------------------------


def query_model(
    endpoint: str,
    question: str,
    *,
    system_prompt: str = "You are a knowledgeable assistant. Answer the question accurately and concisely.",
    timeout: int = 120,
    max_tokens: int = 1024,
) -> ModelResponse:
    """Send a single question to an OpenAI-compatible chat completion endpoint.

    Works with any server that implements ``POST /v1/chat/completions``
    (llama.cpp, vLLM, LiteLLM, Ollama, etc.).

    Args:
        endpoint: Base URL of the inference server, e.g.
                  ``"http://localhost:8080"``.  The path
                  ``/v1/chat/completions`` is appended automatically if not
                  already present.
        question: User message text.
        system_prompt: System message prepended to the conversation.
        timeout: Request timeout in seconds.
        max_tokens: Maximum completion tokens requested.

    Returns:
        A :class:`ModelResponse` with the generated text and metadata.
        On network errors the response has empty text and ``is_refusal=True``.
    """
    url = endpoint.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url += "/v1/chat/completions"

    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    start = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        elapsed = (time.monotonic() - start) * 1000
        log.warning("Model request failed: %s", exc)
        return ModelResponse(text="", latency_ms=elapsed, token_count=0, is_refusal=True)

    elapsed = (time.monotonic() - start) * 1000
    data = resp.json()

    text = ""
    token_count = 0
    if "choices" in data and data["choices"]:
        text = data["choices"][0].get("message", {}).get("content", "")
    if "usage" in data:
        token_count = data["usage"].get("completion_tokens", 0)

    if not token_count and text:
        token_count = len(text.split())  # rough word-based fallback

    return ModelResponse(
        text=text,
        latency_ms=elapsed,
        token_count=token_count,
        is_refusal=_is_refusal(text),
    )


def _is_refusal(text: str) -> bool:
    """Return ``True`` if *text* matches a known refusal pattern or is empty."""
    if not text or not text.strip():
        return True
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def warmup_model(endpoint: str, n: int = 3) -> None:
    """Send *n* trivial requests to prime the model's KV cache.

    Args:
        endpoint: Inference server base URL.
        n: Number of warmup requests.
    """
    log.info("Warming up %s with %d requests...", endpoint, n)
    for i in range(n):
        query_model(endpoint, f"What is {i + 1} + {i + 1}?", timeout=30)


def _safe_int(val: Any, default: int = 0) -> int:
    """Convert *val* to int, returning *default* on failure."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _parse_judge_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from the judge's response.

    Handles markdown code fences (```json ... ```) and uses a
    bracket-counting approach to extract nested JSON objects correctly.

    Args:
        text: Raw text returned by the judge model.

    Returns:
        Parsed dict, or an empty dict if parsing fails.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Fast path: try the full (stripped) text first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Find outermost balanced braces (handles nested JSON correctly)
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


# ---------------------------------------------------------------------------
# PairwiseAssessor
# ---------------------------------------------------------------------------


class PairwiseAssessor:
    """Orchestrates a blinded pairwise LLM evaluation with position bias controls.

    Two models (model_a and model_b) are queried on every question.  A
    separate judge model scores both answers in a blinded A/B format.
    Presentation order is randomised per question to mitigate position bias.
    A subset of questions is re-judged with the order swapped so that bias
    can be measured empirically.

    Args:
        model_a_url: Inference server URL for model A.
        model_b_url: Inference server URL for model B.
        judge_url: Inference server URL for the judge model.
        model_a_label: Human-readable label for model A (used in reports).
        model_b_label: Human-readable label for model B.
        request_timeout: Per-request timeout in seconds for models under test.
        judge_timeout: Per-request timeout in seconds for the judge.
        max_tokens_model: Maximum completion tokens for models under test.
        max_tokens_judge: Maximum completion tokens for the judge.
        warmup_requests: Number of warmup requests sent to each endpoint
                         before the assessment starts.
        position_bias_subset: Number of domain questions that receive
                               swap-order adjudication.
        min_cell_size: Minimum paired samples per category for per-category
                       statistics; smaller groups are aggregated into OTHER.
        seed: Master random seed.
        judge_system_prompt: System prompt sent to the judge for all calls.
        judge_prompt_template: Format string for the pairwise judge prompt.
                                Must contain ``{question}``, ``{gold_answer}``,
                                ``{answer_a}``, ``{answer_b}``.  Defaults to a
                                generic 1–5 rubric.
        general_judge_prompt_template: Format string for the general-knowledge
                                        judge prompt.  Must contain
                                        ``{question}``, ``{expected}``,
                                        ``{answer}``.

    Example::

        from pairwise_llm_eval import PairwiseAssessor, MemoryProvider, Question

        questions = [Question(id="1", question="...", gold_answer="...", category="cat")]
        assessor = PairwiseAssessor(
            model_a_url="http://localhost:8080",
            model_b_url="http://localhost:8081",
            judge_url="http://localhost:8082",
        )
        result = assessor.run_full(MemoryProvider(questions), n_domain=len(questions))
    """

    def __init__(
        self,
        model_a_url: str,
        model_b_url: str,
        judge_url: str,
        *,
        model_a_label: str = "model_a",
        model_b_label: str = "model_b",
        request_timeout: int = 120,
        judge_timeout: int = 180,
        max_tokens_model: int = 1024,
        max_tokens_judge: int = 512,
        warmup_requests: int = 3,
        position_bias_subset: int = 50,
        min_cell_size: int = 20,
        seed: int = 42,
        judge_system_prompt: str = "You are an expert assessment judge. Respond only with valid JSON.",
        judge_prompt_template: str | None = None,
        general_judge_prompt_template: str | None = None,
    ) -> None:
        self.model_a_url = model_a_url
        self.model_b_url = model_b_url
        self.judge_url = judge_url
        self.model_a_label = model_a_label
        self.model_b_label = model_b_label
        self.request_timeout = request_timeout
        self.judge_timeout = judge_timeout
        self.max_tokens_model = max_tokens_model
        self.max_tokens_judge = max_tokens_judge
        self.warmup_requests = warmup_requests
        self.position_bias_subset = position_bias_subset
        self.min_cell_size = min_cell_size
        self.seed = seed
        self._judge_system_prompt = judge_system_prompt
        self._judge_prompt_template = judge_prompt_template or _DEFAULT_JUDGE_PROMPT
        self._general_judge_prompt_template = (
            general_judge_prompt_template or _DEFAULT_GENERAL_JUDGE_PROMPT
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _query(self, endpoint: str, question: str, *, system_prompt: str | None = None) -> ModelResponse:
        return query_model(
            endpoint,
            question,
            system_prompt=system_prompt or "You are a knowledgeable assistant. Answer the question accurately and concisely.",
            timeout=self.request_timeout,
            max_tokens=self.max_tokens_model,
        )

    def _query_judge(self, prompt: str) -> ModelResponse:
        return query_model(
            self.judge_url,
            prompt,
            system_prompt=self._judge_system_prompt,
            timeout=self.judge_timeout,
            max_tokens=self.max_tokens_judge,
        )

    def judge_pair(
        self,
        question: str,
        gold_answer: str,
        model_a_answer: str,
        model_b_answer: str,
        *,
        swap_order: bool = False,
    ) -> JudgeResult:
        """Run the judge on a single question with blinded A/B presentation.

        Args:
            question: The original question text.
            gold_answer: The reference answer.
            model_a_answer: Response from model A.
            model_b_answer: Response from model B.
            swap_order: When ``True``, model B is presented as "Answer 1"
                        and model A as "Answer 2", reversing the usual order.

        Returns:
            A :class:`JudgeResult` with raw scores and parsed reasoning.
        """
        if swap_order:
            answer_a, answer_b = model_b_answer, model_a_answer
            order = "model_b_first"
        else:
            answer_a, answer_b = model_a_answer, model_b_answer
            order = "model_a_first"

        prompt = self._judge_prompt_template.format(
            question=question,
            gold_answer=gold_answer[:2000],
            answer_a=answer_a[:2000],
            answer_b=answer_b[:2000],
        )

        resp = self._query_judge(prompt)
        parsed = _parse_judge_json(resp.text)
        # 0 is used as a sentinel for parse failures; filtered out by compute_metrics.
        score_1 = max(0, min(5, _safe_int(parsed.get("score_1", 0))))
        score_2 = max(0, min(5, _safe_int(parsed.get("score_2", 0))))
        reasoning = parsed.get("reasoning", resp.text[:500])

        return JudgeResult(
            score_1=score_1,
            score_2=score_2,
            reasoning=reasoning,
            order=order,
        )

    def judge_general(
        self,
        question: str,
        expected: str,
        answer: str,
    ) -> tuple[int, str]:
        """Rate a single general-knowledge answer.

        Args:
            question: The general knowledge question.
            expected: The reference answer.
            answer: The model's answer to rate.

        Returns:
            ``(score, reasoning)`` where score is 1–5 (0 on parse failure).
        """
        prompt = self._general_judge_prompt_template.format(
            question=question,
            expected=expected,
            answer=answer[:2000],
        )
        resp = self._query_judge(prompt)
        parsed = _parse_judge_json(resp.text)
        # 0 is used as a sentinel for parse failures; filtered out by compute_metrics.
        score = max(0, min(5, _safe_int(parsed.get("score", 0))))
        reasoning = parsed.get("reasoning", resp.text[:500])
        return score, reasoning

    # ------------------------------------------------------------------ #
    # Public assessment methods                                            #
    # ------------------------------------------------------------------ #

    def warmup(self) -> None:
        """Send warmup requests to all three endpoints.

        Called automatically by :meth:`run_full` unless ``warmup_requests``
        is set to 0.
        """
        if self.warmup_requests > 0:
            warmup_model(self.model_a_url, self.warmup_requests)
            warmup_model(self.model_b_url, self.warmup_requests)
            warmup_model(self.judge_url, self.warmup_requests)

    def run_domain_assessment(self, questions: list[Question]) -> list[QuestionResult]:
        """Evaluate both models on a list of domain questions.

        For each question:

        1. Query model A and model B independently.
        2. Present both answers to the judge in a randomly-chosen order.
        3. For a random subset (``position_bias_subset`` questions), re-judge
           with the presentation order swapped and store the comparison in
           ``metadata["swap_adjudication"]``.

        Args:
            questions: Ordered list of domain questions to evaluate.

        Returns:
            A :class:`QuestionResult` for every question.
        """
        n = len(questions)
        bias_indices = select_bias_subset(n, self.position_bias_subset, self.seed)
        rng = random.Random(self.seed)
        results: list[QuestionResult] = []

        for i, q in enumerate(questions):
            log.info(
                "Domain [%d/%d] category=%s id=%s",
                i + 1, n, q.category, q.id,
            )

            resp_a = self._query(self.model_a_url, q.question)
            resp_b = self._query(self.model_b_url, q.question)

            # Randomise primary presentation order
            primary_swap = rng.random() < 0.5
            judge_result = self.judge_pair(
                q.question, q.gold_answer,
                resp_a.text, resp_b.text,
                swap_order=primary_swap,
            )

            # Map raw score_1/score_2 back to model_a/model_b
            if judge_result.order == "model_a_first":
                score_a = judge_result.score_1
                score_b = judge_result.score_2
            else:
                score_a = judge_result.score_2
                score_b = judge_result.score_1

            metadata: dict[str, Any] = {
                "judge_order": judge_result.order,
            }
            if q.metadata:
                metadata.update(q.metadata)

            # Swap-order adjudication for position bias detection
            if i in bias_indices:
                swap_result = self.judge_pair(
                    q.question, q.gold_answer,
                    resp_a.text, resp_b.text,
                    swap_order=not primary_swap,
                )
                if swap_result.order == "model_a_first":
                    swap_a = swap_result.score_1
                    swap_b = swap_result.score_2
                else:
                    swap_a = swap_result.score_2
                    swap_b = swap_result.score_1

                metadata["swap_adjudication"] = {
                    "swap_model_a_score": swap_a,
                    "swap_model_b_score": swap_b,
                    "swap_reasoning": swap_result.reasoning,
                    "model_a_disagree": abs(score_a - swap_a),
                    "model_b_disagree": abs(score_b - swap_b),
                }

            results.append(
                QuestionResult(
                    question_id=q.id,
                    question_type="domain",
                    question_text=q.question,
                    gold_answer=q.gold_answer[:500],
                    category=q.category,
                    model_a_response={
                        "text": resp_a.text,
                        "latency_ms": resp_a.latency_ms,
                        "token_count": resp_a.token_count,
                        "is_refusal": resp_a.is_refusal,
                    },
                    model_b_response={
                        "text": resp_b.text,
                        "latency_ms": resp_b.latency_ms,
                        "token_count": resp_b.token_count,
                        "is_refusal": resp_b.is_refusal,
                    },
                    judge={
                        "score_1": judge_result.score_1,
                        "score_2": judge_result.score_2,
                        "reasoning": judge_result.reasoning,
                        "order": judge_result.order,
                    },
                    model_a_score=score_a,
                    model_b_score=score_b,
                    metadata=metadata,
                )
            )

        return results

    def run_general_assessment(
        self, questions: list[dict[str, str]]
    ) -> list[QuestionResult]:
        """Evaluate both models on general knowledge questions.

        Each question is judged independently for model A and model B (not
        as a pairwise comparison) so that absolute quality can be assessed for
        catastrophic forgetting detection.

        Args:
            questions: List of dicts with keys ``question``, ``answer``, and
                       ``category``.

        Returns:
            A :class:`QuestionResult` for every question.
        """
        n = len(questions)
        results: list[QuestionResult] = []

        for i, q in enumerate(questions):
            log.info("General [%d/%d] category=%s", i + 1, n, q["category"])

            resp_a = self._query(self.model_a_url, q["question"])
            resp_b = self._query(self.model_b_url, q["question"])

            score_a, reasoning_a = self.judge_general(
                q["question"], q["answer"], resp_a.text
            )
            score_b, reasoning_b = self.judge_general(
                q["question"], q["answer"], resp_b.text
            )

            results.append(
                QuestionResult(
                    question_id=f"general_{i:03d}",
                    question_type="general",
                    question_text=q["question"],
                    gold_answer=q["answer"],
                    category=q["category"],
                    model_a_response={
                        "text": resp_a.text,
                        "latency_ms": resp_a.latency_ms,
                        "token_count": resp_a.token_count,
                        "is_refusal": resp_a.is_refusal,
                    },
                    model_b_response={
                        "text": resp_b.text,
                        "latency_ms": resp_b.latency_ms,
                        "token_count": resp_b.token_count,
                        "is_refusal": resp_b.is_refusal,
                    },
                    judge={
                        "model_a_score": score_a,
                        "model_a_reasoning": reasoning_a,
                        "model_b_score": score_b,
                        "model_b_reasoning": reasoning_b,
                    },
                    model_a_score=score_a,
                    model_b_score=score_b,
                    metadata={"category": q["category"]},
                )
            )

        return results

    def run_full(
        self,
        domain_provider: QuestionProvider,
        n_domain: int = 500,
        general_questions: list[dict[str, str]] | None = None,
        n_general: int = 100,
    ) -> AssessmentResult:
        """Run the complete evaluation: domain + general knowledge.

        This is the primary entry point for running an assessment.  It:

        1. Warms up all endpoints.
        2. Samples ``n_domain`` questions from *domain_provider*.
        3. Runs domain assessment with position-bias controls.
        4. Runs general knowledge assessment (using built-in questions if
           *general_questions* is ``None``).
        5. Computes aggregate statistics.

        Args:
            domain_provider: Any :class:`QuestionProvider` implementation
                              that yields domain-specific questions.
            n_domain: Number of domain questions to sample and evaluate.
            general_questions: Optional override for the general knowledge
                                question pool.  When ``None``, the 100 built-in
                                questions are used (stratified to *n_general*).
            n_general: Number of general knowledge questions to evaluate.

        Returns:
            An :class:`AssessmentResult` containing all raw results and
            computed metrics.
        """
        from .forgetting import sample_general_questions
        from .statistics import compute_metrics

        log.info("=" * 60)
        log.info("Pairwise Evaluation — %s vs %s", self.model_a_label, self.model_b_label)
        log.info("=" * 60)
        log.info("Model A: %s", self.model_a_url)
        log.info("Model B: %s", self.model_b_url)
        log.info("Judge:   %s", self.judge_url)
        log.info("Domain:  %d questions", n_domain)
        log.info("General: %d questions", n_general)

        self.warmup()

        log.info("Sampling domain questions...")
        domain_questions = domain_provider.sample(n_domain)
        log.info("Sampled %d domain questions", len(domain_questions))

        if general_questions is None:
            general_questions = sample_general_questions(n_general, seed=self.seed)
        else:
            general_questions = general_questions[:n_general]

        log.info("Running domain assessment...")
        domain_results = self.run_domain_assessment(domain_questions)

        log.info("Running general knowledge assessment...")
        general_results = self.run_general_assessment(general_questions)

        log.info("Computing metrics...")
        metrics = compute_metrics(
            domain_results,
            general_results,
            min_cell_size=self.min_cell_size,
        )

        # Annotate metrics with model labels
        metrics["model_a_label"] = self.model_a_label
        metrics["model_b_label"] = self.model_b_label

        return AssessmentResult(
            domain_results=domain_results,
            general_results=general_results,
            metrics=metrics,
            model_a_label=self.model_a_label,
            model_b_label=self.model_b_label,
        )

    # Expose as_dict for serialisation convenience
    @staticmethod
    def results_to_dicts(results: list[QuestionResult]) -> list[dict[str, Any]]:
        """Convert a list of :class:`QuestionResult` to JSON-serialisable dicts."""
        return [asdict(r) for r in results]
