"""pairwise-llm-eval — blinded pairwise LLM evaluation with statistical rigour.

Quick start::

    from pairwise_llm_eval import PairwiseAssessor, MemoryProvider, Question

    questions = [
        Question(
            id="q1",
            question="What is the capital of France?",
            gold_answer="Paris",
            category="geography",
        ),
    ]
    assessor = PairwiseAssessor(
        model_a_url="http://localhost:8080",
        model_b_url="http://localhost:8081",
        judge_url="http://localhost:8082",
        model_a_label="baseline",
        model_b_label="fine-tuned",
        warmup_requests=0,  # skip warmup for quick tests
    )
    result = assessor.run_full(MemoryProvider(questions), n_domain=1, n_general=0)
    print(result.metrics)
"""

from .assessor import AssessmentResult, PairwiseAssessor, QuestionResult
from .providers.base import Question, QuestionProvider
from .providers.jsonl import JSONLProvider
from .providers.memory import MemoryProvider

__all__ = [
    "PairwiseAssessor",
    "AssessmentResult",
    "QuestionResult",
    "Question",
    "QuestionProvider",
    "MemoryProvider",
    "JSONLProvider",
]

__version__ = "0.1.0"
