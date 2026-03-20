"""Base types for question providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Question:
    """A single evaluation question with metadata.

    Attributes:
        id: Unique identifier for the question.
        question: The question text sent to models under evaluation.
        gold_answer: The reference answer used by the judge.
        category: Grouping label (e.g. topic area, competency code prefix).
        metadata: Arbitrary extra fields; preserved in output but not used
                  by the core evaluation logic.
    """

    id: str
    question: str
    gold_answer: str
    category: str
    metadata: dict | None = field(default=None)


@runtime_checkable
class QuestionProvider(Protocol):
    """Protocol for objects that supply evaluation questions.

    Implement this protocol to plug any data source — a database, a JSONL
    file, an in-memory list, or a remote API — into :class:`PairwiseAssessor`.

    The ``sample`` method must be deterministic when a global random seed is
    set before calling it.  Implementations that need their own RNG should
    accept an optional ``seed`` parameter.
    """

    def sample(self, n: int) -> list[Question]:
        """Return up to *n* questions for evaluation.

        Args:
            n: Maximum number of questions to return.  Implementations may
               return fewer if the underlying pool is smaller than *n*.

        Returns:
            A list of :class:`Question` instances.
        """
        ...
