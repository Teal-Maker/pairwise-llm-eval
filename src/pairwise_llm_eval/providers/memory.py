"""In-memory question provider."""

from __future__ import annotations

import random

from .base import Question


class MemoryProvider:
    """Provides questions from an in-memory list.

    Useful for small test sets, unit tests, or programmatically constructed
    question pools.

    Args:
        questions: The full pool of questions to sample from.
        seed: Optional random seed for reproducible sampling.  When ``None``
              the global ``random`` state is used.

    Example::

        questions = [
            Question(id="q1", question="What is 2+2?",
                     gold_answer="4", category="math"),
        ]
        provider = MemoryProvider(questions, seed=42)
        sample = provider.sample(10)
    """

    def __init__(self, questions: list[Question], *, seed: int | None = None) -> None:
        self._questions = list(questions)
        self._seed = seed

    def sample(self, n: int) -> list[Question]:
        """Return a random sample of up to *n* questions.

        Args:
            n: Desired sample size.

        Returns:
            A shuffled subset (or all questions if the pool is smaller than *n*).
        """
        pool = list(self._questions)
        rng = random.Random(self._seed)
        rng.shuffle(pool)
        return pool[:n]
