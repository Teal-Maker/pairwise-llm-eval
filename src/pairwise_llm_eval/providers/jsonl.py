"""JSONL file-based question provider with stratified sampling."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .base import Question


class JSONLProvider:
    """Loads questions from a JSONL file with optional stratified sampling.

    Each line in the JSONL file must be a JSON object with at least the
    following keys:

    - ``id`` (str)
    - ``question`` (str)
    - ``gold_answer`` (str)
    - ``category`` (str)

    Any additional keys are preserved in ``Question.metadata``.

    Args:
        path: Path to the JSONL file.
        stratified: When ``True``, :meth:`sample` draws an equal number of
                    questions from each category (round-robin).  When
                    ``False``, questions are drawn uniformly at random.
        seed: Optional random seed for reproducible sampling.

    Example::

        provider = JSONLProvider("questions.jsonl", stratified=True, seed=0)
        sample = provider.sample(200)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        stratified: bool = True,
        seed: int | None = None,
    ) -> None:
        self._path = Path(path)
        self._stratified = stratified
        self._seed = seed
        self._questions: list[Question] = []
        self._load()

    def _load(self) -> None:
        questions: list[Question] = []
        with open(self._path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self._path}:{lineno}: invalid JSON — {exc}"
                    ) from exc

                try:
                    q_id = str(obj["id"])
                    question = str(obj["question"])
                    gold_answer = str(obj["gold_answer"])
                    category = str(obj["category"])
                except KeyError as exc:
                    raise ValueError(
                        f"{self._path}:{lineno}: missing required field {exc}"
                    ) from exc

                extra = {
                    k: v
                    for k, v in obj.items()
                    if k not in ("id", "question", "gold_answer", "category")
                }
                questions.append(
                    Question(
                        id=q_id,
                        question=question,
                        gold_answer=gold_answer,
                        category=category,
                        metadata=extra or None,
                    )
                )
        self._questions = questions

    def sample(self, n: int) -> list[Question]:
        """Return up to *n* questions, optionally stratified by category.

        When ``stratified=True`` the sample draws from each category in
        round-robin order so that no single category dominates the set.

        Args:
            n: Maximum number of questions to return.

        Returns:
            A list of :class:`Question` instances.
        """
        rng = random.Random(self._seed)
        pool = list(self._questions)

        if not self._stratified:
            rng.shuffle(pool)
            return pool[:n]

        # Group by category, shuffle within each group
        by_cat: dict[str, list[Question]] = defaultdict(list)
        for q in pool:
            by_cat[q.category].append(q)

        for cat_list in by_cat.values():
            rng.shuffle(cat_list)

        # Round-robin across sorted categories for determinism
        result: list[Question] = []
        cats = sorted(by_cat.keys())
        indices = {c: 0 for c in cats}

        while len(result) < n:
            added_any = False
            for cat in cats:
                if len(result) >= n:
                    break
                idx = indices[cat]
                if idx < len(by_cat[cat]):
                    result.append(by_cat[cat][idx])
                    indices[cat] += 1
                    added_any = True
            if not added_any:
                break  # pool exhausted

        return result

    @property
    def all_questions(self) -> list[Question]:
        """Return all loaded questions without sampling."""
        return list(self._questions)

    def __len__(self) -> int:
        return len(self._questions)
