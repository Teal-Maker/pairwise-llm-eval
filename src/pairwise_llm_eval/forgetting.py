"""Catastrophic forgetting detection via general knowledge questions.

Provides a loader for the 100 built-in general knowledge questions (10
categories x 10 questions) bundled with this package, and a stratified
sampler that draws evenly across categories when using a subset.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_DATA_FILE = Path(__file__).parent / "general_questions.json"


def load_default_general_questions() -> list[dict[str, str]]:
    """Load the 100 built-in general knowledge questions.

    The questions span ten categories: science, history, geography,
    math_logic, common_sense, language, technology, economics,
    arts_literature, and health_biology.

    Each dict has the keys ``category``, ``question``, and ``answer``.

    Returns:
        A list of 100 question dicts.

    Example::

        from pairwise_llm_eval.forgetting import load_default_general_questions

        questions = load_default_general_questions()
        print(len(questions))   # 100
    """
    with open(_DATA_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def sample_general_questions(n: int, *, seed: int = 42) -> list[dict[str, str]]:
    """Return a stratified sample of up to *n* general knowledge questions.

    When *n* is less than the full pool, questions are drawn in equal
    proportions from each category (round-robin, deterministic).

    Args:
        n: Number of questions to return (capped at pool size).
        seed: Random seed for shuffling within categories.

    Returns:
        A list of at most *n* question dicts.

    Example::

        sample = sample_general_questions(30)
        categories = {q["category"] for q in sample}
        assert len(categories) == 10   # all categories represented
    """
    import random

    all_qs = load_default_general_questions()
    if n >= len(all_qs):
        return all_qs

    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, str]]] = defaultdict(list)
    for q in all_qs:
        by_cat[q["category"]].append(q)

    for cat_list in by_cat.values():
        rng.shuffle(cat_list)

    result: list[dict[str, str]] = []
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
            break

    return result
