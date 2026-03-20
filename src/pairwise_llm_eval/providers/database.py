"""PostgreSQL-backed question provider (optional psycopg2 dependency)."""

from __future__ import annotations

import random
from typing import Any

from .base import Question


class DatabaseProvider:
    """Loads questions from a PostgreSQL database via a user-supplied query.

    Requires the ``database`` extra: ``pip install pairwise-llm-eval[database]``.

    The query must return at least these columns:

    - ``id`` — unique question identifier
    - ``question`` — the question text
    - ``gold_answer`` — the reference answer
    - ``category`` — grouping label

    Additional columns are stored in ``Question.metadata``.

    Args:
        connection_string: A libpq-compatible DSN, e.g.
            ``"postgresql://user:pass@host:5432/dbname"``.
        query: SQL that returns the required columns.  Parameterised queries
            using ``%s`` placeholders are not supported here; bake any
            filters directly into the SQL or use ``query_params``.
        query_params: Optional tuple of parameters passed to ``cursor.execute``
            alongside *query*.
        seed: Optional random seed for reproducible shuffling after fetch.

    Example::

        provider = DatabaseProvider(
            connection_string="postgresql://user:pass@localhost/mydb",
            query='''
                SELECT id::text, question, gold_answer, category
                FROM eval_questions
                WHERE split = 'test'
            ''',
            seed=42,
        )
        sample = provider.sample(500)
    """

    def __init__(
        self,
        connection_string: str,
        query: str,
        *,
        query_params: tuple[Any, ...] | None = None,
        seed: int | None = None,
    ) -> None:
        self._dsn = connection_string
        self._query = query
        self._query_params = query_params or ()
        self._seed = seed

    def _connect(self) -> Any:
        try:
            import psycopg2
            import psycopg2.extras as _extras
        except ImportError as exc:
            raise ImportError(
                "DatabaseProvider requires psycopg2. "
                "Install it with: pip install pairwise-llm-eval[database]"
            ) from exc
        self._psycopg2_extras = _extras
        return psycopg2.connect(self._dsn)

    def sample(self, n: int) -> list[Question]:
        """Fetch all rows from the query, shuffle, then return up to *n*.

        The entire result set is fetched into memory and then shuffled locally
        so that ``n`` is honoured without requiring a ``LIMIT`` in the SQL.

        Args:
            n: Maximum number of questions to return.

        Returns:
            A shuffled subset of the query results.
        """
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=self._psycopg2_extras.DictCursor) as cur:
                cur.execute(self._query, self._query_params)
                rows = cur.fetchall()
        finally:
            conn.close()

        required = {"id", "question", "gold_answer", "category"}
        questions: list[Question] = []
        for row in rows:
            row_dict = dict(row)
            missing = required - row_dict.keys()
            if missing:
                raise ValueError(
                    f"DatabaseProvider query result is missing columns: {missing}. "
                    "Ensure the query returns id, question, gold_answer, category."
                )
            extra = {k: v for k, v in row_dict.items() if k not in required}
            questions.append(
                Question(
                    id=str(row_dict["id"]),
                    question=str(row_dict["question"]),
                    gold_answer=str(row_dict["gold_answer"]),
                    category=str(row_dict["category"]),
                    metadata=extra or None,
                )
            )

        rng = random.Random(self._seed)
        rng.shuffle(questions)
        return questions[:n]
