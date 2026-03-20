"""Tests for report generation."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from pairwise_llm_eval.reporting import write_report


# ---------------------------------------------------------------------------
# Minimal stub matching the fields write_report reads via asdict()
# ---------------------------------------------------------------------------

@dataclass
class _FakeResult:
    question_id: str
    question_type: str
    question_text: str
    gold_answer: str
    category: str
    model_a_response: dict[str, Any]
    model_b_response: dict[str, Any]
    judge: dict[str, Any]
    model_a_score: int
    model_b_score: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_result(q_id: str = "q1", a: int = 3, b: int = 4) -> _FakeResult:
    resp = {"text": "ok", "latency_ms": 100.0, "token_count": 30, "is_refusal": False}
    return _FakeResult(
        question_id=q_id,
        question_type="domain",
        question_text="Q?",
        gold_answer="A",
        category="cat",
        model_a_response=resp,
        model_b_response=resp,
        judge={"score_1": a, "score_2": b, "reasoning": "ok", "order": "model_a_first"},
        model_a_score=a,
        model_b_score=b,
    )


def _minimal_metrics() -> dict[str, Any]:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "domain_count": 2,
        "general_count": 0,
        "domain": {
            "model_a_mean_score": 3.0,
            "model_b_mean_score": 4.0,
            "model_a_median_score": 3.0,
            "model_b_median_score": 4.0,
            "median_diff": 1.0,
            "median_diff_ci_95": [0.5, 1.5],
            "wilcoxon_statistic": 0.0,
            "wilcoxon_p_value": 0.5,
            "model_b_wins": 2,
            "model_a_wins": 0,
            "ties": 0,
            "model_b_win_rate": 1.0,
            "model_b_win_rate_ci_95": [0.8, 1.0],
            "model_a_refusal_rate": 0.0,
            "model_b_refusal_rate": 0.0,
            "model_a_latency_p50": 100.0,
            "model_a_latency_p95": 110.0,
            "model_b_latency_p50": 95.0,
            "model_b_latency_p95": 105.0,
            "model_a_mean_tokens": 30.0,
            "model_b_mean_tokens": 40.0,
        },
        "domain_by_category": {},
        "model_a_label": "baseline",
        "model_b_label": "fine-tuned",
    }


class TestWriteReport:
    def _run(self, tmp_dir: Path, metrics=None, domain=None, general=None, **kwargs):
        m = metrics or _minimal_metrics()
        d = domain or [_make_result("q1"), _make_result("q2")]
        g = general or []
        return write_report(
            m, d, g, tmp_dir,
            model_a_label=kwargs.get("model_a_label", "baseline"),
            model_b_label=kwargs.get("model_b_label", "fine-tuned"),
        )

    def test_creates_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "new_subdir"
            self._run(out)
            assert out.is_dir()

    def test_writes_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            written = self._run(Path(td))
            assert "summary" in written
            assert written["summary"].exists()
            with open(written["summary"]) as f:
                data = json.load(f)
            assert "domain_count" in data

    def test_writes_details_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            written = self._run(Path(td))
            assert "details" in written
            lines = written["details"].read_text().splitlines()
            assert len(lines) == 2  # 2 domain results, 0 general
            for line in lines:
                obj = json.loads(line)
                assert "question_id" in obj

    def test_writes_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            written = self._run(Path(td))
            assert "report" in written
            md = written["report"].read_text()
            assert "# Pairwise Evaluation Report" in md

    def test_markdown_contains_model_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            written = self._run(
                Path(td),
                model_a_label="base-v1",
                model_b_label="lora-v2",
            )
            md = written["report"].read_text()
            assert "base-v1" in md
            assert "lora-v2" in md

    def test_no_bias_file_without_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            written = self._run(Path(td))
            assert "bias" not in written

    def test_bias_file_written_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = _minimal_metrics()
            m["position_bias"] = {
                "n_checked": 5,
                "n_disagree_gt1": 1,
                "disagree_rate": 0.2,
                "details": [],
            }
            written = self._run(Path(td), metrics=m)
            assert "bias" in written
            bias_data = json.loads(written["bias"].read_text())
            assert bias_data["n_checked"] == 5

    def test_forgetting_warning_in_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = _minimal_metrics()
            m["general_count"] = 3
            m["general"] = {
                "model_a_mean_score": 4.0,
                "model_b_mean_score": 2.0,
                "median_diff": -2.0,
                "median_diff_ci_95": [-2.5, -1.5],
                "wilcoxon_p_value": 0.01,
                "model_b_wins": 0,
                "model_a_wins": 3,
                "ties": 0,
                "forgetting_flag": True,
            }
            m["general_by_category"] = {}

            gen = [_make_result(f"g{i}") for i in range(3)]
            for r in gen:
                r.question_type = "general"

            written = self._run(Path(td), metrics=m, general=gen)
            md = written["report"].read_text()
            assert "WARNING" in md
            assert "forgetting" in md.lower()

    def test_bias_warning_in_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = _minimal_metrics()
            m["position_bias"] = {
                "n_checked": 10,
                "n_disagree_gt1": 8,
                "disagree_rate": 0.8,
                "details": [],
            }
            written = self._run(Path(td), metrics=m)
            md = written["report"].read_text()
            assert "WARNING" in md

    def test_no_domain_bias_warning_when_low(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = _minimal_metrics()
            m["position_bias"] = {
                "n_checked": 20,
                "n_disagree_gt1": 1,
                "disagree_rate": 0.05,
                "details": [],
            }
            written = self._run(Path(td), metrics=m)
            md = written["report"].read_text()
            # Low rate — no WARNING about position bias threshold
            assert "Position Bias Check" in md
            lines = [ln for ln in md.split("\n") if "WARNING" in ln and "bias" in ln.lower()]
            assert len(lines) == 0

    def test_returns_path_objects(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            written = self._run(Path(td))
            for v in written.values():
                assert isinstance(v, Path)

    def test_combined_domain_and_general(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = _minimal_metrics()
            m["general_count"] = 2
            m["general"] = {
                "model_a_mean_score": 3.5,
                "model_b_mean_score": 3.5,
                "median_diff": 0.0,
                "median_diff_ci_95": [-0.5, 0.5],
                "wilcoxon_p_value": 1.0,
                "model_b_wins": 1,
                "model_a_wins": 1,
                "ties": 0,
                "forgetting_flag": False,
            }
            m["general_by_category"] = {
                "science": {"n": 2, "model_a_mean": 3.5, "model_b_mean": 3.5, "diff_mean": 0.0}
            }
            gen = [_make_result(f"g{i}") for i in range(2)]
            for r in gen:
                r.question_type = "general"
            written = self._run(Path(td), metrics=m, general=gen)
            md = written["report"].read_text()
            assert "General Knowledge" in md
            assert "science" in md
