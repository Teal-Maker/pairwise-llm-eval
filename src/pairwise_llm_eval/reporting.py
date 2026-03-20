"""Report generation for pairwise evaluation results.

Writes three output files into a directory:

- ``assessment_summary.json`` — full metrics dict
- ``assessment_details.jsonl`` — one JSON object per question result
- ``position_bias_check.json`` — position bias details (if available)
- ``assessment_report.md`` — human-readable Markdown summary
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .assessor import QuestionResult

log = logging.getLogger(__name__)


def write_report(
    metrics: dict[str, Any],
    domain_results: list[QuestionResult],
    general_results: list[QuestionResult],
    output_dir: str | Path,
    *,
    model_a_label: str = "model_a",
    model_b_label: str = "model_b",
    domain_label: str = "domain",
) -> dict[str, Path]:
    """Write JSON, JSONL, and Markdown report files.

    Args:
        metrics: The metrics dict returned by :func:`compute_metrics`.
        domain_results: Domain question results list.
        general_results: General knowledge results list.
        output_dir: Directory in which to write output files.  Created if
                    it does not exist.
        model_a_label: Human-readable label for model A (used in Markdown).
        model_b_label: Human-readable label for model B.
        domain_label: The domain key used in *metrics* (default ``"domain"``).

    Returns:
        A dict mapping output type to the :class:`Path` of the written file::

            {
                "summary": Path(".../assessment_summary.json"),
                "details": Path(".../assessment_details.jsonl"),
                "bias": Path(".../position_bias_check.json"),   # if present
                "report": Path(".../assessment_report.md"),
            }
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    written: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # JSON summary
    # ------------------------------------------------------------------
    summary_path = out / "assessment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    log.info("Summary: %s", summary_path)
    written["summary"] = summary_path

    # ------------------------------------------------------------------
    # JSONL details
    # ------------------------------------------------------------------
    details_path = out / "assessment_details.jsonl"
    with open(details_path, "w", encoding="utf-8") as fh:
        for r in domain_results + general_results:
            fh.write(json.dumps(asdict(r), default=str) + "\n")
    log.info("Details: %s", details_path)
    written["details"] = details_path

    # ------------------------------------------------------------------
    # Position bias JSON
    # ------------------------------------------------------------------
    if "position_bias" in metrics:
        bias_path = out / "position_bias_check.json"
        with open(bias_path, "w", encoding="utf-8") as fh:
            json.dump(metrics["position_bias"], fh, indent=2)
        log.info("Position bias: %s", bias_path)
        written["bias"] = bias_path

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    md_path = out / "assessment_report.md"
    _write_markdown(
        md_path, metrics, ts, model_a_label, model_b_label, domain_label
    )
    log.info("Report: %s", md_path)
    written["report"] = md_path

    return written


def _write_markdown(
    path: Path,
    metrics: dict[str, Any],
    ts: str,
    model_a_label: str,
    model_b_label: str,
    domain_label: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Pairwise Evaluation Report\n\n")
        f.write(f"**Generated**: {ts}\n\n")
        f.write(f"**Model A**: {model_a_label}  \n")
        f.write(f"**Model B**: {model_b_label}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- Domain questions: {metrics.get('domain_count', 0)}\n")
        f.write(f"- General knowledge questions: {metrics.get('general_count', 0)}\n\n")

        # ----------------------------------------------------------
        # Domain results
        # ----------------------------------------------------------
        if domain_label in metrics:
            d = metrics[domain_label]
            f.write("## Domain Assessment\n\n")
            f.write(f"| Metric | {model_a_label} | {model_b_label} |\n")
            f.write("|--------|" + "------|" * 2 + "\n")
            f.write(
                f"| Mean Score | {d['model_a_mean_score']:.2f} | {d['model_b_mean_score']:.2f} |\n"
            )
            f.write(
                f"| Median Score | {d['model_a_median_score']:.1f} | {d['model_b_median_score']:.1f} |\n"
            )
            f.write(
                f"| Refusal Rate | {d['model_a_refusal_rate']:.1%} | {d['model_b_refusal_rate']:.1%} |\n"
            )
            f.write(
                f"| Mean Tokens | {d['model_a_mean_tokens']:.0f} | {d['model_b_mean_tokens']:.0f} |\n"
            )
            f.write(
                f"| Latency p50 | {d['model_a_latency_p50']:.0f}ms | {d['model_b_latency_p50']:.0f}ms |\n"
            )
            f.write(
                f"| Latency p95 | {d['model_a_latency_p95']:.0f}ms | {d['model_b_latency_p95']:.0f}ms |\n\n"
            )

            f.write("### Head-to-Head\n\n")
            f.write(
                f"- **{model_b_label} wins**: {d['model_b_wins']} ({d['model_b_win_rate']:.1%})\n"
            )
            f.write(f"- **{model_a_label} wins**: {d['model_a_wins']}\n")
            f.write(f"- **Ties**: {d['ties']}\n")
            ci = d["median_diff_ci_95"]
            f.write(
                f"- **Median score diff (B - A)**: {d['median_diff']:+.2f} "
                f"[{ci[0]:+.2f}, {ci[1]:+.2f}] 95% CI\n"
            )
            f.write(f"- **Wilcoxon p-value**: {d['wilcoxon_p_value']:.4f}\n\n")

            by_cat_key = f"{domain_label}_by_category"
            if by_cat_key in metrics and metrics[by_cat_key]:
                f.write("### By Category\n\n")
                f.write(
                    f"| Category | n | {model_a_label} Mean | {model_b_label} Mean | "
                    "Median Diff | B Wins | p (FDR) |\n"
                )
                f.write("|----------|---|" + "------|" * 5 + "\n")
                for area, am in sorted(metrics[by_cat_key].items()):
                    p_str = f"{am.get('p_value_fdr', am.get('p_value_raw', 1.0)):.3f}"
                    f.write(
                        f"| {area} | {am['n']} | {am['model_a_mean']:.2f} | "
                        f"{am['model_b_mean']:.2f} | {am['median_diff']:+.2f} | "
                        f"{am['model_b_wins']} | {p_str} |\n"
                    )
                f.write("\n")

        # ----------------------------------------------------------
        # General knowledge results
        # ----------------------------------------------------------
        if "general" in metrics:
            g = metrics["general"]
            f.write("## General Knowledge (Catastrophic Forgetting Check)\n\n")
            f.write(f"| Metric | {model_a_label} | {model_b_label} |\n")
            f.write("|--------|" + "------|" * 2 + "\n")
            f.write(
                f"| Mean Score | {g['model_a_mean_score']:.2f} | {g['model_b_mean_score']:.2f} |\n"
            )
            ci_g = g["median_diff_ci_95"]
            f.write(
                f"| Median Diff | {g['median_diff']:+.2f} "
                f"[{ci_g[0]:+.2f}, {ci_g[1]:+.2f}] | — |\n"
            )
            f.write(f"| Wilcoxon p | {g['wilcoxon_p_value']:.4f} | — |\n")
            f.write(
                f"| B Wins | {g['model_b_wins']} | A Wins: {g['model_a_wins']} | "
                f"Ties: {g['ties']} |\n\n"
            )

            if g.get("forgetting_flag"):
                f.write(
                    "**WARNING: Possible catastrophic forgetting detected** "
                    f"({model_b_label} mean > 0.5 below {model_a_label})\n\n"
                )
            else:
                f.write("No catastrophic forgetting detected.\n\n")

            if "general_by_category" in metrics:
                f.write("### By Category\n\n")
                f.write(
                    f"| Category | n | {model_a_label} Mean | {model_b_label} Mean | Diff |\n"
                )
                f.write("|----------|---|" + "------|" * 3 + "\n")
                for cat, cm in sorted(metrics["general_by_category"].items()):
                    f.write(
                        f"| {cat} | {cm['n']} | {cm['model_a_mean']:.2f} | "
                        f"{cm['model_b_mean']:.2f} | {cm['diff_mean']:+.2f} |\n"
                    )
                f.write("\n")

        # ----------------------------------------------------------
        # Position bias
        # ----------------------------------------------------------
        if "position_bias" in metrics:
            pb = metrics["position_bias"]
            f.write("## Position Bias Check\n\n")
            f.write(f"- Questions checked: {pb['n_checked']}\n")
            f.write(f"- Disagreements (>1 point): {pb['n_disagree_gt1']}\n")
            f.write(f"- Disagree rate: {pb['disagree_rate']:.1%}\n\n")

            if pb.get("disagree_rate", 0) > 0.10:
                f.write("**WARNING**: Position bias rate exceeds 10% threshold.\n\n")
