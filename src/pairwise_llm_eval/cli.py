"""Command-line interface for pairwise-llm-eval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pairwise-llm-eval",
        description=(
            "Blinded pairwise LLM evaluation with position bias controls "
            "and statistical rigour."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal run with domain questions from a JSONL file
  pairwise-llm-eval \\
    --model-a-url http://localhost:8080 \\
    --model-b-url http://localhost:8081 \\
    --judge-url   http://localhost:8082 \\
    --domain-questions questions.jsonl

  # Override labels and output directory
  pairwise-llm-eval \\
    --model-a-url http://localhost:8080 \\
    --model-b-url http://localhost:8081 \\
    --judge-url   http://localhost:8082 \\
    --domain-questions questions.jsonl \\
    --model-a-label base --model-b-label fine-tuned \\
    --output-dir ./results

  # Quick smoke test
  pairwise-llm-eval \\
    --model-a-url http://localhost:8080 \\
    --model-b-url http://localhost:8081 \\
    --judge-url   http://localhost:8082 \\
    --domain-questions questions.jsonl \\
    --domain-count 10 --general-count 10
""",
    )

    # Endpoints
    parser.add_argument(
        "--model-a-url", required=True,
        help="Inference server URL for model A (OpenAI-compatible).",
    )
    parser.add_argument(
        "--model-b-url", required=True,
        help="Inference server URL for model B (OpenAI-compatible).",
    )
    parser.add_argument(
        "--judge-url", required=True,
        help="Inference server URL for the judge model.",
    )

    # Question source
    parser.add_argument(
        "--domain-questions", required=True, metavar="FILE",
        help=(
            "Path to a JSONL file of domain questions. Each line must have: "
            "id, question, gold_answer, category."
        ),
    )

    # Counts
    parser.add_argument(
        "--domain-count", type=int, default=500, metavar="N",
        help="Number of domain questions to sample and evaluate (default: 500).",
    )
    parser.add_argument(
        "--general-count", type=int, default=100, metavar="N",
        help=(
            "Number of general knowledge questions to use "
            "(default: 100, max: 100 built-in questions)."
        ),
    )

    # Labels
    parser.add_argument(
        "--model-a-label", default="model_a", metavar="LABEL",
        help="Human-readable label for model A (default: model_a).",
    )
    parser.add_argument(
        "--model-b-label", default="model_b", metavar="LABEL",
        help="Human-readable label for model B (default: model_b).",
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling and order randomisation (default: 42).",
    )

    # Output
    parser.add_argument(
        "--output-dir", default=".", metavar="DIR",
        help="Directory for output files (default: current directory).",
    )

    # Advanced tuning
    parser.add_argument(
        "--request-timeout", type=int, default=120,
        help="Per-request timeout in seconds for models under test (default: 120).",
    )
    parser.add_argument(
        "--judge-timeout", type=int, default=180,
        help="Per-request timeout in seconds for the judge (default: 180).",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip warmup requests.",
    )
    parser.add_argument(
        "--position-bias-subset", type=int, default=50, metavar="N",
        help=(
            "Number of domain questions to re-judge with swapped order "
            "for position bias detection (default: 50)."
        ),
    )
    parser.add_argument(
        "--min-cell-size", type=int, default=20,
        help=(
            "Minimum paired samples per category for per-category statistics. "
            "Smaller groups aggregated into OTHER (default: 20)."
        ),
    )
    parser.add_argument(
        "--no-stratified", dest="stratified", action="store_false", default=True,
        help="Use uniform random sampling instead of stratified sampling (default: stratified).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Exit code (0 on success, non-zero on error).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate domain questions file
    domain_path = Path(args.domain_questions)
    if not domain_path.exists():
        log.error("Domain questions file not found: %s", domain_path)
        return 1

    # Lazy imports so the package is importable without all deps
    try:
        import numpy as np  # noqa: F401 — check availability
    except ImportError:
        log.error("numpy is required. Install with: pip install pairwise-llm-eval")
        return 1

    from .assessor import PairwiseAssessor
    from .forgetting import sample_general_questions
    from .providers.jsonl import JSONLProvider
    from .reporting import write_report

    np.random.seed(args.seed)

    # Build provider
    provider = JSONLProvider(
        domain_path,
        stratified=args.stratified,
        seed=args.seed,
    )
    log.info("Loaded %d domain questions from %s", len(provider), domain_path)

    # Build assessor
    assessor = PairwiseAssessor(
        model_a_url=args.model_a_url,
        model_b_url=args.model_b_url,
        judge_url=args.judge_url,
        model_a_label=args.model_a_label,
        model_b_label=args.model_b_label,
        request_timeout=args.request_timeout,
        judge_timeout=args.judge_timeout,
        warmup_requests=0 if args.no_warmup else 3,
        position_bias_subset=args.position_bias_subset,
        min_cell_size=args.min_cell_size,
        seed=args.seed,
    )

    # General questions
    general_qs = sample_general_questions(args.general_count, seed=args.seed)

    # Run
    result = assessor.run_full(
        domain_provider=provider,
        n_domain=args.domain_count,
        general_questions=general_qs,
        n_general=args.general_count,
    )

    # Write reports
    output_dir = Path(args.output_dir)
    written = write_report(
        result.metrics,
        result.domain_results,
        result.general_results,
        output_dir,
        model_a_label=args.model_a_label,
        model_b_label=args.model_b_label,
    )

    # Print summary
    m = result.metrics
    print("\n" + "=" * 60)
    print("ASSESSMENT COMPLETE")
    print("=" * 60)

    if "domain" in m:
        d = m["domain"]
        print(f"\nDomain ({args.domain_count} questions):")
        print(
            f"  {args.model_a_label} mean: {d['model_a_mean_score']:.2f}  |  "
            f"{args.model_b_label} mean: {d['model_b_mean_score']:.2f}"
        )
        ci = d["median_diff_ci_95"]
        print(
            f"  Median diff (B - A): {d['median_diff']:+.2f} "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
        )
        print(
            f"  B wins: {d['model_b_wins']} | "
            f"A wins: {d['model_a_wins']} | "
            f"Ties: {d['ties']}"
        )
        print(f"  Wilcoxon p={d['wilcoxon_p_value']:.4f}")

    if "general" in m:
        g = m["general"]
        print(f"\nGeneral Knowledge ({len(result.general_results)} questions):")
        print(
            f"  {args.model_a_label} mean: {g['model_a_mean_score']:.2f}  |  "
            f"{args.model_b_label} mean: {g['model_b_mean_score']:.2f}"
        )
        flag = "YES" if g.get("forgetting_flag") else "No"
        print(f"  Catastrophic forgetting: {flag}")

    if "position_bias" in m:
        pb = m["position_bias"]
        print(
            f"\nPosition Bias: {pb['disagree_rate']:.1%} disagree rate "
            f"({pb['n_disagree_gt1']}/{pb['n_checked']})"
        )

    print(f"\nOutput: {output_dir}/")
    for label, path in written.items():
        print(f"  [{label}] {path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
