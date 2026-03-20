# pairwise-llm-eval

Blinded pairwise LLM evaluation with position bias controls and statistical rigour.

## Why use this?

When comparing two LLM endpoints (e.g. base vs fine-tuned, different model sizes, different prompts), naive side-by-side judging is biased by presentation order: whichever answer the judge sees first tends to score higher. Most evaluation scripts ignore this.

This library runs blinded A/B evaluation where the judge sees answers in randomized order, then re-judges a subset with swapped order to measure the bias directly. Results include Wilcoxon signed-rank tests, FDR-corrected per-category breakdowns, and bootstrap confidence intervals so you can tell whether differences are real or noise.

**Use this when you are:**
- Comparing a fine-tuned model against its base to decide whether the fine-tune was worth it
- Evaluating two model configurations, prompts, or quantization levels head-to-head
- Checking for catastrophic forgetting after domain fine-tuning
- Running evaluations that need to hold up to statistical scrutiny

## Features

- **Blinded A/B evaluation**: presentation order is randomised per question to prevent anchoring bias
- **Position bias detection**: a configurable subset of questions is re-judged with swapped order; disagree rate is reported
- **Statistical rigour**: Wilcoxon signed-rank test, Benjamini-Hochberg FDR correction, bootstrap confidence intervals
- **Catastrophic forgetting detection**: 100 built-in general knowledge questions across 10 categories
- **Pluggable question sources**: in-memory list, JSONL file, or any PostgreSQL query via the `DatabaseProvider`
- **Generic**: works with any OpenAI-compatible inference server (llama.cpp, vLLM, LiteLLM, Ollama, etc.)

## Installation

```bash
pip install pairwise-llm-eval

# With PostgreSQL provider support
pip install pairwise-llm-eval[database]
```

## Quick start

### Command line

```bash
pairwise-llm-eval \
  --model-a-url http://localhost:8080 \
  --model-b-url http://localhost:8081 \
  --judge-url   http://localhost:8082 \
  --domain-questions questions.jsonl \
  --model-a-label baseline \
  --model-b-label fine-tuned \
  --output-dir ./results
```

Domain questions JSONL format (one JSON object per line):

```json
{"id": "q001", "question": "...", "gold_answer": "...", "category": "..."}
```

### Python API

```python
from pairwise_llm_eval import PairwiseAssessor, MemoryProvider, Question

questions = [
    Question(id="q1", question="What is the capital of France?",
             gold_answer="Paris", category="geography"),
]

assessor = PairwiseAssessor(
    model_a_url="http://localhost:8080",
    model_b_url="http://localhost:8081",
    judge_url="http://localhost:8082",
    model_a_label="baseline",
    model_b_label="fine-tuned",
)

# Domain-only evaluation (no general knowledge / forgetting check)
result = assessor.run_full(MemoryProvider(questions), n_domain=len(questions), n_general=0)
print(result.metrics)

# Include catastrophic forgetting check (100 built-in general knowledge questions)
result = assessor.run_full(MemoryProvider(questions), n_domain=len(questions), n_general=50)
```

## Output files

`write_report()` produces files in the output directory:

| File | Contents |
|------|----------|
| `assessment_summary.json` | Full metrics dict (aggregate + per-category + position bias) |
| `assessment_details.jsonl` | One JSON object per question with both model responses and judge scores |
| `position_bias_check.json` | Swap-adjudication breakdown (only written when position bias checking is enabled) |
| `assessment_report.md` | Human-readable Markdown summary with tables |

## Statistical methods

- **Wilcoxon signed-rank test**: non-parametric paired comparison on the score differences; two-sided, reported per-aggregate and per-category
- **Benjamini-Hochberg FDR**: applied across per-category p-values to control the false discovery rate
- **Bootstrap CI**: 10,000-resample bootstrap for the median score difference and win rate (95% CI)
- **Position bias rate**: fraction of swap-adjudicated questions where scores changed by more than 1 point in either direction

## License

Apache-2.0
