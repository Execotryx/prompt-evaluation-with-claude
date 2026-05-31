# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
# Run the pipeline with defaults (score 9.5, 10 iterations, stagnation 3)
python main.py

# Run with custom parameters
python main.py --score 9.0 --iterations 5 --stagnation 2

# Install dependencies (uses uv)
uv sync
```

Requires `ANTHROPIC_API_KEY` in a `.env` file.

## Code style

Use type hints for Python code, including function parameters, return values, class attributes, and non-obvious local variables. Use Google-style docstrings for Python code. Public functions, methods, classes, and non-trivial private helpers should document parameters with `Args:`, returned values with `Returns:`, and raised exceptions with `Raises:` when applicable.

## Architecture

Single-file pipeline ([main.py](main.py)) implementing an iterative prompt evaluation loop for AWS coding tasks. The flow:

1. **EvaluationDatasetGenerator** — creates `evaluation_dataset.json` (10 tasks) once; uses `Codex-haiku-4-5` for cost efficiency
2. **SolutionGenerator** — generates model solutions per task using `Codex-sonnet-4-6`; caches to `solutions.json`
3. **PromptEvaluator** — scores each solution 1–10 with structured strengths/weaknesses; caches to `evaluation_results.json`
4. **TaskRefiner** — rewrites `solution_criteria` for low-scoring tasks based on weaknesses; task descriptions are never modified

Each refinement iteration writes numbered files (`refined_dataset_N.json`, `refined_solutions_N.json`, `refined_evaluation_results_N.json`). The best-scoring dataset is always saved to `best_dataset.json`.

The loop stops when: mean score reaches `--score` threshold, `--iterations` max is hit, or `--stagnation` consecutive non-improving iterations occur. Each iteration always refines from the **best-scoring dataset so far**, not the most recent one.

### Key classes

- **ClaudeClient** — thin wrapper around the Anthropic Messages API; maintains per-instance conversation history; call `reset()` between independent requests; raises `RuntimeError` on truncated responses (`stop_reason == "max_tokens"`)
- **BaseAgent** — abstract base requiring subclasses to implement `_build_prompt(**kwargs)` and `_build_output_config()`
- All agents use structured JSON output via `output_config` (passes `betas=["output-128k-2025-02-19"]` and a JSON schema to the API)

### Caching behavior

Every agent checks whether its output file exists before calling the model — if the file is present it is loaded directly, skipping all API calls. To force a stage to re-run, delete its output file:

```bash
# Re-run evaluation only
del evaluation_results.json

# Re-run a specific refinement iteration
del refined_dataset_1.json refined_solutions_1.json refined_evaluation_results_1.json
```

### Models & token budgets

```python
MODEL = "Codex-sonnet-4-6"      # solutions, evaluation, refinement
DATASET_MODEL = "Codex-haiku-4-5"  # initial dataset generation
MAX_TOKENS = 1024                # base; per-call budget = ceil(MAX_TOKENS * token_multiplier)
```
