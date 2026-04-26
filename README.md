# Prompt Evaluation

An iterative prompt evaluation pipeline that generates, evaluates, and refines solution criteria for AWS coding tasks using the Claude API.

## Setup

**Prerequisites:** Python ≥ 3.12, [uv](https://github.com/astral-sh/uv), and an [Anthropic API key](https://console.anthropic.com/).

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Create a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

## Usage

```bash
# Run with defaults (target score 9.5, up to 10 iterations)
python main.py

# Customize thresholds
python main.py --score 9.0 --iterations 5 --stagnation 2
```

| Flag | Default | Description |
|---|---|---|
| `--score` | `9.5` | Target mean evaluation score (1–10) |
| `--iterations` | `10` | Maximum refinement iterations |
| `--stagnation` | `3` | Stop after N consecutive non-improving iterations |

## How It Works

The pipeline runs an iterative loop across four stages:

1. **Dataset generation** — creates `evaluation_dataset.json` with 10 AWS coding tasks (runs once; uses `claude-haiku-4-5`)
2. **Solution generation** — generates a model solution per task (`claude-sonnet-4-6`); cached to `solutions.json`
3. **Evaluation** — scores each solution 1–10 with structured strengths/weaknesses; cached to `evaluation_results.json`
4. **Refinement** — rewrites `solution_criteria` for weak tasks based on evaluator feedback; task descriptions are never changed

Each refinement iteration produces numbered output files (`refined_dataset_N.json`, `refined_solutions_N.json`, `refined_evaluation_results_N.json`). The best-scoring dataset across all iterations is saved to `best_dataset.json`.

The loop stops when the mean score reaches the `--score` threshold, `--iterations` is exhausted, or `--stagnation` consecutive iterations show no improvement.

## Architecture

The pipeline is implemented as a single module (`main.py`) with four agent classes that each inherit from `BaseAgent`:

| Class | Role | Model |
|---|---|---|
| `EvaluationDatasetGenerator` | Generates the initial 10-task dataset (runs once) | `claude-haiku-4-5` |
| `SolutionGenerator` | Produces a model solution per task | `claude-sonnet-4-6` |
| `PromptEvaluator` | Scores each solution 1–10 with structured strengths/weaknesses | `claude-sonnet-4-6` |
| `TaskRefiner` | Rewrites `solution_criteria` for weak tasks based on evaluator feedback | `claude-sonnet-4-6` |

`ClaudeClient` is a thin wrapper around the Anthropic Messages API that maintains per-instance conversation history for multi-turn interactions and raises `RuntimeError` on truncated responses. All agents use structured JSON output via the `output-128k-2025-02-19` beta.

The `main()` loop always refines from the **best-scoring dataset seen so far**, not the most recent one. If a refinement produces a lower score the best dataset is kept and the stagnation counter increments.

## Output Files

| File | Description |
|---|---|
| `evaluation_dataset.json` | Initial task dataset (generated once) |
| `solutions.json` | Solutions for the initial dataset |
| `evaluation_results.json` | Scores and feedback for initial solutions |
| `refined_dataset_N.json` | Refined dataset after iteration N |
| `refined_solutions_N.json` | Solutions for refined dataset at iteration N |
| `refined_evaluation_results_N.json` | Evaluation results for iteration N |
| `best_dataset.json` | Best-scoring dataset across all iterations |
