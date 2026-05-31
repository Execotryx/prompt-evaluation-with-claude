# Prompt Evaluation

A LangGraph-backed prompt evaluation agent that generates, evaluates, and refines solution criteria for AWS coding tasks using the Claude API.

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

# Increase only the refinement token budget
python main.py --refinement-token-multiplier 4.0

# Use a custom LangGraph checkpoint thread / SQLite database
python main.py --thread-id prompt-evaluation-dev --checkpoint-db langgraph_checkpoints.sqlite
```

| Flag | Default | Description |
|---|---|---|
| `--score` | `9.5` | Target mean evaluation score (1–10) |
| `--iterations` | `10` | Maximum refinement iterations |
| `--stagnation` | `3` | Stop after N consecutive non-improving iterations |
| `--thread-id` | `prompt-evaluation` | LangGraph checkpoint thread ID used for resumable runs |
| `--checkpoint-db` | `langgraph_checkpoints.sqlite` | SQLite database path used for LangGraph checkpoints |
| `--dataset-token-multiplier` | `2.0` | Token multiplier for dataset generation |
| `--solution-token-multiplier` | `2.0` | Token multiplier for solution generation |
| `--evaluation-token-multiplier` | `1.0` | Token multiplier for solution evaluation |
| `--refinement-token-multiplier` | `2.0` | Token multiplier for criteria refinement |
| `--max-token-corrections` | `3` | Maximum automatic max-token corrections; each correction increases only the failed stage multiplier by exactly `1.0` |

## How It Works

The agent runs a deterministic LangGraph policy graph across four model-backed stages:

1. **Dataset generation** — creates `evaluation_dataset.json` with 10 AWS coding tasks (runs once; uses `claude-haiku-4-5`)
2. **Solution generation** — generates a model solution per task (`claude-sonnet-4-6`); cached to `solutions.json`
3. **Evaluation** — scores each solution 1–10 with structured strengths/weaknesses; cached to `evaluation_results.json`
4. **Refinement** — rewrites `solution_criteria` for weak tasks based on evaluator feedback; task descriptions are never changed

Each refinement iteration produces numbered output files (`refined_dataset_N.json`, `refined_solutions_N.json`, `refined_evaluation_results_N.json`). The best-scoring dataset across all iterations is saved to `best_dataset.json`.

The graph stops when the mean score reaches the `--score` threshold, `--iterations` is exhausted, or `--stagnation` consecutive iterations show no improvement. LangGraph also stores orchestration state in the configured SQLite checkpoint database after graph steps, keyed by `--thread-id`.

If a model call is truncated with `stop_reason='max_tokens'`, the graph applies a corrective action before stopping: it increases the failed stage's token multiplier by exactly `1.0` and retries that same stage, up to `--max-token-corrections`.

## Architecture

The project is implemented as a single module (`main.py`) with four model-call helper classes that each inherit from `BaseAgent`:

| Class | Role | Model |
|---|---|---|
| `EvaluationDatasetGenerator` | Generates the initial 10-task dataset (runs once) | `claude-haiku-4-5` |
| `SolutionGenerator` | Produces a model solution per task | `claude-sonnet-4-6` |
| `PromptEvaluator` | Scores each solution 1–10 with structured strengths/weaknesses | `claude-sonnet-4-6` |
| `TaskRefiner` | Rewrites `solution_criteria` for weak tasks based on evaluator feedback | `claude-sonnet-4-6` |

`ClaudeClient` is a thin wrapper around the Anthropic Messages API that maintains per-instance conversation history for multi-turn interactions and raises `RuntimeError` on truncated responses. All model-call helpers use structured JSON output via the `output-128k-2025-02-19` beta.

The LangGraph agent always refines from the **best-scoring dataset seen so far**, not the most recent one. If a refinement produces a lower score the best dataset is kept and the stagnation counter increments.

`build_prompt_evaluation_agent(checkpoint_db)` builds the compiled LangGraph app. The graph state tracks datasets, solutions, evaluations, best score, iteration count, stagnation count, current stage, errors, stop reason, and artifact paths.

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
| `langgraph_checkpoints.sqlite` | Default SQLite checkpoint database for LangGraph orchestration state |
