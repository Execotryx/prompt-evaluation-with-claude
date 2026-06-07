# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run the LangGraph agent with defaults (score 9.5, 10 iterations, stagnation 3)
python main.py

# Run the deterministic no-AI refinement demo
python main.py --demo

# Resume an isolated live run
python main.py --resume-run-id 20260606T143000Z-a1b2c3d4

# Preview/delete live-run folders older than 24 hours
python scripts/cleanup_runs.py --dry-run
python scripts/cleanup_runs.py

# Run with custom stopping limits
python main.py --score 9.0 --iterations 5 --stagnation 2

# Use a specific checkpoint thread / SQLite database
python main.py --thread-id prompt-evaluation-dev --checkpoint-db langgraph_checkpoints.sqlite

# Increase one stage's model output budget
python main.py --refinement-token-multiplier 4.0

# Static type check
uv run --python C:\Users\Execotryx\AppData\Local\Programs\Python\Python313\python.exe pyright

# Compile check
uv run --python C:\Users\Execotryx\AppData\Local\Programs\Python\Python313\python.exe python -m py_compile main.py run_storage.py scripts\cleanup_runs.py tests\test_langgraph_agent.py tests\test_cleanup_runs.py tests\test_run_storage.py

# Unit tests
uv run --python C:\Users\Execotryx\AppData\Local\Programs\Python\Python313\python.exe python -m unittest discover -s tests
```

Requires `ANTHROPIC_API_KEY` in a `.env` file, except when running `python main.py --demo`.

## Code Style

Use type hints for Python code, including function parameters, return values, class attributes, and non-obvious local variables.

Use Google-style docstrings for Python code. Public functions, methods, classes, and non-trivial private helpers should document parameters with `Args:`, returned values with `Returns:`, and raised exceptions with `Raises:` when applicable.

Keep refactors conservative. Prefer explicit typed state updates over dynamic dict-key tricks because Pyright is strict on `TypedDict` usage.

## Architecture

The LangGraph agent and model-call helpers live in [main.py](main.py). Shared artifact names, run paths, manifest timestamps, and archive creation live in [run_storage.py](run_storage.py); both the agent and cleanup script use that storage contract.

The graph flow:

1. **load_or_generate_dataset** - creates or loads `evaluation_dataset.json` (10 tasks); uses `claude-haiku-4-5`
2. **load_or_generate_solutions** - creates or loads `solutions.json`; uses `claude-sonnet-4-6`
3. **load_or_evaluate_solutions** - creates or loads `evaluation_results.json`; uses `claude-sonnet-4-6`
4. **initialize_best** - records the initial best dataset, results, and mean score
5. **decide_next_step** - routes to stop or another refinement pass
6. **refine_dataset** - rewrites `solution_criteria` for weak tasks; task descriptions are never modified
7. **generate_refined_solutions** - creates or loads `refined_solutions_N.json`
8. **evaluate_refined_solutions** - creates or loads `refined_evaluation_results_N.json`
9. **update_best_or_stagnation** - saves improvements to `best_dataset.json` or increments stagnation
10. **finalize** - reports the stop reason and preserves error details when present

Each refinement iteration writes numbered files (`refined_dataset_N.json`, `refined_solutions_N.json`, `refined_evaluation_results_N.json`). The best-scoring dataset is always saved to `best_dataset.json`.

Demo mode (`--demo`) replays the real-run artifacts stored in `demo_data/` and never constructs `ClaudeClient`. It reads the initial files and each numbered refinement round from that folder, writes isolated root-level demo-prefixed copies (`demo_evaluation_dataset.json`, `demo_solutions.json`, `demo_evaluation_results.json`, `demo_refined_dataset_N.json`, `demo_refined_solutions_N.json`, `demo_refined_evaluation_results_N.json`, `demo_best_dataset.json`), and defaults to thread `prompt-evaluation-demo` with checkpoint database `demo_langgraph_checkpoints.sqlite`.

Every live API invocation creates `runs/<run-id>/` and stores all artifacts plus the default checkpoint there. Finalized successful and failed runs contain `run_manifest.json` and `artifacts.zip`. Core link generation is abstracted behind `DownloadLinkProvider`; the CLI adapts `--download-base-url` or `DOWNLOAD_BASE_URL` into `BaseUrlDownloadLinkProvider`. Use `--resume-run-id` to reopen an existing run. `scripts/cleanup_runs.py` removes all direct-child run folders older than 24 hours by default and supports `--dry-run`.

Keep storage behavior centralized: use `RunArtifactPaths` for normal/demo layouts, `run_storage.py` for manifest/archive operations, `_finalize_live_run(...)` for all live-run outcomes, and `_replay_demo_artifact(...)` for checked-in demo fixture replay.

The graph stops when the mean score reaches `--score`, `--iterations` is exhausted, or `--stagnation` consecutive non-improving iterations occur. Each iteration always refines from the **best-scoring dataset so far**, not the most recent one.

If a model call fails because the response was truncated with `stop_reason='max_tokens'`, the graph routes through `apply_token_multiplier_correction`. That node increases only the failed stage's token multiplier by exactly `1.0`, clears the error, and retries the failed stage until `--max-token-corrections` is reached.

### Key Classes And Helpers

- **ClaudeClient** - thin wrapper around the Anthropic Messages API; maintains per-instance conversation history; call `reset()` between independent requests; raises `RuntimeError` on truncated responses (`stop_reason == "max_tokens"`)
- **BaseAgent** - abstract base requiring subclasses to implement `_build_prompt(**kwargs)` and `_build_output_config()`
- **EvaluationDatasetGenerator**, **SolutionGenerator**, **PromptEvaluator**, **TaskRefiner** - model-call wrappers with structured JSON output schemas
- **AgentState** - LangGraph `TypedDict` carrying datasets, solutions, evaluation results, best score, counters, token multipliers, artifact paths, current stage, retry stage, errors, and stop reason
- **RunArtifactPaths** - shared definition of run artifact, checkpoint, manifest, and archive paths
- **PipelineConfig** - typed CLI configuration passed through the CLI adapter
- **_stage_success(...)** and **_node_error(...)** - centralize successful and failed node state updates
- **_generate_solutions_node(...)** and **_evaluate_solutions_node(...)** - shared helpers for initial and refined solution/evaluation nodes
- **_finalize_live_run(...)** and **_replay_demo_artifact(...)** - centralize live finalization and demo replay
- **build_prompt_evaluation_agent(checkpoint_db)** - builds and compiles the LangGraph app using a SQLite checkpointer

All model-call helpers use structured JSON output via `output_config` (passes `betas=["output-128k-2025-02-19"]` and a JSON schema to the API).

### Caching Behavior

Every model-backed helper checks whether its output file exists before calling the model. If the file is present, it is loaded directly and the API call is skipped. To force a stage to re-run, delete its output file:

```bash
# Re-run evaluation only
del evaluation_results.json

# Re-run a specific refinement iteration
del refined_dataset_1.json refined_solutions_1.json refined_evaluation_results_1.json
```

LangGraph also stores orchestration state in `langgraph_checkpoints.sqlite` by default. Use a new `--thread-id` for a fresh checkpoint thread when needed.

Demo mode stores its orchestration state in `demo_langgraph_checkpoints.sqlite` by default and keeps demo artifacts separate from normal cached API artifacts.

### Models And Token Budgets

```python
MODEL = "claude-sonnet-4-6"          # solutions, evaluation, refinement
DATASET_MODEL = "claude-haiku-4-5"   # initial dataset generation
MAX_TOKENS = 1024                    # base; per-call budget = ceil(MAX_TOKENS * token_multiplier)
```

Default token multipliers:

- `--dataset-token-multiplier`: `2.0`
- `--solution-token-multiplier`: `2.0`
- `--evaluation-token-multiplier`: `1.0`
- `--refinement-token-multiplier`: `2.0`
- `--max-token-corrections`: `3`
