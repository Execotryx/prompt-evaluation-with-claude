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

# Run the deterministic no-AI refinement demo
python main.py --demo

# Resume a previous live run
python main.py --resume-run-id 20260606T143000Z-a1b2c3d4

# Form public archive links for a statically served runs directory
python main.py --download-base-url https://downloads.example.com/runs

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
| `--checkpoint-db` | run-local | Override the live run's `langgraph_checkpoints.sqlite` path |
| `--demo` | `False` | Replay the existing real-run artifacts without AI API calls |
| `--runs-root` | `runs` | Parent directory for isolated live-run folders |
| `--resume-run-id` | none | Resume an existing live run and reuse its artifacts/checkpoint |
| `--download-base-url` | `DOWNLOAD_BASE_URL` | Public base URL used to form the run ZIP download URL |
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

Use `python main.py --demo` to replay the real-run artifacts stored in `demo_data/` without constructing an Anthropic client. It reads the initial dataset, solutions, evaluation, and each numbered refinement round from that folder, then copies them into isolated root-level `demo_*.json` outputs while the normal graph evaluates improvement and stopping behavior. The default demo thread and checkpoint database are `prompt-evaluation-demo` and `demo_langgraph_checkpoints.sqlite`.

Every live API invocation creates a fresh `runs/<run-id>/` directory. All generated JSON artifacts and the default SQLite checkpoint stay inside that folder, so concurrent runs cannot reuse or overwrite each other's caches. A completed or failed run writes `run_manifest.json` and `artifacts.zip`; the ZIP includes the manifest and JSON artifacts but excludes the checkpoint database.

Download-link generation is provider-based for programmatic callers: pass an implementation of `DownloadLinkProvider` to `main(download_link_provider=...)`. The provider receives the run ID and finalized archive path, so it can return a public URL, signed storage URL, application route, or another download reference. The CLI is one adapter: `--download-base-url` or `DOWNLOAD_BASE_URL` creates a `BaseUrlDownloadLinkProvider`; without one, the CLI prints the archive's absolute local path.

Use `--resume-run-id` to reopen an existing run folder. The saved thread ID, checkpoint, and artifacts are reused, and the manifest and archive are regenerated after finalization.

## Architecture

The LangGraph pipeline and model-call helpers live in `main.py`. Shared live-run artifact names, paths, manifest timestamps, and ZIP creation live in `run_storage.py`; both the pipeline and `scripts/cleanup_runs.py` use that contract so storage behavior is defined once.

| Class | Role | Model |
|---|---|---|
| `EvaluationDatasetGenerator` | Generates the initial 10-task dataset (runs once) | `claude-haiku-4-5` |
| `SolutionGenerator` | Produces a model solution per task | `claude-sonnet-4-6` |
| `PromptEvaluator` | Scores each solution 1–10 with structured strengths/weaknesses | `claude-sonnet-4-6` |
| `TaskRefiner` | Rewrites `solution_criteria` for weak tasks based on evaluator feedback | `claude-sonnet-4-6` |

`ClaudeClient` is a thin wrapper around the Anthropic Messages API that maintains per-instance conversation history for multi-turn interactions and raises `RuntimeError` on truncated responses. All model-call helpers use structured JSON output via the `output-128k-2025-02-19` beta.

The LangGraph agent always refines from the **best-scoring dataset seen so far**, not the most recent one. If a refinement produces a lower score the best dataset is kept and the stagnation counter increments.

`build_prompt_evaluation_agent(checkpoint_db)` builds the compiled LangGraph app. The graph state tracks datasets, solutions, evaluations, best score, iteration count, stagnation count, current stage, errors, stop reason, and artifact paths.

Graph construction is split into small helpers:

- `_add_graph_nodes(workflow)` registers the node functions.
- `_add_stage_edges(workflow)` wires deterministic and conditional routes from shared route tables.
- `_compile_graph(workflow, checkpoint_db)` attaches the SQLite checkpointer and compiles the app.

Node state updates use `_stage_success(...)` and `_node_error(...)` so stale error fields are cleared consistently. Initial and refined solution/evaluation stages share `_generate_solutions_node(...)` and `_evaluate_solutions_node(...)`, with only the artifact path changing between initial and refined runs.

`RunArtifactPaths` defines normal and demo artifact layouts. `_finalize_live_run(...)` handles manifest and archive finalization for both successful and failed live runs. Demo artifact replay shares `_replay_demo_artifact(...)`, and CLI parsing produces one typed `PipelineConfig`.

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

Demo mode reads the corresponding real-run fixtures from `demo_data/` and writes demo-prefixed copies: `demo_evaluation_dataset.json`, `demo_solutions.json`, `demo_evaluation_results.json`, `demo_refined_dataset_N.json`, `demo_refined_solutions_N.json`, `demo_refined_evaluation_results_N.json`, `demo_best_dataset.json`, and `demo_langgraph_checkpoints.sqlite`.

Live-run output is stored under `runs/<run-id>/` and includes the normal JSON artifacts plus `run_manifest.json`, `artifacts.zip`, and the default `langgraph_checkpoints.sqlite`.

## Cleaning Up Runs

Delete all run folders older than 24 hours:

```bash
python scripts/cleanup_runs.py
python scripts/cleanup_runs.py --dry-run
python scripts/cleanup_runs.py --runs-root D:\prompt-runs --max-age-hours 48
```

Cron example, running hourly:

```cron
0 * * * * cd /path/to/prompt-evaluation && uv run python scripts/cleanup_runs.py
```

Windows Task Scheduler action:

```text
Program: uv
Arguments: run python scripts\cleanup_runs.py --runs-root runs --max-age-hours 24
Start in: E:\sources\python\prompt-evaluation
```

## Development

Use type hints for Python code and Google-style docstrings for public functions, methods, classes, and non-trivial private helpers. Docstrings should include `Args:`, `Returns:`, and `Raises:` when applicable.

Run the checks before handing off changes:

```bash
uv run --python C:\Users\Execotryx\AppData\Local\Programs\Python\Python313\python.exe pyright
uv run --python C:\Users\Execotryx\AppData\Local\Programs\Python\Python313\python.exe python -m py_compile main.py run_storage.py scripts\cleanup_runs.py tests\test_langgraph_agent.py tests\test_cleanup_runs.py tests\test_run_storage.py
uv run --python C:\Users\Execotryx\AppData\Local\Programs\Python\Python313\python.exe python -m unittest discover -s tests
```

The tests cover graph routing, cache reuse, max-token correction, checkpoint error recovery, live-run storage/finalization, cleanup safety, and the shared initial/refined solution and evaluation node helpers.
