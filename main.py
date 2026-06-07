"""Iterative prompt evaluation pipeline for AWS-related code generation tasks.

Generates an evaluation dataset of tasks, produces model solutions, evaluates
them on a 1–10 scale, and repeatedly refines the solution criteria based on
identified weaknesses until a configurable mean-score threshold is reached or
the refinement budget (iterations / stagnation limit) is exhausted.

Module-level constants
----------------------
MODEL : str
    Default model used for solution generation, evaluation, and refinement.
DATASET_MODEL : str
    Lighter model used only for seeding the initial evaluation dataset.
MAX_TOKENS : int
    Base output-token budget; per-call budgets are derived via ``token_multiplier``.
SCORE_THRESHOLD : float
    Default mean-score target (1–10) at which refinement stops.
MAX_REFINEMENT_ITERATIONS : int
    Hard cap on the number of refinement passes.
MAX_STAGNATION_ITERATIONS : int
    Refinement stops early after this many consecutive non-improving passes.
BEST_DATASET_FILE : str
    File path where the highest-scoring dataset is persisted.
"""
import os
import math
import argparse
import sqlite3
import re
import uuid
import anthropic
from abc import ABC, abstractmethod
from collections.abc import Hashable
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
from typing import Literal, Any, Protocol, TypeVar, TypedDict, cast
from json import dumps, loads
from tqdm import tqdm
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from run_storage import (
    BEST_DATASET_FILE,
    DATASET_FILE as INITIAL_DATASET_FILE,
    DEFAULT_CHECKPOINT_DB,
    DEFAULT_RUNS_ROOT,
    EVALUATION_FILE as INITIAL_EVALUATION_FILE,
    RUN_ARCHIVE_FILE,
    RUN_MANIFEST_FILE,
    SOLUTIONS_FILE as INITIAL_SOLUTIONS_FILE,
    RunArtifactPaths,
    create_run_archive,
    load_manifest,
    utc_timestamp,
    write_manifest,
)

MODEL: str = "claude-sonnet-4-6"
DATASET_MODEL: str = "claude-haiku-4-5"
MAX_TOKENS: int = 1024
SCORE_THRESHOLD: float = 9.5
MAX_REFINEMENT_ITERATIONS: int = 10
MAX_STAGNATION_ITERATIONS: int = 3
DEMO_THREAD_ID: str = "prompt-evaluation-demo"
DEMO_ARTIFACTS: RunArtifactPaths = RunArtifactPaths.in_directory(".", prefix="demo_")
DEMO_SOURCE_ARTIFACTS: RunArtifactPaths = RunArtifactPaths.in_directory(Path(__file__).parent / "demo_data")
DEMO_DATASET_FILE: str = str(DEMO_ARTIFACTS.dataset)
DEMO_SOLUTIONS_FILE: str = str(DEMO_ARTIFACTS.solutions)
DEMO_EVALUATION_FILE: str = str(DEMO_ARTIFACTS.evaluation)
DEMO_BEST_DATASET_FILE: str = str(DEMO_ARTIFACTS.best_dataset)
DEMO_REFINED_DATASET_PATTERN: str = str(DEMO_ARTIFACTS.refined_dataset_pattern)
DEMO_REFINED_SOLUTIONS_PATTERN: str = str(DEMO_ARTIFACTS.refined_solutions_pattern)
DEMO_REFINED_EVALUATION_PATTERN: str = str(DEMO_ARTIFACTS.refined_evaluation_pattern)
DEMO_CHECKPOINT_DB: str = str(DEMO_ARTIFACTS.checkpoint)
DEMO_SOURCE_DATASET_FILE: str = str(DEMO_SOURCE_ARTIFACTS.dataset)
DEMO_SOURCE_SOLUTIONS_FILE: str = str(DEMO_SOURCE_ARTIFACTS.solutions)
DEMO_SOURCE_EVALUATION_FILE: str = str(DEMO_SOURCE_ARTIFACTS.evaluation)
DEMO_SOURCE_REFINED_DATASET_PATTERN: str = str(DEMO_SOURCE_ARTIFACTS.refined_dataset_pattern)
DEMO_SOURCE_REFINED_SOLUTIONS_PATTERN: str = str(DEMO_SOURCE_ARTIFACTS.refined_solutions_pattern)
DEMO_SOURCE_REFINED_EVALUATION_PATTERN: str = str(DEMO_SOURCE_ARTIFACTS.refined_evaluation_pattern)
DATASET_TOKEN_MULTIPLIER: float = 2.0
SOLUTION_TOKEN_MULTIPLIER: float = 2.0
EVALUATION_TOKEN_MULTIPLIER: float = 1.0
REFINEMENT_TOKEN_MULTIPLIER: float = 2.0
MAX_TOKEN_CORRECTIONS: int = 3
ValidatedArtifact = TypeVar("ValidatedArtifact")


class ClaudeClient:
    """Wrapper around the Anthropic Messages API with persistent conversation history.

    Maintains a running list of user/assistant message pairs that is sent with
    each request, enabling multi-turn conversations. Call ``reset()`` between
    independent requests to avoid accumulating irrelevant context.
    """

    @property
    def api_key(self) -> str:
        """str: The Anthropic API key, resolved lazily from the ``ANTHROPIC_API_KEY`` environment variable."""
        if not self.__api_key:
            key: str | None = os.getenv("ANTHROPIC_API_KEY")
            if key is not None:
                self.__api_key = key
        return self.__api_key

    @property
    def client(self) -> anthropic.Anthropic:
        """anthropic.Anthropic: The underlying Anthropic client used to make API calls."""
        return self.__client

    @property
    def messages(self) -> list[anthropic.types.MessageParam]:
        """list[anthropic.types.MessageParam]: The current conversation history sent with every request."""
        return self.__messages

    def __init__(self, model: str = MODEL) -> None:
        """Initialise the client, loading ``ANTHROPIC_API_KEY`` from the environment.

        Args:
            model (str): Model ID to use for all calls made through this client.
                Defaults to the module-level ``MODEL`` constant.
        """
        load_dotenv()
        self.__api_key: str = ""
        self.__model: str = model
        self.__client: anthropic.Anthropic = anthropic.Anthropic(api_key=self.api_key)
        self.__messages: list[anthropic.types.MessageParam] = []

    def __add_message(self, role: Literal["user", "assistant"], content: str) -> None:
        """Append a message with the given role to the conversation history.

        Args:
            role (Literal["user", "assistant"]): The role of the message sender.
            content (str): The message text.
        """
        self.messages.append(anthropic.types.MessageParam(role=role, content=content))

    def _add_user_message(self, content: str) -> None:
        """Append a user-role message to the conversation history.

        Args:
            content (str): The user message text.
        """
        self.__add_message(role="user", content=content)

    def _add_assistant_message(self, content: str) -> None:
        """Append an assistant-role message to the conversation history.

        Args:
            content (str): The assistant message text.
        """
        self.__add_message(role="assistant", content=content)

    def reset(self) -> None:
        """Clear the conversation history.

        Call this before sending an independent request to prevent prior
        turns from influencing the model's response and wasting tokens.
        """
        self.__messages = []

    def ask(self, question: str, token_multiplier: float = 1.0, temperature: float = 0.0, **kwargs: Any) -> str:
        """Send a message and return the model's text response.

        The question is appended to the conversation history before the API
        call, and the assistant's reply is appended afterwards, so subsequent
        calls continue the same conversation.

        Args:
            question (str): The user message to send.
            token_multiplier (float): Scales the module-level ``MAX_TOKENS`` constant to
                produce the actual ``max_tokens`` value sent to the API.
                The result is rounded up to the nearest integer, so fractional
                multipliers are safe to use.
            temperature (float): Sampling temperature (0.0 = deterministic).
            **kwargs (Any): Additional parameters forwarded to ``messages.create()``,
                e.g. ``output_config`` for structured output. ``stream`` is
                silently removed if present.

        Returns:
            str: The text content of the first content block in the response.

        Raises:
            RuntimeError: If the model's response was cut off due to hitting
                ``max_tokens``.
        """
        self._add_user_message(content=question.strip())

        max_tokens: int = math.ceil(MAX_TOKENS * token_multiplier)

        params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self.messages,
            "model": self.__model,
        }

        kwargs.pop("stream", None)
        if kwargs:
            params.update(kwargs)

        message: anthropic.types.Message = self.client.messages.create(**params, stream=False)

        if message.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Response was truncated (stop_reason='max_tokens'). "
                f"Increase token_multiplier (current: {token_multiplier}, max_tokens: {max_tokens})."
            )

        block: anthropic.types.TextBlock = message.content[0]  # type: ignore[assignment]
        self._add_assistant_message(content=block.text)
        return block.text


class BaseAgent(ABC):
    """Abstract base class for model-backed agents.

    Consolidates shared client initialisation and enforces a consistent
    interface for prompt and output-config construction across all
    concrete agent types.
    """

    def __init__(self, client: ClaudeClient) -> None:
        """Store the model client for use by subclass methods.

        Args:
            client (ClaudeClient): A ``ClaudeClient`` instance used to query the model.
        """
        self.__client: ClaudeClient = client

    @property
    def client(self) -> ClaudeClient:
        """ClaudeClient: The model client used to make API calls."""
        return self.__client

    @abstractmethod
    def _build_prompt(self, **kwargs: Any) -> str:
        """Build and return the prompt string for a model call.

        Args:
            **kwargs (Any): Prompt-specific context (e.g. task, solution, criteria).

        Returns:
            str: A formatted prompt string ready to pass to ``ClaudeClient.ask()``.
        """
        ...

    @abstractmethod
    def _build_output_config(self) -> dict[str, Any]:
        """Build and return the structured-output config for a model call.

        Returns:
            dict[str, Any]: A dict conforming to the ``output_config`` parameter
            accepted by ``ClaudeClient.ask()``.
        """
        ...


class EvaluationDatasetGenerator(BaseAgent):
    """Generates an evaluation dataset of AWS-related tasks.

    On first run the dataset is produced by the model and persisted to
    ``evaluation_dataset.json``. Subsequent runs load from that file, so
    the model is only called once unless the file is deleted.
    """

    DATASET_FILE: str = INITIAL_DATASET_FILE

    def __init__(self, client: ClaudeClient, output_file: str | None = None) -> None:
        """Initialise the generator with a model client and optional output path.

        Args:
            client (ClaudeClient): A ``ClaudeClient`` instance used to query the model.
            output_file (str | None): Path to write/read the dataset. Defaults to
                ``evaluation_dataset.json``.
        """
        super().__init__(client)
        self._output_file: str = output_file or self.DATASET_FILE

    def _build_prompt(self) -> str:  # type: ignore[override]
        """Return the prompt that asks the model to generate the task dataset."""
        return """
Generate a evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts
that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects,
each representing task that requires Python, JSON, or a Regex to complete. Think step by step when generating the tasks.

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a regular expression.
* Focus on tasks that do not require writing much code

Please generate 10 objects. Respond ONLY with the JSON array, without any additional text, explanation or code brackets.

Example output:
[
    {
        "task": "Description of task",
        "solution_criteria": "Description of what the solution should accomplish, e.g. 'The Python function should take a string as input and return a list of all email addresses found in the string.'"
    },
    ...additional
]
"""

    def _build_output_config(self) -> dict[str, Any]:
        """Return the structured-output config enforcing the dataset schema."""
        return {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "solution_criteria": {"type": "string"},
                        },
                        "required": ["task", "solution_criteria"],
                        "additionalProperties": False,
                    },
                },
            }
        }

    def create_dataset(self, token_multiplier: float = DATASET_TOKEN_MULTIPLIER) -> list[dict[str, Any]]:
        """Generate or load the evaluation dataset.

        If ``evaluation_dataset.json`` does not exist, the model is called to
        produce the dataset which is then saved to that file. If the file
        already exists it is loaded directly without calling the model.

        Args:
            token_multiplier (float): Multiplier applied to ``MAX_TOKENS`` for
                the dataset-generation model call.

        Returns:
            list[dict[str, Any]]: A list of dicts, each with ``"task"`` and
            ``"solution_criteria"`` keys.
        """
        if not os.path.exists(self._output_file):
            response: str = self.client.ask(
                question=self._build_prompt(),
                token_multiplier=token_multiplier,
                output_config=self._build_output_config(),
            )
            with open(self._output_file, "w") as f:
                f.write(dumps(loads(response), indent=4))
        else:
            with open(self._output_file, "r") as f:
                response = f.read()

        return loads(response)


class SolutionGenerator(BaseAgent):
    """Generates model solutions for each task in an evaluation dataset.

    Each solution is produced in a single model call. The generation prompt
    instructs the model to produce the most concise correct solution directly.

    Solutions are persisted to the configured output file after generation.
    On subsequent runs the file is loaded directly, skipping model calls.
    """

    DEFAULT_SOLUTIONS_FILE: str = INITIAL_SOLUTIONS_FILE

    def __init__(self, client: ClaudeClient, output_file: str | None = None) -> None:
        """Initialise the generator with a model client and optional output path.

        Args:
            client (ClaudeClient): A ``ClaudeClient`` instance used to query the model.
            output_file (str | None): Path to write/read solutions. Defaults to
                ``solutions.json``.
        """
        super().__init__(client)
        self._output_file: str = output_file or self.DEFAULT_SOLUTIONS_FILE

    def _build_prompt(self, **kwargs: Any) -> str:
        """Return a prompt asking the model to solve a single task.

        Args:
            **kwargs (Any):
                task (str): The task description.
                solution_criteria (str): Criteria the solution must satisfy.

        Returns:
            str: A formatted prompt string.
        """
        task: str = kwargs["task"]
        solution_criteria: str = kwargs["solution_criteria"]
        return (
            f"Solve the following task following its solution criteria.\n"
            f"Task: \n{task}\n"
            f"Solution Criteria: \n{solution_criteria}\n\n"
            f"If the solution is a Python script, include no comments or docstrings.\n"
            f"Write the most concise solution that fully satisfies the criteria — avoid all unnecessary complexity.\n"
            f"Think step by step and follow the solution criteria strictly."
        )

    def _build_output_config(self) -> dict[str, Any]:
        """Return the structured-output config enforcing the solution schema."""
        return {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "solution": {"type": "string"},
                    },
                    "required": ["solution"],
                    "additionalProperties": False,
                },
            }
        }

    def _solve_task(
        self,
        task: str,
        solution_criteria: str,
        token_multiplier: float = SOLUTION_TOKEN_MULTIPLIER,
    ) -> dict[str, str]:
        """Ask the model to solve a single task.

        Args:
            task (str): The task description.
            solution_criteria (str): Criteria the solution must satisfy.
            token_multiplier (float): Multiplier applied to ``MAX_TOKENS`` for
                this solution-generation model call.

        Returns:
            dict[str, str]: A dict with ``"task"`` and ``"solution"`` keys.
        """
        self.client.reset()
        response: str = self.client.ask(
            question=self._build_prompt(task=task, solution_criteria=solution_criteria),
            token_multiplier=token_multiplier,
            output_config=self._build_output_config(),
        )
        return {"task": task, "solution": loads(response)["solution"]}

    def generate_solutions(
        self,
        eval_dataset: list[dict[str, str]],
        token_multiplier: float = SOLUTION_TOKEN_MULTIPLIER,
    ) -> list[dict[str, str]]:
        """Generate solutions for every task in the evaluation dataset.

        If the output file already exists it is returned directly without
        calling the model. Otherwise each task is solved in sequence and the
        results are saved to the output file.

        Args:
            eval_dataset (list[dict[str, str]]): List of task dicts as produced by
                ``EvaluationDatasetGenerator.create_dataset()``.
            token_multiplier (float): Multiplier applied to ``MAX_TOKENS`` for
                each solution-generation model call.

        Returns:
            list[dict[str, str]]: A list of dicts, each with ``"task"`` and ``"solution"`` keys.
        """
        if os.path.exists(self._output_file):
            with open(self._output_file, "r") as f:
                return loads(f.read())

        solutions: list[dict[str, str]] = [
            self._solve_task(item["task"], item["solution_criteria"], token_multiplier)
            for item in tqdm(eval_dataset, desc="Generating solutions", unit="task")
        ]

        with open(self._output_file, "w") as f:
            f.write(dumps(solutions, indent=4))

        return solutions


class PromptEvaluator(BaseAgent):
    """Evaluates AI-generated solutions against their task and solution criteria.

    Each solution is scored by the model on a 1–10 scale with structured
    feedback. Results are persisted to the configured output file and loaded
    from there on subsequent runs.

    Each result dict includes a ``"task"`` key (added after the model call,
    not enforced by the JSON schema) so downstream consumers — such as
    ``TaskRefiner`` — can match results back to their source tasks by name.
    """

    DEFAULT_EVALUATION_FILE: str = INITIAL_EVALUATION_FILE

    def __init__(self, client: ClaudeClient, output_file: str | None = None) -> None:
        """Initialise the evaluator with a model client and optional output path.

        Args:
            client (ClaudeClient): A ``ClaudeClient`` instance used to query the model.
            output_file (str | None): Path to write/read evaluation results. Defaults to
                ``evaluation_results.json``.
        """
        super().__init__(client)
        self._output_file: str = output_file or self.DEFAULT_EVALUATION_FILE

    def _build_prompt(self, **kwargs: Any) -> str:
        """Return a prompt asking the model to evaluate a single solution.

        Args:
            **kwargs (Any):
                task (str): The original task description.
                solution (str): The AI-generated solution to evaluate.
                solution_criteria (str): The criteria the solution should satisfy.

        Returns:
            str: A formatted evaluation prompt string.
        """
        task: str = kwargs["task"]
        solution: str = kwargs["solution"]
        solution_criteria: str = kwargs["solution_criteria"]
        return f"""
You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

Original Task:
{task}

Solution Criteria:
{solution_criteria}

Solution to Evaluate:
{solution}

Provide your evaluation as a structured JSON object with the following fields, in this specific order:
- "strengths": An array of 1-3 key strengths
- "weaknesses": An array of 1-3 key areas for improvement
- "score": A number between 1-10 measuring how well the solution satisfies the criteria

Respond with JSON. Keep your response concise and direct. Think step by step and be specific in your feedback.
Example response shape:
{{
    "strengths": string[],
    "weaknesses": string[],
    "score": number
}}
    """

    def _build_output_config(self) -> dict[str, Any]:
        """Return the structured-output config enforcing the evaluation schema."""
        return {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "strengths": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "weaknesses": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["score", "strengths", "weaknesses"],
                    "additionalProperties": False,
                },
            }
        }

    def _evaluate_task(
        self,
        task: str,
        solution: str,
        solution_criteria: str,
        token_multiplier: float = EVALUATION_TOKEN_MULTIPLIER,
    ) -> dict[str, Any]:
        """Ask the model to evaluate a single solution and return structured feedback.

        Resets the client's conversation history before the call so that
        prior evaluations do not influence the response.

        The returned dict includes a ``"task"`` key (added after the model
        call, outside the JSON schema) so results can be matched back to
        their source task by name.

        Args:
            task (str): The original task description.
            solution (str): The AI-generated solution to evaluate.
            solution_criteria (str): The criteria the solution should satisfy.
            token_multiplier (float): Multiplier applied to ``MAX_TOKENS`` for
                this evaluation model call.

        Returns:
            dict[str, Any]: A dict with ``"task"`` (str), ``"score"`` (float),
            ``"strengths"`` (list[str]), and ``"weaknesses"`` (list[str]) keys.
        """
        self.client.reset()
        response: str = self.client.ask(
            question=self._build_prompt(task=task, solution=solution, solution_criteria=solution_criteria),
            token_multiplier=token_multiplier,
            output_config=self._build_output_config(),
        )
        result: dict[str, Any] = loads(response)
        result["task"] = task
        return result

    def evaluate_prompts(
        self,
        eval_dataset: list[dict[str, str]],
        solutions: list[dict[str, str]],
        token_multiplier: float = EVALUATION_TOKEN_MULTIPLIER,
    ) -> list[dict[str, Any]]:
        """Evaluate every solution in the solutions list against its task criteria.

        If the output file already exists it is returned directly. Otherwise
        each solution is evaluated in sequence and results are saved to the
        output file.

        Args:
            eval_dataset (list[dict[str, str]]): List of task dicts with ``"task"`` and
                ``"solution_criteria"`` keys.
            solutions (list[dict[str, str]]): List of solution dicts with ``"task"`` and
                ``"solution"`` keys, as produced by
                ``SolutionGenerator.generate_solutions()``.
            token_multiplier (float): Multiplier applied to ``MAX_TOKENS`` for
                each evaluation model call.

        Returns:
            list[dict[str, Any]]: A list of evaluation dicts, each containing
            ``"task"`` (str), ``"score"`` (float), ``"strengths"`` (list[str]),
            and ``"weaknesses"`` (list[str]).
        """
        if os.path.exists(self._output_file):
            with open(self._output_file, "r") as f:
                return loads(f.read())

        solution_map: dict[str, str] = {s["task"]: s["solution"] for s in solutions}

        evaluation_results: list[dict[str, Any]] = [
            self._evaluate_task(item["task"], solution_map[item["task"]], item["solution_criteria"], token_multiplier)
            for item in tqdm(eval_dataset, desc="Evaluating solutions", unit="task")
            if item["task"] in solution_map
        ]

        with open(self._output_file, "w") as f:
            f.write(dumps(evaluation_results, indent=4))

        return evaluation_results


class TaskRefiner(BaseAgent):
    """Refines the solution criteria of evaluation tasks based on identified weaknesses.

    The task description is never modified — only the solution criteria is
    rewritten in a single model call. The generation prompt instructs the model
    to produce concise, targeted criteria directly. The output is always a
    Markdown bulleted list.

    Refined tasks are saved to the configured output file. On subsequent
    runs the file is loaded directly, skipping model calls. After
    refinement, a new round of solution generation and evaluation should
    be run against the refined dataset to measure improvement.
    """

    DEFAULT_REFINED_FILE: str = "refined_dataset.json"

    def __init__(self, client: ClaudeClient, output_file: str | None = None) -> None:
        """Initialise the refiner with a model client and optional output path.

        Args:
            client (ClaudeClient): A ``ClaudeClient`` instance used to query the model.
            output_file (str | None): Path to write/read the refined dataset. Defaults to
                ``refined_dataset.json``.
        """
        super().__init__(client)
        self._output_file: str = output_file or self.DEFAULT_REFINED_FILE

    def _build_prompt(self, **kwargs: Any) -> str:
        """Return a prompt asking the model to rewrite the solution criteria for a task.

        The task description is provided for context only and must not be
        changed. Only the solution criteria is rewritten.

        Args:
            **kwargs (Any):
                task (str): The original task description (context only).
                solution_criteria (str): The original solution criteria.
                weaknesses (list[str]): Weaknesses identified in the
                    evaluated solution for this task.

        Returns:
            str: A formatted refinement prompt string.
        """
        task: str = kwargs["task"]
        solution_criteria: str = kwargs["solution_criteria"]
        weaknesses: list[str] = kwargs["weaknesses"]
        formatted_weaknesses: str = "\n".join(f"- {w}" for w in weaknesses)
        return f"""
You are an expert AWS prompt engineer. Rewrite the solution criteria below so that future
solutions are more likely to address the identified weaknesses.

Task (for context only — do not change):
{task}

Original Solution Criteria:
{solution_criteria}

Weaknesses identified in the evaluated solution:
{formatted_weaknesses}

Instructions:
* Rewrite ONLY the solution criteria. The task description must remain exactly as shown.
* Make the criteria more precise and explicit so each weakness is directly targeted —
  if a weakness was vagueness, add specificity; if it was missing error handling, require it;
  if it was an incorrect AWS API usage, call out the correct one.
* Keep the scope small: the task must remain solvable with a single Python function,
  a single JSON object, or a regular expression.
* Be concise: each bullet should be one specific, direct requirement — no redundant wording.
* Return the rewritten criteria as a Markdown bulleted list and nothing else.
"""

    def _build_output_config(self) -> dict[str, Any]:
        """Return the structured-output config enforcing the refined criteria schema.

        Only ``solution_criteria`` is present — the task is never part of the
        model output and is carried over from the original dataset unchanged.
        """
        return {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "solution_criteria": {"type": "string"},
                    },
                    "required": ["solution_criteria"],
                    "additionalProperties": False,
                },
            }
        }

    def _refine_task(
        self,
        task: str,
        solution_criteria: str,
        weaknesses: list[str],
        token_multiplier: float = REFINEMENT_TOKEN_MULTIPLIER,
    ) -> dict[str, str]:
        """Ask the model to rewrite the solution criteria for a task.

        The task description is passed to the model for context only and is
        never part of the output — it is copied from the input unchanged.
        The history is reset before the call so prior refinements do not bleed in.

        Args:
            task (str): The original task description (carried through unchanged).
            solution_criteria (str): The original solution criteria to rewrite.
            weaknesses (list[str]): Weaknesses identified in the evaluated solution.
            token_multiplier (float): Multiplier applied to ``MAX_TOKENS`` for
                this refinement model call.

        Returns:
            dict[str, str]: A dict with the original ``"task"`` and a rewritten
            ``"solution_criteria"`` as a Markdown bulleted list.
        """
        self.client.reset()
        response: str = self.client.ask(
            question=self._build_prompt(task=task, solution_criteria=solution_criteria, weaknesses=weaknesses),
            token_multiplier=token_multiplier,
            output_config=self._build_output_config(),
        )
        return {"task": task, "solution_criteria": loads(response)["solution_criteria"]}

    def refine_dataset(
        self,
        eval_dataset: list[dict[str, str]],
        evaluation_results: list[dict[str, Any]],
        token_multiplier: float = REFINEMENT_TOKEN_MULTIPLIER,
    ) -> list[dict[str, str]]:
        """Refine every task in the dataset using the corresponding evaluation weaknesses.

        Tasks are matched to their evaluation results by the ``"task"`` key
        present in each result (added by ``PromptEvaluator._evaluate_task``).
        Tasks with no matching evaluation result are carried over unchanged.

        If the output file already exists it is returned directly without
        calling the model. Otherwise each task is refined in sequence and
        the results are saved to the output file.

        Args:
            eval_dataset (list[dict[str, str]]): Original list of task dicts with
                ``"task"`` and ``"solution_criteria"`` keys.
            evaluation_results (list[dict[str, Any]]): Results produced by
                ``PromptEvaluator.evaluate_prompts()``, each containing
                ``"task"`` (str) and ``"weaknesses"`` (list[str]) keys.
            token_multiplier (float): Multiplier applied to ``MAX_TOKENS`` for
                each refinement model call.

        Returns:
            list[dict[str, str]]: A list of refined task dicts, each with ``"task"``
            and ``"solution_criteria"`` keys, in the same order as ``eval_dataset``.
        """
        if os.path.exists(self._output_file):
            with open(self._output_file, "r") as f:
                return loads(f.read())

        weakness_map: dict[str, list[str]] = {
            r["task"]: r["weaknesses"]
            for r in evaluation_results
            if "task" in r and "weaknesses" in r
        }

        refined: list[dict[str, str]] = [
            self._refine_task(item["task"], item["solution_criteria"], weakness_map[item["task"]], token_multiplier)
            if item["task"] in weakness_map
            else item
            for item in tqdm(eval_dataset, desc="Refining tasks", unit="task")
        ]

        with open(self._output_file, "w") as f:
            f.write(dumps(refined, indent=4))

        return refined


def _mean_score(evaluation_results: list[dict[str, Any]]) -> float:
    """Return the arithmetic mean of ``"score"`` across all evaluation results.

    Args:
        evaluation_results (list[dict[str, Any]]): Evaluation result dicts, each
            containing at least a numeric ``"score"`` key.

    Returns:
        float: Mean score across all results.

    Raises:
        ValueError: If ``evaluation_results`` is empty.
    """
    if not evaluation_results:
        raise ValueError("Cannot calculate a mean score from an empty evaluation result set.")
    return sum(r["score"] for r in evaluation_results) / len(evaluation_results)


def _save_best_dataset(dataset: list[dict[str, str]], output_file: str = BEST_DATASET_FILE) -> None:
    """Overwrite ``best_dataset.json`` with the current best dataset.

    Args:
        dataset (list[dict[str, str]]): The dataset to persist, where each dict
            contains ``"task"`` (str) and ``"solution_criteria"`` (str) keys.
        output_file (str): File path to write. Defaults to ``BEST_DATASET_FILE``.

    Returns:
        None.
    """
    with open(output_file, "w") as f:
        f.write(dumps(dataset, indent=4))


class AgentState(TypedDict, total=False):
    """Shared LangGraph state for the prompt-evaluation agent."""

    demo_mode: bool
    dataset: list[dict[str, str]]
    solutions: list[dict[str, str]]
    evaluation_results: list[dict[str, Any]]
    best_dataset: list[dict[str, str]]
    best_results: list[dict[str, Any]]
    best_score: float
    iteration: int
    stagnation_count: int
    score_threshold: float
    max_iterations: int
    max_stagnation: int
    token_correction_count: int
    max_token_corrections: int
    current_stage: str
    failed_stage: str | None
    retry_stage: str | None
    last_error: str | None
    stop_reason: str | None
    dataset_file: str
    solutions_file: str
    evaluation_file: str
    best_dataset_file: str
    refined_dataset_pattern: str
    refined_solutions_pattern: str
    refined_evaluation_pattern: str
    dataset_token_multiplier: float
    solution_token_multiplier: float
    evaluation_token_multiplier: float
    refinement_token_multiplier: float


class AgentErrorState(TypedDict):
    """Graph state update emitted when a node fails."""

    current_stage: str
    failed_stage: str
    retry_stage: None
    last_error: str
    stop_reason: str


class StageSuccessState(TypedDict, total=False):
    """Graph state update emitted when a node completes successfully."""

    current_stage: str
    failed_stage: str | None
    retry_stage: str | None
    last_error: str | None
    stop_reason: str | None
    dataset: list[dict[str, str]]
    solutions: list[dict[str, str]]
    evaluation_results: list[dict[str, Any]]
    best_dataset: list[dict[str, str]]
    best_results: list[dict[str, Any]]
    best_score: float
    iteration: int
    stagnation_count: int


class TokenCorrectionState(TypedDict, total=False):
    """Graph state update emitted after correcting a max-token error."""

    current_stage: str
    failed_stage: str | None
    retry_stage: str
    last_error: str | None
    stop_reason: str | None
    token_correction_count: int
    dataset_token_multiplier: float
    solution_token_multiplier: float
    evaluation_token_multiplier: float
    refinement_token_multiplier: float


class FinalState(TypedDict):
    """Graph state update emitted by the finalization node."""

    current_stage: str
    failed_stage: str | None
    last_error: str | None
    stop_reason: str


class RunResult(TypedDict):
    """Summary returned after a live or demo invocation."""

    run_id: str | None
    run_directory: str | None
    archive_path: str | None
    download_link: str | None
    status: str
    stop_reason: str


class DownloadLinkProvider(Protocol):
    """Builds a downloadable link for a finalized run archive."""

    def create_download_link(self, run_id: str, archive_path: Path) -> str | None:
        """Return a downloadable link for a run archive."""
        ...


class BaseUrlDownloadLinkProvider:
    """Builds run archive links beneath a configured public base URL."""

    def __init__(self, base_url: str) -> None:
        """Store the public base URL used for run archive links."""
        self._base_url: str = base_url.rstrip("/")

    def create_download_link(self, run_id: str, archive_path: Path) -> str:
        """Return the public URL for a run archive."""
        return f"{self._base_url}/{run_id}/{archive_path.name}"


def _state_value(state: AgentState, key: str, default: Any) -> Any:
    """Return a state value, falling back only when the key is absent or None.

    Args:
        state (AgentState): Current graph state.
        key (str): State key to read.
        default (Any): Value to return when the key is absent or maps to ``None``.

    Returns:
        Any: The stored state value or ``default``.
    """
    value = state.get(key)
    return default if value is None else value


def _refined_artifact_path(state: AgentState, key: str, default_pattern: str) -> str:
    """Return the artifact path for the next refinement iteration.

    Args:
        state (AgentState): Current graph state.
        key (str): State key containing a format pattern with ``{iteration}``.
        default_pattern (str): Pattern to use when the key is not present.

    Returns:
        str: Formatted artifact path for ``state["iteration"] + 1``.
    """
    next_iteration = int(_state_value(state, "iteration", 0)) + 1
    pattern: str = _state_value(state, key, default_pattern)
    return pattern.format(iteration=next_iteration)


def _validate_record_list(records: Any, collection_label: str, item_label: str) -> list[dict[str, Any]]:
    """Validate common non-empty record-list and task fields.

    Args:
        records (Any): Candidate record list.
        collection_label (str): Human-readable collection label for errors.
        item_label (str): Human-readable singular record label for errors.

    Returns:
        list[dict[str, Any]]: Records with validated object and task fields.

    Raises:
        ValueError: If the list or a common record field is invalid.
    """
    if not isinstance(records, list) or not records:
        raise ValueError(f"{collection_label} must be a non-empty list.")

    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"{item_label} {index} must be an object.")
        if not isinstance(item.get("task"), str) or not item["task"]:
            raise ValueError(f"{item_label} {index} must include a non-empty task string.")
    return records


def _validate_dataset(dataset: Any) -> list[dict[str, str]]:
    """Validate a task dataset before accepting it into graph state.

    Args:
        dataset (Any): Candidate dataset loaded from cache or generated by a model.

    Returns:
        list[dict[str, str]]: Validated dataset with ``task`` and
            ``solution_criteria`` strings.

    Raises:
        ValueError: If the dataset is empty or any item has an invalid shape.
    """
    records: list[dict[str, Any]] = _validate_record_list(dataset, "Dataset", "Dataset item")
    for index, item in enumerate(records):
        if not isinstance(item.get("solution_criteria"), str) or not item["solution_criteria"]:
            raise ValueError(f"Dataset item {index} must include a non-empty solution_criteria string.")

    return cast(list[dict[str, str]], records)


def _validate_solutions(solutions: Any) -> list[dict[str, str]]:
    """Validate generated solutions before accepting them into graph state.

    Args:
        solutions (Any): Candidate solutions loaded from cache or generated by a model.

    Returns:
        list[dict[str, str]]: Validated solution records with ``task`` and
            ``solution`` strings.

    Raises:
        ValueError: If the solution list is empty or any item has an invalid shape.
    """
    records: list[dict[str, Any]] = _validate_record_list(solutions, "Solutions", "Solution item")
    for index, item in enumerate(records):
        if not isinstance(item.get("solution"), str) or not item["solution"]:
            raise ValueError(f"Solution item {index} must include a non-empty solution string.")

    return cast(list[dict[str, str]], records)


def _validate_evaluation_results(results: Any) -> list[dict[str, Any]]:
    """Validate evaluation results before accepting them into graph state.

    Args:
        results (Any): Candidate evaluation results loaded from cache or generated
            by a model.

    Returns:
        list[dict[str, Any]]: Validated evaluation records containing ``task``,
            numeric ``score``, ``strengths``, and ``weaknesses``.

    Raises:
        ValueError: If the result list is empty or any item has an invalid shape.
    """
    records: list[dict[str, Any]] = _validate_record_list(
        results,
        "Evaluation results",
        "Evaluation result",
    )
    for index, item in enumerate(records):
        if isinstance(item.get("score"), bool) or not isinstance(item.get("score"), (int, float)):
            raise ValueError(f"Evaluation result {index} must include a numeric score.")
        if not isinstance(item.get("strengths"), list):
            raise ValueError(f"Evaluation result {index} must include a strengths list.")
        if not isinstance(item.get("weaknesses"), list):
            raise ValueError(f"Evaluation result {index} must include a weaknesses list.")

    return records


def _load_or_create_json(output_file: str, payload: Any) -> Any:
    """Load a JSON artifact if present, otherwise write and return a default payload.

    Args:
        output_file (str): Artifact path to read or create.
        payload (Any): JSON-serializable fallback payload.

    Returns:
        Any: Existing decoded JSON or the fallback payload after it is written.
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            return loads(f.read())

    with open(output_file, "w") as f:
        f.write(dumps(payload, indent=4))
    return payload


def _load_demo_source(source_file: str) -> Any:
    """Load a checked-in artifact produced by a real AI run.

    Args:
        source_file (str): Real-run artifact path to load.

    Returns:
        Any: Decoded JSON artifact.

    Raises:
        FileNotFoundError: If the required real-run artifact is missing.
    """
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Demo mode requires real-run artifact: {source_file}")
    with open(source_file, "r") as f:
        return loads(f.read())


def _replay_demo_artifact(
    source_file: str,
    output_file: str,
    validator: Callable[[Any], ValidatedArtifact],
) -> ValidatedArtifact:
    """Replay, cache, and validate one checked-in real-run artifact."""
    payload: Any = _load_demo_source(source_file)
    return validator(_load_or_create_json(output_file, payload))


def _demo_dataset(output_file: str) -> list[dict[str, str]]:
    """Replay the initial dataset from a real AI run.

    Args:
        output_file (str): Artifact path used for demo cache behavior.

    Returns:
        list[dict[str, str]]: Validated AWS coding task records.
    """
    return _replay_demo_artifact(DEMO_SOURCE_DATASET_FILE, output_file, _validate_dataset)


def _demo_solutions(output_file: str, iteration: int | None = None) -> list[dict[str, str]]:
    """Replay initial or refined solutions from a real AI run.

    Args:
        output_file (str): Artifact path used for demo cache behavior.
        iteration (int | None): Refined iteration number, or ``None`` for the
            initial solutions.

    Returns:
        list[dict[str, str]]: Validated solution records.
    """
    source_file: str = (
        DEMO_SOURCE_SOLUTIONS_FILE
        if iteration is None
        else DEMO_SOURCE_REFINED_SOLUTIONS_PATTERN.format(iteration=iteration)
    )
    return _replay_demo_artifact(source_file, output_file, _validate_solutions)


def _demo_evaluation_results(
    output_file: str,
    iteration: int | None = None,
) -> list[dict[str, Any]]:
    """Replay initial or refined evaluation results from a real AI run.

    Args:
        output_file (str): Artifact path used for demo cache behavior.
        iteration (int | None): Refined iteration number, or ``None`` for the
            initial evaluation.

    Returns:
        list[dict[str, Any]]: Validated evaluation records.
    """
    source_file: str = (
        DEMO_SOURCE_EVALUATION_FILE
        if iteration is None
        else DEMO_SOURCE_REFINED_EVALUATION_PATTERN.format(iteration=iteration)
    )
    return _replay_demo_artifact(source_file, output_file, _validate_evaluation_results)


def _demo_refined_dataset(output_file: str, iteration: int) -> list[dict[str, str]]:
    """Replay one refined dataset from a real AI run.

    Args:
        output_file (str): Artifact path used for demo cache behavior.
        iteration (int): Refined iteration number to replay.

    Returns:
        list[dict[str, str]]: Validated refined task records.
    """
    source_file: str = DEMO_SOURCE_REFINED_DATASET_PATTERN.format(iteration=iteration)
    return _replay_demo_artifact(source_file, output_file, _validate_dataset)


def _stage_success(stage: str, updates: StageSuccessState | None = None) -> AgentState:
    """Represent a successful node execution in graph state.

    Args:
        stage (str): Name of the node or stage that completed successfully.
        updates (StageSuccessState | None): Additional partial state values
            produced by the node.

    Returns:
        AgentState: Partial state update that records the successful stage and
            clears stale error routing fields.
    """
    update: StageSuccessState = {
        "current_stage": stage,
        "failed_stage": None,
        "retry_stage": None,
        "last_error": None,
    }
    if updates is not None:
        update.update(updates)
    return cast(AgentState, update)


def _node_error(stage: str, exc: Exception) -> AgentState:
    """Represent a recoverable node failure in graph state.

    Args:
        stage (str): Name of the node or stage where the exception occurred.
        exc (Exception): Exception raised by the node.

    Returns:
        AgentErrorState: Partial state update that records the error and routes the
            graph toward finalization.
    """
    update: AgentErrorState = {
        "current_stage": stage,
        "failed_stage": stage,
        "retry_stage": None,
        "last_error": f"{type(exc).__name__}: {exc}",
        "stop_reason": "error",
    }
    return cast(AgentState, update)


def _token_multiplier_key_for_stage(stage: str | None) -> str | None:
    """Return the token multiplier state key controlled by a stage.

    Args:
        stage (str | None): Failed graph stage name.

    Returns:
        str | None: Matching token multiplier state key, or ``None`` when the
            stage has no token multiplier to correct.
    """
    stage_to_key: dict[str, str] = {
        "load_or_generate_dataset": "dataset_token_multiplier",
        "load_or_generate_solutions": "solution_token_multiplier",
        "load_or_evaluate_solutions": "evaluation_token_multiplier",
        "refine_dataset": "refinement_token_multiplier",
        "generate_refined_solutions": "solution_token_multiplier",
        "evaluate_refined_solutions": "evaluation_token_multiplier",
    }
    if stage is None:
        return None
    return stage_to_key.get(stage)


def _is_max_tokens_error(error: str | None) -> bool:
    """Return whether an error string describes model output truncation.

    Args:
        error (str | None): Error string stored in graph state.

    Returns:
        bool: ``True`` when the error is a max-token truncation.
    """
    return bool(error and "stop_reason='max_tokens'" in error)


def _can_correct_token_error(state: AgentState) -> bool:
    """Return whether the graph should apply a token-multiplier correction.

    Args:
        state (AgentState): Current graph state containing error details and
            correction counters.

    Returns:
        bool: ``True`` when the latest error is a max-token truncation, the
            failed stage has a known multiplier, and the correction limit has
            not been reached.
    """
    failed_stage: str | None = state.get("failed_stage")
    correction_count: int = int(_state_value(state, "token_correction_count", 0))
    max_corrections: int = int(_state_value(state, "max_token_corrections", MAX_TOKEN_CORRECTIONS))
    return (
        _is_max_tokens_error(state.get("last_error"))
        and _token_multiplier_key_for_stage(failed_stage) is not None
        and correction_count < max_corrections
    )


def load_or_generate_dataset(state: AgentState) -> AgentState:
    """Load or generate the initial evaluation dataset.

    Args:
        state (AgentState): Current graph state, optionally containing
            ``dataset_file`` and ``dataset_token_multiplier``.

    Returns:
        AgentState: Partial state with validated ``dataset`` on success, or
            ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "load_or_generate_dataset"
    try:
        dataset_file: str = _state_value(state, "dataset_file", EvaluationDatasetGenerator.DATASET_FILE)
        if bool(_state_value(state, "demo_mode", False)):
            return _stage_success(stage, {"dataset": _demo_dataset(dataset_file)})

        token_multiplier: float = float(_state_value(state, "dataset_token_multiplier", DATASET_TOKEN_MULTIPLIER))
        dataset: list[dict[str, Any]] = EvaluationDatasetGenerator(
            ClaudeClient(model=DATASET_MODEL),
            output_file=dataset_file,
        ).create_dataset(token_multiplier=token_multiplier)
        return _stage_success(stage, {"dataset": _validate_dataset(dataset)})
    except Exception as exc:
        return _node_error(stage, exc)


def _generate_solutions_node(state: AgentState, stage: str, output_file: str) -> AgentState:
    """Load or generate solutions for a dataset-backed graph node.

    Args:
        state (AgentState): Current graph state containing ``dataset`` and
            optionally ``solution_token_multiplier``.
        stage (str): Public graph node name to record in state.
        output_file (str): Artifact path used by ``SolutionGenerator``.

    Returns:
        AgentState: Partial state with validated ``solutions`` on success, or
            error details on failure.
    """
    try:
        dataset: list[dict[str, str]] = _validate_dataset(state.get("dataset"))
        if bool(_state_value(state, "demo_mode", False)):
            iteration: int | None = (
                int(_state_value(state, "iteration", 0)) + 1
                if stage == "generate_refined_solutions"
                else None
            )
            return _stage_success(stage, {"solutions": _demo_solutions(output_file, iteration)})

        token_multiplier: float = float(_state_value(state, "solution_token_multiplier", SOLUTION_TOKEN_MULTIPLIER))
        solutions: list[dict[str, str]] = SolutionGenerator(
            ClaudeClient(),
            output_file=output_file,
        ).generate_solutions(dataset, token_multiplier=token_multiplier)
        return _stage_success(stage, {"solutions": _validate_solutions(solutions)})
    except Exception as exc:
        return _node_error(stage, exc)


def load_or_generate_solutions(state: AgentState) -> AgentState:
    """Load or generate solutions for the current dataset.

    Args:
        state (AgentState): Current graph state containing ``dataset`` and
            optionally ``solutions_file`` and ``solution_token_multiplier``.

    Returns:
        AgentState: Partial state with validated ``solutions`` on success, or
            ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "load_or_generate_solutions"
    solutions_file: str = _state_value(state, "solutions_file", SolutionGenerator.DEFAULT_SOLUTIONS_FILE)
    return _generate_solutions_node(state, stage, solutions_file)


def _evaluate_solutions_node(state: AgentState, stage: str, output_file: str) -> AgentState:
    """Load or evaluate solutions for a dataset-backed graph node.

    Args:
        state (AgentState): Current graph state containing ``dataset``,
            ``solutions``, and optionally ``evaluation_token_multiplier``.
        stage (str): Public graph node name to record in state.
        output_file (str): Artifact path used by ``PromptEvaluator``.

    Returns:
        AgentState: Partial state with validated ``evaluation_results`` on
            success, or error details on failure.
    """
    try:
        dataset: list[dict[str, str]] = _validate_dataset(state.get("dataset"))
        solutions: list[dict[str, str]] = _validate_solutions(state.get("solutions"))
        if bool(_state_value(state, "demo_mode", False)):
            iteration: int | None = (
                int(_state_value(state, "iteration", 0)) + 1
                if stage == "evaluate_refined_solutions"
                else None
            )
            return _stage_success(stage, {"evaluation_results": _demo_evaluation_results(output_file, iteration)})

        token_multiplier: float = float(_state_value(state, "evaluation_token_multiplier", EVALUATION_TOKEN_MULTIPLIER))
        results: list[dict[str, Any]] = PromptEvaluator(
            ClaudeClient(),
            output_file=output_file,
        ).evaluate_prompts(dataset, solutions, token_multiplier=token_multiplier)
        return _stage_success(stage, {"evaluation_results": _validate_evaluation_results(results)})
    except Exception as exc:
        return _node_error(stage, exc)


def load_or_evaluate_solutions(state: AgentState) -> AgentState:
    """Load or evaluate solutions for the current dataset.

    Args:
        state (AgentState): Current graph state containing ``dataset``,
            ``solutions``, and optionally ``evaluation_file`` and
            ``evaluation_token_multiplier``.

    Returns:
        AgentState: Partial state with validated ``evaluation_results`` on
            success, or ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "load_or_evaluate_solutions"
    evaluation_file: str = _state_value(state, "evaluation_file", PromptEvaluator.DEFAULT_EVALUATION_FILE)
    return _evaluate_solutions_node(state, stage, evaluation_file)


def initialize_best(state: AgentState) -> AgentState:
    """Initialise best-scoring state from the first evaluation pass.

    Args:
        state (AgentState): Current graph state containing the initial
            ``dataset`` and ``evaluation_results``.

    Returns:
        AgentState: Partial state containing ``best_dataset``, ``best_results``,
            ``best_score``, iteration counters, and the current stage.
    """
    stage = "initialize_best"
    try:
        dataset: list[dict[str, str]] = _validate_dataset(state.get("dataset"))
        results: list[dict[str, Any]] = _validate_evaluation_results(state.get("evaluation_results"))
        best_dataset_file: str = _state_value(state, "best_dataset_file", BEST_DATASET_FILE)
        best_score: float = _mean_score(results)
        _save_best_dataset(dataset, best_dataset_file)
        return _stage_success(
            stage,
            {
                "best_dataset": dataset,
                "best_results": results,
                "best_score": best_score,
                "iteration": int(_state_value(state, "iteration", 0)),
                "stagnation_count": int(_state_value(state, "stagnation_count", 0)),
            },
        )
    except Exception as exc:
        return _node_error(stage, exc)


def decide_next_step(state: AgentState) -> AgentState:
    """Decide whether the graph should stop or run another refinement pass.

    Args:
        state (AgentState): Current graph state containing score, iteration, and
            stagnation counters plus their configured limits.

    Returns:
        AgentState: Partial state with ``stop_reason`` set when execution should
            end, otherwise ``stop_reason`` set to ``None``.
    """
    stage = "decide_next_step"
    try:
        best_score = float(_state_value(state, "best_score", 0.0))
        score_threshold = float(_state_value(state, "score_threshold", SCORE_THRESHOLD))
        iteration = int(_state_value(state, "iteration", 0))
        max_iterations = int(_state_value(state, "max_iterations", MAX_REFINEMENT_ITERATIONS))
        stagnation_count = int(_state_value(state, "stagnation_count", 0))
        max_stagnation = int(_state_value(state, "max_stagnation", MAX_STAGNATION_ITERATIONS))

        stop_reason: str | None = None
        if best_score >= score_threshold:
            stop_reason = "score_threshold"
        elif iteration >= max_iterations:
            stop_reason = "max_iterations"
        elif stagnation_count >= max_stagnation:
            stop_reason = "stagnation"

        return _stage_success(stage, {"stop_reason": stop_reason})
    except Exception as exc:
        return _node_error(stage, exc)


def refine_dataset(state: AgentState) -> AgentState:
    """Refine solution criteria from the best-scoring dataset so far.

    Args:
        state (AgentState): Current graph state containing ``best_dataset``,
            ``best_results``, iteration count, refined artifact path pattern, and
            ``refinement_token_multiplier``.

    Returns:
        AgentState: Partial state with the newly refined ``dataset`` on success,
            or ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "refine_dataset"
    try:
        best_dataset: list[dict[str, str]] = _validate_dataset(state.get("best_dataset"))
        best_results: list[dict[str, Any]] = _validate_evaluation_results(state.get("best_results"))
        output_file: str = _refined_artifact_path(state, "refined_dataset_pattern", "refined_dataset_{iteration}.json")
        if bool(_state_value(state, "demo_mode", False)):
            iteration: int = int(_state_value(state, "iteration", 0)) + 1
            return _stage_success(stage, {"dataset": _demo_refined_dataset(output_file, iteration)})

        token_multiplier: float = float(_state_value(state, "refinement_token_multiplier", REFINEMENT_TOKEN_MULTIPLIER))
        dataset: list[dict[str, str]] = TaskRefiner(
            ClaudeClient(),
            output_file=output_file,
        ).refine_dataset(best_dataset, best_results, token_multiplier=token_multiplier)
        return _stage_success(stage, {"dataset": _validate_dataset(dataset)})
    except Exception as exc:
        return _node_error(stage, exc)


def generate_refined_solutions(state: AgentState) -> AgentState:
    """Load or generate solutions for the refined dataset.

    Args:
        state (AgentState): Current graph state containing the refined
            ``dataset``, iteration count, refined solution artifact pattern, and
            ``solution_token_multiplier``.

    Returns:
        AgentState: Partial state with validated refined ``solutions`` on
            success, or ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "generate_refined_solutions"
    output_file: str = _refined_artifact_path(state, "refined_solutions_pattern", "refined_solutions_{iteration}.json")
    return _generate_solutions_node(state, stage, output_file)


def evaluate_refined_solutions(state: AgentState) -> AgentState:
    """Load or evaluate solutions for the refined dataset.

    Args:
        state (AgentState): Current graph state containing refined ``dataset``,
            ``solutions``, iteration count, refined evaluation artifact pattern,
            and ``evaluation_token_multiplier``.

    Returns:
        AgentState: Partial state with validated refined ``evaluation_results``
            on success, or ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "evaluate_refined_solutions"
    output_file: str = _refined_artifact_path(state, "refined_evaluation_pattern", "refined_evaluation_results_{iteration}.json")
    return _evaluate_solutions_node(state, stage, output_file)


def update_best_or_stagnation(state: AgentState) -> AgentState:
    """Update best dataset tracking after a refinement evaluation.

    Args:
        state (AgentState): Current graph state containing the latest refined
            ``dataset``, ``evaluation_results``, best score, and counters.

    Returns:
        AgentState: Partial state with incremented ``iteration`` and either
            updated best fields or incremented ``stagnation_count``.
    """
    stage = "update_best_or_stagnation"
    try:
        dataset: list[dict[str, str]] = _validate_dataset(state.get("dataset"))
        results: list[dict[str, Any]] = _validate_evaluation_results(state.get("evaluation_results"))
        refined_score: float = _mean_score(results)
        best_score = float(_state_value(state, "best_score", 0.0))
        iteration = int(_state_value(state, "iteration", 0)) + 1
        stagnation_count = int(_state_value(state, "stagnation_count", 0))

        updates: StageSuccessState = {
            "iteration": iteration,
        }

        if refined_score > best_score:
            best_dataset_file: str = _state_value(state, "best_dataset_file", BEST_DATASET_FILE)
            _save_best_dataset(dataset, best_dataset_file)
            updates.update(
                {
                    "best_dataset": dataset,
                    "best_results": results,
                    "best_score": refined_score,
                    "stagnation_count": 0,
                }
            )
        else:
            updates["stagnation_count"] = stagnation_count + 1

        return _stage_success(stage, updates)
    except Exception as exc:
        return _node_error(stage, exc)


def apply_token_multiplier_correction(state: AgentState) -> AgentState:
    """Increase the failed stage's token multiplier by exactly one and retry.

    Args:
        state (AgentState): Current graph state containing ``failed_stage``,
            ``last_error``, and token multiplier values.

    Returns:
        AgentState: Partial state with one token multiplier increased by
            exactly ``1.0``, error fields cleared, and ``retry_stage`` set to
            the failed stage, or an error state if the failed stage has no
            configurable multiplier.
    """
    stage: str | None = state.get("failed_stage")
    multiplier_key: str | None = _token_multiplier_key_for_stage(stage)
    if stage is None or multiplier_key is None:
        return _node_error("apply_token_multiplier_correction", ValueError("No token multiplier is mapped for the failed stage."))

    current_multiplier: float = float(_state_value(state, multiplier_key, 1.0))
    correction_count: int = int(_state_value(state, "token_correction_count", 0))
    update: TokenCorrectionState = {
        "current_stage": "apply_token_multiplier_correction",
        "failed_stage": None,
        "retry_stage": stage,
        "last_error": None,
        "stop_reason": None,
        "token_correction_count": correction_count + 1,
    }
    corrected_multiplier: float = current_multiplier + 1.0
    if multiplier_key == "dataset_token_multiplier":
        update["dataset_token_multiplier"] = corrected_multiplier
    elif multiplier_key == "solution_token_multiplier":
        update["solution_token_multiplier"] = corrected_multiplier
    elif multiplier_key == "evaluation_token_multiplier":
        update["evaluation_token_multiplier"] = corrected_multiplier
    else:
        update["refinement_token_multiplier"] = corrected_multiplier
    return cast(AgentState, update)


def finalize(state: AgentState) -> AgentState:
    """Mark graph execution complete and preserve the final state summary.

    Args:
        state (AgentState): Current graph state, optionally containing an
            existing ``stop_reason``.

    Returns:
        AgentState: Partial state with ``stop_reason`` populated. Successful
            runs set ``current_stage`` to ``"finalize"``; error runs preserve
            the stage where the error occurred in ``failed_stage``.
    """
    stop_reason = state.get("stop_reason")
    if not stop_reason:
        stop_reason = "complete"
    current_stage: str = _state_value(state, "current_stage", "finalize") if state.get("last_error") else "finalize"
    update: FinalState = {
        "current_stage": current_stage,
        "failed_stage": state.get("failed_stage"),
        "last_error": state.get("last_error"),
        "stop_reason": stop_reason,
    }
    return cast(AgentState, update)


def _route_after_stage(state: AgentState, next_node: str) -> str:
    """Route after a normal graph stage.

    Args:
        state (AgentState): Current graph state.
        next_node (str): Node name to route to when there is no error.

    Returns:
        str: Correction node for recoverable max-token errors, ``"finalize"``
            for unrecoverable errors, otherwise ``next_node``.
    """
    if state.get("last_error") or state.get("stop_reason") == "error":
        if _can_correct_token_error(state):
            return "apply_token_multiplier_correction"
        return "finalize"
    return next_node


def _route_after_decision(state: AgentState) -> str:
    """Route after the policy decision node.

    Args:
        state (AgentState): Current graph state after ``decide_next_step``.

    Returns:
        str: ``"finalize"`` when execution should stop, otherwise
            ``"refine_dataset"``.
    """
    if state.get("last_error") or state.get("stop_reason"):
        return "finalize"
    return "refine_dataset"


def _route_after_token_correction(state: AgentState) -> str:
    """Route from token correction back to the failed stage.

    Args:
        state (AgentState): Current graph state after
            ``apply_token_multiplier_correction``.

    Returns:
        str: Stage name to retry, or ``"finalize"`` if no retry stage exists.
    """
    retry_stage: str | None = state.get("retry_stage")
    if state.get("last_error") or retry_stage is None:
        return "finalize"
    return retry_stage


# Checkpoint history helpers.


def _ensure_sqlite_metadata_serializer(checkpointer: SqliteSaver) -> None:
    """Patch older sqlite checkpointers to work with newer LangGraph serializers.

    Args:
        checkpointer (SqliteSaver): SQLite checkpointer used by the compiled graph.

    Returns:
        None.
    """
    serializer: Any = checkpointer.jsonplus_serde
    if hasattr(serializer, "dumps") and hasattr(serializer, "loads"):
        return

    def dump_metadata(value: Any) -> bytes:
        """Serialize checkpoint metadata to bytes.

        Args:
            value (Any): JSON-serializable metadata value.

        Returns:
            bytes: UTF-8 encoded JSON metadata.
        """
        return dumps(value).encode("utf-8")

    def load_metadata(value: bytes | str) -> Any:
        """Deserialize checkpoint metadata.

        Args:
            value (bytes | str): Metadata previously stored by ``dump_metadata``.

        Returns:
            Any: Decoded JSON metadata value.
        """
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return loads(value)

    serializer.dumps = dump_metadata  # type: ignore[attr-defined]
    serializer.loads = load_metadata  # type: ignore[attr-defined]


def _decode_checkpoint_write(value_type: str, value: bytes) -> Any:
    """Decode a LangGraph checkpoint write value.

    Args:
        value_type (str): Serializer type stored in the SQLite ``writes`` table.
        value (bytes): Serialized channel value from the SQLite ``writes`` table.

    Returns:
        Any: Decoded checkpoint channel value.
    """
    checkpointer: SqliteSaver = SqliteSaver(sqlite3.connect(":memory:"))
    _ensure_sqlite_metadata_serializer(checkpointer)
    try:
        return checkpointer.serde.loads_typed((value_type, value))
    finally:
        checkpointer.conn.close()


def _latest_checkpoint_error(checkpoint_db: str, thread_id: str) -> str | None:
    """Return the latest non-empty error detail from checkpoint write history.

    Args:
        checkpoint_db (str): SQLite checkpoint database path.
        thread_id (str): LangGraph checkpoint thread ID.

    Returns:
        str | None: Latest stored ``last_error`` value, or ``None`` when no
            non-empty error detail is present.
    """
    if not os.path.exists(checkpoint_db):
        return None

    conn: sqlite3.Connection = sqlite3.connect(checkpoint_db)
    try:
        rows: list[tuple[str, bytes]] = conn.execute(
            """
            SELECT type, value
            FROM writes
            WHERE thread_id = ?
              AND channel = 'last_error'
              AND length(value) > 0
            ORDER BY rowid DESC
            """,
            (thread_id,),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    for value_type, value in rows:
        decoded: Any = _decode_checkpoint_write(value_type, value)
        if isinstance(decoded, str) and decoded:
            return decoded
    return None


GRAPH_NODES: tuple[tuple[str, Any], ...] = (
    ("load_or_generate_dataset", load_or_generate_dataset),
    ("load_or_generate_solutions", load_or_generate_solutions),
    ("load_or_evaluate_solutions", load_or_evaluate_solutions),
    ("initialize_best", initialize_best),
    ("decide_next_step", decide_next_step),
    ("refine_dataset", refine_dataset),
    ("generate_refined_solutions", generate_refined_solutions),
    ("evaluate_refined_solutions", evaluate_refined_solutions),
    ("update_best_or_stagnation", update_best_or_stagnation),
    ("apply_token_multiplier_correction", apply_token_multiplier_correction),
    ("finalize", finalize),
)
STAGE_EDGE_SPECS: tuple[tuple[str, str], ...] = (
    ("load_or_generate_dataset", "load_or_generate_solutions"),
    ("load_or_generate_solutions", "load_or_evaluate_solutions"),
    ("load_or_evaluate_solutions", "initialize_best"),
    ("initialize_best", "decide_next_step"),
    ("refine_dataset", "generate_refined_solutions"),
    ("generate_refined_solutions", "evaluate_refined_solutions"),
    ("evaluate_refined_solutions", "update_best_or_stagnation"),
    ("update_best_or_stagnation", "decide_next_step"),
)
STAGE_ROUTE_EXITS: dict[Hashable, str] = {
    "apply_token_multiplier_correction": "apply_token_multiplier_correction",
    "finalize": "finalize",
}
TOKEN_CORRECTION_ROUTES: dict[Hashable, str] = {
    "load_or_generate_dataset": "load_or_generate_dataset",
    "load_or_generate_solutions": "load_or_generate_solutions",
    "load_or_evaluate_solutions": "load_or_evaluate_solutions",
    "refine_dataset": "refine_dataset",
    "generate_refined_solutions": "generate_refined_solutions",
    "evaluate_refined_solutions": "evaluate_refined_solutions",
    "finalize": "finalize",
}


# Graph construction helpers.


def _add_graph_nodes(workflow: StateGraph) -> None:
    """Register all LangGraph nodes used by the prompt-evaluation agent.

    Args:
        workflow (StateGraph): Mutable graph being assembled.

    Returns:
        None.
    """
    for node_name, node in GRAPH_NODES:
        workflow.add_node(node_name, node)


def _stage_route_map(next_node: str) -> dict[Hashable, str]:
    """Return the route map for a normal stage transition.

    Args:
        next_node (str): Node reached when the current stage succeeds.

    Returns:
        dict[Hashable, str]: LangGraph route map including success, correction,
            and finalization routes.
    """
    return {next_node: next_node, **STAGE_ROUTE_EXITS}


def _add_stage_edges(workflow: StateGraph) -> None:
    """Register deterministic and conditional routes for the graph.

    Args:
        workflow (StateGraph): Mutable graph being assembled.

    Returns:
        None.
    """
    workflow.add_edge(START, "load_or_generate_dataset")
    for source_node, next_node in STAGE_EDGE_SPECS:
        workflow.add_conditional_edges(
            source_node,
            lambda state, target=next_node: _route_after_stage(state, target),
            _stage_route_map(next_node),
        )
    workflow.add_conditional_edges(
        "decide_next_step",
        _route_after_decision,
        {"refine_dataset": "refine_dataset", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "apply_token_multiplier_correction",
        _route_after_token_correction,
        TOKEN_CORRECTION_ROUTES,
    )
    workflow.add_edge("finalize", END)


def _compile_graph(workflow: StateGraph, checkpoint_db: str) -> Any:
    """Compile a graph with a SQLite checkpointer.

    Args:
        workflow (StateGraph): Graph to compile.
        checkpoint_db (str): SQLite database path used by the LangGraph
            checkpointer.

    Returns:
        Any: Compiled LangGraph application.
    """
    conn: sqlite3.Connection = sqlite3.connect(checkpoint_db, check_same_thread=False)
    checkpointer: SqliteSaver = SqliteSaver(conn)
    _ensure_sqlite_metadata_serializer(checkpointer)
    checkpointer.setup()
    return workflow.compile(checkpointer=checkpointer)


def build_prompt_evaluation_agent(checkpoint_db: str = DEFAULT_CHECKPOINT_DB) -> Any:
    """Build and compile the LangGraph prompt-evaluation agent.

    Args:
        checkpoint_db (str): SQLite database path used by the LangGraph
            checkpointer.

    Returns:
        Any: Compiled LangGraph application ready to invoke with ``AgentState``
            input and a ``thread_id`` config.
    """
    workflow: StateGraph = StateGraph(AgentState)
    _add_graph_nodes(workflow)
    _add_stage_edges(workflow)
    return _compile_graph(workflow, checkpoint_db)


def _generate_run_id() -> str:
    """Return a filesystem-safe unique live-run identifier."""
    timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _validate_run_id(run_id: str) -> str:
    """Validate and return a run identifier.

    Args:
        run_id (str): Candidate run identifier.

    Returns:
        str: Validated run identifier.

    Raises:
        ValueError: If the identifier could escape the runs root.
    """
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
        raise ValueError(f"Invalid run ID: {run_id!r}")
    return run_id


def _cli_download_link_provider(base_url: str | None) -> DownloadLinkProvider | None:
    """Build the CLI-configured download-link provider.

    Args:
        base_url (str | None): Explicit CLI base URL.

    Returns:
        DownloadLinkProvider | None: Base-URL provider when configured via the
        CLI or ``DOWNLOAD_BASE_URL`` environment variable.
    """
    load_dotenv()
    resolved_base_url: str | None = base_url or os.getenv("DOWNLOAD_BASE_URL")
    if not resolved_base_url:
        return None
    return BaseUrlDownloadLinkProvider(resolved_base_url)


def _prepare_live_run(
    runs_root: str,
    resume_run_id: str | None,
    thread_id: str | None,
) -> tuple[str, Path, str, dict[str, Any], RunArtifactPaths]:
    """Create or reopen an isolated live-run directory.

    Args:
        runs_root (str): Parent directory for live runs.
        resume_run_id (str | None): Existing run ID to reopen.
        thread_id (str | None): Explicit checkpoint thread ID.

    Returns:
        tuple[str, Path, str, dict[str, Any], RunArtifactPaths]: Run ID,
        directory, resolved thread ID, manifest, and artifact paths.
    """
    root: Path = Path(runs_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if resume_run_id is not None:
        run_id: str = _validate_run_id(resume_run_id)
        run_directory: Path = (root / run_id).resolve()
        if run_directory.parent != root or not run_directory.is_dir():
            raise FileNotFoundError(f"Run not found for resume: {run_id}")
        artifact_paths: RunArtifactPaths = RunArtifactPaths.in_directory(run_directory)
        manifest: dict[str, Any] = load_manifest(artifact_paths.manifest)
        resolved_thread_id: str = thread_id or str(manifest.get("thread_id") or run_id)
        return run_id, run_directory, resolved_thread_id, manifest, artifact_paths

    while True:
        run_id = _generate_run_id()
        run_directory = root / run_id
        try:
            run_directory.mkdir()
            break
        except FileExistsError:
            continue

    resolved_thread_id = thread_id or run_id
    manifest = {
        "run_id": run_id,
        "created_at": utc_timestamp(),
        "updated_at": utc_timestamp(),
        "status": "running",
        "thread_id": resolved_thread_id,
    }
    resolved_directory: Path = run_directory.resolve()
    return (
        run_id,
        resolved_directory,
        resolved_thread_id,
        manifest,
        RunArtifactPaths.in_directory(resolved_directory),
    )


def _finalize_live_run(
    manifest: dict[str, Any],
    artifact_paths: RunArtifactPaths,
    *,
    status: str,
    stop_reason: str,
    error: str | None,
    best_score: float | None = None,
    completed_iterations: int | None = None,
) -> Path:
    """Persist final live-run metadata and rebuild its downloadable archive."""
    manifest.update(
        {
            "status": status,
            "stop_reason": stop_reason,
            "error": error,
        }
    )
    if best_score is not None:
        manifest["best_score"] = best_score
    if completed_iterations is not None:
        manifest["completed_iterations"] = completed_iterations
    write_manifest(artifact_paths.manifest, manifest)
    return create_run_archive(artifact_paths)


def main(
    score_threshold: float = SCORE_THRESHOLD,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    max_stagnation: int = MAX_STAGNATION_ITERATIONS,
    thread_id: str | None = None,
    checkpoint_db: str | None = None,
    dataset_token_multiplier: float = DATASET_TOKEN_MULTIPLIER,
    solution_token_multiplier: float = SOLUTION_TOKEN_MULTIPLIER,
    evaluation_token_multiplier: float = EVALUATION_TOKEN_MULTIPLIER,
    refinement_token_multiplier: float = REFINEMENT_TOKEN_MULTIPLIER,
    max_token_corrections: int = MAX_TOKEN_CORRECTIONS,
    demo_mode: bool = False,
    runs_root: str = DEFAULT_RUNS_ROOT,
    resume_run_id: str | None = None,
    download_link_provider: DownloadLinkProvider | None = None,
) -> RunResult:
    """Run the LangGraph prompt-evaluation agent with iterative refinement.

    The graph performs the same artifact-producing work as the previous
    pipeline while LangGraph owns orchestration state, deterministic routing,
    and SQLite-backed checkpointing.

    Args:
        score_threshold (float): Target mean score (1–10). Refinement stops once
            the best mean score meets or exceeds this value. Defaults to
            the module-level ``SCORE_THRESHOLD``.
        max_iterations (int): Maximum number of refinement passes before stopping.
            Defaults to the module-level ``MAX_REFINEMENT_ITERATIONS``.
        max_stagnation (int): Stop early after this many consecutive non-improving
            iterations. Defaults to the module-level ``MAX_STAGNATION_ITERATIONS``.
        thread_id (str | None): LangGraph checkpoint thread ID used for
            resumable runs. Defaults to the normal or demo thread based on
            ``demo_mode``.
        checkpoint_db (str | None): SQLite database path used for LangGraph
            checkpoints. Defaults to the normal or demo database based on
            ``demo_mode``.
        dataset_token_multiplier (float): Multiplier applied to ``MAX_TOKENS``
            for dataset-generation model calls.
        solution_token_multiplier (float): Multiplier applied to ``MAX_TOKENS``
            for solution-generation model calls.
        evaluation_token_multiplier (float): Multiplier applied to ``MAX_TOKENS``
            for evaluation model calls.
        refinement_token_multiplier (float): Multiplier applied to ``MAX_TOKENS``
            for task-refinement model calls.
        max_token_corrections (int): Maximum number of automatic corrections
            for max-token truncation errors. Each correction increases only the
            failed stage's multiplier by exactly ``1.0``.
        demo_mode (bool): Run with deterministic local artifacts and no AI API
            calls. Defaults to ``False``.
        runs_root (str): Parent directory for isolated live-run folders.
        resume_run_id (str | None): Existing live-run ID to resume.
        download_link_provider (DownloadLinkProvider | None): Strategy used to
            create a downloadable link after the run archive is finalized.

    Returns:
        RunResult: Run identity, archive location, download URL, and status.
    """
    if demo_mode and resume_run_id is not None:
        raise ValueError("--resume-run-id cannot be used with demo mode.")

    run_id: str | None = None
    run_directory: Path | None = None
    manifest: dict[str, Any] | None = None
    archive_path: Path | None = None

    if demo_mode:
        resolved_thread_id: str = thread_id or DEMO_THREAD_ID
        resolved_checkpoint_db: str = checkpoint_db or DEMO_CHECKPOINT_DB
        artifact_paths: RunArtifactPaths = DEMO_ARTIFACTS
    else:
        run_id, run_directory, resolved_thread_id, manifest, artifact_paths = _prepare_live_run(
            runs_root,
            resume_run_id,
            thread_id,
        )
        resolved_checkpoint_db = checkpoint_db or str(artifact_paths.checkpoint)
        manifest.update(
            {
                "status": "running",
                "thread_id": resolved_thread_id,
                "requested_limits": {
                    "score_threshold": score_threshold,
                    "max_iterations": max_iterations,
                    "max_stagnation": max_stagnation,
                    "max_token_corrections": max_token_corrections,
                },
                "artifacts": artifact_paths.manifest_artifacts(),
            }
        )
        write_manifest(artifact_paths.manifest, manifest)

    print("=== LangGraph prompt-evaluation agent ===")
    if demo_mode:
        print("Mode: demo (no AI API calls)")
    else:
        print("Mode: live AI API")
        print(f"Run ID: {run_id}")
        print(f"Run directory: {run_directory}")
    print(f"Thread: {resolved_thread_id}")
    print(f"Checkpoint DB: {resolved_checkpoint_db}")
    print(
        "Token multipliers: "
        f"dataset={dataset_token_multiplier}, "
        f"solution={solution_token_multiplier}, "
        f"evaluation={evaluation_token_multiplier}, "
        f"refinement={refinement_token_multiplier}"
    )

    graph: Any = build_prompt_evaluation_agent(checkpoint_db=resolved_checkpoint_db)
    state_artifact_paths: dict[str, str] = artifact_paths.state_paths()
    initial_state: AgentState = {
        "demo_mode": demo_mode,
        "score_threshold": score_threshold,
        "max_iterations": max_iterations,
        "max_stagnation": max_stagnation,
        "token_correction_count": 0,
        "max_token_corrections": max_token_corrections,
        "current_stage": "start",
        "failed_stage": None,
        "retry_stage": None,
        "last_error": None,
        "stop_reason": None,
        "iteration": 0,
        "stagnation_count": 0,
        "dataset_file": state_artifact_paths["dataset_file"],
        "solutions_file": state_artifact_paths["solutions_file"],
        "evaluation_file": state_artifact_paths["evaluation_file"],
        "best_dataset_file": state_artifact_paths["best_dataset_file"],
        "refined_dataset_pattern": state_artifact_paths["refined_dataset_pattern"],
        "refined_solutions_pattern": state_artifact_paths["refined_solutions_pattern"],
        "refined_evaluation_pattern": state_artifact_paths["refined_evaluation_pattern"],
        "dataset_token_multiplier": dataset_token_multiplier,
        "solution_token_multiplier": solution_token_multiplier,
        "evaluation_token_multiplier": evaluation_token_multiplier,
        "refinement_token_multiplier": refinement_token_multiplier,
    }
    config: dict[str, dict[str, str]] = {"configurable": {"thread_id": resolved_thread_id}}
    try:
        final_state: AgentState = graph.invoke(initial_state, config)
    except Exception as exc:
        if manifest is not None:
            archive_path = _finalize_live_run(
                manifest,
                artifact_paths,
                status="failed",
                stop_reason="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        raise
    finally:
        checkpointer: Any = getattr(graph, "checkpointer", None)
        connection: Any = getattr(checkpointer, "conn", None)
        if connection is not None:
            connection.close()

    last_error: str | None = final_state.get("last_error")
    stop_reason: str = _state_value(final_state, "stop_reason", "complete")
    best_score: float = float(_state_value(final_state, "best_score", 0.0))
    iteration: int = int(_state_value(final_state, "iteration", 0))
    status: str = "failed" if last_error or stop_reason == "error" else "completed"
    checkpoint_error: str | None = None
    if stop_reason == "error" and not last_error:
        checkpoint_error = _latest_checkpoint_error(resolved_checkpoint_db, resolved_thread_id)
    error_detail: str | None = last_error or checkpoint_error

    if manifest is not None:
        archive_path = _finalize_live_run(
            manifest,
            artifact_paths,
            status=status,
            stop_reason=stop_reason,
            error=error_detail,
            best_score=best_score,
            completed_iterations=iteration,
        )

    if last_error:
        failed_stage: str = _state_value(final_state, "failed_stage", _state_value(final_state, "current_stage", "unknown"))
        current_stage: str = _state_value(final_state, "current_stage", "unknown")
        print(f"Stopped with error in stage {failed_stage} (current stage: {current_stage}): {last_error}")
    elif stop_reason == "error":
        failed_stage: str = _state_value(final_state, "failed_stage", _state_value(final_state, "current_stage", "unknown"))
        current_stage: str = _state_value(final_state, "current_stage", "unknown")
        if checkpoint_error:
            print(f"Stopped with error in stage {failed_stage} (current stage: {current_stage}): {checkpoint_error}")
        else:
            print(
                f"Stopped with error in stage {failed_stage} (current stage: {current_stage}), "
                "but no error details were present in the final graph state."
            )
            print("No non-empty last_error write was found in the checkpoint history.")
    else:
        print(f"Stopped: {stop_reason}")
        print(f"Best mean score: {best_score:.2f}")
        print(f"Refinement passes completed: {iteration}")
        print(f"Best dataset: {_state_value(final_state, 'best_dataset_file', BEST_DATASET_FILE)}")

    formed_download_link: str | None = None
    if download_link_provider is not None and run_id is not None and archive_path is not None:
        formed_download_link = download_link_provider.create_download_link(run_id, archive_path)
    if run_directory is not None:
        print(f"Run directory: {run_directory}")
    if formed_download_link is not None:
        print(f"Download link: {formed_download_link}")
    elif archive_path is not None:
        print(f"Download archive: {archive_path}")

    return {
        "run_id": run_id,
        "run_directory": str(run_directory) if run_directory is not None else None,
        "archive_path": str(archive_path) if archive_path is not None else None,
        "download_link": formed_download_link,
        "status": status,
        "stop_reason": stop_reason,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the prompt-evaluation agent.

    Returns:
        argparse.ArgumentParser: Parser configured with all supported CLI flags.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Prompt evaluation pipeline with iterative refinement."
    )
    parser.add_argument(
        "--score",
        dest="score_threshold",
        type=float,
        default=SCORE_THRESHOLD,
        metavar="N",
        help=f"target mean score threshold, 1–10 (default: {SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--iterations",
        dest="max_iterations",
        type=int,
        default=MAX_REFINEMENT_ITERATIONS,
        metavar="N",
        help=f"maximum number of refinement passes (default: {MAX_REFINEMENT_ITERATIONS})",
    )
    parser.add_argument(
        "--stagnation",
        dest="max_stagnation",
        type=int,
        default=MAX_STAGNATION_ITERATIONS,
        metavar="N",
        help=f"stop after N consecutive non-improving iterations (default: {MAX_STAGNATION_ITERATIONS})",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help=(
            "LangGraph checkpoint thread ID used for resumable runs "
            f"(default: generated live run ID; demo: {DEMO_THREAD_ID})"
        ),
    )
    parser.add_argument(
        "--checkpoint-db",
        type=str,
        default=None,
        help=(
            "SQLite database path used for LangGraph checkpoints "
            f"(default: run-local {DEFAULT_CHECKPOINT_DB}; demo: {DEMO_CHECKPOINT_DB})"
        ),
    )
    parser.add_argument(
        "--demo",
        dest="demo_mode",
        action="store_true",
        help="run a deterministic no-AI demo using demo-prefixed artifacts",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=DEFAULT_RUNS_ROOT,
        help=f"parent directory for isolated live runs (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--resume-run-id",
        type=str,
        default=None,
        help="resume an existing live run by its run ID",
    )
    parser.add_argument(
        "--download-base-url",
        type=str,
        default=None,
        help="public base URL for run ZIP downloads (fallback: DOWNLOAD_BASE_URL)",
    )
    parser.add_argument(
        "--dataset-token-multiplier",
        type=float,
        default=DATASET_TOKEN_MULTIPLIER,
        metavar="N",
        help=f"token multiplier for dataset generation (default: {DATASET_TOKEN_MULTIPLIER})",
    )
    parser.add_argument(
        "--solution-token-multiplier",
        type=float,
        default=SOLUTION_TOKEN_MULTIPLIER,
        metavar="N",
        help=f"token multiplier for solution generation (default: {SOLUTION_TOKEN_MULTIPLIER})",
    )
    parser.add_argument(
        "--evaluation-token-multiplier",
        type=float,
        default=EVALUATION_TOKEN_MULTIPLIER,
        metavar="N",
        help=f"token multiplier for solution evaluation (default: {EVALUATION_TOKEN_MULTIPLIER})",
    )
    parser.add_argument(
        "--refinement-token-multiplier",
        type=float,
        default=REFINEMENT_TOKEN_MULTIPLIER,
        metavar="N",
        help=f"token multiplier for criteria refinement (default: {REFINEMENT_TOKEN_MULTIPLIER})",
    )
    parser.add_argument(
        "--max-token-corrections",
        type=int,
        default=MAX_TOKEN_CORRECTIONS,
        metavar="N",
        help=(
            "maximum automatic max-token corrections; each correction "
            f"increases the failed stage multiplier by exactly 1.0 (default: {MAX_TOKEN_CORRECTIONS})"
        ),
    )
    return parser


@dataclass(frozen=True)
class PipelineConfig:
    """Typed command-line configuration for one pipeline invocation."""

    score_threshold: float
    max_iterations: int
    max_stagnation: int
    thread_id: str | None
    checkpoint_db: str | None
    demo_mode: bool
    runs_root: str
    resume_run_id: str | None
    download_base_url: str | None
    dataset_token_multiplier: float
    solution_token_multiplier: float
    evaluation_token_multiplier: float
    refinement_token_multiplier: float
    max_token_corrections: int


def _parse_pipeline_config() -> PipelineConfig:
    """Parse command-line arguments into a typed pipeline configuration."""
    values: dict[str, Any] = vars(_build_parser().parse_args())
    return PipelineConfig(**values)


def _run_cli(config: PipelineConfig) -> RunResult:
    """Adapt CLI-only configuration and invoke the reusable pipeline API."""
    return main(
        score_threshold=config.score_threshold,
        max_iterations=config.max_iterations,
        max_stagnation=config.max_stagnation,
        thread_id=config.thread_id,
        checkpoint_db=config.checkpoint_db,
        dataset_token_multiplier=config.dataset_token_multiplier,
        solution_token_multiplier=config.solution_token_multiplier,
        evaluation_token_multiplier=config.evaluation_token_multiplier,
        refinement_token_multiplier=config.refinement_token_multiplier,
        max_token_corrections=config.max_token_corrections,
        demo_mode=config.demo_mode,
        runs_root=config.runs_root,
        resume_run_id=config.resume_run_id,
        download_link_provider=_cli_download_link_provider(config.download_base_url),
    )


if __name__ == "__main__":
    _run_cli(_parse_pipeline_config())
