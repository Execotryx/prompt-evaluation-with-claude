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
import anthropic
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Literal, Any, TypedDict
from json import dumps, loads
from tqdm import tqdm
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

MODEL: str = "claude-sonnet-4-6"
DATASET_MODEL: str = "claude-haiku-4-5"
MAX_TOKENS: int = 1024
SCORE_THRESHOLD: float = 9.5
MAX_REFINEMENT_ITERATIONS: int = 10
MAX_STAGNATION_ITERATIONS: int = 3
BEST_DATASET_FILE: str = "best_dataset.json"


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

    DATASET_FILE: str = "evaluation_dataset.json"

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

    def create_dataset(self) -> list[dict[str, Any]]:
        """Generate or load the evaluation dataset.

        If ``evaluation_dataset.json`` does not exist, the model is called to
        produce the dataset which is then saved to that file. If the file
        already exists it is loaded directly without calling the model.

        Returns:
            list[dict[str, Any]]: A list of dicts, each with ``"task"`` and
            ``"solution_criteria"`` keys.
        """
        if not os.path.exists(self._output_file):
            response: str = self.client.ask(
                question=self._build_prompt(),
                token_multiplier=2.0,
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

    DEFAULT_SOLUTIONS_FILE: str = "solutions.json"

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

    def _solve_task(self, task: str, solution_criteria: str) -> dict[str, str]:
        """Ask the model to solve a single task.

        Args:
            task (str): The task description.
            solution_criteria (str): Criteria the solution must satisfy.

        Returns:
            dict[str, str]: A dict with ``"task"`` and ``"solution"`` keys.
        """
        self.client.reset()
        response: str = self.client.ask(
            question=self._build_prompt(task=task, solution_criteria=solution_criteria),
            token_multiplier=2.0,
            output_config=self._build_output_config(),
        )
        return {"task": task, "solution": loads(response)["solution"]}

    def generate_solutions(self, eval_dataset: list[dict[str, str]]) -> list[dict[str, str]]:
        """Generate solutions for every task in the evaluation dataset.

        If the output file already exists it is returned directly without
        calling the model. Otherwise each task is solved in sequence and the
        results are saved to the output file.

        Args:
            eval_dataset (list[dict[str, str]]): List of task dicts as produced by
                ``EvaluationDatasetGenerator.create_dataset()``.

        Returns:
            list[dict[str, str]]: A list of dicts, each with ``"task"`` and ``"solution"`` keys.
        """
        if os.path.exists(self._output_file):
            with open(self._output_file, "r") as f:
                return loads(f.read())

        solutions: list[dict[str, str]] = [
            self._solve_task(item["task"], item["solution_criteria"])
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

    DEFAULT_EVALUATION_FILE: str = "evaluation_results.json"

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

    def _evaluate_task(self, task: str, solution: str, solution_criteria: str) -> dict[str, Any]:
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

        Returns:
            dict[str, Any]: A dict with ``"task"`` (str), ``"score"`` (float),
            ``"strengths"`` (list[str]), and ``"weaknesses"`` (list[str]) keys.
        """
        self.client.reset()
        response: str = self.client.ask(
            question=self._build_prompt(task=task, solution=solution, solution_criteria=solution_criteria),
            token_multiplier=1.0,
            output_config=self._build_output_config(),
        )
        result: dict[str, Any] = loads(response)
        result["task"] = task
        return result

    def evaluate_prompts(self, eval_dataset: list[dict[str, str]], solutions: list[dict[str, str]]) -> list[dict[str, Any]]:
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
            self._evaluate_task(item["task"], solution_map[item["task"]], item["solution_criteria"])
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

    def _refine_task(self, task: str, solution_criteria: str, weaknesses: list[str]) -> dict[str, str]:
        """Ask the model to rewrite the solution criteria for a task.

        The task description is passed to the model for context only and is
        never part of the output — it is copied from the input unchanged.
        The history is reset before the call so prior refinements do not bleed in.

        Args:
            task (str): The original task description (carried through unchanged).
            solution_criteria (str): The original solution criteria to rewrite.
            weaknesses (list[str]): Weaknesses identified in the evaluated solution.

        Returns:
            dict[str, str]: A dict with the original ``"task"`` and a rewritten
            ``"solution_criteria"`` as a Markdown bulleted list.
        """
        self.client.reset()
        response: str = self.client.ask(
            question=self._build_prompt(task=task, solution_criteria=solution_criteria, weaknesses=weaknesses),
            token_multiplier=2.0,
            output_config=self._build_output_config(),
        )
        return {"task": task, "solution_criteria": loads(response)["solution_criteria"]}

    def refine_dataset(
        self,
        eval_dataset: list[dict[str, str]],
        evaluation_results: list[dict[str, Any]],
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
            self._refine_task(item["task"], item["solution_criteria"], weakness_map[item["task"]])
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
    current_stage: str
    last_error: str | None
    stop_reason: str | None
    dataset_file: str
    solutions_file: str
    evaluation_file: str
    best_dataset_file: str
    refined_dataset_pattern: str
    refined_solutions_pattern: str
    refined_evaluation_pattern: str


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
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty list.")

    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be an object.")
        if not isinstance(item.get("task"), str) or not item["task"]:
            raise ValueError(f"Dataset item {index} must include a non-empty task string.")
        if not isinstance(item.get("solution_criteria"), str) or not item["solution_criteria"]:
            raise ValueError(f"Dataset item {index} must include a non-empty solution_criteria string.")

    return dataset


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
    if not isinstance(solutions, list) or not solutions:
        raise ValueError("Solutions must be a non-empty list.")

    for index, item in enumerate(solutions):
        if not isinstance(item, dict):
            raise ValueError(f"Solution item {index} must be an object.")
        if not isinstance(item.get("task"), str) or not item["task"]:
            raise ValueError(f"Solution item {index} must include a non-empty task string.")
        if not isinstance(item.get("solution"), str) or not item["solution"]:
            raise ValueError(f"Solution item {index} must include a non-empty solution string.")

    return solutions


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
    if not isinstance(results, list) or not results:
        raise ValueError("Evaluation results must be a non-empty list.")

    for index, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation result {index} must be an object.")
        if not isinstance(item.get("task"), str) or not item["task"]:
            raise ValueError(f"Evaluation result {index} must include a non-empty task string.")
        if isinstance(item.get("score"), bool) or not isinstance(item.get("score"), (int, float)):
            raise ValueError(f"Evaluation result {index} must include a numeric score.")
        if not isinstance(item.get("strengths"), list):
            raise ValueError(f"Evaluation result {index} must include a strengths list.")
        if not isinstance(item.get("weaknesses"), list):
            raise ValueError(f"Evaluation result {index} must include a weaknesses list.")

    return results


def _node_error(stage: str, exc: Exception) -> AgentState:
    """Represent a recoverable node failure in graph state.

    Args:
        stage (str): Name of the node or stage where the exception occurred.
        exc (Exception): Exception raised by the node.

    Returns:
        AgentState: Partial state update that records the error and routes the
            graph toward finalization.
    """
    return {
        "current_stage": stage,
        "last_error": f"{type(exc).__name__}: {exc}",
        "stop_reason": "error",
    }


def load_or_generate_dataset(state: AgentState) -> AgentState:
    """Load or generate the initial evaluation dataset.

    Args:
        state (AgentState): Current graph state, optionally containing
            ``dataset_file``.

    Returns:
        AgentState: Partial state with validated ``dataset`` on success, or
            ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "load_or_generate_dataset"
    try:
        dataset_file: str = _state_value(state, "dataset_file", EvaluationDatasetGenerator.DATASET_FILE)
        dataset: list[dict[str, Any]] = EvaluationDatasetGenerator(
            ClaudeClient(model=DATASET_MODEL),
            output_file=dataset_file,
        ).create_dataset()
        return {
            "dataset": _validate_dataset(dataset),
            "current_stage": stage,
            "last_error": None,
        }
    except Exception as exc:
        return _node_error(stage, exc)


def load_or_generate_solutions(state: AgentState) -> AgentState:
    """Load or generate solutions for the current dataset.

    Args:
        state (AgentState): Current graph state containing ``dataset`` and
            optionally ``solutions_file``.

    Returns:
        AgentState: Partial state with validated ``solutions`` on success, or
            ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "load_or_generate_solutions"
    try:
        dataset: list[dict[str, str]] = _validate_dataset(state.get("dataset"))
        solutions_file: str = _state_value(state, "solutions_file", SolutionGenerator.DEFAULT_SOLUTIONS_FILE)
        solutions: list[dict[str, str]] = SolutionGenerator(
            ClaudeClient(),
            output_file=solutions_file,
        ).generate_solutions(dataset)
        return {
            "solutions": _validate_solutions(solutions),
            "current_stage": stage,
            "last_error": None,
        }
    except Exception as exc:
        return _node_error(stage, exc)


def load_or_evaluate_solutions(state: AgentState) -> AgentState:
    """Load or evaluate solutions for the current dataset.

    Args:
        state (AgentState): Current graph state containing ``dataset``,
            ``solutions``, and optionally ``evaluation_file``.

    Returns:
        AgentState: Partial state with validated ``evaluation_results`` on
            success, or ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "load_or_evaluate_solutions"
    try:
        dataset: list[dict[str, str]] = _validate_dataset(state.get("dataset"))
        solutions: list[dict[str, str]] = _validate_solutions(state.get("solutions"))
        evaluation_file: str = _state_value(state, "evaluation_file", PromptEvaluator.DEFAULT_EVALUATION_FILE)
        results: list[dict[str, Any]] = PromptEvaluator(
            ClaudeClient(),
            output_file=evaluation_file,
        ).evaluate_prompts(dataset, solutions)
        return {
            "evaluation_results": _validate_evaluation_results(results),
            "current_stage": stage,
            "last_error": None,
        }
    except Exception as exc:
        return _node_error(stage, exc)


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
        return {
            "best_dataset": dataset,
            "best_results": results,
            "best_score": best_score,
            "iteration": int(_state_value(state, "iteration", 0)),
            "stagnation_count": int(_state_value(state, "stagnation_count", 0)),
            "current_stage": stage,
            "last_error": None,
        }
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

        return {
            "current_stage": stage,
            "last_error": None,
            "stop_reason": stop_reason,
        }
    except Exception as exc:
        return _node_error(stage, exc)


def refine_dataset(state: AgentState) -> AgentState:
    """Refine solution criteria from the best-scoring dataset so far.

    Args:
        state (AgentState): Current graph state containing ``best_dataset``,
            ``best_results``, iteration count, and refined artifact path pattern.

    Returns:
        AgentState: Partial state with the newly refined ``dataset`` on success,
            or ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "refine_dataset"
    try:
        best_dataset: list[dict[str, str]] = _validate_dataset(state.get("best_dataset"))
        best_results: list[dict[str, Any]] = _validate_evaluation_results(state.get("best_results"))
        output_file: str = _refined_artifact_path(state, "refined_dataset_pattern", "refined_dataset_{iteration}.json")
        dataset: list[dict[str, str]] = TaskRefiner(
            ClaudeClient(),
            output_file=output_file,
        ).refine_dataset(best_dataset, best_results)
        return {
            "dataset": _validate_dataset(dataset),
            "current_stage": stage,
            "last_error": None,
        }
    except Exception as exc:
        return _node_error(stage, exc)


def generate_refined_solutions(state: AgentState) -> AgentState:
    """Load or generate solutions for the refined dataset.

    Args:
        state (AgentState): Current graph state containing the refined
            ``dataset``, iteration count, and refined solution artifact pattern.

    Returns:
        AgentState: Partial state with validated refined ``solutions`` on
            success, or ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "generate_refined_solutions"
    try:
        dataset: list[dict[str, str]] = _validate_dataset(state.get("dataset"))
        output_file: str = _refined_artifact_path(state, "refined_solutions_pattern", "refined_solutions_{iteration}.json")
        solutions: list[dict[str, str]] = SolutionGenerator(
            ClaudeClient(),
            output_file=output_file,
        ).generate_solutions(dataset)
        return {
            "solutions": _validate_solutions(solutions),
            "current_stage": stage,
            "last_error": None,
        }
    except Exception as exc:
        return _node_error(stage, exc)


def evaluate_refined_solutions(state: AgentState) -> AgentState:
    """Load or evaluate solutions for the refined dataset.

    Args:
        state (AgentState): Current graph state containing refined ``dataset``,
            ``solutions``, iteration count, and refined evaluation artifact pattern.

    Returns:
        AgentState: Partial state with validated refined ``evaluation_results``
            on success, or ``last_error`` and ``stop_reason`` on failure.
    """
    stage = "evaluate_refined_solutions"
    try:
        dataset: list[dict[str, str]] = _validate_dataset(state.get("dataset"))
        solutions: list[dict[str, str]] = _validate_solutions(state.get("solutions"))
        output_file: str = _refined_artifact_path(state, "refined_evaluation_pattern", "refined_evaluation_results_{iteration}.json")
        results: list[dict[str, Any]] = PromptEvaluator(
            ClaudeClient(),
            output_file=output_file,
        ).evaluate_prompts(dataset, solutions)
        return {
            "evaluation_results": _validate_evaluation_results(results),
            "current_stage": stage,
            "last_error": None,
        }
    except Exception as exc:
        return _node_error(stage, exc)


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

        updates: AgentState = {
            "iteration": iteration,
            "current_stage": stage,
            "last_error": None,
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

        return updates
    except Exception as exc:
        return _node_error(stage, exc)


def finalize(state: AgentState) -> AgentState:
    """Mark graph execution complete and preserve the final state summary.

    Args:
        state (AgentState): Current graph state, optionally containing an
            existing ``stop_reason``.

    Returns:
        AgentState: Partial state with ``current_stage`` set to ``"finalize"``
            and ``stop_reason`` populated.
    """
    stop_reason = state.get("stop_reason")
    if not stop_reason:
        stop_reason = "complete"
    return {
        "current_stage": "finalize",
        "stop_reason": stop_reason,
    }


def _route_after_stage(state: AgentState, next_node: str) -> str:
    """Route to finalize after an error, otherwise continue to the next node.

    Args:
        state (AgentState): Current graph state.
        next_node (str): Node name to route to when there is no error.

    Returns:
        str: ``"finalize"`` when an error is present, otherwise ``next_node``.
    """
    if state.get("last_error") or state.get("stop_reason") == "error":
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


def build_prompt_evaluation_agent(checkpoint_db: str = "langgraph_checkpoints.sqlite") -> Any:
    """Build and compile the LangGraph prompt-evaluation agent.

    Args:
        checkpoint_db (str): SQLite database path used by the LangGraph
            checkpointer.

    Returns:
        Any: Compiled LangGraph application ready to invoke with ``AgentState``
            input and a ``thread_id`` config.
    """
    workflow: StateGraph = StateGraph(AgentState)

    workflow.add_node("load_or_generate_dataset", load_or_generate_dataset)
    workflow.add_node("load_or_generate_solutions", load_or_generate_solutions)
    workflow.add_node("load_or_evaluate_solutions", load_or_evaluate_solutions)
    workflow.add_node("initialize_best", initialize_best)
    workflow.add_node("decide_next_step", decide_next_step)
    workflow.add_node("refine_dataset", refine_dataset)
    workflow.add_node("generate_refined_solutions", generate_refined_solutions)
    workflow.add_node("evaluate_refined_solutions", evaluate_refined_solutions)
    workflow.add_node("update_best_or_stagnation", update_best_or_stagnation)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "load_or_generate_dataset")
    workflow.add_conditional_edges(
        "load_or_generate_dataset",
        lambda state: _route_after_stage(state, "load_or_generate_solutions"),
        {"load_or_generate_solutions": "load_or_generate_solutions", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "load_or_generate_solutions",
        lambda state: _route_after_stage(state, "load_or_evaluate_solutions"),
        {"load_or_evaluate_solutions": "load_or_evaluate_solutions", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "load_or_evaluate_solutions",
        lambda state: _route_after_stage(state, "initialize_best"),
        {"initialize_best": "initialize_best", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "initialize_best",
        lambda state: _route_after_stage(state, "decide_next_step"),
        {"decide_next_step": "decide_next_step", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "decide_next_step",
        _route_after_decision,
        {"refine_dataset": "refine_dataset", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "refine_dataset",
        lambda state: _route_after_stage(state, "generate_refined_solutions"),
        {"generate_refined_solutions": "generate_refined_solutions", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "generate_refined_solutions",
        lambda state: _route_after_stage(state, "evaluate_refined_solutions"),
        {"evaluate_refined_solutions": "evaluate_refined_solutions", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "evaluate_refined_solutions",
        lambda state: _route_after_stage(state, "update_best_or_stagnation"),
        {"update_best_or_stagnation": "update_best_or_stagnation", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "update_best_or_stagnation",
        lambda state: _route_after_stage(state, "decide_next_step"),
        {"decide_next_step": "decide_next_step", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)

    conn: sqlite3.Connection = sqlite3.connect(checkpoint_db, check_same_thread=False)
    checkpointer: SqliteSaver = SqliteSaver(conn)
    _ensure_sqlite_metadata_serializer(checkpointer)
    checkpointer.setup()
    return workflow.compile(checkpointer=checkpointer)


def main(
    score_threshold: float = SCORE_THRESHOLD,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    max_stagnation: int = MAX_STAGNATION_ITERATIONS,
    thread_id: str = "prompt-evaluation",
    checkpoint_db: str = "langgraph_checkpoints.sqlite",
) -> None:
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
        thread_id (str): LangGraph checkpoint thread ID used for resumable runs.
        checkpoint_db (str): SQLite database path used for LangGraph checkpoints.

    Returns:
        None.
    """
    print("=== LangGraph prompt-evaluation agent ===")
    print(f"Thread: {thread_id}")
    print(f"Checkpoint DB: {checkpoint_db}")

    graph: Any = build_prompt_evaluation_agent(checkpoint_db=checkpoint_db)
    initial_state: AgentState = {
        "score_threshold": score_threshold,
        "max_iterations": max_iterations,
        "max_stagnation": max_stagnation,
        "iteration": 0,
        "stagnation_count": 0,
        "dataset_file": EvaluationDatasetGenerator.DATASET_FILE,
        "solutions_file": SolutionGenerator.DEFAULT_SOLUTIONS_FILE,
        "evaluation_file": PromptEvaluator.DEFAULT_EVALUATION_FILE,
        "best_dataset_file": BEST_DATASET_FILE,
        "refined_dataset_pattern": "refined_dataset_{iteration}.json",
        "refined_solutions_pattern": "refined_solutions_{iteration}.json",
        "refined_evaluation_pattern": "refined_evaluation_results_{iteration}.json",
    }
    config: dict[str, dict[str, str]] = {"configurable": {"thread_id": thread_id}}
    final_state: AgentState = graph.invoke(initial_state, config)

    last_error: str | None = final_state.get("last_error")
    if last_error:
        print(f"Stopped with error at {final_state.get('current_stage')}: {last_error}")
        return

    best_score = float(_state_value(final_state, "best_score", 0.0))
    iteration = int(_state_value(final_state, "iteration", 0))
    stop_reason = _state_value(final_state, "stop_reason", "complete")
    print(f"Stopped: {stop_reason}")
    print(f"Best mean score: {best_score:.2f}")
    print(f"Refinement passes completed: {iteration}")
    print(f"Best dataset: {_state_value(final_state, 'best_dataset_file', BEST_DATASET_FILE)}")


class PipelineArgs:
    """Parses and exposes command-line arguments for the evaluation pipeline."""

    def __init__(self) -> None:
        """Parse ``sys.argv`` and store the resulting namespace for property access.

        Returns:
            None.
        """
        parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description="Prompt evaluation pipeline with iterative refinement."
        )
        parser.add_argument(
            "--score",
            type=float,
            default=SCORE_THRESHOLD,
            metavar="N",
            help=f"target mean score threshold, 1–10 (default: {SCORE_THRESHOLD})",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=MAX_REFINEMENT_ITERATIONS,
            metavar="N",
            help=f"maximum number of refinement passes (default: {MAX_REFINEMENT_ITERATIONS})",
        )
        parser.add_argument(
            "--stagnation",
            type=int,
            default=MAX_STAGNATION_ITERATIONS,
            metavar="N",
            help=f"stop after N consecutive non-improving iterations (default: {MAX_STAGNATION_ITERATIONS})",
        )
        parser.add_argument(
            "--thread-id",
            type=str,
            default="prompt-evaluation",
            help="LangGraph checkpoint thread ID used for resumable runs",
        )
        parser.add_argument(
            "--checkpoint-db",
            type=str,
            default="langgraph_checkpoints.sqlite",
            help="SQLite database path used for LangGraph checkpoints",
        )
        self.__args = parser.parse_args()

    @property
    def score(self) -> float:
        """Return the target mean score threshold supplied via ``--score``.

        Returns:
            float: Target mean score threshold.
        """
        return self.__args.score

    @property
    def iterations(self) -> int:
        """Return the maximum number of refinement passes.

        Returns:
            int: Value supplied via ``--iterations``.
        """
        return self.__args.iterations

    @property
    def stagnation(self) -> int:
        """Return the maximum consecutive non-improving iterations.

        Returns:
            int: Value supplied via ``--stagnation``.
        """
        return self.__args.stagnation

    @property
    def thread_id(self) -> str:
        """Return the LangGraph checkpoint thread ID.

        Returns:
            str: Value supplied via ``--thread-id``.
        """
        return self.__args.thread_id

    @property
    def checkpoint_db(self) -> str:
        """Return the SQLite checkpoint database path.

        Returns:
            str: Value supplied via ``--checkpoint-db``.
        """
        return self.__args.checkpoint_db


if __name__ == "__main__":
    args = PipelineArgs()
    main(
        score_threshold=args.score,
        max_iterations=args.iterations,
        max_stagnation=args.stagnation,
        thread_id=args.thread_id,
        checkpoint_db=args.checkpoint_db,
    )
