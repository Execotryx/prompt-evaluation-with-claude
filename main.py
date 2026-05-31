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
from collections.abc import Hashable
from dotenv import load_dotenv
from typing import Literal, Any, TypedDict, cast
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
DATASET_TOKEN_MULTIPLIER: float = 2.0
SOLUTION_TOKEN_MULTIPLIER: float = 2.0
EVALUATION_TOKEN_MULTIPLIER: float = 1.0
REFINEMENT_TOKEN_MULTIPLIER: float = 2.0
MAX_TOKEN_CORRECTIONS: int = 3


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
    _add_graph_nodes(workflow)
    _add_stage_edges(workflow)
    return _compile_graph(workflow, checkpoint_db)


def main(
    score_threshold: float = SCORE_THRESHOLD,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    max_stagnation: int = MAX_STAGNATION_ITERATIONS,
    thread_id: str = "prompt-evaluation",
    checkpoint_db: str = "langgraph_checkpoints.sqlite",
    dataset_token_multiplier: float = DATASET_TOKEN_MULTIPLIER,
    solution_token_multiplier: float = SOLUTION_TOKEN_MULTIPLIER,
    evaluation_token_multiplier: float = EVALUATION_TOKEN_MULTIPLIER,
    refinement_token_multiplier: float = REFINEMENT_TOKEN_MULTIPLIER,
    max_token_corrections: int = MAX_TOKEN_CORRECTIONS,
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

    Returns:
        None.
    """
    print("=== LangGraph prompt-evaluation agent ===")
    print(f"Thread: {thread_id}")
    print(f"Checkpoint DB: {checkpoint_db}")
    print(
        "Token multipliers: "
        f"dataset={dataset_token_multiplier}, "
        f"solution={solution_token_multiplier}, "
        f"evaluation={evaluation_token_multiplier}, "
        f"refinement={refinement_token_multiplier}"
    )

    graph: Any = build_prompt_evaluation_agent(checkpoint_db=checkpoint_db)
    initial_state: AgentState = {
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
        "dataset_file": EvaluationDatasetGenerator.DATASET_FILE,
        "solutions_file": SolutionGenerator.DEFAULT_SOLUTIONS_FILE,
        "evaluation_file": PromptEvaluator.DEFAULT_EVALUATION_FILE,
        "best_dataset_file": BEST_DATASET_FILE,
        "refined_dataset_pattern": "refined_dataset_{iteration}.json",
        "refined_solutions_pattern": "refined_solutions_{iteration}.json",
        "refined_evaluation_pattern": "refined_evaluation_results_{iteration}.json",
        "dataset_token_multiplier": dataset_token_multiplier,
        "solution_token_multiplier": solution_token_multiplier,
        "evaluation_token_multiplier": evaluation_token_multiplier,
        "refinement_token_multiplier": refinement_token_multiplier,
    }
    config: dict[str, dict[str, str]] = {"configurable": {"thread_id": thread_id}}
    final_state: AgentState = graph.invoke(initial_state, config)

    last_error: str | None = final_state.get("last_error")
    stop_reason: str = _state_value(final_state, "stop_reason", "complete")
    if last_error:
        failed_stage: str = _state_value(final_state, "failed_stage", _state_value(final_state, "current_stage", "unknown"))
        current_stage: str = _state_value(final_state, "current_stage", "unknown")
        print(f"Stopped with error in stage {failed_stage} (current stage: {current_stage}): {last_error}")
        return

    if stop_reason == "error":
        failed_stage: str = _state_value(final_state, "failed_stage", _state_value(final_state, "current_stage", "unknown"))
        current_stage: str = _state_value(final_state, "current_stage", "unknown")
        checkpoint_error: str | None = _latest_checkpoint_error(checkpoint_db, thread_id)
        if checkpoint_error:
            print(f"Stopped with error in stage {failed_stage} (current stage: {current_stage}): {checkpoint_error}")
            return
        print(
            f"Stopped with error in stage {failed_stage} (current stage: {current_stage}), "
            "but no error details were present in the final graph state."
        )
        print("No non-empty last_error write was found in the checkpoint history.")
        return

    best_score = float(_state_value(final_state, "best_score", 0.0))
    iteration = int(_state_value(final_state, "iteration", 0))
    print(f"Stopped: {stop_reason}")
    print(f"Best mean score: {best_score:.2f}")
    print(f"Refinement passes completed: {iteration}")
    print(f"Best dataset: {_state_value(final_state, 'best_dataset_file', BEST_DATASET_FILE)}")


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


class PipelineArgs:
    """Parses and exposes command-line arguments for the evaluation pipeline."""

    def __init__(self) -> None:
        """Parse ``sys.argv`` and store the resulting namespace for property access.

        Returns:
            None.
        """
        self.__args = _build_parser().parse_args()

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

    @property
    def dataset_token_multiplier(self) -> float:
        """Return the dataset-generation token multiplier.

        Returns:
            float: Value supplied via ``--dataset-token-multiplier``.
        """
        return self.__args.dataset_token_multiplier

    @property
    def solution_token_multiplier(self) -> float:
        """Return the solution-generation token multiplier.

        Returns:
            float: Value supplied via ``--solution-token-multiplier``.
        """
        return self.__args.solution_token_multiplier

    @property
    def evaluation_token_multiplier(self) -> float:
        """Return the evaluation token multiplier.

        Returns:
            float: Value supplied via ``--evaluation-token-multiplier``.
        """
        return self.__args.evaluation_token_multiplier

    @property
    def refinement_token_multiplier(self) -> float:
        """Return the criteria-refinement token multiplier.

        Returns:
            float: Value supplied via ``--refinement-token-multiplier``.
        """
        return self.__args.refinement_token_multiplier

    @property
    def max_token_corrections(self) -> int:
        """Return the maximum automatic max-token corrections.

        Returns:
            int: Value supplied via ``--max-token-corrections``.
        """
        return self.__args.max_token_corrections


if __name__ == "__main__":
    args = PipelineArgs()
    main(
        score_threshold=args.score,
        max_iterations=args.iterations,
        max_stagnation=args.stagnation,
        thread_id=args.thread_id,
        checkpoint_db=args.checkpoint_db,
        dataset_token_multiplier=args.dataset_token_multiplier,
        solution_token_multiplier=args.solution_token_multiplier,
        evaluation_token_multiplier=args.evaluation_token_multiplier,
        refinement_token_multiplier=args.refinement_token_multiplier,
        max_token_corrections=args.max_token_corrections,
    )
