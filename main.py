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
import anthropic
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Literal, Any
from json import dumps, loads
from tqdm import tqdm

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

    def __init__(self, model: str = MODEL):
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

        message = self.client.messages.create(**params, stream=False)

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

    def __init__(self, client: ClaudeClient):
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
        if not os.path.exists(self.DATASET_FILE):
            response: str = self.client.ask(
                question=self._build_prompt(),
                token_multiplier=2.0,
                output_config=self._build_output_config(),
            )
            with open(self.DATASET_FILE, "w") as f:
                f.write(dumps(loads(response), indent=4))
        else:
            with open(self.DATASET_FILE, "r") as f:
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

    def __init__(self, client: ClaudeClient, output_file: str | None = None):
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

    def __init__(self, client: ClaudeClient, output_file: str | None = None):
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

    def __init__(self, client: ClaudeClient, output_file: str | None = None):
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
    """
    return sum(r["score"] for r in evaluation_results) / len(evaluation_results)


def _save_best_dataset(dataset: list[dict[str, str]]) -> None:
    """Overwrite ``best_dataset.json`` with the current best dataset.

    Args:
        dataset (list[dict[str, str]]): The dataset to persist, where each dict
            contains ``"task"`` (str) and ``"solution_criteria"`` (str) keys.
    """
    with open(BEST_DATASET_FILE, "w") as f:
        f.write(dumps(dataset, indent=4))


def main(
    score_threshold: float = SCORE_THRESHOLD,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    max_stagnation: int = MAX_STAGNATION_ITERATIONS,
) -> None:
    """Run the prompt evaluation pipeline with iterative refinement.

    Performs an initial pass (dataset generation → solution generation →
    evaluation), then repeatedly refines the dataset and re-evaluates until
    the mean evaluation score reaches ``score_threshold``,
    ``max_iterations`` is exhausted, or ``max_stagnation`` consecutive
    non-improving iterations occur.

    Each refinement iteration starts from the best-scoring dataset seen so
    far, not necessarily the most recent one. If a refinement produces a
    lower score than the current best, the best dataset is kept unchanged
    and the next iteration refines from it again. Whenever the best score
    improves, ``best_dataset.json`` is updated.

    Each refinement iteration writes its own set of output files suffixed
    with the iteration number (e.g. ``refined_dataset_1.json``,
    ``refined_solutions_1.json``, ``refined_evaluation_results_1.json``),
    so every pass is independently cached and inspectable.

    Args:
        score_threshold (float): Target mean score (1–10). Refinement stops once
            the best mean score meets or exceeds this value. Defaults to
            the module-level ``SCORE_THRESHOLD``.
        max_iterations (int): Maximum number of refinement passes before stopping.
            Defaults to the module-level ``MAX_REFINEMENT_ITERATIONS``.
        max_stagnation (int): Stop early after this many consecutive non-improving
            iterations. Defaults to the module-level ``MAX_STAGNATION_ITERATIONS``.
    """
    print("=== Initial pass ===")

    dataset: list[dict[str, str]] = EvaluationDatasetGenerator(ClaudeClient(model=DATASET_MODEL)).create_dataset()
    solutions: list[dict[str, str]] = SolutionGenerator(ClaudeClient()).generate_solutions(dataset)
    evaluation_results: list[dict[str, Any]] = PromptEvaluator(ClaudeClient()).evaluate_prompts(dataset, solutions)

    best_dataset: list[dict[str, str]] = dataset
    best_results: list[dict[str, Any]] = evaluation_results
    best_score: float = _mean_score(evaluation_results)
    _save_best_dataset(best_dataset)

    stagnation_count: int = 0

    for iteration in range(1, max_iterations + 1):
        print(f"\nBest mean score so far: {best_score:.2f} / {score_threshold:.2f} threshold")

        if best_score >= score_threshold:
            print(f"Threshold reached after {iteration - 1} refinement(s).")
            break

        print(f"=== Refinement pass {iteration} (refining from best, score {best_score:.2f}) ===")

        refined_dataset: list[dict[str, str]] = TaskRefiner(
            ClaudeClient(),
            output_file=f"refined_dataset_{iteration}.json",
        ).refine_dataset(best_dataset, best_results)

        refined_solutions: list[dict[str, str]] = SolutionGenerator(
            ClaudeClient(),
            output_file=f"refined_solutions_{iteration}.json",
        ).generate_solutions(refined_dataset)

        refined_results: list[dict[str, Any]] = PromptEvaluator(
            ClaudeClient(),
            output_file=f"refined_evaluation_results_{iteration}.json",
        ).evaluate_prompts(refined_dataset, refined_solutions)

        refined_score: float = _mean_score(refined_results)

        if refined_score > best_score:
            best_score = refined_score
            best_dataset = refined_dataset
            best_results = refined_results
            stagnation_count = 0
            _save_best_dataset(best_dataset)
            print(f"New best score: {best_score:.2f} — {BEST_DATASET_FILE} updated.")
        else:
            stagnation_count += 1
            print(f"Score did not improve ({refined_score:.2f} <= {best_score:.2f}), stagnation {stagnation_count}/{max_stagnation}.")
            if stagnation_count >= max_stagnation:
                print(f"No improvement for {max_stagnation} consecutive refinement(s), stopping early.")
                break
    else:
        print(f"\nStopped after {max_iterations} refinement(s) (best score: {best_score:.2f}).")


class PipelineArgs:
    """Parses and exposes command-line arguments for the evaluation pipeline."""

    def __init__(self) -> None:
        """Parse ``sys.argv`` and store the resulting namespace for property access."""
        parser = argparse.ArgumentParser(
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
        self.__args = parser.parse_args()

    @property
    def score(self) -> float:
        """float: Target mean score threshold supplied via ``--score``."""
        return self.__args.score

    @property
    def iterations(self) -> int:
        """int: Maximum number of refinement passes supplied via ``--iterations``."""
        return self.__args.iterations

    @property
    def stagnation(self) -> int:
        """int: Maximum consecutive non-improving iterations supplied via ``--stagnation``."""
        return self.__args.stagnation


if __name__ == "__main__":
    args = PipelineArgs()
    main(score_threshold=args.score, max_iterations=args.iterations, max_stagnation=args.stagnation)
