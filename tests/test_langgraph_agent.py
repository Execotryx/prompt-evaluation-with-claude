import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


DATASET = [
    {
        "task": "Task A",
        "solution_criteria": "Criteria A",
    }
]

SOLUTIONS = [
    {
        "task": "Task A",
        "solution": "Solution A",
    }
]


def results(score: float) -> list[dict[str, object]]:
    return [
        {
            "task": "Task A",
            "score": score,
            "strengths": ["ok"],
            "weaknesses": ["needs more"],
        }
    ]


class DummyClaudeClient:
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def ask(self, *args, **kwargs):
        raise AssertionError("Model calls should not run when cache files are present.")


class LangGraphAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_json(self, name: str, payload: object) -> str:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def state(self, *, score: float, max_iterations: int = 1, max_stagnation: int = 1) -> main.AgentState:
        return {
            "score_threshold": 9.5,
            "max_iterations": max_iterations,
            "max_stagnation": max_stagnation,
            "iteration": 0,
            "stagnation_count": 0,
            "dataset_file": self.write_json("evaluation_dataset.json", DATASET),
            "solutions_file": self.write_json("solutions.json", SOLUTIONS),
            "evaluation_file": self.write_json("evaluation_results.json", results(score)),
            "best_dataset_file": str(self.root / "best_dataset.json"),
            "refined_dataset_pattern": str(self.root / "refined_dataset_{iteration}.json"),
            "refined_solutions_pattern": str(self.root / "refined_solutions_{iteration}.json"),
            "refined_evaluation_pattern": str(self.root / "refined_evaluation_results_{iteration}.json"),
        }

    def invoke(self, state: main.AgentState, thread_id: str = "test-thread") -> main.AgentState:
        graph = main.build_prompt_evaluation_agent(str(self.root / "checkpoints.sqlite"))
        try:
            with patch.object(main, "ClaudeClient", DummyClaudeClient):
                return graph.invoke(state, {"configurable": {"thread_id": thread_id}})
        finally:
            graph.checkpointer.conn.close()

    def test_threshold_reached_after_initial_evaluation(self):
        state = self.state(score=10.0)

        final_state = self.invoke(state)

        self.assertEqual(final_state.get("stop_reason"), "score_threshold")
        self.assertEqual(final_state.get("iteration"), 0)
        self.assertEqual(final_state.get("best_score"), 10.0)
        best_dataset_file = final_state.get("best_dataset_file")
        self.assertIsNotNone(best_dataset_file)
        self.assertTrue(os.path.exists(best_dataset_file or ""))

    def test_max_iteration_stop_without_refinement(self):
        state = self.state(score=5.0, max_iterations=0)

        final_state = self.invoke(state)

        self.assertEqual(final_state.get("stop_reason"), "max_iterations")
        self.assertEqual(final_state.get("iteration"), 0)
        self.assertEqual(final_state.get("best_score"), 5.0)

    def test_non_improvement_increments_stagnation_and_stops(self):
        state = self.state(score=5.0, max_iterations=3, max_stagnation=1)
        self.write_json("refined_dataset_1.json", DATASET)
        self.write_json("refined_solutions_1.json", SOLUTIONS)
        self.write_json("refined_evaluation_results_1.json", results(4.0))

        final_state = self.invoke(state)

        self.assertEqual(final_state.get("stop_reason"), "stagnation")
        self.assertEqual(final_state.get("iteration"), 1)
        self.assertEqual(final_state.get("stagnation_count"), 1)
        self.assertEqual(final_state.get("best_score"), 5.0)

    def test_improvement_resets_stagnation_and_updates_best(self):
        refined_dataset = [{"task": "Task A", "solution_criteria": "Better criteria"}]
        state = self.state(score=5.0, max_iterations=1, max_stagnation=3)
        self.write_json("refined_dataset_1.json", refined_dataset)
        self.write_json("refined_solutions_1.json", SOLUTIONS)
        self.write_json("refined_evaluation_results_1.json", results(6.0))

        final_state = self.invoke(state)

        self.assertEqual(final_state.get("stop_reason"), "max_iterations")
        self.assertEqual(final_state.get("iteration"), 1)
        self.assertEqual(final_state.get("stagnation_count"), 0)
        self.assertEqual(final_state.get("best_score"), 6.0)
        self.assertEqual(final_state.get("best_dataset"), refined_dataset)

    def test_refine_dataset_uses_best_dataset_not_current_dataset(self):
        current_dataset = [{"task": "Task A", "solution_criteria": "Current criteria"}]
        best_dataset = [{"task": "Task A", "solution_criteria": "Best criteria"}]
        captured = {}

        class FakeTaskRefiner:
            def __init__(self, *args, **kwargs):
                pass

            def refine_dataset(self, eval_dataset, evaluation_results, token_multiplier=main.REFINEMENT_TOKEN_MULTIPLIER):
                captured["dataset"] = eval_dataset
                captured["results"] = evaluation_results
                captured["token_multiplier"] = token_multiplier
                return eval_dataset

        state: main.AgentState = {
            "dataset": current_dataset,
            "best_dataset": best_dataset,
            "best_results": results(7.0),
            "iteration": 0,
            "refined_dataset_pattern": str(self.root / "refined_dataset_{iteration}.json"),
            "refinement_token_multiplier": 4.0,
        }

        with patch.object(main, "TaskRefiner", FakeTaskRefiner), patch.object(main, "ClaudeClient", DummyClaudeClient):
            update = main.refine_dataset(state)

        self.assertIsNone(update.get("last_error"))
        self.assertEqual(captured["dataset"], best_dataset)
        self.assertEqual(captured["token_multiplier"], 4.0)
        self.assertEqual(update.get("dataset"), best_dataset)

    def test_solution_and_evaluation_nodes_use_expected_artifact_paths(self) -> None:
        solution_paths: list[str | None] = []
        evaluation_paths: list[str | None] = []

        class FakeSolutionGenerator:
            DEFAULT_SOLUTIONS_FILE = "solutions.json"

            def __init__(self, client: object, output_file: str | None = None) -> None:
                solution_paths.append(output_file)

            def generate_solutions(
                self,
                eval_dataset: list[dict[str, str]],
                token_multiplier: float = main.SOLUTION_TOKEN_MULTIPLIER,
            ) -> list[dict[str, str]]:
                return SOLUTIONS

        class FakePromptEvaluator:
            DEFAULT_EVALUATION_FILE = "evaluation_results.json"

            def __init__(self, client: object, output_file: str | None = None) -> None:
                evaluation_paths.append(output_file)

            def evaluate_prompts(
                self,
                eval_dataset: list[dict[str, str]],
                solutions: list[dict[str, str]],
                token_multiplier: float = main.EVALUATION_TOKEN_MULTIPLIER,
            ) -> list[dict[str, object]]:
                return results(7.0)

        state: main.AgentState = {
            "dataset": DATASET,
            "solutions": SOLUTIONS,
            "solutions_file": str(self.root / "initial_solutions.json"),
            "evaluation_file": str(self.root / "initial_evaluation.json"),
            "iteration": 0,
            "refined_solutions_pattern": str(self.root / "refined_solutions_{iteration}.json"),
            "refined_evaluation_pattern": str(self.root / "refined_evaluation_{iteration}.json"),
        }

        with (
            patch.object(main, "SolutionGenerator", FakeSolutionGenerator),
            patch.object(main, "PromptEvaluator", FakePromptEvaluator),
            patch.object(main, "ClaudeClient", DummyClaudeClient),
        ):
            main.load_or_generate_solutions(state)
            main.generate_refined_solutions(state)
            main.load_or_evaluate_solutions(state)
            main.evaluate_refined_solutions(state)

        self.assertEqual(
            solution_paths,
            [
                str(self.root / "initial_solutions.json"),
                str(self.root / "refined_solutions_1.json"),
            ],
        )
        self.assertEqual(
            evaluation_paths,
            [
                str(self.root / "initial_evaluation.json"),
                str(self.root / "refined_evaluation_1.json"),
            ],
        )

    def test_graph_construction_tables_include_expected_routes(self) -> None:
        node_names = {node_name for node_name, _ in main.GRAPH_NODES}
        stage_sources = {source_node for source_node, _ in main.STAGE_EDGE_SPECS}

        self.assertIn("apply_token_multiplier_correction", node_names)
        self.assertIn("finalize", node_names)
        self.assertIn("generate_refined_solutions", stage_sources)
        self.assertEqual(main.STAGE_ROUTE_EXITS["apply_token_multiplier_correction"], "apply_token_multiplier_correction")
        self.assertEqual(main.TOKEN_CORRECTION_ROUTES["evaluate_refined_solutions"], "evaluate_refined_solutions")

    def test_node_error_records_failed_stage(self):
        update: main.AgentState = main._node_error("refine_dataset", RuntimeError("boom"))
        finalized: main.AgentState = main.finalize(update)

        self.assertEqual(finalized.get("current_stage"), "refine_dataset")
        self.assertEqual(finalized.get("failed_stage"), "refine_dataset")
        self.assertEqual(finalized.get("last_error"), "RuntimeError: boom")
        self.assertEqual(finalized.get("stop_reason"), "error")

    def test_max_tokens_error_routes_to_correction(self):
        state: main.AgentState = {
            "failed_stage": "refine_dataset",
            "last_error": "RuntimeError: Response was truncated (stop_reason='max_tokens').",
            "token_correction_count": 0,
            "max_token_corrections": 1,
        }

        route = main._route_after_stage(state, "generate_refined_solutions")

        self.assertEqual(route, "apply_token_multiplier_correction")

    def test_token_correction_increments_failed_stage_multiplier_by_one(self):
        state: main.AgentState = {
            "failed_stage": "refine_dataset",
            "last_error": "RuntimeError: Response was truncated (stop_reason='max_tokens').",
            "refinement_token_multiplier": 2.0,
            "token_correction_count": 0,
        }

        update = main.apply_token_multiplier_correction(state)

        self.assertEqual(update.get("refinement_token_multiplier"), 3.0)
        self.assertEqual(update.get("retry_stage"), "refine_dataset")
        self.assertEqual(update.get("token_correction_count"), 1)
        self.assertIsNone(update.get("last_error"))
        self.assertIsNone(update.get("stop_reason"))

    def test_token_correction_limit_routes_to_finalize(self):
        state: main.AgentState = {
            "failed_stage": "refine_dataset",
            "last_error": "RuntimeError: Response was truncated (stop_reason='max_tokens').",
            "token_correction_count": 1,
            "max_token_corrections": 1,
        }

        route = main._route_after_stage(state, "generate_refined_solutions")

        self.assertEqual(route, "finalize")

    def test_initial_state_can_clear_stale_error_checkpoint(self):
        checkpoint_db = str(self.root / "checkpoints.sqlite")
        graph = main.build_prompt_evaluation_agent(checkpoint_db)
        config = {"configurable": {"thread_id": "stale-error-thread"}}
        stale_state: main.AgentState = {
            "score_threshold": 9.5,
            "max_iterations": 0,
            "max_stagnation": 1,
            "stop_reason": "error",
            "last_error": None,
            "dataset_file": self.write_json("evaluation_dataset.json", DATASET),
            "solutions_file": self.write_json("solutions.json", SOLUTIONS),
            "evaluation_file": self.write_json("evaluation_results.json", results(5.0)),
            "best_dataset_file": str(self.root / "best_dataset.json"),
        }
        fresh_state: main.AgentState = {
            **stale_state,
            "current_stage": "start",
            "failed_stage": None,
            "retry_stage": None,
            "last_error": None,
            "stop_reason": None,
        }

        try:
            with patch.object(main, "ClaudeClient", DummyClaudeClient):
                graph.invoke(stale_state, config)
                final_state = graph.invoke(fresh_state, config)
        finally:
            graph.checkpointer.conn.close()

        self.assertEqual(final_state["stop_reason"], "max_iterations")
        self.assertIsNone(final_state.get("last_error"))

    def test_latest_checkpoint_error_recovers_non_empty_write(self):
        checkpoint_db = str(self.root / "checkpoints.sqlite")
        graph = main.build_prompt_evaluation_agent(checkpoint_db)
        try:
            value_type, value = graph.checkpointer.serde.dumps_typed("RuntimeError: boom")
        finally:
            graph.checkpointer.conn.close()
        conn = sqlite3.connect(checkpoint_db)
        try:
            conn.execute(
                """
                INSERT INTO writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value)
                VALUES (?, '', ?, ?, ?, ?, ?, ?)
                """,
                ("error-history-thread", "checkpoint-1", "task-1", 1, "last_error", value_type, value),
            )
            conn.commit()
        finally:
            conn.close()

        self.assertEqual(main._latest_checkpoint_error(checkpoint_db, "error-history-thread"), "RuntimeError: boom")

    def test_sqlite_checkpoint_persists_final_state(self):
        checkpoint_db = str(self.root / "checkpoints.sqlite")
        state = self.state(score=10.0)
        config = {"configurable": {"thread_id": "persistent-thread"}}
        graph = main.build_prompt_evaluation_agent(checkpoint_db)

        try:
            with patch.object(main, "ClaudeClient", DummyClaudeClient):
                final_state = graph.invoke(state, config)
        finally:
            graph.checkpointer.conn.close()

        reloaded_graph = main.build_prompt_evaluation_agent(checkpoint_db)
        try:
            snapshot = reloaded_graph.get_state(config)
        finally:
            reloaded_graph.checkpointer.conn.close()

        self.assertEqual(final_state.get("stop_reason"), "score_threshold")
        self.assertEqual(snapshot.values["stop_reason"], "score_threshold")
        self.assertEqual(snapshot.values["best_score"], 10.0)


if __name__ == "__main__":
    unittest.main()
