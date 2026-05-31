import json
import os
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

        self.assertEqual(final_state["stop_reason"], "score_threshold")
        self.assertEqual(final_state["iteration"], 0)
        self.assertEqual(final_state["best_score"], 10.0)
        self.assertTrue(os.path.exists(final_state["best_dataset_file"]))

    def test_max_iteration_stop_without_refinement(self):
        state = self.state(score=5.0, max_iterations=0)

        final_state = self.invoke(state)

        self.assertEqual(final_state["stop_reason"], "max_iterations")
        self.assertEqual(final_state["iteration"], 0)
        self.assertEqual(final_state["best_score"], 5.0)

    def test_non_improvement_increments_stagnation_and_stops(self):
        state = self.state(score=5.0, max_iterations=3, max_stagnation=1)
        self.write_json("refined_dataset_1.json", DATASET)
        self.write_json("refined_solutions_1.json", SOLUTIONS)
        self.write_json("refined_evaluation_results_1.json", results(4.0))

        final_state = self.invoke(state)

        self.assertEqual(final_state["stop_reason"], "stagnation")
        self.assertEqual(final_state["iteration"], 1)
        self.assertEqual(final_state["stagnation_count"], 1)
        self.assertEqual(final_state["best_score"], 5.0)

    def test_improvement_resets_stagnation_and_updates_best(self):
        refined_dataset = [{"task": "Task A", "solution_criteria": "Better criteria"}]
        state = self.state(score=5.0, max_iterations=1, max_stagnation=3)
        self.write_json("refined_dataset_1.json", refined_dataset)
        self.write_json("refined_solutions_1.json", SOLUTIONS)
        self.write_json("refined_evaluation_results_1.json", results(6.0))

        final_state = self.invoke(state)

        self.assertEqual(final_state["stop_reason"], "max_iterations")
        self.assertEqual(final_state["iteration"], 1)
        self.assertEqual(final_state["stagnation_count"], 0)
        self.assertEqual(final_state["best_score"], 6.0)
        self.assertEqual(final_state["best_dataset"], refined_dataset)

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

        self.assertIsNone(update["last_error"])
        self.assertEqual(captured["dataset"], best_dataset)
        self.assertEqual(captured["token_multiplier"], 4.0)
        self.assertEqual(update["dataset"], best_dataset)

    def test_node_error_records_failed_stage(self):
        update: main.AgentState = main._node_error("refine_dataset", RuntimeError("boom"))
        finalized: main.AgentState = main.finalize(update)

        self.assertEqual(finalized["current_stage"], "refine_dataset")
        self.assertEqual(finalized["failed_stage"], "refine_dataset")
        self.assertEqual(finalized["last_error"], "RuntimeError: boom")
        self.assertEqual(finalized["stop_reason"], "error")

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

        self.assertEqual(final_state["stop_reason"], "score_threshold")
        self.assertEqual(snapshot.values["stop_reason"], "score_threshold")
        self.assertEqual(snapshot.values["best_score"], 10.0)


if __name__ == "__main__":
    unittest.main()
