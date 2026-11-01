"""Check the Lesson 1 SQL workflow with real SQLite and offline model replies."""

import tempfile
import unittest
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import L1_fast_agent_llm as lesson
from llm_workflows import SQLContext
from sql_db import SQLDatabase


class OfflineModel(BaseChatModel):
    replies: list[AIMessage]
    requests: list[list] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-fixed-workflow"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(messages)
        if not self.replies:
            raise AssertionError("The workflow made an unexpected model request.")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    def bind_tools(self, *args, **kwargs):
        raise AssertionError("A fixed workflow must not let the model select tools.")


def model(*texts):
    return OfflineModel(replies=[AIMessage(content=text) for text in texts])


class FixedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = SQLDatabase.from_uri(
            f"sqlite:///{Path(self.temp.name) / 'workflow.db'}"
        )
        self.addCleanup(self.db._engine.dispose)
        self.db.run("CREATE TABLE Artist (Name TEXT)")
        for number in range(1, 9):
            self.db.run(f"INSERT INTO Artist VALUES ('Artist {number}')")
        self.context = SQLContext(db=self.db)

    def test_sql_runs_in_fixed_order_and_limits_rows(self):
        graph = lesson.build_graph(
            model("SELECT Name FROM Artist", "The first five artists are listed.")
        )
        updates = list(graph.stream(
            {"question": "List artists."},
            context=self.context,
            stream_mode="updates",
        ))
        self.assertEqual(
            [next(iter(update)) for update in updates],
            ["prepare", "generate_sql", "validate_sql", "execute_sql", "summarize"],
        )
        self.assertEqual(
            updates[3]["execute_sql"]["rows"],
            [("Artist 1",), ("Artist 2",), ("Artist 3",), ("Artist 4",), ("Artist 5",)],
        )
        self.assertEqual(
            updates[-1]["summarize"]["answer"],
            "The first five artists are listed.",
        )

    def test_sql_rejects_writes_and_stops_after_two_attempts(self):
        llm = model("DELETE FROM Artist", "DELETE FROM Artist")
        result = lesson.build_graph(llm).invoke(
            {"question": "Delete every artist."}, context=self.context
        )
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["rows"], [])
        self.assertTrue(result["error"])
        self.assertEqual(self.db.run("SELECT COUNT(*) FROM Artist"), "[(8,)]")
        self.assertEqual(len(llm.requests), 2)

    def test_sql_can_repair_one_invalid_query(self):
        llm = model(
            "SELECT Name FROM MissingTable",
            "SELECT Name FROM Artist LIMIT 1",
            "Artist 1.",
        )
        result = lesson.build_graph(llm).invoke(
            {"question": "Name one artist."}, context=self.context
        )
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["rows"], [("Artist 1",)])
        self.assertEqual(result["answer"], "Artist 1.")
        self.assertIn("MissingTable", llm.requests[1][-1].content)

    def test_sql_does_not_execute_model_tool_requests(self):
        llm = OfflineModel(replies=[
            AIMessage(content="", tool_calls=[{
                "name": "execute_sql", "args": {"query": "DELETE FROM Artist"},
                "id": "unwanted",
            }]),
        ])
        result = lesson.build_graph(llm, max_attempts=1).invoke(
            {"question": "Delete artists."}, context=self.context
        )
        self.assertEqual(result["rows"], [])
        self.assertTrue(result["error"])
        self.assertEqual(self.db.run("SELECT COUNT(*) FROM Artist"), "[(8,)]")


if __name__ == "__main__":
    unittest.main()
