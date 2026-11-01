"""Check source selection, clarification, and shared agents without live models."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command
from pydantic import Field

import L1_fast_agent_state as lesson
from sql_db import SQLDatabase


class OfflineModel(BaseChatModel):
    replies: list[AIMessage]
    requests: list[list] = Field(default_factory=list)
    request_tools: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-music-workflow"

    def bind_tools(self, tools, **kwargs):
        return self.bind(
            tools=[convert_to_openai_tool(item) for item in tools], **kwargs
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(messages)
        self.request_tools.append([
            item["function"]["name"] for item in kwargs.get("tools", [])
        ])
        if not self.replies:
            raise AssertionError("The graph made an unexpected model request.")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def route(destination, question="Which single source should I use?"):
    return AIMessage(content=json.dumps({
        "route": destination, "question": question,
    }))


def sql_request(query, call_id="sql-1"):
    return AIMessage(content="", tool_calls=[{
        "name": "execute_sql", "args": {"query": query}, "id": call_id,
    }])


class MusicWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.directory / 'catalog.db'}")
        self.addCleanup(self.db._engine.dispose)
        self.db.run("CREATE TABLE Artist (Name TEXT)")
        self.db.run("INSERT INTO Artist VALUES ('Artist 1')")
        (self.directory / "lesson_notes.md").write_text(
            "# A fictional singer\n\n"
            "Artist 1 recorded Evening Lines after singing on the last ferry.\n"
            "Its soft piano makes the album suitable for a quiet evening.\n"
        )
        (self.directory / "music_history_skill.md").write_text(
            "Explain a recommendation in two short sentences.\n"
            "Use only the supplied history. Say when information is missing.\n"
        )
        self.context = lesson.RuntimeContext(
            db=self.db, markdown_root=self.directory
        )
        self.config = {"configurable": {"thread_id": self.id()}}

    def test_natural_recommendation_reads_separate_prose_and_skill_files(self):
        llm = OfflineModel(replies=[
            route("b"),
            AIMessage(content="Try Evening Lines for its soft piano."),
        ])
        updates = list(lesson.build_graph(llm).stream(
            {"question": "Recommend an album for a quiet evening."},
            config=self.config, context=self.context, stream_mode="updates",
        ))

        self.assertEqual([next(iter(item)) for item in updates], ["a", "b"])
        self.assertEqual(updates[-1]["b"]["report"], "Try Evening Lines for its soft piano.")
        self.assertEqual(llm.request_tools, [[], []])
        self.assertIn("two short sentences", llm.requests[1][0].content)
        self.assertIn("singing on the last ferry", llm.requests[1][1].content)
        self.assertIn("quiet evening", llm.requests[1][1].content)

    def test_exact_fact_runs_sql_without_reading_markdown(self):
        (self.directory / "lesson_notes.md").unlink()
        llm = OfflineModel(replies=[
            route("c"), sql_request("SELECT Name FROM Artist"),
            AIMessage(content="The artist is Artist 1."),
        ])
        updates = list(lesson.build_graph(llm).stream(
            {"question": "What is the exact artist name?"},
            config=self.config, context=self.context, stream_mode="updates",
        ))

        self.assertEqual([next(iter(item)) for item in updates], ["a", "c"])
        self.assertEqual(updates[-1]["c"]["report"], "The artist is Artist 1.")
        self.assertEqual(llm.request_tools, [[], ["execute_sql"], ["execute_sql"]])
        self.assertIn("Artist 1", llm.requests[-1][-1].content)

    def test_sql_agent_discovers_schema_through_its_tool(self):
        llm = OfflineModel(replies=[
            route("c"),
            sql_request("SELECT sql FROM sqlite_master WHERE name = 'Artist'", "schema"),
            sql_request("SELECT Name FROM Artist", "data"),
            AIMessage(content="The artist is Artist 1."),
        ])
        result = lesson.build_graph(llm).invoke(
            {"question": "What artists are in the catalog?"},
            config=self.config, context=self.context,
        )

        self.assertEqual(result["report"], "The artist is Artist 1.")
        self.assertTrue(any(
            item.type == "tool" and "CREATE TABLE Artist" in item.content
            for item in llm.requests[2]
        ))
        self.assertTrue(any(
            item.type == "tool" and "Artist 1" in item.content
            for item in llm.requests[3]
        ))

    def test_clarification_resumes_a_with_original_request_and_reply(self):
        llm = OfflineModel(replies=[
            route("clarify", "Do you want a recommendation or exact catalog facts?"),
            route("b"), AIMessage(content="Try Evening Lines."),
        ])
        graph = lesson.build_graph(llm)
        initial = graph.invoke(
            {"question": "Recommend music and give exact track lengths."},
            config=self.config, context=self.context,
        )

        self.assertEqual(len(llm.requests), 1)
        self.assertIn("recommendation", initial["__interrupt__"][0].value["question"])
        self.assertEqual(graph.get_state(self.config).next, ("clarify",))
        result = graph.invoke(
            Command(resume="Only recommend something calm."),
            config=self.config, context=self.context,
        )

        self.assertEqual(result["report"], "Try Evening Lines.")
        self.assertEqual(result["clarification_rounds"], 1)
        self.assertIn("exact track lengths", llm.requests[1][-1].content)
        self.assertIn("Only recommend something calm.", llm.requests[1][-1].content)
        self.assertEqual(llm.request_tools, [[], [], []])
        self.assertEqual(graph.get_state(self.config).next, ())

    def test_two_unsuccessful_clarification_rounds_end_without_tools(self):
        llm = OfflineModel(replies=[route("clarify") for _ in range(3)])
        graph = lesson.build_graph(llm)
        result = graph.invoke(
            {"question": "Use both sources."}, config=self.config, context=self.context
        )
        self.assertEqual(result["__interrupt__"][0].value["round"], 1)
        result = graph.invoke(
            Command(resume="I still want both."),
            config=self.config, context=self.context,
        )
        self.assertEqual(result["__interrupt__"][0].value["round"], 2)
        result = graph.invoke(
            Command(resume="Use both anyway."),
            config=self.config, context=self.context,
        )

        self.assertNotIn("__interrupt__", result)
        self.assertIn("unable", result["report"].lower())
        self.assertEqual(result["clarification_rounds"], 2)
        self.assertEqual(len(llm.requests), 3)
        self.assertEqual(llm.request_tools, [[], [], []])
        self.assertEqual(graph.get_state(self.config).next, ())

    def test_clear_second_reply_can_still_reach_a_source(self):
        llm = OfflineModel(replies=[
            route("clarify"), route("clarify"),
            route("b"), AIMessage(content="Try Evening Lines."),
        ])
        graph = lesson.build_graph(llm)
        graph.invoke(
            {"question": "Use both sources."}, config=self.config, context=self.context
        )
        graph.invoke(
            Command(resume="Either one."),
            config=self.config, context=self.context,
        )
        result = graph.invoke(
            Command(resume="Just recommend an album."),
            config=self.config, context=self.context,
        )
        self.assertEqual(result["report"], "Try Evening Lines.")
        self.assertEqual(result["clarification_rounds"], 2)
        self.assertEqual(graph.get_state(self.config).next, ())

    def test_blank_replies_consume_clarification_rounds_without_ending_early(self):
        llm = OfflineModel(replies=[route("clarify") for _ in range(3)])
        graph = lesson.build_graph(llm)
        graph.invoke(
            {"question": "Use both sources."}, config=self.config, context=self.context
        )
        result = graph.invoke(
            Command(resume="  "), config=self.config, context=self.context
        )
        self.assertIn("__interrupt__", result)
        self.assertEqual(result["__interrupt__"][0].value["round"], 2)
        result = graph.invoke(
            Command(resume=""), config=self.config, context=self.context
        )
        self.assertNotIn("__interrupt__", result)
        self.assertIn("unable", result["report"].lower())
        self.assertEqual(result["clarification_rounds"], 2)
        self.assertEqual(llm.request_tools, [[], [], []])

    def test_bad_classifier_output_cannot_select_or_execute_tools(self):
        for reply in [
            AIMessage(content="not JSON"),
            route("delete_database"),
            AIMessage(content='{"route": "c", "question": 123}'),
            sql_request("DELETE FROM Artist"),
        ]:
            with self.subTest(reply=reply):
                llm = OfflineModel(replies=[reply])
                graph = lesson.build_graph(llm)
                result = graph.invoke(
                    {"question": "What should I hear?"},
                    config=self.config, context=self.context,
                )
                self.assertIn("__interrupt__", result)
                result = graph.invoke(
                    Command(resume="quit"), config=self.config, context=self.context
                )
                self.assertNotIn("__interrupt__", result)
                self.assertEqual(len(llm.requests), 1)
                self.assertEqual(llm.request_tools, [[]])
                self.assertEqual(self.db.run("SELECT COUNT(*) FROM Artist"), "[(1,)]")

    def test_quit_during_clarification_goes_directly_to_end(self):
        llm = OfflineModel(replies=[route("clarify")])
        graph = lesson.build_graph(llm)
        graph.invoke(
            {"question": "Use both sources."}, config=self.config, context=self.context
        )
        result = graph.invoke(
            Command(resume="quit"), config=self.config, context=self.context
        )
        self.assertIn("ended", result["report"].lower())
        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(graph.get_state(self.config).next, ())

    def test_missing_or_empty_markdown_stops_before_answer_generation(self):
        for name, contents in [
            ("lesson_notes.md", None), ("lesson_notes.md", "  "),
            ("music_history_skill.md", None), ("music_history_skill.md", "  "),
        ]:
            with self.subTest(name=name, contents=contents):
                path = self.directory / name
                original = path.read_text()
                if contents is None:
                    path.unlink()
                else:
                    path.write_text(contents)
                try:
                    llm = OfflineModel(replies=[route("b")])
                    result = lesson.build_graph(llm).invoke(
                        {"question": "Recommend an album."},
                        config=self.config, context=self.context,
                    )
                    self.assertIn("error", result["report"].lower())
                    self.assertEqual(len(llm.requests), 1)
                finally:
                    path.write_text(original)

    def test_markdown_tool_request_is_rejected_without_sql_execution(self):
        llm = OfflineModel(replies=[route("b"), sql_request("DELETE FROM Artist")])
        result = lesson.build_graph(llm).invoke(
            {"question": "Recommend an album."},
            config=self.config, context=self.context,
        )
        self.assertIn("error", result["report"].lower())
        self.assertEqual(llm.request_tools, [[], []])
        self.assertEqual(self.db.run("SELECT COUNT(*) FROM Artist"), "[(1,)]")

    def test_shared_model_and_agent_are_initialized_once_across_threads_and_resume(self):
        second_db = SQLDatabase.from_uri(
            f"sqlite:///{self.directory / 'second-catalog.db'}"
        )
        self.addCleanup(second_db._engine.dispose)
        second_db.run("CREATE TABLE Artist (Name TEXT)")
        second_db.run("INSERT INTO Artist VALUES ('Artist 2')")
        llm = OfflineModel(replies=[
            route("clarify"), route("c"),
            sql_request("SELECT Name FROM Artist", "first-1"),
            sql_request("SELECT Name FROM Artist", "second-1"),
            AIMessage(content="Artist 1."),
            route("c"),
            sql_request("SELECT Name FROM Artist", "first-2"),
            sql_request("SELECT Name FROM Artist", "second-2"),
            AIMessage(content="Artist 2."),
            route("b"), AIMessage(content="Try Evening Lines."),
        ])
        with (
            patch.object(lesson, "create_workflow_model", return_value=llm) as model_factory,
            patch.object(lesson, "create_agent", wraps=lesson.create_agent) as agent_factory,
        ):
            graph = lesson.build_graph()
            graph.invoke(
                {"question": "Both sources."}, config=self.config, context=self.context
            )
            first = graph.invoke(
                Command(resume="Just give the exact artist name."),
                config=self.config, context=self.context,
            )
            second = graph.invoke(
                {"question": "What is the exact artist name?"},
                config={"configurable": {"thread_id": "second-request"}},
                context=lesson.RuntimeContext(db=second_db),
            )
            third = graph.invoke(
                {"question": "Recommend an album."},
                config={"configurable": {"thread_id": "third-request"}},
                context=self.context,
            )
            self.assertEqual(model_factory.call_count, 1)
            self.assertEqual(agent_factory.call_count, 1)
            self.assertIs(agent_factory.call_args.kwargs["tools"][0], lesson.execute_sql)

        self.assertEqual(first["report"], "Artist 1.")
        self.assertEqual(second["report"], "Artist 2.")
        self.assertEqual(third["report"], "Try Evening Lines.")
        second_tools = [
            item for item in second["messages"] if item.type == "tool"
        ]
        self.assertEqual(len(second_tools), 2)
        self.assertTrue(all(item.status == "success" for item in second_tools))
        self.assertTrue(all("Artist 2" in item.content for item in second_tools))
        self.assertEqual(llm.request_tools[-2:], [[], []])

    def test_sql_writes_are_rejected(self):
        llm = OfflineModel(replies=[
            route("c"), sql_request("DELETE FROM Artist"),
            AIMessage(content="The write was rejected."),
        ])
        result = lesson.build_graph(llm).invoke(
            {"question": "Delete the catalog."},
            config=self.config, context=self.context,
        )
        self.assertEqual(self.db.run("SELECT COUNT(*) FROM Artist"), "[(1,)]")
        self.assertIn("Error:", llm.requests[-1][-1].content)
        self.assertIn("rejected", result["report"])

    def test_sql_model_and_tool_limits_remain_per_request(self):
        llm = OfflineModel(replies=[
            route("c"),
            *[sql_request("SELECT Name FROM Artist", f"sql-{number}") for number in range(4)],
        ])
        result = lesson.build_graph(llm).invoke(
            {"question": "List the artists."},
            config=self.config, context=self.context,
        )
        tools = [item for item in result["messages"] if item.type == "tool"]
        self.assertEqual(len(llm.requests), 4)  # One classifier call and three agent calls.
        self.assertEqual(sum(item.status == "success" for item in tools), 2)
        self.assertEqual(sum(item.status == "error" for item in tools), 1)
        self.assertIn("limit", result["report"].lower())

    def test_sql_tool_batch_cannot_exceed_its_execution_limit(self):
        calls = [{
            "name": "execute_sql", "args": {"query": "SELECT Name FROM Artist"},
            "id": f"sql-{number}",
        } for number in range(4)]
        llm = OfflineModel(replies=[
            route("c"), AIMessage(content="", tool_calls=calls),
            AIMessage(content="Only two requests were executed."),
        ])
        result = lesson.build_graph(llm).invoke(
            {"question": "List the artists."},
            config=self.config, context=self.context,
        )
        tools = [item for item in result["messages"] if item.type == "tool"]
        self.assertEqual(sum(item.status == "success" for item in tools), 2)
        self.assertEqual(sum(item.status == "error" for item in tools), 2)

    def test_fictional_studio_catalog_returns_matching_album_facts(self):
        catalog_path = Path(lesson.__file__).parent / "studio/music_catalog.db"
        self.assertTrue(catalog_path.is_file(), "The fictional catalog is missing.")
        db = SQLDatabase.from_uri(f"sqlite:///{catalog_path}")
        self.addCleanup(db._engine.dispose)
        llm = OfflineModel(replies=[
            route("c"),
            sql_request(
                "SELECT Album.Title, Artist.Name, Album.PriceUSD FROM Album "
                "JOIN Artist ON Album.ArtistId = Artist.ArtistId "
                "WHERE Album.Title = 'Evening Lines'"
            ),
            AIMessage(content="Evening Lines is by Mira Vale and costs USD 10.00."),
        ])
        result = lesson.build_graph(llm).invoke(
            {"question": "What is the price and artist of Evening Lines?"},
            config=self.config, context=lesson.RuntimeContext(db=db),
        )
        self.assertIn("Mira Vale", result["report"])
        tool_result = json.loads(llm.requests[-1][-1].content)
        self.assertEqual(tool_result["rows"], [["Evening Lines", "Mira Vale", 10.0]])


if __name__ == "__main__":
    unittest.main()
