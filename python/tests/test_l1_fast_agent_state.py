"""Check source selection, clarification, and shared agents without live models."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tracers.base import BaseTracer
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command
from langsmith import tracing_context
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


def file_request(name, args, call_id="file-1"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": call_id,
    }])


def read_sources():
    return AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"file_path": f"/{name}"}, "id": f"read-{number}"}
        for number, name in enumerate(["music_history_skill.md", "lesson_notes.md"])
    ])


class OfflineTracer(BaseTracer):
    def __init__(self):
        super().__init__()
        self.runs = []

    def _persist_run(self, run):
        self.runs.append(run)


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
            db=self.db
        )
        self.config = {"configurable": {"thread_id": self.id()}}

    def test_natural_recommendation_reads_separate_prose_and_skill_files(self):
        llm = OfflineModel(replies=[
            route("answer_from_markdown"),
            read_sources(),
            AIMessage(content="Try Evening Lines for its soft piano."),
        ])
        updates = list(lesson.build_graph(llm, markdown_root=self.directory).stream(
            {"question": "Recommend an album for a quiet evening."},
            config=self.config, context=self.context, stream_mode="updates",
        ))

        self.assertEqual([next(iter(item)) for item in updates], ["classify_question", "answer_from_markdown"])
        self.assertEqual(updates[-1]["answer_from_markdown"]["report"], "Try Evening Lines for its soft piano.")
        self.assertEqual(llm.request_tools[0], [])
        self.assertIn("read_file", llm.request_tools[1])
        self.assertIn("write_file", llm.request_tools[1])
        self.assertNotIn("execute", llm.request_tools[1])
        self.assertNotIn("execute_sql", llm.request_tools[1])
        tool_text = "\n".join(item.content for item in llm.requests[-1] if item.type == "tool")
        self.assertIn("two short sentences", tool_text)
        self.assertIn("singing on the last ferry", tool_text)
        self.assertIn("quiet evening", tool_text)

    def test_exact_fact_runs_sql_without_reading_markdown(self):
        (self.directory / "lesson_notes.md").unlink()
        llm = OfflineModel(replies=[
            route("answer_from_sql"), sql_request("SELECT Name FROM Artist"),
            AIMessage(content="The artist is Artist 1."),
        ])
        updates = list(lesson.build_graph(llm, markdown_root=self.directory).stream(
            {"question": "What is the exact artist name?"},
            config=self.config, context=self.context, stream_mode="updates",
        ))

        self.assertEqual([next(iter(item)) for item in updates], ["classify_question", "answer_from_sql"])
        self.assertEqual(updates[-1]["answer_from_sql"]["report"], "The artist is Artist 1.")
        self.assertEqual(llm.request_tools, [[], ["execute_sql"], ["execute_sql"]])
        self.assertIn("Artist 1", llm.requests[-1][-1].content)

    def test_sql_agent_discovers_schema_through_its_tool(self):
        llm = OfflineModel(replies=[
            route("answer_from_sql"),
            sql_request("SELECT sql FROM sqlite_master WHERE name = 'Artist'", "schema"),
            sql_request("SELECT Name FROM Artist", "data"),
            AIMessage(content="The artist is Artist 1."),
        ])
        result = lesson.build_graph(llm, markdown_root=self.directory).invoke(
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

    def test_clarification_resumes_classifier_with_original_request_and_reply(self):
        llm = OfflineModel(replies=[
            route("clarify", "Do you want a recommendation or exact catalog facts?"),
            route("answer_from_markdown"), read_sources(), AIMessage(content="Try Evening Lines."),
        ])
        graph = lesson.build_graph(llm, markdown_root=self.directory)
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
        self.assertEqual(llm.request_tools[:2], [[], []])
        self.assertIn("read_file", llm.request_tools[-1])
        self.assertEqual(graph.get_state(self.config).next, ())

    def test_two_unsuccessful_clarification_rounds_end_without_tools(self):
        llm = OfflineModel(replies=[route("clarify") for _ in range(3)])
        graph = lesson.build_graph(llm, markdown_root=self.directory)
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
            route("answer_from_markdown"), read_sources(), AIMessage(content="Try Evening Lines."),
        ])
        graph = lesson.build_graph(llm, markdown_root=self.directory)
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
        graph = lesson.build_graph(llm, markdown_root=self.directory)
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
            AIMessage(content='{"route": "answer_from_sql", "question": 123}'),
            sql_request("DELETE FROM Artist"),
        ]:
            with self.subTest(reply=reply):
                llm = OfflineModel(replies=[reply])
                graph = lesson.build_graph(llm, markdown_root=self.directory)
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
        graph = lesson.build_graph(llm, markdown_root=self.directory)
        graph.invoke(
            {"question": "Use both sources."}, config=self.config, context=self.context
        )
        result = graph.invoke(
            Command(resume="quit"), config=self.config, context=self.context
        )
        self.assertIn("ended", result["report"].lower())
        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(graph.get_state(self.config).next, ())

    def test_missing_or_empty_markdown_returns_file_tool_errors(self):
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
                    llm = OfflineModel(replies=[
                        route("answer_from_markdown"), read_sources(),
                        AIMessage(content="Markdown error: source information is unavailable."),
                    ])
                    result = lesson.build_graph(llm, markdown_root=self.directory).invoke(
                        {"question": "Recommend an album."},
                        config=self.config, context=self.context,
                    )
                    self.assertIn("error", result["report"].lower())
                    tool_text = "\n".join(
                        item.content for item in llm.requests[-1] if item.type == "tool"
                    )
                    self.assertTrue("Error" in tool_text or "empty" in tool_text, tool_text)
                finally:
                    path.write_text(original)

    def test_markdown_tool_request_is_rejected_without_sql_execution(self):
        llm = OfflineModel(replies=[
            route("answer_from_markdown"), sql_request("DELETE FROM Artist"),
            AIMessage(content="Markdown error: SQL tools are unavailable."),
        ])
        result = lesson.build_graph(llm, markdown_root=self.directory).invoke(
            {"question": "Recommend an album."},
            config=self.config, context=self.context,
        )
        self.assertIn("error", result["report"].lower())
        self.assertTrue(all("execute_sql" not in tools for tools in llm.request_tools))
        self.assertEqual(self.db.run("SELECT COUNT(*) FROM Artist"), "[(1,)]")

    def test_shared_model_and_agent_are_initialized_once_across_threads_and_resume(self):
        second_db = SQLDatabase.from_uri(
            f"sqlite:///{self.directory / 'second-catalog.db'}"
        )
        self.addCleanup(second_db._engine.dispose)
        second_db.run("CREATE TABLE Artist (Name TEXT)")
        second_db.run("INSERT INTO Artist VALUES ('Artist 2')")
        llm = OfflineModel(replies=[
            route("clarify"), route("answer_from_sql"),
            *[
                sql_request("SELECT Name FROM Artist", f"first-{number}")
                for number in range(lesson.MAX_TOOL_CALLS)
            ],
            AIMessage(content="Artist 1."),
            route("answer_from_sql"),
            *[
                sql_request("SELECT Name FROM Artist", f"second-{number}")
                for number in range(lesson.MAX_TOOL_CALLS)
            ],
            AIMessage(content="Artist 2."),
            route("answer_from_markdown"), read_sources(), AIMessage(content="Try Evening Lines."),
        ])
        with (
            patch.object(lesson, "create_workflow_model", return_value=llm) as model_factory,
            patch.object(lesson, "create_agent", wraps=lesson.create_agent) as agent_factory,
            patch.object(lesson, "create_deep_agent", wraps=lesson.create_deep_agent) as markdown_factory,
        ):
            graph = lesson.build_graph(markdown_root=self.directory)
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
            self.assertEqual(markdown_factory.call_count, 1)
            self.assertIs(agent_factory.call_args.kwargs["tools"][0], lesson.execute_sql)

        self.assertEqual(first["report"], "Artist 1.")
        self.assertEqual(second["report"], "Artist 2.")
        self.assertEqual(third["report"], "Try Evening Lines.")
        second_tools = [
            item for item in second["messages"] if item.type == "tool"
        ]
        self.assertEqual(len(second_tools), lesson.MAX_TOOL_CALLS)
        self.assertTrue(all(item.status == "success" for item in second_tools))
        self.assertTrue(all("Artist 2" in item.content for item in second_tools))
        self.assertIn("read_file", llm.request_tools[-1])

    def test_markdown_agent_writes_and_edits_only_inside_its_folder(self):
        llm = OfflineModel(replies=[
            route("answer_from_markdown"),
            file_request("write_file", {"file_path": "/draft.md", "content": "Calm music."}),
            file_request("edit_file", {
                "file_path": "/draft.md", "old_string": "Calm", "new_string": "Quiet",
            }, "edit-1"),
            AIMessage(content="Draft saved."),
        ])
        lesson.build_graph(llm, markdown_root=self.directory).invoke(
            {"question": "Save a draft about calm music."},
            config=self.config, context=self.context,
        )
        self.assertEqual((self.directory / "draft.md").read_text(), "Quiet music.")

    def test_markdown_agent_rejects_read_write_and_edit_outside_its_folder(self):
        outside = self.directory / "outside"
        outside.mkdir()
        (outside / "private.md").write_text("private source")
        root = self.directory / "allowed"
        root.mkdir()
        (root / "escape").symlink_to(outside, target_is_directory=True)
        for file_path in ["../outside/private.md", "/escape/private.md"]:
            with self.subTest(file_path=file_path):
                llm = OfflineModel(replies=[
                    route("answer_from_markdown"),
                    file_request("read_file", {"file_path": file_path}, "read-outside"),
                    file_request("write_file", {
                        "file_path": file_path, "content": "changed",
                    }, "write-outside"),
                    file_request("edit_file", {
                        "file_path": file_path,
                        "old_string": "private", "new_string": "changed",
                    }, "edit-outside"),
                    AIMessage(content="Access was denied."),
                ])
                result = lesson.build_graph(llm, markdown_root=root).invoke(
                    {"question": "Read and change the outside file."},
                    config=self.config, context=self.context,
                )
                tools = [item for item in result["messages"] if item.type == "tool"]
                self.assertEqual(len(tools), 3)
                self.assertTrue(all("Error" in item.content for item in tools), tools)
                self.assertTrue(all("private source" not in item.content for item in tools))
                self.assertEqual((outside / "private.md").read_text(), "private source")

    def test_markdown_agent_maps_absolute_paths_under_the_virtual_root(self):
        llm = OfflineModel(replies=[
            route("answer_from_markdown"),
            file_request("write_file", {"file_path": "/nested/draft.md", "content": "Draft."}),
            file_request("read_file", {"file_path": "/nested/draft.md"}, "read-draft"),
            AIMessage(content="Draft saved."),
        ])
        result = lesson.build_graph(llm, markdown_root=self.directory).invoke(
            {"question": "Save and read a draft."},
            config=self.config, context=self.context,
        )
        self.assertEqual((self.directory / "nested/draft.md").read_text(), "Draft.")
        self.assertTrue(any(
            item.type == "tool" and "Draft." in item.content for item in result["messages"]
        ))

    def test_markdown_search_and_listing_cannot_follow_outside_symlinks(self):
        outside = self.directory / "outside"
        outside.mkdir()
        (outside / "private.md").write_text("private source")
        root = self.directory / "allowed"
        root.mkdir()
        (root / "escape").symlink_to(outside, target_is_directory=True)
        llm = OfflineModel(replies=[
            route("answer_from_markdown"),
            AIMessage(content="", tool_calls=[
                {"name": "ls", "args": {"path": "/escape"}, "id": "list-outside"},
                {"name": "glob", "args": {"pattern": "*.md", "path": "/escape"}, "id": "glob-outside"},
                {"name": "grep", "args": {"pattern": "private", "path": "/escape"}, "id": "grep-outside"},
            ]),
            AIMessage(content="Access was denied."),
        ])
        result = lesson.build_graph(llm, markdown_root=root).invoke(
            {"question": "Search outside the folder."},
            config=self.config, context=self.context,
        )
        tools = [item for item in result["messages"] if item.type == "tool"]
        self.assertEqual(len(tools), 3)
        by_call = {item.tool_call_id: item for item in tools}
        self.assertEqual(by_call["list-outside"].status, "error")
        self.assertEqual(by_call["glob-outside"].status, "error")
        self.assertEqual(by_call["grep-outside"].content, "No matches found")
        self.assertTrue(all("private source" not in item.content for item in tools))

    def test_markdown_tool_budget_resets_for_each_request(self):
        calls = [{
            "name": "read_file", "args": {"file_path": "/lesson_notes.md"},
            "id": f"read-{number}",
        } for number in range(7)]
        llm = OfflineModel(replies=[
            reply
            for _ in range(2)
            for reply in [
                route("answer_from_markdown"),
                AIMessage(content="", tool_calls=calls),
                AIMessage(content="Only six file calls were executed."),
            ]
        ])
        graph = lesson.build_graph(llm, markdown_root=self.directory)
        for number in range(2):
            result = graph.invoke(
                {"question": "Read the history."},
                config={"configurable": {"thread_id": f"markdown-budget-{number}"}},
                context=self.context,
            )
            tools = [item for item in result["messages"] if item.type == "tool"]
            self.assertEqual(sum(item.status == "success" for item in tools), 6)
            self.assertEqual(sum(item.status == "error" for item in tools), 1)

    def test_large_markdown_question_does_not_create_hidden_history_files(self):
        llm = OfflineModel(replies=[AIMessage(content="Source information is unavailable.")])
        agent = lesson.create_markdown_agent(llm, markdown_root=self.directory)
        before = set(self.directory.rglob("*"))
        agent.invoke({"messages": [HumanMessage(content="music " * 35000)]})
        self.assertEqual(set(self.directory.rglob("*")), before)
        self.assertEqual(len(llm.requests), 1)

    def test_markdown_model_budget_includes_all_calls_with_large_file_results(self):
        (self.directory / "lesson_notes.md").write_text("music " * 12400)
        llm = OfflineModel(
            profile={"max_input_tokens": 30000},
            replies=[
                file_request("read_file", {"file_path": "/lesson_notes.md"}, f"read-{number}")
                for number in range(4)
            ],
        )
        agent = lesson.create_markdown_agent(llm, markdown_root=self.directory)
        before = set(self.directory.rglob("*"))
        result = agent.invoke({"messages": [HumanMessage(content="Read the music history.")]})
        self.assertEqual(len(llm.requests), 4)
        self.assertIn("limit", result["messages"][-1].text.lower())
        self.assertEqual(set(self.directory.rglob("*")), before)

    def test_markdown_agent_cannot_run_shell_commands_or_delegate(self):
        for name, args in [
            ("execute", {"command": "touch /must-not-exist"}),
            ("task", {"subagent_type": "general-purpose", "description": "Write an outside file."}),
        ]:
            with self.subTest(name=name):
                llm = OfflineModel(replies=[
                    route("answer_from_markdown"), file_request(name, args),
                    AIMessage(content="The tool is unavailable."),
                ])
                result = lesson.build_graph(llm, markdown_root=self.directory).invoke(
                    {"question": "Run a command."},
                    config=self.config, context=self.context,
                )
                tools = [item for item in result["messages"] if item.type == "tool"]
                self.assertEqual(len(tools), 1)
                self.assertEqual(tools[0].status, "error")
                self.assertEqual(len(llm.requests), 3)

    def test_run_question_registers_tracing_with_the_requested_run_id(self):
        run_id = uuid4()
        tracer = OfflineTracer()
        llm = OfflineModel(replies=[
            route("answer_from_markdown"), read_sources(), AIMessage(content="Try Evening Lines."),
        ])
        graph = lesson.build_graph(llm, markdown_root=self.directory)
        with (
            tracing_context(enabled=False),
            patch.object(lesson, "tracing_is_enabled", return_value=True, create=True),
            patch.object(lesson, "LangChainTracer", return_value=tracer, create=True) as tracer_factory,
            patch("builtins.print"),
        ):
            result = lesson.run_question(graph, "Recommend calm music.", self.context, run_id=run_id)
        self.assertEqual(result["report"], "Try Evening Lines.")
        tracer_factory.assert_called_once_with()
        self.assertEqual(len(tracer.runs), 1)
        root = tracer.runs[0]
        self.assertEqual(root.id, run_id)
        self.assertEqual(root.name, "music_question")
        self.assertEqual(root.extra["metadata"]["question_run_id"], str(run_id))
        children = list(root.child_runs)
        traced_names = []
        while children:
            child = children.pop()
            self.assertEqual(child.trace_id, run_id)
            traced_names.append(child.name)
            children.extend(child.child_runs)
        self.assertIn("read_file", traced_names)
        self.assertIn("markdown_agent", traced_names)

    def test_clarification_resumes_with_a_new_run_id_and_the_same_thread(self):
        initial_run_id = uuid4()
        tracer = OfflineTracer()
        llm = OfflineModel(replies=[
            route("clarify"), route("answer_from_sql"),
            sql_request("SELECT Name FROM Artist"), AIMessage(content="Artist 1."),
        ])
        graph = lesson.build_graph(llm, markdown_root=self.directory)
        with (
            tracing_context(enabled=False),
            patch.object(lesson, "tracing_is_enabled", return_value=True, create=True),
            patch.object(lesson, "LangChainTracer", return_value=tracer, create=True),
            patch("builtins.input", return_value="Only exact catalog facts."),
            patch("builtins.print"),
        ):
            result = lesson.run_question(graph, "Use both sources.", self.context, run_id=initial_run_id)
        self.assertEqual(result["report"], "Artist 1.")
        self.assertEqual(len(tracer.runs), 2)
        initial, resumed = tracer.runs
        self.assertEqual(initial.id, initial_run_id)
        self.assertNotEqual(resumed.id, initial.id)
        self.assertEqual(
            initial.extra["metadata"]["thread_id"],
            resumed.extra["metadata"]["thread_id"],
        )
        self.assertEqual(resumed.extra["metadata"]["question_run_id"], str(initial_run_id))
        children = list(resumed.child_runs)
        traced_names = []
        while children:
            child = children.pop()
            self.assertEqual(child.trace_id, resumed.id)
            traced_names.append(child.name)
            children.extend(child.child_runs)
        self.assertIn("execute_sql", traced_names)

    def test_run_question_does_not_register_a_tracer_when_tracing_is_disabled(self):
        llm = OfflineModel(replies=[
            route("answer_from_markdown"), read_sources(), AIMessage(content="Try Evening Lines."),
        ])
        graph = lesson.build_graph(llm, markdown_root=self.directory)
        with (
            tracing_context(enabled=False),
            patch.object(lesson, "tracing_is_enabled", return_value=False, create=True),
            patch.object(lesson, "LangChainTracer", create=True) as tracer_factory,
            patch("builtins.print"),
        ):
            result = lesson.run_question(graph, "Recommend calm music.", self.context)
        self.assertEqual(result["report"], "Try Evening Lines.")
        tracer_factory.assert_not_called()

    def test_sql_writes_are_rejected(self):
        llm = OfflineModel(replies=[
            route("answer_from_sql"), sql_request("DELETE FROM Artist"),
            AIMessage(content="The write was rejected."),
        ])
        result = lesson.build_graph(llm, markdown_root=self.directory).invoke(
            {"question": "Delete the catalog."},
            config=self.config, context=self.context,
        )
        self.assertEqual(self.db.run("SELECT COUNT(*) FROM Artist"), "[(1,)]")
        self.assertIn("Error:", llm.requests[-1][-1].content)
        self.assertIn("rejected", result["report"])

    def test_sql_model_and_tool_limits_remain_per_request(self):
        llm = OfflineModel(replies=[
            route("answer_from_sql"),
            *[
                sql_request("SELECT Name FROM Artist", f"sql-{number}")
                for number in range(lesson.MAX_MODEL_CALLS + 1)
            ],
        ])
        result = lesson.build_graph(llm, markdown_root=self.directory).invoke(
            {"question": "List the artists."},
            config=self.config, context=self.context,
        )
        tools = [item for item in result["messages"] if item.type == "tool"]
        self.assertEqual(len(llm.requests), 1 + lesson.MAX_MODEL_CALLS)
        self.assertEqual(sum(item.status == "success" for item in tools), lesson.MAX_TOOL_CALLS)
        self.assertEqual(
            sum(item.status == "error" for item in tools),
            lesson.MAX_MODEL_CALLS - lesson.MAX_TOOL_CALLS,
        )
        self.assertIn("limit", result["report"].lower())

    def test_sql_tool_batch_cannot_exceed_its_execution_limit(self):
        calls = [{
            "name": "execute_sql", "args": {"query": "SELECT Name FROM Artist"},
            "id": f"sql-{number}",
        } for number in range(lesson.MAX_TOOL_CALLS + 2)]
        llm = OfflineModel(replies=[
            route("answer_from_sql"), AIMessage(content="", tool_calls=calls),
            AIMessage(content="Only the permitted requests were executed."),
        ])
        result = lesson.build_graph(llm, markdown_root=self.directory).invoke(
            {"question": "List the artists."},
            config=self.config, context=self.context,
        )
        tools = [item for item in result["messages"] if item.type == "tool"]
        self.assertEqual(sum(item.status == "success" for item in tools), lesson.MAX_TOOL_CALLS)
        self.assertEqual(sum(item.status == "error" for item in tools), 2)

    def test_fictional_studio_catalog_returns_matching_album_facts(self):
        catalog_path = Path(lesson.__file__).parent / "studio/music_catalog.db"
        self.assertTrue(catalog_path.is_file(), "The fictional catalog is missing.")
        db = SQLDatabase.from_uri(f"sqlite:///{catalog_path}")
        self.addCleanup(db._engine.dispose)
        llm = OfflineModel(replies=[
            route("answer_from_sql"),
            sql_request(
                "SELECT Album.Title, Artist.Name, Album.PriceUSD FROM Album "
                "JOIN Artist ON Album.ArtistId = Artist.ArtistId "
                "WHERE Album.Title = 'Evening Lines'"
            ),
            AIMessage(content="Evening Lines is by Mira Vale and costs USD 10.00."),
        ])
        result = lesson.build_graph(llm, markdown_root=self.directory).invoke(
            {"question": "What is the price and artist of Evening Lines?"},
            config=self.config, context=lesson.RuntimeContext(db=db),
        )
        self.assertIn("Mira Vale", result["report"])
        tool_result = json.loads(llm.requests[-1][-1].content)
        self.assertEqual(tool_result["rows"], [["Evening Lines", "Mira Vale", 10.0]])


if __name__ == "__main__":
    unittest.main()
