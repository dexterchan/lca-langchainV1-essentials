"""Provide small model tasks and code-enforced controls for fixed workflows."""

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from sql_db import SQLDatabase

PUBLIC_TABLES = {"Album", "Artist", "Genre", "Playlist", "PlaylistTrack", "Track"}
MAX_ROWS = 5


def plain_content(message):
    """Accept text output and reject model requests to execute tools."""
    if message.tool_calls:
        raise ValueError("The model must return data, not tool requests.")
    if not isinstance(message.content, str) or not message.content.strip():
        raise ValueError("The model returned no text.")
    return message.content.strip()


def unfence(text):
    """Remove one surrounding Markdown code fence."""
    match = re.fullmatch(r"```(?:sql|json)?\s*\n(.*?)\n```", text.strip(), re.DOTALL)
    return match.group(1).strip() if match else text.strip()


@dataclass
class SQLContext:
    db: SQLDatabase
    is_employee: bool = True


class SQLState(TypedDict, total=False):
    question: str
    schema: str
    sql: str
    rows: list
    columns: list[str]
    answer: str
    error: str
    attempts: int
    approval: str
    history: Annotated[list[BaseMessage], add_messages]


@contextmanager
def read_connection(db, allowed_tables=None):
    """Open SQLite in read-only mode and enforce the permitted operations."""
    path = Path(db._engine.url.database).resolve()
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    allowed = (
        {name.casefold() for name in allowed_tables}
        if allowed_tables is not None else None
    )

    def authorize(action, name, detail, database, source):
        if action == sqlite3.SQLITE_READ:
            if allowed is not None and name.casefold() not in allowed:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            return (
                sqlite3.SQLITE_DENY
                if (detail or "").casefold() == "load_extension"
                else sqlite3.SQLITE_OK
            )
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    ticks = 0

    def stop_long_query():
        nonlocal ticks
        ticks += 1
        return int(ticks > 100)

    connection.set_authorizer(authorize)
    connection.set_progress_handler(stop_long_query, 1000)
    try:
        yield connection
    finally:
        connection.close()


def validate_query(db, query, allowed_tables=None):
    """Check one bounded read-only query before the execution node runs."""
    if len(query) > 4096 or not re.match(r"^\s*(SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError("Return one SELECT or read-only WITH query of at most 4096 characters.")
    with read_connection(db, allowed_tables) as connection:
        connection.execute("EXPLAIN " + query).fetchall()


def read_query(db, query, allowed_tables=None):
    """Execute a checked read-only query and return at most five rows."""
    validate_query(db, query, allowed_tables)
    with read_connection(db, allowed_tables) as connection:
        cursor = connection.execute(query)
        columns = [column[0] for column in cursor.description]
        rows = [
            tuple(value[:300] if isinstance(value, str) else value for value in row)
            for row in cursor.fetchmany(MAX_ROWS)
        ]
    return columns, rows


class SQLNodes:
    """Implement SQL tasks while the lesson file defines every graph edge."""

    def __init__(self, model, *, max_attempts=2, restrict_tables=False):
        """Set the model, attempt limit, and optional public table policy."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self.model = model
        self.max_attempts = max_attempts
        self.restrict_tables = restrict_tables

    def allowed_tables(self, context):
        """Choose table access from trusted runtime context."""
        return PUBLIC_TABLES if self.restrict_tables and not context.is_employee else None

    def prepare(self, state: SQLState, runtime: Runtime[SQLContext]):
        """Load the permitted schema and reset fields for this question."""
        # Schema discovery is a fixed code operation, not a model-selected tool.
        with read_connection(runtime.context.db) as connection:
            tables = connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        allowed = self.allowed_tables(runtime.context)
        schema = "\n".join(
            definition for name, definition in tables
            if definition and (allowed is None or name in allowed)
        )
        updates = {
            "schema": schema, "sql": "", "rows": [], "columns": [],
            "answer": "", "error": "", "attempts": 0, "approval": "",
        }
        if allowed is not None:
            # Public turns must not inherit private data from a saved conversation.
            updates["history"] = Overwrite([])
        return updates

    def generate_sql(self, state: SQLState):
        """Ask the model for one SQL query, never for a workflow decision."""
        instructions = (
            "Return exactly one SQLite SELECT or read-only WITH query. "
            "Return SQL only. Do not request tools or choose workflow steps. "
            "Use only the supplied schema. Prefer explicit columns and LIMIT 5."
        )
        request = f"Schema:\n{state['schema']}\n\nQuestion: {state['question']}"
        if state["error"]:
            request += f"\nThe previous query failed. Correct this error: {state['error']}"
        response = self.model.invoke([
            SystemMessage(content=instructions),
            *state.get("history", [])[-6:],
            HumanMessage(content=request),
        ])
        try:
            query, error = unfence(plain_content(response)), ""
        except ValueError as exception:
            query, error = "", str(exception)
        return {"sql": query, "error": error, "attempts": state["attempts"] + 1}

    def validate_sql(self, state: SQLState, runtime: Runtime[SQLContext]):
        """Reject invalid syntax, writes, and unauthorized table access."""
        if state["error"]:
            return {"error": state["error"]}
        try:
            validate_query(
                runtime.context.db, state["sql"],
                self.allowed_tables(runtime.context),
            )
            return {"error": ""}
        except (ValueError, sqlite3.Error) as exception:
            return {"error": str(exception)}

    def execute_sql(self, state: SQLState, runtime: Runtime[SQLContext]):
        """Run the validated query with database-level read-only controls."""
        try:
            columns, rows = read_query(
                runtime.context.db, state["sql"],
                self.allowed_tables(runtime.context),
            )
            return {"columns": columns, "rows": rows, "error": ""}
        except (ValueError, sqlite3.Error) as exception:
            return {"columns": [], "rows": [], "error": str(exception)}

    def route(self, state: SQLState):
        """Choose success, retry, or stop using code and an attempt counter."""
        if not state["error"]:
            return "ready"
        return "retry" if state["attempts"] < self.max_attempts else "stop"

    def summarize(self, state: SQLState):
        """Ask the model to describe only the returned database rows."""
        response = self.model.invoke([
            SystemMessage(content=(
                "Answer the question using only the supplied SQL results. "
                "Results may be truncated to five rows. Do not call a partial list complete. "
                "Do not request tools or propose workflow steps."
            )),
            HumanMessage(content=json.dumps({
                "question": state["question"], "sql": state["sql"],
                "columns": state["columns"], "rows": state["rows"], "row_limit": MAX_ROWS,
            }, default=str)),
        ])
        try:
            answer, error = plain_content(response), ""
        except ValueError as exception:
            answer, error = "The model did not return a valid summary.", str(exception)
        history_answer = (
            f"{answer}\nSQL used: {state['sql']}\n"
            f"Rows: {json.dumps(state['rows'], default=str)}"
        )
        return {
            "answer": answer, "error": error,
            "history": [
                HumanMessage(content=state["question"]),
                AIMessage(content=history_answer),
            ],
        }

    def stop(self, state: SQLState):
        """Return a controlled failure without another model call."""
        return {"answer": f"The workflow stopped: {state['error']}"}
