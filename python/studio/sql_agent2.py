"""SQL agent for studio."""

import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

from sql_db import SQLDatabase

# Load the common model setup from the parent Python directory.
LESSON_DIR = Path(__file__).resolve().parents[1]
if str(LESSON_DIR) not in sys.path:
    sys.path.append(str(LESSON_DIR))

from llm import create_model  # noqa: E402 - Requires the parent directory on sys.path.

llm = create_model()

# database is from:
# url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

@tool
def execute_sql(query: str) -> str:
    """Execute a query and return results."""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


SYSTEM_PROMPT = """You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows of output unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
"""
# currently, studio lacks support for passing in runtime context

agent = create_agent(
    model=llm,
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
)

# Example:
# question = "Which genre on average has the longest tracks?"
# for step in agent.stream({"messages": [{"role":"user","content": question}]}, stream_mode="values"):
#    step["messages"][-1].pretty_print()
