"""Interactive Python version of L8_dynamic.ipynb.

Run the # %% cells in order in VS Code or Spyder.
See README.md for uv setup and terminal commands.
"""

# %% [markdown]
# # Dynamic Prompt
# <img src="./assets/LC_DynamicPrompts.png" width="500">
#
# ## Setup

# %% [markdown]
# Load and/or check for needed environmental variables

# %%
import sys
from pathlib import Path

# Find the lesson files from the script or the interactive working folder.
LESSON_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (LESSON_DIR / "env_utils.py").is_file():
    LESSON_DIR = LESSON_DIR / "python"
if str(LESSON_DIR) not in sys.path:
    sys.path.insert(0, str(LESSON_DIR))

from llm import create_model
from env_utils import doublecheck_env

llm = create_model()

# Check and print results
doublecheck_env(str(LESSON_DIR / "example.env"))

# %%
from sql_db import SQLDatabase

db = SQLDatabase.from_uri(f"sqlite:///{(LESSON_DIR / 'Chinook.db').as_posix()}")

# %%
from dataclasses import dataclass


@dataclass
class RuntimeContext:
    is_employee: bool
    db: SQLDatabase

# %%
from langchain_core.tools import tool
from langgraph.runtime import get_runtime

@tool
def execute_sql(query: str) -> str:
    """Execute a SQLite command and return results."""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

# %%
SYSTEM_PROMPT_TEMPLATE = """You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows unless the user explicitly asks otherwise.
{table_limits}
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
"""

# %% [markdown]
# ## Build a Dynamic Prompt
# Utilize runtime context and middleware to generate a dynamic prompt.

# %%
from langchain.agents.middleware.types import ModelRequest, dynamic_prompt


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    if not request.runtime.context.is_employee:
        table_limits = "- Limit access to these tables: Album, Artist, Genre, Playlist, PlaylistTrack, Track."
    else:
        table_limits = ""

    return SYSTEM_PROMPT_TEMPLATE.format(table_limits=table_limits)

# %% [markdown]
# Include middleware in `create_agent`.

# %%
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[execute_sql],
    middleware=[dynamic_system_prompt],
    context_schema=RuntimeContext,
)

# %%
question = "What is the most costly purchase by Frank Harris?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    context=RuntimeContext(is_employee=False, db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %%
question = "What is the most costly purchase by Frank Harris?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    context=RuntimeContext(is_employee=True, db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %%
