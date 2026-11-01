"""Interactive Python version of L6_memory.ipynb.

Run the # %% cells in order in VS Code or Spyder.
See README.md for uv setup and terminal commands.
"""

# %% [markdown]
# # Memory
#
# <div style="display: flex; justify-content: flex-start; gap: 10px;">
#   <img src="./assets/LC_Memory_before.png" style="width:500px; border:1px solid #ccc; border-radius:6px;">
# </div>
#
# Persisting messages, or 'agent state' between invocations of the agent.

# %% [markdown]
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
SYSTEM_PROMPT = """You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
"""

# %%
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=RuntimeContext,
)

# %% [markdown]
# ## Repeated Queries

# %%
question = "This is Frank Harris, What was the total on my last invoice?"
steps = []

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
    context=RuntimeContext(db=db),
):
    step["messages"][-1].pretty_print()
    steps.append(step)

# %%
question = "What were the titles?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %% [markdown]
# ## Add memory

# %%
from langgraph.checkpoint.memory import InMemorySaver

# %%
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

agent = create_agent(
    model=llm,
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=RuntimeContext,
    checkpointer=InMemorySaver(),
)

# %%
question = "This is Frank Harris, What was the total on my last invoice?"
steps = []

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    {"configurable": {"thread_id": "1"}},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
    steps.append(step)

# %%
question = "What were the titles?"
steps = []

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    {"configurable": {"thread_id": "1"}},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
    steps.append(step)

# %% [markdown]
# ## Try your own queries
# Now that there is memory, check the agents recall!

# %%
question = "Your Question Here?"
steps = []

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    {"configurable": {"thread_id": "1"}},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
    steps.append(step)

# %%

# %%
