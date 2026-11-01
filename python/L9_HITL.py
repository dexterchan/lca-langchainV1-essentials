"""Interactive Python version of L9_HITL.ipynb.

Run the # %% cells in order in VS Code or Spyder.
See README.md for uv setup and terminal commands.
"""

# %% [markdown]
# # Middleware: Human In The Loop
# <img src="./assets/LC_HITL.png" width="300">
#

# %% [markdown]
# ## Setup

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
- If the database is offline, ask user to try again later without further comment.
"""

# %% [markdown]
# ## Middleware

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=llm,
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
    context_schema=RuntimeContext,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"execute_sql": {"allowed_decisions": ["approve", "reject"]}},
        ),
    ],
)

# %%
from langgraph.types import Command

question = "What are the names of all the employees?"

config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config=config,
    context=RuntimeContext(db=db)
)

if "__interrupt__" in result:
    description = result['__interrupt__'][-1].value['action_requests'][-1]['description']
    print(f"\033[1;3;31m{80 * '-'}\033[0m")
    print(
        f"\033[1;3;31m Interrupt:{description}\033[0m"
    )

    result = agent.invoke(
        Command(
            resume={
                "decisions": [{"type": "reject", "message": "the database is offline."}]
            }
        ),
        config=config,  # Same thread ID to resume the paused conversation
        context=RuntimeContext(db=db),
    )
    print(f"\033[1;3;31m{80 * '-'}\033[0m")

print(result["messages"][-1].content)

# %%
config = {"configurable": {"thread_id": "2"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config=config,
    context=RuntimeContext(db=db)
)

while "__interrupt__" in result:
    description = result['__interrupt__'][-1].value['action_requests'][-1]['description']
    print(f"\033[1;3;31m{80 * '-'}\033[0m")
    print(
        f"\033[1;3;31m Interrupt:{description}\033[0m"
    )

    result = agent.invoke(
        Command(
            resume={"decisions": [{"type": "approve"}]}
        ),
        config=config,  # Same thread ID to resume the paused conversation
        context=RuntimeContext(db=db),
    )

for msg in result["messages"]:
    msg.pretty_print()

# %%
