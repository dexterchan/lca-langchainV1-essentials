"""Interactive Python version of L1_fast_agent.ipynb.

Run the # %% cells in order in VS Code or Spyder.
See README.md for uv setup and terminal commands.
"""

# %% [markdown]
# # Fast Agent - Build a SQL Agent fast!
#
# <img src="./assets/LC_L1_Top.png" align="left" width="500">

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

from env_utils import doublecheck_env, doublecheck_pkgs
from llm import create_model

llm = create_model()

# Check and print results
doublecheck_env(str(LESSON_DIR / "example.env"))  # check environmental variables
doublecheck_pkgs(pyproject_path=LESSON_DIR / "pyproject.toml", verbose=True)   # check packages

# %%
from sql_db import SQLDatabase

db = SQLDatabase.from_uri(f"sqlite:///{(LESSON_DIR / 'Chinook.db').as_posix()}")

# %% [markdown]
# Define the runtime context to provide the agent and tools with access to the database.

# %%
from dataclasses import dataclass

from sql_db import SQLDatabase


# define context structure to support dependency injection
@dataclass
class RuntimeContext:
    db: SQLDatabase

# %% [markdown]
# <b>⚠️ Security Note:</b> This demo does not include a filter on LLM-generated commands. In production, you would want to limit the scope of LLM-generated commands. ⚠️
# This tool will connect to the database. Note the use of `get_runtime` to access the graph **runtime context**.

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

# %% [markdown]
# Add a system prompt to define your agents behavior.

# %%
SYSTEM_PROMPT = """You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows of output unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
"""

# %% [markdown]
# Create your agent! Add a model, tools, a prompt, and the runtime access, and go!  You can choose many agents from our [integrations](https://docs.langchain.com/oss/python/integrations/providers) list.

# %%
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=RuntimeContext,
)

# %% [markdown]
# Here's a display of the agent ReAct Loop.

# %%
# Print the agent graph as Mermaid text. This also works in a terminal.
print(agent.get_graph(xray=True).draw_mermaid())

# %% [markdown]
# Run some queries. Notice:
# - The agent does not have the database schema and will need to discover it independently.
# - The agent may make mistakes! By returning error messages, the agent can self-correct its queries.
# - Notice you invoke the agent with `agent.stream`.
#     - This command and the `pretty_print` display the **messages** that communicate information between the model and the tools.
# - Notice the agent doesn't remember the schema between invocations... More on this later!

# %%
question = "Which table has the largest number of entries?"

for step in agent.stream(
    {"messages": question},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %%
question = "Which genre on average has the longest tracks?"

for step in agent.stream(
    {"messages": question},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %%
question = "Please list all of the tables"

for step in agent.stream(
    {"messages": question},
    context={"db": db},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %% [markdown]
# **Create your own query here!**  Add some questions of your own.

# %%
question = "TRY YOUR OWN QUERY HERE"

for step in agent.stream(
    {"messages": question},
    context={"db": db},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %% [markdown]
# ### Let's try this in Studio

# %%
