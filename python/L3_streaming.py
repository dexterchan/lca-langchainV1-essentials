"""Interactive Python version of L3_streaming.ipynb.

Run the # %% cells in order in VS Code or Spyder.
See README.md for uv setup and terminal commands.
"""

# %% [markdown]
# # Streaming
#
# <img src="./assets/LC_streaming.png" width="400">
#
# Streaming reduces the latency between generating data and the user receiving it.
# There are two types frequently used with Agents:

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
from langchain.agents import create_agent

# %%
agent = create_agent(
    model=llm,
    system_prompt="You are a full-stack comedian",
)

# %% [markdown]
# ## No Streaming (invoke)

# %%
result = agent.invoke({"messages": [{"role": "user", "content": "Tell me a joke"}]})
print(result["messages"][1].content)

# %% [markdown]
# ## values
# You have seen this streaming mode in our examples so far.

# %%
# Stream = values
for step in agent.stream(
    {"messages": [{"role": "user", "content": "Tell me a Dad joke"}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %% [markdown]
# ## messages
# Messages stream data token by token - the lowest latency possible. This is perfect for interactive applications like chatbots.

# %%
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "Write me a family friendly poem."}]},
    stream_mode="messages",
):
    print(f"{token.content}", end="")

# %% [markdown]
# ## Tools can stream too!
# Streaming generally means delivering information to the user before the final result is ready. There are many cases where this is useful. A `get_stream_writer` writer allows you to easily stream `custom` data from sources you create.

# %%
from langchain.agents import create_agent
from langgraph.config import get_stream_writer


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    # stream any arbitrary data
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=llm,
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["values", "custom"],
):
    print(chunk)

# %%
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["custom"],
):
    print(chunk)

# %% [markdown]
# ## Try different modes on your own!
# Modify the stream mode and the select to produce different results.

# %%
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["values", "custom"],
):
    if chunk[0] == "custom":
        print(chunk[1])
