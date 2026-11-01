"""Interactive Python version of L2_messages.ipynb.

Run the # %% cells in order in VS Code or Spyder.
See README.md for uv setup and terminal commands.
"""

# %% [markdown]
# # ✉️ Messages
#   <img src="./assets/LC_Messages.png" width="500">
#
# Messages are the fundamental unit of context for models in LangChain. They represent the input and output of models, carrying both the content and metadata needed to represent the state of a conversation when interacting with an LLM.

# %% [markdown]
# ## Setup

# %% [markdown]
# Load and/or check for needed environmental variables

# %%
import sys
from pathlib import Path
from pprint import pp

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

# %% [markdown]
# ## Human👨‍💻 and AI 🤖 Messages

# %%
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

agent = create_agent(
    model=llm,
    system_prompt="You are a full-stack comedian"
)

# %%
human_msg = HumanMessage("Hello, how are you?")

result = agent.invoke({"messages": [human_msg]})

# %%
print(result["messages"][-1].content)

# %%
print(type(result["messages"][-1]))

# %%
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}\n")

# %% [markdown]
# ### Altenative formats
# #### Strings
# There are situations where LangChain can infer the role from the context, and a simple string is enough to create a message.

# %%
agent = create_agent(
    model=llm,
    system_prompt="You are a terse sports poet.",  # This is a SystemMessage under the hood
)

# %%
result = agent.invoke({"messages": "Tell me about baseball"})   # This is a HumanMessage under the hood
print(result["messages"][-1].content)

# %% [markdown]
# #### Dictionaries

# %%
result = agent.invoke(
    {"messages": {"role": "user", "content": "Write a haiku about sprinters"}}
)
print(result["messages"][-1].content)

# %% [markdown]
# There are multiple roles:
# ```python
# messages = [
#     {"role": "system", "content": "You are a sports poetry expert who completes haikus that have been started"},
#     {"role": "user", "content": "Write a haiku about sprinters"},
#     {"role": "assistant", "content": "Feet don't fail me..."}
# ]
# ```

# %% [markdown]
# ## Output Format
# ### messages
# Let's create a tool so agent will create some tool messages.

# %%
from langchain_core.tools import tool

@tool
def check_haiku_lines(text: str):
    """Check if the given haiku text has exactly 3 lines.

    Returns None if it's correct, otherwise an error message.
    """
    # Split the text into lines, ignoring leading/trailing spaces
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    print(f"checking haiku, it has {len(lines)} lines:\n {text}")

    if len(lines) != 3:
        return f"Incorrect! This haiku has {len(lines)} lines. A haiku must have exactly 3 lines."
    return "Correct, this haiku has 3 lines."

# %%
agent = create_agent(
    model=llm,
    tools=[check_haiku_lines],
    system_prompt="You are a sports poet who only writes Haiku. You always check your work.",
)

# %%
result = agent.invoke({"messages": "Please write me a poem"})

# %%
print(result["messages"][-1].content)

# %%
print(len(result["messages"]))

# %%
for i, msg in enumerate(result["messages"]):
    msg.pretty_print()

# %% [markdown]
# ### Other useful information
# Above, the print messages have just been selecting pieces of the information stored in the messages list. Let's dig into all the information that is available!

# %%
pp(result)

# %% [markdown]
# You can select just the last message, and you can see where the final message is coming from.

# %%
pp(result["messages"][-1])

# %%
pp(result["messages"][-1].usage_metadata)

# %%
pp(result["messages"][-1].response_metadata)

# %% [markdown]
# ### Try it on your own!
# Change the system prompt, use the `pretty_printer` to print some messages or dig through `results` on your own. Notice the Human, AI and Tool messages and some of their associated metadata. Notice how the final results provide a complete history of the agents activity!

# %%
agent = create_agent(
    model=llm,
    tools=[check_haiku_lines],
    system_prompt="Your SYSTEM prompt here",
)

# %%
for i, msg in enumerate(result["messages"]):
    msg.pretty_print()
