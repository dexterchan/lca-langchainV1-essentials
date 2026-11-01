"""Interactive Python version of L7_structuredOutput.ipynb.

Run the # %% cells in order in VS Code or Spyder.
See README.md for uv setup and terminal commands.
"""

# %% [markdown]
# # Structured Output
# <img src="./assets/LC_StructuredOutput.png" width="500">

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
# ## Structured Output Example

# %%
from typing_extensions import TypedDict

from langchain.agents import create_agent


class ContactInfo(TypedDict):
    name: str
    email: str
    phone: str


agent = create_agent(model=llm, response_format=ContactInfo)

recorded_conversation = """We talked with John Doe. He works over at Example. His number is, let's see,
five, five, five, one two three, four, five, six seven. Did you get that?
And, his email was john at example.com. He wanted to order 50 boxes of Captain Crunch."""

result = agent.invoke(
    {"messages": [{"role": "user", "content": recorded_conversation}]}
)

pp(result["structured_response"])

# %% [markdown]
# #### Multiple data types are supported

# %% [markdown]
# * pydantic `BaseModel`
# * `TypedDict`
# * `dataclasses`
# * json schema (dict)

# %%
from langchain.agents import create_agent
from pydantic import BaseModel


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


agent = create_agent(model=llm, response_format=ContactInfo)

recorded_conversation = """ We talked with John Doe. He works over at Example. His number is, let's see,
five, five, five, one two three, four, five, six seven. Did you get that?
And, his email was john at example.com. He wanted to order 50 boxes of Captain Crunch."""

result = agent.invoke(
    {"messages": [{"role": "user", "content": recorded_conversation}]}
)

pp(result["structured_response"])

# %%
