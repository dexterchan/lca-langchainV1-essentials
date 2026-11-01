"""Interactive Python version of L4_tools.ipynb.

Run the # %% cells in order in VS Code or Spyder.
See README.md for uv setup and terminal commands.
"""

# %% [markdown]
# # Tools  🔨🪚🔩⚙️⚒️🧲🧰🔧🪛🔩⚙️🔦🧭🔗⛓️🧮
# Tools allow agents to 'Act' in the real world.
# Careful descriptions can help your agent discover how to use your tools.
#
# LangChain supports many tool formats and tool sets. Here we will cover some common cases, but check the [docs](https://docs.langchain.com/oss/python/langchain/tools) for more information.

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

# %% [markdown]
# ## Calculator example
#
# In this example, the docstring and inferred arguments and argument types are used by the LLM to detetermine when and how to call the tool.

# %%
from typing import Literal

from langchain.tools import tool


@tool
def real_number_calculator(
    a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"]
) -> float:
    """Perform basic arithmetic operations on two real numbers."""
    print("🧮 Invoking calculator tool")
    # Perform the specified operation
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b
    else:
        raise ValueError(f"Invalid operation: {operation}")

# %%
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[real_number_calculator],
    system_prompt="You are a helpful assistant",
)

# %% [markdown]
# This invokes your calculator tool.

# %%
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is 3.1125 * 4.1234"}]}
)
print(result["messages"][-1].content)

# %% [markdown]
# We can check the **metadata** in [LangSmith Observability](https://smith.langchain.com/public/b77bde6c-f0ad-4256-bfab-7d514ece3405/r) to see this.

# %% [markdown]
# The tool description can have a big impact.
# This may not  invoke your calculator tool because the inputs are integers.  (results vary from run to run)

# %%
result = agent.invoke({"messages": [{"role": "user", "content": "what is 3 * 4"}]})
print(result["messages"][-1].content)

# %% [markdown]
# This often does not invoke the tool though the input are real numbers. (results vary from run to run)

# %%
result = agent.invoke({"messages": [{"role": "user", "content": "what is 3.0 * 4.0"}]})
print(result["messages"][-1].content)

# %% [markdown]
# ## Adding a more detailed description
# While a basic description is often sufficient, LangChain has support for enhanced descriptions. The example below uses one method: Google Style argument descriptions. Used with `parse_docstring=True`, this will parse and pass the arg descriptions to the model. You can rename the tool and change its description. This can be effective when you are sharing a standard tool but would like agent-specific instructions.

# %%
from typing import Literal

from langchain.tools import tool


@tool(
    "calculator",
    parse_docstring=True,
    description=(
        "Perform basic arithmetic operations on two real numbers."
        "Use this whenever you have operations on any numbers, even if they are integers."
    ),
)
def real_number_calculator(
    a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"]
) -> float:
    """Perform basic arithmetic operations on two real numbers.

    Args:
        a (float): The first number.
        b (float): The second number.
        operation (Literal["add", "subtract", "multiply", "divide"]):
            The arithmetic operation to perform.

            - `"add"`: Returns the sum of `a` and `b`.
            - `"subtract"`: Returns the result of `a - b`.
            - `"multiply"`: Returns the product of `a` and `b`.
            - `"divide"`: Returns the result of `a / b`. Raises an error if `b` is zero.

    Returns:
        float: The numerical result of the specified operation.

    Raises:
        ValueError: If an invalid operation is provided or division by zero is attempted.
    """
    print("🧮  Invoking calculator tool")
    # Perform the specified operation
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b
    else:
        raise ValueError(f"Invalid operation: {operation}")

# %%
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[real_number_calculator],
    system_prompt="You are a helpful assistant",
)

# %%
result = agent.invoke({"messages": [{"role": "user", "content": "what is 3.0 * 4.0"}]})
print(result["messages"][-1].content)

# %% [markdown]
# Let's check our [LangSmith Observability trace](https://smith.langchain.com/public/7d65902c-bd3c-4fc6-bbd3-7c1d66566fda/r) to see the tool description.

# %%
result = agent.invoke({"messages": [{"role": "user", "content": "what is 3 * 4"}]})
print(result["messages"][-1].content)

# %% [markdown]
# ## Try your own.
# Create a tool of your own and try it here!

# %%

@tool
def your_tool(
    a: float, b: float,
) -> float:
    """Perform your favorite operation.

    Args:
        a (float): operator a description
        b (float): operator b description

    Returns:
        float: description
    """
    pass

# %%
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[your_tool],
    system_prompt="You are a helpful assistant",
)
