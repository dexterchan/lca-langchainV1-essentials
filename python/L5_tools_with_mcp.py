"""Interactive Python version of L5_tools_with_mcp.ipynb.

Run the # %% cells in order in VS Code or Spyder.
See README.md for uv setup and terminal commands.
"""

# %% [markdown]
# # Tools with MCP ⏰
#
# The Model Context Protocol (MCP) provides a standardized way to connect AI agents to external tools and data sources. Let's connect to an MCP server using `langchain-mcp-adapters`.

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
import asyncio
from concurrent.futures import ThreadPoolExecutor

import sys

# MCP's Windows subprocess fallback needs a real stderr file handle.
# Jupyter replaces sys.stderr with an object whose fileno() is unsupported.
if sys.platform == "win32":
    sys.stderr = sys.__stderr__

from langchain_mcp_adapters.client import MultiServerMCPClient
import shutil
import tempfile
from pathlib import Path

# Use the official Python MCP time server. The npm server previously used here
# does not install on Windows. Resolving uvx gives subprocess an absolute path,
# which also avoids the Windows uvx.exe/command lookup difference.
uvx = shutil.which("uvx")
if uvx is None:
    raise RuntimeError("uvx was not found. Install uv from https://docs.astral.sh/uv/.")

mcp_client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": uvx,
            "args": [
                "--cache-dir",
                str(Path(tempfile.gettempdir()) / "mcp-uv-cache"),
                # mcp-server-time (mcp>=1.23.0, no upper bound) breaks against
                # mcp 2.0.0+, which renamed McpError to MCPError:
                # https://github.com/modelcontextprotocol/servers/issues/4570
                "--with",
                "mcp<2",
                "mcp-server-time",
            ],
        }
    },
)

# Load tools from the MCP server
# Use a worker thread because an interactive console may already run an event loop.
with ThreadPoolExecutor(max_workers=1) as executor:
    mcp_tools = executor.submit(asyncio.run, mcp_client.get_tools()).result()
print(f"Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")

# %% [markdown]
# Create an agent with the MCP-provided time tools.

# %%
from langchain.agents import create_agent

agent_with_mcp = create_agent(
    model=llm,
    tools=mcp_tools,
    system_prompt="You are a helpful assistant",
)

# %% [markdown]
# Ask about the current time in San Francisco.

# %%
# The MCP tools are async, so run the agent in a worker thread too.
with ThreadPoolExecutor(max_workers=1) as executor:
    result = executor.submit(
        asyncio.run,
        agent_with_mcp.ainvoke(
            {"messages": [{"role": "user", "content": "What's the time in SF right now?"}]}
        ),
    ).result()
for msg in result["messages"]:
    msg.pretty_print()

# %%
