"""Run Lesson 1 with fixed SQL states and explicit LangGraph edges."""

# %% [markdown]
# # Lesson 1: A fixed SQL workflow
# START -> prepare -> generate_sql -> validate_sql -> execute_sql -> summarize -> END
# An invalid query returns to generate_sql only while the attempt limit permits it.
# The model cannot select tools, add nodes, or choose an edge.

# %%
import sys
from pathlib import Path

LESSON_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (LESSON_DIR / "llm_workflows.py").is_file():
    LESSON_DIR = LESSON_DIR / "python"
if str(LESSON_DIR) not in sys.path:
    sys.path.insert(0, str(LESSON_DIR))

from langgraph.graph import END, START, StateGraph
from llm import create_workflow_model
from llm_workflows import SQLContext, SQLNodes, SQLState
from sql_db import SQLDatabase

RuntimeContext = SQLContext

# %% [markdown]
# ## Define every node and edge
# The model returns SQL text. Python chooses the next state.
# Validation and execution enforce read-only access independently of the prompt.
# At most two SQL drafts are requested by default.

# %%
def build_graph(model, *, checkpointer=None, max_attempts=2):
    """Build a fixed SQL workflow with bounded correction attempts."""
    nodes = SQLNodes(model, max_attempts=max_attempts)
    builder = StateGraph(SQLState, context_schema=SQLContext)
    builder.add_node("prepare", nodes.prepare)
    builder.add_node("generate_sql", nodes.generate_sql)
    builder.add_node("validate_sql", nodes.validate_sql)
    builder.add_node("execute_sql", nodes.execute_sql)
    builder.add_node("summarize", nodes.summarize)
    builder.add_node("stop", nodes.stop)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "generate_sql")
    builder.add_edge("generate_sql", "validate_sql")
    builder.add_conditional_edges("validate_sql", nodes.route, {
        "ready": "execute_sql", "retry": "generate_sql", "stop": "stop",
    })
    builder.add_conditional_edges("execute_sql", nodes.route, {
        "ready": "summarize", "retry": "generate_sql", "stop": "stop",
    })
    builder.add_edge("summarize", END)
    builder.add_edge("stop", END)
    return builder.compile(checkpointer=checkpointer)

# %% [markdown]
# ## Create the model and inspect the state diagram
# Importing this file does not load keys or run model requests.
# Run this cell and the next cells in order for the live lesson.

# %%
if __name__ == "__main__":
    llm = create_workflow_model()
    db = SQLDatabase.from_uri(f"sqlite:///{(LESSON_DIR / 'Chinook.db').as_posix()}")
    graph = build_graph(llm)
    print(graph.get_graph().draw_mermaid())

# %%
if __name__ == "__main__":
    questions = [
        "Which table has the largest number of entries?",
        "Which genre on average has the longest tracks?",
        "Please list all of the tables",
    ]
    for question in questions:
        print(f"\nQuestion: {question}")
        for update in graph.stream(
            {"question": question}, context=SQLContext(db=db),
            config={"recursion_limit": 20}, stream_mode="updates",
        ):
            for node, changes in update.items():
                print(f"\nNode: {node}")
                for field in ("attempts", "sql", "error", "rows", "answer"):
                    if field in changes:
                        print(f"{field}: {changes[field]}")
