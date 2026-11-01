# Music routing implementation plan

> **For agentic workers:** Execute the tasks inline, using the agreed design and the checks below.

**Goal:** Demonstrate model-based source selection, separate Markdown skills and prose, and a bounded clarification loop.

**Architecture:** A classifies the question and returns `Command`. B reads separate instruction and history files. C invokes a SQL agent created once during graph setup. A clarification node pauses with `interrupt`, then returns to A after a reply.

**Tech Stack:** Python 3.11–3.13, LangGraph, LangChain `create_agent`, SQLite, and the shared OpenRouter model factory.

**Spec:** The user's accepted Q1–Q8 decisions, recorded in the constraints below.

## Global constraints

- Fictional history and matching SQL catalog data stay in `python/studio`.
- Skill instructions and history data occupy separate Markdown files.
- A uses a model prompt to select a source.
- History and recommendations select B. Exact catalog facts select C.
- Mixed or unclear requests ask the user to clarify.
- B uses only Markdown. Missing information produces an honest inability to answer.
- Clarification pauses and resumes the same thread, then returns to A.
- After two unsuccessful clarification rounds, quit to `END`.
- Initialize the model, SQL agent, tools, and checkpointer once per graph.
- Keep all node functions outside `build_graph`.
- Retain SQL read-only enforcement, three agent model calls, and two tool executions per SQL run.
- Preserve the original lessons and both existing `Chinook.db` files.

## Task 1: Update behavioral checks

**File:** `python/tests/test_l1_fast_agent_state.py`

- [x] Replace obsolete prefix and Markdown-section tests with natural-question routing tests.
- [x] Test actual graph interruption, resume, return to A, cancellation, and termination after two rounds.
- [x] Verify malformed classifier output cannot execute tools.
- [x] Verify separate Markdown files supply the answer prompt and missing files stop before answer generation.
- [x] Reuse one graph for multiple requests and verify model/agent factories run once.
- [x] Retain real SQLite schema discovery, read-only rejection, and agent call-limit checks.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_l1_fast_agent_state.py -v` and confirm the new behavior fails before implementation.

## Task 2: Supply the fictional sources

**Files:** `python/studio/lesson_notes.md`, `python/studio/music_history_skill.md`, `python/studio/music_catalog.sql`, `python/studio/music_catalog.db`

- [x] Write prose about Mira Vale, Noah Reed, Lea Moss, and Kai Rowan.
- [x] Put recommendation and source-grounding instructions in the separate skill file.
- [x] Create `Artist`, `Genre`, `Album`, and `Track` tables with matching fictional names.
- [x] Generate the new SQLite fixture from its readable SQL seed, using exclusive file creation to avoid replacing another database.

## Task 3: Implement the graph

**File:** `python/L1_fast_agent_state.py`

- [x] Replace prefix parsing with one model classification call and a closed set of permitted destinations.
- [x] Read fixed Markdown paths under the Studio directory without the old section parser.
- [x] Add a top-level clarification node using `interrupt` and `Command`.
- [x] Bind shared dependencies with `partial` and supply explicit graph destinations for the diagram.
- [x] Compile with one checkpointer and create one reusable SQL agent with its checkpointing disabled.
- [x] Add interactive examples that reuse the graph and database while giving each new question a separate thread.
- [x] Run the focused graph tests.

## Task 4: Document and verify

**File:** `python/README.md`

- [x] Document natural questions, separate sources, resume, termination, and initialization scope.
- [x] Include the actual state diagram and link to the Studio files.
- [x] Run all tests with `.venv/bin/python -m unittest discover -s tests -v`.
- [x] Run Ruff on the changed Python files and `git diff --check`.
- [x] Request a read-only review and resolve material findings.
- [x] Report that no live model requests ran.
