# Relais

A framework for building multi-step AI agent pipelines on Claude — through the Claude
Agent SDK, or in-process on the Anthropic Messages API (see [Engines](#engines)).

*Relais* (French for "relay") - agents relay data to each other through pipeline steps.

## Overview

Relais is an AI state machine operated by config. You declare tools, instructions,
agents, and routing; each step is driven by an agent that can use only the tools it
was granted and does exactly one thing before handing off. It enables you to:

- Define multi-step pipelines where each step is driven by an explicit agent
- Grant each agent a fixed tool set that is **enforced**, not merely suggested — an
  agent physically cannot call a tool it was not given
- Route between steps based on a step's structured output (conditional branching)
- Inject dynamic context via hooks (sync or async)
- Isolate steps onto separate agents (separate clients, separate context) or share
  one agent across steps
- Persist run results to SQLite and log structured step events to spool

### Core model

- **Agent** — owns a model, a `max_turns` budget, a `steps` budget, a tool set, and
  the engine its steps run on. A live instance opens its engine state on first run
  (a CLI client, or a transcript) and keeps it across the steps it participates in,
  so its conversation context carries from step to step. With
  `steps=N` the instance expires after N steps and the next route to that agent
  starts a fresh instance with clean context; `steps=None` (default) persists for
  the whole run. Two different agents always have isolated context.
- **Step** — names an instruction file, the tools available *for that step*, the
  `response_tool` whose output is captured, the agent that runs it, and routing rules.
- **response_tool** — every step must declare one. Only that tool's output is captured,
  used for routing, and passed to the next step. Text responses are not propagated.

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

# With dev dependencies
uv pip install -e ".[dev]"
```

## Quick Start

```python
from pathlib import Path
from relais import Pipeline, PipelineStep, PipelineAgent, tool, Annotated

# Define a tool. Its return value becomes the step's structured output.
@tool("greet", "Greet the user")
async def greet(name: Annotated[str, "The name of the person to greet"]) -> dict:
    return {"content": [{"type": "text", "text": f"Hello, {name}!"}]}

# Define the agent that runs the step. It owns max_turns / model and is granted
# exactly the tools it may use.
greeter = PipelineAgent(name="greeter", tools=[greet], max_turns=3)

# Create a pipeline
pipeline = Pipeline.create(
    name="greeting_pipeline",
    steps={
        "greet": PipelineStep(
            name="greet",
            instruction="greet",       # loads instructions/greet.md
            response_tool="greet",     # this tool's output is captured + routed
            tools=[greet],             # tools available for this step
            agent=greeter,             # required: the agent that runs the step
            next={"default": None}     # ends pipeline
        )
    },
    start_step="greet",
    instructions_dir=Path("./instructions"),
    db_config="./pipeline.db"
)

# Initialize and run
pipeline.initialize_db()
run_id = pipeline.run("Please greet Alice!")
state = pipeline.get_run(run_id)
print(state.step_results)
```

## Features

### Conditional Routing

Route to different steps based on tool results:

Routing reads a field from the step's `response_tool` output:

```python
PipelineStep(
    name="analyze",
    instruction="analyze",
    response_tool="classify_request",
    tools=[classify_request],
    agent=PipelineAgent(name="classifier", tools=[classify_request], max_turns=2),
    next={
        "field": "category",  # read from classify_request's output
        "routes": [
            {"equals": "question", "goto": "answer_question"},
            {"equals": "task", "goto": "perform_task"},
        ],
        "default": "chat"
    }
)
```

### Hooks for Dynamic Context

Inject runtime data into step context:

Hooks run before the step and their output is injected as `[Hook Data]`. They may
be sync or async:

```python
def get_current_time():
    return {"timestamp": datetime.now().isoformat()}

async def get_user_preferences():
    return {"theme": "dark", "language": "en"}

PipelineStep(
    name="personalized_response",
    instruction="respond",
    response_tool="respond_tool",
    tools=[respond_tool],
    hooks=[get_current_time, get_user_preferences],
    agent=PipelineAgent(name="responder", tools=[respond_tool]),
    next={"default": None}
)
```

### Isolated vs. shared agents

Give two steps **different** agents and they get separate engine state and isolated
conversation context — the later step sees the earlier one only through
`[Previous Step Output]`:

```python
researcher = PipelineAgent(name="researcher", tools=[search], max_turns=5)
summarizer = PipelineAgent(name="summarizer", tools=[create_summary], max_turns=3)

steps = {
    "research": PipelineStep(
        name="research", instruction="research", response_tool="search",
        tools=[search], agent=researcher, next={"default": "summarize"},
    ),
    "summarize": PipelineStep(
        name="summarize", instruction="summarize", response_tool="create_summary",
        tools=[create_summary], agent=summarizer, next={"default": None},
    ),
}
```

Give two steps the **same** agent and it persists across them, keeping its
conversation live in its engine state (its CLI client, or its transcript).

**Step budget (`steps=N`).** When you want an agent to carry context across a
bounded unit of work and then start clean, give it a step budget:

```python
# A worker that runs draft -> review as one attempt, then resets on retry.
worker = PipelineAgent(name="worker", tools=[...], steps=2)

steps = {
    "draft":  PipelineStep(name="draft", ..., agent=worker, next={"default": "review"}),
    "review": PipelineStep(
        name="review", ..., agent=worker,
        next={"field": "ok",
              "routes": [{"equals": False, "goto": "draft"}],  # retry: loop back
              "default": None},
    ),
}
```

The instance spends its 2-step budget on `draft`→`review`. If `review` loops back
to `draft`, the expired instance is retired and a **fresh** worker (clean context)
takes the retry. Size `N` to a loop body to reset on each loop-back; use a large
`N` (or `None`) for a refinement loop that should *remember* prior attempts.

### Logging & debugging (spool)

A run executes start-to-finish in one process. Instead of a pause/resume debug
mode, relais logs structured step events to [spool](https://github.com/bm13563/spool):
each pipeline writes its own queryable JSONL stream
(`~/.local/share/tdl-crypto/logs/relais.<pipeline>.jsonl`) with events like
`run_started`, `step_start` (including the full context), `step_done` (turns,
routing data, next step), and `run_completed`. After a run, inspect results via
`pipeline.get_run(run_id)` and trace what happened with the spool TUI or reader API.

### Tool Gating (defined, not recommended)

The tools listed on a step are the *only* tools that step's agent can successfully
call. This is enforced at runtime: every tool invocation is checked against the
current step's allow-list, and a call to any other tool is refused before the tool
body runs and is excluded from routing data. Telling the model about a tool in an
instruction does not grant access — only the step's `tools` list does.

```python
# secret_tool is registered on the pipeline but NOT in this step's tools list.
# Even if the instruction tells the model to call it, the body never executes.
PipelineStep(
    name="gated",
    instruction="gated",
    response_tool="allowed_tool",
    tools=[allowed_tool],            # the only callable tool here
    agent=PipelineAgent(name="gated_agent", tools=[allowed_tool]),
    next={"default": None},
)
```

### Engines

Each agent's steps run on an **engine**, picked per agent with `PipelineAgent(engine=...)`:

| | `"sdk"` (default) | `"api"` |
|---|---|---|
| Runs a step in | a Claude Code CLI subprocess per agent, driven by the Claude Agent SDK | the host's own process, on the Anthropic Messages API (`anthropic` SDK) |
| Needs | the bundled CLI (ships with `claude-agent-sdk`) and a key or `claude` login | an API key: `env={"ANTHROPIC_API_KEY": key}` on the agent, else the process environment |
| Agent context | the CLI session | a transcript — the API messages the agent has accumulated |
| `session_id` / `resume` | the Claude Code session id | the transcript id |
| Tools | the pipeline's tools (served as an in-process MCP server) plus Claude Code built-ins | the pipeline's tools only — every request offers the agent's tools |

```python
analyst = PipelineAgent(name="analyst", tools=[search, report], engine="api", model="sonnet")
```

On both engines the step's `tools` list is enforced by the registry: the model is
offered the agent's whole tool set, and a call to a tool outside the current step is
refused before the tool body runs. (The api engine keeps the offered set the same on
every step on purpose — a thinking block is bound to the tools it was produced under.)

**Transcripts** live in the run database (`db_config`), table `transcripts`: the api
engine saves an agent's transcript after every step, under the id each step reports as
its `session_id`. Pass it back as `PipelineAgent(resume=...)` on a later run to continue
that conversation. (Run `pipeline.initialize_db()` once on an existing database to add
the table.) A transcript continues only under the prefix it was made with — the agent's
model and tools (and relais's system prompt), which its thinking is bound to. If the
transcript is unknown, or that prefix has changed since (say, a tool added to the agent),
the agent begins a fresh transcript under the same id: the handle stays valid, the
model's memory of the earlier conversation is dropped, and an info line is logged.

**SDK-only agent fields** — the api engine ignores them: `permission_mode`,
`builtin_tools`, `env` beyond `ANTHROPIC_API_KEY`, and the pipeline's `cwd` (as well as
the CLI's setting sources, which relais always switches off). Everything else applies on
both: `model` (the api engine maps `opus` / `sonnet` / `haiku` to `claude-opus-5-5` /
`claude-sonnet-5-5` / `claude-haiku-5-5`; any other value is sent as a model id),
`max_turns`, `thinking`, `effort`, `resume`, `steps`, `tools`.

`thinking` is a tri-state. `None` (the default) sends nothing and leaves it to the
model — and the 5.x models think adaptively by default, so on the api engine `None` is
not "no thinking". `True` turns it on (sdk: a thinking-token budget; api: adaptive
thinking). `False` turns it off on the api engine (`{"type": "disabled"}`; the sdk engine
sets nothing, as for `None`). Claude Opus 5.5 rejects disabled thinking, so an agent on it
must not pass `False` — lower `effort` instead.

The api engine streams each model turn, so `astream()` sees the same `text_delta` and
`tool_use` events, plus a `usage` event after every turn; every engine reports a step's
summed token counts on `step_done` as `usage`.

### Tool Definition

Tools must be async and return the MCP content format. Use `Annotated` to describe parameters,
and default values to mark parameters as optional:

```python
from relais import tool, Annotated

@tool("search", "Search for information")
async def search(
    query: Annotated[str, "The search query"],
    limit: Annotated[int, "Maximum results to return"] = 10,
) -> dict:
    results = perform_search(query, limit)
    return {
        "content": [{"type": "text", "text": json.dumps(results)}]
    }
```

## Running Examples

Examples are in the `examples/pipelines/` directory.

### Prerequisites

**Authentication.** Set `ANTHROPIC_API_KEY` in the environment. On the sdk engine the Agent
SDK drives a bundled Claude Code CLI (no separate install) and that process reads the key —
relais itself never touches credentials; on the api engine the `anthropic` client in the
host process reads it. API-key auth bills at API rates and is the route for anything that
runs for other people: Anthropic does not permit third-party products to ride on a claude.ai
subscription login. With no key set, the bundled CLI (sdk engine) falls back to the `claude` login on that
machine; Anthropic's terms treat that as ordinary individual use of the Agent SDK on your own
subscription — fine for your own experiments, never for anything that runs for other people,
and a product may never offer a claude.ai login of its own
(https://code.claude.com/docs/en/legal-and-compliance, "Authentication and credential use").
A host that keeps the key in its own settings hands it in per agent instead of relying on the
process environment: `PipelineAgent(..., env={"ANTHROPIC_API_KEY": key})`.

### Run Examples

```bash
# Simple greeting pipeline
uv run examples/pipelines/simple_greeting.py

# Research pipeline with an isolated research agent
uv run examples/pipelines/research.py

# Routing pipeline with conditional branching
uv run examples/pipelines/routing.py

# Hooks pipeline with dynamic context
uv run examples/pipelines/hooks_pipeline.py
```

## Running Tests

Tests are organized into unit, integration, and end-to-end categories.

```bash
# Run all tests
uv run pytest

# Run only unit tests (fast, no external dependencies)
uv run pytest tests/unit/

# Run only integration tests (requires SQLite)
uv run pytest tests/integration/

# Run only e2e tests (real model calls; auto-skip without credentials)
uv run pytest tests/e2e/

# Run tests by marker
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m e2e

# Run with coverage
uv run pytest --cov=relais

# Run a specific test file
uv run pytest tests/unit/test_pipeline.py

# Run a specific test
uv run pytest tests/unit/test_step.py::test_resolve_next_default
```

## Project Structure

```
relais/
├── src/relais/
│   ├── __init__.py      # Public API exports
│   ├── pipeline.py      # High-level Pipeline class
│   ├── step.py          # PipelineStep definition
│   ├── agent.py         # PipelineAgent definition + lifecycle
│   ├── executor.py      # Pipeline orchestration: context, routing, state
│   ├── engine.py        # The engine seam: StepOutcome, Engine
│   ├── engines/         # sdk.py (Claude Agent SDK), api.py (Messages API)
│   ├── tools.py         # Tool registry, @tool decorator, per-step gating
│   ├── state.py         # SQLite persistence for pipeline runs and transcripts
│   ├── logging_config.py # spool logging setup
│   └── utils.py         # Utilities
├── examples/
│   ├── pipelines/       # Example pipeline scripts
│   ├── instructions/    # Markdown instruction files
│   ├── tools/           # Example tool definitions
│   ├── hooks/           # Example hook functions
│   └── config.py        # Shared example configuration
├── tests/
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── e2e/             # End-to-end tests
└── pyproject.toml
```

## API Reference

### Pipeline

```python
Pipeline.create(
    name: str,                       # Unique pipeline identifier
    steps: Dict[str, PipelineStep],  # Step definitions (each must carry an agent)
    start_step: str,                 # First step to execute
    instructions_dir: Path,          # Path to instruction markdown files
    db_config: str,                  # SQLite database path
    cwd: str = None,                 # Working directory for file operations
    verbose: bool = False,           # Print step output + token usage to console
)

pipeline.run(initial_input: str, args: dict = None) -> str          # sync; returns run_id
await pipeline.arun(initial_input: str, args: dict = None) -> str   # the same, from a running event loop
async for event in pipeline.astream(initial_input, args)            # the run's events as they happen:
                                                                     # step_start, text_delta, tool_use, usage,
                                                                     # step_done, run_done
pipeline.get_run(run_id: str) -> PipelineRunState
pipeline.start_conversation(args: dict = None) -> Conversation       # docs/conversations.md
pipeline.initialize_db()
```

Model and turn budget are configured on the agent, not on `Pipeline.create`.

### PipelineStep

```python
PipelineStep(
    name: str,                          # Step identifier
    instruction: str,                   # Instruction file name (without .md)
    instruction_text: str = None,       # The instruction inline instead (no file read)
    max_visits: int = None,             # Cap on entries per run; past it, the default route
    reads: List[str] = [],              # Earlier steps whose outputs this step is shown ([Output of <step>])
    response_tool: str,                 # Required: tool whose output is captured + routed
    tools: List[Union[str, Callable]] = [],  # Tools available for this step (enforced)
    hooks: List[Callable] = [],         # Context injection functions (sync or async)
    agent: PipelineAgent = None,        # Required at pipeline-create time
    next: dict = {"default": None},     # Routing rules
)
```

### PipelineAgent

```python
PipelineAgent(
    name: str,                          # Unique agent identifier
    tools: List[Union[str, Callable]] = [],  # Tools this agent may use
    steps: int = None,                  # None = persists all run; N = expire after N steps
    max_turns: int = 10,                # Max model round-trips per step
    model: str = "opus",                # Model (opus, sonnet, haiku — or a model id)
    thinking: bool = None,              # None = model default; True = on; False = off (api; not Opus 5.5)
    engine: str = "sdk",                # "sdk" (Claude Code CLI) or "api" (Messages API) — see Engines
    permission_mode: str = "acceptEdits",  # sdk only: permission mode for built-in tools
    effort: str = None,                 # Reasoning effort (low … max); None = default
    resume: str = None,                 # Session (sdk) / transcript (api) id to resume
    env: dict = None,                   # Extra env for this agent's Claude Code subprocess,
                                        # e.g. {"ANTHROPIC_API_KEY": key} from the host's settings
                                        # (the api engine reads only ANTHROPIC_API_KEY)
    builtin_tools: list = None,         # sdk only: Claude Code built-ins the client may have at all;
                                        # [] = none (an agent inside a server), None = the SDK default
)
```

### @tool Decorator

```python
from relais import tool, Annotated

@tool(name: str, description: str)
async def my_tool(
    required_param: Annotated[str, "Description of the parameter"],
    optional_param: Annotated[int, "Optional with default"] = 10,
) -> dict:
    return {
        "content": [{"type": "text", "text": "result"}]
    }
```

- Parameter types are inferred from type hints
- Use `Annotated[type, "description"]` to add parameter descriptions
- Parameters with default values are optional; others are required

## License

MIT
