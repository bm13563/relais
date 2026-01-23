# Relais

A framework for building multi-step AI agent pipelines using the Claude Agent SDK.

*Relais* (French for "relay") - agents relay data to each other through pipeline steps.

## Overview

Relais provides a high-level abstraction for orchestrating AI agent workflows. It enables you to:

- Define multi-step pipelines with custom tools at each step
- Route between steps based on tool results (conditional branching)
- Inject dynamic context via hooks
- Run isolated subagents for research or specialized tasks
- Ground agent responses to pipeline data (reduce hallucination from training data)
- Persist pipeline state to SQLite

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
from relais import Pipeline, PipelineStep, tool, Annotated

# Define a tool
@tool("greet", "Greet the user")
async def greet(name: Annotated[str, "The name of the person to greet"]) -> dict:
    return {"content": [{"type": "text", "text": f"Hello, {name}!"}]}

# Create a pipeline
pipeline = Pipeline.create(
    name="greeting_pipeline",
    steps={
        "greet": PipelineStep(
            name="greet",
            instruction="greet",  # loads instructions/greet.md
            max_turns=3,
            tools=[greet],
            next={"default": None}  # ends pipeline
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

```python
PipelineStep(
    name="analyze",
    instruction="analyze",
    tools=[classify_request],
    next={
        "field": "category",
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

```python
def get_current_time():
    return {"timestamp": datetime.now().isoformat()}

def get_user_preferences():
    return {"theme": "dark", "language": "en"}

PipelineStep(
    name="personalized_response",
    instruction="respond",
    tools=[respond_tool],
    hooks=[get_current_time, get_user_preferences],
    next={"default": None}
)
```

### Subagents

Run isolated agents that don't share conversation context:

```python
PipelineStep(
    name="research",
    instruction="research",
    tools=[search],
    subagent=True,  # isolated session
    subagent_model="opus",  # optional: use different model
    next={"default": "summarize"}
)
```

### Grounded Mode

Constrain agent to only use pipeline data (reduces training data hallucination):

```python
pipeline = Pipeline.create(
    name="grounded_pipeline",
    grounded=True,  # applies to all steps
    # ...
)

# Or per-subagent:
PipelineStep(
    name="research",
    subagent=True,
    subagent_grounded=True,  # only for this subagent
    # ...
)
```

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

Authentication is handled via Claude Code CLI (run `claude` to authenticate).

### Run Examples

```bash
# Simple greeting pipeline
uv run examples/pipelines/simple_greeting.py

# Research pipeline with subagent
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

# Run only e2e tests (requires Claude Code CLI auth)
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
│   ├── executor.py      # Pipeline execution engine
│   ├── tools.py         # Tool registry and @tool decorator
│   ├── state.py         # SQLite state persistence
│   ├── router.py        # Command routing
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
    name: str,                    # Unique pipeline identifier
    steps: Dict[str, PipelineStep],  # Step definitions
    start_step: str,              # First step to execute
    instructions_dir: Path,       # Path to instruction markdown files
    db_config: str,               # SQLite database path
    model: str = "sonnet",        # Model for all steps (sonnet, opus, haiku)
    grounded: bool = False,       # Constrain to pipeline data only
    cwd: str = None               # Working directory for file operations
)

pipeline.run(initial_input: str, args: dict = None) -> str  # Returns run_id
pipeline.get_run(run_id: str) -> PipelineRunState
pipeline.resume(run_id: str, user_input: str = None)
pipeline.initialize_db()
```

### PipelineStep

```python
PipelineStep(
    name: str,                    # Step identifier
    instruction: str,             # Instruction file name (without .md)
    next: dict,                   # Routing rules
    max_turns: int = 10,          # Max API round-trips
    tools: List[Union[str, Callable]],  # Available tools
    hooks: List[Callable] = [],   # Context injection functions
    temperature: float = None,    # Temperature override
    subagent: bool = False,       # Run as isolated subagent
    subagent_model: str = None,   # Model override for subagent
    subagent_grounded: bool = None  # Grounding override for subagent
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
