# Relais Pipeline Architecture

## Pipeline Flow

```
Pipeline.run(initial_input)
    │
    ▼
PipelineOrchestrator._execute_pipeline()
    │
    ├─► For each step in pipeline:
    │       │
    │       ▼
    │   _build_step_context()
    │       │
    │       ├── [User Input]            ← initial_input (passed to every step)
    │       ├── [Pipeline Args]         ← args (passed to every step)
    │       ├── [Previous Step Output]  ← routing_data from previous step's tool call
    │       ├── [Current Step]          ← step.name
    │       ├── [Hook Data]             ← step.get_hook_data()
    │       ├── [Instructions]          ← step.instruction.md
    │       └── [Pipeline Context]      ← PIPELINE_STEP_INSTRUCTION constant
    │       │
    │       ▼
    │   _execute_step(step, context, agent)
    │       │
    │       ├── set_current_step(step.name, step.tools)  ← per-step tool gate
    │       ├── engine.run_step(...) on the agent's engine (agent.engine)
    │       │       │
    │       │       ├── sdk: create/reuse the agent's ClaudeSDKClient,
    │       │       │        client.query(context), tools served as an MCP server
    │       │       ├── api: create/reuse the agent's transcript, run the
    │       │       │        Messages API tool loop in-process, offering
    │       │       │        the agent's tools (the same on every step)
    │       │       ├── Every tool call goes through the registry wrapper:
    │       │       │       ├── refuses tools not in step.tools
    │       │       │       └── appends allowed results to registry._tool_results
    │       │       └── Returns a StepOutcome (text, tool calls, turns, error, usage)
    │       │
    │       ▼
    │       Capture step.response_tool's output via
    │       registry.get_tool_result(step.response_tool)
    │       (raises ResponseToolNotCalled if it was never called)
    │       │
    │       └── Return StepExecutionResult(routing_data=...)
    │       │
    │       ▼
    │   step.resolve_next(routing_data)
    │       │
    │       └── Determines next step from routing rules
    │
    └─► Loop until next_step is None
```

## Context Passing Between Steps

The **critical path** for context between steps:

```
Step A: agent calls its response_tool
    │
    ▼
Tool wrapper in tools.py (_create_args_wrapper)
    │
    ├── is_tool_allowed(name)? refuse if not in the current step's tools
    ├── Executes the actual tool function
    └── Appends (tool_name, result) to registry._tool_results
    │
    ▼
_execute_step: registry.get_tool_result(step.response_tool)
    │
    └── _extract_from_mcp_content() → JSON dict (or {"response": text})
    │
    ▼
routing_data = extracted dict
    │
    ▼
previous_result = routing_data  (for next iteration)
    │
    ▼
_build_step_context() for Step B
    │
    └── [Previous Step Output]\n{json.dumps(previous_result)}
```

## Agent Lifecycle

```
PipelineAgent (live instance)
    │
    ├── Opens its engine state on first run (agent.client):
    │   sdk → a ClaudeSDKClient (context lives in the CLI session)
    │   api → a transcript (context is the Messages API messages,
    │         saved to SQLite after every step)
    │   Reuses it across the steps this instance participates in
    │
    ├── steps=None → persists for the whole run
    │
    └── steps=N    → consume_step() after each step
                     is_expired() at 0 → engine.close(agent) now;
                     next route to this agent clones a fresh
                     instance with clean context (resets on loop-back)
```

A run executes start-to-finish in one process. There is no pause/resume; two
different agents have two different engine states and therefore isolated context.

## Key Files

| File | Purpose |
|------|---------|
| pipeline.py | High-level API: Pipeline.create(), Pipeline.run() |
| executor.py | Core execution: _execute_pipeline(), _execute_step() |
| engine.py | The engine seam: StepOutcome, Engine, StepEvents |
| engines/sdk.py | sdk engine: Claude Code CLI via the Claude Agent SDK |
| engines/api.py | api engine: in-process tool loop on the Messages API |
| tools.py | Tool registration, per-step gating, result capture (engine-neutral) |
| step.py | PipelineStep definition, routing logic |
| agent.py | PipelineAgent definition + its engine state |
| state.py | SQLite persistence for pipeline runs (and api transcripts) |
| logging_config.py | events through the standard `logging` module (a NullHandler on "relais"; the host's handlers decide) |

## Data Flow Summary

1. **User Input** → passed to every step unchanged
2. **Tool Result** → captured by the registry wrapper → becomes [Previous Step Output] for next step
3. **Step Results** → persisted to SQLite for inspection (get_run / list_runs)
4. **Step Events** → logged as events through the standard `logging` module, wherever the host sends them
