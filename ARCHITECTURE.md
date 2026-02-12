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
    │       ├── [Previous Conversation] ← debug mode only, from accumulated_messages
    │       ├── [User Input]            ← initial_input (passed to every step)
    │       ├── [Previous Step Output]  ← routing_data from previous step's tool call
    │       ├── [Current Step]          ← step.name
    │       ├── [Hook Data]             ← step.get_hook_data()
    │       ├── [Instructions]          ← step.instruction.md
    │       └── [Pipeline Context]      ← PIPELINE_STEP_INSTRUCTION constant
    │       │
    │       ▼
    │   _execute_step(step, context, agent)
    │       │
    │       ├── Create/reuse ClaudeSDKClient for agent
    │       ├── Send context via client.query(context)
    │       ├── Stream response, capture messages
    │       ├── Tool calls execute via MCP server
    │       │       │
    │       │       └── Tool wrapper captures result to
    │       │           ToolRegistry._last_tool_result
    │       │
    │       └── Return StepExecutionResult
    │       │
    │       ▼
    │   _get_routing_data_from_registry()
    │       │
    │       └── Retrieves tool result captured by MCP wrapper
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
Step A tool call
    │
    ▼
Tool wrapper in tools.py (_create_args_wrapper)
    │
    ├── Executes the actual tool function
    └── Saves result: registry._last_tool_result = (tool_name, result)
    │
    ▼
_execute_step returns
    │
    ▼
_get_routing_data_from_registry()
    │
    ├── Calls registry.get_last_tool_result()
    └── Extracts JSON from MCP content format
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
PipelineAgent
    │
    ├── steps=None (persistent)
    │       │
    │       └── Agent lives for entire pipeline
    │           Client reused across all steps
    │           Conversation history maintained in SDK
    │
    └── steps=N (limited)
            │
            └── Agent expires after N steps
                New client created when expired
                Conversation history injected as context
```

## Key Files

| File | Purpose |
|------|---------|
| pipeline.py | High-level API: Pipeline.create(), Pipeline.run() |
| executor.py | Core execution: _execute_pipeline(), _execute_step() |
| tools.py | MCP tool registration, result capture |
| step.py | PipelineStep definition, routing logic |
| agent.py | PipelineAgent lifecycle |
| state.py | SQLite persistence for pipeline runs |
| agent_state.py | SQLite persistence for agent state |

## Data Flow Summary

1. **User Input** → passed to every step unchanged
2. **Tool Result** → captured by MCP wrapper → becomes [Previous Step Output] for next step
3. **Conversation History** → only used in debug mode for resume
4. **Step Results** → persisted to SQLite for debugging/auditing
