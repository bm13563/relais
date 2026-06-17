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
    │       ├── set_current_step(step.name, step.tools)  ← per-step tool gate
    │       ├── Create/reuse ClaudeSDKClient for agent
    │       ├── Send context via client.query(context)
    │       ├── Stream response, capture messages
    │       ├── Tool calls execute via MCP server
    │       │       │
    │       │       ├── Wrapper refuses tools not in step.tools
    │       │       └── Allowed results appended to registry._tool_results
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
| state.py | SQLite persistence for pipeline runs and agent state |

## Data Flow Summary

1. **User Input** → passed to every step unchanged
2. **Tool Result** → captured by MCP wrapper → becomes [Previous Step Output] for next step
3. **Conversation History** → only used in debug mode for resume
4. **Step Results** → persisted to SQLite for debugging/auditing
