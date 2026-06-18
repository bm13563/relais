# Conversational pipelines

Status: **implemented** on the `conversations` branch. Verified live.

## What it is

A normal pipeline run is fire-and-forget: `pipeline.run(input)` executes every
step to the end, then tears the agents down. A **conversation** runs the same
pipeline but **suspends** at any step marked `await_input`, returns control to
the caller, and **resumes** when the caller provides the human's next message —
with the pipeline's agents kept alive in RAM between turns, so they retain full
context across the whole conversation.

This is a multi-agent pipeline you can *converse with* — the routing, handoff,
and tool-gating are the point. A plain single chat agent could not express an
investigate → cross-check → summarize structure.

## The model

One new concept: **a step can yield to the human after it runs** (`await_input`).
The human's reply then flows **forward** as the input to the next step. Routing
is otherwise unchanged — ordinary `next={...}` rules on ordinary steps.

```python
advisor = Pipeline.create(
    name="trade_advisor",
    start_step="await",
    steps={
        # Entry: a pure-park await step (no agent). The pipeline boots awaiting the
        # opening message. Pure-park steps need no agent and no response_tool.
        "await": PipelineStep(
            name="await", instruction="-", await_input=True, next={"default": "investigate"},
        ),
        "investigate": PipelineStep(
            name="investigate", instruction="investigate", response_tool="findings",
            tools=[query_db], agent=analyst, next={"default": "summarize"},
        ),
        # Mid-pipeline await: the agent runs, produces the summary the human sees,
        # THEN suspends waiting for the reply.
        "summarize": PipelineStep(
            name="summarize", instruction="summarize", response_tool="summary",
            agent=advisor, await_input=True, next={"default": "decide"},
        ),
        # An ordinary routing step whose input is the human's reply. Nothing special.
        "decide": PipelineStep(
            name="decide", instruction="decide", response_tool="verdict",
            agent=advisor,
            next={"field": "action",
                  "routes": [{"equals": "dig_deeper", "goto": "investigate"}],
                  "default": "await"},   # loop back to wait for the next message
        ),
    },
    ...
)
```

Entry-park and mid-pipeline await are the **same mechanism**: there is state, the
state is "awaiting input", and it does not change until input is provided. The
only difference is whether a step ran before the wait (mid: yes, produces output;
entry: no, nothing to show yet).

Routing after an await is an **ordinary step**. The `decide` step above is a
normal conditional-routing step — its only novelty is that its input arrived from
a human. There is no special "route step" type and `await_input` runs no router
itself. Ending the conversation is an explicit caller action (`end_conversation`),
not something an agent decides — so a conversational pipeline typically loops
between work steps and awaits forever until the caller ends it.

## API

```python
convo = pipeline.start_conversation(args=None)   # returns a Conversation handle
convo.id                                          # conversation id

turn = convo.continue_conversation("why should I trade this?")
turn.output     # the suspending step's routing data (what the human sees); None for a pure-park entry
turn.step       # name of the step that produced this turn's output
turn.awaiting   # True if suspended waiting for the next input; False if the pipeline ended

# ... later, same convo, agents still warm with full context ...
turn = convo.continue_conversation("but what about the funding rate?")

convo.end_conversation()   # disconnect agents, clear state, remove from registry
```

`continue_conversation` only advances a conversation that is currently awaiting
input (its resting state between turns). It is **synchronous** — it blocks until
the turn completes — so it is safe to call from sync code.

## Implementation

### Why a background event loop (the substance)

A live `ClaudeSDKClient` is bound to the asyncio loop and **task** it was created
in; its connect/query/disconnect must all happen in the same task (anyio
cancel-scope rule). `pipeline.run()` uses `asyncio.run()`, which creates a loop,
runs the pipeline, and destroys the loop — so held clients cannot survive to the
next request.

Therefore conversations use a **single persistent background event loop** on a
daemon thread (the conversation runtime, in `conversation.py`). All conversation
agent clients live there. Each conversation has **one long-lived driver
coroutine** that owns its agents for the conversation's lifetime; turns are
delivered to it via an `asyncio.Queue` + per-turn `Future`. This keeps every
client's whole lifecycle inside one task.

`continue_conversation` submits to the runtime via
`asyncio.run_coroutine_threadsafe(...).result()` and blocks. Because it blocks,
call it from a **sync context** — e.g. a FastAPI `def` route (which runs in the
threadpool, so blocking does not stall the server's loop). Fire-and-forget
`pipeline.run()` is untouched and never uses this runtime.

This was de-risked with a spike before building: a naive "one task per turn"
approach fails at `disconnect()` with `Attempted to exit cancel scope in a
different task`. The single-driver-task pattern (client lifecycle in one task,
turns via a queue) resolves it. Verified: a client created on the background loop,
driven across multiple sync calls, with memory retained, no cross-loop or
cancel-scope error.

### Executor

The step loop was extracted into `_run_segment(start_step, input, run_agents,
..., conversational)`, which runs from a step until the pipeline ends or (in
conversational mode) a step with `await_input` suspends. It returns
`{suspended, next_step, step, output}`. `run_agents` is passed in and persists
across calls, so agents keep their live clients (and context) between turns. The
caller owns teardown. Fire-and-forget `_execute_pipeline` is now a thin wrapper
over `_run_segment(conversational=False)` and behaves identically (await_input is
ignored there). Tool-gating, step budgets, and DB step-history are unaffected.

### Lifecycle / eviction

A module-level registry (`_conversations`) holds active conversations — they hold
live clients in RAM, so eviction is required, not optional:

- **End**: a turn that runs to the pipeline's end (`awaiting=False`) tears down
  and removes the conversation.
- **Explicit**: `end_conversation()`.
- **Idle**: conversations idle longer than `DEFAULT_IDLE_TTL` (30 min) are
  disconnected and evicted; checked lazily when the registry is touched.
- **Overlap**: a second `continue_conversation` while one is in progress on the
  same conversation is rejected (a conversation is single-threaded by nature).

## Non-goals

- Pausing arbitrarily *within* a step (mid-agent). Suspension is only at step
  boundaries marked `await_input`.
- Surviving process restart. Conversations are RAM-only by design — that is the
  trade for not rehydrating context from stored text (which degraded output and
  was the reason the old "session mode" was removed). The run's step history
  remains in SQLite for audit; the live agents are gone on restart.
- A generic single chat-agent primitive. This is specifically multi-agent
  pipelines you can converse with.

## FastAPI usage (the daytrader case)

```python
# A sync def route — runs in FastAPI's threadpool, so blocking is fine.
@router.post("/chat")
def chat(req: ChatRequest):
    convo = get_or_start(req.conversation_id, advisor)   # registry lookup / start_conversation
    turn = convo.continue_conversation(req.messages[-1].content)
    return {"reply": turn.output, "awaiting": turn.awaiting}
```
