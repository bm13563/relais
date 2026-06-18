"""Conversational pipelines.

A normal pipeline run is fire-and-forget. A *conversation* runs the same pipeline
but suspends at any step marked ``await_input``, returns control to the caller,
and resumes when the caller provides the human's reply — with the pipeline's
agents kept alive (in RAM) between turns, so they retain full context.

The hard constraint this module solves: a live ClaudeSDKClient is bound to the
asyncio loop and task it was created in, and its connect/disconnect must happen
in the same task (anyio cancel-scope rule). So:

- relais runs ONE persistent background event loop on a daemon thread (the
  conversation runtime). All conversation agent clients live there.
- Each conversation has ONE long-lived driver coroutine on that loop that owns
  its agents for the whole conversation. continue_conversation() signals the
  driver via a queue/future rather than spawning a new task per turn.
- continue_conversation() is synchronous to the caller (submits to the loop and
  blocks for the result), so it is safe to call from a sync context — e.g. a
  FastAPI ``def`` route running in the threadpool.

Fire-and-forget ``pipeline.run()`` is untouched and never uses this runtime.
"""

from __future__ import annotations
import asyncio
import threading
import time
import uuid
from typing import Dict, Optional

from .logging_config import get_logger

log = get_logger("relais.conversation")

# Idle conversations are evicted after this many seconds as a safety net, even
# though callers are expected to end_conversation() explicitly.
DEFAULT_IDLE_TTL = 1800.0


class _Runtime:
    """A single persistent background event loop shared by all conversations."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._run, name="relais-conversations", daemon=True
                )
                self._thread.start()
            return self._loop

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro):
        """Run a coroutine on the background loop and block for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop()).result()


_runtime = _Runtime()


class Turn:
    """The result of one conversational turn.

    Attributes:
        output: The routing data produced by the step that suspended (what the
            human sees), or None for a pure-park entry suspension.
        step: Name of the step that produced this turn's output.
        awaiting: True if the pipeline is suspended waiting for the next input.
            False only if the pipeline ran to its end during this turn.
    """

    def __init__(self, output, step, awaiting):
        self.output = output
        self.step = step
        self.awaiting = awaiting

    def to_dict(self):
        return {"output": self.output, "step": self.step, "awaiting": self.awaiting}


class Conversation:
    """A live, resumable run of a pipeline, driven turn by turn.

    Created via ``Pipeline.start_conversation()``. Its agents live on the shared
    conversation runtime for the whole conversation. Call ``continue_conversation``
    once per human message; call ``end_conversation`` when done.
    """

    def __init__(self, orchestrator, config, args=None, idle_ttl=DEFAULT_IDLE_TTL):
        self.id = uuid.uuid4().hex
        self._orchestrator = orchestrator
        self._config = config
        self._args = dict(args) if args else {}
        self._idle_ttl = idle_ttl
        self._last_active = time.time()
        self._closed = False
        self._turn_lock = threading.Lock()  # reject overlapping turns on one conversation

        # Driver state (lives on the runtime loop).
        self._inbox: Optional[asyncio.Queue] = None
        self._driver_task = None
        self._run_id = None

    def _start(self):
        # Create the inbox and driver task on the runtime loop.
        self._run_id = self._orchestrator.state_manager.create_pipeline_run(
            pipeline_name=self._config.name,
            start_step=self._config.start_step,
            args=self._args,
        )
        _runtime.submit(self._spawn_driver())
        log.info("conversation_started", conversation=self.id, run=self._run_id, pipeline=self._config.name)

    async def _spawn_driver(self):
        self._inbox = asyncio.Queue()
        self._driver_task = asyncio.ensure_future(self._driver())

    async def _driver(self):
        """One long-lived task owning this conversation's agents for its lifetime.

        Keeps run_agents warm across turns. Each turn message runs a segment from
        the parked step to the next await_input (or the end). All client
        connect/query/disconnect happen inside THIS task, satisfying anyio.
        """
        orch = self._orchestrator
        run_agents: Dict[str, "object"] = {}
        mcp_server = orch.tool_registry.create_mcp_server()
        next_step = self._config.start_step
        try:
            while True:
                text, fut = await self._inbox.get()
                if text is _CLOSE:
                    fut.set_result(None)
                    return
                try:
                    seg = await orch._run_segment(
                        self._run_id, self._config, next_step, text, self._args,
                        run_agents, mcp_server, conversational=True,
                    )
                except Exception as e:
                    fut.set_exception(e)
                    continue

                if seg["suspended"]:
                    next_step = seg["next_step"]
                    fut.set_result(Turn(seg["output"], seg["step"], awaiting=True))
                else:
                    orch.state_manager.complete_pipeline(self._run_id)
                    fut.set_result(Turn(seg["output"], seg["step"], awaiting=False))
                    return
        finally:
            for agent in reversed(list(run_agents.values())):
                if agent.has_client():
                    await orch._safe_disconnect(agent)

    def continue_conversation(self, text: str) -> Turn:
        """Provide the next human input; run forward to the next suspension or the
        end. Returns a Turn. Blocks until the turn completes (safe from sync code)."""
        if self._closed:
            raise RuntimeError("conversation is closed")
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("a turn is already in progress for this conversation")
        try:
            if self._inbox is None:
                self._start()
            self._last_active = time.time()
            turn = _runtime.submit(self._send(text))
            if not turn.awaiting:
                self._closed = True
                _conversations.pop(self.id, None)
            return turn
        finally:
            self._turn_lock.release()

    async def _send(self, text):
        fut = _runtime.loop().create_future()
        await self._inbox.put((text, fut))
        return await fut

    def end_conversation(self):
        """Tear down: disconnect agents, clear state, remove from the registry."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._inbox is not None:
                _runtime.submit(self._send(_CLOSE))
        finally:
            _conversations.pop(self.id, None)
            log.info("conversation_ended", conversation=self.id)

    def is_idle(self, now=None):
        now = now if now is not None else time.time()
        return (now - self._last_active) > self._idle_ttl


_CLOSE = object()

# Active conversations, by id. Holds live agents in RAM — eviction is required.
_conversations: Dict[str, Conversation] = {}


def register(conversation: Conversation):
    _evict_idle()
    _conversations[conversation.id] = conversation


def get(conversation_id: str) -> Optional[Conversation]:
    return _conversations.get(conversation_id)


def _evict_idle():
    now = time.time()
    for cid, convo in list(_conversations.items()):
        if convo.is_idle(now):
            log.info("conversation_evicted_idle", conversation=cid)
            convo.end_conversation()
