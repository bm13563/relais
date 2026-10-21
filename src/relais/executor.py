"""Pipeline orchestration: steps, context, routing, state.

The orchestrator walks a pipeline step by step. For each step it builds the
context prompt, gates the registry to the step's tools, and hands the step to
its agent's engine (relais.engine — the Claude Code CLI via the Agent SDK, or
the Messages API in-process), which runs the model's turn loop and reports a
StepOutcome. Everything after that is engine-neutral and lives here: the
response tool's captured output becomes the step's routing data, an error before
the answer is a StepError, routing picks the next step, the result is persisted.

Each agent keeps its engine state (a CLI client, a transcript) for the whole
pipeline run, so its conversation context carries across its steps. A run
executes start-to-finish in one process; results are persisted to SQLite for
after-the-fact inspection. Step-level detail is logged to spool as structured
events (see `spool` — query the JSONL stream to debug a run).
"""

from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .step import PipelineStep
from .tools import ToolRegistry, tool_names
from .state import SQLiteStateManager
from .agent import PipelineAgent
from .engine import Engine
from .engines import ENGINES
from .logging_config import get_logger


# Pipeline context appended to every step's prompt.
PIPELINE_STEP_INSTRUCTION = """[Pipeline Context]
You are one step in a multi-step pipeline. Focus your thinking on the specific task for this step. Don't worry about any future steps, they will be handled by other agents, focus all of your thinking on this step.

Do not make multiple parallel tool calls — call tools one at a time.

The next step in the pipeline will only see the output of your response tool. So the entire focus of your effort should go into that call. If you have 10 available turns, use the response tool after 2, then continue reasoning for another 8 turns, that will be 8 turns of context wasted. Call your response tool once you have completed your reasoning."""


class ResponseToolNotCalled(Exception):
    """Raised when a step's declared response_tool was not called by the agent."""
    pass


class StepError(Exception):
    """Raised when a step ends in error (e.g. rate limit) before its response_tool ran.

    Carries the step and agent names, the engine's error kind (e.g. "rate_limit")
    and its result text so a caller can decide what to do (sleep, switch model).
    """
    def __init__(self, step, agent, kind, text):
        super().__init__(f"step '{step}' on agent '{agent}' failed ({kind}): {text}")
        self.step = step
        self.agent = agent
        self.kind = kind
        self.text = text


@dataclass
class PipelineConfig:
    """Configuration for a pipeline.

    Attributes:
        name: Unique identifier for the pipeline
        steps: Step definitions keyed by name
        start_step: Name of the first step to execute
        instructions_dir: Path to instruction markdown files
        agents: PipelineAgent instances keyed by name
        cwd: Working directory for file operations
        verbose: If True, print step output to console
    """
    name: str
    steps: Dict[str, PipelineStep]
    start_step: str
    instructions_dir: str
    agents: Dict[str, 'PipelineAgent'] = None
    cwd: Optional[str] = None
    verbose: bool = False

    def __post_init__(self):
        if self.agents is None:
            self.agents = {}


@dataclass
class StepExecutionResult:
    """Result from executing a single pipeline step."""
    step_name: str
    final_response: str
    tool_results: List[dict]
    turns_used: int
    stop_reason: str  # 'success' or 'error'
    routing_data: Optional[dict] = None
    # The step's continuity handle — the Claude Code session (sdk) or the
    # transcript (api) it ran in — what a host passes back as
    # PipelineAgent.resume to continue the conversation on a later run.
    session_id: Optional[str] = None
    # Token counts summed over the step (relais.engine.USAGE_KEYS); {} if unknown.
    usage: dict = field(default_factory=dict)


class PipelineOrchestrator:
    """Runs pipelines step by step, each step on its agent's engine.

    Execution flow:
    1. Build the context prompt for the current step.
    2. Run the step on its agent's engine (its engine state made once, reused).
    3. Capture the step's response_tool output as routing data.
    4. Apply routing rules to pick the next step.
    5. Persist the step result to SQLite.
    6. Repeat until a step routes to None.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        state_manager: SQLiteStateManager,
        instructions_dir: Path,
        cwd: str = None
    ):
        self.tool_registry = tool_registry
        self.state_manager = state_manager
        self.instructions_dir = instructions_dir
        self.cwd = cwd
        self.pipelines: Dict[str, PipelineConfig] = {}
        # Replaced with a per-pipeline logger in register_pipeline().
        self.log = get_logger("relais")
        # The run's event sink while Pipeline.astream() drives it (None otherwise):
        # an async callable that receives text_delta / tool_use / step_done events
        # as they happen. One streaming run at a time per orchestrator.
        self._on_event = None
        # One engine per name, made on first use (see _engine()).
        self._engines: Dict[str, Engine] = {}

    def register_pipeline(self, config: PipelineConfig) -> None:
        """Register a pipeline configuration."""
        self.log = get_logger(f"relais.{config.name}")
        self.log.info("pipeline_registered", pipeline=config.name, steps=len(config.steps))
        self.pipelines[config.name] = config

    @staticmethod
    def _clone_agent(template: PipelineAgent) -> PipelineAgent:
        """Create a fresh runtime instance from a config template.

        Resets the step budget (steps_remaining) via __post_init__.
        """
        return PipelineAgent(
            name=template.name,
            tools=list(template.tools),
            steps=template.steps,
            max_turns=template.max_turns,
            model=template.model,
            thinking=template.thinking,
            permission_mode=template.permission_mode,
            effort=template.effort,
            resume=template.resume,
            env=template.env,
            builtin_tools=template.builtin_tools,
            engine=template.engine,
        )

    def start_pipeline(
        self,
        pipeline_name: str,
        initial_input: str,
        args: dict = None,
    ) -> str:
        """Start and run a pipeline to completion. Returns the run ID.

        Owns its own event loop (asyncio.run), so it is for SYNC callers only —
        from inside a running loop use start_pipeline_async().
        """
        return asyncio.run(self.start_pipeline_async(pipeline_name, initial_input, args))

    async def start_pipeline_async(
        self,
        pipeline_name: str,
        initial_input: str,
        args: dict = None,
    ) -> str:
        """The async twin of start_pipeline: awaits the run inside the CALLER's
        event loop (an async web framework, an async job worker). Every engine
        client the run opens lives and dies within this one awaiting task, which
        is what the anyio cancel-scope rule needs. Returns the run ID."""
        config = self.pipelines.get(pipeline_name)
        if not config:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        run_args = dict(args) if args else {}
        run_id = self.state_manager.create_pipeline_run(
            pipeline_name=pipeline_name,
            start_step=config.start_step,
            args=run_args,
        )
        # The run, not its input: the input is the host's content (a person's
        # question) — it is in the run database; the log says a run began.
        self.log.info(
            "run_started", run=run_id, pipeline=pipeline_name,
            start_step=config.start_step, input_chars=len(initial_input),
        )

        try:
            await self._execute_pipeline(run_id, config, initial_input, args=run_args)
        except Exception as e:
            self.log.error("run_failed", run=run_id, pipeline=pipeline_name, error=str(e))
            self.state_manager.complete_pipeline(run_id, status='failed')
            raise

        return run_id

    async def _execute_pipeline(
        self,
        run_id: str,
        config: PipelineConfig,
        initial_input: str,
        args: dict = None,
    ) -> None:
        """Run a pipeline start-to-finish in one process (fire-and-forget).

        Each agent is instantiated once and keeps its engine state across every
        step it runs. await_input is ignored here — a fire-and-forget run does
        not suspend.
        """
        run_agents: Dict[str, PipelineAgent] = {}
        try:
            await self._run_segment(
                run_id, config, config.start_step, initial_input, args,
                run_agents, conversational=False,
            )
            self.state_manager.complete_pipeline(run_id)
            self.log.info("run_completed", run=run_id, pipeline=config.name)
        finally:
            for agent in reversed(list(run_agents.values())):
                if agent.has_client():
                    await self._safe_close(agent)

    async def _run_segment(
        self,
        run_id: str,
        config: PipelineConfig,
        start_step: str,
        step_input: str,
        args: dict,
        run_agents: Dict[str, PipelineAgent],
        conversational: bool,
        step_images: list = None,
    ) -> dict:
        """Run steps from start_step until the pipeline ends or (in conversational
        mode) a step with await_input suspends the run.

        run_agents persists across calls so agents keep their engine state (and
        context) between conversational turns. The caller owns teardown.

        Returns a dict: {"suspended": bool, "next_step": str|None, "step": str|None,
        "output": dict|None}. suspended=True means a step with await_input ran (or,
        for a pure-park entry, parked) and the run is now waiting for input at
        "next_step".
        """
        current_step_name = start_step
        previous_result = None
        last_output = None
        last_step_name = None
        visits: Dict[str, int] = {}
        # Every step's latest output by name — what a step's `reads` draws on.
        outputs: Dict[str, dict] = {}

        while current_step_name:
            step = config.steps.get(current_step_name)
            if not step:
                raise ValueError(f"Step not found: {current_step_name}")

            # The visit cap (max_visits): past it, the step is skipped and its
            # default route taken — a bounded loop, decided here, not by an agent.
            visits[current_step_name] = visits.get(current_step_name, 0) + 1
            if step.max_visits is not None and visits[current_step_name] > step.max_visits:
                next_name = step.next.get("default") if isinstance(step.next, dict) else None
                self.log.info("visit_cap", run=run_id, step=current_step_name, visits=visits[current_step_name], next=next_name)
                current_step_name = next_name
                continue

            # Pure-park await step (no agent): it runs nothing; it just establishes
            # the wait. Only meaningful as an entry/standalone suspension point.
            if step.await_input and step.agent is None:
                if conversational and step_input is None:
                    next_name = step.resolve_next({})
                    return {"suspended": True, "next_step": next_name, "step": step.name, "output": None}
                # Input has arrived (or non-conversational): fall through to the next step.
                current_step_name = step.resolve_next({})
                continue

            # Route (pass-through) step: no LLM. Run hooks, merge their dicts into
            # routing data, and resolve the next step from it. For cheap, deterministic
            # branching (e.g. "is the session finished?") that needs no reasoning.
            if step.route:
                hook_results = await step.get_hook_data()
                routing_data = {}
                for hr in hook_results:
                    if isinstance(hr, dict):
                        routing_data.update(hr)
                next_step_name = step.resolve_next(routing_data)
                # The route, not the data: a step's output is the host's content (a
                # review, an answer, a person's figures) and belongs in the run database,
                # not in a log file with no retention. The keys say what shape came back.
                self.log.info(
                    "route", run=run_id, step=current_step_name,
                    keys=sorted(routing_data) if isinstance(routing_data, dict) else type(routing_data).__name__,
                    next=next_step_name,
                )
                outputs[current_step_name] = routing_data
                previous_result = routing_data
                current_step_name = next_step_name
                continue

            agent_name = step.agent.name
            agent = run_agents.get(agent_name)
            if agent is not None and agent.is_expired():
                if agent.has_client():
                    await self._safe_close(agent)
                agent = None
            if agent is None:
                agent = self._clone_agent(step.agent)
                run_agents[agent_name] = agent

            context = await self._build_step_context(
                step=step,
                previous_result=previous_result,
                initial_input=step_input,
                instructions_dir=Path(config.instructions_dir),
                args=args,
                earlier={name: outputs[name] for name in step.reads if name in outputs},
            )
            images = step_images
            step_input = None  # human/initial input is consumed by the first step only
            step_images = None  # images ride with that same first step only

            self.log.info(
                "step_start", run=run_id, step=current_step_name, agent=agent_name,
                tools=tool_names(step.tools), max_turns=agent.max_turns, model=agent.model or "opus",
                engine=agent.engine,
            )
            # DEBUG ONLY (spool writes info and up by default): the whole prompt, which
            # is the host's content — never raise relais's logger to debug in production.
            self.log.debug("step_context", run=run_id, step=current_step_name, context=context)

            await self._emit({"type": "step_start", "run": run_id, "step": current_step_name, "agent": agent_name})
            result = await self._execute_step(step=step, context=context, agent=agent, images=images)
            routing_data = result.routing_data or {}
            outputs[current_step_name] = routing_data
            next_step_name = step.resolve_next(routing_data)

            self.log.info(
                "step_done", run=run_id, step=current_step_name, agent=agent_name,
                turns=result.turns_used, stop=result.stop_reason,
                next=next_step_name or "END", usage=result.usage,
                output_keys=sorted(routing_data) if isinstance(routing_data, dict) else type(routing_data).__name__,
            )
            self.state_manager.update_pipeline_step(
                run_id=run_id,
                current_step=next_step_name or current_step_name,
                step_result={
                    'step': current_step_name,
                    'turns_used': result.turns_used,
                    'stop_reason': result.stop_reason,
                    'routing_data': routing_data,
                    'final_response': result.final_response,
                    'session_id': result.session_id,
                },
            )
            await self._emit({
                "type": "step_done", "run": run_id, "step": current_step_name,
                "routing_data": routing_data, "session_id": result.session_id,
                "turns": result.turns_used, "next": next_step_name, "usage": result.usage,
            })

            agent.consume_step()
            if agent.is_expired() and agent.has_client():
                self.log.debug("agent_expired", run=run_id, agent=agent_name)
                await self._safe_close(agent)

            last_output = routing_data
            last_step_name = current_step_name

            # Suspend AFTER this step if it awaits input (conversational mode).
            if conversational and step.await_input:
                return {"suspended": True, "next_step": next_step_name, "step": last_step_name, "output": routing_data}

            previous_result = routing_data
            current_step_name = next_step_name

        return {"suspended": False, "next_step": None, "step": last_step_name, "output": last_output}

    def _engine(self, agent: PipelineAgent) -> Engine:
        """The engine this agent's steps run on — one per name, made on first use."""
        if agent.engine not in self._engines:
            self._engines[agent.engine] = ENGINES[agent.engine](cwd=self.cwd, state_manager=self.state_manager)
        return self._engines[agent.engine]

    async def _safe_close(self, agent: PipelineAgent) -> None:
        """Release an agent's engine state through its engine, logging (not
        raising) on error. The engine clears agent.client either way."""
        try:
            await self._engine(agent).close(agent)
        except Exception as e:
            self.log.warning("close_error", agent=agent.name, error=str(e))

    async def _execute_step(
        self,
        step: PipelineStep,
        context: str,
        agent: PipelineAgent,
        images: list = None,
    ) -> StepExecutionResult:
        """Run one step on its agent's engine and capture routing data."""
        # Per-step tool scoping. This is the HARD constraint: every tool call goes
        # through the registry's wrapper, which consults is_tool_allowed() and
        # refuses any tool not in step.tools — whatever the engine offered the model.
        self.tool_registry.set_current_step(step.name, step.tools or [])

        outcome = await self._engine(agent).run_step(
            agent=agent, step=step, prompt=context, images=images or [],
            registry=self.tool_registry, emit=self._on_event, verbose=self._verbose,
        )
        for call in outcome.tool_calls:
            self.log.debug("tool_call", step=step.name, tool=call["tool"], input=call["input"])

        # Routing data is the output of the step's declared response tool.
        # response_tool is required on every step, so absence is a hard failure.
        # A step whose response tool WAS called has its answer: an error the engine
        # reports after that — the turn cap hit while the model carried on past
        # its answer — is logged, not raised (the agent said what it had to say).
        captured = self.tool_registry.get_tool_result(step.response_tool)
        if (outcome.is_error or outcome.error_kind) and not captured:
            raise StepError(
                step.name, agent.name, outcome.error_kind or "result_error",
                outcome.result_text or outcome.final_text,
            )
        if outcome.is_error or outcome.error_kind:
            self.log.warning(
                "step_error_after_answer", step=step.name, agent=agent.name,
                kind=outcome.error_kind, turns=outcome.turns_used,
            )
        if not captured:
            raise ResponseToolNotCalled(
                f"Step '{step.name}' requires response tool '{step.response_tool}' "
                f"but it was not called by the agent."
            )
        _, raw_result = captured
        if isinstance(raw_result, dict) and "content" in raw_result:
            routing_data = self._extract_from_mcp_content(raw_result["content"])
        else:
            routing_data = raw_result

        return StepExecutionResult(
            step_name=step.name,
            final_response=outcome.final_text,
            tool_results=outcome.tool_calls,
            turns_used=outcome.turns_used,
            stop_reason='error' if outcome.is_error else 'success',
            routing_data=routing_data,
            session_id=outcome.session_id,
            usage=outcome.usage,
        )

    async def _emit(self, event: dict) -> None:
        """Hand an event to the streaming sink, if a stream is listening."""
        if self._on_event is not None:
            await self._on_event(event)

    @property
    def _verbose(self) -> bool:
        # Resolved per-call from the active pipeline; cheap and avoids threading
        # config through _execute_step.
        return any(c.verbose for c in self.pipelines.values())

    async def _build_step_context(
        self,
        step: PipelineStep,
        previous_result: dict,
        initial_input: str,
        instructions_dir: Path,
        args: dict = None,
        earlier: dict = None,
    ) -> str:
        """Build the context prompt for a step. `earlier` is {step name: output}
        for the steps this one `reads` — shown after the previous step's."""
        sections = []

        if initial_input:
            sections.append(f"[User Input]\n{initial_input}")

        if args:
            sections.append(f"[Pipeline Args]\n{json.dumps(args, indent=2)}")

        if previous_result:
            sections.append(f"[Previous Step Output]\n{json.dumps(previous_result, indent=2)}")

        for name, output in (earlier or {}).items():
            sections.append(f"[Output of {name}]\n{json.dumps(output, indent=2)}")

        sections.append(f"[Current Step]\n{step.name}")

        hook_data = await step.get_hook_data()
        if hook_data:
            sections.append(f"[Hook Data]\n{json.dumps(hook_data, indent=2)}")

        if step.instruction_text is not None:
            sections.append(f"[Instructions]\n{step.instruction_text}")
        else:
            instruction_path = instructions_dir / f"{step.instruction}.md"
            if instruction_path.exists():
                sections.append(f"[Instructions]\n{instruction_path.read_text()}")
            else:
                self.log.warning("instruction_missing", step=step.name, path=str(instruction_path))

        names = tool_names(step.tools)
        if names:
            sections.append(
                f"[Available Tools]\n"
                f"For this step, you should use: {', '.join(names)}\n"
                f"Other tools are not available for this step and will be blocked if called."
            )

        if step.response_tool and self._on_event is not None:
            # A streamed run: the host shows the prose as it is written, so the
            # agent writes first and hands over last.
            sections.append(
                f"[Response Tool]\n"
                f"You MUST call '{step.response_tool}' to complete this step — AFTER your "
                f"answer, as your last act. Your prose IS shown to the person as you write "
                f"it; only the output of '{step.response_tool}' is captured and passed to "
                f"the next step."
            )
        elif step.response_tool:
            sections.append(
                f"[Response Tool]\n"
                f"You MUST call '{step.response_tool}' to complete this step. "
                f"All tools work normally, but only the output of '{step.response_tool}' "
                f"is captured and passed to the next step. Your text responses are not "
                f"visible — only the output of '{step.response_tool}' is used."
            )

        sections.append(PIPELINE_STEP_INSTRUCTION)
        return "\n\n".join(sections)

    def _extract_from_mcp_content(self, content_list: list) -> Optional[dict]:
        """Extract routing data from MCP content format (the first text block)."""
        for item in content_list:
            if isinstance(item, dict) and item.get("type") == "text":
                try:
                    return json.loads(item.get("text", "{}"))
                except json.JSONDecodeError:
                    return {"response": item.get("text")}
        return None

