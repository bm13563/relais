"""Pipeline execution engine using the Claude Agent SDK.

Each agent connects one live ClaudeSDKClient on its first step and keeps it for
the whole pipeline run, so its conversation context lives in the SDK client. A
run executes start-to-finish in one process; results are persisted to SQLite for
after-the-fact inspection. Step-level detail is logged to spool as structured
events (see `spool` — query the JSONL stream to debug a run).
"""

from __future__ import annotations
import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from .step import PipelineStep
from .tools import ToolRegistry
from .state import SQLiteStateManager
from .agent import PipelineAgent
from .logging_config import get_logger


# Pipeline context appended to every step's prompt.
PIPELINE_STEP_INSTRUCTION = """[Pipeline Context]
You are one step in a multi-step pipeline. Focus your thinking on the specific task for this step. Don't worry about any future steps, they will be handled by other agents, focus all of your thinking on this step.

Do not make multiple parallel tool calls — call tools one at a time.

The next step in the pipeline will only see the output of your response tool. So the entire focus of your effort should go into that call. If you have 10 available turns, use the response tool after 2, then continue reasoning for another 8 turns, that will be 8 turns of context wasted. Call your response tool once you have completed your reasoning."""


class ResponseToolNotCalled(Exception):
    """Raised when a step's declared response_tool was not called by the agent."""
    pass


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


class PipelineOrchestrator:
    """Runs pipelines step by step using the Claude Agent SDK.

    Execution flow:
    1. Build the context prompt for the current step.
    2. Run the step on its agent's live client (created once, reused).
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

    def register_pipeline(self, config: PipelineConfig) -> None:
        """Register a pipeline configuration."""
        self.log = get_logger(f"relais.{config.name}")
        self.log.info("pipeline_registered", pipeline=config.name, steps=len(config.steps))
        self.pipelines[config.name] = config

    @staticmethod
    def _clone_agent(template: PipelineAgent) -> PipelineAgent:
        """Create a fresh runtime agent from a config template."""
        return PipelineAgent(
            name=template.name,
            tools=list(template.tools),
            max_turns=template.max_turns,
            model=template.model,
            thinking=template.thinking,
        )

    def start_pipeline(
        self,
        pipeline_name: str,
        initial_input: str,
        args: dict = None,
    ) -> str:
        """Start and run a pipeline to completion. Returns the run ID."""
        return asyncio.run(self._start_pipeline_async(pipeline_name, initial_input, args))

    async def _start_pipeline_async(
        self,
        pipeline_name: str,
        initial_input: str,
        args: dict = None,
    ) -> str:
        config = self.pipelines.get(pipeline_name)
        if not config:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        run_args = dict(args) if args else {}
        run_id = self.state_manager.create_pipeline_run(
            pipeline_name=pipeline_name,
            start_step=config.start_step,
            args=run_args,
        )
        self.log.info(
            "run_started", run=run_id, pipeline=pipeline_name,
            start_step=config.start_step, input=initial_input[:200],
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
        """Run the pipeline loop start-to-finish in one process.

        Each agent is instantiated once (cloned from its config template) and
        keeps its live client across every step it runs.
        """
        current_step_name = config.start_step
        previous_result = None
        step_count = 0
        run_agents: Dict[str, PipelineAgent] = {}  # name -> live agent (preserves client)

        mcp_server = self.tool_registry.create_mcp_server()

        try:
            while current_step_name:
                step_count += 1
                step = config.steps.get(current_step_name)
                if not step:
                    raise ValueError(f"Step not found: {current_step_name}")

                # One live agent per name for the whole run.
                agent_name = step.agent.name
                agent = run_agents.get(agent_name)
                if agent is None:
                    agent = self._clone_agent(step.agent)
                    run_agents[agent_name] = agent

                context = await self._build_step_context(
                    step=step,
                    previous_result=previous_result,
                    initial_input=initial_input,
                    instructions_dir=Path(config.instructions_dir),
                    args=args,
                )

                self.log.info(
                    "step_start", run=run_id, n=step_count, step=current_step_name,
                    agent=agent_name, tools=_tool_names(step.tools),
                    max_turns=agent.max_turns, model=agent.model or "opus",
                )
                self.log.debug("step_context", run=run_id, step=current_step_name, context=context)

                result = await self._execute_step(
                    step=step, context=context, mcp_server=mcp_server, agent=agent,
                )

                routing_data = result.routing_data or {}
                next_step_name = step.resolve_next(routing_data)

                self.log.info(
                    "step_done", run=run_id, step=current_step_name, agent=agent_name,
                    turns=result.turns_used, stop=result.stop_reason,
                    next=next_step_name or "END", routing_data=routing_data,
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
                    },
                )

                previous_result = routing_data
                current_step_name = next_step_name

            self.state_manager.complete_pipeline(run_id)
            self.log.info("run_completed", run=run_id, pipeline=config.name, steps=step_count)
        finally:
            # Disconnect clients newest-first. Each ClaudeSDKClient enters an
            # anyio task scope on connect(); anyio requires those scopes to be
            # exited in reverse order. Teardown errors are logged, not raised.
            for agent in reversed(list(run_agents.values())):
                if agent.has_client():
                    try:
                        await agent.disconnect()
                    except Exception as e:
                        self.log.warning("disconnect_error", agent=agent.name, error=str(e))
                    finally:
                        agent.client = None

    async def _execute_step(
        self,
        step: PipelineStep,
        context: str,
        mcp_server,
        agent: PipelineAgent,
    ) -> StepExecutionResult:
        """Run one step on its agent's live client and capture routing data."""
        # Per-step tool scoping. This is the HARD constraint: the MCP wrapper
        # consults is_tool_allowed() on every call and refuses any tool not in
        # step.tools, regardless of the SDK's own allowed_tools list.
        self.tool_registry.set_current_step(step.name, step.tools or [])

        model = agent.model or "opus"
        thinking = agent.thinking or False

        reusing_client = agent.has_client()
        if not reusing_client:
            # allowed_tools is the SDK's per-client filter, set once at connect().
            # Use the agent's full tool set; per-step scoping is enforced by the
            # wrapper above.
            allowed_tools = self.tool_registry.get_allowed_tools(agent.tools)
            options_kwargs = {
                "max_turns": agent.max_turns,
                "model": model,
                "mcp_servers": {self.tool_registry.name: mcp_server},
                "allowed_tools": allowed_tools,
                "permission_mode": "acceptEdits",
                "cwd": self.cwd,
            }
            if thinking:
                options_kwargs["max_thinking_tokens"] = 60000
            client = ClaudeSDKClient(options=ClaudeAgentOptions(**options_kwargs))
            await client.connect()
            agent.set_client(client)
        else:
            client = agent.client

        await client.query(context)

        tool_results = []
        final_response = ""
        turns_used = 0
        is_error = False

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_response = block.text
                        if self._verbose:
                            print(f"\n[{step.name}] {block.text}")
                    elif isinstance(block, ToolUseBlock):
                        self.log.debug("tool_call", step=step.name, tool=block.name, input=block.input)
                        tool_results.append({"id": block.id, "tool": block.name, "input": block.input})
                        if self._verbose:
                            print(f"\n[{step.name}] Tool: {block.name}  {block.input}")
            elif isinstance(message, ResultMessage):
                turns_used = message.num_turns
                is_error = message.is_error

        # Routing data is the output of the step's declared response tool.
        # response_tool is required on every step, so absence is a hard failure.
        captured = self.tool_registry.get_tool_result(step.response_tool)
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
            final_response=final_response,
            tool_results=tool_results,
            turns_used=turns_used,
            stop_reason='error' if is_error else 'success',
            routing_data=routing_data,
        )

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
    ) -> str:
        """Build the context prompt for a step."""
        sections = []

        if initial_input:
            sections.append(f"[User Input]\n{initial_input}")

        if args:
            sections.append(f"[Pipeline Args]\n{json.dumps(args, indent=2)}")

        if previous_result:
            sections.append(f"[Previous Step Output]\n{json.dumps(previous_result, indent=2)}")

        sections.append(f"[Current Step]\n{step.name}")

        hook_data = await step.get_hook_data()
        if hook_data:
            sections.append(f"[Hook Data]\n{json.dumps(hook_data, indent=2)}")

        instruction_path = instructions_dir / f"{step.instruction}.md"
        if instruction_path.exists():
            sections.append(f"[Instructions]\n{instruction_path.read_text()}")
        else:
            self.log.warning("instruction_missing", step=step.name, path=str(instruction_path))

        tool_names = _tool_names(step.tools)
        if tool_names:
            sections.append(
                f"[Available Tools]\n"
                f"For this step, you should use: {', '.join(tool_names)}\n"
                f"Other tools are not available for this step and will be blocked if called."
            )

        if step.response_tool:
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


def _tool_names(tools) -> List[str]:
    """Normalize a tools list (strings or @tool functions) to names."""
    names = []
    for t in tools or []:
        if callable(t) and hasattr(t, '_tool_name'):
            names.append(t._tool_name)
        elif isinstance(t, str):
            names.append(t)
    return names
