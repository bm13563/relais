"""Pipeline execution engine using Claude Agent SDK.

Uses ClaudeSDKClient for all step execution since custom MCP tools require it.
The SDK handles the agentic loop internally - we configure it via ClaudeAgentOptions.

Note: query() does NOT support custom tools - only ClaudeSDKClient does.
"""

from __future__ import annotations
import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
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
from .agent_state import AgentStateManager
from .logging_config import get_logger

log = get_logger('executor')

# Pipeline context injected at end of every step
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
        steps: Dictionary of step definitions keyed by name
        start_step: Name of the first step to execute
        instructions_dir: Path to instruction markdown files
        agents: Dictionary of PipelineAgent instances keyed by name
        cwd: Working directory for file operations
        verbose: If True, print full step output to console
    """
    name: str
    steps: Dict[str, PipelineStep]
    start_step: str
    instructions_dir: str
    agents: Dict[str, 'PipelineAgent'] = None
    cwd: Optional[str] = None
    verbose: bool = False

    def __post_init__(self):
        """Initialize agents dict if None."""
        if self.agents is None:
            self.agents = {}


@dataclass
class StepExecutionResult:
    """Result from executing a single pipeline step."""
    step_name: str
    final_response: str
    tool_results: List[dict]
    turns_used: int
    stop_reason: str  # 'success', 'max_turns', 'error'
    routing_data: Optional[dict] = None
    session_id: Optional[str] = None
    messages: Optional[List[dict]] = None  # Captured messages for context persistence


class PipelineOrchestrator:
    """Main orchestrator for pipeline execution using Claude Agent SDK.

    Uses ClaudeSDKClient for continuous sessions and query() for isolated
    subagent execution. The SDK handles max_turns internally.

    Execution flow:
    1. Build context for current step (instructions, previous result)
    2. Execute step via SDK (main session or isolated query)
    3. Extract routing data from tool results
    4. Apply routing rules to determine next step
    5. Persist state to SQLite
    6. Repeat until pipeline ends
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        state_manager: SQLiteStateManager,
        instructions_dir: Path,
        cwd: str = None
    ):
        """Initialize the orchestrator.

        Args:
            tool_registry: Registry with custom tools
            state_manager: SQLite state persistence
            instructions_dir: Path to instruction markdown files
            cwd: Working directory for file operations
        """
        self.tool_registry = tool_registry
        self.state_manager = state_manager
        self.instructions_dir = instructions_dir
        self.cwd = cwd
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.log = get_logger('orchestrator')
        self.context_log_path = Path("pipeline.context")

        # Create agent state manager (use same db path as state_manager with .agents suffix)
        agent_db_path = state_manager.db_path.replace('.db', '.agents.db')
        self.agent_state_manager = AgentStateManager.create(agent_db_path)
        self.agent_state_manager.initialize_schema()

    def register_pipeline(self, config: PipelineConfig) -> None:
        """Register a pipeline configuration."""
        self.log.info(f"Registering pipeline '{config.name}' with {len(config.steps)} steps")
        self.log.debug(f"Pipeline steps: {list(config.steps.keys())}")
        self.pipelines[config.name] = config

    def _get_or_create_agent(
        self,
        run_id: str,
        step: PipelineStep,
        config: PipelineConfig
    ) -> PipelineAgent:
        """Get or create an agent for a pipeline step.

        Args:
            run_id: UUID of the pipeline run
            step: The current step
            config: Pipeline configuration

        Returns:
            PipelineAgent instance for this step
        """
        # Get agent from step (required)
        if step.agent is None:
            raise ValueError(
                f"Step '{step.name}' is missing required 'agent' parameter. "
                f"Every step must have an explicit agent assigned."
            )

        agent_template = step.agent
        agent_name = agent_template.name

        # Try to load existing agent state from database
        agent = self.agent_state_manager.load_agent(run_id, agent_name)

        # If agent exists and is expired, clear it and create new one
        if agent and agent.is_expired():
            self.log.info(f"Agent '{agent_name}' expired, creating new instance")
            self.agent_state_manager.delete_agent(run_id, agent_name)
            agent = None

        # If no agent state exists, clone from template
        if not agent:
            agent = PipelineAgent(
                name=agent_template.name,
                tools=agent_template.tools,  # Copy tools from template
                steps=agent_template.steps,
                max_turns=agent_template.max_turns,
                model=agent_template.model,
                thinking=agent_template.thinking,
            )
            self.log.info(f"Created agent '{agent_name}' from template: steps={agent.steps}, max_turns={agent.max_turns}, tools={len(agent.tools)}")

            # Save new agent
            self.agent_state_manager.save_agent(run_id, agent)

        return agent

    def start_pipeline(
        self,
        pipeline_name: str,
        initial_input: str,
        args: dict = None,
        session: str = None
    ) -> str:
        """Start a new pipeline run or continue an existing session.

        Args:
            pipeline_name: Name of registered pipeline
            initial_input: Initial prompt/input
            args: Pipeline arguments
            session: Optional session name for debug mode. When provided:
                - Checks for an existing active session with this name
                - If found, resumes from where it left off
                - If not found (or completed), starts a new run
                - Pipeline pauses after each step completes

        Returns:
            Run ID (UUID)
        """
        return asyncio.run(self._start_pipeline_async(
            pipeline_name, initial_input, args, session
        ))

    async def _start_pipeline_async(
        self,
        pipeline_name: str,
        initial_input: str,
        args: dict = None,
        session: str = None
    ) -> str:
        """Async implementation of start_pipeline."""
        config = self.pipelines.get(pipeline_name)
        if not config:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        # Check for existing active session
        existing_run = None
        start_from_step = None
        if session:
            existing_run = self.state_manager.get_active_session(pipeline_name, session)
            if existing_run:
                self.log.info(f"Resuming session '{session}' from step '{existing_run.current_step}'")
                run_id = existing_run.id
                start_from_step = existing_run.current_step
                args = existing_run.args  # Use stored args
                initial_input = args.get('_initial_input', initial_input)  # Restore original input
                # Mark as running again
                self.state_manager.resume_pipeline(run_id)
            else:
                self.log.info(f"Starting new session '{session}' for pipeline '{pipeline_name}'")

        if not existing_run:
            self.log.info(f"Starting pipeline '{pipeline_name}'")
            self.log.debug(f"Initial input: {initial_input[:200]}...")

            # Store initial_input in args for session resume
            run_args = dict(args) if args else {}
            if session:
                run_args['_initial_input'] = initial_input

            run_id = self.state_manager.create_pipeline_run(
                pipeline_name=pipeline_name,
                start_step=config.start_step,
                args=run_args,
                session=session
            )
            args = run_args  # Use the args we stored
            self.log.info(f"Created pipeline run: {run_id}")

        try:
            await self._execute_pipeline(
                run_id, config, initial_input,
                start_from_step=start_from_step,
                session=session
            )
        except Exception as e:
            self.log.error(f"Pipeline execution failed: {e}")
            self.state_manager.complete_pipeline(run_id, status='failed')
            raise

        return run_id

    async def _execute_pipeline(
        self,
        run_id: str,
        config: PipelineConfig,
        initial_input: str,
        start_from_step: str = None,
        session: str = None
    ) -> None:
        """Execute the pipeline loop.

        Main session steps share a single ClaudeSDKClient to maintain conversation
        context. Subagent steps get isolated client instances.

        When session is provided, pipeline runs in debug mode:
        - Executes one step at a time
        - Pauses after each step completes
        - Call run() again with same session to continue
        """
        current_step_name = start_from_step or config.start_step
        previous_result = None
        step_count = 0
        accumulated_messages = []  # Accumulate messages across main session steps
        run_agents: Dict[str, PipelineAgent] = {}  # Track agents within this run (preserves clients)

        # In session mode, load previous result and conversation history from stored state
        if session and start_from_step:
            run_state = self.state_manager.get_pipeline_run(run_id)
            if run_state:
                # Load accumulated conversation history
                if run_state.conversation_history:
                    accumulated_messages = run_state.conversation_history
                    self.log.debug(f"Loaded {len(accumulated_messages)} messages from previous session")
                # Get the last step's routing data as previous_result
                if run_state.step_results:
                    step_results = run_state.step_results
                    if step_results:
                        last_step = list(step_results.keys())[-1] if step_results else None
                        if last_step and 'routing_data' in step_results[last_step]:
                            previous_result = step_results[last_step]['routing_data']
                            self.log.debug(f"Loaded previous result from step '{last_step}': {previous_result}")

        # Clear/create context log file for this pipeline run
        try:
            with open(self.context_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Pipeline Context Log - Run ID: {run_id}\n")
                f.write(f"Pipeline: {config.name}\n")
                f.write(f"Started at: {datetime.now().isoformat()}\n")
        except Exception as e:
            self.log.warning(f"Failed to initialize context log: {e}")

        # Create MCP server with registered tools
        mcp_server = self.tool_registry.create_mcp_server()

        self.log.info(f"[{run_id}] Beginning pipeline execution from '{current_step_name}'")

        try:
            while current_step_name:
                step_count += 1
                step = config.steps.get(current_step_name)
                if not step:
                    raise ValueError(f"Step not found: {current_step_name}")

                self.log.info(f"[{run_id}] === Step {step_count}: '{current_step_name}' ===")

                # Get the agent for this step - check run_agents first to preserve client
                agent_name = step.agent.name
                if agent_name in run_agents:
                    agent = run_agents[agent_name]
                    # Check if cached agent has expired - if so, create a fresh one
                    if agent.is_expired():
                        self.log.info(f"Agent '{agent_name}' expired, creating fresh instance")
                        del run_agents[agent_name]
                        agent = self._get_or_create_agent(run_id, step, config)
                        run_agents[agent_name] = agent
                    else:
                        self.log.debug(f"Reusing agent '{agent_name}' from current run")
                else:
                    agent = self._get_or_create_agent(run_id, step, config)
                    run_agents[agent_name] = agent

                self.log.debug(f"[{run_id}] max_turns={agent.max_turns}, tools={step.tools}, agent={agent_name}")

                # Build context prompt - always include initial_input so all steps can see original request
                # In debug mode resume, inject previous conversation history
                if session and agent.is_persistent():
                    prev_msgs = accumulated_messages
                elif session and agent.conversation_history:
                    # For isolated agents with steps > 1, use their stored conversation history
                    prev_msgs = agent.conversation_history
                else:
                    prev_msgs = None
                    
                context = await self._build_step_context(
                    step=step,
                    previous_result=previous_result,
                    initial_input=initial_input,
                    instructions_dir=Path(config.instructions_dir),
                    config=config,
                    previous_messages=prev_msgs
                )

                # Log full context for debugging
                agent_step_info = None
                if agent.steps is not None:
                    current_agent_step = agent.steps - agent.steps_remaining + 1
                    agent_step_info = f"{current_agent_step}/{agent.steps}"
                self._log_step_context(current_step_name, context, step_count, agent_name, agent_step_info)

                # Execute step
                self.log.info(f"[{run_id}] Running with agent '{agent_name}' (steps={agent.steps})")
                result = await self._execute_step(
                    step=step,
                    context=context,
                    mcp_server=mcp_server,
                    run_id=run_id,
                    config=config,
                    agent=agent
                )

                self.log.info(f"[{run_id}] Step completed - turns={result.turns_used}, reason={result.stop_reason}")

                # Accumulate messages for debug mode resume (persistent agents use accumulated_messages)
                if agent.is_persistent() and result.messages:
                    accumulated_messages.extend(result.messages)
                    self.log.debug(f"Accumulated {len(result.messages)} messages, total now {len(accumulated_messages)}")

                # Determine next step
                routing_data = result.routing_data or {}
                next_step_name = step.resolve_next(routing_data)
                self.log.info(f"[{run_id}] Routing: '{current_step_name}' -> '{next_step_name or 'END'}'")

                # Persist state
                self.state_manager.update_pipeline_step(
                    run_id=run_id,
                    current_step=next_step_name or current_step_name,
                    conversation_history=accumulated_messages,  # Persist for debug mode resume
                    step_result={
                        'step': current_step_name,
                        'turns_used': result.turns_used,
                        'stop_reason': result.stop_reason,
                        'routing_data': routing_data,
                        'final_response': result.final_response
                    }
                )

                # Session mode: pause after each step
                if session and next_step_name:
                    self.log.info(f"[{run_id}] Session mode: pausing after '{current_step_name}', next step is '{next_step_name}'")
                    self.state_manager.pause_pipeline(run_id)
                    print(f"\n{'='*60}")
                    print(f"SESSION '{session}' PAUSED")
                    print(f"  Completed: {current_step_name}")
                    print(f"  Next step: {next_step_name}")
                    print(f"  Run again with session='{session}' to continue")
                    print(f"{'='*60}\n")
                    return  # Exit without completing - will resume later

                previous_result = routing_data
                current_step_name = next_step_name

            self.log.info(f"[{run_id}] Pipeline completed after {step_count} steps")
            self.state_manager.complete_pipeline(run_id)
            if session:
                print(f"\n{'='*60}")
                print(f"SESSION '{session}' COMPLETED")
                print(f"  Total steps: {step_count}")
                print(f"{'='*60}\n")
        finally:
            # Disconnect all agents that have clients
            for agent in run_agents.values():
                if agent.has_client():
                    await agent.disconnect()

    async def _execute_step(
        self,
        step: PipelineStep,
        context: str,
        mcp_server,
        run_id: str,
        config: PipelineConfig,
        agent: PipelineAgent
    ) -> StepExecutionResult:
        """Execute a pipeline step with the given agent.

        Handles both persistent (steps=None) and limited (steps=N) agents:
        - Creates/reuses client stored on agent.client
        - Processes response and captures messages to agent.conversation_history
        - Manages agent lifecycle (consume_step, expiration) for non-persistent agents
        - Persists agent state to database
        """
        # Set current step for tool validation (soft constraint - MCP blocks unauthorized tools)
        self.tool_registry.set_current_step(step.name, step.tools or [])

        # Build client options from agent settings
        model = agent.model or "opus"
        thinking = agent.thinking or False
        # For persistent agents, --allowedTools is set once at CLI startup and
        # can't change between steps, so use agent.tools (the full set).
        # Per-step scoping is enforced by the MCP wrapper's is_tool_allowed().
        # For non-persistent agents, use step.tools for per-step scoping.
        if agent.is_persistent():
            allowed_tools = self.tool_registry.get_allowed_tools(agent.tools)
        elif step.tools:
            allowed_tools = self.tool_registry.get_allowed_tools(step.tools)
        else:
            allowed_tools = self.tool_registry.get_allowed_tools(agent.tools)

        # Check if agent already has a client - reuse it across steps
        reusing_client = agent.has_client()

        agent_instance_id = None

        if reusing_client:
            self.log.info(f"Continuing agent '{agent.name}' for '{step.name}' (steps_remaining={agent.steps_remaining})")
        else:
            agent_instance_id = str(uuid.uuid4())
            self.log.info(f"Starting agent '{agent.name}' ({agent_instance_id}) for '{step.name}'")

            # Log to database
            self.state_manager.log_subagent_spawn(
                parent_pipeline_id=run_id,
                subagent_id=agent_instance_id,
                step_name=step.name
            )

        options_kwargs = {
            "max_turns": agent.max_turns,
            "model": model,
            "mcp_servers": {self.tool_registry.name: mcp_server},
            "allowed_tools": allowed_tools,
            "permission_mode": "acceptEdits",
            "cwd": config.cwd or self.cwd,
        }
        if thinking:
            options_kwargs["max_thinking_tokens"] = 60000

        options = ClaudeAgentOptions(**options_kwargs)

        self.log.info(f"Step options: max_turns={agent.max_turns}, model={model}, allowed_tools={allowed_tools}")

        # Initialize tracking
        tool_results = []
        final_response = ""
        turns_used = 0
        is_error = False
        captured_messages = []

        # Record the user query
        captured_messages.append({"role": "user", "content": context})

        # Get or create client
        if reusing_client:
            client = agent.client
            await client.query(context)
        else:
            client = ClaudeSDKClient(options=options)
            await client.connect()
            agent.set_client(client)
            await client.query(context)

        try:
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    msg_content = []  # For captured_messages (debug mode conversation history)
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            final_response = block.text  # For logging
                            msg_content.append({"type": "text", "text": block.text})
                            if config.verbose:
                                print(f"\n[{step.name}] {block.text}")
                        elif isinstance(block, ToolUseBlock):
                            self.log.info(f"Tool call: {block.name} (id={block.id})")
                            msg_content.append({"type": "tool_use", "tool": block.name, "input": block.input})
                            if config.verbose:
                                print(f"\n[{step.name}] Tool: {block.name}")
                                print(f"  Input: {block.input}")
                            tool_call = {
                                "id": block.id,
                                "tool": block.name,
                                "input": block.input
                            }
                            tool_results.append(tool_call)  # For logging
                    if msg_content:
                        captured_messages.append({"role": "assistant", "content": msg_content})

                elif isinstance(message, ResultMessage):
                    turns_used = message.num_turns  # For logging
                    is_error = message.is_error
                    self.log.info(f"Result: turns={turns_used}, error={is_error}")
                    if config.verbose and message.usage:
                        u = message.usage
                        input_tokens = u.get('input_tokens', 0) + u.get('cache_creation_input_tokens', 0) + u.get('cache_read_input_tokens', 0)
                        output_tokens = u.get('output_tokens', 0)
                        cache_read = u.get('cache_read_input_tokens', 0)
                        print(f"\n[{step.name}] Tokens: {input_tokens:,} in ({cache_read:,} cached) / {output_tokens:,} out")

        finally:
            # Add captured messages to agent's conversation history
            agent.add_messages(captured_messages)

            # Handle lifecycle for non-persistent agents
            if not agent.is_persistent():
                agent.consume_step()
                self.log.debug(f"Agent '{agent.name}' consumed step, remaining={agent.steps_remaining}")

                if agent.is_expired():
                    self.log.info(f"Agent '{agent.name}' expired, disconnecting client")
                    await agent.disconnect()
                    agent.client = None
                    self.agent_state_manager.delete_agent(run_id, agent.name)
                else:
                    self.agent_state_manager.save_agent(run_id, agent)
            else:
                # Persistent agents still save state (for debug mode resume)
                self.agent_state_manager.save_agent(run_id, agent)

        # Get routing data from response tool
        if step.response_tool:
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
        else:
            routing_data = self._get_routing_data_from_registry()

        result = StepExecutionResult(
            step_name=step.name,
            final_response=final_response,
            tool_results=tool_results,
            turns_used=turns_used,
            stop_reason='error' if is_error else 'success',
            routing_data=routing_data,
            messages=captured_messages
        )

        # Log completion if we started a new agent instance
        if agent_instance_id:
            self.state_manager.log_subagent_complete(
                subagent_id=agent_instance_id,
                result={
                    'final_response': final_response,
                    'tool_results': tool_results,
                    'routing_data': result.routing_data
                },
                turns_used=turns_used
            )

        return result

    async def _build_step_context(
        self,
        step: PipelineStep,
        previous_result: dict,
        initial_input: str,
        instructions_dir: Path,
        config: PipelineConfig,
        previous_messages: List[dict] = None
    ) -> str:
        """Build the context prompt for a step."""
        sections = []

        # Inject previous conversation history (for debug mode resume)
        if previous_messages:
            conversation_text = self._format_conversation_history(previous_messages)
            if conversation_text:
                sections.append(f"[Previous Conversation]\n{conversation_text}")

        if initial_input:
            sections.append(f"[User Input]\n{initial_input}")

        if previous_result:
            sections.append(f"[Previous Step Output]\n{json.dumps(previous_result, indent=2)}")

        sections.append(f"[Current Step]\n{step.name}")

        # Execute hooks
        hook_data = await step.get_hook_data()
        if hook_data:
            self.log.debug(f"Hook data: {len(hook_data)} items")
            sections.append(f"[Hook Data]\n{json.dumps(hook_data, indent=2)}")

        # Load instruction
        instruction_path = instructions_dir / f"{step.instruction}.md"
        if instruction_path.exists():
            instruction = instruction_path.read_text()
            sections.append(f"[Instructions]\n{instruction}")
        else:
            self.log.warning(f"Instruction not found: {instruction_path}")

        # Add available tools section (soft constraint - tells agent what tools to use)
        if step.tools:
            tool_names = []
            for t in step.tools:
                if callable(t) and hasattr(t, '_tool_name'):
                    tool_names.append(t._tool_name)
                elif isinstance(t, str):
                    tool_names.append(t)
            if tool_names:
                tools_text = ", ".join(tool_names)
                sections.append(
                    f"[Available Tools]\n"
                    f"For this step, you should use: {tools_text}\n"
                    f"Other tools are not available for this step and will be blocked if called."
                )

        # Inject response tool requirement
        if step.response_tool:
            sections.append(
                f"[Response Tool]\n"
                f"You MUST call '{step.response_tool}' to complete this step. "
                f"All tools work normally, but only the output of '{step.response_tool}' "
                f"is captured and passed to the next step. Your text responses are not "
                f"visible — only the output of '{step.response_tool}' is used."
            )

        # Always end with pipeline step instruction
        sections.append(PIPELINE_STEP_INSTRUCTION)

        return "\n\n".join(sections)

    def _get_routing_data_from_registry(self) -> Optional[dict]:
        """Get routing data from MCP tool capture.

        This is the preferred method - the MCP wrapper captures tool output directly.
        """
        captured = self.tool_registry.get_last_tool_result()
        if not captured or not isinstance(captured, tuple) or len(captured) != 2:
            return None

        tool_name, result = captured
        self.log.debug(f"Got routing data from registry for {tool_name}: {str(result)[:200]}")

        # Handle MCP wrapper format {"content": [...]}
        if isinstance(result, dict) and "content" in result:
            return self._extract_from_mcp_content(result["content"])
        return result

    def _extract_from_mcp_content(self, content_list: list) -> Optional[dict]:
        """Extract routing data from MCP content format."""
        for item in content_list:
            if isinstance(item, dict) and item.get("type") == "text":
                try:
                    return json.loads(item.get("text", "{}"))
                except json.JSONDecodeError:
                    return {"response": item.get("text")}
        return None

    def _format_conversation_history(self, messages: List[dict]) -> str:
        """Format captured messages as readable conversation history.

        Args:
            messages: List of captured messages with role and content

        Returns:
            Formatted conversation text
        """
        if not messages:
            return ""

        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                # User messages are context strings - pass through without truncation
                # Caller (pipeline) can handle truncation if needed
                if isinstance(content, str):
                    lines.append(f"[User Query]\n{content}")
            elif role == "assistant":
                # Assistant messages have structured content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                lines.append(f"[Assistant]\n{block.get('text', '')}")
                            elif block.get("type") == "tool_use":
                                tool = block.get("tool", "unknown")
                                inp = block.get("input", {})
                                lines.append(f"[Tool Call: {tool}]\n{json.dumps(inp, indent=2)}")
                elif isinstance(content, str):
                    lines.append(f"[Assistant]\n{content}")

        return "\n\n".join(lines)

    def _log_step_context(self, step_name: str, context: str, step_num: int, agent_name: str, agent_step: str = None) -> None:
        """Log the full context being sent to a step for debugging.

        Args:
            step_name: Name of the step
            context: The full context string
            step_num: The step number in the pipeline
            agent_name: Name of the agent executing this step
            agent_step: Agent step info like "2/3" for limited agents
        """
        try:
            with open(self.context_log_path, 'a', encoding='utf-8') as f:
                separator = "=" * 80
                f.write(f"\n{separator}\n")
                agent_info = f"{agent_name} {agent_step}" if agent_step else agent_name
                f.write(f"STEP {step_num}: {step_name} ({agent_info})\n")
                f.write(f"{separator}\n\n")
                f.write(context)
                f.write(f"\n\n{separator}\n\n")
        except Exception as e:
            self.log.warning(f"Failed to write context log: {e}")
