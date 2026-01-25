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
    ToolResultBlock,
)

from .step import PipelineStep
from .tools import ToolRegistry
from .state import SQLiteStateManager
from .logging_config import get_logger

log = get_logger('executor')

# Grounding prompt injected when step.grounded=True to constrain model to pipeline data only
GROUNDING_PROMPT = """[Pipeline: {pipeline_name} | Step: {step_name}]

You are one step in a multi-step pipeline. Each step receives data from the previous step and passes results to the next.

Your ONLY information sources:
- [User Input]: The original request
- [Previous Step Output]: Data from the prior step (if any)
- Tool results: Responses from tools you call during THIS step

Do not use knowledge from your training data. If the provided data is insufficient, state what is missing rather than filling gaps with outside knowledge. You are a data processor in a chain - transform what you receive, nothing more."""

# Tool usage prompt injected into all pipeline steps
TOOL_USAGE_PROMPT = """[Tool Usage Rules]
IMPORTANT: Call each tool exactly ONCE per step. Do not make multiple parallel tool calls.

After calling a tool, STOP and wait for the result. The pipeline will route you to the next step based on the tool's output. Making multiple tool calls will cause only the last one to be processed - the others will be lost."""


@dataclass
class PipelineConfig:
    """Configuration for a pipeline.

    Attributes:
        name: Unique identifier for the pipeline
        steps: Dictionary of step definitions keyed by name
        start_step: Name of the first step to execute
        instructions_dir: Path to instruction markdown files
        model: Model for all steps (sonnet, opus, haiku)
        thinking: If True, enable extended thinking with max budget
        grounded: If True, all steps use grounded mode
        cwd: Working directory for file operations
        verbose: If True, print full step output to console
    """
    name: str
    steps: Dict[str, PipelineStep]
    start_step: str
    instructions_dir: str
    model: str = "sonnet"
    thinking: bool = False
    grounded: bool = False
    cwd: Optional[str] = None
    verbose: bool = False


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


@dataclass
class SubagentConfig:
    """Configuration for spawning a subagent."""
    step: PipelineStep
    context: str
    parent_pipeline_id: str


class PipelineOrchestrator:
    """Main orchestrator for pipeline execution using Claude Agent SDK.

    Uses ClaudeSDKClient for continuous sessions and query() for isolated
    subagent execution. The SDK handles max_turns internally.

    Execution flow:
    1. Build context for current step (instructions, args, previous result)
    2. Execute step via SDK (main session or isolated query)
    3. Extract routing data from tool results
    4. Apply routing rules to determine next step
    5. Persist state to SQLiteben@ben-Inspiron-3505:~/Documents/GitHub/tdl-ai$ uv run pipelines/eda.py
    6. Repeat until pipeline ends
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        state_manager: SQLiteStateManager,
        instructions_dir: Path,
        model: str = "sonnet",
        cwd: str = None
    ):
        """Initialize the orchestrator.

        Args:
            tool_registry: Registry with custom tools
            state_manager: SQLite state persistence
            instructions_dir: Path to instruction markdown files
            model: Model for all steps (sonnet, opus, haiku)
            cwd: Working directory for file operations
        """
        self.tool_registry = tool_registry
        self.state_manager = state_manager
        self.instructions_dir = instructions_dir
        self.model = model
        self.cwd = cwd
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.log = get_logger('orchestrator')
        self.context_log_path = Path("pipeline.context")

    def register_pipeline(self, config: PipelineConfig) -> None:
        """Register a pipeline configuration."""
        self.log.info(f"Registering pipeline '{config.name}' with {len(config.steps)} steps")
        self.log.debug(f"Pipeline steps: {list(config.steps.keys())}")
        self.pipelines[config.name] = config

    def start_pipeline(
        self,
        pipeline_name: str,
        initial_input: str,
        args: dict = None
    ) -> str:
        """Start a new pipeline run.

        Args:
            pipeline_name: Name of registered pipeline
            initial_input: Initial prompt/input
            args: Pipeline arguments

        Returns:
            Run ID (UUID)
        """
        return asyncio.run(self._start_pipeline_async(
            pipeline_name, initial_input, args
        ))

    async def _start_pipeline_async(
        self,
        pipeline_name: str,
        initial_input: str,
        args: dict = None
    ) -> str:
        """Async implementation of start_pipeline."""
        self.log.info(f"Starting pipeline '{pipeline_name}'")
        self.log.debug(f"Initial input: {initial_input[:200]}...")

        config = self.pipelines.get(pipeline_name)
        if not config:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        run_id = self.state_manager.create_pipeline_run(
            pipeline_name=pipeline_name,
            start_step=config.start_step,
            args=args
        )
        self.log.info(f"Created pipeline run: {run_id}")

        try:
            await self._execute_pipeline(run_id, config, initial_input, args or {})
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
        args: dict,
        start_from_step: str = None
    ) -> None:
        """Execute the pipeline loop.

        Main session steps share a single ClaudeSDKClient to maintain conversation
        context. Subagent steps get isolated client instances.
        """
        current_step_name = start_from_step or config.start_step
        previous_result = None
        step_count = 0

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

        # Create main session client that persists across non-subagent steps
        main_client = await self._create_main_client(mcp_server, config)

        try:
            while current_step_name:
                step_count += 1
                step = config.steps.get(current_step_name)
                if not step:
                    raise ValueError(f"Step not found: {current_step_name}")

                self.log.info(f"[{run_id}] === Step {step_count}: '{current_step_name}' ===")
                self.log.debug(f"[{run_id}] max_turns={step.max_turns}, tools={step.tools}, subagent={step.subagent}")

                # Build context prompt - always include initial_input so all steps can see original request
                context = self._build_step_context(
                    step=step,
                    args=args,
                    previous_result=previous_result,
                    initial_input=initial_input,
                    instructions_dir=Path(config.instructions_dir),
                    config=config
                )

                # Log full context for debugging
                self._log_step_context(current_step_name, context, step_count)

                # Execute step
                if step.subagent:
                    self.log.info(f"[{run_id}] Running as SUBAGENT (isolated)")
                    result = await self._execute_subagent_step(
                        step=step,
                        context=context,
                        mcp_server=mcp_server,
                        run_id=run_id,
                        config=config
                    )
                else:
                    self.log.info(f"[{run_id}] Running in MAIN session (persistent)")
                    result = await self._execute_main_step(
                        step=step,
                        context=context,
                        client=main_client,
                        config=config
                    )

                self.log.info(f"[{run_id}] Step completed - turns={result.turns_used}, reason={result.stop_reason}")

                # Determine next step
                routing_data = result.routing_data or {}
                next_step_name = step.resolve_next(routing_data)
                self.log.info(f"[{run_id}] Routing: '{current_step_name}' -> '{next_step_name or 'END'}'")

                # Persist state
                self.state_manager.update_pipeline_step(
                    run_id=run_id,
                    current_step=next_step_name or current_step_name,
                    conversation_history=[],  # SDK manages conversation
                    step_result={
                        'step': current_step_name,
                        'turns_used': result.turns_used,
                        'stop_reason': result.stop_reason,
                        'routing_data': routing_data,
                        'final_response': result.final_response[:1000]
                    }
                )

                previous_result = routing_data
                current_step_name = next_step_name

            self.log.info(f"[{run_id}] Pipeline completed after {step_count} steps")
            self.state_manager.complete_pipeline(run_id)
        finally:
            # Always disconnect main client
            await main_client.disconnect()

    async def _create_main_client(
        self,
        mcp_server,
        config: PipelineConfig
    ) -> ClaudeSDKClient:
        """Create and connect the main session client.

        This client persists across all non-subagent steps to maintain
        conversation context.

        Note: allowed_tools is set to ALL tools from non-subagent steps since
        we can't change it dynamically. Step instructions guide tool selection.
        """
        model = config.model or self.model

        # Collect all tools from non-subagent steps
        all_main_tools = []
        for step in config.steps.values():
            if not step.subagent and step.tools:
                all_main_tools.extend(step.tools)
        allowed_tools = self.tool_registry.get_allowed_tools(all_main_tools) if all_main_tools else None

        options_kwargs = {
            "model": model,
            "mcp_servers": {self.tool_registry.name: mcp_server},
            "allowed_tools": allowed_tools,
            "permission_mode": "acceptEdits",
            "cwd": config.cwd or self.cwd,
        }
        if config.thinking:
            options_kwargs["max_thinking_tokens"] = 60000

        options = ClaudeAgentOptions(**options_kwargs)
        client = ClaudeSDKClient(options=options)
        await client.connect()

        self.log.info(f"Created main session client: model={model}, allowed_tools={allowed_tools}")
        return client

    async def _execute_main_step(
        self,
        step: PipelineStep,
        context: str,
        client: ClaudeSDKClient,
        config: PipelineConfig
    ) -> StepExecutionResult:
        """Execute a step using the persistent main session client.

        The client maintains conversation history across steps, so the model
        remembers previous interactions.
        """
        # Set current step for tool validation
        self.tool_registry.set_current_step(step.name, step.tools or [])

        allowed_tools = self.tool_registry.get_allowed_tools(step.tools)

        self.log.info(f"SDK options: max_turns={step.max_turns}, allowed_tools={allowed_tools}")

        tool_calls_by_id = {}  # Track tool calls by ID for matching with results
        tool_results = []
        final_response = ""
        turns_used = 0
        session_id = None

        # Send query to existing client (maintains conversation history)
        await client.query(context)

        async for message in client.receive_response():
            self.log.debug(f"Received message type: {type(message).__name__}")

            # Check for tool results in any message with content
            if hasattr(message, 'content') and isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        # Match result to its tool call by ID
                        self.log.debug(f"ToolResultBlock attrs: {dir(block)}")
                        tool_use_id = getattr(block, 'tool_use_id', None)
                        self.log.debug(f"ToolResultBlock tool_use_id={tool_use_id}, content type={type(block.content).__name__}")
                        if tool_use_id and tool_use_id in tool_calls_by_id:
                            tool_calls_by_id[tool_use_id]["output"] = block.content
                            self.log.debug(f"Tool result for {tool_use_id}: {type(block.content).__name__}")
                        else:
                            self.log.debug(f"Tool result without matching call: tool_use_id={tool_use_id}, known_ids={list(tool_calls_by_id.keys())}")

            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_response = block.text
                        self.log.debug(f"Text response: {block.text[:200]}...")
                        if config.verbose:
                            print(f"\n[{step.name}] {block.text}")
                    elif isinstance(block, ToolUseBlock):
                        self.log.info(f"Tool call: {block.name} (id={block.id})")
                        if config.verbose:
                            print(f"\n[{step.name}] Tool: {block.name}")
                            print(f"  Input: {block.input}")
                        tool_call = {
                            "id": block.id,
                            "tool": block.name,
                            "input": block.input
                        }
                        tool_calls_by_id[block.id] = tool_call
                        tool_results.append(tool_call)

            elif isinstance(message, ResultMessage):
                turns_used = message.num_turns
                session_id = message.session_id
                self.log.info(f"Result: turns={turns_used}, error={message.is_error}")
                if config.verbose and message.usage:
                    u = message.usage
                    # Total input = non-cached + cache_creation + cache_read
                    input_tokens = u.get('input_tokens', 0) + u.get('cache_creation_input_tokens', 0) + u.get('cache_read_input_tokens', 0)
                    output_tokens = u.get('output_tokens', 0)
                    cache_read = u.get('cache_read_input_tokens', 0)
                    print(f"\n[{step.name}] Tokens: {input_tokens:,} in ({cache_read:,} cached) / {output_tokens:,} out")

        routing_data = self._extract_routing_data(tool_results)
        self.log.debug(f"Tool results for routing: {tool_results}")
        self.log.debug(f"Extracted routing data: {routing_data}")

        return StepExecutionResult(
            step_name=step.name,
            final_response=final_response,
            tool_results=tool_results,
            turns_used=turns_used,
            stop_reason='success' if not tool_results or final_response else 'max_turns',
            routing_data=routing_data,
            session_id=session_id
        )

    async def _execute_subagent_step(
        self,
        step: PipelineStep,
        context: str,
        mcp_server,
        run_id: str,
        config: PipelineConfig
    ) -> StepExecutionResult:
        """Execute a step as an isolated subagent.

        Uses a fresh ClaudeSDKClient instance which creates a new session,
        providing isolation from the main conversation.

        Note: We use ClaudeSDKClient instead of query() because query() does NOT
        support custom MCP tools - only ClaudeSDKClient does.
        """
        # Set current step for tool validation
        self.tool_registry.set_current_step(step.name, step.tools or [])

        subagent_id = str(uuid.uuid4())
        self.log.info(f"Spawning subagent {subagent_id} for '{step.name}'")

        # Log to database
        self.state_manager.log_subagent_spawn(
            parent_pipeline_id=run_id,
            subagent_id=subagent_id,
            step_name=step.name
        )

        # Use subagent-specific overrides or fall back to pipeline config
        model = step.subagent_model or config.model or self.model
        thinking = step.subagent_thinking if step.subagent_thinking is not None else config.thinking
        allowed_tools = self.tool_registry.get_allowed_tools(step.tools)

        options_kwargs = {
            "max_turns": step.max_turns,
            "model": model,
            "mcp_servers": {self.tool_registry.name: mcp_server},
            "allowed_tools": allowed_tools,
            "permission_mode": "acceptEdits",
            "cwd": config.cwd or self.cwd,
        }
        if thinking:
            options_kwargs["max_thinking_tokens"] = 60000

        options = ClaudeAgentOptions(**options_kwargs)

        self.log.info(f"Subagent options: max_turns={step.max_turns}, model={model}, allowed_tools={allowed_tools}, thinking={thinking}")

        tool_calls_by_id = {}  # Track tool calls by ID for matching with results
        tool_results = []
        final_response = ""
        turns_used = 0

        # New ClaudeSDKClient instance = fresh session = isolation guaranteed
        async with ClaudeSDKClient(options=options) as client:
            await client.query(context)

            async for message in client.receive_response():
                # Check for tool results in any message with content (comes as UserMessage)
                if hasattr(message, 'content') and isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            tool_use_id = getattr(block, 'tool_use_id', None)
                            if tool_use_id and tool_use_id in tool_calls_by_id:
                                tool_calls_by_id[tool_use_id]["output"] = block.content
                                self.log.debug(f"[subagent] Tool result for {tool_use_id}")

                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            final_response = block.text
                            if config.verbose:
                                print(f"\n[{step.name}:subagent] {block.text}")
                        elif isinstance(block, ToolUseBlock):
                            self.log.info(f"[subagent] Tool: {block.name} (id={block.id})")
                            if config.verbose:
                                print(f"\n[{step.name}:subagent] Tool: {block.name}")
                                print(f"  Input: {block.input}")
                            tool_call = {
                                "id": block.id,
                                "tool": block.name,
                                "input": block.input
                            }
                            tool_calls_by_id[block.id] = tool_call
                            tool_results.append(tool_call)

                elif isinstance(message, ResultMessage):
                    turns_used = message.num_turns
                    if config.verbose and message.usage:
                        u = message.usage
                        input_tokens = u.get('input_tokens', 0) + u.get('cache_creation_input_tokens', 0) + u.get('cache_read_input_tokens', 0)
                        output_tokens = u.get('output_tokens', 0)
                        cache_read = u.get('cache_read_input_tokens', 0)
                        print(f"\n[{step.name}:subagent] Tokens: {input_tokens:,} in ({cache_read:,} cached) / {output_tokens:,} out")

        result = StepExecutionResult(
            step_name=step.name,
            final_response=final_response,
            tool_results=tool_results,
            turns_used=turns_used,
            stop_reason='success',
            routing_data=self._extract_routing_data(tool_results)
        )

        # Log completion
        self.state_manager.log_subagent_complete(
            subagent_id=subagent_id,
            result={
                'final_response': final_response,
                'tool_results': tool_results,
                'routing_data': result.routing_data
            },
            turns_used=turns_used
        )

        return result

    def _build_step_context(
        self,
        step: PipelineStep,
        args: dict,
        previous_result: dict,
        initial_input: str,
        instructions_dir: Path,
        config: PipelineConfig
    ) -> str:
        """Build the context prompt for a step."""
        sections = []

        # Determine if grounded: subagent steps use subagent_grounded, others use config.grounded
        if step.subagent:
            is_grounded = step.subagent_grounded if step.subagent_grounded is not None else config.grounded
        else:
            is_grounded = config.grounded

        if is_grounded:
            sections.append(GROUNDING_PROMPT.format(
                pipeline_name=config.name,
                step_name=step.name
            ))

        # Always include tool usage rules
        sections.append(TOOL_USAGE_PROMPT)

        if initial_input:
            sections.append(f"[User Input]\n{initial_input}")

        if previous_result:
            sections.append(f"[Previous Step Output]\n{json.dumps(previous_result, indent=2)}")

        sections.append(f"[Current Step]\n{step.name}")

        if args:
            sections.append(f"[Pipeline Args]\n{json.dumps(args, indent=2)}")

        # Execute hooks
        hook_data = step.get_hook_data()
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

        return "\n\n".join(sections)

    def _extract_routing_data(self, tool_results: List[dict]) -> Optional[dict]:
        """Extract routing data from the last tool result that has successful output.

        Note: Not all tool calls may have output - the SDK only executes allowed tools.
        Tool calls to unauthorized tools return permission error messages which we skip.
        """
        if not tool_results:
            return None

        # Find the last tool result that has successful output (not a permission error)
        last_result = None
        for result in reversed(tool_results):
            if "output" in result:
                output = result.get("output")
                # Skip permission error messages from unauthorized tool calls
                if isinstance(output, str) and "requested permissions to use" in output:
                    self.log.debug(f"Skipping permission error output for routing: {result.get('tool')}")
                    continue
                # Skip tool validation errors (tools called from wrong pipeline step)
                if isinstance(output, dict) and "content" in output:
                    content = output.get("content", [])
                    if content and isinstance(content[0], dict):
                        text = content[0].get("text", "")
                        if "is only available in specific pipeline steps" in text:
                            self.log.debug(f"Skipping tool validation error for routing: {result.get('tool')}")
                            continue
                last_result = result
                break

        if not last_result:
            return None

        output = last_result.get("output")

        # Handle SDK tool result format
        if isinstance(output, str):
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                return {"response": output}
        elif isinstance(output, dict):
            # Check for MCP wrapper format {"content": [...]}
            if "content" in output and isinstance(output["content"], list):
                return self._extract_from_mcp_content(output["content"])
            return output
        elif isinstance(output, list):
            # MCP tool result content format
            return self._extract_from_mcp_content(output)
        return None

    def _extract_from_mcp_content(self, content_list: list) -> Optional[dict]:
        """Extract routing data from MCP content format."""
        for item in content_list:
            if isinstance(item, dict) and item.get("type") == "text":
                try:
                    return json.loads(item.get("text", "{}"))
                except json.JSONDecodeError:
                    return {"response": item.get("text")}
        return None

    def _log_step_context(self, step_name: str, context: str, step_num: int) -> None:
        """Log the full context being sent to a step for debugging.

        Args:
            step_name: Name of the step
            context: The full context string
            step_num: The step number in the pipeline
        """
        try:
            with open(self.context_log_path, 'a', encoding='utf-8') as f:
                separator = "=" * 80
                f.write(f"\n{separator}\n")
                f.write(f"STEP {step_num}: {step_name}\n")
                f.write(f"{separator}\n\n")
                f.write(context)
                f.write(f"\n\n{separator}\n\n")
        except Exception as e:
            self.log.warning(f"Failed to write context log: {e}")
