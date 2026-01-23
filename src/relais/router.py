"""Command router for pipeline execution.

This module provides command-based routing to start pipelines,
either from CLI input or programmatic invocation.
"""

from __future__ import annotations
import json
import sys
from typing import Callable, Dict, Optional

from .utils import parse_command
from .pipeline import Pipeline


class PipelineRouter:
    """Routes commands to pipelines.

    This class parses commands from user input and starts the appropriate
    pipeline. It can be used as a CLI entry point or programmatically.

    Usage:
        router = PipelineRouter(command_prefix="#")

        # Register pipeline factories
        router.register("analyze", create_analyze_pipeline)
        router.register("learn", create_learn_pipeline)

        # Handle a command
        result = router.handle_prompt("#analyze some input")

        # Or run as CLI
        router.run()
    """

    def __init__(self, command_prefix: str = "#"):
        """Initialize the router.

        Args:
            command_prefix: The prefix that triggers commands (default: "#")
        """
        self.command_prefix = command_prefix
        self.pipelines: Dict[str, Callable[..., Pipeline]] = {}

    def register(self, command: str, create_fn: Callable[..., Pipeline]) -> None:
        """Register a pipeline creation function for a command.

        Args:
            command: The command name (without prefix)
            create_fn: A function that takes optional args and returns a Pipeline
        """
        self.pipelines[command] = create_fn

    def start(self, name: str, initial_input: str = "", args: dict = None) -> Optional[str]:
        """Start a pipeline by name.

        Args:
            name: Pipeline/command name
            initial_input: Initial input for the pipeline
            args: Arguments to pass to the pipeline

        Returns:
            Run ID if pipeline started, None if not found
        """
        if name not in self.pipelines:
            return None

        pipeline = self.pipelines[name](args)
        return pipeline.run(initial_input, args)

    def handle_prompt(self, prompt: str) -> Optional[str]:
        """Handle a user prompt, potentially starting a pipeline.

        Args:
            prompt: The user's input

        Returns:
            Run ID if a command was found and pipeline started, None otherwise
        """
        parsed = parse_command(prompt, self.command_prefix)
        if not parsed or parsed['command'] not in self.pipelines:
            return None

        # Extract initial input from the args portion of the command
        initial_input = parsed.get('args', '') or ''

        return self.start(parsed['command'], initial_input)

    def run(self) -> None:
        """Run as a CLI handler, reading from stdin.

        Expects JSON input with 'prompt' and optionally 'args' fields.
        Outputs the run ID as JSON.
        """
        if len(sys.argv) < 2:
            return

        action = sys.argv[1]

        if action == "start":
            try:
                hook_input = json.loads(sys.stdin.read())
                prompt = hook_input.get('prompt', '')
                args = hook_input.get('args', {})

                # Parse command from prompt
                parsed = parse_command(prompt, self.command_prefix)
                if not parsed or parsed['command'] not in self.pipelines:
                    return

                # Start pipeline
                initial_input = parsed.get('args', '') or ''
                pipeline = self.pipelines[parsed['command']](args)
                run_id = pipeline.run(initial_input, args)

                print(json.dumps({"run_id": run_id}, indent=2))
            except Exception as e:
                print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)

        elif action == "status":
            # Check status of a run
            try:
                hook_input = json.loads(sys.stdin.read())
                run_id = hook_input.get('run_id')
                pipeline_name = hook_input.get('pipeline')

                if not run_id or pipeline_name not in self.pipelines:
                    print(json.dumps({"error": "Missing run_id or pipeline"}))
                    return

                pipeline = self.pipelines[pipeline_name]({})
                state = pipeline.get_run(run_id)

                if state:
                    print(json.dumps({
                        "run_id": state.id,
                        "status": state.status,
                        "current_step": state.current_step,
                        "created_at": str(state.created_at),
                        "updated_at": str(state.updated_at)
                    }, indent=2))
                else:
                    print(json.dumps({"error": "Run not found"}))
            except Exception as e:
                print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
