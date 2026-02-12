"""
Relais - A framework for building multi-step AI agent pipelines.

This package provides:
- Pipeline: High-level interface for creating and running pipelines
- PipelineStep: Individual steps with instructions, hooks, tools, and routing
- PipelineOrchestrator: Low-level execution engine
- ToolRegistry: Central registry for pipeline tools
- SQLiteStateManager: State persistence in SQLite
- PipelineRouter: Command routing for starting pipelines
"""

from typing import Annotated

from .step import PipelineStep
from .tools import ToolRegistry, ToolDefinition, ToolResponse, tool
from .state import SQLiteStateManager, PipelineRunState
from .executor import (
    PipelineOrchestrator,
    PipelineConfig,
    StepExecutionResult,
)
from .pipeline import Pipeline, cleanup_all_pipeline_states
from .router import PipelineRouter
from .utils import parse_command, read_markdown
from .logging_config import setup_logging, get_logger
from .agent import PipelineAgent
from .agent_state import AgentStateManager

__all__ = [
    # High-level API
    "Pipeline",
    "PipelineStep",
    "PipelineRouter",

    # Execution engine
    "PipelineOrchestrator",
    "PipelineConfig",
    "StepExecutionResult",

    # Agents
    "PipelineAgent",
    "AgentStateManager",

    # Tools
    "Annotated",
    "tool",
    "ToolRegistry",
    "ToolDefinition",
    "ToolResponse",

    # State management
    "SQLiteStateManager",
    "PipelineRunState",

    # Logging
    "setup_logging",
    "get_logger",

    # Utilities
    "cleanup_all_pipeline_states",
    "parse_command",
    "read_markdown",
]
