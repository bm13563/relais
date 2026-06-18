"""
Relais - A framework for building multi-step AI agent pipelines.

This package provides:
- Pipeline: High-level interface for creating and running pipelines
- PipelineStep: Individual steps with instructions, hooks, tools, and routing
- PipelineOrchestrator: Low-level execution engine
- ToolRegistry: Central registry for pipeline tools
- SQLiteStateManager: State persistence in SQLite
"""

from typing import Annotated

from .step import PipelineStep
from .tools import ToolRegistry, ToolDefinition, ToolResponse, tool
from .state import SQLiteStateManager, PipelineRunState
from .executor import (
    PipelineOrchestrator,
    PipelineConfig,
    StepExecutionResult,
    ResponseToolNotCalled,
)
from .pipeline import Pipeline, cleanup_all_pipeline_states
from .conversation import Conversation, Turn
from .utils import read_markdown
from .logging_config import setup_logging, get_logger
from .agent import PipelineAgent

__all__ = [
    # High-level API
    "Pipeline",
    "PipelineStep",
    "Conversation",
    "Turn",

    # Execution engine
    "PipelineOrchestrator",
    "PipelineConfig",
    "StepExecutionResult",
    "ResponseToolNotCalled",

    # Agents
    "PipelineAgent",

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
    "read_markdown",
]
