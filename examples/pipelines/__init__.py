"""Pipeline factory functions for example pipelines."""

from .simple_greeting import create_simple_greeting
from .routing import create_routing_pipeline
from .research import create_research_pipeline
from .hooks_pipeline import create_hooks_pipeline

__all__ = [
    "create_simple_greeting",
    "create_routing_pipeline",
    "create_research_pipeline",
    "create_hooks_pipeline",
]
