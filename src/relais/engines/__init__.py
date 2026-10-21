"""The engines a pipeline step can run on, by the name PipelineAgent.engine takes.

See relais.engine for the seam they implement.
"""

from .sdk import SdkEngine
from .api import ApiEngine

ENGINES = {"sdk": SdkEngine, "api": ApiEngine}

__all__ = ["ENGINES", "SdkEngine", "ApiEngine"]
