"""The engines a pipeline step can run on, by the name PipelineAgent.engine takes.

See relais.engine for the seam they implement.
"""

from .sdk import SdkEngine

ENGINES = {"sdk": SdkEngine}

__all__ = ["ENGINES", "SdkEngine"]
