"""The engines a pipeline step can run on, by the name PipelineAgent.engine takes.

See relais.engine for the seam they implement. An engine's module — and with it
its SDK — is imported the first time a step needs that engine, not at `import
relais`: the api engine brings the Anthropic SDK, the sdk engine the Claude
Agent SDK, and a host that runs one engine should not hold the other in memory.
Both stay installed dependencies, so either engine works wherever relais is
installed.
"""

from importlib import import_module

# name → (module, class): the names PipelineAgent.engine accepts. The module is
# imported by engine_class(), on first use.
ENGINES = {
    "sdk": ("relais.engines.sdk", "SdkEngine"),
    "api": ("relais.engines.api", "ApiEngine"),
}


def engine_class(name: str) -> type:
    """The engine class for a name in ENGINES, importing its module now."""
    module, cls = ENGINES[name]
    return getattr(import_module(module), cls)


__all__ = ["ENGINES", "engine_class"]
