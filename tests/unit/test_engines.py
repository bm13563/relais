"""Unit tests for engines/__init__.py — each engine's module, and its SDK, is
imported when a step first needs that engine, not at `import relais`.

The import checks run in a fresh interpreter: this one has long since imported
both engines through the other tests."""

import subprocess
import sys

import pytest

from relais.engines import ENGINES, engine_class
from relais.engines.api import ApiEngine
from relais.engines.sdk import SdkEngine


def loaded_after(code: str) -> dict:
    """Which SDKs a fresh interpreter holds after running `code`."""
    probe = code + "\nimport sys; print('anthropic' in sys.modules, 'claude_agent_sdk' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    anthropic, claude_agent_sdk = out.split()
    return {"anthropic": anthropic == "True", "claude_agent_sdk": claude_agent_sdk == "True"}


class TestEnginesLoadOnFirstUse:
    def test_import_relais_loads_neither_sdk(self):
        assert loaded_after("import relais") == {"anthropic": False, "claude_agent_sdk": False}

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("api", {"anthropic": True, "claude_agent_sdk": False}),
            ("sdk", {"anthropic": False, "claude_agent_sdk": True}),
        ],
    )
    def test_an_engine_loads_its_own_sdk_only(self, name, expected):
        code = f"from relais.engines import engine_class; engine_class({name!r})"
        assert loaded_after(code) == expected


class TestEngineClass:
    def test_each_name_resolves_to_its_engine(self):
        assert set(ENGINES) == {"api", "sdk"}
        assert engine_class("api") is ApiEngine
        assert engine_class("sdk") is SdkEngine
