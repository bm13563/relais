"""Unit tests for step.py - PipelineStep class."""

import pytest

from relais.step import PipelineStep


class TestPipelineStepCreation:
    """Tests for PipelineStep instantiation and defaults."""

    def test_minimal_step(self):
        """Test creating step with minimal required fields."""
        step = PipelineStep(
            name="minimal",
            instruction="test"
        )
        assert step.name == "minimal"
        assert step.instruction == "test"
        assert step.max_turns == 10  # default
        assert step.tools == []  # default
        assert step.hooks == []  # default
        assert step.agent is None  # default
        assert step.next == {"default": None}  # default - ends pipeline

    def test_fully_configured_step(self):
        """Test creating step with all fields specified."""
        hooks = [lambda: {"data": "value"}]
        step = PipelineStep(
            name="full",
            instruction="full_instruction",
            next={"default": "next_step"},
            max_turns=15,
            tools=["tool1", "tool2", "tool3"],
            hooks=hooks,
            agent="custom_agent"
        )
        assert step.name == "full"
        assert step.instruction == "full_instruction"
        assert step.next == {"default": "next_step"}
        assert step.max_turns == 15
        assert step.tools == ["tool1", "tool2", "tool3"]
        assert step.hooks == hooks
        assert step.agent == "custom_agent"

    def test_step_with_conditional_routing(self):
        """Test step with conditional routing configuration."""
        step = PipelineStep(
            name="router",
            instruction="router_instruction",
            next={
                "field": "status",
                "routes": [
                    {"equals": "success", "goto": "success_handler"},
                    {"equals": "failure", "goto": "failure_handler"},
                ],
                "default": "unknown_handler"
            }
        )
        assert step.next["field"] == "status"
        assert len(step.next["routes"]) == 2
        assert step.next["default"] == "unknown_handler"


class TestResolveNext:
    """Tests for PipelineStep.resolve_next method."""

    def test_resolve_next_simple_default(self):
        """Test simple default routing (always goes to same step)."""
        step = PipelineStep(
            name="simple",
            instruction="test",
            next={"default": "next_step"}
        )
        result = step.resolve_next({})
        assert result == "next_step"

    def test_resolve_next_ends_pipeline(self):
        """Test routing that ends pipeline (default None)."""
        step = PipelineStep(
            name="terminal",
            instruction="test",
            next={"default": None}
        )
        result = step.resolve_next({})
        assert result is None

    def test_resolve_next_conditional_match_first(self):
        """Test conditional routing matches first route."""
        step = PipelineStep(
            name="router",
            instruction="test",
            next={
                "field": "category",
                "routes": [
                    {"equals": "A", "goto": "handle_a"},
                    {"equals": "B", "goto": "handle_b"},
                ],
                "default": "fallback"
            }
        )
        result = step.resolve_next({"category": "A"})
        assert result == "handle_a"

    def test_resolve_next_conditional_match_second(self):
        """Test conditional routing matches second route."""
        step = PipelineStep(
            name="router",
            instruction="test",
            next={
                "field": "category",
                "routes": [
                    {"equals": "A", "goto": "handle_a"},
                    {"equals": "B", "goto": "handle_b"},
                ],
                "default": "fallback"
            }
        )
        result = step.resolve_next({"category": "B"})
        assert result == "handle_b"

    def test_resolve_next_conditional_no_match_uses_default(self):
        """Test conditional routing falls back to default when no match."""
        step = PipelineStep(
            name="router",
            instruction="test",
            next={
                "field": "category",
                "routes": [
                    {"equals": "A", "goto": "handle_a"},
                    {"equals": "B", "goto": "handle_b"},
                ],
                "default": "fallback"
            }
        )
        result = step.resolve_next({"category": "C"})
        assert result == "fallback"

    def test_resolve_next_conditional_missing_field(self):
        """Test conditional routing when field is missing from result."""
        step = PipelineStep(
            name="router",
            instruction="test",
            next={
                "field": "category",
                "routes": [
                    {"equals": "A", "goto": "handle_a"},
                ],
                "default": "fallback"
            }
        )
        result = step.resolve_next({"other_field": "value"})
        assert result == "fallback"

    def test_resolve_next_conditional_empty_result(self):
        """Test conditional routing with empty result dict."""
        step = PipelineStep(
            name="router",
            instruction="test",
            next={
                "field": "category",
                "routes": [
                    {"equals": "A", "goto": "handle_a"},
                ],
                "default": "fallback"
            }
        )
        result = step.resolve_next({})
        assert result == "fallback"

    def test_resolve_next_conditional_none_value(self):
        """Test conditional routing when field value is None."""
        step = PipelineStep(
            name="router",
            instruction="test",
            next={
                "field": "category",
                "routes": [
                    {"equals": None, "goto": "handle_none"},
                    {"equals": "A", "goto": "handle_a"},
                ],
                "default": "fallback"
            }
        )
        result = step.resolve_next({"category": None})
        assert result == "handle_none"

    def test_resolve_next_conditional_integer_value(self):
        """Test conditional routing with integer values."""
        step = PipelineStep(
            name="router",
            instruction="test",
            next={
                "field": "score",
                "routes": [
                    {"equals": 100, "goto": "perfect"},
                    {"equals": 0, "goto": "fail"},
                ],
                "default": "partial"
            }
        )
        assert step.resolve_next({"score": 100}) == "perfect"
        assert step.resolve_next({"score": 0}) == "fail"
        assert step.resolve_next({"score": 50}) == "partial"

    def test_resolve_next_conditional_boolean_value(self):
        """Test conditional routing with boolean values."""
        step = PipelineStep(
            name="router",
            instruction="test",
            next={
                "field": "success",
                "routes": [
                    {"equals": True, "goto": "success_handler"},
                    {"equals": False, "goto": "failure_handler"},
                ],
                "default": "unknown"
            }
        )
        assert step.resolve_next({"success": True}) == "success_handler"
        assert step.resolve_next({"success": False}) == "failure_handler"

    def test_resolve_next_no_routes_key(self):
        """Test conditional routing when routes key is missing."""
        step = PipelineStep(
            name="router",
            instruction="test",
            next={
                "field": "category",
                "default": "fallback"
            }
        )
        result = step.resolve_next({"category": "A"})
        assert result == "fallback"

    def test_resolve_next_empty_routes(self):
        """Test conditional routing with empty routes list."""
        step = PipelineStep(
            name="router",
            instruction="test",
            next={
                "field": "category",
                "routes": [],
                "default": "fallback"
            }
        )
        result = step.resolve_next({"category": "A"})
        assert result == "fallback"


class TestGetInstruction:
    """Tests for PipelineStep.get_instruction method."""

    def test_get_instruction_basic(self, test_instructions_dir):
        """Test loading instruction file."""
        step = PipelineStep(
            name="test",
            instruction="test_step"
        )
        content = step.get_instruction(test_instructions_dir)
        assert "# Test Step" in content
        assert "Do the test task" in content

    def test_get_instruction_different_file(self, test_instructions_dir):
        """Test loading different instruction file."""
        step = PipelineStep(
            name="greet",
            instruction="greet"
        )
        content = step.get_instruction(test_instructions_dir)
        assert "# Greeting" in content

    def test_get_instruction_file_not_found(self, test_instructions_dir):
        """Test that missing instruction file raises error."""
        step = PipelineStep(
            name="missing",
            instruction="nonexistent_instruction"
        )
        with pytest.raises(FileNotFoundError):
            step.get_instruction(test_instructions_dir)


class TestGetHookData:
    """Tests for PipelineStep.get_hook_data method."""

    def test_get_hook_data_no_hooks(self):
        """Test getting hook data when no hooks defined."""
        step = PipelineStep(
            name="no_hooks",
            instruction="test"
        )
        result = step.get_hook_data()
        assert result == []

    def test_get_hook_data_single_hook(self):
        """Test getting data from single hook."""
        def my_hook():
            return {"key": "value"}

        step = PipelineStep(
            name="hooked",
            instruction="test",
            hooks=[my_hook]
        )
        result = step.get_hook_data()
        assert result == [{"key": "value"}]

    def test_get_hook_data_multiple_hooks(self):
        """Test getting data from multiple hooks."""
        def hook1():
            return {"a": 1}

        def hook2():
            return {"b": 2}

        def hook3():
            return {"c": 3}

        step = PipelineStep(
            name="multi_hooked",
            instruction="test",
            hooks=[hook1, hook2, hook3]
        )
        result = step.get_hook_data()
        assert result == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_get_hook_data_different_return_types(self):
        """Test hooks returning different data types."""
        def string_hook():
            return "just a string"

        def list_hook():
            return [1, 2, 3]

        def dict_hook():
            return {"nested": {"data": True}}

        step = PipelineStep(
            name="varied",
            instruction="test",
            hooks=[string_hook, list_hook, dict_hook]
        )
        result = step.get_hook_data()
        assert result[0] == "just a string"
        assert result[1] == [1, 2, 3]
        assert result[2] == {"nested": {"data": True}}

    def test_get_hook_data_hook_returns_none(self):
        """Test hook that returns None."""
        def none_hook():
            return None

        step = PipelineStep(
            name="none_hook",
            instruction="test",
            hooks=[none_hook]
        )
        result = step.get_hook_data()
        assert result == [None]

    def test_get_hook_data_hook_with_side_effects(self):
        """Test that hooks are actually executed."""
        call_count = {"count": 0}

        def counting_hook():
            call_count["count"] += 1
            return call_count["count"]

        step = PipelineStep(
            name="counting",
            instruction="test",
            hooks=[counting_hook]
        )

        # First call
        result1 = step.get_hook_data()
        assert result1 == [1]

        # Second call
        result2 = step.get_hook_data()
        assert result2 == [2]

    def test_get_hook_data_execution_order(self):
        """Test that hooks execute in order."""
        results = []

        def hook_a():
            results.append("a")
            return "a"

        def hook_b():
            results.append("b")
            return "b"

        def hook_c():
            results.append("c")
            return "c"

        step = PipelineStep(
            name="ordered",
            instruction="test",
            hooks=[hook_a, hook_b, hook_c]
        )
        step.get_hook_data()
        assert results == ["a", "b", "c"]


class TestStepEquality:
    """Tests for step equality and identity."""

    def test_steps_with_same_config_are_equal(self):
        """Test that identically configured steps are equal."""
        step1 = PipelineStep(
            name="test",
            instruction="test_instruction",
            max_turns=5,
            tools=["tool1"]
        )
        step2 = PipelineStep(
            name="test",
            instruction="test_instruction",
            max_turns=5,
            tools=["tool1"]
        )
        assert step1 == step2

    def test_steps_with_different_names_not_equal(self):
        """Test that steps with different names are not equal."""
        step1 = PipelineStep(name="step1", instruction="test")
        step2 = PipelineStep(name="step2", instruction="test")
        assert step1 != step2

    def test_steps_with_different_tools_not_equal(self):
        """Test that steps with different tools are not equal."""
        step1 = PipelineStep(name="test", instruction="test", tools=["a"])
        step2 = PipelineStep(name="test", instruction="test", tools=["b"])
        assert step1 != step2
