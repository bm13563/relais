"""Unit tests for utils.py - utility functions."""

import pytest
from pathlib import Path

from relais.utils import read_markdown, parse_command


class TestReadMarkdown:
    """Tests for read_markdown function."""

    def test_read_markdown_with_base_dir(self, tmp_path):
        """Test reading markdown file with base directory."""
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Hello\nThis is a test.")

        content = read_markdown("test.md", tmp_path)
        assert content == "# Hello\nThis is a test."

    def test_read_markdown_without_base_dir(self, tmp_path):
        """Test reading markdown file with absolute path."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Content\nSome content here.")

        content = read_markdown(str(test_file))
        assert content == "# Content\nSome content here."

    def test_read_markdown_nested_path(self, tmp_path):
        """Test reading markdown from nested directory."""
        nested_dir = tmp_path / "level1" / "level2"
        nested_dir.mkdir(parents=True)
        test_file = nested_dir / "nested.md"
        test_file.write_text("# Nested\nNested content.")

        content = read_markdown("level1/level2/nested.md", tmp_path)
        assert content == "# Nested\nNested content."

    def test_read_markdown_unicode_content(self, tmp_path):
        """Test reading markdown with unicode characters."""
        test_file = tmp_path / "unicode.md"
        test_file.write_text("# 你好\nEmoji: 🎉 🚀 ✨")

        content = read_markdown("unicode.md", tmp_path)
        assert "你好" in content
        assert "🎉" in content

    def test_read_markdown_empty_file(self, tmp_path):
        """Test reading empty markdown file."""
        test_file = tmp_path / "empty.md"
        test_file.write_text("")

        content = read_markdown("empty.md", tmp_path)
        assert content == ""

    def test_read_markdown_file_not_found(self, tmp_path):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_markdown("nonexistent.md", tmp_path)

    def test_read_markdown_multiline(self, tmp_path):
        """Test reading multiline markdown content."""
        content_text = """# Title

## Section 1
Some text here.

## Section 2
More text here.

- Item 1
- Item 2
- Item 3
"""
        test_file = tmp_path / "multiline.md"
        test_file.write_text(content_text)

        content = read_markdown("multiline.md", tmp_path)
        assert "# Title" in content
        assert "## Section 1" in content
        assert "- Item 1" in content


class TestParseCommand:
    """Tests for parse_command function."""

    def test_parse_command_basic(self):
        """Test basic command parsing."""
        result = parse_command("#learn python basics")
        assert result is not None
        assert result["command"] == "learn"
        assert result["args"] == "python basics"

    def test_parse_command_no_args(self):
        """Test command with no arguments."""
        result = parse_command("#help")
        assert result is not None
        assert result["command"] == "help"
        assert result["args"] is None

    def test_parse_command_different_prefix(self):
        """Test command with different prefix."""
        result = parse_command("/analyze this text", prefix="/")
        assert result is not None
        assert result["command"] == "analyze"
        assert result["args"] == "this text"

    def test_parse_command_no_match(self):
        """Test when no command is found."""
        result = parse_command("just regular text")
        assert result is None

    def test_parse_command_empty_string(self):
        """Test with empty string."""
        result = parse_command("")
        assert result is None

    def test_parse_command_only_prefix(self):
        """Test with only prefix character."""
        result = parse_command("#")
        assert result is None

    def test_parse_command_with_whitespace(self):
        """Test command with leading/trailing whitespace."""
        result = parse_command("  #start something   ")
        assert result is not None
        assert result["command"] == "start"
        assert result["args"] == "something"

    def test_parse_command_multiword_args(self):
        """Test command with multi-word arguments."""
        result = parse_command("#search find all python files")
        assert result is not None
        assert result["command"] == "search"
        assert result["args"] == "find all python files"

    def test_parse_command_special_chars_in_args(self):
        """Test command with special characters in arguments."""
        result = parse_command("#query SELECT * FROM users WHERE id = 1")
        assert result is not None
        assert result["command"] == "query"
        assert result["args"] == "SELECT * FROM users WHERE id = 1"

    def test_parse_command_at_sign_prefix(self):
        """Test with @ prefix."""
        result = parse_command("@mention someone", prefix="@")
        assert result is not None
        assert result["command"] == "mention"
        assert result["args"] == "someone"

    def test_parse_command_escaped_prefix(self):
        """Test with prefix that needs regex escaping."""
        result = parse_command("$run script", prefix="$")
        assert result is not None
        assert result["command"] == "run"
        assert result["args"] == "script"

    def test_parse_command_numbers_in_command(self):
        """Test command containing numbers."""
        result = parse_command("#test123 some args")
        assert result is not None
        assert result["command"] == "test123"
        assert result["args"] == "some args"

    def test_parse_command_underscore_in_command(self):
        """Test command with underscore."""
        result = parse_command("#my_command arg1 arg2")
        assert result is not None
        assert result["command"] == "my_command"
        assert result["args"] == "arg1 arg2"

    def test_parse_command_hyphen_not_allowed(self):
        """Test that hyphen in command causes no match.

        The regex expects command to be followed by whitespace or end-of-string.
        A hyphen after alphanumeric chars doesn't match this pattern.
        """
        result = parse_command("#my-command arg")
        # Hyphen after "my" doesn't match expected pattern (whitespace or $)
        assert result is None

    def test_parse_command_case_sensitivity(self):
        """Test that command parsing preserves case."""
        result = parse_command("#MyCommand Args")
        assert result is not None
        assert result["command"] == "MyCommand"
        assert result["args"] == "Args"

    def test_parse_command_in_middle_of_text(self):
        """Test command appearing in middle of text."""
        # The regex uses $ so it should match at end of string
        result = parse_command("some text #command args")
        assert result is not None
        assert result["command"] == "command"
        assert result["args"] == "args"
