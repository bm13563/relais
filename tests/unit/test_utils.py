"""Unit tests for utils.py - utility functions."""

import pytest
from pathlib import Path

from relais.utils import read_markdown


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
