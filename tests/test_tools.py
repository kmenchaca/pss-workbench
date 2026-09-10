"""Tests for pss/tools.py - agentic tool definitions and executors."""

import os
import tempfile
import time
import sys
import subprocess
from pathlib import Path

import pytest

from pss.tools import (
    TOOL_DEFINITIONS,
    ToolExecutor,
    ToolResult,
    execute_list_directory,
    execute_read_file,
    execute_run_command,
    execute_search_files,
    get_tool_definitions,
)


class TestToolDefinitions:
    """Test tool definition structure."""

    def test_all_tools_have_required_fields(self):
        """Each tool definition should have name, description, and input_schema."""
        for name, defn in TOOL_DEFINITIONS.items():
            assert "name" in defn
            assert "description" in defn
            assert "input_schema" in defn
            assert defn["name"] == name

    def test_get_tool_definitions_filters_correctly(self):
        """get_tool_definitions should return only requested tools."""
        tools = get_tool_definitions(["read_file", "list_directory"])
        assert len(tools) == 2
        names = [t["name"] for t in tools]
        assert "read_file" in names
        assert "list_directory" in names

    def test_get_tool_definitions_ignores_unknown(self):
        """Unknown tool names should be ignored."""
        tools = get_tool_definitions(["read_file", "unknown_tool"])
        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"


class TestReadFile:
    """Test read_file tool executor."""

    def test_read_existing_file(self, tmp_path):
        """Should read file contents successfully."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = execute_read_file(str(test_file))
        assert result.success
        assert result.output == "hello world"
        assert result.error is None

    def test_read_nonexistent_file(self, tmp_path):
        """Should return error for missing file."""
        result = execute_read_file(str(tmp_path / "missing.txt"))
        assert not result.success
        assert "not found" in result.error.lower()

    def test_read_with_working_dir(self, tmp_path):
        """Should resolve relative paths from working_dir."""
        test_file = tmp_path / "subdir" / "test.txt"
        test_file.parent.mkdir()
        test_file.write_text("content")

        result = execute_read_file("subdir/test.txt", str(tmp_path))
        assert result.success
        assert result.output == "content"

    def test_read_truncates_large_files(self, tmp_path):
        """Should truncate files larger than max_size."""
        test_file = tmp_path / "large.txt"
        test_file.write_text("x" * 200_000)  # 200KB

        result = execute_read_file(str(test_file))
        assert result.success
        assert "[TRUNCATED" in result.output
        assert len(result.output) < 200_000


class TestListDirectory:
    """Test list_directory tool executor."""

    def test_list_directory(self, tmp_path):
        """Should list files and directories."""
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.py").write_text("b")
        (tmp_path / "subdir").mkdir()

        result = execute_list_directory(str(tmp_path))
        assert result.success
        assert "file1.txt" in result.output
        assert "file2.py" in result.output
        assert "subdir/" in result.output

    def test_list_nonexistent_directory(self, tmp_path):
        """Should return error for missing directory."""
        result = execute_list_directory(str(tmp_path / "missing"))
        assert not result.success
        assert "not found" in result.error.lower()

    def test_list_with_working_dir(self, tmp_path):
        """Should resolve relative paths from working_dir."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "test.txt").write_text("x")

        result = execute_list_directory("subdir", str(tmp_path))
        assert result.success
        assert "test.txt" in result.output


class TestSearchFiles:
    """Test search_files tool executor."""

    def test_search_finds_matches(self, tmp_path):
        """Should find pattern matches in files."""
        (tmp_path / "file1.py").write_text("def hello():\n    pass")
        (tmp_path / "file2.py").write_text("def world():\n    pass")

        result = execute_search_files("def .*\\(\\)", str(tmp_path), "*.py")
        assert result.success
        assert "hello" in result.output
        assert "world" in result.output

    def test_search_no_matches(self, tmp_path):
        """Should return 'No matches found' when nothing matches."""
        (tmp_path / "file.txt").write_text("nothing here")

        result = execute_search_files("nonexistent", str(tmp_path))
        assert result.success
        assert "no matches" in result.output.lower()

    def test_search_invalid_regex(self, tmp_path):
        """Should handle invalid regex gracefully."""
        result = execute_search_files("[invalid", str(tmp_path))
        assert not result.success
        assert "regex" in result.error.lower()

    def test_search_with_file_pattern(self, tmp_path):
        """Should filter by file pattern."""
        (tmp_path / "file.py").write_text("pattern here")
        (tmp_path / "file.txt").write_text("pattern here too")

        result = execute_search_files("pattern", str(tmp_path), "*.py")
        assert result.success
        assert "file.py" in result.output
        # Should not include .txt file
        assert "file.txt" not in result.output


class TestRunCommand:
    """Test run_command tool executor."""

    def test_run_simple_command(self):
        """Should run command and capture output."""
        # Use a cross-platform command
        if os.name == "nt":
            result = execute_run_command("echo hello")
        else:
            result = execute_run_command("echo hello")

        assert result.success
        assert "hello" in result.output

    def test_run_failing_command(self):
        """Should capture non-zero exit codes."""
        if os.name == "nt":
            result = execute_run_command("cmd /c exit 1")
        else:
            result = execute_run_command("false")

        assert not result.success
        assert result.error is not None

    def test_run_with_working_dir(self, tmp_path):
        """Should run from specified working directory."""
        if os.name == "nt":
            result = execute_run_command("cd", working_dir=str(tmp_path))
        else:
            result = execute_run_command("pwd", working_dir=str(tmp_path))

        assert result.success
        # Output should contain the tmp_path
        assert str(tmp_path).lower() in result.output.lower() or tmp_path.name in result.output

    def test_run_timeout(self):
        """Should timeout long-running commands."""
        started = time.monotonic()
        if os.name == "nt":
            result = execute_run_command("ping -n 100 localhost", timeout=1)
        else:
            result = execute_run_command("sleep 100", timeout=1)

        assert not result.success
        assert "timed" in result.error.lower()  # "timed out"
        assert time.monotonic() - started < 8, "Inherited stdout must not delay timeout until the child exits"

    @pytest.mark.skipif(os.name != "nt", reason="Windows process-tree regression")
    def test_timeout_terminates_descendant_process(self, tmp_path):
        import ctypes
        from ctypes import wintypes
        script = tmp_path / "spawn_child.py"
        script.write_text(
            "import subprocess, sys, time\n"
            "from pathlib import Path\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "Path('child.pid').write_text(str(child.pid))\n"
            "time.sleep(60)\n", encoding="utf-8"
        )
        command = f'"{sys.executable}" "{script}"'
        started = time.monotonic()
        result = execute_run_command(command, timeout=2, working_dir=str(tmp_path))
        assert not result.success and "timed out" in result.error
        assert time.monotonic() - started < 9
        child_pid = int((tmp_path / "child.pid").read_text())
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, child_pid)  # SYNCHRONIZE, no write access
        if handle:
            terminated = kernel.WaitForSingleObject(handle, 1000) == 0
            kernel.CloseHandle(handle)
            if not terminated:
                # Exact test-created PID only; leave no sleeping regression child.
                subprocess.run(["taskkill", "/PID", str(child_pid), "/T", "/F"],
                               capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
            assert terminated, "Timed-out command left its child running"

    def test_run_bounds_large_output(self):
        command = f'"{sys.executable}" -c "import sys; print(\'x\'*60000); print(\'problem\', file=sys.stderr)"'
        result = execute_run_command(command)
        assert result.success
        assert "TRUNCATED" in result.output
        assert len(result.output) < 50100


class TestToolExecutor:
    """Test the ToolExecutor class."""

    def test_executor_dispatches_correctly(self, tmp_path):
        """Executor should dispatch to correct tool function."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        executor = ToolExecutor(working_dir=str(tmp_path))

        result = executor.execute("read_file", {"path": "test.txt"})
        assert result.success
        assert result.output == "content"

    def test_executor_unknown_tool(self):
        """Should return error for unknown tools."""
        executor = ToolExecutor()
        result = executor.execute("nonexistent_tool", {})
        assert not result.success
        assert "unknown tool" in result.error.lower()

    def test_format_result_for_message(self):
        """Should format results appropriately."""
        executor = ToolExecutor()

        success_result = ToolResult(success=True, output="file contents")
        assert executor.format_result_for_message(success_result) == "file contents"

        error_result = ToolResult(success=False, output="", error="File not found")
        formatted = executor.format_result_for_message(error_result)
        assert "Error" in formatted
        assert "File not found" in formatted
