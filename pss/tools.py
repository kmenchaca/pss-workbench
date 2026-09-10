"""Agentic tools for PSS branches.

This module defines tools that branches can use during exploration.
Tools are read-only to avoid conflicts between parallel branches.
Write actions should be deferred to the synthesis action plan.
"""

import logging
import os
import re
import subprocess
import signal
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Commands that are likely to write/modify state — blocked in parallel mode
WRITE_COMMAND_PATTERNS = re.compile(
    r'(?:^|\s|&&|\|\||;)'  # preceded by start/whitespace/operator
    r'(?:rm|rmdir|mv|cp|mkdir|touch|chmod|chown|tee|dd|truncate|install'
    r'|pip install|npm install|apt|brew|git commit|git push|git checkout'
    r'|sed -i|awk .* >)'
    r'(?:\s|$)',
    re.IGNORECASE,
)
WRITE_REDIRECT_PATTERN = re.compile(r'[^2]?>{1,2}\s*[^&]')


@dataclass
class ToolResult:
    """Result of executing a tool."""

    success: bool
    output: str
    error: str | None = None


# Tool definitions following the same pattern as gates.py
TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "read_file": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)",
                }
            },
            "required": ["path"],
        },
    },
    "list_directory": {
        "name": "list_directory",
        "description": "List files and directories at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)",
                    "default": ".",
                }
            },
            "required": [],
        },
    },
    "search_files": {
        "name": "search_files",
        "description": "Search for a regex pattern across files in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)",
                    "default": ".",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to search (e.g. '*.py')",
                    "default": "*",
                },
            },
            "required": ["pattern"],
        },
    },
    "run_command": {
        "name": "run_command",
        "description": "Execute a shell command and return stdout/stderr. Use for running tests, builds, or other commands.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
    },
}


def _is_write_command(command: str) -> bool:
    """Heuristic check for commands that modify state."""
    if WRITE_REDIRECT_PATTERN.search(command):
        return True
    if WRITE_COMMAND_PATTERNS.search(command):
        return True
    return False


def get_tool_definitions(tool_names: list[str]) -> list[dict[str, Any]]:
    """Get tool definitions for the specified tool names."""
    return [TOOL_DEFINITIONS[name] for name in tool_names if name in TOOL_DEFINITIONS]


# Tool executor implementations


def execute_read_file(path: str, working_dir: str | None = None) -> ToolResult:
    """Read a file and return its contents."""
    try:
        if working_dir:
            full_path = Path(working_dir) / path
        else:
            full_path = Path(path)

        if not full_path.exists():
            return ToolResult(
                success=False, output="", error=f"File not found: {full_path}"
            )

        if not full_path.is_file():
            return ToolResult(
                success=False, output="", error=f"Not a file: {full_path}"
            )

        # Limit file size to avoid memory issues
        max_size = 100_000  # 100KB
        if full_path.stat().st_size > max_size:
            content = full_path.read_text(encoding="utf-8", errors="replace")[
                :max_size
            ]
            content += f"\n\n[TRUNCATED: File exceeds {max_size} bytes]"
        else:
            content = full_path.read_text(encoding="utf-8", errors="replace")

        return ToolResult(success=True, output=content)

    except PermissionError:
        return ToolResult(
            success=False, output="", error=f"Permission denied: {path}"
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Error reading file: {e}")


def execute_list_directory(path: str = ".", working_dir: str | None = None) -> ToolResult:
    """List contents of a directory."""
    try:
        if working_dir:
            full_path = Path(working_dir) / path
        else:
            full_path = Path(path)

        if not full_path.exists():
            return ToolResult(
                success=False, output="", error=f"Directory not found: {full_path}"
            )

        if not full_path.is_dir():
            return ToolResult(
                success=False, output="", error=f"Not a directory: {full_path}"
            )

        entries = []
        for entry in sorted(full_path.iterdir()):
            if entry.is_dir():
                entries.append(f"{entry.name}/")
            else:
                size = entry.stat().st_size
                entries.append(f"{entry.name} ({size} bytes)")

        return ToolResult(success=True, output="\n".join(entries))

    except PermissionError:
        return ToolResult(
            success=False, output="", error=f"Permission denied: {path}"
        )
    except Exception as e:
        return ToolResult(
            success=False, output="", error=f"Error listing directory: {e}"
        )


def execute_search_files(
    pattern: str,
    path: str = ".",
    file_pattern: str = "*",
    working_dir: str | None = None,
) -> ToolResult:
    """Search for a regex pattern in files."""
    try:
        if working_dir:
            base_path = Path(working_dir) / path
        else:
            base_path = Path(path)

        if not base_path.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {base_path}"
            )

        regex = re.compile(pattern)
        results = []
        max_results = 100  # Limit results to avoid huge output

        for file_path in base_path.rglob(file_pattern):
            if not file_path.is_file():
                continue

            # Skip binary files and large files
            try:
                if file_path.stat().st_size > 100_000:
                    continue

                content = file_path.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        rel_path = file_path.relative_to(base_path)
                        results.append(f"{rel_path}:{i}: {line.strip()[:200]}")

                        if len(results) >= max_results:
                            results.append(
                                f"\n[TRUNCATED: Found {max_results}+ matches]"
                            )
                            return ToolResult(success=True, output="\n".join(results))
            except (UnicodeDecodeError, PermissionError):
                continue

        if not results:
            return ToolResult(success=True, output="No matches found.")

        return ToolResult(success=True, output="\n".join(results))

    except re.error as e:
        return ToolResult(success=False, output="", error=f"Invalid regex: {e}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Error searching: {e}")


def execute_run_command(
    command: str, timeout: int = 30, working_dir: str | None = None
) -> ToolResult:
    """Execute a shell command, bounding waits and terminating its tree on timeout."""
    try:
        # File-backed capture avoids communicate() waiting indefinitely when a
        # descendant inherits a pipe after the shell is terminated (Windows).
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(
                command, shell=True, stdout=stdout, stderr=stderr, cwd=working_dir,
                start_new_session=os.name != "nt",
                creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
                if os.name == "nt" else 0,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                cleanup_error = _terminate_command_tree(process)
                return ToolResult(
                    success=False, output="",
                    error=f"Command timed out after {timeout} seconds"
                    + (f"; {cleanup_error}" if cleanup_error else ""),
                )
            stdout.seek(0)
            stderr.seek(0)
            # Read at most the displayed cap plus one character for truncation.
            captured_stdout = stdout.read(50_001).decode(errors="replace")
            captured_stderr = stderr.read(50_001).decode(errors="replace")

        output_parts = []
        if captured_stdout:
            output_parts.append(captured_stdout)
        if captured_stderr:
            output_parts.append(f"STDERR:\n{captured_stderr}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        # Truncate very long output
        max_output = 50_000
        if len(output) > max_output:
            output = output[:max_output] + f"\n\n[TRUNCATED: Output exceeds {max_output} chars]"

        return ToolResult(
            success=process.returncode == 0,
            output=output,
            error=f"Exit code: {process.returncode}" if process.returncode != 0 else None,
        )

    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            output="",
            error=f"Command timed out after {timeout} seconds",
        )
    except Exception as e:
        return ToolResult(
            success=False, output="", error=f"Error running command: {e}"
        )


def _terminate_command_tree(process: subprocess.Popen) -> str | None:
    """Terminate only the command's process tree; never use image-name matching."""
    error = None
    try:
        if os.name == "nt":
            taskkill = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/taskkill.exe"
            result = subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode != 0:
                error = "process-tree cleanup could not be confirmed"
        else:
            # start_new_session gives this command its own group, including
            # ordinary descendants, without touching the host's process group.
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        error = "process-tree cleanup could not be confirmed"
    finally:
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                # It may have exited between poll() and kill().
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            error = "command cleanup timed out"
    return error


# Tool executor registry
TOOL_EXECUTORS: dict[str, Callable[..., ToolResult]] = {
    "read_file": execute_read_file,
    "list_directory": execute_list_directory,
    "search_files": execute_search_files,
    "run_command": execute_run_command,
}


class ToolExecutor:
    """Executes tools and tracks usage."""

    def __init__(self, working_dir: str | None = None, timeout: int = 30, read_only: bool = False):
        self.working_dir = working_dir or os.getcwd()
        self.default_timeout = timeout
        self.read_only = read_only

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments."""
        if tool_name not in TOOL_EXECUTORS:
            return ToolResult(
                success=False, output="", error=f"Unknown tool: {tool_name}"
            )

        executor = TOOL_EXECUTORS[tool_name]

        # Inject working_dir and handle defaults
        if tool_name == "read_file":
            return executor(arguments.get("path", ""), self.working_dir)
        elif tool_name == "list_directory":
            return executor(arguments.get("path", "."), self.working_dir)
        elif tool_name == "search_files":
            return executor(
                arguments.get("pattern", ""),
                arguments.get("path", "."),
                arguments.get("file_pattern", "*"),
                self.working_dir,
            )
        elif tool_name == "run_command":
            command = arguments.get("command", "")
            if self.read_only and _is_write_command(command):
                logger.warning(f"[PSS] Blocked write command in read-only mode: {command[:80]}")
                return ToolResult(
                    success=False,
                    output="",
                    error="Write operations are blocked in parallel mode to prevent "
                          "conflicts between branches. Defer writes to the synthesis action plan.",
                )
            return executor(
                command,
                arguments.get("timeout", self.default_timeout),
                self.working_dir,
            )
        else:
            return ToolResult(
                success=False, output="", error=f"No handler for tool: {tool_name}"
            )

    def format_result_for_message(self, result: ToolResult) -> str:
        """Format a tool result for inclusion in a message."""
        if result.success:
            return result.output
        else:
            return f"Error: {result.error}\n{result.output}" if result.output else f"Error: {result.error}"
