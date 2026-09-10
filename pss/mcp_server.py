"""PSS MCP Server - Expose branching exploration to Claude Code.

This server runs locally and uses the user's own API keys.
All exploration costs are paid by the user, not the PSS maintainers.

Installation:
    # Install PSS with MCP support
    pip install pss[mcp]

    # Or with uv
    uv pip install pss[mcp]

    # Register with Claude Code
    claude mcp add --transport stdio pss -- pss-mcp

Usage:
    Once registered, Claude Code can use these tools:
    - pss_explore: Parallel branching exploration for complex questions
    - pss_investigate: Agentic code investigation with synthesis
    - pss_research: Deep, thorough analysis with multiple perspectives
    - pss_debug: Parallel bug hunting with evidence voting
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Check if MCP is available
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from pss.config import PSSConfig, load_config
from pss.harness import run_pss
from pss.presets import get_preset, apply_preset_to_config, PRESETS
from pss.providers import create_provider
from pss.types import PSSResult


def _get_config_path() -> Path:
    """Get the PSS config file path."""
    # Check for project-local config first
    local_config = Path("pss_config.yaml")
    if local_config.exists():
        return local_config

    # Check user home directory
    home_config = Path.home() / ".config" / "pss" / "config.yaml"
    if home_config.exists():
        return home_config

    # Fall back to project-local (will use defaults)
    return local_config


def _format_result(result: list | PSSResult) -> str:
    """Format PSS result for MCP response."""
    if isinstance(result, PSSResult):
        # Synthesis available - return unified output
        if result.synthesis:
            output_parts = [result.synthesis.unified_output]

            # Add confidence
            if result.synthesis.confidence > 0:
                output_parts.append(f"\n\n**Confidence**: {result.synthesis.confidence:.0%}")

            # Add dissenting views if any
            if result.synthesis.dissenting_views:
                output_parts.append("\n\n**Dissenting Views**:")
                for view in result.synthesis.dissenting_views:
                    output_parts.append(f"- {view}")

            # Add action plan if present
            if result.synthesis.action_plan:
                output_parts.append("\n\n**Suggested Actions**:")
                for action in result.synthesis.action_plan:
                    output_parts.append(f"- {action}")

            # Add cost info
            output_parts.append(f"\n\n*PSS explored {len(result.leaves)} branches, cost: ${result.total_usage.cost:.4f}*")

            return "\n".join(output_parts)
        else:
            # No synthesis - return best leaf
            if result.leaves:
                best = max(result.leaves, key=lambda l: len(l.output or ""))
                return best.output or "(No output)"
            return "(No results)"
    else:
        # List of contexts (no synthesis)
        if result:
            best = max(result, key=lambda l: len(l.output or ""))
            total_cost = sum(c.usage.cost for c in result)
            output = best.output or "(No output)"
            return f"{output}\n\n*PSS explored {len(result)} branches, cost: ${total_cost:.4f}*"
        return "(No results)"


def _run_pss_with_preset(
    prompt: str,
    preset_name: str,
    working_dir: str | None = None,
    max_branches: int | None = None,
    synthesis: bool | None = None,
) -> str:
    """Run PSS with a given preset and return formatted result."""
    # Load config
    config_path = _get_config_path()
    config = load_config(config_path)

    # Apply preset
    preset = get_preset(preset_name)
    if preset:
        apply_preset_to_config(preset, config)

    # Apply overrides
    if max_branches is not None:
        config.max_contexts = max_branches

    if working_dir is not None:
        config.agentic_working_dir = working_dir

    if synthesis is not None:
        config.synthesis_enabled = synthesis

    # Enable parallel execution for speed
    config.parallel = True

    # Create provider
    provider = create_provider(config)

    # Run PSS
    result = run_pss(prompt, config, provider)

    return _format_result(result)


# Tool definitions
TOOLS = [
    Tool(
        name="pss_explore",
        description="""Run parallel branching exploration on a complex question.

Use this when:
- A problem might have multiple valid approaches
- You want diverse perspectives on a topic
- Single-shot reasoning might miss alternatives
- Brainstorming or ideation tasks

The tool spawns multiple exploration branches that each take different approaches,
then optionally synthesizes findings into a unified response.""",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question or topic to explore"
                },
                "max_branches": {
                    "type": "integer",
                    "description": "Maximum number of exploration branches (default: 10)",
                    "default": 10
                },
                "synthesis": {
                    "type": "boolean",
                    "description": "Whether to synthesize findings from all branches (default: true)",
                    "default": True
                }
            },
            "required": ["prompt"]
        }
    ),
    Tool(
        name="pss_investigate",
        description="""Agentic code investigation with file access and synthesis.

Use this when:
- Investigating a codebase for bugs, patterns, or understanding
- Need to read files and search code across multiple angles
- Want multiple investigation branches merged into findings

Each branch can read files, search code, and run commands (read-only).
Findings are synthesized into a structured report.""",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "What to investigate (e.g., 'Find security vulnerabilities in src/')"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Directory to investigate (default: current directory)"
                },
                "max_branches": {
                    "type": "integer",
                    "description": "Maximum investigation branches (default: 8)",
                    "default": 8
                }
            },
            "required": ["prompt"]
        }
    ),
    Tool(
        name="pss_research",
        description="""Deep, thorough analysis with multiple perspectives.

Use this when:
- Need comprehensive research on a topic
- Want to explore a subject from multiple angles
- Require thorough, well-reasoned analysis

Spawns branches that each dive deep into different aspects,
then synthesizes into a comprehensive analysis.""",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The research question or topic"
                },
                "max_branches": {
                    "type": "integer",
                    "description": "Maximum research branches (default: 5)",
                    "default": 5
                }
            },
            "required": ["prompt"]
        }
    ),
    Tool(
        name="pss_debug",
        description="""Parallel bug hunting with evidence voting.

Use this when:
- Hunting for bugs in code
- Need multiple debugging approaches tried in parallel
- Want findings ranked by how many branches found them

Each branch investigates independently, then findings are
merged with voting to surface bugs found by multiple branches.""",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Bug description or area to investigate"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Directory containing the code (default: current directory)"
                },
                "max_branches": {
                    "type": "integer",
                    "description": "Maximum debugging branches (default: 6)",
                    "default": 6
                }
            },
            "required": ["prompt"]
        }
    ),
    Tool(
        name="pss_reasoning",
        description="""Mathematical and logical reasoning with multiple solution attempts.

Use this when:
- Solving math problems that might need multiple approaches
- Working through logical puzzles
- Problems where one approach might fail but another succeeds

Uses early checkpoints to try different solution strategies quickly.""",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The math or logic problem to solve"
                },
                "max_branches": {
                    "type": "integer",
                    "description": "Maximum solution attempts (default: 10)",
                    "default": 10
                }
            },
            "required": ["prompt"]
        }
    ),
]


def create_server() -> "Server":
    """Create and configure the MCP server."""
    if not MCP_AVAILABLE:
        raise RuntimeError(
            "MCP support not installed. Install with: pip install pss[mcp]"
        )

    server = Server("pss")

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        prompt = arguments.get("prompt", "")
        max_branches = arguments.get("max_branches")
        working_dir = arguments.get("working_dir")
        synthesis = arguments.get("synthesis", True)

        # Map tool name to preset
        preset_map = {
            "pss_explore": "explore",
            "pss_investigate": "investigate",
            "pss_research": "research",
            "pss_debug": "debug",
            "pss_reasoning": "reasoning",
        }

        preset_name = preset_map.get(name, "explore")

        # Run PSS (blocking - run in thread pool to not block event loop)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _run_pss_with_preset,
            prompt,
            preset_name,
            working_dir,
            max_branches,
            synthesis,
        )

        return [TextContent(type="text", text=result)]

    return server


async def run_server():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP support not installed.", file=sys.stderr)
        print("Install with: pip install pss[mcp]", file=sys.stderr)
        print("Or: uv pip install pss[mcp]", file=sys.stderr)
        sys.exit(1)

    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """Entry point for pss-mcp command."""
    # Load environment variables from .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Check for required API keys
    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: No API keys found.", file=sys.stderr)
        print("Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY environment variable.", file=sys.stderr)

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
