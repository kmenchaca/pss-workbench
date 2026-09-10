"""Tests for MCP server module."""

import pytest
from unittest.mock import patch, MagicMock


class TestMCPServerModule:
    """Tests for MCP server loading and tool definitions."""

    def test_mcp_server_imports(self):
        """Test that MCP server module imports correctly."""
        from pss.mcp_server import TOOLS, _get_config_path, _format_result

        # Should have 5 tools
        assert len(TOOLS) == 5

    def test_tool_names(self):
        """Test that expected tools are defined."""
        from pss.mcp_server import TOOLS

        tool_names = {t.name for t in TOOLS}
        expected = {"pss_explore", "pss_investigate", "pss_research", "pss_debug", "pss_reasoning"}
        assert tool_names == expected

    def test_tool_schemas_valid(self):
        """Test that tool schemas have required fields."""
        from pss.mcp_server import TOOLS

        for tool in TOOLS:
            assert tool.name is not None
            assert tool.description is not None
            assert tool.inputSchema is not None
            assert "properties" in tool.inputSchema
            assert "prompt" in tool.inputSchema["properties"]

    def test_format_result_with_list(self):
        """Test formatting a list of contexts."""
        from pss.mcp_server import _format_result
        from pss.types import Context, TokenUsage

        # Create mock contexts
        ctx1 = Context(
            id="test1",
            parent_id=None,
            messages=[],
            output="Result 1",
        )
        ctx1.usage = TokenUsage(input_tokens=100, output_tokens=50, cost=0.001)

        ctx2 = Context(
            id="test2",
            parent_id=None,
            messages=[],
            output="Result 2 with more content",
        )
        ctx2.usage = TokenUsage(input_tokens=100, output_tokens=50, cost=0.001)

        result = _format_result([ctx1, ctx2])

        # Should return the longer output (ctx2)
        assert "Result 2" in result
        assert "2 branches" in result
        assert "$0.002" in result  # total cost

    def test_format_result_with_pss_result(self):
        """Test formatting a PSSResult with synthesis."""
        from pss.mcp_server import _format_result
        from pss.types import Context, PSSResult, SynthesisResult, SearchTree, TokenUsage

        # Create mock synthesis result
        synthesis = SynthesisResult(
            unified_output="Synthesized findings from all branches.",
            confidence=0.85,
            dissenting_views=["Branch 1 disagreed on X"],
            action_plan=["Fix bug A", "Add test B"],
        )
        synthesis.usage = TokenUsage(input_tokens=500, output_tokens=200, cost=0.005)

        ctx = Context(id="test", parent_id=None, messages=[], output="Leaf output")
        ctx.usage = TokenUsage(input_tokens=100, output_tokens=50, cost=0.001)

        tree = SearchTree()
        tree.contexts["test"] = ctx
        tree.leaves = ["test"]

        pss_result = PSSResult(
            leaves=[ctx],
            tree=tree,
            total_usage=TokenUsage(input_tokens=600, output_tokens=250, cost=0.006),
            synthesis=synthesis,
        )

        result = _format_result(pss_result)

        assert "Synthesized findings" in result
        assert "85%" in result  # confidence
        assert "Dissenting Views" in result
        assert "Branch 1 disagreed" in result
        assert "Suggested Actions" in result
        assert "Fix bug A" in result

    def test_format_result_empty_list(self):
        """Test formatting an empty result."""
        from pss.mcp_server import _format_result

        result = _format_result([])
        assert "(No results)" in result

    def test_get_config_path_returns_path(self):
        """Test that config path function returns a Path."""
        from pss.mcp_server import _get_config_path
        from pathlib import Path

        path = _get_config_path()
        assert isinstance(path, Path)


class TestMCPServerCreation:
    """Tests for MCP server creation (requires mcp package)."""

    def test_create_server_without_mcp(self):
        """Test that server creation fails gracefully without MCP."""
        from pss import mcp_server

        # Temporarily set MCP_AVAILABLE to False
        original = mcp_server.MCP_AVAILABLE
        mcp_server.MCP_AVAILABLE = False

        try:
            with pytest.raises(RuntimeError, match="MCP support not installed"):
                mcp_server.create_server()
        finally:
            mcp_server.MCP_AVAILABLE = original

    @pytest.mark.skipif(
        not pytest.importorskip("mcp", reason="MCP package not installed"),
        reason="MCP package required"
    )
    def test_create_server_with_mcp(self):
        """Test that server creation works with MCP installed."""
        from pss.mcp_server import create_server, MCP_AVAILABLE

        if not MCP_AVAILABLE:
            pytest.skip("MCP package not installed")

        server = create_server()
        assert server is not None
        assert server.name == "pss"
