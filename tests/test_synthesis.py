"""Tests for pss/synthesis.py - synthesis phase logic."""

from unittest.mock import MagicMock, patch

import pytest

from pss.config import PSSConfig
from pss.synthesis import (
    BranchSummary,
    _build_branch_summary,
    _build_synthesis_prompt,
    _extract_cross_branch_evidence,
    _format_evidence_summary,
    _summarize_tool_traces,
    synthesize,
)
from pss.types import Context, SynthesisResult, TokenUsage, ToolTrace


class TestToolTraceSummary:
    """Test tool trace summarization."""

    def test_empty_traces(self):
        """Should handle empty trace list."""
        result = _summarize_tool_traces([])
        assert "no tools used" in result.lower()

    def test_single_trace(self):
        """Should format a single tool trace."""
        traces = [
            ToolTrace(
                tool_name="read_file",
                arguments={"path": "test.py"},
                result="def hello(): pass",
                success=True,
                timestamp=1234567890.0,
            )
        ]
        result = _summarize_tool_traces(traces)
        assert "read_file" in result
        assert "test.py" in result
        assert "OK" in result
        assert "def hello" in result

    def test_failed_trace(self):
        """Should indicate failed tool calls."""
        traces = [
            ToolTrace(
                tool_name="read_file",
                arguments={"path": "missing.py"},
                result="",
                success=False,
                timestamp=1234567890.0,
            )
        ]
        result = _summarize_tool_traces(traces)
        assert "FAILED" in result

    def test_truncates_long_results(self):
        """Should truncate long results."""
        traces = [
            ToolTrace(
                tool_name="read_file",
                arguments={"path": "big.py"},
                result="x" * 1000,
                success=True,
                timestamp=1234567890.0,
            )
        ]
        result = _summarize_tool_traces(traces)
        assert "..." in result
        assert len(result) < 1000


class TestBranchSummary:
    """Test branch summary building."""

    def test_builds_summary_from_context(self):
        """Should extract relevant info from context."""
        ctx = Context(
            id="root->abc123",
            parent_id="root",
            messages=[],
            branch_reason="investigate auth module",
            output="Found bug in auth.py",
            tool_traces=[
                ToolTrace(
                    tool_name="read_file",
                    arguments={"path": "auth.py"},
                    result="code here",
                    success=True,
                    timestamp=1234567890.0,
                )
            ],
        )

        summary = _build_branch_summary(ctx)
        assert summary.branch_id == "root->abc123"
        assert summary.branch_reason == "investigate auth module"
        assert summary.final_output == "Found bug in auth.py"
        assert len(summary.tool_traces) == 1
        assert "read_file" in summary.tool_summary


class TestCrossBranchEvidence:
    """Test cross-branch evidence extraction."""

    def test_finds_common_patterns(self):
        """Should detect when multiple branches find same thing."""
        summaries = [
            BranchSummary(
                branch_id="root",
                branch_reason=None,
                tool_traces=[],
                final_output="Found a bug in the auth module",
                confidence=0.8,
                tool_summary="",
            ),
            BranchSummary(
                branch_id="root->abc",
                branch_reason="check auth",
                tool_traces=[],
                final_output="Bug detected in authentication",
                confidence=0.9,
                tool_summary="",
            ),
            BranchSummary(
                branch_id="root->def",
                branch_reason="check database",
                tool_traces=[],
                final_output="Database looks correct",
                confidence=0.7,
                tool_summary="",
            ),
        ]

        evidence = _extract_cross_branch_evidence(summaries)
        assert "found bug" in evidence
        assert len(evidence["found bug"]) == 2  # Two branches found bugs
        assert "verified correct" in evidence

    def test_handles_empty_outputs(self):
        """Should handle branches with no output."""
        summaries = [
            BranchSummary(
                branch_id="root",
                branch_reason=None,
                tool_traces=[],
                final_output=None,
                confidence=None,
                tool_summary="",
            ),
        ]

        evidence = _extract_cross_branch_evidence(summaries)
        assert evidence == {}


class TestEvidenceSummaryFormatting:
    """Test evidence summary formatting."""

    def test_formats_evidence(self):
        """Should format evidence with counts."""
        evidence = {
            "found bug": ["root", "root->abc"],
            "test passed": ["root->def"],
        }

        result = _format_evidence_summary(evidence, total_branches=3)
        assert "found bug" in result
        assert "2/3" in result
        assert "test passed" in result
        assert "1/3" in result

    def test_empty_evidence(self):
        """Should handle empty evidence."""
        result = _format_evidence_summary({}, total_branches=3)
        assert "no cross-branch" in result.lower()


class TestSynthesisPrompt:
    """Test synthesis prompt building."""

    def test_builds_complete_prompt(self):
        """Should include all required sections."""
        summaries = [
            BranchSummary(
                branch_id="root",
                branch_reason=None,
                tool_traces=[],
                final_output="Analysis complete",
                confidence=0.8,
                tool_summary="(no tools used)",
            ),
        ]
        evidence = {"found issue": ["root"]}

        prompt = _build_synthesis_prompt(
            "What's wrong with the code?",
            summaries,
            evidence,
            "merge",
        )

        assert "What's wrong with the code?" in prompt
        assert "BRANCH 1" in prompt
        assert "root" in prompt
        assert "Analysis complete" in prompt
        assert "CROSS-BRANCH EVIDENCE" in prompt
        assert "YOUR TASK" in prompt


class TestSynthesize:
    """Test the main synthesize function."""

    def test_returns_empty_result_for_no_leaves(self):
        """Should handle empty leaves list."""
        config = PSSConfig()
        result = synthesize([], "test prompt", config)

        assert result.unified_output == "No branches produced output."
        assert result.confidence == 0.0

    @patch("pss.synthesis.get_synthesis_provider")
    def test_calls_provider_with_correct_prompt(self, mock_get_provider):
        """Should construct and send synthesis prompt."""
        # Setup mock provider
        mock_provider = MagicMock()
        mock_provider.chat.return_value = (
            "Synthesized output",
            [],
            1000,
            TokenUsage(input_tokens=500, output_tokens=500, cost=0.01),
        )
        mock_get_provider.return_value = mock_provider

        # Create test leaves
        leaves = [
            Context(
                id="root",
                parent_id=None,
                messages=[],
                output="First finding",
                tool_traces=[],
            ),
            Context(
                id="root->abc",
                parent_id="root",
                messages=[],
                output="Second finding",
                branch_reason="alternative approach",
                tool_traces=[],
            ),
        ]

        config = PSSConfig(synthesis_enabled=True)
        result = synthesize(leaves, "Test prompt", config)

        # Verify provider was called
        mock_provider.chat.assert_called_once()
        call_args = mock_provider.chat.call_args
        messages = call_args[0][0]

        # Check prompt contains expected content
        prompt = messages[0]["content"]
        assert "Test prompt" in prompt
        assert "root" in prompt
        assert "First finding" in prompt
        assert "Second finding" in prompt

    @patch("pss.synthesis.get_synthesis_provider")
    def test_parses_tool_call_result(self, mock_get_provider):
        """Should parse structured result from tool call."""
        mock_provider = MagicMock()
        mock_provider.chat.return_value = (
            "",
            [
                {
                    "name": "synthesis_result",
                    "arguments": {
                        "unified_output": "Merged findings",
                        "confidence": 0.85,
                        "key_findings": ["finding1", "finding2"],
                        "action_plan": ["step1", "step2"],
                    },
                }
            ],
            1000,
            TokenUsage(input_tokens=500, output_tokens=500, cost=0.01),
        )
        mock_get_provider.return_value = mock_provider

        leaves = [
            Context(id="root", parent_id=None, messages=[], output="output", tool_traces=[]),
        ]

        config = PSSConfig(synthesis_enabled=True)
        result = synthesize(leaves, "Test", config)

        assert result.unified_output == "Merged findings"
        assert result.confidence == 0.85
        assert result.action_plan == ["step1", "step2"]

    @patch("pss.synthesis.get_synthesis_provider")
    def test_falls_back_to_text_response(self, mock_get_provider):
        """Should use text response if no tool call."""
        mock_provider = MagicMock()
        mock_provider.chat.return_value = (
            "Plain text synthesis",
            [],  # No tool calls
            1000,
            TokenUsage(input_tokens=500, output_tokens=500, cost=0.01),
        )
        mock_get_provider.return_value = mock_provider

        leaves = [
            Context(id="root", parent_id=None, messages=[], output="output", tool_traces=[]),
        ]

        config = PSSConfig(synthesis_enabled=True)
        result = synthesize(leaves, "Test", config)

        assert result.unified_output == "Plain text synthesis"
        assert result.confidence == 0.5  # Default confidence
