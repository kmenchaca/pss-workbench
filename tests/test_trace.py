"""Tests for the trace module."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from pss.trace import (
    ExplorationTrace,
    analyze_failure_effectiveness,
    analyze_signal_utility,
    analyze_spawn_confidence,
    analyze_velocity_patterns,
    build_trace_from_result,
    generate_trace_summary,
    list_traces,
    load_traces,
    save_trace,
    serialize_config,
    serialize_context,
    serialize_failure,
    serialize_search_tree,
    serialize_signal,
    serialize_synthesis,
    serialize_velocity_snapshot,
)
from pss.types import (
    Context,
    PSSResult,
    SearchTree,
    SpawnMetadata,
    SpawnReason,
    SynthesisResult,
    TerminationReason,
    TokenUsage,
    ToolTrace,
    VelocitySnapshot,
)
from pss.config import PSSConfig


class TestExplorationTrace:
    """Tests for ExplorationTrace dataclass."""

    def test_create_empty_trace(self):
        """Test creating an empty trace."""
        trace = ExplorationTrace()
        assert trace.id is not None
        assert len(trace.id) == 12
        assert trace.timestamp > 0
        assert trace.prompt == ""
        assert trace.leaves == []
        assert trace.num_leaves == 0

    def test_create_trace_with_data(self):
        """Test creating a trace with data."""
        trace = ExplorationTrace(
            prompt="Test prompt",
            num_leaves=3,
            total_usage={"input_tokens": 100, "output_tokens": 50, "cost": 0.01},
        )
        assert trace.prompt == "Test prompt"
        assert trace.num_leaves == 3
        assert trace.total_usage["cost"] == 0.01

    def test_add_event(self):
        """Test adding events to trace."""
        trace = ExplorationTrace()
        trace.add_event("gate_decision", {"branch_id": "root", "action": "continue"})

        assert len(trace.events) == 1
        assert trace.events[0]["type"] == "gate_decision"
        assert trace.events[0]["data"]["action"] == "continue"
        assert "timestamp" in trace.events[0]

    def test_to_dict(self):
        """Test converting trace to dict."""
        trace = ExplorationTrace(prompt="Test", num_leaves=2)
        d = trace.to_dict()

        assert isinstance(d, dict)
        assert d["prompt"] == "Test"
        assert d["num_leaves"] == 2
        assert "id" in d
        assert "timestamp" in d

    def test_save_and_load(self):
        """Test saving and loading a trace."""
        trace = ExplorationTrace(
            prompt="Test prompt",
            num_leaves=5,
            human_rating=4,
            human_notes="Good exploration",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_trace.json"
            trace.save(path)

            assert path.exists()

            # Load it back
            loaded = ExplorationTrace.load(path)
            assert loaded.id == trace.id
            assert loaded.prompt == trace.prompt
            assert loaded.num_leaves == trace.num_leaves
            assert loaded.human_rating == 4
            assert loaded.human_notes == "Good exploration"


class TestSerializeContext:
    """Tests for context serialization."""

    def test_serialize_basic_context(self):
        """Test serializing a basic context."""
        ctx = Context(
            id="root",
            parent_id=None,
            messages=[{"role": "user", "content": "Hello"}],
            token_count=100,
            token_budget=50000,
            status="running",
            gates_seen=1,
        )

        result = serialize_context(ctx)

        assert result["id"] == "root"
        assert result["parent_id"] is None
        assert result["status"] == "running"
        assert result["gates_seen"] == 1
        assert result["message_count"] == 1

    def test_serialize_context_with_output(self):
        """Test serializing a context that terminated with output."""
        ctx = Context(
            id="root->abc123",
            parent_id="root",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Response here"},
            ],
            status="terminated",
            output="Final output",
            termination_reason=TerminationReason.COMPLETED,
        )

        result = serialize_context(ctx)

        assert result["output"] == "Final output"
        assert result["termination_reason"] == "completed"
        assert result["last_assistant_message"] == "Response here"

    def test_serialize_context_with_spawn_metadata(self):
        """Test serializing a context with spawn metadata."""
        ctx = Context(
            id="root->abc123",
            parent_id="root",
            spawn_metadata=SpawnMetadata(
                spawn_reason=SpawnReason.GATE_BRANCH,
                spawn_edit="Explore alternative A",
                spawn_gate_number=2,
                spawn_confidence=0.8,
                parent_velocity=0.6,
            ),
        )

        result = serialize_context(ctx)

        assert result["spawn_metadata"]["spawn_reason"] == "gate_branch"
        assert result["spawn_metadata"]["spawn_edit"] == "Explore alternative A"
        assert result["spawn_metadata"]["spawn_confidence"] == 0.8

    def test_serialize_context_with_velocity_history(self):
        """Test serializing a context with velocity history."""
        ctx = Context(
            id="root",
            parent_id=None,
            velocity_history=[
                VelocitySnapshot(
                    timestamp=1000.0,
                    tokens_in_window=500,
                    novelty_rate=0.6,
                    assertion_rate=0.5,
                    uncertainty_trend=0.2,
                    repetition_score=0.1,
                    question_density=0.3,
                ),
            ],
        )

        result = serialize_context(ctx)

        assert len(result["velocity_history"]) == 1
        assert result["velocity_history"][0]["novelty_rate"] == 0.6

    def test_serialize_context_with_tool_traces(self):
        """Test serializing a context with tool traces."""
        ctx = Context(
            id="root",
            parent_id=None,
            tool_traces=[
                ToolTrace(
                    tool_name="read_file",
                    arguments={"path": "/tmp/test.txt"},
                    result="File contents here",
                    success=True,
                    timestamp=1000.0,
                    tokens_used=50,
                ),
            ],
        )

        result = serialize_context(ctx)

        assert len(result["tool_traces"]) == 1
        assert result["tool_traces"][0]["tool_name"] == "read_file"
        assert result["tool_traces"][0]["success"] is True


class TestSerializeVelocitySnapshot:
    """Tests for velocity snapshot serialization."""

    def test_serialize_velocity_snapshot(self):
        """Test serializing a velocity snapshot."""
        v = VelocitySnapshot(
            timestamp=1000.0,
            tokens_in_window=500,
            novelty_rate=0.6,
            assertion_rate=0.5,
            uncertainty_trend=0.2,
            repetition_score=0.1,
            question_density=0.3,
        )

        result = serialize_velocity_snapshot(v)

        assert result["timestamp"] == 1000.0
        assert result["tokens_in_window"] == 500
        assert result["novelty_rate"] == 0.6
        assert "overall_velocity" in result


class TestSerializeSearchTree:
    """Tests for search tree serialization."""

    def test_serialize_search_tree(self):
        """Test serializing a search tree."""
        tree = SearchTree()
        tree.root_id = "root"
        tree.contexts["root"] = Context(id="root", parent_id=None)
        tree.contexts["root->abc"] = Context(id="root->abc", parent_id="root")
        tree.leaves = ["root->abc"]

        result = serialize_search_tree(tree)

        assert result["root_id"] == "root"
        assert result["leaves"] == ["root->abc"]
        assert "root" in result["contexts"]
        assert "root->abc" in result["contexts"]


class TestSerializeSynthesis:
    """Tests for synthesis result serialization."""

    def test_serialize_synthesis(self):
        """Test serializing a synthesis result."""
        synthesis = SynthesisResult(
            unified_output="Combined output",
            confidence=0.9,
            evidence_summary={"finding1": 2, "finding2": 1},
            dissenting_views=["Alternative view"],
            action_plan=["Step 1", "Step 2"],
            usage=TokenUsage(input_tokens=100, output_tokens=200, cost=0.05),
        )

        result = serialize_synthesis(synthesis)

        assert result["unified_output"] == "Combined output"
        assert result["confidence"] == 0.9
        assert result["evidence_summary"]["finding1"] == 2
        assert result["action_plan"] == ["Step 1", "Step 2"]
        assert result["usage"]["cost"] == 0.05


class TestSerializeConfig:
    """Tests for config serialization."""

    def test_serialize_config(self):
        """Test serializing a config."""
        config = PSSConfig(
            provider="anthropic",
            model="claude-3-haiku",
            soft_gate_tokens=10000,
            hard_gate_tokens=40000,
            synthesis_enabled=True,
            synthesis_strategy="vote",
        )

        result = serialize_config(config)

        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-3-haiku"
        assert result["soft_gate_tokens"] == 10000
        assert result["synthesis_enabled"] is True
        assert result["synthesis_strategy"] == "vote"


class TestBuildTraceFromResult:
    """Tests for building traces from PSS results."""

    def test_build_trace_from_pss_result(self):
        """Test building a trace from a PSSResult."""
        # Create a mock PSSResult
        ctx1 = Context(
            id="root",
            parent_id=None,
            status="terminated",
            output="Output 1",
            termination_reason=TerminationReason.COMPLETED,
            usage=TokenUsage(input_tokens=100, output_tokens=50, cost=0.01),
        )
        ctx2 = Context(
            id="root->abc",
            parent_id="root",
            status="terminated",
            output="Output 2",
            termination_reason=TerminationReason.STUCK,
            usage=TokenUsage(input_tokens=150, output_tokens=75, cost=0.02),
        )

        tree = SearchTree()
        tree.contexts["root"] = ctx1
        tree.contexts["root->abc"] = ctx2
        tree.leaves = ["root", "root->abc"]

        result = PSSResult(
            leaves=[ctx1, ctx2],
            tree=tree,
            total_usage=TokenUsage(input_tokens=250, output_tokens=125, cost=0.03),
        )

        config = PSSConfig()

        trace = build_trace_from_result(
            prompt="Test prompt",
            config=config,
            result=result,
            elapsed_seconds=10.5,
            tree=tree,
        )

        assert trace.prompt == "Test prompt"
        assert trace.elapsed_seconds == 10.5
        assert trace.num_leaves == 2
        assert len(trace.leaves) == 2
        assert trace.total_usage["cost"] == 0.03
        assert trace.leaf_termination_reasons["completed"] == 1
        assert trace.leaf_termination_reasons["stuck"] == 1


class TestTraceFileOperations:
    """Tests for trace file operations."""

    def test_save_and_list_traces(self):
        """Test saving traces and listing them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir)

            # Save a few traces
            for i in range(3):
                trace = ExplorationTrace(prompt=f"Test {i}")
                save_trace(trace, trace_dir)
                time.sleep(0.1)  # Small delay for distinct timestamps

            # List them
            traces = list_traces(trace_dir)
            assert len(traces) == 3

            # Should be sorted by time (newest first)
            loaded = load_traces(trace_dir)
            assert len(loaded) == 3

    def test_list_traces_empty_dir(self):
        """Test listing traces from empty/nonexistent directory."""
        traces = list_traces(Path("/nonexistent/path"))
        assert traces == []

    def test_load_traces_with_limit(self):
        """Test loading traces with a limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir)

            # Save several traces
            for i in range(5):
                trace = ExplorationTrace(prompt=f"Test {i}")
                save_trace(trace, trace_dir)

            # Load with limit
            loaded = load_traces(trace_dir, limit=2)
            assert len(loaded) == 2


class TestAnalysisFunctions:
    """Tests for trace analysis functions."""

    def test_analyze_velocity_patterns(self):
        """Test velocity pattern analysis."""
        trace = ExplorationTrace(
            leaves=[
                {
                    "id": "root",
                    "termination_reason": "completed",
                },
                {
                    "id": "root->abc",
                    "termination_reason": "stuck",
                },
            ],
            velocity_snapshots={
                "root": [
                    {"overall_velocity": 0.7, "timestamp": 1000},
                    {"overall_velocity": 0.8, "timestamp": 2000},
                ],
                "root->abc": [
                    {"overall_velocity": 0.3, "timestamp": 1000},
                    {"overall_velocity": 0.2, "timestamp": 2000},
                ],
            },
        )

        result = analyze_velocity_patterns(trace)

        assert result["branches_with_velocity"] == 2
        assert result["avg_final_velocity"] == 0.5  # (0.8 + 0.2) / 2
        assert result["successful_branch_avg_velocity"] == 0.8
        assert result["stuck_branch_avg_velocity"] == 0.2

    def test_analyze_failure_effectiveness(self):
        """Test failure analysis."""
        trace = ExplorationTrace(
            failures=[
                {"failure_type": "dead_end", "scope": "local"},
                {"failure_type": "dead_end", "scope": "direction"},
                {"failure_type": "circular", "scope": "local"},
            ],
        )

        result = analyze_failure_effectiveness(trace)

        assert result["total_failures_extracted"] == 3
        assert result["failures_by_type"]["dead_end"] == 2
        assert result["failures_by_type"]["circular"] == 1
        assert result["failures_by_scope"]["local"] == 2
        assert result["failures_by_scope"]["direction"] == 1

    def test_analyze_signal_utility(self):
        """Test signal analysis."""
        trace = ExplorationTrace(
            signals=[
                {"signal_type": "discovery", "delivered_to": ["root->abc"]},
                {"signal_type": "discovery", "delivered_to": []},
                {"signal_type": "warning", "delivered_to": ["root->def"]},
            ],
        )

        result = analyze_signal_utility(trace)

        assert result["total_signals"] == 3
        assert result["signals_by_type"]["discovery"] == 2
        assert result["signals_by_type"]["warning"] == 1
        assert result["delivery_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_analyze_spawn_confidence(self):
        """Test spawn confidence analysis."""
        trace = ExplorationTrace(
            leaves=[
                {
                    "id": "root->abc",
                    "termination_reason": "completed",
                    "spawn_metadata": {"spawn_confidence": 0.8},
                },
                {
                    "id": "root->def",
                    "termination_reason": "stuck",
                    "spawn_metadata": {"spawn_confidence": 0.4},
                },
            ],
        )

        result = analyze_spawn_confidence(trace)

        assert result["branches_with_confidence"] == 2
        assert result["successful_avg_confidence"] == 0.8
        assert result["stuck_avg_confidence"] == 0.4


class TestGenerateTraceSummary:
    """Tests for trace summary generation."""

    def test_generate_summary(self):
        """Test generating a trace summary."""
        trace = ExplorationTrace(
            id="abc123456789",
            prompt="Test exploration prompt",
            elapsed_seconds=45.5,
            total_usage={"input_tokens": 1000, "output_tokens": 500, "cost": 0.05},
            total_contexts_created=5,
            num_leaves=3,
            leaf_termination_reasons={"completed": 2, "stuck": 1},
        )

        summary = generate_trace_summary(trace)

        assert "abc123456789" in summary
        assert "Test exploration" in summary
        assert "45.5s" in summary
        assert "$0.0500" in summary
        assert "completed: 2" in summary

    def test_generate_summary_with_rating(self):
        """Test generating a summary with human rating."""
        trace = ExplorationTrace(
            id="abc123456789",
            prompt="Test",
            human_rating=4,
            human_notes="Good exploration",
        )

        summary = generate_trace_summary(trace)

        assert "4/5" in summary
        assert "Good exploration" in summary
