"""Tests for gate injection and parsing."""

from pss.gates import (
    CHECKPOINT_DECISION_TOOL,
    create_gate_prompt,
    parse_gate_decision,
)
from pss.types import Context, DiversityState
from pss.config import PSSConfig


def test_checkpoint_tool_schema():
    """Test that the checkpoint tool has required fields."""
    assert CHECKPOINT_DECISION_TOOL["name"] == "checkpoint_decision"
    assert "input_schema" in CHECKPOINT_DECISION_TOOL
    schema = CHECKPOINT_DECISION_TOOL["input_schema"]
    assert "action" in schema["properties"]
    assert schema["properties"]["action"]["enum"] == ["continue", "branch", "terminate"]


def test_first_gate_prompt_is_surprise():
    """Test that the first gate prompt doesn't reveal prior knowledge of gates."""
    ctx = Context(id="test", parent_id=None, gates_seen=0)
    prompt = create_gate_prompt(ctx)
    assert "---CHECKPOINT---" in prompt
    assert "You've been exploring for a while" in prompt


def test_subsequent_gate_prompt_is_shorter():
    """Test that subsequent gates (after the first 2) are more concise."""
    # First two gates (0, 1) use the early checkpoint prompt
    # Gate 2+ uses the shorter "Decision point" prompt
    ctx = Context(id="test", parent_id=None, gates_seen=2)
    prompt = create_gate_prompt(ctx)
    assert "---CHECKPOINT---" in prompt
    assert "Decision point" in prompt
    # Should be shorter than early gates
    first_ctx = Context(id="test", parent_id=None, gates_seen=0)
    first_prompt = create_gate_prompt(first_ctx)
    assert len(prompt) < len(first_prompt)


def test_force_flag_adds_urgency():
    """Test that force=True adds urgency to the prompt."""
    ctx = Context(id="test", parent_id=None, gates_seen=0)
    normal_prompt = create_gate_prompt(ctx, force=False)
    forced_prompt = create_gate_prompt(ctx, force=True)
    assert "MUST decide now" in forced_prompt
    assert "MUST decide now" not in normal_prompt


def test_parse_continue_decision():
    """Test parsing a continue decision."""
    tool_calls = [
        {
            "name": "checkpoint_decision",
            "arguments": {"action": "continue"},
        }
    ]
    decision = parse_gate_decision(tool_calls)
    assert decision is not None
    assert decision.action == "continue"
    assert decision.branches == []


def test_parse_branch_decision():
    """Test parsing a branch decision."""
    tool_calls = [
        {
            "name": "checkpoint_decision",
            "arguments": {
                "action": "branch",
                "branches": ["try approach A", "try approach B"],
                "also_continue": True,
                "reasoning": "multiple promising paths",
            },
        }
    ]
    decision = parse_gate_decision(tool_calls)
    assert decision is not None
    assert decision.action == "branch"
    assert len(decision.branches) == 2
    assert decision.also_continue is True
    assert decision.reasoning == "multiple promising paths"


def test_parse_terminate_decision():
    """Test parsing a terminate decision."""
    tool_calls = [
        {
            "name": "checkpoint_decision",
            "arguments": {
                "action": "terminate",
                "output": "Final answer: 42",
                "confidence": 0.95,
            },
        }
    ]
    decision = parse_gate_decision(tool_calls)
    assert decision is not None
    assert decision.action == "terminate"
    assert decision.output == "Final answer: 42"
    assert decision.confidence == 0.95


def test_parse_ignores_other_tools():
    """Test that other tool calls are ignored."""
    tool_calls = [
        {"name": "some_other_tool", "arguments": {}},
        {"name": "checkpoint_decision", "arguments": {"action": "continue"}},
    ]
    decision = parse_gate_decision(tool_calls)
    assert decision is not None
    assert decision.action == "continue"


def test_parse_no_checkpoint_returns_none():
    """Test that missing checkpoint_decision returns None."""
    tool_calls = [{"name": "some_other_tool", "arguments": {}}]
    decision = parse_gate_decision(tool_calls)
    assert decision is None


def test_parse_empty_returns_none():
    """Test that empty tool_calls returns None."""
    decision = parse_gate_decision([])
    assert decision is None


def test_continue_with_branches_becomes_branch():
    """Test that continue + branches auto-converts to branch + also_continue."""
    tool_calls = [
        {
            "name": "checkpoint_decision",
            "arguments": {
                "action": "continue",
                "branches": ["try approach A", "try approach B"],
            },
        }
    ]
    decision = parse_gate_decision(tool_calls)
    assert decision is not None
    # Should auto-convert to branch with also_continue=True
    assert decision.action == "branch"
    assert decision.also_continue is True
    assert len(decision.branches) == 2


def test_continue_with_empty_branches_stays_continue():
    """Test that continue without branches stays as continue."""
    tool_calls = [
        {
            "name": "checkpoint_decision",
            "arguments": {
                "action": "continue",
                "branches": [],
            },
        }
    ]
    decision = parse_gate_decision(tool_calls)
    assert decision is not None
    assert decision.action == "continue"
    assert decision.also_continue is False


def test_short_branches_are_filtered():
    """Test that very short branch descriptions are filtered out."""
    tool_calls = [
        {
            "name": "checkpoint_decision",
            "arguments": {
                "action": "branch",
                "branches": ["try A", "this is a valid branch description", "B"],
            },
        }
    ]
    decision = parse_gate_decision(tool_calls)
    assert decision is not None
    # Only the valid one should remain (>5 chars)
    assert len(decision.branches) == 1
    assert "valid branch" in decision.branches[0]


# v0.5: Diversity-aware gate prompts


def test_first_gate_has_no_diversity_info():
    """First gate (Heisenberg trick) should NOT include diversity info."""
    ctx = Context(id="test", parent_id=None, gates_seen=0)
    config = PSSConfig(
        diversity_aware_enabled=True,
        diversity_min_threshold=0.3,
        diversity_target=0.5,
    )
    diversity_state = DiversityState(
        current_score=0.2,
        interpretation="low",
        num_leaves=3,
    )

    prompt = create_gate_prompt(
        ctx,
        diversity_state=diversity_state,
        config=config,
    )

    # First gate should NOT have diversity guidance (Heisenberg trick)
    assert "Diversity" not in prompt
    assert "---CHECKPOINT---" in prompt


def test_subsequent_gate_includes_diversity_guidance():
    """Subsequent gates (after first 2) should include diversity guidance when enabled."""
    # First two gates (0, 1) are "early" and don't include diversity info
    # Gate 2+ is "subsequent" and includes diversity guidance
    ctx = Context(id="test", parent_id=None, gates_seen=2)
    config = PSSConfig(
        diversity_aware_enabled=True,
        diversity_min_threshold=0.3,
        diversity_target=0.5,
    )
    diversity_state = DiversityState(
        current_score=0.2,
        interpretation="low",
        num_leaves=3,
    )

    prompt = create_gate_prompt(
        ctx,
        diversity_state=diversity_state,
        config=config,
    )

    # Subsequent gates SHOULD have diversity guidance
    assert "Diversity Alert" in prompt
    assert "too similar" in prompt


def test_subsequent_gate_no_diversity_when_disabled():
    """Subsequent gates should NOT include diversity when disabled."""
    # Use gates_seen=2 to get "subsequent" prompt
    ctx = Context(id="test", parent_id=None, gates_seen=2)
    config = PSSConfig(
        diversity_aware_enabled=False,  # Disabled
    )
    diversity_state = DiversityState(
        current_score=0.2,
        interpretation="low",
        num_leaves=3,
    )

    prompt = create_gate_prompt(
        ctx,
        diversity_state=diversity_state,
        config=config,
    )

    # Should NOT have diversity guidance when disabled
    assert "Diversity" not in prompt


def test_gate_prompt_without_diversity_state():
    """Gate prompt should work without diversity state (backward compat)."""
    ctx = Context(id="test", parent_id=None, gates_seen=1)

    # Call without diversity_state or config
    prompt = create_gate_prompt(ctx)

    assert "---CHECKPOINT---" in prompt
    assert "Diversity" not in prompt
