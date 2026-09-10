"""Tests for the exploration module."""


from pss.exploration import CONTINUATION_PROMPTS, determine_gate_type
from pss.types import Context


class TestDetermineGateType:
    """Tests for determine_gate_type function."""

    def test_no_gate_below_threshold(self):
        """Test no gate fires when below threshold."""
        ctx = Context(id="test", parent_id=None, messages=[], gates_seen=0)
        result = determine_gate_type(
            ctx, new_total=1000, soft_gate=5000, hard_gate=50000
        )
        assert result == "none"

    def test_first_soft_gate(self):
        """Test first soft gate fires at soft_gate threshold."""
        ctx = Context(id="test", parent_id=None, messages=[], gates_seen=0)

        # Just below
        result = determine_gate_type(
            ctx, new_total=4999, soft_gate=5000, hard_gate=50000
        )
        assert result == "none"

        # At threshold
        result = determine_gate_type(
            ctx, new_total=5000, soft_gate=5000, hard_gate=50000
        )
        assert result == "soft"

        # Above threshold
        result = determine_gate_type(
            ctx, new_total=7000, soft_gate=5000, hard_gate=50000
        )
        assert result == "soft"

    def test_subsequent_soft_gates(self):
        """Test subsequent soft gates fire at multiples of soft_gate."""
        # After first gate (gates_seen=1), next fires at 2*soft_gate
        ctx = Context(id="test", parent_id=None, messages=[], gates_seen=1)

        result = determine_gate_type(
            ctx, new_total=5000, soft_gate=5000, hard_gate=50000
        )
        assert result == "none"  # Below 2*5000=10000

        result = determine_gate_type(
            ctx, new_total=10000, soft_gate=5000, hard_gate=50000
        )
        assert result == "soft"

        # After second gate (gates_seen=2), next fires at 3*soft_gate=15000
        ctx = Context(id="test", parent_id=None, messages=[], gates_seen=2)

        result = determine_gate_type(
            ctx, new_total=14999, soft_gate=5000, hard_gate=50000
        )
        assert result == "none"

        result = determine_gate_type(
            ctx, new_total=15000, soft_gate=5000, hard_gate=50000
        )
        assert result == "soft"

    def test_hard_gate_takes_precedence(self):
        """Test hard gate fires and takes precedence over soft."""
        ctx = Context(id="test", parent_id=None, messages=[], gates_seen=0)

        # At hard gate threshold
        result = determine_gate_type(
            ctx, new_total=50000, soft_gate=5000, hard_gate=50000
        )
        assert result == "hard"

        # Above hard gate
        result = determine_gate_type(
            ctx, new_total=60000, soft_gate=5000, hard_gate=50000
        )
        assert result == "hard"

    def test_hard_gate_overrides_soft_gate_timing(self):
        """Test hard gate fires even when soft gate would also fire."""
        # If soft=5000, hard=10000, and tokens=10000
        # Both would fire, but hard takes precedence
        ctx = Context(id="test", parent_id=None, messages=[], gates_seen=1)
        result = determine_gate_type(
            ctx, new_total=10000, soft_gate=5000, hard_gate=10000
        )
        assert result == "hard"


class TestContinuationPrompts:
    """Tests for continuation prompt configuration."""

    def test_prompts_exist(self):
        """Test that continuation prompts are defined."""
        assert len(CONTINUATION_PROMPTS) > 0

    def test_prompts_are_strings(self):
        """Test that all prompts are non-empty strings."""
        for prompt in CONTINUATION_PROMPTS:
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_prompts_are_varied(self):
        """Test that prompts are not all identical."""
        # Should have at least 3 different prompts to avoid pattern detection
        unique_prompts = set(CONTINUATION_PROMPTS)
        assert len(unique_prompts) >= 3
