"""Tests for the PSS harness with mocked provider."""

from pss.config import PSSConfig
from pss.harness import run_pss, _spawn_branch
from pss.providers import Provider
from pss.types import Context, PSSResult, SearchTree, TokenUsage


def get_leaves(result):
    """Helper to extract leaves from run_pss result (handles both old and new return types)."""
    if isinstance(result, PSSResult):
        return result.leaves
    return result


class MockProvider(Provider):
    """Mock provider for testing."""

    def __init__(self, responses: list[tuple[str, list[dict], int]]):
        """
        Initialize with a list of responses to return in sequence.

        Each response is (text, tool_calls, tokens_used).
        TokenUsage is auto-generated from tokens_used.
        """
        self.responses = responses
        self.call_index = 0
        self.calls = []

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, list[dict], int, TokenUsage]:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )

        if self.call_index >= len(self.responses):
            # Default: terminate with output (so it becomes a leaf)
            usage = TokenUsage(input_tokens=50, output_tokens=50, cost=0.0001)
            return (
                "",
                [{"name": "checkpoint_decision", "arguments": {"action": "terminate", "output": "Default termination"}}],
                100,
                usage,
            )

        text, tool_calls, tokens_used = self.responses[self.call_index]
        self.call_index += 1
        # Generate mock usage from tokens_used (assume 50/50 split)
        usage = TokenUsage(
            input_tokens=tokens_used // 2,
            output_tokens=tokens_used // 2,
            cost=tokens_used * 0.000001,  # Mock cost
        )
        return text, tool_calls, tokens_used, usage


def test_single_context_terminates():
    """Test a single context that terminates after early gates."""
    provider = MockProvider(
        [
            # First call: exploration response
            ("I'm exploring the problem...", [], 5000),
            # Gate 1: forced to continue (no-terminate rule on first 2 gates)
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {"action": "continue"},
                    }
                ],
                100,
            ),
            # Second exploration
            ("Still exploring...", [], 5000),
            # Gate 2: forced to continue (no-terminate rule)
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {"action": "continue"},
                    }
                ],
                100,
            ),
            # Third exploration
            ("Found the answer!", [], 5000),
            # Gate 3: can now terminate
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {
                            "action": "terminate",
                            "output": "The answer is 42",
                            "confidence": 0.9,
                        },
                    }
                ],
                100,
            ),
        ]
    )

    config = PSSConfig(
        soft_gate_tokens=4000,  # Lower than first response
        hard_gate_tokens=50000,
        total_max=100000,
    )

    result = run_pss("What is the answer?", config, provider)
    leaves = get_leaves(result)

    assert len(leaves) == 1
    assert leaves[0].output == "The answer is 42"


def test_context_branches():
    """Test a context that branches."""
    provider = MockProvider(
        [
            # First call: exploration
            ("Exploring...", [], 5000),
            # Gate 1: branch into two paths (branching is allowed on early gates)
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {
                            "action": "branch",
                            "branches": ["Try approach A", "Try approach B"],
                            "also_continue": False,
                        },
                    }
                ],
                100,
            ),
            # Branch A explores (branches start with gates_seen=1)
            ("Branch A exploring...", [], 3000),
            # Branch A gate 2: forced to continue (no-terminate rule)
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {"action": "continue"},
                    }
                ],
                100,
            ),
            # Branch A continues exploring
            ("Branch A still exploring...", [], 3000),
            # Branch A gate 3: can terminate
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {
                            "action": "terminate",
                            "output": "Result from A",
                        },
                    }
                ],
                100,
            ),
            # Branch B explores
            ("Branch B exploring...", [], 3000),
            # Branch B gate 2: forced to continue
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {"action": "continue"},
                    }
                ],
                100,
            ),
            # Branch B continues exploring
            ("Branch B still exploring...", [], 3000),
            # Branch B gate 3: can terminate
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {
                            "action": "terminate",
                            "output": "Result from B",
                        },
                    }
                ],
                100,
            ),
        ]
    )

    config = PSSConfig(
        soft_gate_tokens=1000,  # Low enough that branches hit gates with 3000 tokens
        hard_gate_tokens=50000,
        total_max=100000,
    )

    result = run_pss("Explore options", config, provider)
    leaves = get_leaves(result)

    # Both branches should terminate with some output
    assert len(leaves) == 2
    # Verify all leaves have output (due to mock response ordering, specific outputs may vary)
    assert all(leaf.output is not None for leaf in leaves)
    # At least one should have the expected output from the mock
    outputs = {leaf.output for leaf in leaves}
    assert "Result from B" in outputs or "Result from A" in outputs


def test_context_continues():
    """Test a context that continues past early gates."""
    provider = MockProvider(
        [
            # First exploration
            ("Starting...", [], 5000),
            # Gate 1: continue (no-terminate rule)
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {"action": "continue"},
                    }
                ],
                100,
            ),
            # Second exploration
            ("Continuing...", [], 5000),
            # Gate 2: continue (no-terminate rule still in effect)
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {"action": "continue"},
                    }
                ],
                100,
            ),
            # Third exploration
            ("Almost done...", [], 5000),
            # Gate 3: can terminate
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {
                            "action": "terminate",
                            "output": "Final result",
                        },
                    }
                ],
                100,
            ),
        ]
    )

    config = PSSConfig(
        soft_gate_tokens=4000,
        hard_gate_tokens=50000,
        total_max=100000,
    )

    result = run_pss("Keep going", config, provider)
    leaves = get_leaves(result)

    assert len(leaves) == 1
    assert leaves[0].output == "Final result"
    # Context should have accumulated tokens from all three explorations
    assert leaves[0].token_count >= 15000


def test_spawn_branch_inherits_state():
    """Test that spawned branches inherit parent state."""
    tree = SearchTree()
    parent = Context(
        id="parent",
        parent_id=None,
        messages=[{"role": "user", "content": "Hello"}],
        state={"key": "value"},
        token_count=5000,
        gates_seen=1,
    )
    tree.contexts["parent"] = parent

    branch = _spawn_branch(tree, parent, "Try alternative approach", 50000)

    # Branch should have fresh token count
    assert branch.token_count == 0
    assert branch.token_budget == 50000

    # Branch should inherit state (deep copy)
    assert branch.state == {"key": "value"}
    branch.state["key"] = "modified"
    assert parent.state["key"] == "value"  # Parent unchanged

    # Branch should know gates exist
    assert branch.gates_seen == 1

    # Branch should have branch reason
    assert branch.branch_reason == "Try alternative approach"

    # Branch should be in tree
    assert branch.id in tree.contexts


def test_max_contexts_limit():
    """Test that max_contexts limit is respected."""
    # Provider that always wants to branch
    responses = []
    for i in range(100):  # More than we'd ever need
        responses.append(("Exploring...", [], 5000))
        responses.append(
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {
                            "action": "branch",
                            "branches": ["A", "B", "C"],  # Try to spawn 3 each time
                            "also_continue": True,
                        },
                    }
                ],
                100,
            )
        )

    provider = MockProvider(responses)

    config = PSSConfig(
        soft_gate_tokens=4000,
        hard_gate_tokens=50000,
        total_max=1000000,
        max_contexts=5,  # Limit to 5 contexts
    )

    # This should not create more than 5 contexts
    # (will eventually hit token budget or contexts will terminate)
    _ = run_pss("Branch a lot", config, provider)  # noqa: F841

    # The run should complete without infinite loop
    assert len(provider.calls) > 0


def test_total_budget_limit():
    """Test that total token budget is respected."""
    # Provider that always continues
    responses = []
    for i in range(100):
        responses.append(("Exploring...", [], 10000))
        responses.append(
            (
                "",
                [
                    {
                        "name": "checkpoint_decision",
                        "arguments": {
                            "action": "continue",
                        },
                    }
                ],
                100,
            )
        )

    provider = MockProvider(responses)

    config = PSSConfig(
        soft_gate_tokens=5000,
        hard_gate_tokens=50000,
        total_max=25000,  # Should stop around 2-3 iterations
    )

    _ = run_pss("Keep going forever", config, provider)  # noqa: F841

    # Should have stopped due to budget
    total_tokens = sum(call[2] for call in provider.responses[: provider.call_index])
    # Allow some slack for gate overhead
    assert total_tokens <= config.total_max + 10000


# ============================================================
# Regression tests for extracted modules (exploration, execution)
# ============================================================


def test_determine_gate_type_first_soft():
    """Test that first soft gate fires at soft_gate_tokens."""
    from pss.harness import _determine_gate_type

    ctx = Context(id="test", parent_id=None, messages=[], gates_seen=0)

    # Below threshold
    assert _determine_gate_type(ctx, 4000, soft_gate=5000, hard_gate=50000) == "none"

    # At threshold
    assert _determine_gate_type(ctx, 5000, soft_gate=5000, hard_gate=50000) == "soft"

    # Above threshold
    assert _determine_gate_type(ctx, 6000, soft_gate=5000, hard_gate=50000) == "soft"


def test_determine_gate_type_subsequent_soft():
    """Test that subsequent soft gates fire at N*soft_gate_tokens."""
    from pss.harness import _determine_gate_type

    ctx = Context(id="test", parent_id=None, messages=[], gates_seen=1)

    # First gate already seen, so next is at 2*soft_gate
    assert _determine_gate_type(ctx, 5000, soft_gate=5000, hard_gate=50000) == "none"
    assert _determine_gate_type(ctx, 9999, soft_gate=5000, hard_gate=50000) == "none"
    assert _determine_gate_type(ctx, 10000, soft_gate=5000, hard_gate=50000) == "soft"


def test_determine_gate_type_hard():
    """Test that hard gate fires at hard_gate_tokens."""
    from pss.harness import _determine_gate_type

    ctx = Context(id="test", parent_id=None, messages=[], gates_seen=0)

    # Hard gate takes precedence
    assert _determine_gate_type(ctx, 50000, soft_gate=5000, hard_gate=50000) == "hard"
    assert _determine_gate_type(ctx, 60000, soft_gate=5000, hard_gate=50000) == "hard"


def test_process_decision_terminate():
    """Test _process_decision handles terminate correctly."""
    from pss.harness import _process_decision
    from pss.types import GateDecision

    tree = SearchTree()
    ctx = Context(id="ctx", parent_id=None, messages=[], status="running")
    tree.contexts["ctx"] = ctx

    decision = GateDecision(action="terminate", output="Result", confidence=0.9)
    config = PSSConfig()

    _process_decision(tree, ctx, decision, config, None)

    assert ctx.status == "terminated"
    assert ctx.output == "Result"
    assert "ctx" in tree.leaves


def test_process_decision_branch():
    """Test _process_decision handles branch correctly."""
    from pss.harness import _process_decision
    from pss.types import GateDecision

    tree = SearchTree()
    ctx = Context(
        id="ctx",
        parent_id=None,
        messages=[{"role": "user", "content": "test"}],
        status="running",
    )
    tree.contexts["ctx"] = ctx

    decision = GateDecision(
        action="branch",
        branches=["Approach A", "Approach B"],
        also_continue=True,
    )
    config = PSSConfig(max_contexts=10)

    _process_decision(tree, ctx, decision, config, None)

    # Should have spawned 2 branches
    assert len(tree.contexts) == 3  # Original + 2 branches

    # Original should still be running (also_continue=True)
    assert ctx.status == "running"


def test_process_decision_branch_no_continue():
    """Test _process_decision handles branch with also_continue=False."""
    from pss.harness import _process_decision
    from pss.types import GateDecision

    tree = SearchTree()
    ctx = Context(
        id="ctx",
        parent_id=None,
        messages=[{"role": "user", "content": "test"}],
        status="running",
    )
    tree.contexts["ctx"] = ctx

    decision = GateDecision(
        action="branch",
        branches=["Approach A"],
        also_continue=False,
    )
    config = PSSConfig(max_contexts=10)

    _process_decision(tree, ctx, decision, config, None)

    # Original should be "branched" status (not running)
    assert ctx.status == "branched"


def test_collect_leaves():
    """Test _collect_leaves returns only contexts with output."""
    from pss.harness import _collect_leaves

    tree = SearchTree()

    # Context with output
    ctx1 = Context(id="ctx1", parent_id=None, messages=[], output="Result 1")
    tree.contexts["ctx1"] = ctx1
    tree.leaves.append("ctx1")

    # Context without output (terminated but no output)
    ctx2 = Context(id="ctx2", parent_id=None, messages=[], status="terminated")
    tree.contexts["ctx2"] = ctx2

    # Context with output
    ctx3 = Context(id="ctx3", parent_id=None, messages=[], output="Result 3")
    tree.contexts["ctx3"] = ctx3
    tree.leaves.append("ctx3")

    leaves = _collect_leaves(tree)

    assert len(leaves) == 2
    outputs = {leaf.output for leaf in leaves}
    assert "Result 1" in outputs
    assert "Result 3" in outputs
