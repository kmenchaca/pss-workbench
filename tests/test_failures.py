"""Tests for pss.failures module."""

import pytest

from pss.failures import (
    FailureRegistry,
    FailureScope,
    FailureSummary,
    FailureType,
    extract_failure_from_context,
    format_failures_for_injection,
    _detect_failure_type,
    _extract_failure_reason,
    _calculate_failure_confidence,
)
from pss.types import Context, TerminationReason


def make_failure(
    branch_id: str = "test_branch",
    failure_type: FailureType = FailureType.DEAD_END,
    what_attempted: str = "test approach",
    why_failed: str = "it didn't work",
    confidence: float = 0.7,
    scope: FailureScope = FailureScope.LOCAL,
    branch_path: list[str] | None = None,
) -> FailureSummary:
    """Helper to create a FailureSummary for testing."""
    return FailureSummary(
        branch_id=branch_id,
        failure_type=failure_type,
        what_was_attempted=what_attempted,
        why_it_failed=why_failed,
        evidence=["test evidence"],
        suggested_avoidance=f"Avoid {what_attempted}",
        confidence=confidence,
        scope=scope,
        branch_path=branch_path or [],
    )


def make_context(
    id: str,
    branch_reason: str = "test direction",
    output: str | None = None,
    termination_reason: TerminationReason | None = None,
    messages: list[dict] | None = None,
) -> Context:
    """Helper to create a Context for testing."""
    return Context(
        id=id,
        parent_id=None,
        branch_reason=branch_reason,
        output=output,
        termination_reason=termination_reason,
        messages=messages or [],
    )


class TestFailureRegistry:
    """Tests for FailureRegistry operations."""

    def test_add_failure(self):
        """Can add a failure to the registry."""
        registry = FailureRegistry()
        failure = make_failure()

        registry.add_failure(failure)

        assert len(registry.failures) == 1
        assert registry.failures[0] == failure

    def test_add_merges_similar_failures(self):
        """Similar failures are merged instead of duplicated."""
        registry = FailureRegistry()

        f1 = make_failure(branch_id="branch1", what_attempted="try OAuth authentication")
        original_confidence = f1.confidence
        f2 = make_failure(branch_id="branch2", what_attempted="try OAuth authentication")

        registry.add_failure(f1)
        registry.add_failure(f2)

        # Should merge into one
        assert len(registry.failures) == 1
        # Branch ID should include both
        assert "branch1" in registry.failures[0].branch_id
        assert "branch2" in registry.failures[0].branch_id
        # Confidence should increase from original
        assert registry.failures[0].confidence > original_confidence

    def test_merge_escalates_scope(self):
        """Multiple siblings failing escalates to DIRECTION scope."""
        registry = FailureRegistry()

        f1 = make_failure(branch_id="b1", what_attempted="use PKCE flow", scope=FailureScope.LOCAL)
        f2 = make_failure(branch_id="b2", what_attempted="use PKCE flow", scope=FailureScope.LOCAL)

        registry.add_failure(f1)
        registry.add_failure(f2)

        # Should escalate to DIRECTION
        assert registry.failures[0].scope == FailureScope.DIRECTION

    def test_get_relevant_failures_keyword_match(self):
        """Failures are returned based on keyword matching."""
        registry = FailureRegistry()

        registry.add_failure(make_failure(what_attempted="implement OAuth login"))
        registry.add_failure(make_failure(what_attempted="connect to database"))

        # Search for OAuth-related
        results = registry.get_relevant_failures("try OAuth authentication", max_failures=5)

        assert len(results) == 1
        assert "OAuth" in results[0].what_was_attempted

    def test_get_relevant_failures_respects_max(self):
        """Max failures limit is respected."""
        registry = FailureRegistry()

        for i in range(10):
            registry.add_failure(make_failure(
                branch_id=f"b{i}",
                what_attempted=f"approach {i} for authentication",
            ))

        results = registry.get_relevant_failures("authentication", max_failures=3)

        assert len(results) <= 3

    def test_get_global_failures(self):
        """Global failures are returned regardless of direction."""
        registry = FailureRegistry()

        local = make_failure(what_attempted="local thing", scope=FailureScope.LOCAL)
        global_f = make_failure(what_attempted="API down", scope=FailureScope.GLOBAL)

        registry.add_failure(local)
        registry.add_failure(global_f)

        results = registry.get_global_failures()

        assert len(results) == 1
        assert results[0].scope == FailureScope.GLOBAL

    def test_age_all_advances_lifecycle(self):
        """Aging advances failures through lifecycle."""
        registry = FailureRegistry(stale_after_branches=2, archive_after_branches=4)
        failure = make_failure()
        registry.add_failure(failure)

        assert failure.status == "active"

        registry.age_all()
        registry.age_all()
        assert failure.status == "stale"

        registry.age_all()
        registry.age_all()
        assert failure.status == "archived"

    def test_archived_failures_excluded_from_relevance(self):
        """Archived failures are not returned in relevance search."""
        registry = FailureRegistry(archive_after_branches=1)
        failure = make_failure(what_attempted="test approach")
        registry.add_failure(failure)

        # Age to archived
        registry.age_all()
        registry.age_all()
        assert failure.status == "archived"

        # Should not be returned
        results = registry.get_relevant_failures("test approach", max_failures=5)
        assert len(results) == 0


class TestContradictionHandling:
    """Tests for contradiction detection and handling."""

    def test_contradiction_archives_local_failure(self):
        """Successful branch contradicting LOCAL failure archives it."""
        registry = FailureRegistry()
        failure = make_failure(
            what_attempted="OAuth flow",
            scope=FailureScope.LOCAL,
            branch_path=["auth", "OAuth"],
        )
        registry.add_failure(failure)

        contradicted = registry.handle_contradiction(
            "success_branch",
            ["authentication", "OAuth", "token refresh"],
        )

        assert len(contradicted) == 1
        assert failure.status == "archived"
        assert "Contradicted by success_branch" in failure.evidence

    def test_contradiction_downgrades_direction_scope(self):
        """Successful branch contradicting DIRECTION failure downgrades it."""
        registry = FailureRegistry()
        failure = make_failure(
            what_attempted="OAuth approach",
            scope=FailureScope.DIRECTION,
            confidence=0.8,
            branch_path=["auth", "OAuth"],
        )
        registry.add_failure(failure)

        registry.handle_contradiction(
            "success_branch",
            ["authentication", "OAuth", "success"],
        )

        assert failure.scope == FailureScope.LOCAL
        assert failure.confidence == 0.4  # halved

    def test_contradiction_downgrades_global_to_direction(self):
        """GLOBAL failures downgrade to DIRECTION."""
        registry = FailureRegistry()
        failure = make_failure(
            scope=FailureScope.GLOBAL,
            branch_path=["api", "request"],
        )
        registry.add_failure(failure)

        registry.handle_contradiction(
            "success_branch",
            ["api", "request", "retry"],
        )

        assert failure.scope == FailureScope.DIRECTION

    def test_no_contradiction_for_different_paths(self):
        """Non-overlapping paths don't trigger contradiction."""
        registry = FailureRegistry()
        failure = make_failure(
            branch_path=["database", "connection"],
        )
        registry.add_failure(failure)

        contradicted = registry.handle_contradiction(
            "success_branch",
            ["api", "authentication"],
        )

        assert len(contradicted) == 0
        assert failure.status == "active"


class TestFailureExtraction:
    """Tests for extracting failures from contexts."""

    def test_only_extracts_from_stuck_contexts(self):
        """Extraction only works for STUCK/KILLED termination."""
        # COMPLETED context - should return None
        completed = make_context(
            "c1",
            output="success",
            termination_reason=TerminationReason.COMPLETED,
        )
        assert extract_failure_from_context(completed) is None

        # BUDGET context - should return None
        budget = make_context(
            "c2",
            termination_reason=TerminationReason.BUDGET,
        )
        assert extract_failure_from_context(budget) is None

    def test_extracts_from_stuck_context(self):
        """Extracts failure from STUCK context with messages."""
        ctx = make_context(
            "stuck_branch",
            branch_reason="try recursive approach",
            termination_reason=TerminationReason.STUCK,
            messages=[
                {"role": "assistant", "content": "I'm going in circles. This approach keeps failing."},
            ],
        )

        failure = extract_failure_from_context(ctx)

        assert failure is not None
        assert failure.branch_id == "stuck_branch"
        assert failure.what_was_attempted == "try recursive approach"

    def test_returns_none_for_empty_messages(self):
        """Returns None if no assistant messages to analyze."""
        ctx = make_context(
            "empty",
            termination_reason=TerminationReason.STUCK,
            messages=[],
        )

        assert extract_failure_from_context(ctx) is None


class TestFailureTypeDetection:
    """Tests for failure type detection patterns."""

    def test_detects_circular(self):
        """Detects circular/repetitive patterns."""
        assert _detect_failure_type("i'm going in circles here") == FailureType.CIRCULAR
        assert _detect_failure_type("tried this before with same result") == FailureType.CIRCULAR

    def test_detects_blocked(self):
        """Detects blocked/external issues."""
        assert _detect_failure_type("permission denied error") == FailureType.BLOCKED
        assert _detect_failure_type("api error 503") == FailureType.BLOCKED
        assert _detect_failure_type("connection timeout") == FailureType.BLOCKED

    def test_detects_invalid_assumption(self):
        """Detects invalid assumption patterns."""
        assert _detect_failure_type("my assumption was wrong") == FailureType.INVALID_ASSUMPTION
        assert _detect_failure_type("the api doesn't exist") == FailureType.INVALID_ASSUMPTION

    def test_detects_contradicted(self):
        """Detects contradiction patterns."""
        assert _detect_failure_type("this contradicts the requirements") == FailureType.CONTRADICTED

    def test_defaults_to_dead_end(self):
        """Unknown patterns default to DEAD_END."""
        assert _detect_failure_type("this just doesn't work") == FailureType.DEAD_END


class TestConfidenceCalculation:
    """Tests for failure confidence calculation."""

    def test_explicit_language_increases_confidence(self):
        """Definite language increases confidence."""
        high_conf = _calculate_failure_confidence(
            "this definitely won't work. confirmed failure.",
            FailureType.DEAD_END,
        )
        low_conf = _calculate_failure_confidence(
            "this might not work",
            FailureType.DEAD_END,
        )

        assert high_conf > low_conf

    def test_hedging_decreases_confidence(self):
        """Hedging language decreases confidence."""
        hedged = _calculate_failure_confidence(
            "maybe this won't work, not sure",
            FailureType.DEAD_END,
        )
        direct = _calculate_failure_confidence(
            "this failed with an error",
            FailureType.DEAD_END,
        )

        assert hedged < direct

    def test_multiple_signals_increase_confidence(self):
        """Multiple failure signals increase confidence."""
        many_signals = _calculate_failure_confidence(
            "failed with error. cannot proceed. won't work.",
            FailureType.DEAD_END,
        )
        few_signals = _calculate_failure_confidence(
            "didn't succeed",
            FailureType.DEAD_END,
        )

        assert many_signals > few_signals

    def test_confidence_bounded(self):
        """Confidence is bounded between 0.2 and 0.95."""
        very_low = _calculate_failure_confidence(
            "maybe possibly uncertain might",
            FailureType.DEAD_END,
        )
        very_high = _calculate_failure_confidence(
            "definitely confirmed verified failed error cannot won't work",
            FailureType.BLOCKED,
        )

        assert very_low >= 0.2
        assert very_high <= 0.95


class TestFailureFormatting:
    """Tests for failure injection formatting."""

    def test_format_empty_list(self):
        """Empty list returns empty string."""
        result = format_failures_for_injection([])
        assert result == ""

    def test_format_includes_header(self):
        """Formatted output includes header."""
        failures = [make_failure()]
        result = format_failures_for_injection(failures)

        assert "Known Dead Ends" in result

    def test_format_includes_scope(self):
        """Formatted output includes scope."""
        failures = [make_failure(scope=FailureScope.DIRECTION)]
        result = format_failures_for_injection(failures)

        assert "DIRECTION" in result

    def test_format_includes_avoidance(self):
        """Formatted output includes avoidance suggestion."""
        failures = [make_failure(what_attempted="bad approach")]
        result = format_failures_for_injection(failures)

        assert "Avoid" in result

    def test_format_respects_max_chars(self):
        """Long failure lists are truncated."""
        failures = [
            make_failure(
                what_attempted=f"very long approach description number {i} " * 10
            )
            for i in range(10)
        ]

        result = format_failures_for_injection(failures, max_chars=500)

        assert len(result) <= 600  # some overhead allowed
        assert "omitted" in result


class TestFailureSummaryDataclass:
    """Tests for FailureSummary dataclass."""

    def test_default_values(self):
        """FailureSummary has sensible defaults."""
        failure = FailureSummary(
            branch_id="test",
            failure_type=FailureType.DEAD_END,
            what_was_attempted="test",
            why_it_failed="failed",
            evidence=[],
            suggested_avoidance="avoid",
            confidence=0.5,
            scope=FailureScope.LOCAL,
        )

        assert failure.age_branches == 0
        assert failure.status == "active"
        assert failure.branch_depth == 0
        assert failure.branch_path == []
        assert failure.sibling_outcomes == {}
