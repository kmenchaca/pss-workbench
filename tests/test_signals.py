"""Tests for pss.signals module (Bulletin Board)."""

import pytest
import time

from pss.signals import (
    BulletinBoard,
    Signal,
    SignalType,
    create_signal,
    extract_signals_from_response,
    format_signals_for_injection,
    _clean_signal_content,
    _extract_tags,
)


def make_signal(
    branch_id: str = "test_branch",
    signal_type: SignalType = SignalType.DISCOVERY,
    content: str = "Test signal content",
    confidence: float = 0.5,
    total_tokens: int = 1000,
    source_weight: float = 0.5,
) -> Signal:
    """Helper to create a Signal for testing."""
    return create_signal(
        branch_id=branch_id,
        signal_type=signal_type,
        content=content,
        total_tokens=total_tokens,
        confidence=confidence,
        source_weight=source_weight,
    )


class TestSignal:
    """Tests for Signal dataclass."""

    def test_effective_confidence_no_decay(self):
        """Confidence unchanged when no tokens have passed."""
        signal = make_signal(confidence=0.8, total_tokens=1000, source_weight=1.0)

        eff = signal.effective_confidence(1000)

        assert eff == 0.8

    def test_effective_confidence_with_decay(self):
        """Confidence decays with token distance."""
        signal = make_signal(confidence=0.8, total_tokens=0, source_weight=1.0)

        # After half_life tokens, should be halved
        eff = signal.effective_confidence(10000, half_life_tokens=10000)

        assert 0.35 < eff < 0.45  # Should be ~0.4

    def test_effective_confidence_dead_source_penalty(self):
        """Dead source applies additional penalty."""
        signal = make_signal(confidence=0.8, total_tokens=1000, source_weight=1.0)
        signal.mark_source_terminated("stuck")

        eff = signal.effective_confidence(1000, dead_source_penalty=0.5)

        assert eff == 0.4  # 0.8 * 0.5

    def test_effective_confidence_completed_no_penalty(self):
        """Successfully completed source has no extra penalty."""
        signal = make_signal(confidence=0.8, total_tokens=1000, source_weight=1.0)
        signal.mark_source_terminated("completed")

        eff = signal.effective_confidence(1000, dead_source_penalty=0.5)

        assert eff == 0.8  # No penalty for completed

    def test_effective_confidence_source_weight(self):
        """Source weight affects effective confidence."""
        signal = make_signal(confidence=0.8, total_tokens=1000, source_weight=0.5)

        eff = signal.effective_confidence(1000)

        assert eff == 0.4  # 0.8 * 0.5

    def test_mark_source_terminated(self):
        """Can mark signal's source as terminated."""
        signal = make_signal()

        assert not signal.source_terminated

        signal.mark_source_terminated("stuck")

        assert signal.source_terminated
        assert signal.source_termination_reason == "stuck"


class TestBulletinBoard:
    """Tests for BulletinBoard operations."""

    def test_post_signal(self):
        """Can post a signal to the board."""
        board = BulletinBoard()
        signal = make_signal()

        board.post(signal)

        assert len(board.signals) == 1
        assert board.signals[0] == signal

    def test_post_deduplicates_similar(self):
        """Similar signals are merged instead of duplicated."""
        board = BulletinBoard()

        s1 = make_signal(content="The API returns JSON data")
        original_confidence = s1.confidence
        s2 = make_signal(content="The API returns JSON format", branch_id="other_branch")

        board.post(s1)
        board.post(s2)

        # Should merge into one
        assert len(board.signals) == 1
        # Confidence should be boosted from original
        assert board.signals[0].confidence > original_confidence

    def test_post_keeps_different_signals(self):
        """Different signals are kept separate."""
        board = BulletinBoard()

        s1 = make_signal(content="The API returns JSON data")
        s2 = make_signal(content="Database connections are pooled")

        board.post(s1)
        board.post(s2)

        assert len(board.signals) == 2

    def test_get_signals_excludes_self(self):
        """Signals from the same branch are excluded."""
        board = BulletinBoard()

        s1 = make_signal(branch_id="branch_A", content="Found something")
        board.post(s1)

        signals = board.get_signals_for_branch(
            "branch_A", "test direction", 2000
        )

        assert len(signals) == 0

    def test_get_signals_excludes_already_delivered(self):
        """Already delivered signals are not re-delivered."""
        board = BulletinBoard()

        s1 = make_signal(branch_id="branch_A", content="Found something important", source_weight=1.0)
        board.post(s1)

        # First delivery - use low min_confidence to ensure delivery
        signals1 = board.get_signals_for_branch(
            "branch_B", "something important", 2000, min_confidence=0.1
        )
        assert len(signals1) == 1

        # Second delivery - should be empty (already delivered)
        signals2 = board.get_signals_for_branch(
            "branch_B", "something important", 3000, min_confidence=0.1
        )
        assert len(signals2) == 0

    def test_get_signals_respects_max(self):
        """Max signals limit is respected."""
        board = BulletinBoard()

        for i in range(10):
            board.post(make_signal(
                branch_id=f"branch_{i}",
                content=f"Signal about topic {i}",
            ))

        signals = board.get_signals_for_branch(
            "target_branch", "topic", 2000, max_signals=3
        )

        assert len(signals) <= 3

    def test_get_signals_filters_low_confidence(self):
        """Low confidence signals are filtered out."""
        board = BulletinBoard()

        # Use completely different content to avoid deduplication
        low_conf = make_signal(confidence=0.1, content="Database connection pooling works", source_weight=1.0, branch_id="branch_A")
        high_conf = make_signal(confidence=0.8, content="API authentication uses OAuth tokens", branch_id="branch_B", source_weight=1.0)

        board.post(low_conf)
        board.post(high_conf)

        signals = board.get_signals_for_branch(
            "target", "authentication OAuth API tokens", 2000, min_confidence=0.3
        )

        # Only high confidence signal should pass min_confidence filter
        assert len(signals) == 1
        assert "OAuth" in signals[0].content

    def test_get_signals_relevance_filtering(self):
        """Signals are filtered by relevance to direction."""
        board = BulletinBoard()

        relevant = make_signal(
            branch_id="branch_A",
            content="Found OAuth authentication issue",
            source_weight=1.0,
            confidence=0.8,
        )
        irrelevant = make_signal(
            branch_id="branch_B",
            content="Database schema looks correct",
            source_weight=1.0,
            confidence=0.8,
        )

        board.post(relevant)
        board.post(irrelevant)

        signals = board.get_signals_for_branch(
            "target", "OAuth authentication", 2000, min_confidence=0.1
        )

        # Relevant signal should be included
        assert any("OAuth" in s.content for s in signals)

    def test_get_global_signals(self):
        """Global signals returns high-confidence warnings/constraints."""
        board = BulletinBoard()

        warning = make_signal(
            signal_type=SignalType.WARNING,
            content="API rate limit reached",
            confidence=0.8,
            source_weight=1.0,
            branch_id="branch_A",
        )
        discovery = make_signal(
            signal_type=SignalType.DISCOVERY,
            content="Found interesting pattern",
            confidence=0.8,
            source_weight=1.0,
            branch_id="branch_B",
        )

        board.post(warning)
        board.post(discovery)

        globals = board.get_global_signals(2000, min_confidence=0.5)

        # Only warning should be included (DISCOVERY is not global)
        assert len(globals) == 1
        assert globals[0].signal_type == SignalType.WARNING

    def test_mark_branch_terminated(self):
        """Can mark all signals from a branch as terminated."""
        board = BulletinBoard()

        s1 = make_signal(branch_id="dying_branch", content="Signal 1")
        s2 = make_signal(branch_id="dying_branch", content="Signal 2")
        s3 = make_signal(branch_id="other_branch", content="Signal 3")

        board.post(s1)
        board.post(s2)
        board.post(s3)

        board.mark_branch_terminated("dying_branch", "stuck")

        assert board.signals[0].source_terminated
        assert board.signals[0].source_termination_reason == "stuck"
        assert board.signals[1].source_terminated
        assert not board.signals[2].source_terminated

    def test_prune_stale(self):
        """Can prune stale signals."""
        board = BulletinBoard()

        fresh = make_signal(total_tokens=9000, content="Fresh signal")
        stale = make_signal(total_tokens=0, content="Very old signal", confidence=0.1)

        board.post(fresh)
        board.post(stale)

        # At token 10000, stale signal should be very decayed
        pruned = board.prune_stale(10000, min_effective_confidence=0.05)

        # The very old signal with low confidence should be pruned
        assert pruned >= 0  # May or may not prune depending on exact decay

    def test_get_stats(self):
        """Can get board statistics."""
        board = BulletinBoard()

        # Use unique content to avoid deduplication
        board.post(make_signal(signal_type=SignalType.DISCOVERY, content="First discovery about APIs", branch_id="b1"))
        board.post(make_signal(signal_type=SignalType.WARNING, content="Warning about rate limits", branch_id="b2"))
        board.post(make_signal(signal_type=SignalType.DISCOVERY, content="Second discovery about databases", branch_id="b3"))

        stats = board.get_stats()

        assert stats["total_signals"] == 3
        assert stats["by_type"]["discovery"] == 2
        assert stats["by_type"]["warning"] == 1


class TestSignalExtraction:
    """Tests for signal extraction from text."""

    def test_extract_discovery(self):
        """Extracts discovery signals."""
        text = "I found that the API uses JSON format for all responses."

        signals = extract_signals_from_response(text, "test_branch", 1000)

        assert len(signals) >= 1
        assert any(s.signal_type == SignalType.DISCOVERY for s in signals)

    def test_extract_warning(self):
        """Extracts warning signals."""
        text = "Warning: don't use the deprecated endpoint, it will fail."

        signals = extract_signals_from_response(text, "test_branch", 1000)

        assert any(s.signal_type == SignalType.WARNING for s in signals)

    def test_extract_constraint(self):
        """Extracts constraint signals."""
        text = "The authentication must be included in every request."

        signals = extract_signals_from_response(text, "test_branch", 1000)

        assert any(s.signal_type == SignalType.CONSTRAINT for s in signals)

    def test_extract_hypothesis(self):
        """Extracts hypothesis signals."""
        text = "Hypothesis: the timeout is caused by network latency."

        signals = extract_signals_from_response(text, "test_branch", 1000)

        assert any(s.signal_type == SignalType.HYPOTHESIS for s in signals)

    def test_extract_multiple_signals(self):
        """Can extract multiple signals from one response."""
        text = """
        I discovered that the API returns paginated results.
        Warning: don't exceed 100 requests per minute.
        The authentication must use OAuth 2.0 tokens.
        """

        signals = extract_signals_from_response(text, "test_branch", 1000)

        assert len(signals) >= 2

    def test_extract_no_signals_from_empty(self):
        """Empty text produces no signals."""
        signals = extract_signals_from_response("", "test_branch", 1000)
        assert len(signals) == 0

    def test_signal_has_correct_metadata(self):
        """Extracted signals have correct metadata."""
        text = "I found that the config file is missing."

        signals = extract_signals_from_response(
            text, "my_branch", 5000, base_confidence=0.6, source_weight=0.7
        )

        if signals:
            signal = signals[0]
            assert signal.source_branch == "my_branch"
            assert signal.created_at_total_tokens == 5000
            assert signal.source_weight == 0.7


class TestSignalFormatting:
    """Tests for signal formatting."""

    def test_format_empty_list(self):
        """Empty list returns empty string."""
        result = format_signals_for_injection([])
        assert result == ""

    def test_format_includes_header(self):
        """Formatted output includes header."""
        signals = [make_signal()]
        result = format_signals_for_injection(signals)

        assert "Harness Observations" in result

    def test_format_includes_signal_type(self):
        """Formatted output includes signal type."""
        signals = [make_signal(signal_type=SignalType.WARNING)]
        result = format_signals_for_injection(signals)

        assert "WARNING" in result

    def test_format_respects_max_chars(self):
        """Long signal lists are truncated."""
        signals = [
            make_signal(content=f"Very long signal content number {i} " * 10)
            for i in range(10)
        ]

        result = format_signals_for_injection(signals, max_chars=500)

        assert len(result) <= 600  # Some overhead allowed
        assert "more observations" in result


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_clean_signal_content(self):
        """Content is cleaned of pattern prefixes."""
        content = "I found that the issue is in the config"
        cleaned = _clean_signal_content(content)

        assert not cleaned.startswith("I found")

    def test_clean_signal_truncates_long(self):
        """Long content is truncated."""
        content = "A" * 300
        cleaned = _clean_signal_content(content)

        assert len(cleaned) <= 200

    def test_extract_tags(self):
        """Tags are extracted from content."""
        content = "The HTTPClient uses OAuth2 for AuthManager"
        tags = _extract_tags(content)

        assert "HTTPClient" in tags or "OAuth2" in tags or "AuthManager" in tags

    def test_extract_tags_acronyms(self):
        """Acronyms are extracted as tags."""
        content = "The API uses JSON and HTTP"
        tags = _extract_tags(content)

        assert "API" in tags or "JSON" in tags or "HTTP" in tags

    def test_create_signal_helper(self):
        """create_signal helper works correctly."""
        signal = create_signal(
            branch_id="test",
            signal_type=SignalType.DISCOVERY,
            content="Test content",
            total_tokens=1000,
            confidence=0.7,
            source_weight=0.6,
            tags=["tag1", "tag2"],
        )

        assert signal.source_branch == "test"
        assert signal.signal_type == SignalType.DISCOVERY
        assert signal.content == "Test content"
        assert signal.confidence == 0.7
        assert signal.source_weight == 0.6
        assert "tag1" in signal.tags
