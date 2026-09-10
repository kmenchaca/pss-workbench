"""Tests for pss.velocity module."""

import pytest
import time

from pss.velocity import (
    VelocityTracker,
    _compute_novelty_rate,
    _compute_assertion_rate,
    _compute_uncertainty_trend,
    _compute_repetition_score,
    _compute_question_density,
    format_velocity_display,
)
from pss.types import VelocitySnapshot


class TestNoveltyRate:
    """Tests for novelty rate computation."""

    def test_high_novelty_with_new_terms(self):
        """New technical terms increase novelty."""
        prev = "We discussed the API."
        curr = "The HTTPClient handles ResponseError cases with TokenManager."

        novelty = _compute_novelty_rate(curr, prev)

        # Should be high since HTTPClient, ResponseError, TokenManager are new
        assert novelty > 0.5

    def test_low_novelty_with_repeated_terms(self):
        """Repeated terms decrease novelty."""
        prev = "The HTTPClient handles ResponseError with TokenManager."
        curr = "The HTTPClient also uses TokenManager for ResponseError recovery."

        novelty = _compute_novelty_rate(curr, prev)

        # Should be low since terms are repeated
        assert novelty < 0.5

    def test_novelty_with_empty_prev(self):
        """First chunk has word-level novelty."""
        curr = "The ConfigManager loads Settings from FileHandler."

        novelty = _compute_novelty_rate(curr, "")

        # Should have high novelty for first content
        assert novelty > 0.3

    def test_empty_text_returns_zero(self):
        """Empty text returns zero novelty."""
        assert _compute_novelty_rate("", "previous content") == 0.0
        assert _compute_novelty_rate("   ", "previous content") == 0.0


class TestAssertionRate:
    """Tests for assertion rate computation."""

    def test_high_assertion_rate(self):
        """Assertive language increases assertion rate."""
        text = """
        The solution is clear. This definitely works. Therefore, we should
        proceed. I found that the issue was in the config. Obviously, this
        must be fixed immediately.
        """

        rate = _compute_assertion_rate(text)

        assert rate > 0.6

    def test_low_assertion_rate(self):
        """Non-assertive language has low rate."""
        text = """
        Hmm, let me think about this. What if we tried something else?
        I wonder what would happen. Maybe we could explore this direction.
        """

        rate = _compute_assertion_rate(text)

        assert rate < 0.4

    def test_empty_text_returns_zero(self):
        """Empty text returns zero assertion rate."""
        assert _compute_assertion_rate("") == 0.0
        assert _compute_assertion_rate("   ") == 0.0


class TestUncertaintyTrend:
    """Tests for uncertainty trend computation."""

    def test_confident_language_positive_trend(self):
        """Confident language gives positive trend."""
        text = "This definitely works. I'm certain this is correct. Confirmed."

        trend = _compute_uncertainty_trend(text)

        assert trend > 0

    def test_hedging_language_negative_trend(self):
        """Hedging language gives negative trend."""
        text = "Maybe this could work. Perhaps we should try. Not sure though."

        trend = _compute_uncertainty_trend(text)

        assert trend < 0

    def test_neutral_language_near_zero(self):
        """Neutral language gives trend near zero."""
        text = "The function takes an input and returns output."

        trend = _compute_uncertainty_trend(text)

        assert -0.3 < trend < 0.3

    def test_trend_bounded(self):
        """Trend is bounded between -1 and 1."""
        very_uncertain = "maybe perhaps possibly might could uncertain unclear"
        very_confident = "definitely certainly clearly obviously confirmed verified"

        assert _compute_uncertainty_trend(very_uncertain) >= -1.0
        assert _compute_uncertainty_trend(very_confident) <= 1.0


class TestRepetitionScore:
    """Tests for repetition score computation."""

    def test_high_repetition(self):
        """Repetitive content has high score."""
        prev = "The function processes the input data and returns results."
        curr = "The function processes the input data and stores results."

        score = _compute_repetition_score(curr, prev)

        assert score > 0.3

    def test_low_repetition(self):
        """Novel content has low score."""
        prev = "Let me analyze the database schema."
        curr = "The authentication flow uses OAuth tokens."

        score = _compute_repetition_score(curr, prev)

        assert score < 0.3

    def test_circular_patterns_detected(self):
        """Explicit circular patterns increase score."""
        text = "As I said before, let me try again with the same approach."

        score = _compute_repetition_score(text, "different content")

        assert score > 0.5

    def test_empty_text_returns_zero(self):
        """Empty text returns zero repetition."""
        assert _compute_repetition_score("", "prev") == 0.0
        assert _compute_repetition_score("curr", "") == 0.0


class TestQuestionDensity:
    """Tests for question density computation."""

    def test_high_question_density(self):
        """Questions increase density."""
        text = """
        What should we do? How does this work? Why isn't it responding?
        Where is the config file? Which option should we choose?
        """

        density = _compute_question_density(text)

        assert density > 0.5

    def test_low_question_density(self):
        """Statements have low density."""
        text = """
        The system processes requests. It validates input data.
        Then it stores results in the database. Finally it returns a response.
        """

        density = _compute_question_density(text)

        assert density < 0.3

    def test_empty_text_returns_zero(self):
        """Empty text returns zero question density."""
        assert _compute_question_density("") == 0.0
        assert _compute_question_density("   ") == 0.0


class TestVelocitySnapshot:
    """Tests for VelocitySnapshot overall_velocity computation."""

    def test_high_velocity_snapshot(self):
        """High-quality metrics produce high velocity."""
        snapshot = VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.8,
            assertion_rate=0.7,
            uncertainty_trend=0.5,  # positive = confident
            repetition_score=0.1,  # low = good
            question_density=0.1,  # low = focused
        )

        velocity = snapshot.overall_velocity

        assert velocity > 0.6

    def test_low_velocity_snapshot(self):
        """Poor metrics produce low velocity."""
        snapshot = VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.1,
            assertion_rate=0.2,
            uncertainty_trend=-0.5,  # negative = uncertain
            repetition_score=0.8,  # high = repetitive
            question_density=0.8,  # high = asking not telling
        )

        velocity = snapshot.overall_velocity

        assert velocity < 0.4

    def test_velocity_bounded(self):
        """Velocity is bounded between 0 and 1."""
        # Best case
        best = VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=1.0,
            assertion_rate=1.0,
            uncertainty_trend=1.0,
            repetition_score=0.0,
            question_density=0.0,
        )

        # Worst case
        worst = VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.0,
            assertion_rate=0.0,
            uncertainty_trend=-1.0,
            repetition_score=1.0,
            question_density=1.0,
        )

        assert 0.0 <= best.overall_velocity <= 1.0
        assert 0.0 <= worst.overall_velocity <= 1.0


class TestVelocityTracker:
    """Tests for VelocityTracker."""

    def test_measure_adds_snapshot(self):
        """Measuring adds snapshot to history."""
        tracker = VelocityTracker()

        tracker.measure(
            "The HTTPClient handles requests efficiently.",
            tokens_in_chunk=100,
            prev_window_text="",
        )

        assert len(tracker.snapshots) == 1

    def test_should_measure_interval(self):
        """Should measure respects interval."""
        tracker = VelocityTracker(measurement_interval_tokens=500)

        assert tracker.should_measure(600)
        assert tracker.should_measure(500)
        assert not tracker.should_measure(400)

    def test_get_current_velocity(self):
        """Get current velocity returns latest snapshot."""
        tracker = VelocityTracker()

        # Default before any measurements
        assert tracker.get_current_velocity() == 0.5

        tracker.measure("Some assertive content.", 100, "")
        velocity = tracker.get_current_velocity()

        assert 0.0 <= velocity <= 1.0

    def test_get_trend(self):
        """Get trend computes velocity direction."""
        tracker = VelocityTracker()

        # Add declining snapshots
        tracker.snapshots = [
            VelocitySnapshot(
                timestamp=time.time(), tokens_in_window=100,
                novelty_rate=0.8, assertion_rate=0.7, uncertainty_trend=0.5,
                repetition_score=0.1, question_density=0.1
            ),
            VelocitySnapshot(
                timestamp=time.time(), tokens_in_window=100,
                novelty_rate=0.5, assertion_rate=0.5, uncertainty_trend=0.0,
                repetition_score=0.3, question_density=0.3
            ),
            VelocitySnapshot(
                timestamp=time.time(), tokens_in_window=100,
                novelty_rate=0.2, assertion_rate=0.2, uncertainty_trend=-0.5,
                repetition_score=0.6, question_density=0.5
            ),
        ]

        trend = tracker.get_trend()

        # Should be negative (declining)
        assert trend < 0

    def test_get_average_velocity(self):
        """Get average velocity over recent snapshots."""
        tracker = VelocityTracker()

        tracker.snapshots = [
            VelocitySnapshot(
                timestamp=time.time(), tokens_in_window=100,
                novelty_rate=0.6, assertion_rate=0.6, uncertainty_trend=0.0,
                repetition_score=0.2, question_density=0.2
            ),
            VelocitySnapshot(
                timestamp=time.time(), tokens_in_window=100,
                novelty_rate=0.6, assertion_rate=0.6, uncertainty_trend=0.0,
                repetition_score=0.2, question_density=0.2
            ),
        ]

        avg = tracker.get_average_velocity()

        assert 0.4 < avg < 0.7

    def test_is_plateauing(self):
        """Detects plateau when velocity is low and stable."""
        tracker = VelocityTracker()

        # Add low, stable velocity snapshots
        for _ in range(3):
            tracker.snapshots.append(VelocitySnapshot(
                timestamp=time.time(), tokens_in_window=100,
                novelty_rate=0.1, assertion_rate=0.1, uncertainty_trend=-0.2,
                repetition_score=0.7, question_density=0.6
            ))

        assert tracker.is_plateauing()

    def test_not_plateauing_with_high_velocity(self):
        """Not plateauing when velocity is high."""
        tracker = VelocityTracker()

        for _ in range(3):
            tracker.snapshots.append(VelocitySnapshot(
                timestamp=time.time(), tokens_in_window=100,
                novelty_rate=0.8, assertion_rate=0.7, uncertainty_trend=0.3,
                repetition_score=0.1, question_density=0.1
            ))

        assert not tracker.is_plateauing()

    def test_reset(self):
        """Reset clears all state."""
        tracker = VelocityTracker()

        tracker.measure("content", 100, "")
        tracker.measure("more content", 100, "content")

        tracker.reset()

        assert len(tracker.snapshots) == 0
        assert tracker._total_tokens_seen == 0


class TestFormatVelocityDisplay:
    """Tests for velocity display formatting."""

    def test_format_high_velocity(self):
        """High velocity shows appropriate status."""
        snapshot = VelocitySnapshot(
            timestamp=time.time(), tokens_in_window=100,
            novelty_rate=0.8, assertion_rate=0.7, uncertainty_trend=0.5,
            repetition_score=0.1, question_density=0.1
        )

        display = format_velocity_display(snapshot)

        assert "high" in display
        assert "Velocity:" in display
        assert "Novelty:" in display

    def test_format_low_velocity(self):
        """Low velocity shows appropriate status."""
        snapshot = VelocitySnapshot(
            timestamp=time.time(), tokens_in_window=100,
            novelty_rate=0.1, assertion_rate=0.2, uncertainty_trend=-0.5,
            repetition_score=0.8, question_density=0.7
        )

        display = format_velocity_display(snapshot)

        assert "low" in display

    def test_format_shows_all_metrics(self):
        """Display shows all metric components."""
        snapshot = VelocitySnapshot(
            timestamp=time.time(), tokens_in_window=100,
            novelty_rate=0.5, assertion_rate=0.5, uncertainty_trend=0.0,
            repetition_score=0.5, question_density=0.5
        )

        display = format_velocity_display(snapshot)

        assert "Novelty" in display
        assert "Assertions" in display
        assert "Repetition" in display
        assert "Questions" in display
