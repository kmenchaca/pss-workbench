"""Tests for lazy branching (v1.3)."""

import pytest

from pss.lazy_branching import (
    BranchSignal,
    LazyBranchingConfig,
    analyze_initial_response,
    check_early_convergence,
    should_continue_branching,
)


class TestAnalyzeInitialResponse:
    """Tests for analyze_initial_response()."""

    def test_short_direct_answer_is_converged(self):
        """Short, direct answers should signal convergence."""
        response = "The capital of France is Paris."
        signal, confidence, reason = analyze_initial_response(response, token_count=300)
        assert signal == BranchSignal.CONVERGED
        assert confidence >= 0.5

    def test_direct_answer_pattern(self):
        """Direct answer patterns should signal convergence."""
        responses = [
            "The answer is 42.",
            "It is approximately 3.14159.",
            "Yes, that is correct.",
            "No, that would not work.",
        ]
        for response in responses:
            signal, _, _ = analyze_initial_response(response, token_count=300)
            assert signal == BranchSignal.CONVERGED, f"Expected CONVERGED for '{response}'"

    def test_multiple_approaches_is_diverging(self):
        """Responses mentioning multiple approaches should signal divergence."""
        responses = [
            "There are several approaches to this problem. First, we could...",
            "Here are some options to consider: 1. Option A, 2. Option B...",
            "On one hand, you could do X. On the other hand, Y might be better.",
        ]
        for response in responses:
            signal, _, _ = analyze_initial_response(response, token_count=500)
            assert signal == BranchSignal.DIVERGING, f"Expected DIVERGING for '{response[:50]}'"

    def test_exploration_language_is_diverging(self):
        """Exploration language should signal divergence."""
        responses = [
            "Let's explore the different possibilities here...",
            "We could also consider an alternative approach where...",
            "Another way to solve this would be to...",
        ]
        for response in responses:
            signal, _, _ = analyze_initial_response(response, token_count=500)
            assert signal == BranchSignal.DIVERGING, f"Expected DIVERGING for '{response[:50]}'"

    def test_low_velocity_suggests_convergence(self):
        """Low velocity should suggest convergence."""
        response = "This is a response."
        signal, confidence, _ = analyze_initial_response(
            response, token_count=300, velocity=0.3
        )
        assert signal == BranchSignal.CONVERGED
        assert confidence >= 0.5

    def test_high_velocity_suggests_divergence(self):
        """High velocity should suggest divergence."""
        response = "This is a longer exploratory response with many ideas..."
        signal, _, _ = analyze_initial_response(
            response, token_count=500, velocity=0.8
        )
        assert signal == BranchSignal.DIVERGING

    def test_too_early_is_uncertain(self):
        """Responses with too few tokens should be uncertain."""
        response = "The answer is 42."
        signal, _, reason = analyze_initial_response(response, token_count=50)
        assert signal == BranchSignal.UNCERTAIN
        assert "too_early" in reason


class TestShouldContinueBranching:
    """Tests for should_continue_branching()."""

    def test_converged_response_should_not_branch(self):
        """Converged responses should not continue branching."""
        response = "The answer is Paris."
        should_branch, reason = should_continue_branching(response, token_count=300)
        assert should_branch is False
        assert "converged" in reason

    def test_diverging_response_should_branch(self):
        """Diverging responses should continue branching."""
        response = "There are several approaches. First... Second... Third..."
        should_branch, reason = should_continue_branching(response, token_count=500)
        assert should_branch is True
        assert "diverging" in reason

    def test_uncertain_defaults_to_branching(self):
        """Uncertain responses should default to branching."""
        response = "Hello, this is a generic response."
        should_branch, reason = should_continue_branching(response, token_count=300)
        assert should_branch is True
        assert "uncertain" in reason


class TestCheckEarlyConvergence:
    """Tests for check_early_convergence()."""

    def test_short_low_velocity_is_converged(self):
        """Short response with low velocity should be converged."""
        response = "Paris."
        is_converged = check_early_convergence(response, token_count=250, velocity=0.3)
        assert is_converged is True

    def test_too_early_is_not_converged(self):
        """Too early to judge should not be converged."""
        response = "Paris."
        is_converged = check_early_convergence(response, token_count=100, velocity=0.3)
        assert is_converged is False

    def test_past_window_is_not_converged(self):
        """Past convergence window should not be converged."""
        response = "Paris."
        is_converged = check_early_convergence(response, token_count=1000, velocity=0.3)
        assert is_converged is False

    def test_high_confidence_pattern_is_converged(self):
        """High confidence patterns should trigger convergence."""
        response = "The answer is definitely 42."
        is_converged = check_early_convergence(response, token_count=300)
        assert is_converged is True


class TestLazyBranchingConfig:
    """Tests for LazyBranchingConfig."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = LazyBranchingConfig()
        assert config.enabled is True
        assert config.min_tokens_to_judge == 200
        assert config.convergence_window_tokens == 750
        assert config.convergence_velocity == 0.5

    def test_disabled_config_returns_uncertain(self):
        """Disabled config should always return uncertain."""
        config = LazyBranchingConfig(enabled=False)
        response = "The answer is 42."
        signal, _, reason = analyze_initial_response(response, token_count=300, config=config)
        assert signal == BranchSignal.UNCERTAIN
        assert "disabled" in reason


class TestRealWorldExamples:
    """Tests with real-world-like responses."""

    def test_factual_answer_converges(self):
        """A factual answer like 'capital of France' should converge."""
        response = """The capital of France is Paris. Paris has been the capital
        since the 10th century and is home to about 2 million people in the city
        proper, with over 12 million in the greater metropolitan area."""
        signal, _, _ = analyze_initial_response(response, token_count=400, velocity=0.4)
        assert signal == BranchSignal.CONVERGED

    def test_brainstorm_response_diverges(self):
        """A brainstorming response should diverge."""
        response = """There are many creative uses for a paperclip! Here are some ideas:

        1. Bookmark - bend it to mark your page
        2. Phone SIM ejector - the pointed end works perfectly
        3. Zipper pull - attach to a broken zipper
        4. Wire stripper - for thin wires
        5. Picture hanger - bend and tape to back of frame

        Let me explore some more unconventional uses..."""
        signal, _, _ = analyze_initial_response(response, token_count=500)
        assert signal == BranchSignal.DIVERGING

    def test_ambiguous_response_is_uncertain(self):
        """An ambiguous response should be uncertain."""
        response = """I'd be happy to help you with that. Could you provide more
        context about what you're trying to achieve? There might be different
        approaches depending on your specific situation."""
        signal, _, _ = analyze_initial_response(response, token_count=400)
        # Could be either uncertain or diverging (hedging language)
        assert signal in (BranchSignal.UNCERTAIN, BranchSignal.DIVERGING)
