"""Lazy branching for PSS v1.3.

Instead of classifying prompts upfront, let the initial exploration
reveal whether branching is needed. A converged response plateaus
quickly with low velocity. A diverging response sustains exploration.

This is more accurate than prompt pattern-matching because it uses
the actual LLM response to determine intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class BranchSignal(Enum):
    """Signal from initial response analysis."""

    CONVERGED = "converged"  # Answer found, no branching needed
    DIVERGING = "diverging"  # Multiple directions, branch
    UNCERTAIN = "uncertain"  # Can't tell, default behavior


@dataclass
class LazyBranchingConfig:
    """Configuration for lazy branching."""

    enabled: bool = True

    # Token thresholds
    min_tokens_to_judge: int = 200  # Too early before this
    convergence_window_tokens: int = 750  # Check for convergence within this

    # Velocity thresholds (from empirical analysis)
    convergence_velocity: float = 0.5  # Below this = probably converged
    divergence_velocity: float = 0.65  # Above this = probably diverging

    # Response characteristics
    max_converged_length: int = 300  # Converged responses are usually short
    min_diverged_length: int = 400  # Diverging responses tend to be longer


# Patterns that suggest the response has converged (single answer)
CONVERGENCE_RESPONSE_PATTERNS = [
    # Direct factual answers (anywhere in response)
    (r"\bthe capital of .+ is\s+\w+", 0.85),  # "The capital of France is Paris"
    (r"\b\w+ is the capital of", 0.85),  # "Paris is the capital of France"
    (r"\bthe answer is\b", 0.8),
    (r"\bwritten by\s+\w+", 0.8),  # "written by Shakespeare"
    (r"\bthe author is\s+\w+", 0.8),

    # Short definitive statements at start
    (r"^(it is|yes|no)\b", 0.8),  # "It is approximately...", "Yes, that is correct"

    # Specific factual indicators
    (r"\bis\s+(paris|london|tokyo|berlin|madrid|rome|beijing|washington|moscow|delhi)", 0.8),
    (r"\b(equals?|=)\s*[\d,\.]+", 0.8),  # Math answers like "equals 105"
    (r"\bhas\s+\d+\s+(legs?|sides?|letters?)", 0.8),  # "has 8 legs"
    (r"\bis\s+approximately\s+[\d,\.]+", 0.8),  # "is approximately 3.14"

    # Confident conclusions (at start or after newline)
    (r"(^|\n)(in summary|to summarize|therefore|thus|so)\b", 0.6),
    (r"(^|\n)(the capital|the author|the result|the solution)\b", 0.7),

    # Short definitive statements
    (r"^\w+\s+\w+\s*[.!]$", 0.5),  # Two-word sentences like "Paris."
]

# Patterns that suggest the response is diverging (multiple directions)
DIVERGENCE_RESPONSE_PATTERNS = [
    # Multiple approaches
    (r"\b(there are (several|multiple|various|many)|here are some)\b", 0.9),
    (r"\b(on one hand|on the other hand|alternatively)\b", 0.8),
    (r"\b(it depends|depending on|various factors)\b", 0.8),

    # Lists and options
    (r"\b(first|second|third|1\.|2\.|3\.)\b.*\b(first|second|third|1\.|2\.|3\.)\b", 0.8),
    (r"\b(option [a-c]|approach [a-c]|method [a-c])\b", 0.7),

    # Exploration language
    (r"\b(let's (explore|examine|consider|look at))\b", 0.7),
    (r"\b(we (can|could|might|should) (also|consider))\b", 0.6),
    (r"\b(another (way|approach|option|method))\b", 0.8),

    # Hedging
    (r"\b(however|although|but|yet|while)\b", 0.5),
    (r"\b(perhaps|maybe|possibly|potentially)\b", 0.5),
]


def analyze_initial_response(
    response_text: str,
    token_count: int,
    velocity: float | None = None,
    config: LazyBranchingConfig | None = None,
) -> tuple[BranchSignal, float, str]:
    """Analyze initial response to determine if branching is needed.

    Args:
        response_text: The LLM's initial response
        token_count: Tokens used so far
        velocity: Current velocity (if available)
        config: Optional configuration

    Returns:
        Tuple of (signal, confidence, reason)
    """
    if config is None:
        config = LazyBranchingConfig()

    if not config.enabled:
        return (BranchSignal.UNCERTAIN, 0.0, "lazy_branching_disabled")

    # Too early to judge
    if token_count < config.min_tokens_to_judge:
        return (BranchSignal.UNCERTAIN, 0.0, "too_early")

    text_lower = response_text.lower().strip()
    text_length = len(response_text)

    # Score convergence and divergence signals
    convergence_score = 0.0
    divergence_score = 0.0
    convergence_reason = ""
    divergence_reason = ""

    # Check response patterns
    for pattern, weight in CONVERGENCE_RESPONSE_PATTERNS:
        if re.search(pattern, text_lower, re.MULTILINE):
            if weight > convergence_score:
                convergence_score = weight
                convergence_reason = f"pattern:{pattern[:30]}"

    for pattern, weight in DIVERGENCE_RESPONSE_PATTERNS:
        if re.search(pattern, text_lower, re.MULTILINE):
            if weight > divergence_score:
                divergence_score = weight
                divergence_reason = f"pattern:{pattern[:30]}"

    # Factor in velocity if available
    if velocity is not None:
        if velocity < config.convergence_velocity:
            convergence_score = max(convergence_score, 0.6)
            if not convergence_reason:
                convergence_reason = f"low_velocity:{velocity:.2f}"
        elif velocity > config.divergence_velocity:
            divergence_score = max(divergence_score, 0.6)
            if not divergence_reason:
                divergence_reason = f"high_velocity:{velocity:.2f}"

    # Factor in response length
    if text_length < config.max_converged_length and token_count < config.convergence_window_tokens:
        convergence_score += 0.2
        if not convergence_reason:
            convergence_reason = f"short_response:{text_length}"
    elif text_length > config.min_diverged_length:
        divergence_score += 0.1

    # Determine signal
    if convergence_score > divergence_score and convergence_score >= 0.5:
        return (BranchSignal.CONVERGED, convergence_score, convergence_reason)
    elif divergence_score > convergence_score and divergence_score >= 0.5:
        return (BranchSignal.DIVERGING, divergence_score, divergence_reason)
    else:
        return (BranchSignal.UNCERTAIN, max(convergence_score, divergence_score), "no_clear_signal")


def should_continue_branching(
    response_text: str,
    token_count: int,
    velocity: float | None = None,
    config: LazyBranchingConfig | None = None,
) -> tuple[bool, str]:
    """Determine if exploration should continue with branching.

    Args:
        response_text: The LLM's response so far
        token_count: Tokens used
        velocity: Current velocity
        config: Optional configuration

    Returns:
        Tuple of (should_branch, reason)
    """
    if config is None:
        config = LazyBranchingConfig()

    signal, confidence, reason = analyze_initial_response(
        response_text, token_count, velocity, config
    )

    if signal == BranchSignal.CONVERGED:
        return (False, f"converged:{reason}")
    elif signal == BranchSignal.DIVERGING:
        return (True, f"diverging:{reason}")
    else:
        # Default to branching for uncertain
        return (True, f"uncertain:{reason}")


def check_early_convergence(
    response_text: str,
    token_count: int,
    velocity: float | None = None,
    config: LazyBranchingConfig | None = None,
) -> bool:
    """Quick check for early convergence.

    Use this in the exploration loop to detect if we can skip branching.

    Args:
        response_text: Response text so far
        token_count: Tokens used
        velocity: Current velocity
        config: Configuration

    Returns:
        True if converged (no branching needed), False otherwise
    """
    if config is None:
        config = LazyBranchingConfig()

    text_lower = response_text.lower().strip()

    # For VERY short responses with high-confidence patterns, always check
    # A response like "The capital of France is Paris." is clearly converged
    if len(response_text) < 100:
        for pattern, weight in CONVERGENCE_RESPONSE_PATTERNS:
            if weight >= 0.8 and re.search(pattern, text_lower):
                return True

    # Quick heuristics for longer responses
    if token_count < config.min_tokens_to_judge:
        return False  # Too early for longer responses

    if token_count > config.convergence_window_tokens:
        return False  # Past the window for early convergence

    # Check velocity (fastest signal)
    if velocity is not None and velocity < config.convergence_velocity:
        # Low velocity + short response = converged
        if len(response_text) < config.max_converged_length:
            return True

    # Check response patterns
    for pattern, weight in CONVERGENCE_RESPONSE_PATTERNS:
        if weight >= 0.7 and re.search(pattern, text_lower):
            return True

    return False


def format_lazy_branching_result(
    response_text: str,
    token_count: int,
    velocity: float | None = None,
    config: LazyBranchingConfig | None = None,
) -> str:
    """Format lazy branching analysis for display."""
    signal, confidence, reason = analyze_initial_response(
        response_text, token_count, velocity, config
    )
    should_branch, branch_reason = should_continue_branching(
        response_text, token_count, velocity, config
    )

    return (
        f"Signal: {signal.value} (confidence: {confidence:.0%})\n"
        f"Should branch: {should_branch}\n"
        f"Reason: {reason}\n"
        f"Tokens: {token_count}, Velocity: {velocity or 'N/A'}"
    )
