"""Convergent task detection for PSS v1.2.

Detects prompts that have single correct answers and shouldn't branch.
These tasks waste tokens exploring when they should just answer.

Examples of convergent tasks:
- "What is the capital of France?" → Just answer "Paris"
- "How many legs does a spider have?" → Just answer "8"
- "Who wrote Hamlet?" → Just answer "Shakespeare"

Examples of divergent tasks (should branch):
- "List creative uses for a brick" → Many valid answers
- "Write a story about..." → Many valid approaches
- "Analyze the implications of..." → Multiple perspectives needed
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Classification of task type."""

    CONVERGENT = "convergent"  # Single correct answer, don't branch
    DIVERGENT = "divergent"    # Multiple valid approaches, branch freely
    UNCERTAIN = "uncertain"    # Can't determine, use default behavior


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection."""

    enabled: bool = True

    # If True, uncertain tasks are treated as divergent (branch)
    # If False, uncertain tasks are treated as convergent (don't branch)
    branch_on_uncertain: bool = True

    # Minimum confidence to classify as convergent
    convergent_confidence_threshold: float = 0.7


# Patterns that suggest convergent (single-answer) tasks
CONVERGENT_PATTERNS = [
    # Factual questions
    (r"\bwhat is the\b.*\bof\b", 0.8),
    (r"\bwho is the\b.*\bof\b", 0.8),
    (r"\bwho wrote\b", 0.8),
    (r"\bwho invented\b", 0.8),
    (r"\bwho discovered\b", 0.8),
    (r"\bwhen did\b.*\b(happen|occur|start|end|begin)\b", 0.8),
    (r"\bwhen was\b.*\b(born|founded|created|built)\b", 0.8),
    (r"\bhow many\b.*\b(are there|does|do)\b", 0.7),
    (r"\bhow much\b.*\b(is|does|do)\b", 0.7),
    (r"\bwhere is\b.*\blocated\b", 0.8),
    (r"\bwhere was\b.*\bborn\b", 0.8),

    # Definitional questions
    (r"\bdefine\b", 0.7),
    (r"\bwhat does\b.*\bmean\b", 0.7),
    (r"\bwhat is the definition of\b", 0.8),
    (r"\bwhat is the meaning of\b", 0.8),

    # Calculation/conversion
    (r"\bconvert\b.*\bto\b", 0.8),
    (r"\bcalculate\b", 0.7),
    (r"\bcompute\b", 0.7),
    (r"\bwhat is\b.*\bplus\b", 0.9),
    (r"\bwhat is\b.*\bminus\b", 0.9),
    (r"\bwhat is\b.*\btimes\b", 0.9),
    (r"\bwhat is\b.*\bdivided by\b", 0.9),
    (r"\b\d+\s*[\+\-\*\/]\s*\d+\b", 0.9),

    # Yes/no questions (often convergent)
    (r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|had)\b.*\?$", 0.6),

    # Spelling/grammar
    (r"\bhow do you spell\b", 0.9),
    (r"\bspell\b.*\bcorrectly\b", 0.9),
]

# Patterns that suggest divergent (multi-answer) tasks
DIVERGENT_PATTERNS = [
    # Creative tasks
    (r"\bwrite\b.*\b(story|poem|essay|article|script)\b", 0.9),
    (r"\bcreate\b", 0.7),
    (r"\bcompose\b", 0.8),
    (r"\bimagine\b", 0.8),
    (r"\bdesign\b", 0.7),

    # Exploration/brainstorming
    (r"\blist\b.*\b(ways|ideas|options|approaches|methods)\b", 0.9),
    (r"\bbrainstorm\b", 0.9),
    (r"\bexplore\b", 0.8),
    (r"\bmap out\b", 0.9),
    (r"\bgenerate\b.*\bideas\b", 0.9),

    # Analysis (often benefits from multiple perspectives)
    (r"\banalyze\b", 0.6),
    (r"\bevaluate\b", 0.6),
    (r"\bcompare\b.*\band\b", 0.7),
    (r"\bpros and cons\b", 0.8),
    (r"\badvantages and disadvantages\b", 0.8),

    # Open-ended
    (r"\bwhat are\b.*\bways\b", 0.8),
    (r"\bwhat are\b.*\boptions\b", 0.8),
    (r"\bwhat could\b", 0.6),
    (r"\bwhat might\b", 0.6),
    (r"\bwhat if\b", 0.7),
    (r"\bhow might\b", 0.7),
    (r"\bhow could\b", 0.7),

    # Multi-perspective
    (r"\bdifferent perspectives\b", 0.9),
    (r"\bmultiple approaches\b", 0.9),
    (r"\bvarious ways\b", 0.8),
]


def detect_task_type(
    prompt: str,
    config: ConvergenceConfig | None = None,
) -> tuple[TaskType, float, str]:
    """Detect whether a prompt is convergent or divergent.

    Args:
        prompt: The user's prompt
        config: Optional configuration

    Returns:
        Tuple of (task_type, confidence, reason)
    """
    if config is None:
        config = ConvergenceConfig()

    if not config.enabled:
        return (TaskType.UNCERTAIN, 0.0, "detection_disabled")

    prompt_lower = prompt.lower().strip()

    # Check divergent patterns first (these override convergent)
    max_divergent_score = 0.0
    divergent_reason = ""

    for pattern, weight in DIVERGENT_PATTERNS:
        if re.search(pattern, prompt_lower):
            if weight > max_divergent_score:
                max_divergent_score = weight
                divergent_reason = f"matches_divergent_pattern:{pattern}"

    # Check convergent patterns
    max_convergent_score = 0.0
    convergent_reason = ""

    for pattern, weight in CONVERGENT_PATTERNS:
        if re.search(pattern, prompt_lower):
            if weight > max_convergent_score:
                max_convergent_score = weight
                convergent_reason = f"matches_convergent_pattern:{pattern}"

    # Determine result
    if max_divergent_score > max_convergent_score:
        if max_divergent_score >= config.convergent_confidence_threshold:
            return (TaskType.DIVERGENT, max_divergent_score, divergent_reason)
    elif max_convergent_score > max_divergent_score:
        if max_convergent_score >= config.convergent_confidence_threshold:
            return (TaskType.CONVERGENT, max_convergent_score, convergent_reason)

    # No clear signal
    return (TaskType.UNCERTAIN, max(max_divergent_score, max_convergent_score), "no_clear_pattern")


def should_branch(
    prompt: str,
    config: ConvergenceConfig | None = None,
) -> tuple[bool, str]:
    """Determine if a prompt should use branching exploration.

    Args:
        prompt: The user's prompt
        config: Optional configuration

    Returns:
        Tuple of (should_branch, reason)
    """
    if config is None:
        config = ConvergenceConfig()

    task_type, confidence, reason = detect_task_type(prompt, config)

    if task_type == TaskType.CONVERGENT:
        return (False, f"convergent_task:{reason}")
    elif task_type == TaskType.DIVERGENT:
        return (True, f"divergent_task:{reason}")
    else:
        # Uncertain - use config to decide
        return (config.branch_on_uncertain, f"uncertain_task:{reason}")


def format_convergence_result(
    prompt: str,
    config: ConvergenceConfig | None = None,
) -> str:
    """Format convergence detection result for display."""
    task_type, confidence, reason = detect_task_type(prompt, config)
    branch, branch_reason = should_branch(prompt, config)

    return (
        f"Task type: {task_type.value} (confidence: {confidence:.0%})\n"
        f"Should branch: {branch}\n"
        f"Reason: {reason}"
    )
