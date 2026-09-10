"""Verification strategies for PSS outputs.

Each strategy returns a float score between 0.0 and 1.0:
- 0.0 = completely failed
- 0.5 = partial success
- 1.0 = fully passed

The harness uses these scores to determine early exit when
score >= config.good_enough_threshold.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pss.config import PSSConfig


@dataclass
class VerificationResult:
    """Result of a verification check."""

    passed: bool
    score: float
    reason: str | None = None


def create_eval_fn(config: PSSConfig) -> Callable[[str], float] | None:
    """
    Create an eval function based on config.verification_strategy.

    Returns None if strategy is "none".
    """
    strategy = config.verification_strategy

    if strategy == "none":
        return None

    elif strategy == "programmatic":
        test_command = config.test_command or "pytest"
        return lambda output: programmatic_verify(output, test_command).score

    elif strategy == "checkbox":
        return lambda output: checkbox_verify(output).score

    elif strategy == "reviewer":
        # Reviewer strategy requires an LLM call - return a placeholder
        # that always returns 0.5 (neutral). Full implementation would
        # need provider access.
        return lambda output: 0.5

    return None


def programmatic_verify(
    output: str,
    test_command: str,
    work_dir: str | Path | None = None,
    timeout: int = 60,
) -> VerificationResult:
    """
    Run a test command and return pass/fail based on exit code.

    This is the "programmatic" verification strategy - useful for code tasks
    where tests can verify correctness.

    Args:
        output: The output to verify (may be written to a temp file if needed)
        test_command: Shell command to run (e.g., "pytest", "npm test")
        work_dir: Working directory for the command (defaults to cwd)
        timeout: Maximum seconds to wait for command

    Returns:
        VerificationResult with score 1.0 if exit code 0, else 0.0
    """
    try:
        result = subprocess.run(
            test_command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return VerificationResult(
                passed=True,
                score=1.0,
                reason="Tests passed",
            )
        else:
            return VerificationResult(
                passed=False,
                score=0.0,
                reason=f"Tests failed (exit code {result.returncode}): {result.stderr[:200]}",
            )

    except subprocess.TimeoutExpired:
        return VerificationResult(
            passed=False,
            score=0.0,
            reason=f"Test command timed out after {timeout}s",
        )
    except Exception as e:
        return VerificationResult(
            passed=False,
            score=0.0,
            reason=f"Test execution error: {str(e)}",
        )


def checkbox_verify(output: str) -> VerificationResult:
    """
    Verify based on checkbox completion in markdown.

    Counts [x] (done) vs [ ] (todo) checkboxes and returns
    completion ratio as the score.

    Args:
        output: Text containing markdown checkboxes

    Returns:
        VerificationResult with score = completed / total
    """
    # Match markdown checkboxes: [ ] or [x] or [X]
    done_pattern = r"\[x\]|\[X\]"
    todo_pattern = r"\[ \]"

    done_count = len(re.findall(done_pattern, output))
    todo_count = len(re.findall(todo_pattern, output))
    total = done_count + todo_count

    if total == 0:
        # No checkboxes found - assume complete
        return VerificationResult(
            passed=True,
            score=1.0,
            reason="No checkboxes found",
        )

    completion = done_count / total
    passed = completion >= 1.0

    return VerificationResult(
        passed=passed,
        score=completion,
        reason=f"{done_count}/{total} items completed ({completion:.0%})",
    )


def keyword_verify(
    output: str,
    required_keywords: list[str],
    forbidden_keywords: list[str] | None = None,
) -> VerificationResult:
    """
    Verify based on presence/absence of keywords.

    Useful for simple content checks without running code.

    Args:
        output: Text to check
        required_keywords: All of these must be present
        forbidden_keywords: None of these should be present

    Returns:
        VerificationResult with score based on keyword matches
    """
    forbidden_keywords = forbidden_keywords or []
    output_lower = output.lower()

    # Check required keywords
    required_found = sum(1 for kw in required_keywords if kw.lower() in output_lower)
    required_total = len(required_keywords)

    # Check forbidden keywords
    forbidden_found = sum(1 for kw in forbidden_keywords if kw.lower() in output_lower)

    if required_total == 0:
        required_score = 1.0
    else:
        required_score = required_found / required_total

    # Penalize for forbidden keywords
    if forbidden_found > 0:
        penalty = min(0.5, forbidden_found * 0.1)
        final_score = max(0.0, required_score - penalty)
    else:
        final_score = required_score

    passed = final_score >= 0.8 and forbidden_found == 0

    return VerificationResult(
        passed=passed,
        score=final_score,
        reason=f"Required: {required_found}/{required_total}, Forbidden: {forbidden_found}",
    )


def length_verify(
    output: str,
    min_words: int | None = None,
    max_words: int | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> VerificationResult:
    """
    Verify based on output length.

    Useful for ensuring outputs aren't too short (incomplete)
    or too long (rambling).

    Args:
        output: Text to check
        min_words: Minimum word count
        max_words: Maximum word count
        min_chars: Minimum character count
        max_chars: Maximum character count

    Returns:
        VerificationResult with score 1.0 if within bounds, else 0.0
    """
    word_count = len(output.split())
    char_count = len(output)

    issues = []

    if min_words and word_count < min_words:
        issues.append(f"Too short: {word_count} words < {min_words}")
    if max_words and word_count > max_words:
        issues.append(f"Too long: {word_count} words > {max_words}")
    if min_chars and char_count < min_chars:
        issues.append(f"Too short: {char_count} chars < {min_chars}")
    if max_chars and char_count > max_chars:
        issues.append(f"Too long: {char_count} chars > {max_chars}")

    if issues:
        return VerificationResult(
            passed=False,
            score=0.0,
            reason="; ".join(issues),
        )

    return VerificationResult(
        passed=True,
        score=1.0,
        reason=f"Length OK: {word_count} words, {char_count} chars",
    )


def combined_verify(
    output: str,
    verifiers: list[tuple[Callable[[str], VerificationResult], float]],
) -> VerificationResult:
    """
    Combine multiple verifiers with weights.

    Args:
        output: Text to verify
        verifiers: List of (verifier_fn, weight) tuples

    Returns:
        VerificationResult with weighted average score
    """
    if not verifiers:
        return VerificationResult(passed=True, score=1.0, reason="No verifiers")

    total_weight = sum(w for _, w in verifiers)
    weighted_score = 0.0
    reasons = []

    for verifier, weight in verifiers:
        result = verifier(output)
        weighted_score += result.score * weight
        if result.reason:
            reasons.append(result.reason)

    final_score = weighted_score / total_weight if total_weight > 0 else 0.0
    passed = final_score >= 0.8

    return VerificationResult(
        passed=passed,
        score=final_score,
        reason="; ".join(reasons),
    )
