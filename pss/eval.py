"""Evaluation module for PSS.

This module provides metrics and utilities for evaluating PSS performance
against baselines like single-shot and best-of-N sampling.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from pss.types import Context


@dataclass
class EvalResult:
    """Result of an evaluation run."""

    task: str
    method: str  # "pss", "best_of_n", "single_shot"
    num_samples: int
    num_correct: int
    pass_at_1: float
    pass_at_k: float  # k = num_samples
    diversity_score: float | None = None
    self_bleu: float | None = None
    distinct_1: float | None = None
    distinct_2: float | None = None
    majority_vote_correct: bool | None = None
    total_cost: float = 0.0
    total_tokens: int = 0
    raw_outputs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def efficiency(self) -> float:
        """Diversity per dollar spent."""
        if self.total_cost <= 0 or self.diversity_score is None:
            return 0.0
        return self.diversity_score / self.total_cost

    @property
    def pass_at_k_gain(self) -> float:
        """Relative improvement of pass@k over pass@1."""
        if self.pass_at_1 <= 0:
            return float("inf") if self.pass_at_k > 0 else 0.0
        return (self.pass_at_k - self.pass_at_1) / self.pass_at_1


# =============================================================================
# Core Metrics
# =============================================================================


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric (unbiased estimator).

    This is the probability that at least one of k samples is correct,
    given n total samples with c correct ones.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to consider

    Returns:
        Probability that at least one of k samples is correct

    Reference:
        Chen et al., "Evaluating Large Language Models Trained on Code"
        https://arxiv.org/abs/2107.03374
    """
    if n < k:
        return 1.0 if c > 0 else 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    # Use the formula: 1 - C(n-c, k) / C(n, k)
    # Computed as product to avoid large factorials
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def pass_at_k_from_results(
    results: list[bool],
    k: int | None = None,
) -> tuple[float, float]:
    """
    Calculate pass@1 and pass@k from a list of boolean results.

    Args:
        results: List of True/False for each sample
        k: Number of samples for pass@k (default: len(results))

    Returns:
        (pass@1, pass@k)
    """
    n = len(results)
    c = sum(results)
    if k is None:
        k = n

    p1 = pass_at_k(n, c, 1)
    pk = pass_at_k(n, c, min(k, n))

    return p1, pk


def self_bleu(outputs: list[str], n: int = 4, smoothing: bool = True) -> float:
    """
    Calculate Self-BLEU score (lower = more diverse).

    Each output is scored against all other outputs as references.
    The average BLEU score measures how similar outputs are to each other.

    Args:
        outputs: List of text outputs
        n: Maximum n-gram order (default: 4)
        smoothing: Whether to use smoothing for short texts

    Returns:
        Average Self-BLEU score (0-1, lower = more diverse)
    """
    if len(outputs) < 2:
        return 0.0

    def get_ngrams(text: str, n: int) -> list[tuple]:
        """Extract n-grams from text."""
        tokens = text.lower().split()
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def bleu_score(candidate: str, references: list[str], max_n: int) -> float:
        """Calculate BLEU score for a single candidate."""
        candidate_tokens = candidate.lower().split()
        if len(candidate_tokens) == 0:
            return 0.0

        # Collect reference n-grams
        ref_ngram_counts: dict[int, Counter] = {i: Counter() for i in range(1, max_n + 1)}
        for ref in references:
            for i in range(1, max_n + 1):
                ngrams = get_ngrams(ref, i)
                for ng in ngrams:
                    ref_ngram_counts[i][ng] = max(
                        ref_ngram_counts[i][ng], Counter(ngrams)[ng]
                    )

        # Calculate precision for each n-gram order
        precisions = []
        for i in range(1, max_n + 1):
            candidate_ngrams = get_ngrams(candidate, i)
            if not candidate_ngrams:
                if smoothing:
                    precisions.append(1.0 / (len(candidate_tokens) + 1))
                else:
                    precisions.append(0.0)
                continue

            matches = sum(
                min(count, ref_ngram_counts[i][ng])
                for ng, count in Counter(candidate_ngrams).items()
            )
            precision = matches / len(candidate_ngrams)
            if smoothing and precision == 0:
                precision = 1.0 / (len(candidate_ngrams) + 1)
            precisions.append(precision)

        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            return 0.0

        log_precision = sum(math.log(p) for p in precisions) / len(precisions)
        return math.exp(log_precision)

    # Calculate Self-BLEU
    scores = []
    for i, output in enumerate(outputs):
        refs = [outputs[j] for j in range(len(outputs)) if j != i]
        scores.append(bleu_score(output, refs, n))

    return sum(scores) / len(scores)


def distinct_n(outputs: list[str], n: int = 1) -> float:
    """
    Calculate Distinct-n metric.

    Ratio of unique n-grams to total n-grams across all outputs.
    Higher = more diverse vocabulary.

    Args:
        outputs: List of text outputs
        n: N-gram order (1 for unigrams, 2 for bigrams, etc.)

    Returns:
        Ratio of unique n-grams to total n-grams (0-1)
    """
    if not outputs:
        return 0.0

    all_ngrams = []
    for output in outputs:
        tokens = output.lower().split()
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)


def majority_vote(
    outputs: list[str],
    extract_answer: Callable[[str], str | None],
) -> tuple[str | None, int]:
    """
    Find the majority answer from outputs.

    Args:
        outputs: List of text outputs
        extract_answer: Function to extract answer from output

    Returns:
        (majority_answer, count)
    """
    answers = [extract_answer(output) for output in outputs]
    valid_answers = [a for a in answers if a is not None]

    if not valid_answers:
        return None, 0

    counter = Counter(valid_answers)
    most_common = counter.most_common(1)[0]
    return most_common


def weighted_vote(
    outputs: list[str],
    confidences: list[float],
    extract_answer: Callable[[str], str | None],
) -> tuple[str | None, float]:
    """
    Find the weighted majority answer.

    Args:
        outputs: List of text outputs
        confidences: Confidence scores for each output
        extract_answer: Function to extract answer from output

    Returns:
        (weighted_answer, total_weight)
    """
    if len(outputs) != len(confidences):
        raise ValueError("outputs and confidences must have same length")

    answer_weights: dict[str, float] = {}
    for output, conf in zip(outputs, confidences):
        answer = extract_answer(output)
        if answer is not None:
            answer_weights[answer] = answer_weights.get(answer, 0) + conf

    if not answer_weights:
        return None, 0.0

    best_answer = max(answer_weights, key=answer_weights.get)
    return best_answer, answer_weights[best_answer]


# =============================================================================
# Answer Extractors
# =============================================================================


def extract_number(text: str) -> str | None:
    """Extract a number from text (for math tasks)."""
    # Look for boxed answers first (common in math)
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()

    # Look for "answer is X" or "= X" patterns
    patterns = [
        r"(?:answer|result|solution)\s*(?:is|=|:)\s*(-?\d+(?:\.\d+)?)",
        r"=\s*(-?\d+(?:\.\d+)?)\s*$",
        r"(-?\d+(?:\.\d+)?)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    return None


def extract_yes_no(text: str) -> str | None:
    """Extract yes/no answer from text."""
    text_lower = text.lower()

    # Look for explicit yes/no
    if re.search(r"\b(yes|correct|true|right)\b", text_lower):
        if not re.search(r"\b(no|incorrect|false|wrong)\b", text_lower):
            return "yes"
    if re.search(r"\b(no|incorrect|false|wrong)\b", text_lower):
        if not re.search(r"\b(yes|correct|true|right)\b", text_lower):
            return "no"

    return None


def extract_multiple_choice(text: str) -> str | None:
    """Extract A/B/C/D answer from text."""
    # Look for "answer is X" pattern
    match = re.search(
        r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-Da-d])\)?",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # Look for standalone letter at end
    match = re.search(r"\b([A-Da-d])\b\s*$", text)
    if match:
        return match.group(1).upper()

    return None


# =============================================================================
# Evaluation Utilities
# =============================================================================


def evaluate_outputs(
    outputs: list[str],
    verifier: Callable[[str], bool],
    task: str = "unknown",
    method: str = "unknown",
    total_cost: float = 0.0,
    total_tokens: int = 0,
    compute_diversity: bool = True,
) -> EvalResult:
    """
    Evaluate a set of outputs against a verifier.

    Args:
        outputs: List of output strings
        verifier: Function that returns True if output is correct
        task: Task name
        method: Method name (e.g., "pss", "best_of_n")
        total_cost: Total cost in USD
        total_tokens: Total tokens used
        compute_diversity: Whether to compute diversity metrics

    Returns:
        EvalResult with all metrics
    """
    # Verify each output
    results = [verifier(output) for output in outputs]
    num_correct = sum(results)

    # Calculate pass@k
    p1, pk = pass_at_k_from_results(results)

    # Calculate diversity metrics
    div_score = None
    sb = None
    d1 = None
    d2 = None

    if compute_diversity and len(outputs) >= 2:
        sb = self_bleu(outputs)
        d1 = distinct_n(outputs, 1)
        d2 = distinct_n(outputs, 2)

        # Try to compute embedding-based diversity
        try:
            from pss.diversity import compute_diversity_from_texts

            diversity_result = compute_diversity_from_texts(outputs)
            if diversity_result:
                div_score = diversity_result.score
        except Exception:
            pass

    return EvalResult(
        task=task,
        method=method,
        num_samples=len(outputs),
        num_correct=num_correct,
        pass_at_1=p1,
        pass_at_k=pk,
        diversity_score=div_score,
        self_bleu=sb,
        distinct_1=d1,
        distinct_2=d2,
        total_cost=total_cost,
        total_tokens=total_tokens,
        raw_outputs=outputs,
    )


def evaluate_with_answer_key(
    outputs: list[str],
    correct_answer: str,
    extract_answer: Callable[[str], str | None],
    normalize: Callable[[str], str] | None = None,
    task: str = "unknown",
    method: str = "unknown",
    total_cost: float = 0.0,
    total_tokens: int = 0,
) -> EvalResult:
    """
    Evaluate outputs against a known correct answer.

    Args:
        outputs: List of output strings
        correct_answer: The correct answer
        extract_answer: Function to extract answer from output
        normalize: Optional function to normalize answers before comparison
        task: Task name
        method: Method name
        total_cost: Total cost
        total_tokens: Total tokens

    Returns:
        EvalResult with metrics including majority vote accuracy
    """
    if normalize is None:
        normalize = lambda x: x.strip().lower()

    correct_normalized = normalize(correct_answer)

    def verifier(output: str) -> bool:
        extracted = extract_answer(output)
        if extracted is None:
            return False
        return normalize(extracted) == correct_normalized

    result = evaluate_outputs(
        outputs, verifier, task, method, total_cost, total_tokens
    )

    # Add majority vote result
    majority_answer, count = majority_vote(outputs, extract_answer)
    if majority_answer is not None:
        result.majority_vote_correct = normalize(majority_answer) == correct_normalized

    return result


def leaves_to_outputs(leaves: list[Context]) -> list[str]:
    """Extract output strings from leaf contexts."""
    outputs = []
    for leaf in leaves:
        if leaf.output:
            outputs.append(leaf.output)
        else:
            # Fall back to last assistant message
            for msg in reversed(leaf.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    outputs.append(msg["content"])
                    break
    return outputs


def format_eval_result(result: EvalResult, verbose: bool = False) -> str:
    """Format an evaluation result for display."""
    lines = [
        f"=== {result.task} ({result.method}) ===",
        f"Samples: {result.num_samples} ({result.num_correct} correct)",
        f"Pass@1: {result.pass_at_1:.2%}",
        f"Pass@{result.num_samples}: {result.pass_at_k:.2%}",
    ]

    if result.pass_at_k > result.pass_at_1:
        lines.append(f"Pass@k gain: +{result.pass_at_k_gain:.0%}")

    if result.majority_vote_correct is not None:
        lines.append(
            f"Majority vote: {'correct' if result.majority_vote_correct else 'incorrect'}"
        )

    if result.diversity_score is not None:
        lines.append(f"Diversity (embedding): {result.diversity_score:.2f}")

    if result.self_bleu is not None:
        lines.append(f"Self-BLEU: {result.self_bleu:.2f} (lower = more diverse)")

    if result.distinct_1 is not None:
        lines.append(f"Distinct-1: {result.distinct_1:.2f}")
        lines.append(f"Distinct-2: {result.distinct_2:.2f}")

    if result.total_cost > 0:
        lines.append(f"Cost: ${result.total_cost:.4f}")
        if result.diversity_score is not None:
            lines.append(f"Efficiency: {result.efficiency:.1f} diversity/$")

    if verbose and result.raw_outputs:
        lines.append("")
        lines.append("Outputs:")
        for i, output in enumerate(result.raw_outputs, 1):
            preview = output[:100].replace("\n", " ")
            if len(output) > 100:
                preview += "..."
            lines.append(f"  {i}. {preview}")

    return "\n".join(lines)


def compare_results(results: list[EvalResult]) -> str:
    """Format a comparison of multiple evaluation results."""
    if not results:
        return "No results to compare"

    lines = [
        f"=== Comparison: {results[0].task} ===",
        "",
        f"{'Method':<15} {'Pass@1':<10} {'Pass@k':<10} {'Diversity':<10} {'Cost':<10}",
        "-" * 55,
    ]

    for r in results:
        div_str = f"{r.diversity_score:.2f}" if r.diversity_score else "N/A"
        cost_str = f"${r.total_cost:.4f}" if r.total_cost > 0 else "N/A"
        lines.append(
            f"{r.method:<15} {r.pass_at_1:<10.2%} {r.pass_at_k:<10.2%} {div_str:<10} {cost_str:<10}"
        )

    return "\n".join(lines)
