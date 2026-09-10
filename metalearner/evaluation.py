"""Strategy evaluation for the meta-learner system.

This module provides functionality to evaluate and compare strategies,
including statistical significance testing and A/B testing frameworks.
"""

import math
import random
import time
from typing import Any, Callable, Sequence

from .types import (
    EvaluationResult,
    Strategy,
    StrategyResult,
)


class StrategyEvaluator:
    """Evaluates strategies against problems and compares performance."""

    def __init__(
        self,
        ground_truth_checker: Callable[[str, str], bool] | None = None,
        quality_scorer: Callable[[str, str], float] | None = None,
    ):
        """Initialize the evaluator.

        Args:
            ground_truth_checker: Function to check if output matches expected.
            quality_scorer: Function to score output quality (0.0-1.0).
        """
        self.ground_truth_checker = ground_truth_checker or self._default_checker
        self.quality_scorer = quality_scorer or self._default_scorer

    def _default_checker(self, output: str, expected: str) -> bool:
        """Default ground truth checker - exact match."""
        return output.strip().lower() == expected.strip().lower()

    def _default_scorer(self, output: str, problem: str) -> float:
        """Default quality scorer based on output characteristics."""
        if not output.strip():
            return 0.0

        score = 0.5  # Base score for any output

        # Length appropriateness (not too short, not too long)
        word_count = len(output.split())
        if 50 <= word_count <= 500:
            score += 0.2
        elif 20 <= word_count <= 1000:
            score += 0.1

        # Structure bonus
        if any(marker in output for marker in ["##", "1.", "- ", "* "]):
            score += 0.1

        # Completeness indicators
        if any(
            indicator in output.lower()
            for indicator in ["therefore", "conclusion", "answer", "result", "solution"]
        ):
            score += 0.1

        # Penalize obvious issues
        if "error" in output.lower() or "unable" in output.lower():
            score -= 0.2

        return max(0.0, min(1.0, score))

    async def evaluate_strategy(
        self,
        strategy: Strategy,
        problem: str,
        executor: Callable[[Strategy, str], Any],
        ground_truth: str | None = None,
    ) -> StrategyResult:
        """Evaluate a strategy on a single problem.

        Args:
            strategy: The strategy to evaluate.
            problem: The problem to solve.
            executor: Async function to execute strategy on problem.
            ground_truth: Expected answer if known.

        Returns:
            StrategyResult with evaluation metrics.
        """
        start_time = time.time()
        error = None
        output = ""
        tokens_used = 0

        try:
            result = await executor(strategy, problem)
            if isinstance(result, dict):
                output = result.get("output", str(result))
                tokens_used = result.get("tokens_used", 0)
            else:
                output = str(result)
        except Exception as e:
            error = str(e)
            output = ""

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Determine success
        if error:
            success = False
            quality_score = 0.0
        elif ground_truth:
            success = self.ground_truth_checker(output, ground_truth)
            quality_score = 1.0 if success else self.quality_scorer(output, problem)
        else:
            success = bool(output.strip())
            quality_score = self.quality_scorer(output, problem)

        return StrategyResult(
            strategy_id=strategy.id,
            success=success,
            tokens_used=tokens_used,
            quality_score=quality_score,
            problem_type="unknown",
            execution_time_ms=execution_time_ms,
            output=output,
            error=error,
        )

    def evaluate_strategy_sync(
        self,
        strategy: Strategy,
        problem: str,
        output: str,
        tokens_used: int = 0,
        ground_truth: str | None = None,
    ) -> StrategyResult:
        """Synchronous evaluation with pre-computed output.

        Args:
            strategy: The strategy that was used.
            problem: The problem that was solved.
            output: The output produced.
            tokens_used: Number of tokens used.
            ground_truth: Expected answer if known.

        Returns:
            StrategyResult with evaluation metrics.
        """
        if ground_truth:
            success = self.ground_truth_checker(output, ground_truth)
            quality_score = 1.0 if success else self.quality_scorer(output, problem)
        else:
            success = bool(output.strip())
            quality_score = self.quality_scorer(output, problem)

        return StrategyResult(
            strategy_id=strategy.id,
            success=success,
            tokens_used=tokens_used,
            quality_score=quality_score,
            problem_type="unknown",
            output=output,
        )


async def compare_strategies(
    strategies: list[Strategy],
    problems: list[tuple[str, str | None]],  # (problem, ground_truth)
    executor: Callable[[Strategy, str], Any],
    evaluator: StrategyEvaluator | None = None,
) -> EvaluationResult:
    """Compare multiple strategies head-to-head.

    Args:
        strategies: List of strategies to compare.
        problems: List of (problem, ground_truth) tuples.
        executor: Async function to execute strategies.
        evaluator: Evaluator to use, or default.

    Returns:
        EvaluationResult with rankings and metrics.
    """
    evaluator = evaluator or StrategyEvaluator()
    results_by_strategy: dict[str, list[StrategyResult]] = {
        s.id: [] for s in strategies
    }

    # Run all combinations
    for strategy in strategies:
        for problem, ground_truth in problems:
            result = await evaluator.evaluate_strategy(
                strategy, problem, executor, ground_truth
            )
            results_by_strategy[strategy.id].append(result)

    # Calculate metrics
    metrics: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        results = results_by_strategy[strategy.id]
        if results:
            metrics[strategy.id] = {
                "success_rate": sum(r.success for r in results) / len(results),
                "avg_quality": sum(r.quality_score for r in results) / len(results),
                "avg_tokens": sum(r.tokens_used for r in results) / len(results),
                "sample_count": len(results),
            }
        else:
            metrics[strategy.id] = {
                "success_rate": 0.0,
                "avg_quality": 0.0,
                "avg_tokens": 0.0,
                "sample_count": 0,
            }

    # Rank strategies by combined score
    def combined_score(strategy: Strategy) -> float:
        m = metrics[strategy.id]
        # Weight: 50% success, 30% quality, 20% efficiency (inverse tokens)
        token_score = 1.0 / (1.0 + m["avg_tokens"] / 1000)  # Normalize tokens
        return 0.5 * m["success_rate"] + 0.3 * m["avg_quality"] + 0.2 * token_score

    rankings = [(s, combined_score(s)) for s in strategies]
    rankings.sort(key=lambda x: x[1], reverse=True)

    # Statistical significance (pairwise)
    significance: dict[str, float] = {}
    if len(strategies) >= 2:
        top = strategies[0]
        for other in strategies[1:]:
            p_value = _calculate_significance(
                results_by_strategy[top.id],
                results_by_strategy[other.id],
            )
            significance[f"{top.id}_vs_{other.id}"] = p_value

    return EvaluationResult(
        rankings=rankings,
        metrics=metrics,
        statistical_significance=significance,
        best_strategy=rankings[0][0] if rankings else None,
    )


def _calculate_significance(
    results1: list[StrategyResult],
    results2: list[StrategyResult],
) -> float:
    """Calculate statistical significance between two result sets.

    Uses a simple proportion test (z-test approximation).

    Args:
        results1: Results from first strategy.
        results2: Results from second strategy.

    Returns:
        Approximate p-value.
    """
    if not results1 or not results2:
        return 1.0

    n1, n2 = len(results1), len(results2)
    p1 = sum(r.success for r in results1) / n1
    p2 = sum(r.success for r in results2) / n2

    # Pooled proportion
    p = (p1 * n1 + p2 * n2) / (n1 + n2)

    if p == 0 or p == 1:
        return 1.0 if p1 == p2 else 0.0

    # Standard error
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    if se == 0:
        return 1.0 if p1 == p2 else 0.0

    # Z-score
    z = abs(p1 - p2) / se

    # Approximate p-value (two-tailed)
    # Using simplified approximation
    p_value = 2 * (1 - _norm_cdf(z))

    return p_value


def _norm_cdf(z: float) -> float:
    """Approximation of normal CDF."""
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if z >= 0 else -1
    z = abs(z) / math.sqrt(2)

    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)

    return 0.5 * (1.0 + sign * y)


class ABTest:
    """A/B testing framework for strategies."""

    def __init__(
        self,
        strategy_a: Strategy,
        strategy_b: Strategy,
        evaluator: StrategyEvaluator | None = None,
        min_samples: int = 30,
        significance_threshold: float = 0.05,
    ):
        """Initialize A/B test.

        Args:
            strategy_a: First strategy (control).
            strategy_b: Second strategy (treatment).
            evaluator: Evaluator to use.
            min_samples: Minimum samples before concluding.
            significance_threshold: P-value threshold for significance.
        """
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.evaluator = evaluator or StrategyEvaluator()
        self.min_samples = min_samples
        self.significance_threshold = significance_threshold

        self.results_a: list[StrategyResult] = []
        self.results_b: list[StrategyResult] = []
        self.rng = random.Random()

    def select_strategy(self) -> Strategy:
        """Randomly select a strategy for the next trial.

        Returns:
            Either strategy_a or strategy_b.
        """
        return self.strategy_a if self.rng.random() < 0.5 else self.strategy_b

    async def run_trial(
        self,
        problem: str,
        executor: Callable[[Strategy, str], Any],
        ground_truth: str | None = None,
    ) -> tuple[Strategy, StrategyResult]:
        """Run a single A/B test trial.

        Args:
            problem: Problem to solve.
            executor: Function to execute strategy.
            ground_truth: Expected answer if known.

        Returns:
            Tuple of (selected_strategy, result).
        """
        strategy = self.select_strategy()
        result = await self.evaluator.evaluate_strategy(
            strategy, problem, executor, ground_truth
        )

        if strategy.id == self.strategy_a.id:
            self.results_a.append(result)
        else:
            self.results_b.append(result)

        return strategy, result

    def add_result(self, strategy: Strategy, result: StrategyResult) -> None:
        """Add a result from external evaluation.

        Args:
            strategy: The strategy that was used.
            result: The result of using the strategy.
        """
        if strategy.id == self.strategy_a.id:
            self.results_a.append(result)
        else:
            self.results_b.append(result)

    def is_conclusive(self) -> bool:
        """Check if we have enough data to conclude.

        Returns:
            True if test can be concluded.
        """
        if len(self.results_a) < self.min_samples:
            return False
        if len(self.results_b) < self.min_samples:
            return False

        p_value = _calculate_significance(self.results_a, self.results_b)
        return p_value < self.significance_threshold

    def get_winner(self) -> tuple[Strategy | None, float]:
        """Get the winning strategy if test is conclusive.

        Returns:
            Tuple of (winning_strategy, confidence) or (None, 0.0).
        """
        if not self.is_conclusive():
            return None, 0.0

        rate_a = sum(r.success for r in self.results_a) / len(self.results_a)
        rate_b = sum(r.success for r in self.results_b) / len(self.results_b)

        p_value = _calculate_significance(self.results_a, self.results_b)
        confidence = 1.0 - p_value

        if rate_a > rate_b:
            return self.strategy_a, confidence
        else:
            return self.strategy_b, confidence

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for the A/B test.

        Returns:
            Dictionary with test statistics.
        """
        n_a, n_b = len(self.results_a), len(self.results_b)

        rate_a = sum(r.success for r in self.results_a) / n_a if n_a else 0.0
        rate_b = sum(r.success for r in self.results_b) / n_b if n_b else 0.0

        quality_a = sum(r.quality_score for r in self.results_a) / n_a if n_a else 0.0
        quality_b = sum(r.quality_score for r in self.results_b) / n_b if n_b else 0.0

        p_value = _calculate_significance(self.results_a, self.results_b)

        return {
            "strategy_a": {
                "id": self.strategy_a.id,
                "name": self.strategy_a.name,
                "samples": n_a,
                "success_rate": rate_a,
                "avg_quality": quality_a,
            },
            "strategy_b": {
                "id": self.strategy_b.id,
                "name": self.strategy_b.name,
                "samples": n_b,
                "success_rate": rate_b,
                "avg_quality": quality_b,
            },
            "p_value": p_value,
            "is_conclusive": self.is_conclusive(),
            "lift": (rate_b - rate_a) / rate_a if rate_a > 0 else 0.0,
        }


def calculate_pass_at_k(
    results: Sequence[StrategyResult],
    k: int = 1,
) -> float:
    """Calculate pass@k metric.

    Args:
        results: List of results to evaluate.
        k: Number of attempts.

    Returns:
        Probability of at least one success in k attempts.
    """
    if not results:
        return 0.0

    n = len(results)
    c = sum(1 for r in results if r.success)

    if n < k:
        # Not enough samples
        return c / n if n > 0 else 0.0

    # Unbiased estimator for pass@k
    # 1 - C(n-c, k) / C(n, k)
    if n - c < k:
        return 1.0

    # Calculate using log to avoid overflow
    log_num = sum(math.log(n - c - i) for i in range(k))
    log_den = sum(math.log(n - i) for i in range(k))

    return 1.0 - math.exp(log_num - log_den)


def calculate_token_efficiency(results: Sequence[StrategyResult]) -> float:
    """Calculate token efficiency (success per 1000 tokens).

    Args:
        results: List of results.

    Returns:
        Success rate per 1000 tokens.
    """
    if not results:
        return 0.0

    total_tokens = sum(r.tokens_used for r in results)
    successes = sum(1 for r in results if r.success)

    if total_tokens == 0:
        return successes * 1000.0  # Perfect efficiency

    return (successes / total_tokens) * 1000


def calculate_quality_adjusted_score(
    results: Sequence[StrategyResult],
    success_weight: float = 0.5,
    quality_weight: float = 0.3,
    efficiency_weight: float = 0.2,
) -> float:
    """Calculate a weighted composite score.

    Args:
        results: List of results.
        success_weight: Weight for success rate.
        quality_weight: Weight for quality score.
        efficiency_weight: Weight for token efficiency.

    Returns:
        Weighted composite score.
    """
    if not results:
        return 0.0

    n = len(results)
    success_rate = sum(1 for r in results if r.success) / n
    avg_quality = sum(r.quality_score for r in results) / n

    total_tokens = sum(r.tokens_used for r in results)
    efficiency = 1.0 / (1.0 + total_tokens / (n * 1000))  # Normalized

    return (
        success_weight * success_rate
        + quality_weight * avg_quality
        + efficiency_weight * efficiency
    )
