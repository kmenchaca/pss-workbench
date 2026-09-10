"""Main meta-learning harness for the meta-learner system.

This module provides the main harness that orchestrates problem solving
using learned strategies and continuous improvement.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .classification import classify_problem
from .database import PerformanceDB, get_db
from .evaluation import StrategyEvaluator
from .learning import LearningLoop
from .mutation import StrategyMutator
from .recommendation import StrategyRecommender
from .strategies import apply_strategy, list_builtin_strategies
from .types import (
    ClassificationResult,
    LearningState,
    MetaConfig,
    RecommendationResult,
    Strategy,
    StrategyResult,
)


@dataclass
class SolveResult:
    """Result of solving a problem with the meta-harness."""

    problem: str
    classification: ClassificationResult
    recommendation: RecommendationResult
    strategy_used: Strategy
    result: StrategyResult
    prompt_generated: str
    output: str
    elapsed_ms: int
    fallback_used: bool = False


@dataclass
class BatchResult:
    """Result of solving multiple problems."""

    problems_solved: int
    successes: int
    total_tokens: int
    total_time_ms: int
    results: list[SolveResult] = field(default_factory=list)
    learning_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successes / max(1, self.problems_solved)

    @property
    def avg_tokens_per_problem(self) -> float:
        """Calculate average tokens per problem."""
        return self.total_tokens / max(1, self.problems_solved)


class MetaHarness:
    """Main harness for meta-learning exploration.

    The MetaHarness orchestrates:
    1. Problem classification
    2. Strategy recommendation
    3. Prompt generation
    4. Execution
    5. Result evaluation
    6. Learning and improvement
    """

    def __init__(
        self,
        config: MetaConfig | None = None,
        db: PerformanceDB | None = None,
        llm_executor: Callable[[str], Any] | None = None,
    ):
        """Initialize the meta-harness.

        Args:
            config: Configuration settings.
            db: Performance database.
            llm_executor: Async function to execute LLM prompts.
        """
        self.config = config or MetaConfig()
        self.db = db or get_db(self.config.db_path)
        self.llm_executor = llm_executor

        self.recommender = StrategyRecommender(self.db, self.config)
        self.evaluator = StrategyEvaluator()
        self.learning_loop = LearningLoop(self.config, self.db)
        self.mutator = StrategyMutator()

        self.solve_count = 0
        self.total_tokens = 0

    def set_executor(self, executor: Callable[[str], Any]) -> None:
        """Set the LLM executor.

        Args:
            executor: Async function to execute prompts.
        """
        self.llm_executor = executor

    async def solve(
        self,
        problem: str,
        ground_truth: str | None = None,
        strategy_override: Strategy | None = None,
    ) -> SolveResult:
        """Solve a problem using learned strategies.

        Args:
            problem: The problem to solve.
            ground_truth: Expected answer if known.
            strategy_override: Force specific strategy.

        Returns:
            SolveResult with outcome.
        """
        if not self.llm_executor:
            raise ValueError("No LLM executor set. Call set_executor() first.")

        start_time = time.time()

        # Classify problem
        classification = classify_problem(problem)

        # Get recommendation (or use override)
        if strategy_override:
            recommendation = RecommendationResult(
                strategy=strategy_override,
                confidence=1.0,
                explanation="Strategy override",
            )
        else:
            recommendation = self.recommender.recommend(problem)

        strategy = recommendation.strategy

        # Generate prompt using strategy
        prompt = apply_strategy(strategy, problem)

        # Execute
        fallback_used = False
        try:
            result = await self._execute_with_strategy(
                strategy, prompt, problem, ground_truth
            )
        except Exception as e:
            # Try fallback
            if recommendation.fallback:
                strategy = recommendation.fallback
                prompt = apply_strategy(strategy, problem)
                result = await self._execute_with_strategy(
                    strategy, prompt, problem, ground_truth
                )
                fallback_used = True
            else:
                # Create failed result
                result = StrategyResult(
                    strategy_id=strategy.id,
                    success=False,
                    tokens_used=0,
                    quality_score=0.0,
                    problem_type=classification.problem_type.id,
                    error=str(e),
                )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Update result with problem type
        result.problem_type = classification.problem_type.id

        # Record result for learning
        self.db.record_result(strategy, result.problem_type, result)

        self.solve_count += 1
        self.total_tokens += result.tokens_used

        return SolveResult(
            problem=problem,
            classification=classification,
            recommendation=recommendation,
            strategy_used=strategy,
            result=result,
            prompt_generated=prompt,
            output=result.output,
            elapsed_ms=elapsed_ms,
            fallback_used=fallback_used,
        )

    async def _execute_with_strategy(
        self,
        strategy: Strategy,
        prompt: str,
        problem: str,
        ground_truth: str | None,
    ) -> StrategyResult:
        """Execute a prompt and evaluate the result.

        Args:
            strategy: Strategy being used.
            prompt: The generated prompt.
            problem: Original problem.
            ground_truth: Expected answer.

        Returns:
            StrategyResult.
        """
        start_time = time.time()

        # Execute LLM
        response = await self.llm_executor(prompt)

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Parse response
        if isinstance(response, dict):
            output = response.get("output", response.get("content", str(response)))
            tokens_used = response.get("tokens_used", response.get("usage", {}).get("total_tokens", 0))
        else:
            output = str(response)
            tokens_used = 0

        # Evaluate
        result = self.evaluator.evaluate_strategy_sync(
            strategy, problem, output, tokens_used, ground_truth
        )
        result.execution_time_ms = execution_time_ms

        return result

    async def solve_batch(
        self,
        problems: list[tuple[str, str | None]],
        learn: bool = True,
    ) -> BatchResult:
        """Solve multiple problems.

        Args:
            problems: List of (problem, ground_truth) tuples.
            learn: Whether to run learning after batch.

        Returns:
            BatchResult with outcomes.
        """
        start_time = time.time()
        results = []
        successes = 0
        total_tokens = 0

        for problem, ground_truth in problems:
            result = await self.solve(problem, ground_truth)
            results.append(result)

            if result.result.success:
                successes += 1
            total_tokens += result.result.tokens_used

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Learning iteration
        learning_metrics = {}
        if learn and len(problems) >= 5:
            new_strategies = self.learning_loop.evolve_population()
            learning_metrics = {
                "new_strategies": len(new_strategies),
                "population_size": len(self.learning_loop.state.strategies),
            }

        return BatchResult(
            problems_solved=len(problems),
            successes=successes,
            total_tokens=total_tokens,
            total_time_ms=elapsed_ms,
            results=results,
            learning_metrics=learning_metrics,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get harness statistics.

        Returns:
            Dictionary with statistics.
        """
        db_stats = self.db.get_statistics()

        return {
            "solve_count": self.solve_count,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_solve": (
                self.total_tokens / max(1, self.solve_count)
            ),
            "strategies_available": len(self.learning_loop.state.strategies),
            "learning_iterations": self.learning_loop.state.iteration,
            **db_stats,
        }

    def get_best_strategies(
        self,
        problem_type: str | None = None,
        top_n: int = 5,
    ) -> list[tuple[Strategy, float]]:
        """Get the best performing strategies.

        Args:
            problem_type: Optional filter by problem type.
            top_n: Number of strategies to return.

        Returns:
            List of (strategy, score) tuples.
        """
        rankings = self.db.strategy_rankings(problem_type)
        return rankings[:top_n]

    def add_strategy(self, strategy: Strategy) -> None:
        """Add a new strategy to the pool.

        Args:
            strategy: Strategy to add.
        """
        self.learning_loop.state.strategies.append(strategy)

    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the pool.

        Args:
            strategy_id: ID of strategy to remove.

        Returns:
            True if removed.
        """
        strategy = self.db.get_strategy(strategy_id)
        if strategy:
            return self.learning_loop.retire_strategy(strategy)
        return False


async def create_harness(
    config: MetaConfig | None = None,
    llm_executor: Callable[[str], Any] | None = None,
) -> MetaHarness:
    """Create and initialize a MetaHarness.

    Args:
        config: Configuration settings.
        llm_executor: LLM executor function.

    Returns:
        Initialized MetaHarness.
    """
    harness = MetaHarness(config)
    if llm_executor:
        harness.set_executor(llm_executor)
    return harness


class SimplifiedHarness:
    """Simplified harness for basic usage without learning."""

    def __init__(self, llm_executor: Callable[[str], Any]):
        """Initialize simplified harness.

        Args:
            llm_executor: Async function to execute prompts.
        """
        self.llm_executor = llm_executor
        self.recommender = StrategyRecommender()
        self.evaluator = StrategyEvaluator()

    async def solve(self, problem: str) -> dict[str, Any]:
        """Solve a problem with automatic strategy selection.

        Args:
            problem: The problem to solve.

        Returns:
            Dictionary with result.
        """
        # Classify and recommend
        classification = classify_problem(problem)
        recommendation = self.recommender.recommend(problem)

        # Generate prompt
        prompt = apply_strategy(recommendation.strategy, problem)

        # Execute
        response = await self.llm_executor(prompt)

        output = str(response) if not isinstance(response, dict) else response.get("output", str(response))

        return {
            "problem": problem,
            "strategy": recommendation.strategy.name,
            "confidence": recommendation.confidence,
            "output": output,
            "problem_type": classification.problem_type.name,
        }


def run_harness_sync(
    problem: str,
    llm_executor: Callable[[str], str],
    config: MetaConfig | None = None,
) -> SolveResult:
    """Synchronous wrapper for running the harness.

    Args:
        problem: Problem to solve.
        llm_executor: Sync LLM executor.
        config: Configuration.

    Returns:
        SolveResult.
    """
    async def async_executor(prompt: str) -> str:
        return llm_executor(prompt)

    harness = MetaHarness(config)
    harness.set_executor(async_executor)

    return asyncio.run(harness.solve(problem))
