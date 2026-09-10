"""Self-improvement bootstrap for the meta-learner system.

This module provides recursive self-improvement capabilities where
the meta-learner uses itself to optimize its own strategies.
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .classification import classify_problem
from .database import PerformanceDB, get_db
from .learning import LearningLoop
from .mutation import StrategyMutator, crossover_strategies, random_strategy
from .recommendation import StrategyRecommender
from .strategies import BUILTIN_STRATEGIES, apply_strategy, list_builtin_strategies
from .types import (
    MetaConfig,
    Strategy,
    StrategyResult,
)


@dataclass
class BootstrapResult:
    """Result of a bootstrap optimization cycle."""

    depth: int
    iterations: int
    initial_success_rate: float
    final_success_rate: float
    improvement: float
    strategies_created: int
    strategies_retired: int
    best_strategy: Strategy | None
    metrics_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MetaMetaResult:
    """Result of meta-meta optimization (optimizing the optimizer)."""

    meta_iterations: int
    meta_strategies_tested: int
    best_meta_strategy: dict[str, Any]
    improvement_over_baseline: float


class BootstrapOptimizer:
    """Optimizes the meta-learner using itself."""

    def __init__(
        self,
        config: MetaConfig | None = None,
        db: PerformanceDB | None = None,
        llm_executor: Callable[[str], Any] | None = None,
        max_depth: int = 3,
    ):
        """Initialize bootstrap optimizer.

        Args:
            config: Configuration settings.
            db: Performance database.
            llm_executor: Async LLM executor.
            max_depth: Maximum recursion depth.
        """
        self.config = config or MetaConfig()
        self.db = db or get_db()
        self.llm_executor = llm_executor
        self.max_depth = min(max_depth, self.config.bootstrap_depth)

        self.learning_loop = LearningLoop(self.config, self.db)
        self.mutator = StrategyMutator()
        self.recommender = StrategyRecommender(self.db, self.config)

        self.rng = random.Random()

    def set_executor(self, executor: Callable[[str], Any]) -> None:
        """Set the LLM executor.

        Args:
            executor: Async function to execute prompts.
        """
        self.llm_executor = executor

    async def bootstrap_optimization(
        self,
        problems: list[tuple[str, str | None]],
        iterations: int = 10,
        depth: int = 0,
    ) -> BootstrapResult:
        """Run bootstrap optimization cycle.

        Uses the meta-learner to improve itself by:
        1. Running current strategies on problems
        2. Analyzing what works and what doesn't
        3. Generating improved strategies
        4. Recursively optimizing (up to max_depth)

        Args:
            problems: Training problems with optional ground truth.
            iterations: Number of optimization iterations.
            depth: Current recursion depth.

        Returns:
            BootstrapResult with optimization outcomes.
        """
        if depth > self.max_depth:
            return BootstrapResult(
                depth=depth,
                iterations=0,
                initial_success_rate=0.0,
                final_success_rate=0.0,
                improvement=0.0,
                strategies_created=0,
                strategies_retired=0,
                best_strategy=None,
            )

        # Measure initial performance
        initial_results = await self._evaluate_population(problems)
        initial_success_rate = self._calculate_success_rate(initial_results)

        metrics_history = []
        strategies_created = 0
        strategies_retired = 0

        for i in range(iterations):
            # Run learning iteration
            metrics = await self.learning_loop.run_iteration(
                problems,
                self._create_executor_wrapper(),
            )
            metrics_history.append(metrics)

            # Analyze failures and generate improvements
            failures = [r for r in initial_results if not r.success]
            if failures:
                new_strategies = await self._generate_improvements(failures)
                strategies_created += len(new_strategies)

            # Retire underperformers
            retired = self._retire_underperformers()
            strategies_retired += len(retired)

            # Recursive optimization at deeper level
            if depth < self.max_depth and i % 3 == 0:
                # Sub-sample problems for recursive optimization
                sub_problems = self.rng.sample(
                    problems,
                    min(len(problems) // 2, 5),
                )
                if sub_problems:
                    await self.bootstrap_optimization(
                        sub_problems,
                        iterations=3,
                        depth=depth + 1,
                    )

        # Measure final performance
        final_results = await self._evaluate_population(problems)
        final_success_rate = self._calculate_success_rate(final_results)

        # Get best strategy
        rankings = self.db.strategy_rankings()
        best_strategy = rankings[0][0] if rankings else None

        return BootstrapResult(
            depth=depth,
            iterations=iterations,
            initial_success_rate=initial_success_rate,
            final_success_rate=final_success_rate,
            improvement=final_success_rate - initial_success_rate,
            strategies_created=strategies_created,
            strategies_retired=strategies_retired,
            best_strategy=best_strategy,
            metrics_history=metrics_history,
        )

    async def _evaluate_population(
        self,
        problems: list[tuple[str, str | None]],
    ) -> list[StrategyResult]:
        """Evaluate current strategy population.

        Args:
            problems: Problems to evaluate on.

        Returns:
            List of results.
        """
        results = []

        for problem, ground_truth in problems:
            recommendation = self.recommender.recommend(problem)
            prompt = apply_strategy(recommendation.strategy, problem)

            try:
                response = await self.llm_executor(prompt)
                output = str(response) if not isinstance(response, dict) else response.get("output", str(response))
                tokens = response.get("tokens_used", 0) if isinstance(response, dict) else 0

                success = True
                if ground_truth:
                    success = ground_truth.lower() in output.lower()

                result = StrategyResult(
                    strategy_id=recommendation.strategy.id,
                    success=success,
                    tokens_used=tokens,
                    quality_score=0.7 if success else 0.3,
                    problem_type="bootstrap",
                    output=output,
                )
                results.append(result)

            except Exception as e:
                results.append(StrategyResult(
                    strategy_id=recommendation.strategy.id,
                    success=False,
                    tokens_used=0,
                    quality_score=0.0,
                    problem_type="bootstrap",
                    error=str(e),
                ))

        return results

    def _calculate_success_rate(self, results: list[StrategyResult]) -> float:
        """Calculate success rate from results."""
        if not results:
            return 0.0
        return sum(1 for r in results if r.success) / len(results)

    async def _generate_improvements(
        self,
        failures: list[StrategyResult],
    ) -> list[Strategy]:
        """Generate improved strategies based on failures.

        Args:
            failures: Failed results to learn from.

        Returns:
            List of new strategies.
        """
        new_strategies = []

        # Group failures by strategy
        by_strategy: dict[str, list[StrategyResult]] = {}
        for result in failures:
            if result.strategy_id not in by_strategy:
                by_strategy[result.strategy_id] = []
            by_strategy[result.strategy_id].append(result)

        # Mutate frequently failing strategies
        for strategy_id, strategy_failures in by_strategy.items():
            if len(strategy_failures) >= 2:
                strategy = self.db.get_strategy(strategy_id)
                if strategy:
                    mutation = self.mutator.mutate(strategy)
                    new_strategies.append(mutation.new_strategy)
                    self.learning_loop.state.strategies.append(mutation.new_strategy)

        # Add random exploration
        if self.rng.random() < 0.2:
            random_strat = random_strategy(self.rng)
            new_strategies.append(random_strat)
            self.learning_loop.state.strategies.append(random_strat)

        return new_strategies

    def _retire_underperformers(self) -> list[Strategy]:
        """Retire poorly performing strategies.

        Returns:
            List of retired strategies.
        """
        retired = []
        rankings = self.db.strategy_rankings()

        if len(rankings) <= self.config.elite_count:
            return retired

        # Find strategies to retire
        for strategy, score in rankings[self.config.population_size:]:
            if strategy.id not in BUILTIN_STRATEGIES:
                if self.learning_loop.retire_strategy(strategy):
                    retired.append(strategy)

        return retired

    def _create_executor_wrapper(self) -> Callable[[Strategy, str], Any]:
        """Create executor wrapper for learning loop."""
        async def executor(strategy: Strategy, problem: str) -> dict[str, Any]:
            prompt = apply_strategy(strategy, problem)
            response = await self.llm_executor(prompt)

            if isinstance(response, dict):
                return response

            return {
                "output": str(response),
                "tokens_used": 0,
            }

        return executor


async def bootstrap_optimization(
    problems: list[tuple[str, str | None]],
    llm_executor: Callable[[str], Any],
    iterations: int = 10,
    config: MetaConfig | None = None,
) -> BootstrapResult:
    """Run bootstrap optimization.

    Args:
        problems: Training problems.
        llm_executor: LLM executor.
        iterations: Number of iterations.
        config: Configuration.

    Returns:
        BootstrapResult.
    """
    optimizer = BootstrapOptimizer(config)
    optimizer.set_executor(llm_executor)
    return await optimizer.bootstrap_optimization(problems, iterations)


class MetaMetaOptimizer:
    """Optimizes the meta-learner's own configuration and algorithms.

    This is one level above regular bootstrap - it optimizes HOW the
    meta-learner optimizes, not just the strategies it uses.
    """

    def __init__(
        self,
        llm_executor: Callable[[str], Any],
        base_config: MetaConfig | None = None,
    ):
        """Initialize meta-meta optimizer.

        Args:
            llm_executor: LLM executor.
            base_config: Base configuration to optimize from.
        """
        self.llm_executor = llm_executor
        self.base_config = base_config or MetaConfig()
        self.rng = random.Random()

    async def optimize_meta_strategy(
        self,
        problems: list[tuple[str, str | None]],
        meta_iterations: int = 5,
    ) -> MetaMetaResult:
        """Optimize the meta-learning strategy itself.

        Args:
            problems: Problems to train/test on.
            meta_iterations: Number of meta-iterations.

        Returns:
            MetaMetaResult with outcomes.
        """
        # Define meta-strategy parameters to optimize
        meta_params = [
            ("learning_rate", 0.05, 0.5),
            ("exploration_rate", 0.05, 0.3),
            ("mutation_rate", 0.1, 0.5),
            ("tournament_size", 2, 5),
        ]

        # Baseline performance
        baseline_result = await self._run_with_config(problems, self.base_config)
        baseline_score = baseline_result.final_success_rate

        best_config = self.base_config
        best_score = baseline_score
        configs_tested = 1

        for _ in range(meta_iterations):
            # Generate candidate configurations
            candidates = self._generate_config_variants(meta_params)

            for config in candidates:
                configs_tested += 1
                result = await self._run_with_config(problems, config)

                if result.final_success_rate > best_score:
                    best_score = result.final_success_rate
                    best_config = config

        return MetaMetaResult(
            meta_iterations=meta_iterations,
            meta_strategies_tested=configs_tested,
            best_meta_strategy=best_config.to_dict(),
            improvement_over_baseline=best_score - baseline_score,
        )

    async def _run_with_config(
        self,
        problems: list[tuple[str, str | None]],
        config: MetaConfig,
    ) -> BootstrapResult:
        """Run bootstrap with specific config.

        Args:
            problems: Problems to test on.
            config: Configuration to test.

        Returns:
            BootstrapResult.
        """
        optimizer = BootstrapOptimizer(config)
        optimizer.set_executor(self.llm_executor)
        return await optimizer.bootstrap_optimization(
            problems,
            iterations=3,  # Fewer iterations for efficiency
        )

    def _generate_config_variants(
        self,
        params: list[tuple[str, float, float]],
    ) -> list[MetaConfig]:
        """Generate config variants.

        Args:
            params: Parameter ranges to vary.

        Returns:
            List of variant configurations.
        """
        variants = []

        for param_name, min_val, max_val in params:
            # Create variant with this parameter changed
            config_dict = self.base_config.to_dict()
            config_dict[param_name] = self.rng.uniform(min_val, max_val)
            variants.append(MetaConfig.from_dict(config_dict))

        # Also try a fully random config
        config_dict = {}
        for param_name, min_val, max_val in params:
            if isinstance(min_val, int):
                config_dict[param_name] = self.rng.randint(int(min_val), int(max_val))
            else:
                config_dict[param_name] = self.rng.uniform(min_val, max_val)
        variants.append(MetaConfig.from_dict(config_dict))

        return variants


async def optimize_meta_strategy(
    problems: list[tuple[str, str | None]],
    llm_executor: Callable[[str], Any],
    meta_iterations: int = 5,
) -> MetaMetaResult:
    """Optimize the meta-learning strategy.

    Args:
        problems: Training problems.
        llm_executor: LLM executor.
        meta_iterations: Number of meta-iterations.

    Returns:
        MetaMetaResult.
    """
    optimizer = MetaMetaOptimizer(llm_executor)
    return await optimizer.optimize_meta_strategy(problems, meta_iterations)
