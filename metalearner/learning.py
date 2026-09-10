"""Learning loop for the meta-learner system.

This module provides the core learning loop that improves strategies
over time based on performance feedback.
"""

import random
from datetime import datetime
from typing import Any, Callable

from .classification import classify_problem
from .database import PerformanceDB, get_db
from .evaluation import StrategyEvaluator
from .mutation import (
    StrategyMutator,
    crossover_strategies,
    random_strategy,
)
from .recommendation import StrategyRecommender
from .strategies import BUILTIN_STRATEGIES, list_builtin_strategies
from .types import (
    LearningState,
    MetaConfig,
    Strategy,
    StrategyResult,
)


class LearningLoop:
    """Manages the learning process for strategy improvement."""

    def __init__(
        self,
        config: MetaConfig | None = None,
        db: PerformanceDB | None = None,
        rng: random.Random | None = None,
    ):
        """Initialize the learning loop.

        Args:
            config: Configuration for learning.
            db: Performance database.
            rng: Random number generator.
        """
        self.config = config or MetaConfig()
        self.db = db or get_db()
        self.rng = rng or random.Random()

        self.mutator = StrategyMutator(self.rng)
        self.recommender = StrategyRecommender(self.db, self.config, self.rng)
        self.evaluator = StrategyEvaluator()

        self.state = LearningState(
            strategies=list_builtin_strategies(),
        )

    def update_recommendations(self, results: list[tuple[Strategy, StrategyResult]]) -> None:
        """Update recommendation weights based on new results.

        Args:
            results: List of (strategy, result) tuples.
        """
        for strategy, result in results:
            # Record to database
            self.db.record_result(strategy, result.problem_type, result)

            # Apply forgetting factor to old results
            self._apply_forgetting(strategy.id, result.problem_type)

    def _apply_forgetting(self, strategy_id: str, problem_type: str) -> None:
        """Apply forgetting factor to decay old results.

        Args:
            strategy_id: Strategy to update.
            problem_type: Problem type to update.
        """
        record = self.db.query_performance(strategy_id, problem_type)
        if not record or not hasattr(record, "results"):
            return

        # Forgetting is simulated by giving more weight to recent results
        # This is handled in the database querying, not by modifying results

    def discover_strategy(
        self,
        problem_type: str,
        executor: Callable[[Strategy, str], Any] | None = None,
    ) -> Strategy | None:
        """Try to discover a new effective strategy for a problem type.

        Args:
            problem_type: Type of problem to optimize for.
            executor: Optional executor for testing.

        Returns:
            New strategy if discovered, None otherwise.
        """
        # Get top strategies for this problem type
        rankings = self.db.strategy_rankings(problem_type)

        if len(rankings) < 2:
            # Not enough data - try random
            new_strategy = random_strategy(self.rng)
            self.state.strategies.append(new_strategy)
            self.state.discoveries += 1
            return new_strategy

        # Try crossover of top performers
        top_strategies = [s for s, _ in rankings[:min(3, len(rankings))]]

        if len(top_strategies) >= 2:
            parent1 = self.rng.choice(top_strategies)
            parent2 = self.rng.choice([s for s in top_strategies if s.id != parent1.id])
            new_strategy = crossover_strategies(parent1, parent2, self.rng)
        else:
            # Mutate top performer
            mutation = self.mutator.mutate(top_strategies[0])
            new_strategy = mutation.new_strategy

        self.state.strategies.append(new_strategy)
        self.state.discoveries += 1

        return new_strategy

    def retire_strategy(self, strategy: Strategy) -> bool:
        """Remove an underperforming strategy from the active pool.

        Args:
            strategy: Strategy to retire.

        Returns:
            True if retired, False if protected.
        """
        # Never retire built-in strategies
        if strategy.id in BUILTIN_STRATEGIES:
            return False

        # Check if strategy is in elite
        rankings = self.db.strategy_rankings()
        elite_ids = {s.id for s, _ in rankings[: self.config.elite_count]}

        if strategy.id in elite_ids:
            return False

        # Retire the strategy
        self.state.strategies = [s for s in self.state.strategies if s.id != strategy.id]
        self.state.retired_strategies.append(strategy)

        return True

    def evolve_population(self) -> list[Strategy]:
        """Evolve the strategy population through selection and mutation.

        Returns:
            List of new strategies created.
        """
        new_strategies = []

        # Selection: tournament selection for parents
        def tournament_select() -> Strategy:
            candidates = self.rng.sample(
                self.state.strategies,
                min(self.config.tournament_size, len(self.state.strategies)),
            )
            # Score candidates
            scores = []
            for s in candidates:
                records = self.db.query_performance(s.id)
                if records and isinstance(records, dict):
                    avg_success = sum(
                        r.success_rate for r in records.values()
                    ) / max(1, len(records))
                else:
                    avg_success = 0.5  # Default for unknown
                scores.append((s, avg_success))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[0][0]

        # Mutation: create mutated variants
        for _ in range(int(len(self.state.strategies) * self.config.mutation_rate)):
            if self.rng.random() < 0.3:
                # Crossover
                parent1 = tournament_select()
                parent2 = tournament_select()
                if parent1.id != parent2.id:
                    child = crossover_strategies(parent1, parent2, self.rng)
                    new_strategies.append(child)
                    self.state.mutations_applied += 1
            else:
                # Mutation
                parent = tournament_select()
                mutation = self.mutator.mutate(parent)
                new_strategies.append(mutation.new_strategy)
                self.state.mutations_applied += 1

        # Add new strategies to population
        self.state.strategies.extend(new_strategies)

        # Cull population if too large
        self._cull_population()

        return new_strategies

    def _cull_population(self) -> None:
        """Remove excess strategies to maintain population size."""
        if len(self.state.strategies) <= self.config.population_size:
            return

        # Get rankings
        rankings = self.db.strategy_rankings()
        ranked_ids = [s.id for s, _ in rankings]

        # Keep elite
        elite_ids = set(ranked_ids[: self.config.elite_count])

        # Keep built-ins
        builtin_ids = set(BUILTIN_STRATEGIES.keys())

        # Score all strategies
        strategy_scores: list[tuple[Strategy, float]] = []
        for strategy in self.state.strategies:
            if strategy.id in elite_ids or strategy.id in builtin_ids:
                score = float("inf")  # Never cull
            else:
                # Score based on ranking
                try:
                    rank = ranked_ids.index(strategy.id)
                    score = 1.0 / (rank + 1)
                except ValueError:
                    score = 0.0  # Unranked

            strategy_scores.append((strategy, score))

        # Sort by score descending and keep top N
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        keep = strategy_scores[: self.config.population_size]

        retired = [
            s for s, _ in strategy_scores[self.config.population_size :]
            if s.id not in builtin_ids
        ]

        self.state.strategies = [s for s, _ in keep]
        self.state.retired_strategies.extend(retired)

    async def run_iteration(
        self,
        problems: list[tuple[str, str | None]],
        executor: Callable[[Strategy, str], Any],
    ) -> dict[str, Any]:
        """Run a single learning iteration.

        Args:
            problems: List of (problem, ground_truth) tuples.
            executor: Async function to execute strategies.

        Returns:
            Dictionary with iteration metrics.
        """
        self.state.iteration += 1

        results: list[tuple[Strategy, StrategyResult]] = []
        successes = 0
        total_tokens = 0

        for problem, ground_truth in problems:
            # Classify and recommend
            classification = classify_problem(problem)
            recommendation = self.recommender.recommend(problem)

            # Execute strategy
            result = await self.evaluator.evaluate_strategy(
                recommendation.strategy,
                problem,
                executor,
                ground_truth,
            )

            # Update result with problem type
            result.problem_type = classification.problem_type.id

            results.append((recommendation.strategy, result))

            if result.success:
                successes += 1
            total_tokens += result.tokens_used

        # Update learning
        self.update_recommendations(results)

        # Evolve population periodically
        new_strategies = []
        if self.state.iteration % 5 == 0:
            new_strategies = self.evolve_population()

        # Record metrics
        metrics = {
            "iteration": self.state.iteration,
            "problems_solved": len(problems),
            "success_rate": successes / max(1, len(problems)),
            "total_tokens": total_tokens,
            "population_size": len(self.state.strategies),
            "new_strategies": len(new_strategies),
            "mutations_applied": self.state.mutations_applied,
            "discoveries": self.state.discoveries,
        }

        self.state.performance_history.append(metrics)

        return metrics


async def meta_learn(
    problems: list[tuple[str, str | None]],
    iterations: int,
    executor: Callable[[Strategy, str], Any],
    config: MetaConfig | None = None,
) -> LearningState:
    """Run the meta-learning loop.

    Args:
        problems: List of (problem, ground_truth) tuples.
        iterations: Number of iterations to run.
        executor: Async function to execute strategies.
        config: Optional configuration.

    Returns:
        Final learning state.
    """
    loop = LearningLoop(config)

    for i in range(iterations):
        # Shuffle problems each iteration
        shuffled = list(problems)
        loop.rng.shuffle(shuffled)

        await loop.run_iteration(shuffled, executor)

    return loop.state


def update_recommendations(
    results: list[tuple[Strategy, StrategyResult]],
    config: MetaConfig | None = None,
) -> None:
    """Update recommendations based on results.

    Args:
        results: List of (strategy, result) tuples.
        config: Optional configuration.
    """
    loop = LearningLoop(config)
    loop.update_recommendations(results)


def discover_strategy(
    problem_type: str,
    config: MetaConfig | None = None,
) -> Strategy | None:
    """Try to discover a new strategy.

    Args:
        problem_type: Problem type to optimize for.
        config: Optional configuration.

    Returns:
        New strategy if discovered.
    """
    loop = LearningLoop(config)
    return loop.discover_strategy(problem_type)


def retire_strategy(
    strategy: Strategy,
    config: MetaConfig | None = None,
) -> bool:
    """Retire an underperforming strategy.

    Args:
        strategy: Strategy to retire.
        config: Optional configuration.

    Returns:
        True if retired.
    """
    loop = LearningLoop(config)
    return loop.retire_strategy(strategy)


class IncrementalLearner:
    """Learns incrementally from each interaction."""

    def __init__(
        self,
        config: MetaConfig | None = None,
        db: PerformanceDB | None = None,
    ):
        """Initialize incremental learner.

        Args:
            config: Configuration.
            db: Performance database.
        """
        self.config = config or MetaConfig()
        self.db = db or get_db()
        self.recommender = StrategyRecommender(self.db, self.config)
        self.evaluator = StrategyEvaluator()
        self.interaction_count = 0

    def learn_from_interaction(
        self,
        problem: str,
        strategy: Strategy,
        output: str,
        success: bool | None = None,
        tokens_used: int = 0,
        ground_truth: str | None = None,
    ) -> StrategyResult:
        """Learn from a single interaction.

        Args:
            problem: The problem solved.
            strategy: The strategy used.
            output: The output produced.
            success: Whether it succeeded (if known).
            tokens_used: Tokens used.
            ground_truth: Expected answer if known.

        Returns:
            The evaluated result.
        """
        # Evaluate the result
        result = self.evaluator.evaluate_strategy_sync(
            strategy, problem, output, tokens_used, ground_truth
        )

        if success is not None:
            result.success = success

        # Classify problem
        classification = classify_problem(problem)
        result.problem_type = classification.problem_type.id

        # Record result
        self.db.record_result(strategy, result.problem_type, result)

        self.interaction_count += 1

        return result

    def get_learning_summary(self) -> dict[str, Any]:
        """Get summary of learning progress.

        Returns:
            Dictionary with learning summary.
        """
        stats = self.db.get_statistics()

        return {
            "interactions": self.interaction_count,
            "total_results": stats["total_results"],
            "strategies_seen": stats["total_strategies"],
            "problem_types_seen": stats["total_problem_types"],
            "overall_success_rate": stats["overall_success_rate"],
        }
