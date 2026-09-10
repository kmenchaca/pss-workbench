"""Strategy recommendation for the meta-learner system.

This module provides functionality to recommend strategies based on
problem classification and historical performance data.
"""

import random
from typing import Any, Callable

from .classification import classify_problem, get_problem_type
from .database import PerformanceDB, get_db
from .strategies import BUILTIN_STRATEGIES, get_builtin_strategy
from .types import (
    MetaConfig,
    RecommendationResult,
    Strategy,
)


class StrategyRecommender:
    """Recommends strategies based on problem analysis and history."""

    def __init__(
        self,
        db: PerformanceDB | None = None,
        config: MetaConfig | None = None,
        rng: random.Random | None = None,
    ):
        """Initialize the recommender.

        Args:
            db: Performance database to use.
            config: Configuration settings.
            rng: Random number generator.
        """
        self.db = db or get_db()
        self.config = config or MetaConfig()
        self.rng = rng or random.Random()

    def recommend(self, problem: str) -> RecommendationResult:
        """Recommend a strategy for a problem.

        Args:
            problem: The problem text.

        Returns:
            RecommendationResult with recommended strategy.
        """
        # Classify the problem
        classification = classify_problem(problem)
        problem_type = classification.problem_type

        # Check exploration vs exploitation
        if self.rng.random() < self.config.exploration_rate:
            return self._explore(problem_type)

        # Try to get best strategy from history
        best, score = self.db.best_strategy_for(
            problem_type.id,
            min_samples=self.config.min_samples_for_recommendation,
        )

        if best and score > 0.5:
            # Use historical best
            explanation = self._generate_explanation(
                best, problem_type, "historical performance"
            )
            alternatives = self._get_alternatives(problem_type, exclude=best.id)
            fallback = self._get_fallback(problem_type, exclude=best.id)

            return RecommendationResult(
                strategy=best,
                confidence=score * classification.confidence,
                explanation=explanation,
                alternatives=alternatives,
                fallback=fallback,
            )

        # Fall back to problem type's recommended strategies
        return self._recommend_from_type(problem_type, classification.confidence)

    def _explore(self, problem_type: Any) -> RecommendationResult:
        """Choose a random strategy for exploration.

        Args:
            problem_type: The problem type.

        Returns:
            RecommendationResult with random strategy.
        """
        strategies = list(BUILTIN_STRATEGIES.values())
        strategy = self.rng.choice(strategies)

        return RecommendationResult(
            strategy=strategy,
            confidence=0.3,  # Low confidence for exploration
            explanation=f"Exploration mode: trying {strategy.name} to gather data",
            alternatives=[(s, 0.3) for s in strategies if s.id != strategy.id][:3],
            fallback=get_builtin_strategy("minimal"),
        )

    def _recommend_from_type(
        self,
        problem_type: Any,
        confidence: float,
    ) -> RecommendationResult:
        """Recommend based on problem type's defaults.

        Args:
            problem_type: The problem type.
            confidence: Classification confidence.

        Returns:
            RecommendationResult.
        """
        recommended_ids = problem_type.recommended_strategies

        if recommended_ids:
            strategy_id = recommended_ids[0]
            strategy = (
                self.db.get_strategy(strategy_id)
                or get_builtin_strategy(strategy_id)
            )

            if strategy:
                explanation = self._generate_explanation(
                    strategy, problem_type, "problem type match"
                )
                alternatives = [
                    (get_builtin_strategy(sid), 0.5)
                    for sid in recommended_ids[1:4]
                    if get_builtin_strategy(sid)
                ]
                fallback = self._get_fallback(problem_type, exclude=strategy.id)

                return RecommendationResult(
                    strategy=strategy,
                    confidence=confidence * 0.7,
                    explanation=explanation,
                    alternatives=alternatives,
                    fallback=fallback,
                )

        # Ultimate fallback to chain of thought
        cot = get_builtin_strategy("chain_of_thought")
        if cot:
            return RecommendationResult(
                strategy=cot,
                confidence=0.4,
                explanation="Default: Chain of Thought works well for most problems",
                alternatives=[],
                fallback=get_builtin_strategy("minimal"),
            )

        # Should never happen, but provide ultimate fallback
        return RecommendationResult(
            strategy=list(BUILTIN_STRATEGIES.values())[0],
            confidence=0.2,
            explanation="Fallback strategy",
            alternatives=[],
            fallback=None,
        )

    def _get_alternatives(
        self,
        problem_type: Any,
        exclude: str | None = None,
        max_count: int = 3,
    ) -> list[tuple[Strategy, float]]:
        """Get alternative strategies.

        Args:
            problem_type: The problem type.
            exclude: Strategy ID to exclude.
            max_count: Maximum alternatives to return.

        Returns:
            List of (strategy, confidence) tuples.
        """
        rankings = self.db.strategy_rankings(
            problem_type.id,
            metric="success_rate",
        )

        alternatives = [
            (s, score) for s, score in rankings
            if s.id != exclude
        ][:max_count]

        # Add type recommendations if not enough
        if len(alternatives) < max_count:
            for sid in problem_type.recommended_strategies:
                if sid != exclude and len(alternatives) < max_count:
                    strategy = get_builtin_strategy(sid)
                    if strategy and not any(s.id == sid for s, _ in alternatives):
                        alternatives.append((strategy, 0.5))

        return alternatives

    def _get_fallback(
        self,
        problem_type: Any,
        exclude: str | None = None,
    ) -> Strategy | None:
        """Get a fallback strategy.

        Args:
            problem_type: The problem type.
            exclude: Strategy ID to exclude.

        Returns:
            Fallback strategy or None.
        """
        # Try minimal first
        minimal = get_builtin_strategy("minimal")
        if minimal and minimal.id != exclude:
            return minimal

        # Try chain of thought
        cot = get_builtin_strategy("chain_of_thought")
        if cot and cot.id != exclude:
            return cot

        # Any other strategy
        for strategy in BUILTIN_STRATEGIES.values():
            if strategy.id != exclude:
                return strategy

        return None

    def _generate_explanation(
        self,
        strategy: Strategy,
        problem_type: Any,
        basis: str,
    ) -> str:
        """Generate explanation for recommendation.

        Args:
            strategy: The recommended strategy.
            problem_type: The problem type.
            basis: Basis for recommendation.

        Returns:
            Explanation string.
        """
        record = self.db.query_performance(strategy.id, problem_type.id)

        if record and hasattr(record, "success_rate"):
            return (
                f"Recommending {strategy.name} for {problem_type.name} problems "
                f"based on {basis}. "
                f"Historical success rate: {record.success_rate:.1%} "
                f"({record.sample_count} samples)"
            )

        return (
            f"Recommending {strategy.name} for {problem_type.name} problems "
            f"based on {basis}. {strategy.description}"
        )

    def explain_recommendation(
        self,
        problem: str,
        strategy: Strategy,
    ) -> str:
        """Generate detailed explanation of why a strategy was recommended.

        Args:
            problem: The problem text.
            strategy: The recommended strategy.

        Returns:
            Detailed explanation string.
        """
        classification = classify_problem(problem)

        lines = [
            f"Strategy Recommendation: {strategy.name}",
            "",
            f"Problem Classification:",
            f"  - Type: {classification.problem_type.name}",
            f"  - Category: {classification.problem_type.category.value}",
            f"  - Confidence: {classification.confidence:.1%}",
            f"  - Features: {', '.join(classification.features)}",
            "",
            f"Strategy Description:",
            f"  {strategy.description}",
            "",
        ]

        # Add performance data if available
        record = self.db.query_performance(
            strategy.id,
            classification.problem_type.id,
        )

        if record and hasattr(record, "success_rate"):
            lines.extend([
                f"Historical Performance on {classification.problem_type.name}:",
                f"  - Success rate: {record.success_rate:.1%}",
                f"  - Average quality: {record.avg_quality:.2f}",
                f"  - Average tokens: {record.avg_tokens:.0f}",
                f"  - Sample count: {record.sample_count}",
            ])
        else:
            lines.append("No historical performance data available.")

        return "\n".join(lines)


def recommend_strategy(problem: str) -> RecommendationResult:
    """Recommend a strategy for a problem.

    Args:
        problem: The problem text.

    Returns:
        RecommendationResult with recommendation.
    """
    recommender = StrategyRecommender()
    return recommender.recommend(problem)


def explain_recommendation(problem: str, strategy: Strategy) -> str:
    """Explain why a strategy was recommended.

    Args:
        problem: The problem text.
        strategy: The recommended strategy.

    Returns:
        Explanation string.
    """
    recommender = StrategyRecommender()
    return recommender.explain_recommendation(problem, strategy)


def fallback_strategy(problem: str) -> Strategy:
    """Get a fallback strategy for a problem.

    Args:
        problem: The problem text.

    Returns:
        A safe fallback strategy.
    """
    classification = classify_problem(problem)
    recommender = StrategyRecommender()
    fallback = recommender._get_fallback(classification.problem_type)

    if fallback:
        return fallback

    # Ultimate fallback
    return list(BUILTIN_STRATEGIES.values())[0]


class UCBRecommender(StrategyRecommender):
    """Recommender using Upper Confidence Bound for exploration/exploitation.

    UCB balances exploration and exploitation by adding a confidence bonus
    to strategies with fewer samples.
    """

    def __init__(
        self,
        db: PerformanceDB | None = None,
        config: MetaConfig | None = None,
        exploration_weight: float = 2.0,
    ):
        """Initialize UCB recommender.

        Args:
            db: Performance database.
            config: Configuration.
            exploration_weight: Weight for exploration bonus.
        """
        super().__init__(db, config)
        self.exploration_weight = exploration_weight
        self.total_trials = 0

    def recommend(self, problem: str) -> RecommendationResult:
        """Recommend using UCB algorithm.

        Args:
            problem: The problem text.

        Returns:
            RecommendationResult.
        """
        import math

        classification = classify_problem(problem)
        problem_type = classification.problem_type

        # Get all strategies with scores
        all_strategies = list(BUILTIN_STRATEGIES.values())
        all_strategies.extend(
            s for s in self.db.get_all_strategies()
            if s.id not in BUILTIN_STRATEGIES
        )

        self.total_trials += 1
        ucb_scores: list[tuple[Strategy, float]] = []

        for strategy in all_strategies:
            record = self.db.query_performance(strategy.id, problem_type.id)

            if record and hasattr(record, "success_rate") and record.sample_count > 0:
                # Exploitation: average reward
                avg_reward = record.success_rate

                # Exploration: UCB bonus
                exploration_bonus = math.sqrt(
                    self.exploration_weight
                    * math.log(self.total_trials)
                    / record.sample_count
                )

                ucb_score = avg_reward + exploration_bonus
            else:
                # No data - maximum exploration bonus
                ucb_score = 1.0 + self.exploration_weight

            ucb_scores.append((strategy, ucb_score))

        # Select best UCB score
        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        best_strategy, best_score = ucb_scores[0]

        # Get record for confidence calculation
        record = self.db.query_performance(best_strategy.id, problem_type.id)
        if record and hasattr(record, "sample_count") and record.sample_count > 0:
            confidence = min(0.9, record.sample_count / 20)  # Cap at 90%
        else:
            confidence = 0.3

        return RecommendationResult(
            strategy=best_strategy,
            confidence=confidence * classification.confidence,
            explanation=f"UCB selection: {best_strategy.name} (UCB score: {best_score:.3f})",
            alternatives=[(s, score) for s, score in ucb_scores[1:4]],
            fallback=self._get_fallback(problem_type, exclude=best_strategy.id),
        )


class ThompsonSamplingRecommender(StrategyRecommender):
    """Recommender using Thompson Sampling for exploration/exploitation.

    Thompson Sampling samples from posterior distributions to balance
    exploration and exploitation in a principled Bayesian way.
    """

    def __init__(
        self,
        db: PerformanceDB | None = None,
        config: MetaConfig | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        """Initialize Thompson Sampling recommender.

        Args:
            db: Performance database.
            config: Configuration.
            prior_alpha: Prior alpha for Beta distribution.
            prior_beta: Prior beta for Beta distribution.
        """
        super().__init__(db, config)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def recommend(self, problem: str) -> RecommendationResult:
        """Recommend using Thompson Sampling.

        Args:
            problem: The problem text.

        Returns:
            RecommendationResult.
        """
        classification = classify_problem(problem)
        problem_type = classification.problem_type

        all_strategies = list(BUILTIN_STRATEGIES.values())
        all_strategies.extend(
            s for s in self.db.get_all_strategies()
            if s.id not in BUILTIN_STRATEGIES
        )

        samples: list[tuple[Strategy, float]] = []

        for strategy in all_strategies:
            record = self.db.query_performance(strategy.id, problem_type.id)

            if record and hasattr(record, "results") and record.results:
                successes = sum(1 for r in record.results if r.success)
                failures = len(record.results) - successes

                alpha = self.prior_alpha + successes
                beta = self.prior_beta + failures
            else:
                alpha = self.prior_alpha
                beta = self.prior_beta

            # Sample from Beta distribution
            sample = self.rng.betavariate(alpha, beta)
            samples.append((strategy, sample))

        # Select highest sample
        samples.sort(key=lambda x: x[1], reverse=True)
        best_strategy, best_sample = samples[0]

        # Calculate confidence
        record = self.db.query_performance(best_strategy.id, problem_type.id)
        if record and hasattr(record, "sample_count") and record.sample_count > 0:
            confidence = min(0.9, record.sample_count / 20)
        else:
            confidence = 0.3

        return RecommendationResult(
            strategy=best_strategy,
            confidence=confidence * classification.confidence,
            explanation=f"Thompson Sampling: {best_strategy.name} (sample: {best_sample:.3f})",
            alternatives=[(s, score) for s, score in samples[1:4]],
            fallback=self._get_fallback(problem_type, exclude=best_strategy.id),
        )
