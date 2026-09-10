"""Meta-Learner Integration - Wire Meta-Learner as master system.

This module integrates the Meta-Learner to control the unified harness,
learning from outcomes to improve system selection over time.

The Meta-Learner acts as the "brain" that:
1. Classifies incoming tasks
2. Recommends the best system(s) based on learned patterns
3. Records outcomes to improve future recommendations
4. Evolves its routing strategies through experience

Example usage:
    from unified import LearnerControlledHarness

    harness = LearnerControlledHarness()
    result = await harness.solve("Should we adopt microservices?")
    # Meta-learner automatically selects best system and learns from outcome
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import SystemType, get_system_type
from .harness import UnifiedHarness, SystemResult
from .meta_harness import MetaHarness, MetaResult, TaskAnalysis


# Map Meta-Learner problem types to unified systems
PROBLEM_TYPE_TO_SYSTEM = {
    "factual": [SystemType.KNOWLEDGE],
    "analytical": [SystemType.PERSONAS, SystemType.ARGSWARM],
    "creative": [SystemType.DREAMLOGIC, SystemType.EVOLUTION],
    "planning": [SystemType.TEMPORAL, SystemType.EMBODIED],
    "security": [SystemType.REDTEAM],
    "debate": [SystemType.ARGSWARM, SystemType.NEGOTIATE],
    "optimization": [SystemType.EVOLUTION, SystemType.METALEARNER],
    "negotiation": [SystemType.NEGOTIATE, SystemType.PERSONAS],
    "multi_perspective": [SystemType.PERSONAS, SystemType.ARGSWARM],
}


@dataclass
class LearnerResult:
    """Result from learner-controlled run.

    Attributes:
        result: The system result
        classification: How the task was classified
        strategy_used: Meta-learner strategy that was applied
        learning_recorded: Whether outcome was recorded for learning
        confidence: Confidence in the recommendation
    """

    result: SystemResult
    classification: str
    strategy_used: str
    learning_recorded: bool
    confidence: float


@dataclass
class LearningStats:
    """Statistics about meta-learning.

    Attributes:
        total_tasks: Total tasks processed
        success_rate: Overall success rate
        system_performance: Performance by system
        classification_distribution: Tasks by classification
    """

    total_tasks: int
    success_rate: float
    system_performance: dict[str, float]
    classification_distribution: dict[str, int]


class LearnerControlledHarness:
    """Harness controlled by the Meta-Learner.

    The Meta-Learner acts as the master system that:
    - Classifies tasks to determine best approach
    - Selects optimal system(s) from the 10 available
    - Records outcomes to improve future selections
    - Adapts its routing strategy through experience
    """

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "meta-llama/llama-3.1-8b-instruct",
        learning_enabled: bool = True,
    ):
        """Initialize the learner-controlled harness.

        Args:
            provider: LLM provider
            model: Model to use
            learning_enabled: Whether to record outcomes for learning
        """
        self.provider = provider
        self.model = model
        self.learning_enabled = learning_enabled

        # Initialize components
        self.unified = UnifiedHarness(provider=provider, model=model)
        self.meta = MetaHarness(provider=provider, model=model)

        # Import meta-learner components
        try:
            from metalearner import (
                ProblemClassifier,
                PerformanceDB,
                StrategyRecommender,
                record_result,
            )
            self.classifier = ProblemClassifier()
            self.db = PerformanceDB()
            self.recommender = StrategyRecommender()
            self._record_result = record_result
            self.metalearner_available = True
        except ImportError:
            self.metalearner_available = False
            self.classifier = None
            self.db = None
            self.recommender = None

        # Track performance
        self.history: list[dict[str, Any]] = []

    def classify_task(self, prompt: str) -> str:
        """Classify a task using the meta-learner.

        Args:
            prompt: The task prompt

        Returns:
            Classification string
        """
        if self.metalearner_available and self.classifier:
            try:
                from metalearner import classify_problem
                result = classify_problem(prompt)
                return result.problem_type.value if hasattr(result, 'problem_type') else str(result)
            except Exception:
                pass

        # Fallback classification based on keywords
        return self._fallback_classify(prompt)

    def _fallback_classify(self, prompt: str) -> str:
        """Fallback classification using keywords.

        Args:
            prompt: The task prompt

        Returns:
            Classification string
        """
        prompt_lower = prompt.lower()

        classifications = {
            "debate": ["debate", "argue", "for and against", "pros and cons"],
            "security": ["security", "vulnerability", "attack", "hack"],
            "creative": ["creative", "imagine", "brainstorm", "dream"],
            "planning": ["plan", "future", "scenario", "forecast"],
            "analytical": ["analyze", "evaluate", "compare", "assess"],
            "negotiation": ["negotiate", "deal", "agreement", "compromise"],
            "optimization": ["optimize", "improve", "evolve", "better"],
        }

        for category, keywords in classifications.items():
            if any(kw in prompt_lower for kw in keywords):
                return category

        return "analytical"  # Default

    def get_recommended_systems(self, classification: str) -> list[SystemType]:
        """Get recommended systems for a classification.

        Args:
            classification: Task classification

        Returns:
            List of recommended system types
        """
        # Check learned preferences first
        if self.metalearner_available and self.db:
            try:
                from metalearner import best_strategy_for
                best = best_strategy_for(classification)
                if best:
                    # Map strategy to system if possible
                    strategy_name = best.name if hasattr(best, 'name') else str(best)
                    mapped = self._strategy_to_systems(strategy_name)
                    if mapped:
                        return mapped
            except Exception:
                pass

        # Fall back to predefined mapping
        return PROBLEM_TYPE_TO_SYSTEM.get(
            classification,
            [SystemType.PERSONAS]  # Default
        )

    def _strategy_to_systems(self, strategy_name: str) -> list[SystemType]:
        """Map a meta-learner strategy to unified systems.

        Args:
            strategy_name: Strategy name

        Returns:
            List of system types
        """
        strategy_mapping = {
            "chain_of_thought": [SystemType.PERSONAS],
            "decomposition": [SystemType.TEMPORAL, SystemType.EMBODIED],
            "adversarial": [SystemType.REDTEAM, SystemType.ARGSWARM],
            "analogical": [SystemType.DREAMLOGIC, SystemType.KNOWLEDGE],
            "tool_heavy": [SystemType.EMBODIED, SystemType.EVOLUTION],
            "minimal": [SystemType.KNOWLEDGE],
            "exhaustive": [SystemType.PERSONAS, SystemType.ARGSWARM, SystemType.TEMPORAL],
        }

        return strategy_mapping.get(
            strategy_name.lower(),
            [SystemType.PERSONAS]
        )

    async def solve(
        self,
        prompt: str,
        max_systems: int = 2,
    ) -> LearnerResult:
        """Solve a task using learner-controlled system selection.

        Args:
            prompt: The task to solve
            max_systems: Maximum systems to run

        Returns:
            LearnerResult with outcome
        """
        # 1. Classify the task
        classification = self.classify_task(prompt)

        # 2. Get recommended systems
        systems = self.get_recommended_systems(classification)[:max_systems]

        # 3. Calculate confidence based on past performance
        confidence = self._get_confidence(classification, systems)

        # 4. Run the system(s)
        if len(systems) == 1:
            result = await self.unified.run(systems[0].value, prompt)
            strategy_used = systems[0].value
        else:
            # Run multiple and take best
            results = await self.unified.run_multiple(
                [s.value for s in systems],
                prompt,
                parallel=True,
            )
            # Pick the successful one with best confidence
            successful = [r for r in results if r.success]
            result = successful[0] if successful else results[0]
            strategy_used = "+".join(s.value for s in systems)

        # 5. Record outcome for learning
        learning_recorded = False
        if self.learning_enabled:
            learning_recorded = self._record_outcome(
                prompt, classification, systems, result
            )

        # 6. Track in local history
        self.history.append({
            "prompt": prompt[:100],
            "classification": classification,
            "systems": [s.value for s in systems],
            "success": result.success,
        })

        return LearnerResult(
            result=result,
            classification=classification,
            strategy_used=strategy_used,
            learning_recorded=learning_recorded,
            confidence=confidence,
        )

    async def solve_with_feedback(
        self,
        prompt: str,
        feedback_fn: Optional[Callable[[SystemResult], float]] = None,
    ) -> LearnerResult:
        """Solve with explicit feedback for learning.

        Args:
            prompt: The task
            feedback_fn: Function to evaluate result quality (returns 0-1)

        Returns:
            LearnerResult
        """
        result = await self.solve(prompt)

        # If feedback function provided, use it to rate the result
        if feedback_fn and result.result.success:
            quality = feedback_fn(result.result)
            self._record_quality_feedback(
                result.classification,
                result.strategy_used,
                quality,
            )

        return result

    def _get_confidence(
        self,
        classification: str,
        systems: list[SystemType],
    ) -> float:
        """Calculate confidence in recommendation.

        Args:
            classification: Task classification
            systems: Recommended systems

        Returns:
            Confidence score (0-1)
        """
        # Check historical performance
        relevant = [
            h for h in self.history
            if h["classification"] == classification
        ]

        if not relevant:
            return 0.5  # Default confidence

        success_rate = sum(1 for h in relevant if h["success"]) / len(relevant)
        return 0.3 + (0.7 * success_rate)  # Scale to 0.3-1.0

    def _record_outcome(
        self,
        prompt: str,
        classification: str,
        systems: list[SystemType],
        result: SystemResult,
    ) -> bool:
        """Record outcome for meta-learning.

        Args:
            prompt: Original prompt
            classification: Task classification
            systems: Systems used
            result: Outcome

        Returns:
            True if recorded successfully
        """
        if not self.metalearner_available:
            return False

        try:
            from metalearner import record_result, StrategyResult

            for system in systems:
                strategy_result = StrategyResult(
                    strategy_id=system.value,
                    problem_id=prompt[:50],
                    success=result.success,
                    quality_score=0.8 if result.success else 0.2,
                    tokens_used=result.tokens_used,
                    time_taken=result.elapsed_seconds,
                )
                record_result(strategy_result)

            return True
        except Exception:
            return False

    def _record_quality_feedback(
        self,
        classification: str,
        strategy: str,
        quality: float,
    ) -> None:
        """Record quality feedback for learning.

        Args:
            classification: Task classification
            strategy: Strategy used
            quality: Quality score (0-1)
        """
        if self.metalearner_available and self.db:
            try:
                from metalearner import PerformanceRecord

                record = PerformanceRecord(
                    strategy_id=strategy,
                    problem_type=classification,
                    success=quality > 0.5,
                    quality_score=quality,
                )
                # Would need to actually store this
            except Exception:
                pass

    def get_stats(self) -> LearningStats:
        """Get learning statistics.

        Returns:
            LearningStats
        """
        if not self.history:
            return LearningStats(
                total_tasks=0,
                success_rate=0.0,
                system_performance={},
                classification_distribution={},
            )

        total = len(self.history)
        successes = sum(1 for h in self.history if h["success"])

        # System performance
        system_results: dict[str, list[bool]] = {}
        for h in self.history:
            for sys in h["systems"]:
                if sys not in system_results:
                    system_results[sys] = []
                system_results[sys].append(h["success"])

        system_performance = {
            sys: sum(results) / len(results)
            for sys, results in system_results.items()
        }

        # Classification distribution
        classification_dist: dict[str, int] = {}
        for h in self.history:
            cls = h["classification"]
            classification_dist[cls] = classification_dist.get(cls, 0) + 1

        return LearningStats(
            total_tasks=total,
            success_rate=successes / total,
            system_performance=system_performance,
            classification_distribution=classification_dist,
        )

    def suggest_improvements(self) -> list[str]:
        """Suggest improvements based on learning.

        Returns:
            List of suggestions
        """
        suggestions = []
        stats = self.get_stats()

        if stats.total_tasks < 10:
            suggestions.append("Run more tasks to gather learning data.")
            return suggestions

        # Check for underperforming systems
        for sys, perf in stats.system_performance.items():
            if perf < 0.5:
                suggestions.append(
                    f"Consider avoiding {sys} - low success rate ({perf:.0%})"
                )

        # Check for classification gaps
        if len(stats.classification_distribution) < 4:
            suggestions.append(
                "Try more diverse task types to expand learning."
            )

        # Overall success rate
        if stats.success_rate < 0.7:
            suggestions.append(
                f"Overall success rate is {stats.success_rate:.0%}. "
                "Consider adjusting system selection or prompts."
            )

        return suggestions or ["System is performing well."]


# =============================================================================
# Convenience Functions
# =============================================================================


async def learner_solve(
    prompt: str,
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.1-8b-instruct",
) -> LearnerResult:
    """Convenience function for learner-controlled solving.

    Args:
        prompt: Task to solve
        provider: LLM provider
        model: Model to use

    Returns:
        LearnerResult
    """
    harness = LearnerControlledHarness(provider=provider, model=model)
    return await harness.solve(prompt)


def classify_for_routing(prompt: str) -> dict[str, Any]:
    """Classify a prompt and return routing info.

    Args:
        prompt: The prompt to classify

    Returns:
        Dictionary with classification and recommended systems
    """
    harness = LearnerControlledHarness()
    classification = harness.classify_task(prompt)
    systems = harness.get_recommended_systems(classification)

    return {
        "classification": classification,
        "recommended_systems": [s.value for s in systems],
        "primary_system": systems[0].value if systems else "personas",
    }
