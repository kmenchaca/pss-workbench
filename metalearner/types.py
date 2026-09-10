"""Type definitions for the meta-learner system.

This module defines all the core data structures used throughout the
meta-learning system for strategy representation, evaluation, and learning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MutationType(Enum):
    """Types of mutations that can be applied to strategies."""
    PARAMETER_TWEAK = "parameter_tweak"
    PROMPT_EDIT = "prompt_edit"
    COMBINE = "combine"
    SIMPLIFY = "simplify"
    EXTEND = "extend"


class ProblemCategory(Enum):
    """High-level problem categorizations."""
    FACTUAL = "factual"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    TECHNICAL = "technical"
    PLANNING = "planning"


@dataclass
class Strategy:
    """An exploration strategy that can be applied to problems.

    Attributes:
        id: Unique identifier for the strategy.
        name: Human-readable name.
        description: Detailed description of how the strategy works.
        parameters: Configuration parameters for the strategy.
        prompt_template: Template for generating prompts using this strategy.
        created_at: When the strategy was created.
        parent_ids: IDs of strategies this was derived from (for genealogy).
    """
    id: str
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    prompt_template: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    parent_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize strategy to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "prompt_template": self.prompt_template,
            "created_at": self.created_at.isoformat(),
            "parent_ids": self.parent_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Strategy":
        """Deserialize strategy from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            prompt_template=data.get("prompt_template", ""),
            created_at=created_at,
            parent_ids=data.get("parent_ids", []),
        )


@dataclass
class StrategyResult:
    """Outcome of applying a strategy to a problem.

    Attributes:
        strategy_id: ID of the strategy that was used.
        success: Whether the strategy succeeded.
        tokens_used: Number of tokens consumed.
        quality_score: Quality rating from 0.0 to 1.0.
        problem_type: Category of the problem.
        execution_time_ms: How long execution took.
        output: The actual output produced.
        error: Error message if failed.
        timestamp: When this result was recorded.
    """
    strategy_id: str
    success: bool
    tokens_used: int
    quality_score: float
    problem_type: str
    execution_time_ms: int = 0
    output: str = ""
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "success": self.success,
            "tokens_used": self.tokens_used,
            "quality_score": self.quality_score,
            "problem_type": self.problem_type,
            "execution_time_ms": self.execution_time_ms,
            "output": self.output,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyResult":
        """Deserialize result from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        return cls(
            strategy_id=data["strategy_id"],
            success=data["success"],
            tokens_used=data["tokens_used"],
            quality_score=data["quality_score"],
            problem_type=data["problem_type"],
            execution_time_ms=data.get("execution_time_ms", 0),
            output=data.get("output", ""),
            error=data.get("error"),
            timestamp=timestamp,
        )


@dataclass
class ProblemType:
    """Categorization of a problem type with associated metadata.

    Attributes:
        id: Unique identifier for the problem type.
        name: Human-readable name.
        category: High-level category (FACTUAL, CREATIVE, etc.).
        features: Extracted features that identify this problem type.
        recommended_strategies: Strategy IDs known to work well.
        description: Detailed description of this problem type.
    """
    id: str
    name: str
    category: ProblemCategory
    features: list[str] = field(default_factory=list)
    recommended_strategies: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize problem type to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "features": self.features,
            "recommended_strategies": self.recommended_strategies,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProblemType":
        """Deserialize problem type from dictionary."""
        category = data.get("category", "analytical")
        if isinstance(category, str):
            category = ProblemCategory(category)

        return cls(
            id=data["id"],
            name=data["name"],
            category=category,
            features=data.get("features", []),
            recommended_strategies=data.get("recommended_strategies", []),
            description=data.get("description", ""),
        )


@dataclass
class PerformanceRecord:
    """Historical performance data for a strategy on a problem type.

    Attributes:
        strategy_id: ID of the strategy.
        problem_type: Type of problems this record covers.
        results: List of individual results.
        success_rate: Calculated success rate.
        avg_quality: Average quality score.
        avg_tokens: Average tokens used.
        sample_count: Number of samples.
    """
    strategy_id: str
    problem_type: str
    results: list[StrategyResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate from results."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def avg_quality(self) -> float:
        """Calculate average quality score."""
        if not self.results:
            return 0.0
        return sum(r.quality_score for r in self.results) / len(self.results)

    @property
    def avg_tokens(self) -> float:
        """Calculate average tokens used."""
        if not self.results:
            return 0.0
        return sum(r.tokens_used for r in self.results) / len(self.results)

    @property
    def sample_count(self) -> int:
        """Return number of samples."""
        return len(self.results)

    def to_dict(self) -> dict[str, Any]:
        """Serialize performance record to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "problem_type": self.problem_type,
            "results": [r.to_dict() for r in self.results],
            "success_rate": self.success_rate,
            "avg_quality": self.avg_quality,
            "avg_tokens": self.avg_tokens,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceRecord":
        """Deserialize performance record from dictionary."""
        return cls(
            strategy_id=data["strategy_id"],
            problem_type=data["problem_type"],
            results=[StrategyResult.from_dict(r) for r in data.get("results", [])],
        )


@dataclass
class StrategyMutation:
    """A modification applied to a strategy to create a variant.

    Attributes:
        original_id: ID of the original strategy.
        mutation_type: Type of mutation applied.
        new_strategy: The resulting mutated strategy.
        description: Description of what was changed.
        timestamp: When the mutation was created.
    """
    original_id: str
    mutation_type: MutationType
    new_strategy: Strategy
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize mutation to dictionary."""
        return {
            "original_id": self.original_id,
            "mutation_type": self.mutation_type.value,
            "new_strategy": self.new_strategy.to_dict(),
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyMutation":
        """Deserialize mutation from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        return cls(
            original_id=data["original_id"],
            mutation_type=MutationType(data["mutation_type"]),
            new_strategy=Strategy.from_dict(data["new_strategy"]),
            description=data.get("description", ""),
            timestamp=timestamp,
        )


@dataclass
class MetaConfig:
    """Configuration for the meta-learning system.

    Attributes:
        learning_rate: How quickly to update strategy weights.
        forgetting_factor: Rate at which old results decay.
        exploration_rate: Probability of trying non-optimal strategies.
        min_samples_for_recommendation: Minimum samples before recommending.
        mutation_rate: Probability of mutating strategies.
        tournament_size: Size of tournament for selection.
        population_size: Number of strategies to maintain.
        elite_count: Number of top strategies to preserve.
        max_strategy_age: Maximum age before retirement consideration.
        bootstrap_depth: Maximum recursion depth for bootstrap.
        db_path: Path to performance database.
    """
    learning_rate: float = 0.1
    forgetting_factor: float = 0.95
    exploration_rate: float = 0.1
    min_samples_for_recommendation: int = 3
    mutation_rate: float = 0.2
    tournament_size: int = 3
    population_size: int = 20
    elite_count: int = 5
    max_strategy_age: int = 100
    bootstrap_depth: int = 3
    db_path: str = "metalearner_db.json"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "forgetting_factor": self.forgetting_factor,
            "exploration_rate": self.exploration_rate,
            "min_samples_for_recommendation": self.min_samples_for_recommendation,
            "mutation_rate": self.mutation_rate,
            "tournament_size": self.tournament_size,
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "max_strategy_age": self.max_strategy_age,
            "bootstrap_depth": self.bootstrap_depth,
            "db_path": self.db_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetaConfig":
        """Deserialize config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ClassificationResult:
    """Result of classifying a problem.

    Attributes:
        problem_type: The identified problem type.
        confidence: Confidence in the classification (0.0 to 1.0).
        features: Features extracted from the problem.
        alternative_types: Other possible classifications with confidence.
    """
    problem_type: ProblemType
    confidence: float
    features: list[str] = field(default_factory=list)
    alternative_types: list[tuple[ProblemType, float]] = field(default_factory=list)


@dataclass
class RecommendationResult:
    """Result of recommending a strategy.

    Attributes:
        strategy: The recommended strategy.
        confidence: Confidence in the recommendation.
        explanation: Why this strategy was recommended.
        alternatives: Other strategies considered.
        fallback: Fallback strategy if primary fails.
    """
    strategy: Strategy
    confidence: float
    explanation: str = ""
    alternatives: list[tuple[Strategy, float]] = field(default_factory=list)
    fallback: Strategy | None = None


@dataclass
class EvaluationResult:
    """Result of evaluating strategies.

    Attributes:
        rankings: Strategies ranked by performance.
        metrics: Per-strategy metrics.
        statistical_significance: P-values for comparisons.
        best_strategy: The top-performing strategy.
    """
    rankings: list[tuple[Strategy, float]] = field(default_factory=list)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    statistical_significance: dict[str, float] = field(default_factory=dict)
    best_strategy: Strategy | None = None


@dataclass
class LearningState:
    """State of the learning loop.

    Attributes:
        iteration: Current iteration number.
        strategies: Active strategies.
        retired_strategies: Strategies that have been retired.
        performance_history: Performance over time.
        mutations_applied: Count of mutations applied.
        discoveries: New strategies discovered.
    """
    iteration: int = 0
    strategies: list[Strategy] = field(default_factory=list)
    retired_strategies: list[Strategy] = field(default_factory=list)
    performance_history: list[dict[str, float]] = field(default_factory=list)
    mutations_applied: int = 0
    discoveries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize learning state to dictionary."""
        return {
            "iteration": self.iteration,
            "strategies": [s.to_dict() for s in self.strategies],
            "retired_strategies": [s.to_dict() for s in self.retired_strategies],
            "performance_history": self.performance_history,
            "mutations_applied": self.mutations_applied,
            "discoveries": self.discoveries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningState":
        """Deserialize learning state from dictionary."""
        return cls(
            iteration=data.get("iteration", 0),
            strategies=[Strategy.from_dict(s) for s in data.get("strategies", [])],
            retired_strategies=[Strategy.from_dict(s) for s in data.get("retired_strategies", [])],
            performance_history=data.get("performance_history", []),
            mutations_applied=data.get("mutations_applied", 0),
            discoveries=data.get("discoveries", 0),
        )
