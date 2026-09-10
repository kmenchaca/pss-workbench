"""Dataclasses for the Temporal Scenario Planner.

This module defines the core data structures for timeline simulation,
event modeling, actor behavior, and scenario comparison.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable


class EventCategory(Enum):
    """Categories of events that can occur in a timeline."""
    MARKET = "market"
    COMPETITIVE = "competitive"
    TECHNICAL = "technical"
    REGULATORY = "regulatory"
    SOCIAL = "social"
    INTERNAL = "internal"


class BehaviorModel(Enum):
    """Behavior models for actors."""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    REACTIVE = "reactive"
    PROACTIVE = "proactive"


class DecisionStatus(Enum):
    """Status of a decision point."""
    PENDING = "pending"
    CHOSEN = "chosen"
    DEFERRED = "deferred"


@dataclass
class Event:
    """Something that happens in the timeline.

    Attributes:
        id: Unique identifier for the event.
        date: When the event occurs.
        description: Human-readable description.
        cause: What caused this event (event id or decision id).
        effects: Dict mapping state keys to changes.
        probability: Likelihood of this event (0.0 to 1.0).
        category: Type of event.
        magnitude: Scale of impact (-10 to 10).
    """
    id: str
    date: datetime
    description: str
    cause: str | None = None
    effects: dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    category: EventCategory = EventCategory.INTERNAL
    magnitude: int = 0


@dataclass
class ActorState:
    """Current state of an actor.

    Attributes:
        resources: Available resources (money, people, etc.).
        position: Current market/strategic position.
        capabilities: What the actor can do.
        relationships: Standing with other actors.
        morale: Internal sentiment (-1.0 to 1.0).
    """
    resources: dict[str, float] = field(default_factory=dict)
    position: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    relationships: dict[str, float] = field(default_factory=dict)
    morale: float = 0.0


@dataclass
class Actor:
    """An agent in the scenario.

    Attributes:
        id: Unique identifier.
        name: Human-readable name.
        behavior_model: How the actor tends to behave.
        current_state: Actor's current state.
        objectives: What the actor is trying to achieve.
        decision_history: Past decisions made by this actor.
    """
    id: str
    name: str
    behavior_model: BehaviorModel
    current_state: ActorState = field(default_factory=ActorState)
    objectives: list[str] = field(default_factory=list)
    decision_history: list[str] = field(default_factory=list)


@dataclass
class Decision:
    """A choice point in the timeline.

    Attributes:
        id: Unique identifier.
        date: When the decision must be made.
        description: What's being decided.
        options: Available choices.
        chosen: The selected option (if decided).
        rationale: Why this choice was made.
        consequences: Predicted effects of this decision.
        actor_id: Which actor makes this decision.
        status: Current status of the decision.
    """
    id: str
    date: datetime
    description: str
    options: list[str] = field(default_factory=list)
    chosen: str | None = None
    rationale: str | None = None
    consequences: dict[str, Any] = field(default_factory=dict)
    actor_id: str | None = None
    status: DecisionStatus = DecisionStatus.PENDING


@dataclass
class TimelineMetrics:
    """Metrics for evaluating a timeline.

    Attributes:
        success_score: Overall success (0.0 to 1.0).
        risk_score: Overall risk level (0.0 to 1.0).
        opportunity_score: Growth potential (0.0 to 1.0).
        stability_score: How stable the state is (0.0 to 1.0).
        custom_metrics: User-defined metrics.
    """
    success_score: float = 0.5
    risk_score: float = 0.5
    opportunity_score: float = 0.5
    stability_score: float = 0.5
    custom_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TimelineState:
    """State at a point in time.

    Attributes:
        date: Current date in the timeline.
        events: Events that have occurred up to this point.
        actor_states: Current state of each actor.
        metrics: Evaluation metrics.
        context: Additional state context.
        parent_state_id: ID of the state this branched from.
    """
    date: datetime
    events: list[Event] = field(default_factory=list)
    actor_states: dict[str, ActorState] = field(default_factory=dict)
    metrics: TimelineMetrics = field(default_factory=TimelineMetrics)
    context: dict[str, Any] = field(default_factory=dict)
    parent_state_id: str | None = None


@dataclass
class Timeline:
    """Sequence of states representing a simulated future.

    Attributes:
        id: Unique identifier.
        initial_state: Starting state.
        events: All events in this timeline.
        decisions: All decisions made.
        final_state: Ending state.
        plausibility: How realistic this timeline is (0.0 to 1.0).
        probability: Likelihood of this timeline occurring.
        branch_point: Decision that created this branch (if any).
    """
    id: str
    initial_state: TimelineState
    events: list[Event] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    final_state: TimelineState | None = None
    plausibility: float = 1.0
    probability: float = 1.0
    branch_point: str | None = None


@dataclass
class RobustDecision:
    """A decision that performs well across scenarios.

    Attributes:
        decision: The decision being analyzed.
        avg_outcome: Average outcome across timelines.
        min_outcome: Worst-case outcome.
        max_outcome: Best-case outcome.
        variance: How much outcomes vary.
        robustness_score: How robust this decision is (0.0 to 1.0).
    """
    decision: Decision
    avg_outcome: float = 0.0
    min_outcome: float = 0.0
    max_outcome: float = 0.0
    variance: float = 0.0
    robustness_score: float = 0.0


@dataclass
class BrittleDecision:
    """A decision that only works in specific scenarios.

    Attributes:
        decision: The decision being analyzed.
        success_scenarios: Timelines where this works.
        failure_scenarios: Timelines where this fails.
        critical_assumptions: What must be true for success.
        brittleness_score: How brittle this decision is (0.0 to 1.0).
    """
    decision: Decision
    success_scenarios: list[str] = field(default_factory=list)
    failure_scenarios: list[str] = field(default_factory=list)
    critical_assumptions: list[str] = field(default_factory=list)
    brittleness_score: float = 0.0


@dataclass
class KeyUncertainty:
    """An uncertainty that significantly affects outcomes.

    Attributes:
        description: What is uncertain.
        impact_range: Range of potential impacts.
        affected_timelines: Which timelines are affected.
        resolution_date: When this uncertainty resolves.
        controllable: Whether we can influence this.
    """
    description: str
    impact_range: tuple[float, float] = (0.0, 1.0)
    affected_timelines: list[str] = field(default_factory=list)
    resolution_date: datetime | None = None
    controllable: bool = False


@dataclass
class ScenarioComparison:
    """Analysis across timelines.

    Attributes:
        timelines: All timelines being compared.
        best_case: Timeline with best outcome.
        worst_case: Timeline with worst outcome.
        most_likely: Timeline with highest probability.
        robust_decisions: Decisions that work across scenarios.
        brittle_decisions: Decisions that only work in some scenarios.
        key_uncertainties: Uncertainties that matter most.
        expected_value: Probability-weighted outcome.
        recommendation: Suggested course of action.
    """
    timelines: list[Timeline] = field(default_factory=list)
    best_case: Timeline | None = None
    worst_case: Timeline | None = None
    most_likely: Timeline | None = None
    robust_decisions: list[RobustDecision] = field(default_factory=list)
    brittle_decisions: list[BrittleDecision] = field(default_factory=list)
    key_uncertainties: list[KeyUncertainty] = field(default_factory=list)
    expected_value: float = 0.0
    recommendation: str = ""


@dataclass
class CausalNode:
    """A node in the causal graph.

    Attributes:
        id: Unique identifier (event or decision id).
        node_type: 'event' or 'decision'.
        description: What this node represents.
        causes: IDs of nodes that caused this.
        effects: IDs of nodes caused by this.
        strength: Impact strength (0.0 to 1.0).
    """
    id: str
    node_type: str
    description: str
    causes: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    strength: float = 1.0


@dataclass
class LeveragePoint:
    """A decision point with outsized impact.

    Attributes:
        node_id: ID of the decision or event.
        impact_score: How much this affects outcomes.
        downstream_count: Number of effects downstream.
        reversibility: How reversible this decision is (0.0 to 1.0).
        timing_sensitivity: How sensitive to timing.
    """
    node_id: str
    impact_score: float = 0.0
    downstream_count: int = 0
    reversibility: float = 0.5
    timing_sensitivity: float = 0.5


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation.

    Attributes:
        n_runs: Number of simulation runs.
        metric_name: Name of the metric analyzed.
        mean: Mean value across runs.
        std_dev: Standard deviation.
        percentiles: Key percentiles (5th, 25th, 50th, 75th, 95th).
        distribution: Full distribution of values.
        confidence_interval: 95% confidence interval.
    """
    n_runs: int
    metric_name: str
    mean: float = 0.0
    std_dev: float = 0.0
    percentiles: dict[int, float] = field(default_factory=dict)
    distribution: list[float] = field(default_factory=list)
    confidence_interval: tuple[float, float] = (0.0, 0.0)


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis.

    Attributes:
        input_name: Name of the input varied.
        sensitivity_score: How sensitive output is to this input.
        correlation: Correlation between input and output.
        elasticity: Percent change in output per percent change in input.
        breakpoints: Values where behavior changes significantly.
    """
    input_name: str
    sensitivity_score: float = 0.0
    correlation: float = 0.0
    elasticity: float = 0.0
    breakpoints: list[float] = field(default_factory=list)
