"""Temporal Scenario Planner.

A temporal simulation framework for exploring different futures through
branching timelines, actor modeling, and scenario synthesis.

Example usage:

    from datetime import datetime, timedelta
    from temporal import (
        TemporalHarness,
        HarnessConfig,
        create_initial_state,
        Decision,
        create_actor,
        BehaviorModel,
    )

    # Create initial state
    initial_state = create_initial_state(
        date=datetime(2024, 1, 1),
        context={"market_size": 1000000000},
    )

    # Create initial decision
    decision = Decision(
        id="strategy_choice",
        date=datetime(2024, 1, 1),
        description="Choose market entry strategy",
        options=["aggressive expansion", "steady growth", "wait and see"],
    )

    # Create harness and run
    harness = TemporalHarness(
        initial_state=initial_state,
        initial_decision=decision,
    )
    comparison = harness.run()

    print(comparison.recommendation)
"""

# Types
from .types import (
    Actor,
    ActorState,
    BehaviorModel,
    BrittleDecision,
    CausalNode,
    Decision,
    DecisionStatus,
    Event,
    EventCategory,
    KeyUncertainty,
    LeveragePoint,
    MonteCarloResult,
    RobustDecision,
    ScenarioComparison,
    SensitivityResult,
    Timeline,
    TimelineMetrics,
    TimelineState,
)

# State management
from .state import (
    advance_time,
    apply_event,
    checkpoint_state,
    create_initial_state,
    diff_states,
    merge_states,
    state_to_summary,
)

# Actor modeling
from .actors import (
    BaseActor,
    CompetitorActor,
    MarketActor,
    RegulatorActor,
    Situation,
    UserActor,
    create_actor,
    simulate_actor_interaction,
)

# Events
from .events import (
    black_swan_injection,
    chain_events,
    filter_events_by_probability,
    generate_events,
    summarize_events,
)

# Simulation
from .simulation import (
    branch_at_decision,
    continue_timeline,
    plausibility_check,
    simulate_step,
    simulate_timeline,
    terminate_implausible,
    timeline_summary,
)

# Causality
from .causality import (
    CausalGraph,
    build_causal_graph,
    counterfactual_analysis,
    critical_path,
    find_common_causes,
    find_divergence_point,
    find_leverage_points,
    merge_graphs,
    trace_cause,
    trace_effects,
    visualize_graph,
)

# Synthesis
from .synthesis import (
    compare_timelines,
    divergence_analysis,
    expected_value,
    find_brittle_decisions,
    find_robust_decisions,
    identify_key_uncertainties,
    scenario_matrix,
    synthesis_summary,
)

# Harness
from .harness import (
    CatastropheWarning,
    HarnessConfig,
    TemporalHarness,
    create_scenario_with_actors,
    quick_scenario_analysis,
)

# Monte Carlo
from .montecarlo import (
    conditional_value_at_risk,
    confidence_intervals,
    monte_carlo_simulate,
    monte_carlo_summary,
    multi_metric_analysis,
    probability_distribution,
    scenario_probability,
    sensitivity_analysis,
    value_at_risk,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Actor",
    "ActorState",
    "BehaviorModel",
    "BrittleDecision",
    "CausalNode",
    "Decision",
    "DecisionStatus",
    "Event",
    "EventCategory",
    "KeyUncertainty",
    "LeveragePoint",
    "MonteCarloResult",
    "RobustDecision",
    "ScenarioComparison",
    "SensitivityResult",
    "Timeline",
    "TimelineMetrics",
    "TimelineState",
    # State management
    "advance_time",
    "apply_event",
    "checkpoint_state",
    "create_initial_state",
    "diff_states",
    "merge_states",
    "state_to_summary",
    # Actor modeling
    "BaseActor",
    "CompetitorActor",
    "MarketActor",
    "RegulatorActor",
    "Situation",
    "UserActor",
    "create_actor",
    "simulate_actor_interaction",
    # Events
    "black_swan_injection",
    "chain_events",
    "filter_events_by_probability",
    "generate_events",
    "summarize_events",
    # Simulation
    "branch_at_decision",
    "continue_timeline",
    "plausibility_check",
    "simulate_step",
    "simulate_timeline",
    "terminate_implausible",
    "timeline_summary",
    # Causality
    "CausalGraph",
    "build_causal_graph",
    "counterfactual_analysis",
    "critical_path",
    "find_common_causes",
    "find_divergence_point",
    "find_leverage_points",
    "merge_graphs",
    "trace_cause",
    "trace_effects",
    "visualize_graph",
    # Synthesis
    "compare_timelines",
    "divergence_analysis",
    "expected_value",
    "find_brittle_decisions",
    "find_robust_decisions",
    "identify_key_uncertainties",
    "scenario_matrix",
    "synthesis_summary",
    # Harness
    "CatastropheWarning",
    "HarnessConfig",
    "TemporalHarness",
    "create_scenario_with_actors",
    "quick_scenario_analysis",
    # Monte Carlo
    "conditional_value_at_risk",
    "confidence_intervals",
    "monte_carlo_simulate",
    "monte_carlo_summary",
    "multi_metric_analysis",
    "probability_distribution",
    "scenario_probability",
    "sensitivity_analysis",
    "value_at_risk",
]
