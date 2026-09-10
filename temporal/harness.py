"""Main temporal harness for the Temporal Scenario Planner.

This module provides the TemporalHarness class that orchestrates
timeline simulation, branching, and synthesis.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from .actors import BaseActor, create_actor
from .causality import build_causal_graph, find_leverage_points, CausalGraph
from .events import black_swan_injection
from .simulation import (
    branch_at_decision,
    plausibility_check,
    simulate_timeline,
    terminate_implausible,
    timeline_summary,
)
from .state import checkpoint_state, create_initial_state, state_to_summary
from .synthesis import (
    compare_timelines,
    synthesis_summary,
    ScenarioComparison,
)
from .types import (
    Actor,
    ActorState,
    BehaviorModel,
    Decision,
    DecisionStatus,
    Event,
    Timeline,
    TimelineState,
)


@dataclass
class HarnessConfig:
    """Configuration for the temporal harness.

    Attributes:
        max_timelines: Maximum number of timelines to simulate.
        simulation_horizon: How far to simulate into the future.
        plausibility_threshold: Minimum plausibility for timeline survival.
        branch_on_all_decisions: Branch at every decision vs. only critical ones.
        black_swan_probability: Chance of black swan event injection.
        step_size: Size of simulation steps.
        parallel_execution: Whether to run simulations in parallel.
    """
    max_timelines: int = 10
    simulation_horizon: timedelta = timedelta(days=365)
    plausibility_threshold: float = 0.1
    branch_on_all_decisions: bool = False
    black_swan_probability: float = 0.05
    step_size: timedelta = timedelta(days=7)
    parallel_execution: bool = False


@dataclass
class CatastropheWarning:
    """Warning about potential catastrophic outcome.

    Attributes:
        timeline_id: Timeline where catastrophe occurs.
        description: What goes wrong.
        trigger_decision: Decision that leads to catastrophe.
        severity: How bad is it (0.0 to 1.0).
        preventable: Can this be avoided?
    """
    timeline_id: str
    description: str
    trigger_decision: str | None = None
    severity: float = 0.5
    preventable: bool = True


class TemporalHarness:
    """Main harness for temporal scenario planning.

    The harness manages the full lifecycle of timeline simulation:
    1. Initialize with starting state and initial decision
    2. Branch at decision points to explore alternatives
    3. Simulate each timeline forward
    4. Apply gate checks for plausibility
    5. Propagate catastrophe warnings backward
    6. Synthesize comparison across scenarios
    """

    def __init__(
        self,
        initial_state: TimelineState,
        initial_decision: Decision | None = None,
        actors: list[BaseActor] | None = None,
        config: HarnessConfig | None = None,
        llm_provider: Callable[[str], str] | None = None,
    ):
        """Initialize the temporal harness.

        Args:
            initial_state: Starting state for all timelines.
            initial_decision: First decision to branch on.
            actors: Actors participating in the simulation.
            config: Harness configuration.
            llm_provider: Optional LLM for event generation and decisions.
        """
        self.initial_state = initial_state
        self.initial_decision = initial_decision
        self.actors = actors or []
        self.config = config or HarnessConfig()
        self.llm_provider = llm_provider

        self.timelines: list[Timeline] = []
        self.terminated_timelines: list[Timeline] = []
        self.catastrophe_warnings: list[CatastropheWarning] = []
        self.causal_graphs: dict[str, CausalGraph] = {}

        self._run_id = f"run_{uuid.uuid4().hex[:8]}"

    def run(self) -> ScenarioComparison:
        """Execute the full temporal simulation.

        Returns:
            ScenarioComparison with analysis results.
        """
        # Initialize timelines
        self._initialize_timelines()

        # Simulate each timeline
        self._simulate_all()

        # Gate checks
        self._apply_gate_checks()

        # Build causal graphs
        self._build_causal_graphs()

        # Check for catastrophes
        self._detect_catastrophes()

        # Synthesize results
        return self._synthesize()

    def _initialize_timelines(self) -> None:
        """Initialize timelines from initial decision."""
        if self.initial_decision and self.initial_decision.options:
            # Branch on initial decision
            base_timeline = Timeline(
                id=f"timeline_{self._run_id}_base",
                initial_state=self.initial_state,
            )

            self.timelines = branch_at_decision(
                base_timeline,
                self.initial_decision,
            )
        else:
            # Single timeline
            self.timelines = [Timeline(
                id=f"timeline_{self._run_id}_0",
                initial_state=self.initial_state,
            )]

    def _simulate_all(self) -> None:
        """Simulate all timelines."""
        simulated: list[Timeline] = []

        for timeline in self.timelines:
            if len(simulated) >= self.config.max_timelines:
                break

            # Simulate this timeline
            simulated_timeline = simulate_timeline(
                initial_state=timeline.initial_state,
                decisions=timeline.decisions,
                horizon=self.config.simulation_horizon,
                actors=self.actors,
                step_size=self.config.step_size,
                llm_provider=self.llm_provider,
            )

            # Copy over branch point
            simulated_timeline.branch_point = timeline.branch_point

            # Maybe inject black swan
            if (
                self.config.black_swan_probability > 0
                and uuid.uuid4().int % 100 < self.config.black_swan_probability * 100
            ):
                from .state import apply_event
                swan = black_swan_injection(simulated_timeline.final_state)  # type: ignore
                simulated_timeline.final_state = apply_event(
                    simulated_timeline.final_state, swan  # type: ignore
                )
                simulated_timeline.events.append(swan)

            simulated.append(simulated_timeline)

            # Branch at decision points within simulation
            if self.config.branch_on_all_decisions:
                for decision in simulated_timeline.decisions:
                    if len(simulated) >= self.config.max_timelines:
                        break
                    if decision.status == DecisionStatus.CHOSEN:
                        # Find unexamined options
                        other_options = [
                            opt for opt in decision.options
                            if opt != decision.chosen
                        ]
                        if other_options:
                            new_branches = branch_at_decision(
                                simulated_timeline,
                                decision,
                                other_options,
                            )
                            simulated.extend(new_branches[:2])  # Limit branching

        self.timelines = simulated

    def _apply_gate_checks(self) -> None:
        """Apply plausibility gate checks."""
        viable, terminated = terminate_implausible(
            self.timelines,
            self.config.plausibility_threshold,
        )

        self.timelines = viable
        self.terminated_timelines.extend(terminated)

    def _build_causal_graphs(self) -> None:
        """Build causal graphs for all timelines."""
        for timeline in self.timelines:
            self.causal_graphs[timeline.id] = build_causal_graph(timeline)

    def _detect_catastrophes(self) -> None:
        """Detect catastrophic outcomes and create warnings."""
        for timeline in self.timelines + self.terminated_timelines:
            if not timeline.final_state:
                continue

            metrics = timeline.final_state.metrics

            # Check for catastrophic outcomes
            if metrics.success_score < 0.1:
                self._add_catastrophe_warning(
                    timeline,
                    "Critical failure: success score below 10%",
                    severity=0.9,
                )

            if metrics.risk_score > 0.9:
                self._add_catastrophe_warning(
                    timeline,
                    "Extreme risk: risk score above 90%",
                    severity=0.8,
                )

            if metrics.stability_score < 0.1:
                self._add_catastrophe_warning(
                    timeline,
                    "Collapse: stability score below 10%",
                    severity=0.95,
                )

            # Check for catastrophic events
            for event in timeline.events:
                if event.magnitude <= -7:
                    self._add_catastrophe_warning(
                        timeline,
                        f"Severe negative event: {event.description}",
                        severity=abs(event.magnitude) / 10,
                    )

    def _add_catastrophe_warning(
        self,
        timeline: Timeline,
        description: str,
        severity: float,
    ) -> None:
        """Add a catastrophe warning with cause tracing."""
        trigger = None

        # Try to find triggering decision
        if timeline.id in self.causal_graphs:
            graph = self.causal_graphs[timeline.id]
            leverage = find_leverage_points(graph)
            if leverage:
                trigger = leverage[0].node_id

        self.catastrophe_warnings.append(CatastropheWarning(
            timeline_id=timeline.id,
            description=description,
            trigger_decision=trigger,
            severity=severity,
            preventable=trigger is not None,
        ))

    def _synthesize(self) -> ScenarioComparison:
        """Synthesize results across timelines."""
        return compare_timelines(self.timelines)

    def add_decision(self, decision: Decision) -> None:
        """Add a decision to be evaluated.

        Args:
            decision: Decision to add.
        """
        for timeline in self.timelines:
            timeline.decisions.append(decision)

    def add_actor(self, actor: BaseActor) -> None:
        """Add an actor to the simulation.

        Args:
            actor: Actor to add.
        """
        self.actors.append(actor)

    def get_timeline(self, timeline_id: str) -> Timeline | None:
        """Get a specific timeline by ID.

        Args:
            timeline_id: ID of timeline to retrieve.

        Returns:
            Timeline or None if not found.
        """
        for timeline in self.timelines + self.terminated_timelines:
            if timeline.id == timeline_id:
                return timeline
        return None

    def get_warnings_for_decision(self, decision_id: str) -> list[CatastropheWarning]:
        """Get warnings related to a specific decision.

        Args:
            decision_id: Decision ID to check.

        Returns:
            List of related warnings.
        """
        return [
            w for w in self.catastrophe_warnings
            if w.trigger_decision == decision_id
        ]

    def summary(self) -> str:
        """Get a summary of the harness state.

        Returns:
            Human-readable summary.
        """
        lines = [
            f"Temporal Harness Run: {self._run_id}",
            "=" * 50,
            f"Active timelines: {len(self.timelines)}",
            f"Terminated timelines: {len(self.terminated_timelines)}",
            f"Catastrophe warnings: {len(self.catastrophe_warnings)}",
            f"Actors: {len(self.actors)}",
        ]

        if self.timelines:
            lines.append("\nTimeline summaries:")
            for tl in self.timelines[:3]:
                lines.append(f"\n{timeline_summary(tl)}")

        if self.catastrophe_warnings:
            lines.append("\nCatastrophe warnings:")
            for w in self.catastrophe_warnings[:3]:
                lines.append(
                    f"  - [{w.severity:.0%}] {w.description} "
                    f"(timeline: {w.timeline_id})"
                )

        return "\n".join(lines)


def quick_scenario_analysis(
    starting_date: datetime,
    initial_context: dict[str, Any],
    decision_description: str,
    decision_options: list[str],
    horizon_days: int = 365,
    num_timelines: int = 5,
    llm_provider: Callable[[str], str] | None = None,
) -> ScenarioComparison:
    """Quick helper for running a scenario analysis.

    Args:
        starting_date: When to start simulation.
        initial_context: Initial state context.
        decision_description: What decision is being made.
        decision_options: Available options.
        horizon_days: How many days to simulate.
        num_timelines: Maximum timelines to generate.
        llm_provider: Optional LLM.

    Returns:
        ScenarioComparison results.
    """
    # Create initial state
    initial_state = create_initial_state(
        date=starting_date,
        context=initial_context,
    )

    # Create initial decision
    initial_decision = Decision(
        id="initial_decision",
        date=starting_date,
        description=decision_description,
        options=decision_options,
    )

    # Create and run harness
    config = HarnessConfig(
        max_timelines=num_timelines,
        simulation_horizon=timedelta(days=horizon_days),
    )

    harness = TemporalHarness(
        initial_state=initial_state,
        initial_decision=initial_decision,
        config=config,
        llm_provider=llm_provider,
    )

    return harness.run()


def create_scenario_with_actors(
    starting_date: datetime,
    player_objectives: list[str],
    competitor_configs: list[dict[str, Any]],
    market_volatility: float = 0.3,
    decision: Decision | None = None,
    llm_provider: Callable[[str], str] | None = None,
) -> TemporalHarness:
    """Create a harness with pre-configured actors.

    Args:
        starting_date: When to start.
        player_objectives: What the player is trying to achieve.
        competitor_configs: List of competitor configurations.
        market_volatility: Market volatility level.
        decision: Initial decision to branch on.
        llm_provider: Optional LLM.

    Returns:
        Configured TemporalHarness ready to run.
    """
    # Create initial state
    initial_state = create_initial_state(
        date=starting_date,
        actors={
            "player": ActorState(
                resources={"capital": 1000000, "headcount": 50},
                morale=0.7,
            ),
        },
        context={
            "market_size": 1000000000,
            "player_market_share": 0.1,
        },
    )

    # Create actors
    actors: list[BaseActor] = []

    # Add market actor
    market = create_actor(
        actor_type="market",
        actor_id="market",
        name="Market",
        behavior=BehaviorModel.REACTIVE,
        volatility=market_volatility,
        llm_provider=llm_provider,
    )
    actors.append(market)

    # Add competitors
    for i, config in enumerate(competitor_configs):
        competitor = create_actor(
            actor_type="competitor",
            actor_id=f"competitor_{i}",
            name=config.get("name", f"Competitor {i}"),
            behavior=BehaviorModel(config.get("behavior", "reactive")),
            market_share=config.get("market_share", 0.2),
            aggressiveness=config.get("aggressiveness", 0.5),
            objectives=config.get("objectives", ["Maximize market share"]),
            llm_provider=llm_provider,
        )
        actors.append(competitor)

    # Create harness
    return TemporalHarness(
        initial_state=initial_state,
        initial_decision=decision,
        actors=actors,
        llm_provider=llm_provider,
    )
