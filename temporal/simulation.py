"""Timeline simulation for the Temporal Scenario Planner.

This module handles the step-by-step simulation of timelines,
branching at decision points, and plausibility checking.
"""

import uuid
import math
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Callable

from .actors import BaseActor, Situation
from .events import (
    chain_events,
    filter_events_by_probability,
    generate_events,
)
from .state import (
    advance_time,
    apply_event,
    checkpoint_state,
    state_to_summary,
)
from .types import (
    Decision,
    DecisionStatus,
    Event,
    Timeline,
    TimelineMetrics,
    TimelineState,
)


def generate_timeline_id() -> str:
    """Generate a unique timeline ID."""
    return f"timeline_{uuid.uuid4().hex[:8]}"


def simulate_timeline(
    initial_state: TimelineState,
    decisions: list[Decision],
    horizon: timedelta,
    actors: list[BaseActor] | None = None,
    step_size: timedelta | None = None,
    llm_provider: Callable[[str], str] | None = None,
) -> Timeline:
    """Run a timeline forward from initial state.

    Args:
        initial_state: Starting state.
        decisions: Decisions to make at their respective dates.
        horizon: How far into the future to simulate.
        actors: Actors participating in the simulation.
        step_size: Size of each simulation step (default 7 days).
        llm_provider: Optional LLM for event generation.

    Returns:
        The simulated timeline.
    """
    step_size = step_size or timedelta(days=7)
    actors = actors or []

    timeline = Timeline(
        id=generate_timeline_id(),
        initial_state=deepcopy(initial_state),
        events=[],
        decisions=[],
        probability=1.0,
    )

    current_state = deepcopy(initial_state)
    end_date = initial_state.date + horizon
    pending_decisions = list(decisions)

    while current_state.date < end_date:
        # Check for decisions at this step
        due_decisions = [d for d in pending_decisions if d.date <= current_state.date]
        for original_decision in due_decisions:
            current_state, executed_decision = _execute_decision(
                current_state, original_decision, actors, llm_provider
            )
            timeline.decisions.append(executed_decision)
            pending_decisions.remove(original_decision)

        # Generate and apply events
        events = generate_events(
            current_state,
            num_events=2,
            llm_provider=llm_provider,
        )

        # Filter by probability
        occurred_events = filter_events_by_probability(events, sample=True)

        for event in occurred_events:
            current_state = apply_event(current_state, event)
            timeline.events.append(event)

            # Chain reactions
            chained = chain_events(event, current_state, llm_provider=llm_provider)
            occurred_chained = filter_events_by_probability(chained, sample=True)
            for chained_event in occurred_chained:
                current_state = apply_event(current_state, chained_event)
                timeline.events.append(chained_event)

        # Advance time
        current_state = advance_time(current_state, step_size)

        # Update timeline probability based on events
        timeline.probability *= _calculate_step_probability(occurred_events)

    timeline.final_state = current_state
    timeline.plausibility = _calculate_plausibility(timeline)

    return timeline


def _execute_decision(
    state: TimelineState,
    decision: Decision,
    actors: list[BaseActor],
    llm_provider: Callable[[str], str] | None = None,
) -> tuple[TimelineState, Decision]:
    """Execute a decision and return updated state.

    Args:
        state: Current state.
        decision: Decision to execute.
        actors: Available actors.
        llm_provider: Optional LLM.

    Returns:
        Tuple of (new_state, updated_decision).
    """
    decision = deepcopy(decision)

    # Find the actor making this decision
    actor = None
    if decision.actor_id:
        actor = next((a for a in actors if a.id == decision.actor_id), None)

    if decision.status == DecisionStatus.CHOSEN and decision.chosen in decision.options:
        # A branch fixes this choice; simulation must not choose it again.
        pass
    elif actor and decision.options:
        # Actor makes the decision
        situation = Situation(
            state=state,
            decision=decision,
            other_actors={a.id: a.state for a in actors if a.id != actor.id},
            recent_events=state.events[-5:],
        )
        chosen = actor.decide(situation)
        decision.chosen = chosen
        decision.rationale = f"{actor.name} chose based on {actor.behavior_model.value} strategy"
    elif decision.options and llm_provider:
        # LLM makes the decision
        chosen = _llm_decide(state, decision, llm_provider)
        decision.chosen = chosen
    elif decision.options:
        # Default to first option
        decision.chosen = decision.options[0]

    decision.status = DecisionStatus.CHOSEN

    # Apply decision consequences
    new_state = deepcopy(state)
    for key, value in decision.consequences.items():
        if key.startswith("metric:"):
            metric_name = key.split(":", 1)[1]
            if hasattr(new_state.metrics, metric_name):
                current = getattr(new_state.metrics, metric_name)
                setattr(new_state.metrics, metric_name, max(0.0, min(1.0, current + value)))
        else:
            new_state.context[key] = value

    return new_state, decision


def _llm_decide(
    state: TimelineState,
    decision: Decision,
    llm_provider: Callable[[str], str],
) -> str:
    """Use LLM to make a decision."""
    prompt = f"""Timeline simulation decision point.

Current state:
{state_to_summary(state)}

Decision: {decision.description}
Options: {', '.join(decision.options)}

Choose the most reasonable option given the current state.
Respond with just the option text."""

    try:
        response = llm_provider(prompt).strip()
        for opt in decision.options:
            if opt.lower() in response.lower() or response.lower() in opt.lower():
                return opt
    except Exception:
        pass

    return decision.options[0] if decision.options else ""


def _calculate_step_probability(events: list[Event]) -> float:
    """Calculate probability contribution from a step's events."""
    if not events:
        return 1.0

    # Product of event probabilities
    prob = 1.0
    for event in events:
        prob *= event.probability

    return prob


def _calculate_plausibility(timeline: Timeline) -> float:
    """Calculate overall plausibility of a timeline.

    Considers:
    - Event probability chain
    - Internal consistency
    - Metric reasonableness
    """
    # Joint event likelihood shrinks exponentially with horizon, even when every
    # event is ordinary. Keep it in probability, but use a length-normalized
    # event score for the plausibility gate. This is a heuristic, not a calibrated
    # probability that the scenario is true. Log-space avoids long-run underflow.
    if timeline.events:
        probabilities = [max(0.0, min(1.0, e.probability)) for e in timeline.events]
        base_plausibility = (
            math.exp(sum(math.log(p) for p in probabilities) / len(probabilities))
            if all(p > 0 for p in probabilities) else 0.0
        )
    else:
        base_plausibility = timeline.probability

    # Check for metric extremes
    if timeline.final_state:
        metrics = timeline.final_state.metrics
        extreme_count = sum([
            metrics.success_score < 0.1 or metrics.success_score > 0.9,
            metrics.risk_score < 0.1 or metrics.risk_score > 0.9,
            metrics.stability_score < 0.1,
        ])
        # Penalize extreme outcomes
        base_plausibility *= 0.9 ** extreme_count

    # Check for too many events
    expected_events = len(timeline.decisions) * 2 + 5
    if len(timeline.events) > expected_events * 2:
        base_plausibility *= 0.8

    return max(0.0, min(1.0, base_plausibility))


def branch_at_decision(
    timeline: Timeline,
    decision: Decision,
    options: list[str] | None = None,
) -> list[Timeline]:
    """Create alternate timelines for each decision option.

    Args:
        timeline: Timeline to branch from.
        decision: Decision point.
        options: Override options (uses decision.options if None).

    Returns:
        List of branched timelines, one per option.
    """
    options = options or decision.options
    if not options:
        return [timeline]

    # Find state just before decision
    branch_state = _get_state_at_decision(timeline, decision)
    if not branch_state:
        branch_state = timeline.initial_state

    branches: list[Timeline] = []
    for option in options:
        # Create new timeline
        new_timeline = Timeline(
            id=generate_timeline_id(),
            initial_state=deepcopy(branch_state),
            events=[],
            decisions=[],
            branch_point=decision.id,
        )

        # Create decision with this option chosen
        new_decision = deepcopy(decision)
        new_decision.chosen = option
        new_decision.status = DecisionStatus.CHOSEN
        new_timeline.decisions.append(new_decision)

        branches.append(new_timeline)

    return branches


def _get_state_at_decision(timeline: Timeline, decision: Decision) -> TimelineState | None:
    """Get the state at a specific decision point."""
    # Reconstruct state up to decision date
    state = deepcopy(timeline.initial_state)

    for event in timeline.events:
        if event.date > decision.date:
            break
        state = apply_event(state, event)

    return state


def plausibility_check(
    timeline: Timeline,
    min_plausibility: float = 0.1,
) -> tuple[bool, list[str]]:
    """Check if a timeline is plausible.

    Args:
        timeline: Timeline to check.
        min_plausibility: Minimum plausibility threshold.

    Returns:
        Tuple of (is_plausible, list_of_issues).
    """
    issues: list[str] = []

    # Check overall plausibility
    if timeline.plausibility < min_plausibility:
        issues.append(f"Overall plausibility too low: {timeline.plausibility:.2f}")

    # Check for metric inconsistencies
    if timeline.final_state:
        m = timeline.final_state.metrics

        # Very high success with very high risk is suspicious
        if m.success_score > 0.8 and m.risk_score > 0.8:
            issues.append("Inconsistent: high success with high risk")

        # Very low stability should impact success
        if m.stability_score < 0.2 and m.success_score > 0.7:
            issues.append("Inconsistent: high success with very low stability")

    # Check for event contradictions
    contradiction = _check_event_contradictions(timeline.events)
    if contradiction:
        issues.append(f"Contradictory events: {contradiction}")

    is_plausible = len(issues) == 0
    return is_plausible, issues


def _check_event_contradictions(events: list[Event]) -> str | None:
    """Check for contradictory events."""
    # Simple check: same-day events with opposite magnitudes
    by_date: dict[str, list[Event]] = {}
    for event in events:
        date_key = event.date.strftime("%Y-%m-%d")
        if date_key not in by_date:
            by_date[date_key] = []
        by_date[date_key].append(event)

    for date_key, day_events in by_date.items():
        if len(day_events) < 2:
            continue

        # Check for opposite effects
        magnitudes = [e.magnitude for e in day_events]
        if max(magnitudes) > 5 and min(magnitudes) < -5:
            return f"Extreme opposite events on {date_key}"

    return None


def terminate_implausible(
    timelines: list[Timeline],
    threshold: float = 0.1,
) -> tuple[list[Timeline], list[Timeline]]:
    """Gate check to terminate implausible timelines.

    Args:
        timelines: Timelines to check.
        threshold: Minimum plausibility.

    Returns:
        Tuple of (viable_timelines, terminated_timelines).
    """
    viable: list[Timeline] = []
    terminated: list[Timeline] = []

    for timeline in timelines:
        is_plausible, _ = plausibility_check(timeline, threshold)
        if is_plausible:
            viable.append(timeline)
        else:
            terminated.append(timeline)

    return viable, terminated


def continue_timeline(
    timeline: Timeline,
    additional_horizon: timedelta,
    additional_decisions: list[Decision] | None = None,
    actors: list[BaseActor] | None = None,
    llm_provider: Callable[[str], str] | None = None,
) -> Timeline:
    """Continue simulating an existing timeline.

    Args:
        timeline: Timeline to continue.
        additional_horizon: How much further to simulate.
        additional_decisions: New decisions to include.
        actors: Actors for the simulation.
        llm_provider: Optional LLM.

    Returns:
        Extended timeline.
    """
    if not timeline.final_state:
        raise ValueError("Cannot continue timeline without final_state")

    # Simulate from final state
    continuation = simulate_timeline(
        initial_state=timeline.final_state,
        decisions=additional_decisions or [],
        horizon=additional_horizon,
        actors=actors,
        llm_provider=llm_provider,
    )

    # Merge into original timeline
    extended = deepcopy(timeline)
    extended.events.extend(continuation.events)
    extended.decisions.extend(continuation.decisions)
    extended.final_state = continuation.final_state
    extended.plausibility = (timeline.plausibility + continuation.plausibility) / 2

    return extended


def simulate_step(
    state: TimelineState,
    step_size: timedelta,
    actors: list[BaseActor] | None = None,
    pending_decisions: list[Decision] | None = None,
    llm_provider: Callable[[str], str] | None = None,
) -> tuple[TimelineState, list[Event], list[Decision]]:
    """Simulate a single step of the timeline.

    Args:
        state: Current state.
        step_size: Duration of this step.
        actors: Available actors.
        pending_decisions: Decisions that might be due.
        llm_provider: Optional LLM.

    Returns:
        Tuple of (new_state, events_occurred, decisions_made).
    """
    actors = actors or []
    pending_decisions = pending_decisions or []

    current_state = deepcopy(state)
    events_occurred: list[Event] = []
    decisions_made: list[Decision] = []

    # Handle due decisions
    due_decisions = [d for d in pending_decisions if d.date <= current_state.date]
    for decision in due_decisions:
        current_state, executed = _execute_decision(
            current_state, decision, actors, llm_provider
        )
        decisions_made.append(executed)

    # Generate and apply events
    events = generate_events(current_state, num_events=2, llm_provider=llm_provider)
    occurred = filter_events_by_probability(events, sample=True)

    for event in occurred:
        current_state = apply_event(current_state, event)
        events_occurred.append(event)

    # Advance time
    current_state = advance_time(current_state, step_size)

    return current_state, events_occurred, decisions_made


def timeline_summary(timeline: Timeline) -> str:
    """Create a human-readable summary of a timeline.

    Args:
        timeline: Timeline to summarize.

    Returns:
        Summary string.
    """
    lines = [
        f"Timeline {timeline.id}",
        f"Plausibility: {timeline.plausibility:.2f}",
        f"Probability: {timeline.probability:.2f}",
        f"Events: {len(timeline.events)}",
        f"Decisions: {len(timeline.decisions)}",
    ]

    if timeline.branch_point:
        lines.append(f"Branched from: {timeline.branch_point}")

    if timeline.decisions:
        lines.append("\nKey decisions:")
        for d in timeline.decisions[:3]:
            lines.append(f"  - {d.description}: {d.chosen}")

    if timeline.final_state:
        lines.append("\nFinal state:")
        lines.append(f"  Success: {timeline.final_state.metrics.success_score:.2f}")
        lines.append(f"  Risk: {timeline.final_state.metrics.risk_score:.2f}")
        lines.append(f"  Stability: {timeline.final_state.metrics.stability_score:.2f}")

    return "\n".join(lines)
