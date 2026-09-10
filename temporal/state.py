"""Timeline state management for the Temporal Scenario Planner.

This module handles state transitions, checkpointing, and state comparison
for timeline simulation.
"""

import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

from .types import (
    ActorState,
    Event,
    TimelineMetrics,
    TimelineState,
)


def create_initial_state(
    date: datetime,
    actors: dict[str, ActorState] | None = None,
    context: dict[str, Any] | None = None,
) -> TimelineState:
    """Create a new initial timeline state.

    Args:
        date: Starting date for the timeline.
        actors: Initial actor states.
        context: Additional context data.

    Returns:
        A new TimelineState with the given initial conditions.
    """
    return TimelineState(
        date=date,
        events=[],
        actor_states=actors or {},
        metrics=TimelineMetrics(),
        context=context or {},
        parent_state_id=None,
    )


def advance_time(state: TimelineState, duration: timedelta) -> TimelineState:
    """Move the timeline state forward by a duration.

    Creates a new state with updated date. Does not modify the original state.

    Args:
        state: Current timeline state.
        duration: How far to advance.

    Returns:
        New TimelineState with updated date.
    """
    new_state = deepcopy(state)
    new_state.date = state.date + duration
    new_state.parent_state_id = f"state_{id(state)}"
    return new_state


def apply_event(state: TimelineState, event: Event) -> TimelineState:
    """Update state from an event.

    Applies the event's effects to the state, creating a new state.
    Does not modify the original state.

    Args:
        state: Current timeline state.
        event: Event to apply.

    Returns:
        New TimelineState with event effects applied.
    """
    new_state = deepcopy(state)
    new_state.events.append(event)

    # Apply effects to context
    for key, value in event.effects.items():
        if key.startswith("actor:"):
            # Effect targets an actor
            actor_id = key.split(":", 1)[1]
            if actor_id in new_state.actor_states:
                _apply_actor_effect(new_state.actor_states[actor_id], value)
        elif key.startswith("metric:"):
            # Effect targets a metric
            metric_name = key.split(":", 1)[1]
            _apply_metric_effect(new_state.metrics, metric_name, value)
        else:
            # Effect targets context
            _apply_context_effect(new_state.context, key, value)

    # Update metrics based on event magnitude
    _update_metrics_from_event(new_state.metrics, event)

    return new_state


def _apply_actor_effect(actor_state: ActorState, effect: dict[str, Any]) -> None:
    """Apply an effect to an actor's state."""
    if "resources" in effect:
        for resource, delta in effect["resources"].items():
            current = actor_state.resources.get(resource, 0.0)
            actor_state.resources[resource] = current + delta

    if "morale" in effect:
        actor_state.morale = max(-1.0, min(1.0, actor_state.morale + effect["morale"]))

    if "capabilities" in effect:
        for cap in effect.get("capabilities_add", []):
            if cap not in actor_state.capabilities:
                actor_state.capabilities.append(cap)
        for cap in effect.get("capabilities_remove", []):
            if cap in actor_state.capabilities:
                actor_state.capabilities.remove(cap)

    if "relationships" in effect:
        for other_id, delta in effect["relationships"].items():
            current = actor_state.relationships.get(other_id, 0.0)
            actor_state.relationships[other_id] = max(-1.0, min(1.0, current + delta))


def _apply_metric_effect(metrics: TimelineMetrics, metric_name: str, value: float) -> None:
    """Apply an effect to timeline metrics."""
    if hasattr(metrics, metric_name):
        current = getattr(metrics, metric_name)
        setattr(metrics, metric_name, max(0.0, min(1.0, current + value)))
    else:
        current = metrics.custom_metrics.get(metric_name, 0.0)
        metrics.custom_metrics[metric_name] = current + value


def _apply_context_effect(context: dict[str, Any], key: str, value: Any) -> None:
    """Apply an effect to state context."""
    if isinstance(value, dict) and "delta" in value:
        current = context.get(key, 0)
        context[key] = current + value["delta"]
    elif isinstance(value, dict) and "set" in value:
        context[key] = value["set"]
    else:
        context[key] = value


def _update_metrics_from_event(metrics: TimelineMetrics, event: Event) -> None:
    """Update metrics based on event characteristics."""
    # Positive magnitude events increase success/opportunity
    if event.magnitude > 0:
        metrics.success_score = min(1.0, metrics.success_score + event.magnitude * 0.02)
        metrics.opportunity_score = min(1.0, metrics.opportunity_score + event.magnitude * 0.01)
    elif event.magnitude < 0:
        metrics.success_score = max(0.0, metrics.success_score + event.magnitude * 0.02)
        metrics.risk_score = min(1.0, metrics.risk_score - event.magnitude * 0.02)

    # Low probability events affect stability
    if event.probability < 0.3:
        metrics.stability_score = max(0.0, metrics.stability_score - 0.05)


def checkpoint_state(state: TimelineState) -> tuple[str, TimelineState]:
    """Save state for branching.

    Creates a checkpoint ID and a deep copy of the state.

    Args:
        state: State to checkpoint.

    Returns:
        Tuple of (checkpoint_id, state_copy).
    """
    checkpoint_id = f"checkpoint_{uuid.uuid4().hex[:8]}"
    state_copy = deepcopy(state)
    state_copy.parent_state_id = checkpoint_id
    return checkpoint_id, state_copy


def diff_states(s1: TimelineState, s2: TimelineState) -> dict[str, Any]:
    """Calculate what changed between two states.

    Args:
        s1: First state (typically earlier).
        s2: Second state (typically later).

    Returns:
        Dict describing the differences.
    """
    diff: dict[str, Any] = {
        "time_elapsed": s2.date - s1.date,
        "events_added": [],
        "actor_changes": {},
        "metric_changes": {},
        "context_changes": {},
    }

    # Find new events
    s1_event_ids = {e.id for e in s1.events}
    for event in s2.events:
        if event.id not in s1_event_ids:
            diff["events_added"].append(event)

    # Compare actor states
    all_actors = set(s1.actor_states.keys()) | set(s2.actor_states.keys())
    for actor_id in all_actors:
        a1 = s1.actor_states.get(actor_id)
        a2 = s2.actor_states.get(actor_id)

        if a1 is None and a2 is not None:
            diff["actor_changes"][actor_id] = {"status": "added", "state": a2}
        elif a1 is not None and a2 is None:
            diff["actor_changes"][actor_id] = {"status": "removed"}
        elif a1 is not None and a2 is not None:
            actor_diff = _diff_actor_states(a1, a2)
            if actor_diff:
                diff["actor_changes"][actor_id] = actor_diff

    # Compare metrics
    diff["metric_changes"] = _diff_metrics(s1.metrics, s2.metrics)

    # Compare context
    all_keys = set(s1.context.keys()) | set(s2.context.keys())
    for key in all_keys:
        v1 = s1.context.get(key)
        v2 = s2.context.get(key)
        if v1 != v2:
            diff["context_changes"][key] = {"from": v1, "to": v2}

    return diff


def _diff_actor_states(a1: ActorState, a2: ActorState) -> dict[str, Any]:
    """Calculate differences between two actor states."""
    changes: dict[str, Any] = {}

    # Resource changes
    all_resources = set(a1.resources.keys()) | set(a2.resources.keys())
    resource_changes = {}
    for resource in all_resources:
        v1 = a1.resources.get(resource, 0)
        v2 = a2.resources.get(resource, 0)
        if v1 != v2:
            resource_changes[resource] = {"from": v1, "to": v2, "delta": v2 - v1}
    if resource_changes:
        changes["resources"] = resource_changes

    # Morale change
    if a1.morale != a2.morale:
        changes["morale"] = {"from": a1.morale, "to": a2.morale}

    # Capability changes
    cap_added = set(a2.capabilities) - set(a1.capabilities)
    cap_removed = set(a1.capabilities) - set(a2.capabilities)
    if cap_added or cap_removed:
        changes["capabilities"] = {"added": list(cap_added), "removed": list(cap_removed)}

    # Relationship changes
    all_relations = set(a1.relationships.keys()) | set(a2.relationships.keys())
    rel_changes = {}
    for other_id in all_relations:
        v1 = a1.relationships.get(other_id, 0)
        v2 = a2.relationships.get(other_id, 0)
        if v1 != v2:
            rel_changes[other_id] = {"from": v1, "to": v2}
    if rel_changes:
        changes["relationships"] = rel_changes

    return changes


def _diff_metrics(m1: TimelineMetrics, m2: TimelineMetrics) -> dict[str, dict[str, float]]:
    """Calculate differences between two metric sets."""
    changes: dict[str, dict[str, float]] = {}

    for metric in ["success_score", "risk_score", "opportunity_score", "stability_score"]:
        v1 = getattr(m1, metric)
        v2 = getattr(m2, metric)
        if v1 != v2:
            changes[metric] = {"from": v1, "to": v2}

    # Custom metrics
    all_custom = set(m1.custom_metrics.keys()) | set(m2.custom_metrics.keys())
    for key in all_custom:
        v1 = m1.custom_metrics.get(key, 0)
        v2 = m2.custom_metrics.get(key, 0)
        if v1 != v2:
            changes[f"custom:{key}"] = {"from": v1, "to": v2}

    return changes


def merge_states(
    base: TimelineState,
    states: list[TimelineState],
    strategy: str = "average",
) -> TimelineState:
    """Merge multiple states into one.

    Args:
        base: Base state to start from.
        states: States to merge.
        strategy: How to merge ('average', 'optimistic', 'pessimistic').

    Returns:
        Merged TimelineState.
    """
    if not states:
        return deepcopy(base)

    merged = deepcopy(base)

    # Use the latest date
    merged.date = max(s.date for s in states)

    # Merge events (union of all events)
    all_events: dict[str, Event] = {}
    for s in states:
        for event in s.events:
            if event.id not in all_events:
                all_events[event.id] = event
    merged.events = list(all_events.values())

    # Merge metrics based on strategy
    if strategy == "average":
        merged.metrics = _average_metrics([s.metrics for s in states])
    elif strategy == "optimistic":
        merged.metrics = _best_metrics([s.metrics for s in states])
    elif strategy == "pessimistic":
        merged.metrics = _worst_metrics([s.metrics for s in states])

    return merged


def _average_metrics(metrics_list: list[TimelineMetrics]) -> TimelineMetrics:
    """Average multiple metrics."""
    n = len(metrics_list)
    return TimelineMetrics(
        success_score=sum(m.success_score for m in metrics_list) / n,
        risk_score=sum(m.risk_score for m in metrics_list) / n,
        opportunity_score=sum(m.opportunity_score for m in metrics_list) / n,
        stability_score=sum(m.stability_score for m in metrics_list) / n,
    )


def _best_metrics(metrics_list: list[TimelineMetrics]) -> TimelineMetrics:
    """Take the best of each metric."""
    return TimelineMetrics(
        success_score=max(m.success_score for m in metrics_list),
        risk_score=min(m.risk_score for m in metrics_list),
        opportunity_score=max(m.opportunity_score for m in metrics_list),
        stability_score=max(m.stability_score for m in metrics_list),
    )


def _worst_metrics(metrics_list: list[TimelineMetrics]) -> TimelineMetrics:
    """Take the worst of each metric."""
    return TimelineMetrics(
        success_score=min(m.success_score for m in metrics_list),
        risk_score=max(m.risk_score for m in metrics_list),
        opportunity_score=min(m.opportunity_score for m in metrics_list),
        stability_score=min(m.stability_score for m in metrics_list),
    )


def state_to_summary(state: TimelineState) -> str:
    """Create a human-readable summary of a state.

    Args:
        state: State to summarize.

    Returns:
        Human-readable summary string.
    """
    lines = [
        f"Timeline State at {state.date.isoformat()}",
        f"Events: {len(state.events)}",
        f"Actors: {len(state.actor_states)}",
        "Metrics:",
        f"  Success: {state.metrics.success_score:.2f}",
        f"  Risk: {state.metrics.risk_score:.2f}",
        f"  Opportunity: {state.metrics.opportunity_score:.2f}",
        f"  Stability: {state.metrics.stability_score:.2f}",
    ]

    if state.events:
        lines.append("Recent events:")
        for event in state.events[-3:]:
            lines.append(f"  - {event.description}")

    return "\n".join(lines)
