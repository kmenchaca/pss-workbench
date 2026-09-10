"""Event system for the Temporal Scenario Planner.

This module provides event generation, chaining, and black swan injection
for timeline simulation.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable

from .types import (
    Event,
    EventCategory,
    TimelineState,
)


# Event templates by category
EVENT_TEMPLATES: dict[EventCategory, list[dict[str, Any]]] = {
    EventCategory.MARKET: [
        {
            "description": "Market expands by {pct}%",
            "effects": {"metric:opportunity_score": 0.1},
            "magnitude_range": (2, 5),
            "base_probability": 0.3,
        },
        {
            "description": "Market contracts by {pct}%",
            "effects": {"metric:risk_score": 0.1},
            "magnitude_range": (-5, -2),
            "base_probability": 0.2,
        },
        {
            "description": "New market segment emerges",
            "effects": {"metric:opportunity_score": 0.15},
            "magnitude_range": (3, 6),
            "base_probability": 0.1,
        },
        {
            "description": "Price war erupts",
            "effects": {"metric:risk_score": 0.15},
            "magnitude_range": (-4, -2),
            "base_probability": 0.15,
        },
    ],
    EventCategory.COMPETITIVE: [
        {
            "description": "Competitor launches new product",
            "effects": {"metric:risk_score": 0.1},
            "magnitude_range": (-3, -1),
            "base_probability": 0.25,
        },
        {
            "description": "Competitor exits market",
            "effects": {"metric:opportunity_score": 0.2},
            "magnitude_range": (3, 6),
            "base_probability": 0.05,
        },
        {
            "description": "Competitor acquires key partner",
            "effects": {"metric:risk_score": 0.15},
            "magnitude_range": (-4, -2),
            "base_probability": 0.1,
        },
        {
            "description": "Competitor faces scandal",
            "effects": {"metric:opportunity_score": 0.1},
            "magnitude_range": (2, 4),
            "base_probability": 0.08,
        },
    ],
    EventCategory.TECHNICAL: [
        {
            "description": "Technology breakthrough achieved",
            "effects": {"metric:success_score": 0.15},
            "magnitude_range": (4, 7),
            "base_probability": 0.1,
        },
        {
            "description": "Critical system failure",
            "effects": {"metric:stability_score": -0.2},
            "magnitude_range": (-6, -3),
            "base_probability": 0.05,
        },
        {
            "description": "New integration opportunity",
            "effects": {"metric:opportunity_score": 0.1},
            "magnitude_range": (2, 4),
            "base_probability": 0.15,
        },
        {
            "description": "Technical debt becomes critical",
            "effects": {"metric:risk_score": 0.1},
            "magnitude_range": (-3, -1),
            "base_probability": 0.2,
        },
    ],
    EventCategory.REGULATORY: [
        {
            "description": "New favorable regulation passed",
            "effects": {"metric:opportunity_score": 0.15},
            "magnitude_range": (3, 5),
            "base_probability": 0.1,
        },
        {
            "description": "Restrictive regulation proposed",
            "effects": {"metric:risk_score": 0.15},
            "magnitude_range": (-4, -2),
            "base_probability": 0.15,
        },
        {
            "description": "Compliance deadline approaching",
            "effects": {"metric:stability_score": -0.1},
            "magnitude_range": (-2, 0),
            "base_probability": 0.2,
        },
        {
            "description": "Industry receives regulatory approval",
            "effects": {"metric:success_score": 0.1},
            "magnitude_range": (2, 4),
            "base_probability": 0.08,
        },
    ],
    EventCategory.SOCIAL: [
        {
            "description": "Positive media coverage",
            "effects": {"metric:success_score": 0.1},
            "magnitude_range": (2, 4),
            "base_probability": 0.2,
        },
        {
            "description": "Public backlash over issue",
            "effects": {"metric:risk_score": 0.15},
            "magnitude_range": (-5, -2),
            "base_probability": 0.1,
        },
        {
            "description": "Consumer trend shifts favorably",
            "effects": {"metric:opportunity_score": 0.1},
            "magnitude_range": (2, 4),
            "base_probability": 0.15,
        },
        {
            "description": "Key talent joins organization",
            "effects": {"metric:success_score": 0.1},
            "magnitude_range": (2, 3),
            "base_probability": 0.12,
        },
    ],
    EventCategory.INTERNAL: [
        {
            "description": "Process improvement implemented",
            "effects": {"metric:stability_score": 0.1},
            "magnitude_range": (1, 3),
            "base_probability": 0.25,
        },
        {
            "description": "Internal conflict arises",
            "effects": {"metric:stability_score": -0.15},
            "magnitude_range": (-3, -1),
            "base_probability": 0.15,
        },
        {
            "description": "Cost savings realized",
            "effects": {"metric:success_score": 0.08},
            "magnitude_range": (1, 3),
            "base_probability": 0.2,
        },
        {
            "description": "Key employee departure",
            "effects": {"metric:risk_score": 0.1},
            "magnitude_range": (-3, -1),
            "base_probability": 0.1,
        },
    ],
}


# Black swan events (rare, high-impact)
BLACK_SWAN_EVENTS: list[dict[str, Any]] = [
    {
        "description": "Global pandemic disrupts operations",
        "effects": {"metric:stability_score": -0.4, "metric:risk_score": 0.3},
        "magnitude": -8,
        "category": EventCategory.SOCIAL,
    },
    {
        "description": "Major cybersecurity breach",
        "effects": {"metric:stability_score": -0.3, "metric:success_score": -0.2},
        "magnitude": -7,
        "category": EventCategory.TECHNICAL,
    },
    {
        "description": "Revolutionary technology makes product obsolete",
        "effects": {"metric:risk_score": 0.4, "metric:opportunity_score": -0.3},
        "magnitude": -9,
        "category": EventCategory.TECHNICAL,
    },
    {
        "description": "Major acquisition opportunity arises",
        "effects": {"metric:opportunity_score": 0.4},
        "magnitude": 8,
        "category": EventCategory.MARKET,
    },
    {
        "description": "Industry-changing regulation passed",
        "effects": {"metric:stability_score": -0.2},
        "magnitude": -6,
        "category": EventCategory.REGULATORY,
    },
    {
        "description": "Unexpected market boom in sector",
        "effects": {"metric:opportunity_score": 0.3, "metric:success_score": 0.2},
        "magnitude": 7,
        "category": EventCategory.MARKET,
    },
]


def generate_event_id() -> str:
    """Generate a unique event ID."""
    return f"event_{uuid.uuid4().hex[:8]}"


def generate_events(
    state: TimelineState,
    context: dict[str, Any] | None = None,
    num_events: int = 3,
    categories: list[EventCategory] | None = None,
    llm_provider: Callable[[str], str] | None = None,
) -> list[Event]:
    """Generate plausible events for a state.

    Args:
        state: Current timeline state.
        context: Additional context for event generation.
        num_events: Number of events to generate.
        categories: Categories to draw from (all if None).
        llm_provider: Optional LLM for more creative events.

    Returns:
        List of generated events.
    """
    events: list[Event] = []
    context = context or {}
    categories = categories or list(EventCategory)

    # Adjust probabilities based on state
    prob_modifiers = _calculate_probability_modifiers(state)

    for _ in range(num_events):
        # Pick a category weighted by state
        category = _select_category(categories, state)
        templates = EVENT_TEMPLATES.get(category, [])

        if not templates:
            continue

        # Select and instantiate a template
        template = random.choice(templates)
        event = _instantiate_template(template, state.date, category, prob_modifiers)

        # Optionally enhance with LLM
        if llm_provider and random.random() < 0.3:
            event = _enhance_event_with_llm(event, state, llm_provider)

        events.append(event)

    return events


def _calculate_probability_modifiers(state: TimelineState) -> dict[EventCategory, float]:
    """Calculate probability modifiers based on state."""
    modifiers: dict[EventCategory, float] = {}

    # High risk increases negative event probability
    if state.metrics.risk_score > 0.6:
        modifiers[EventCategory.COMPETITIVE] = 1.3
        modifiers[EventCategory.REGULATORY] = 1.2

    # Low stability increases internal event probability
    if state.metrics.stability_score < 0.4:
        modifiers[EventCategory.INTERNAL] = 1.4

    # High opportunity increases market events
    if state.metrics.opportunity_score > 0.6:
        modifiers[EventCategory.MARKET] = 1.3

    return modifiers


def _select_category(categories: list[EventCategory], state: TimelineState) -> EventCategory:
    """Select an event category based on state."""
    weights = []
    for cat in categories:
        weight = 1.0

        # Adjust weights based on recent events
        recent_same_cat = sum(1 for e in state.events[-5:] if e.category == cat)
        weight *= 0.7 ** recent_same_cat  # Reduce probability if category is saturated

        weights.append(weight)

    # Normalize weights
    total = sum(weights)
    if total == 0:
        return random.choice(categories)

    normalized = [w / total for w in weights]
    return random.choices(categories, weights=normalized, k=1)[0]


def _instantiate_template(
    template: dict[str, Any],
    base_date: datetime,
    category: EventCategory,
    prob_modifiers: dict[EventCategory, float],
) -> Event:
    """Create an event from a template."""
    # Generate magnitude within range
    mag_range = template.get("magnitude_range", (-3, 3))
    magnitude = random.randint(mag_range[0], mag_range[1])

    # Calculate probability
    base_prob = template.get("base_probability", 0.5)
    modifier = prob_modifiers.get(category, 1.0)
    probability = min(1.0, base_prob * modifier)

    # Generate description with placeholders
    description = template["description"]
    if "{pct}" in description:
        pct = abs(magnitude) * 3
        description = description.format(pct=pct)

    return Event(
        id=generate_event_id(),
        date=base_date + timedelta(days=random.randint(1, 30)),
        description=description,
        cause=None,
        effects=dict(template.get("effects", {})),
        probability=probability,
        category=category,
        magnitude=magnitude,
    )


def _enhance_event_with_llm(
    event: Event,
    state: TimelineState,
    llm_provider: Callable[[str], str],
) -> Event:
    """Enhance an event description using LLM."""
    prompt = f"""Given this generic event in a business simulation:
Event: {event.description}
Category: {event.category.value}
Magnitude: {event.magnitude}

Current context:
- Success score: {state.metrics.success_score:.2f}
- Risk score: {state.metrics.risk_score:.2f}
- Recent events: {', '.join(e.description for e in state.events[-3:])}

Make the event description more specific and realistic.
Respond with just the improved description (one sentence)."""

    try:
        enhanced_description = llm_provider(prompt).strip()
        if enhanced_description and len(enhanced_description) < 200:
            event.description = enhanced_description
    except Exception:
        pass  # Keep original if LLM fails

    return event


def chain_events(
    event: Event,
    state: TimelineState,
    max_chain_length: int = 3,
    llm_provider: Callable[[str], str] | None = None,
) -> list[Event]:
    """Generate a cascade of consequences from an event.

    Args:
        event: The triggering event.
        state: Current timeline state.
        max_chain_length: Maximum number of chained events.
        llm_provider: Optional LLM for creative chaining.

    Returns:
        List of consequent events.
    """
    chained: list[Event] = []

    # Only chain high-magnitude events
    if abs(event.magnitude) < 3:
        return chained

    current_event = event
    for i in range(max_chain_length):
        # Probability of chaining decreases
        if random.random() > 0.6 - (i * 0.15):
            break

        next_event = _generate_consequence(current_event, state, llm_provider)
        if next_event:
            chained.append(next_event)
            current_event = next_event

    return chained


def _generate_consequence(
    cause_event: Event,
    state: TimelineState,
    llm_provider: Callable[[str], str] | None = None,
) -> Event | None:
    """Generate a consequence event from a cause."""
    # Determine likely consequence category
    consequence_categories = _get_consequence_categories(cause_event.category)

    if not consequence_categories:
        return None

    category = random.choice(consequence_categories)
    templates = EVENT_TEMPLATES.get(category, [])

    if not templates:
        return None

    # Pick template matching the polarity of the cause
    matching = [t for t in templates if _same_polarity(t, cause_event.magnitude)]
    if not matching:
        matching = templates

    template = random.choice(matching)

    # Create consequence event
    consequence = _instantiate_template(
        template,
        cause_event.date,
        category,
        {},
    )
    consequence.cause = cause_event.id
    consequence.date = cause_event.date + timedelta(days=random.randint(7, 60))

    # Dampen the magnitude
    consequence.magnitude = int(cause_event.magnitude * 0.6)

    return consequence


def _get_consequence_categories(cause_category: EventCategory) -> list[EventCategory]:
    """Get categories that can follow from a cause category."""
    mappings = {
        EventCategory.MARKET: [EventCategory.COMPETITIVE, EventCategory.INTERNAL],
        EventCategory.COMPETITIVE: [EventCategory.MARKET, EventCategory.INTERNAL],
        EventCategory.TECHNICAL: [EventCategory.INTERNAL, EventCategory.MARKET],
        EventCategory.REGULATORY: [EventCategory.MARKET, EventCategory.COMPETITIVE],
        EventCategory.SOCIAL: [EventCategory.REGULATORY, EventCategory.MARKET],
        EventCategory.INTERNAL: [EventCategory.TECHNICAL, EventCategory.SOCIAL],
    }
    return mappings.get(cause_category, [])


def _same_polarity(template: dict[str, Any], magnitude: int) -> bool:
    """Check if template matches the magnitude polarity."""
    mag_range = template.get("magnitude_range", (0, 0))
    avg_mag = (mag_range[0] + mag_range[1]) / 2
    return (avg_mag >= 0) == (magnitude >= 0)


def black_swan_injection(
    state: TimelineState,
    force_type: str | None = None,
) -> Event:
    """Force an unlikely but impactful event.

    Args:
        state: Current timeline state.
        force_type: Type of black swan ('positive' or 'negative').

    Returns:
        A black swan event.
    """
    candidates = BLACK_SWAN_EVENTS.copy()

    if force_type == "positive":
        candidates = [e for e in candidates if e["magnitude"] > 0]
    elif force_type == "negative":
        candidates = [e for e in candidates if e["magnitude"] < 0]

    if not candidates:
        candidates = BLACK_SWAN_EVENTS

    template = random.choice(candidates)

    return Event(
        id=generate_event_id(),
        date=state.date + timedelta(days=random.randint(1, 14)),
        description=template["description"],
        cause="black_swan",
        effects=dict(template["effects"]),
        probability=0.01,  # Black swans are rare
        category=template["category"],
        magnitude=template["magnitude"],
    )


def filter_events_by_probability(
    events: list[Event],
    threshold: float = 0.5,
    sample: bool = True,
) -> list[Event]:
    """Filter events based on their probability.

    Args:
        events: Events to filter.
        threshold: Minimum probability (if not sampling).
        sample: If True, randomly sample based on probability.

    Returns:
        Filtered list of events.
    """
    if sample:
        return [e for e in events if random.random() < e.probability]
    else:
        return [e for e in events if e.probability >= threshold]


def summarize_events(events: list[Event]) -> str:
    """Create a human-readable summary of events.

    Args:
        events: Events to summarize.

    Returns:
        Summary string.
    """
    if not events:
        return "No events occurred."

    by_category: dict[EventCategory, list[Event]] = {}
    for event in events:
        if event.category not in by_category:
            by_category[event.category] = []
        by_category[event.category].append(event)

    lines = [f"Summary of {len(events)} events:"]
    for category, cat_events in by_category.items():
        lines.append(f"\n{category.value.title()} ({len(cat_events)} events):")
        for event in cat_events[:3]:
            mag_str = f"+{event.magnitude}" if event.magnitude > 0 else str(event.magnitude)
            lines.append(f"  - {event.description} [{mag_str}]")
        if len(cat_events) > 3:
            lines.append(f"  ... and {len(cat_events) - 3} more")

    return "\n".join(lines)
