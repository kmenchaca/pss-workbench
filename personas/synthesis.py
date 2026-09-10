"""Worldview Synthesis - Reconciling multiple persona perspectives.

Combines persona responses without losing attribution, identifies
genuine tensions, and creates multi-perspective outputs.
"""

from typing import Optional

from .library import PersonaLibrary
from .types import (
    EnsembleResult,
    Persona,
    PersonaResponse,
    WorldviewDelta,
)
from .worldview import (
    compare_all_pairs,
    extract_all,
    find_consensus_points,
    find_unique_insights,
    worldview_diversity_score,
)


# ============================================================================
# Synthesis Strategies
# ============================================================================


def synthesize_perspectives(
    responses: list[PersonaResponse],
    library: Optional[PersonaLibrary] = None,
    preserve_attribution: bool = True,
) -> str:
    """Combine persona perspectives without averaging away differences.

    Creates a synthesis that maintains the distinct voices and
    attributions rather than creating a bland average.

    Args:
        responses: List of persona responses
        library: PersonaLibrary for persona names
        preserve_attribution: Whether to include "The X said..." attribution

    Returns:
        Synthesized text combining perspectives
    """
    if library is None:
        library = PersonaLibrary()

    if not responses:
        return "No responses to synthesize."

    # Extract worldview elements from all responses
    enriched = [extract_all(r) for r in responses]

    parts: list[str] = []

    # Section 1: Overview
    parts.append("## Multi-Perspective Analysis\n")
    parts.append(f"Analyzed from {len(responses)} distinct perspectives.\n")

    # Section 2: Individual Perspectives
    parts.append("\n## Individual Perspectives\n")
    for response in enriched:
        persona = library.get(response.persona_id)
        name = persona.name if persona else response.persona_id.title()

        if preserve_attribution:
            parts.append(f"\n### {name}\n")
        else:
            parts.append(f"\n### Perspective {responses.index(response) + 1}\n")

        # Summarize their key points
        if response.risks_identified:
            parts.append(f"**Risks identified:** {', '.join(response.risks_identified[:3])}\n")
        if response.opportunities:
            parts.append(f"**Opportunities:** {', '.join(response.opportunities[:3])}\n")
        if response.assumptions:
            parts.append(f"**Key assumptions:** {', '.join(response.assumptions[:2])}\n")

    # Section 3: Consensus Points
    consensus = find_consensus_points(enriched)
    if consensus:
        parts.append("\n## Consensus Points\n")
        parts.append("Multiple perspectives agree on:\n")
        for point in consensus[:5]:
            parts.append(f"- {point}\n")

    # Section 4: Tensions
    tensions = identify_tensions(enriched)
    if tensions:
        parts.append("\n## Key Tensions\n")
        parts.append("Areas where perspectives genuinely disagree:\n")
        for tension in tensions[:5]:
            parts.append(f"- {tension}\n")

    # Section 5: Unique Insights
    unique = find_unique_insights(enriched, library)
    if unique:
        parts.append("\n## Unique Insights\n")
        parts.append("Perspectives that only one viewpoint identified:\n")
        for persona_id, insights in unique.items():
            persona = library.get(persona_id)
            name = persona.name if persona else persona_id.title()
            parts.append(f"\n**{name} uniquely noted:**\n")
            for insight in insights[:2]:
                parts.append(f"- {insight[:150]}...\n" if len(insight) > 150 else f"- {insight}\n")

    return "".join(parts)


def identify_tensions(
    responses: list[PersonaResponse],
) -> list[str]:
    """Identify genuine disagreements between personas.

    Goes beyond simple differences to find actual contradictions
    or fundamental disagreements in worldview.

    Args:
        responses: List of persona responses

    Returns:
        List of tension descriptions
    """
    deltas = compare_all_pairs(responses)

    tensions: list[str] = []
    seen_tensions: set[str] = set()

    for delta in deltas:
        for tension in delta.tensions:
            # Normalize to avoid duplicates
            key = tension.lower().strip()
            if key not in seen_tensions:
                tensions.append(f"{delta.persona_a} vs {delta.persona_b}: {tension}")
                seen_tensions.add(key)

    # Sort by severity (from the delta)
    deltas_by_severity = sorted(deltas, key=lambda d: d.severity, reverse=True)
    if deltas_by_severity and deltas_by_severity[0].severity > 0.5:
        high_severity = deltas_by_severity[0]
        tensions.insert(
            0, f"High tension between {high_severity.persona_a} and {high_severity.persona_b}"
        )

    return tensions


def attribute_insights(
    synthesis: str,
    responses: list[PersonaResponse],
    library: Optional[PersonaLibrary] = None,
) -> dict[str, list[str]]:
    """Map insights back to the personas who generated them.

    Takes a synthesis text and attributes which personas contributed
    which insights. Returns "The engineer noted...", "The optimist sees..." style attribution.

    Args:
        synthesis: Synthesized text
        responses: Original responses
        library: PersonaLibrary for names

    Returns:
        Map of insight to list of persona names who contributed
    """
    if library is None:
        library = PersonaLibrary()

    attributions: dict[str, list[str]] = {}

    # Find unique insights per persona
    unique = find_unique_insights(responses, library)

    for persona_id, insights in unique.items():
        persona = library.get(persona_id)
        name = persona.name if persona else persona_id.title()

        for insight in insights:
            # Create attribution phrase
            short_insight = insight[:100] + "..." if len(insight) > 100 else insight
            if short_insight not in attributions:
                attributions[short_insight] = []
            attributions[short_insight].append(name)

    return attributions


# ============================================================================
# Multi-Perspective Brief
# ============================================================================


def multi_perspective_brief(
    responses: list[PersonaResponse],
    library: Optional[PersonaLibrary] = None,
    max_length: int = 2000,
) -> str:
    """Create a structured brief showing all perspectives.

    Designed for decision-makers who need to see different
    viewpoints at a glance.

    Args:
        responses: List of persona responses
        library: PersonaLibrary for persona names
        max_length: Maximum length of output

    Returns:
        Formatted brief
    """
    if library is None:
        library = PersonaLibrary()

    if not responses:
        return "No perspectives available."

    enriched = [extract_all(r) for r in responses]

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("MULTI-PERSPECTIVE BRIEF")
    lines.append("=" * 60)
    lines.append("")

    # Quick summary
    lines.append(f"Perspectives analyzed: {len(responses)}")
    diversity = worldview_diversity_score(enriched)
    lines.append(f"Viewpoint diversity: {diversity:.0%}")
    lines.append("")

    # Each perspective in brief
    lines.append("-" * 60)
    lines.append("PERSPECTIVE SUMMARIES")
    lines.append("-" * 60)

    for response in enriched:
        persona = library.get(response.persona_id)
        name = persona.name if persona else response.persona_id.title()
        style = persona.thinking_style.value if persona else "unknown"

        lines.append(f"\n[{name.upper()}] ({style} thinker)")

        # Key point from response
        key_point = _extract_key_point(response.content)
        lines.append(f"  Key point: {key_point}")

        # Top risk if any
        if response.risks_identified:
            lines.append(f"  Top risk: {response.risks_identified[0][:80]}...")

        # Top opportunity if any
        if response.opportunities:
            lines.append(f"  Top opportunity: {response.opportunities[0][:80]}...")

    # Tension summary
    tensions = identify_tensions(enriched)
    if tensions:
        lines.append("")
        lines.append("-" * 60)
        lines.append("KEY TENSIONS")
        lines.append("-" * 60)
        for tension in tensions[:3]:
            lines.append(f"  - {tension}")

    # Consensus
    consensus = find_consensus_points(enriched)
    if consensus:
        lines.append("")
        lines.append("-" * 60)
        lines.append("CONSENSUS POINTS")
        lines.append("-" * 60)
        for point in consensus[:3]:
            lines.append(f"  - {point}")

    lines.append("")
    lines.append("=" * 60)

    result = "\n".join(lines)

    # Truncate if too long
    if len(result) > max_length:
        result = result[: max_length - 100] + "\n\n[Truncated for length]"

    return result


def _extract_key_point(content: str, max_length: int = 150) -> str:
    """Extract the key point from a response.

    Takes the first meaningful sentence as the key point.

    Args:
        content: Full response content
        max_length: Maximum length of extracted point

    Returns:
        Key point string
    """
    # Split into sentences
    sentences = content.replace("\n", " ").split(".")
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]

    if not sentences:
        return content[:max_length] + "..." if len(content) > max_length else content

    # Return first substantial sentence
    first = sentences[0]
    if len(first) > max_length:
        return first[:max_length] + "..."
    return first


# ============================================================================
# Ensemble Result Building
# ============================================================================


def build_ensemble_result(
    responses: list[PersonaResponse],
    library: Optional[PersonaLibrary] = None,
) -> EnsembleResult:
    """Build a complete ensemble result from persona responses.

    Args:
        responses: List of persona responses
        library: PersonaLibrary for names

    Returns:
        Complete EnsembleResult
    """
    if library is None:
        library = PersonaLibrary()

    # Enrich responses with extractions
    enriched = [extract_all(r) for r in responses]

    # Generate synthesis
    synthesis = synthesize_perspectives(enriched, library)

    # Find dominant themes (consensus)
    consensus = find_consensus_points(enriched)
    dominant_themes = [c.split(": ", 1)[-1] if ": " in c else c for c in consensus]

    # Find blind spots (what categories are missing?)
    covered_categories = set()
    for response in enriched:
        persona = library.get(response.persona_id)
        if persona:
            covered_categories.add(persona.category.value)

    all_categories = {"risk", "opportunity", "technical", "user", "strategic", "analytical"}
    missing = all_categories - covered_categories
    blind_spots = [f"Missing {cat} perspective" for cat in missing]

    # Get worldview deltas
    deltas = compare_all_pairs(enriched)

    # Calculate diversity
    diversity = worldview_diversity_score(enriched)

    # Build attributed insights
    attributed = attribute_insights(synthesis, enriched, library)

    return EnsembleResult(
        responses=enriched,
        synthesis=synthesis,
        dominant_themes=dominant_themes,
        blind_spots=blind_spots,
        worldview_deltas=deltas,
        diversity_score=diversity,
        attributed_insights=attributed,
    )


# ============================================================================
# Reconciliation Strategies
# ============================================================================


def reconcile_with_vote(
    responses: list[PersonaResponse],
    options: list[str],
    library: Optional[PersonaLibrary] = None,
) -> dict[str, list[str]]:
    """Have personas vote on options based on their perspectives.

    Args:
        responses: Persona responses
        options: Options to vote on
        library: PersonaLibrary

    Returns:
        Map of option to list of persona IDs who support it
    """
    if library is None:
        library = PersonaLibrary()

    votes: dict[str, list[str]] = {opt: [] for opt in options}

    # Simple heuristic: assign based on persona category alignment
    for response in responses:
        persona = library.get(response.persona_id)
        if not persona:
            continue

        # Check response content for option mentions
        content_lower = response.content.lower()
        for option in options:
            if option.lower() in content_lower:
                votes[option].append(response.persona_id)

    return votes


def reconcile_with_priority(
    responses: list[PersonaResponse],
    priority_order: list[str],
    library: Optional[PersonaLibrary] = None,
) -> str:
    """Reconcile by prioritizing certain personas' views.

    Args:
        responses: Persona responses
        priority_order: List of persona IDs in priority order
        library: PersonaLibrary

    Returns:
        Reconciled synthesis prioritizing given order
    """
    if library is None:
        library = PersonaLibrary()

    # Sort responses by priority
    response_map = {r.persona_id: r for r in responses}
    sorted_responses = []

    for persona_id in priority_order:
        if persona_id in response_map:
            sorted_responses.append(response_map[persona_id])

    # Add any responses not in priority list at the end
    for response in responses:
        if response.persona_id not in priority_order:
            sorted_responses.append(response)

    # Synthesize with priority weighting
    parts = ["## Prioritized Synthesis\n"]
    parts.append("(Perspectives ordered by priority)\n")

    for i, response in enumerate(sorted_responses):
        persona = library.get(response.persona_id)
        name = persona.name if persona else response.persona_id.title()
        priority_label = f"Priority {i + 1}" if i < len(priority_order) else "Additional"

        parts.append(f"\n### [{priority_label}] {name}\n")
        key_point = _extract_key_point(response.content)
        parts.append(f"{key_point}\n")

    return "".join(parts)


def deliberate(
    responses: list[PersonaResponse],
    rounds: int = 1,
    library: Optional[PersonaLibrary] = None,
) -> str:
    """Simulate deliberation between personas.

    Creates a structured back-and-forth considering different views.

    Args:
        responses: Persona responses
        rounds: Number of deliberation rounds to simulate
        library: PersonaLibrary

    Returns:
        Deliberation transcript
    """
    if library is None:
        library = PersonaLibrary()

    enriched = [extract_all(r) for r in responses]
    deltas = compare_all_pairs(enriched)

    parts = ["## Deliberation\n"]

    for round_num in range(rounds):
        parts.append(f"\n### Round {round_num + 1}\n")

        # Each persona responds to tensions
        for response in enriched:
            persona = library.get(response.persona_id)
            name = persona.name if persona else response.persona_id.title()

            # Find tensions involving this persona
            my_tensions = [d for d in deltas if response.persona_id in (d.persona_a, d.persona_b)]

            if my_tensions:
                parts.append(f"\n**{name}:** ")
                tension = my_tensions[0]
                other_id = (
                    tension.persona_b if tension.persona_a == response.persona_id else tension.persona_a
                )
                other_persona = library.get(other_id)
                other_name = other_persona.name if other_persona else other_id.title()

                if tension.tensions:
                    parts.append(f"Regarding the tension with {other_name}: {tension.tensions[0]}\n")
                else:
                    parts.append(f"I see common ground with {other_name}.\n")
            else:
                parts.append(f"\n**{name}:** My position stands.\n")

    # Final synthesis
    parts.append("\n### Deliberation Summary\n")
    consensus = find_consensus_points(enriched)
    if consensus:
        parts.append("Points of agreement: ")
        parts.append(", ".join(c.split(": ", 1)[-1] for c in consensus[:3]))
        parts.append("\n")

    remaining_tensions = identify_tensions(enriched)
    if remaining_tensions:
        parts.append("Unresolved tensions: ")
        parts.append(", ".join(remaining_tensions[:3]))
        parts.append("\n")

    return "".join(parts)
