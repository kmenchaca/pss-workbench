"""Worldview Analysis - Extracting and comparing persona perspectives.

Provides tools to extract assumptions, risks, and opportunities from
persona responses, and compare worldviews to find agreements and tensions.
"""

import re
from typing import Optional

from .library import PersonaLibrary
from .types import (
    Persona,
    PersonaResponse,
    WorldviewDelta,
)


# ============================================================================
# Extraction Patterns
# ============================================================================

# Patterns that indicate assumptions
ASSUMPTION_PATTERNS = [
    r"assum(?:e|ing|ed|ption)",
    r"expect(?:ing|ed|ation)",
    r"presum(?:e|ing|ed|ption)",
    r"tak(?:e|ing|en) for granted",
    r"given that",
    r"based on the premise",
    r"if we assume",
    r"assuming",
    r"implies? that",
    r"must be",
    r"should be",
    r"would be",
]

# Patterns that indicate risks
RISK_PATTERNS = [
    r"risk(?:s|y|ed|ing)?",
    r"danger(?:s|ous)?",
    r"threat(?:s|en)?",
    r"vulnerab(?:le|ility|ilities)",
    r"fail(?:s|ed|ure|ing)?",
    r"problem(?:s|atic)?",
    r"issue(?:s)?",
    r"concern(?:s|ed|ing)?",
    r"worry|worries|worried",
    r"could go wrong",
    r"might not work",
    r"potential(?:ly)? negative",
    r"downside",
]

# Patterns that indicate opportunities
OPPORTUNITY_PATTERNS = [
    r"opportunit(?:y|ies)",
    r"potential(?:ly)? positive",
    r"benefit(?:s|ed|ing)?",
    r"advantage(?:s)?",
    r"upside",
    r"could (?:help|improve|enable)",
    r"might (?:lead to|result in) (?:good|positive|better)",
    r"possibilit(?:y|ies)",
    r"promising",
    r"exciting",
    r"gain(?:s)?",
]


# ============================================================================
# Content Extraction
# ============================================================================


def extract_sentences_with_pattern(content: str, patterns: list[str]) -> list[str]:
    """Extract sentences containing any of the given patterns.

    Args:
        content: Text to extract from
        patterns: Regex patterns to match

    Returns:
        List of matching sentences
    """
    # Split into sentences (rough approximation)
    sentences = re.split(r"[.!?]+", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    matches: list[str] = []
    combined_pattern = "|".join(f"({p})" for p in patterns)

    for sentence in sentences:
        if re.search(combined_pattern, sentence, re.IGNORECASE):
            matches.append(sentence)

    return matches


def extract_assumptions(response: PersonaResponse) -> list[str]:
    """Extract assumptions from a persona's response.

    Identifies statements where the persona is making assumptions,
    whether explicit or implicit.

    Args:
        response: The persona's response

    Returns:
        List of extracted assumptions
    """
    # First use pre-extracted if available
    if response.assumptions:
        return response.assumptions

    content = response.content
    assumptions = extract_sentences_with_pattern(content, ASSUMPTION_PATTERNS)

    # Also look for implicit assumptions (declarative statements without evidence)
    # These often start with "This is", "It will", "They would"
    implicit_patterns = [
        r"^(?:this|it|they|we)\s+(?:is|are|will|would|should)\s+",
        r"^(?:the|a|an)\s+\w+\s+(?:is|are|will)\s+",
        r"clearly|obviously|certainly|definitely",
    ]

    implicit = extract_sentences_with_pattern(content, implicit_patterns)

    # Deduplicate and limit
    all_assumptions = list(dict.fromkeys(assumptions + implicit))
    return all_assumptions[:10]  # Limit to top 10


def extract_risks(response: PersonaResponse) -> list[str]:
    """Extract risks identified by a persona.

    Args:
        response: The persona's response

    Returns:
        List of identified risks
    """
    if response.risks_identified:
        return response.risks_identified

    content = response.content
    risks = extract_sentences_with_pattern(content, RISK_PATTERNS)

    return list(dict.fromkeys(risks))[:10]


def extract_opportunities(response: PersonaResponse) -> list[str]:
    """Extract opportunities identified by a persona.

    Args:
        response: The persona's response

    Returns:
        List of identified opportunities
    """
    if response.opportunities:
        return response.opportunities

    content = response.content
    opportunities = extract_sentences_with_pattern(content, OPPORTUNITY_PATTERNS)

    return list(dict.fromkeys(opportunities))[:10]


def extract_all(response: PersonaResponse) -> PersonaResponse:
    """Extract all worldview elements from a response.

    Creates a new response with populated extraction fields.

    Args:
        response: Original response

    Returns:
        New PersonaResponse with extractions populated
    """
    return PersonaResponse(
        persona_id=response.persona_id,
        content=response.content,
        assumptions=extract_assumptions(response),
        risks_identified=extract_risks(response),
        opportunities=extract_opportunities(response),
        confidence=response.confidence,
        raw_response=response.raw_response,
    )


# ============================================================================
# Worldview Comparison
# ============================================================================


def compare_worldviews(
    response_a: PersonaResponse,
    response_b: PersonaResponse,
) -> WorldviewDelta:
    """Compare worldviews of two personas.

    Identifies agreements, tensions, and unique insights.

    Args:
        response_a: First persona's response
        response_b: Second persona's response

    Returns:
        WorldviewDelta capturing the comparison
    """
    # Extract elements from both
    assumptions_a = set(extract_assumptions(response_a))
    assumptions_b = set(extract_assumptions(response_b))

    risks_a = set(extract_risks(response_a))
    risks_b = set(extract_risks(response_b))

    opps_a = set(extract_opportunities(response_a))
    opps_b = set(extract_opportunities(response_b))

    # Find overlaps (simplified - just looking for exact matches)
    # In practice, would use semantic similarity
    agreements: list[str] = []
    tensions: list[str] = []

    # Check for similar risk assessments (agreement)
    risk_overlap = risks_a & risks_b
    if risk_overlap:
        agreements.extend([f"Both identify risk: {r}" for r in list(risk_overlap)[:3]])

    # Check for contradictory views (tensions)
    # If one sees risk and other sees opportunity in similar areas
    for risk in risks_a:
        for opp in opps_b:
            if _content_overlap(risk, opp):
                tensions.append(f"{response_a.persona_id} sees risk, {response_b.persona_id} sees opportunity")
                break

    for risk in risks_b:
        for opp in opps_a:
            if _content_overlap(risk, opp):
                tensions.append(f"{response_b.persona_id} sees risk, {response_a.persona_id} sees opportunity")
                break

    # Unique insights
    unique_a = list(risks_a - risks_b)[:3] + list(opps_a - opps_b)[:3]
    unique_b = list(risks_b - risks_a)[:3] + list(opps_b - opps_a)[:3]

    # Calculate severity based on number of tensions
    severity = min(1.0, len(tensions) * 0.2)

    return WorldviewDelta(
        persona_a=response_a.persona_id,
        persona_b=response_b.persona_id,
        agreements=agreements,
        tensions=tensions,
        unique_to_a=unique_a,
        unique_to_b=unique_b,
        severity=severity,
    )


def _content_overlap(text_a: str, text_b: str, threshold: int = 3) -> bool:
    """Check if two pieces of text have significant word overlap.

    Simple heuristic - in practice would use embeddings.

    Args:
        text_a: First text
        text_b: Second text
        threshold: Minimum number of shared words

    Returns:
        True if texts have significant overlap
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    # Remove common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "and",
        "or",
        "but",
        "if",
        "because",
        "that",
        "this",
        "it",
    }

    words_a = words_a - stop_words
    words_b = words_b - stop_words

    overlap = words_a & words_b
    return len(overlap) >= threshold


def compare_all_pairs(
    responses: list[PersonaResponse],
) -> list[WorldviewDelta]:
    """Compare all pairs of persona responses.

    Args:
        responses: List of persona responses

    Returns:
        List of WorldviewDeltas for each pair
    """
    deltas: list[WorldviewDelta] = []

    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            delta = compare_worldviews(responses[i], responses[j])
            deltas.append(delta)

    return deltas


# ============================================================================
# Diversity Scoring
# ============================================================================


def worldview_diversity_score(responses: list[PersonaResponse]) -> float:
    """Calculate how diverse the worldviews actually are.

    A high score means personas are providing genuinely different perspectives.
    A low score means they're saying similar things.

    Args:
        responses: List of persona responses

    Returns:
        Diversity score from 0.0 (identical) to 1.0 (maximally diverse)
    """
    if len(responses) < 2:
        return 0.0

    # Collect all unique insights
    all_assumptions: set[str] = set()
    all_risks: set[str] = set()
    all_opportunities: set[str] = set()

    per_persona_counts: list[int] = []

    for response in responses:
        assumptions = extract_assumptions(response)
        risks = extract_risks(response)
        opps = extract_opportunities(response)

        all_assumptions.update(assumptions)
        all_risks.update(risks)
        all_opportunities.update(opps)

        per_persona_counts.append(len(assumptions) + len(risks) + len(opps))

    total_unique = len(all_assumptions) + len(all_risks) + len(all_opportunities)
    total_all = sum(per_persona_counts)

    if total_all == 0:
        return 0.0

    # Diversity is ratio of unique to total
    # Higher ratio means less overlap (more diverse)
    diversity = total_unique / max(total_all, 1)

    # Compare worldview deltas
    deltas = compare_all_pairs(responses)
    if deltas:
        # More tensions = more diversity
        avg_tensions = sum(len(d.tensions) for d in deltas) / len(deltas)
        tension_factor = min(1.0, avg_tensions * 0.2)
        diversity = (diversity + tension_factor) / 2

    return min(1.0, diversity)


def find_consensus_points(responses: list[PersonaResponse]) -> list[str]:
    """Find points where multiple personas agree.

    Args:
        responses: List of persona responses

    Returns:
        List of consensus points
    """
    if len(responses) < 2:
        return []

    # Count occurrences of each risk and opportunity
    risk_counts: dict[str, int] = {}
    opp_counts: dict[str, int] = {}

    for response in responses:
        for risk in extract_risks(response):
            # Normalize for comparison
            key = risk.lower().strip()
            risk_counts[key] = risk_counts.get(key, 0) + 1

        for opp in extract_opportunities(response):
            key = opp.lower().strip()
            opp_counts[key] = opp_counts.get(key, 0) + 1

    # Find items mentioned by multiple personas
    threshold = max(2, len(responses) // 2)  # At least 2 or half of personas

    consensus: list[str] = []

    for risk, count in risk_counts.items():
        if count >= threshold:
            consensus.append(f"Risk consensus ({count}/{len(responses)}): {risk}")

    for opp, count in opp_counts.items():
        if count >= threshold:
            consensus.append(f"Opportunity consensus ({count}/{len(responses)}): {opp}")

    return consensus


def find_unique_insights(
    responses: list[PersonaResponse],
    library: Optional[PersonaLibrary] = None,
) -> dict[str, list[str]]:
    """Find insights that only one persona identified.

    Args:
        responses: List of persona responses
        library: PersonaLibrary for persona names

    Returns:
        Map of persona_id to their unique insights
    """
    if library is None:
        library = PersonaLibrary()

    # Collect all items by persona
    persona_items: dict[str, set[str]] = {}

    for response in responses:
        items: set[str] = set()
        items.update(extract_risks(response))
        items.update(extract_opportunities(response))
        items.update(extract_assumptions(response))
        persona_items[response.persona_id] = items

    # Find what's unique to each
    unique_insights: dict[str, list[str]] = {}

    for persona_id, items in persona_items.items():
        # Get items from all other personas
        other_items: set[str] = set()
        for other_id, other_set in persona_items.items():
            if other_id != persona_id:
                other_items.update(other_set)

        # Find unique items
        unique = items - other_items
        if unique:
            unique_insights[persona_id] = list(unique)

    return unique_insights


# ============================================================================
# Worldview Summary
# ============================================================================


def summarize_worldview(response: PersonaResponse) -> str:
    """Create a summary of a persona's worldview from their response.

    Args:
        response: The persona's response

    Returns:
        Summary string
    """
    assumptions = extract_assumptions(response)
    risks = extract_risks(response)
    opportunities = extract_opportunities(response)

    parts = [f"Persona: {response.persona_id}"]

    if assumptions:
        parts.append(f"\nKey Assumptions ({len(assumptions)}):")
        for a in assumptions[:3]:
            parts.append(f"  - {a[:100]}...")

    if risks:
        parts.append(f"\nRisks Identified ({len(risks)}):")
        for r in risks[:3]:
            parts.append(f"  - {r[:100]}...")

    if opportunities:
        parts.append(f"\nOpportunities ({len(opportunities)}):")
        for o in opportunities[:3]:
            parts.append(f"  - {o[:100]}...")

    return "\n".join(parts)
