"""Persona Consistency Checking - Ensuring personas stay in character.

Provides tools to detect when a persona drifts from its defined identity,
score consistency, and reinforce persona characteristics.
"""

import re
from typing import Optional

from .library import PersonaLibrary
from .types import (
    ConsistencyScore,
    Persona,
    PersonaResponse,
    ThinkingStyle,
)


# ============================================================================
# Consistency Markers
# ============================================================================

# Phrases that indicate specific thinking styles
THINKING_STYLE_MARKERS: dict[ThinkingStyle, list[str]] = {
    ThinkingStyle.CAUTIOUS: [
        "risk",
        "danger",
        "careful",
        "warning",
        "concern",
        "potential issue",
        "might fail",
        "could go wrong",
        "be careful",
        "watch out",
    ],
    ThinkingStyle.EXPLORATORY: [
        "opportunity",
        "possibility",
        "could",
        "might",
        "explore",
        "imagine",
        "what if",
        "consider",
        "potential",
        "exciting",
    ],
    ThinkingStyle.SYSTEMATIC: [
        "first",
        "then",
        "next",
        "step",
        "process",
        "framework",
        "methodology",
        "structured",
        "systematic",
        "procedure",
    ],
    ThinkingStyle.INTUITIVE: [
        "feel",
        "sense",
        "experience",
        "pattern",
        "reminds me",
        "similar to",
        "historically",
        "typically",
        "usually",
        "in my experience",
    ],
    ThinkingStyle.QUESTIONING: [
        "why",
        "how",
        "what if",
        "assume",
        "question",
        "challenge",
        "really",
        "actually",
        "but",
        "however",
    ],
    ThinkingStyle.CREATIVE: [
        "novel",
        "different",
        "alternative",
        "unconventional",
        "innovative",
        "radical",
        "reimagine",
        "transform",
        "breakthrough",
        "paradigm",
    ],
}

# Persona-specific markers
PERSONA_MARKERS: dict[str, list[str]] = {
    "pessimist": [
        "will fail",
        "won't work",
        "problem",
        "issue",
        "concern",
        "worst case",
        "disaster",
        "risk",
        "danger",
        "mistake",
    ],
    "optimist": [
        "opportunity",
        "exciting",
        "potential",
        "positive",
        "benefit",
        "advantage",
        "success",
        "growth",
        "improve",
        "possible",
    ],
    "security_engineer": [
        "vulnerability",
        "attack",
        "exploit",
        "threat",
        "security",
        "breach",
        "malicious",
        "compromise",
        "risk",
        "exposure",
    ],
    "first_principles": [
        "fundamental",
        "basic",
        "assumption",
        "why",
        "root",
        "core",
        "principle",
        "underlying",
        "essential",
        "derive",
    ],
    "novice": [
        "confused",
        "understand",
        "explain",
        "what does",
        "mean",
        "how does",
        "why",
        "clarify",
        "simple",
        "basic",
    ],
    "veteran": [
        "experience",
        "seen",
        "before",
        "historically",
        "pattern",
        "reminds me",
        "learned",
        "mistake",
        "over the years",
        "typical",
    ],
    "adversary": [
        "exploit",
        "defeat",
        "attack",
        "undermine",
        "weakness",
        "vulnerability",
        "sabotage",
        "circumvent",
        "bypass",
        "counter",
    ],
    "customer": [
        "user",
        "experience",
        "frustrating",
        "confusing",
        "easy",
        "intuitive",
        "want",
        "need",
        "helpful",
        "usability",
    ],
    "regulator": [
        "compliance",
        "regulation",
        "legal",
        "policy",
        "requirement",
        "audit",
        "documentation",
        "liability",
        "governance",
        "standard",
    ],
    "innovator": [
        "radical",
        "different",
        "novel",
        "breakthrough",
        "transform",
        "reimagine",
        "disrupt",
        "unconventional",
        "paradigm",
        "revolutionary",
    ],
}

# Phrases that break character for certain personas
CHARACTER_BREAKS: dict[str, list[str]] = {
    "pessimist": [
        "this is exciting",
        "great opportunity",
        "will definitely work",
        "no concerns",
        "nothing could go wrong",
    ],
    "optimist": [
        "this will fail",
        "no hope",
        "doomed",
        "impossible",
        "give up",
    ],
    "security_engineer": [
        "trust everyone",
        "security doesn't matter",
        "no need to worry",
        "completely safe",
    ],
    "novice": [
        "obviously",
        "as everyone knows",
        "clearly",
        "it's simple",
        "any expert would",
    ],
    "veteran": [
        "i've never seen",
        "completely new to me",
        "no prior experience",
    ],
}


# ============================================================================
# Consistency Checking
# ============================================================================


def check_consistency(
    persona: Persona,
    response: PersonaResponse,
    strict: bool = False,
) -> ConsistencyScore:
    """Check if a response is consistent with a persona's identity.

    Analyzes the response content for:
    - Presence of thinking style markers
    - Presence of persona-specific markers
    - Absence of character-breaking phrases

    Args:
        persona: The persona definition
        response: The persona's response to check
        strict: If True, apply stricter consistency requirements

    Returns:
        ConsistencyScore with detailed consistency information
    """
    content_lower = response.content.lower()
    inconsistent_elements: list[str] = []
    score = 1.0

    # Check for thinking style markers
    style_markers = THINKING_STYLE_MARKERS.get(persona.thinking_style, [])
    style_marker_count = sum(1 for marker in style_markers if marker in content_lower)

    if style_markers:
        # Expect at least some markers in a meaningful response
        min_expected = 2 if strict else 1
        if style_marker_count < min_expected and len(response.content) > 100:
            score -= 0.2
            inconsistent_elements.append(
                f"Missing thinking style markers for {persona.thinking_style.value} "
                f"(found {style_marker_count}, expected >= {min_expected})"
            )

    # Check for persona-specific markers
    persona_markers = PERSONA_MARKERS.get(persona.id, [])
    persona_marker_count = sum(1 for marker in persona_markers if marker in content_lower)

    if persona_markers:
        min_expected = 3 if strict else 1
        if persona_marker_count < min_expected and len(response.content) > 100:
            score -= 0.2
            inconsistent_elements.append(
                f"Missing persona-specific markers for {persona.name} "
                f"(found {persona_marker_count}, expected >= {min_expected})"
            )

    # Check for character breaks
    breaks = CHARACTER_BREAKS.get(persona.id, [])
    for break_phrase in breaks:
        if break_phrase in content_lower:
            score -= 0.3
            inconsistent_elements.append(f"Character break detected: '{break_phrase}'")

    # Ensure score stays in valid range
    score = max(0.0, min(1.0, score))

    drift_detected = score < 0.7
    reinforcement_needed = score < 0.5

    return ConsistencyScore(
        persona_id=persona.id,
        score=score,
        drift_detected=drift_detected,
        inconsistent_elements=inconsistent_elements,
        reinforcement_needed=reinforcement_needed,
    )


def detect_drift(
    persona: Persona,
    responses_over_time: list[PersonaResponse],
    window_size: int = 3,
) -> ConsistencyScore:
    """Detect character degradation over multiple responses.

    Looks for patterns of decreasing consistency over time,
    indicating the persona is drifting from its identity.

    Args:
        persona: The persona definition
        responses_over_time: Responses in chronological order
        window_size: Number of recent responses to weight more heavily

    Returns:
        ConsistencyScore reflecting drift over time
    """
    if not responses_over_time:
        return ConsistencyScore(persona_id=persona.id, score=1.0)

    # Check each response
    scores: list[float] = []
    all_inconsistent: list[str] = []

    for response in responses_over_time:
        result = check_consistency(persona, response)
        scores.append(result.score)
        all_inconsistent.extend(result.inconsistent_elements)

    # Calculate overall score with recency weighting
    if len(scores) <= window_size:
        # Use simple average for short sequences
        avg_score = sum(scores) / len(scores)
    else:
        # Weight recent responses more heavily
        recent = scores[-window_size:]
        older = scores[:-window_size]
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older) if older else 1.0

        # 70% weight on recent, 30% on older
        avg_score = 0.7 * recent_avg + 0.3 * older_avg

    # Check for declining trend
    drift_detected = False
    if len(scores) >= 3:
        # Compare first third to last third
        third = len(scores) // 3
        first_third_avg = sum(scores[:third]) / third if third > 0 else 1.0
        last_third_avg = sum(scores[-third:]) / third if third > 0 else 1.0

        if first_third_avg - last_third_avg > 0.2:
            drift_detected = True
            all_inconsistent.append(
                f"Declining consistency trend: {first_third_avg:.2f} -> {last_third_avg:.2f}"
            )

    return ConsistencyScore(
        persona_id=persona.id,
        score=avg_score,
        drift_detected=drift_detected or avg_score < 0.7,
        inconsistent_elements=all_inconsistent,
        reinforcement_needed=avg_score < 0.5,
    )


# ============================================================================
# Persona Reinforcement
# ============================================================================


def reinforce_persona(
    persona: Persona,
    context: str,
    issues: Optional[list[str]] = None,
) -> str:
    """Generate a reinforcement prompt to strengthen persona identity.

    Creates a prompt that reminds the persona of their identity
    and addresses any specific consistency issues.

    Args:
        persona: The persona to reinforce
        context: Current conversation context
        issues: Specific consistency issues to address

    Returns:
        Reinforcement prompt to inject
    """
    parts = [
        f"Remember: You are {persona.name}.",
        f"\n\nYour perspective: {persona.description}",
        f"\n\nYour thinking style: {persona.thinking_style.value}",
    ]

    if persona.biases:
        parts.append(f"\n\nYour known biases: {', '.join(persona.biases)}")

    if persona.strengths:
        parts.append(f"\n\nYour strengths: {', '.join(persona.strengths)}")

    if issues:
        parts.append("\n\n--- STAY IN CHARACTER ---")
        parts.append("\nYou have been drifting. Remember to:")
        for issue in issues:
            parts.append(f"\n- Address: {issue}")

    parts.append("\n\n--- CONTINUE YOUR ANALYSIS ---")
    parts.append(f"\n\nContext: {context[:500]}...")  # Truncate long context

    return "".join(parts)


def create_reinforcement_injection(
    persona: Persona,
    consistency_score: ConsistencyScore,
) -> Optional[str]:
    """Create an injection to reinforce persona if needed.

    Args:
        persona: The persona definition
        consistency_score: Recent consistency check result

    Returns:
        Reinforcement text to inject, or None if not needed
    """
    if not consistency_score.reinforcement_needed:
        return None

    parts = [
        "\n\n[SYSTEM: Character reinforcement required]\n",
        f"You are {persona.name}. Stay in character.\n",
    ]

    if consistency_score.inconsistent_elements:
        parts.append("Issues detected:\n")
        for element in consistency_score.inconsistent_elements[:3]:  # Limit to top 3
            parts.append(f"- {element}\n")

    parts.append(f"\nRemember your perspective: {persona.description}\n")
    parts.append("[END SYSTEM]\n")

    return "".join(parts)


# ============================================================================
# Batch Consistency Checking
# ============================================================================


def check_all_responses(
    responses: list[PersonaResponse],
    library: Optional[PersonaLibrary] = None,
) -> dict[str, ConsistencyScore]:
    """Check consistency for all responses in a batch.

    Args:
        responses: List of persona responses
        library: PersonaLibrary to look up persona definitions

    Returns:
        Map of persona_id to ConsistencyScore
    """
    if library is None:
        library = PersonaLibrary()

    results: dict[str, ConsistencyScore] = {}

    for response in responses:
        persona = library.get(response.persona_id)
        if persona:
            results[response.persona_id] = check_consistency(persona, response)
        else:
            # Unknown persona - can't check consistency
            results[response.persona_id] = ConsistencyScore(
                persona_id=response.persona_id,
                score=0.5,
                inconsistent_elements=["Unknown persona - cannot verify consistency"],
            )

    return results


def identify_weakest_personas(
    responses: list[PersonaResponse],
    library: Optional[PersonaLibrary] = None,
    threshold: float = 0.7,
) -> list[tuple[str, ConsistencyScore]]:
    """Identify personas with weakest consistency.

    Args:
        responses: List of persona responses
        library: PersonaLibrary to look up personas
        threshold: Score below which a persona is considered weak

    Returns:
        List of (persona_id, ConsistencyScore) for weak personas, sorted by score
    """
    scores = check_all_responses(responses, library)

    weak = [(pid, score) for pid, score in scores.items() if score.score < threshold]

    # Sort by score ascending (weakest first)
    weak.sort(key=lambda x: x[1].score)

    return weak


# ============================================================================
# Response Filtering
# ============================================================================


def filter_inconsistent_content(
    persona: Persona,
    content: str,
) -> tuple[str, list[str]]:
    """Filter out content that breaks character.

    Attempts to remove or flag content that's inconsistent
    with the persona's identity.

    Args:
        persona: The persona definition
        content: Content to filter

    Returns:
        Tuple of (filtered_content, list of removed/flagged items)
    """
    breaks = CHARACTER_BREAKS.get(persona.id, [])
    removed: list[str] = []
    filtered = content

    for break_phrase in breaks:
        if break_phrase in content.lower():
            # Mark the phrase
            pattern = re.compile(re.escape(break_phrase), re.IGNORECASE)
            matches = pattern.findall(content)
            for match in matches:
                removed.append(match)
            # Replace with [FILTERED]
            filtered = pattern.sub("[FILTERED - out of character]", filtered)

    return filtered, removed
