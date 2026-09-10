"""Type definitions for the Persona Ensemble system.

Dataclasses defining personas, responses, worldview deltas, and performance tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PersonaCategory(Enum):
    """Categories of persona perspectives."""

    RISK = "risk"  # Focus on risks, failures, threats
    OPPORTUNITY = "opportunity"  # Focus on possibilities, gains
    TECHNICAL = "technical"  # Technical/engineering focus
    USER = "user"  # End-user/customer focus
    STRATEGIC = "strategic"  # High-level/business focus
    ANALYTICAL = "analytical"  # First principles, fundamentals


class ThinkingStyle(Enum):
    """How a persona approaches problems."""

    CAUTIOUS = "cautious"  # Careful, risk-averse
    EXPLORATORY = "exploratory"  # Open to possibilities
    SYSTEMATIC = "systematic"  # Methodical, structured
    INTUITIVE = "intuitive"  # Pattern-matching, experience-based
    QUESTIONING = "questioning"  # Challenges assumptions
    CREATIVE = "creative"  # Novel approaches


@dataclass
class Persona:
    """Identity definition for a branch persona.

    Attributes:
        id: Unique identifier for the persona
        name: Human-readable name
        system_prompt: System prompt to inject for this persona
        thinking_style: How this persona approaches problems
        biases: Known biases this persona has (for attribution)
        strengths: What this persona is good at spotting
        category: Which category this persona belongs to
        description: Brief description of the persona
    """

    id: str
    name: str
    system_prompt: str
    thinking_style: ThinkingStyle
    biases: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    category: PersonaCategory = PersonaCategory.ANALYTICAL
    description: str = ""


@dataclass
class PersonaResponse:
    """Response from a persona during exploration.

    Attributes:
        persona_id: Which persona generated this response
        content: The actual response content
        assumptions: Assumptions the persona made (extracted)
        risks_identified: Risks the persona identified
        opportunities: Opportunities the persona identified
        confidence: How confident the persona is in their response
        raw_response: Original unprocessed response
    """

    persona_id: str
    content: str
    assumptions: list[str] = field(default_factory=list)
    risks_identified: list[str] = field(default_factory=list)
    opportunities: list[str] = field(default_factory=list)
    confidence: float = 0.5
    raw_response: str = ""


@dataclass
class WorldviewDelta:
    """Differences between two persona outputs.

    Captures what two personas agree on, where they disagree,
    and genuine tensions in their worldviews.

    Attributes:
        persona_a: ID of first persona
        persona_b: ID of second persona
        agreements: Points both personas agree on
        tensions: Genuine disagreements between personas
        unique_to_a: Insights only persona A had
        unique_to_b: Insights only persona B had
        severity: How significant the disagreements are (0-1)
    """

    persona_a: str
    persona_b: str
    agreements: list[str] = field(default_factory=list)
    tensions: list[str] = field(default_factory=list)
    unique_to_a: list[str] = field(default_factory=list)
    unique_to_b: list[str] = field(default_factory=list)
    severity: float = 0.0


@dataclass
class EnsembleResult:
    """Combined output from all personas in an ensemble.

    Attributes:
        responses: Individual responses from each persona
        synthesis: Combined synthesis of all perspectives
        dominant_themes: Themes that appeared across multiple personas
        blind_spots: Perspectives or considerations that were missing
        worldview_deltas: Pairwise comparisons between personas
        diversity_score: How diverse the responses actually were (0-1)
        attributed_insights: Map of insight to persona(s) who generated it
    """

    responses: list[PersonaResponse] = field(default_factory=list)
    synthesis: str = ""
    dominant_themes: list[str] = field(default_factory=list)
    blind_spots: list[str] = field(default_factory=list)
    worldview_deltas: list[WorldviewDelta] = field(default_factory=list)
    diversity_score: float = 0.0
    attributed_insights: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class PersonaPerformance:
    """Tracking data for persona effectiveness.

    Attributes:
        persona_id: Which persona this tracks
        total_uses: How many times this persona has been used
        success_rate: Fraction of times persona provided useful output
        unique_insights: Count of insights only this persona provided
        redundancy_rate: Fraction of times persona duplicated others
        problem_types: Which problem types this persona excels at
        avg_confidence: Average confidence in responses
        last_used: Timestamp of last use (ISO format)
    """

    persona_id: str
    total_uses: int = 0
    success_rate: float = 0.0
    unique_insights: int = 0
    redundancy_rate: float = 0.0
    problem_types: dict[str, float] = field(default_factory=dict)
    avg_confidence: float = 0.5
    last_used: Optional[str] = None


@dataclass
class ConsistencyScore:
    """Score for how consistent a persona stayed in character.

    Attributes:
        persona_id: Which persona was scored
        score: Overall consistency score (0-1)
        drift_detected: Whether character drift was detected
        inconsistent_elements: Specific elements that broke character
        reinforcement_needed: Whether persona needs reinforcement
    """

    persona_id: str
    score: float = 1.0
    drift_detected: bool = False
    inconsistent_elements: list[str] = field(default_factory=list)
    reinforcement_needed: bool = False


@dataclass
class AssignmentResult:
    """Result of persona assignment to branches.

    Attributes:
        assignments: Map of branch_id to persona
        coverage: Which categories are covered
        missing_categories: Categories not covered
        rationale: Why these personas were chosen
    """

    assignments: dict[str, Persona] = field(default_factory=dict)
    coverage: list[PersonaCategory] = field(default_factory=list)
    missing_categories: list[PersonaCategory] = field(default_factory=list)
    rationale: str = ""
