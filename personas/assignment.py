"""Persona Assignment - Selecting and assigning personas to branches.

Handles intelligent selection of personas based on problem type,
detection of missing perspectives, and dynamic spawning.
"""

import random
from typing import Optional

from .library import BUILTIN_PERSONAS, PERSONAS_BY_CATEGORY, PersonaLibrary
from .types import (
    AssignmentResult,
    Persona,
    PersonaCategory,
    PersonaResponse,
)


# ============================================================================
# Problem Type Detection
# ============================================================================

# Keywords that suggest certain categories should be prioritized
CATEGORY_KEYWORDS: dict[PersonaCategory, list[str]] = {
    PersonaCategory.RISK: [
        "risk",
        "danger",
        "threat",
        "security",
        "vulnerability",
        "fail",
        "problem",
        "issue",
        "concern",
        "attack",
    ],
    PersonaCategory.OPPORTUNITY: [
        "opportunity",
        "growth",
        "expand",
        "improve",
        "innovation",
        "new",
        "potential",
        "possibility",
        "advantage",
    ],
    PersonaCategory.TECHNICAL: [
        "technical",
        "code",
        "system",
        "architecture",
        "implementation",
        "engineering",
        "performance",
        "scalability",
        "infrastructure",
    ],
    PersonaCategory.USER: [
        "user",
        "customer",
        "experience",
        "usability",
        "interface",
        "adoption",
        "onboarding",
        "feedback",
        "satisfaction",
    ],
    PersonaCategory.STRATEGIC: [
        "strategy",
        "business",
        "market",
        "competition",
        "regulatory",
        "compliance",
        "policy",
        "governance",
        "long-term",
    ],
    PersonaCategory.ANALYTICAL: [
        "analyze",
        "understand",
        "why",
        "how",
        "cause",
        "root",
        "fundamental",
        "principle",
        "assumption",
    ],
}


def detect_relevant_categories(problem: str) -> list[PersonaCategory]:
    """Detect which persona categories are most relevant to a problem.

    Args:
        problem: The problem description

    Returns:
        List of relevant categories, ordered by relevance
    """
    problem_lower = problem.lower()
    scores: dict[PersonaCategory, int] = {cat: 0 for cat in PersonaCategory}

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in problem_lower:
                scores[category] += 1

    # Sort by score descending, then by category name for stability
    sorted_categories = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0].value),
    )

    # Return categories with non-zero scores, or all if none matched
    relevant = [cat for cat, score in sorted_categories if score > 0]
    if not relevant:
        # Return all categories if no keywords matched
        return list(PersonaCategory)

    return relevant


# ============================================================================
# Persona Assignment
# ============================================================================


def assign_personas(
    problem: str,
    num_branches: int,
    library: Optional[PersonaLibrary] = None,
    required_personas: Optional[list[str]] = None,
    excluded_personas: Optional[list[str]] = None,
    ensure_category_coverage: bool = True,
) -> AssignmentResult:
    """Select appropriate personas for a problem.

    Intelligently selects personas based on:
    - Problem content analysis
    - Category coverage requirements
    - Required/excluded constraints

    Args:
        problem: The problem description
        num_branches: Number of branches to assign personas to
        library: PersonaLibrary to select from (uses defaults if None)
        required_personas: Persona IDs that must be included
        excluded_personas: Persona IDs to exclude
        ensure_category_coverage: Try to cover all categories

    Returns:
        AssignmentResult with assignments and coverage info
    """
    if library is None:
        library = PersonaLibrary()

    required = required_personas or []
    excluded = set(excluded_personas or [])

    # Get available personas (excluding excluded)
    available = [p for p in library.list_all() if p.id not in excluded]

    if not available:
        return AssignmentResult(
            assignments={},
            coverage=[],
            missing_categories=list(PersonaCategory),
            rationale="No personas available after exclusions",
        )

    # Start with required personas
    selected: list[Persona] = []
    for persona_id in required:
        persona = library.get(persona_id)
        if persona and persona.id not in excluded:
            selected.append(persona)

    # Detect relevant categories
    relevant_categories = detect_relevant_categories(problem)

    # Fill remaining slots
    remaining_slots = num_branches - len(selected)
    selected_ids = {p.id for p in selected}

    if ensure_category_coverage and remaining_slots > 0:
        # Try to cover each relevant category
        for category in relevant_categories:
            if remaining_slots <= 0:
                break

            # Find a persona from this category not yet selected
            category_personas = [
                p for p in library.list_by_category(category) if p.id not in selected_ids and p.id not in excluded
            ]

            if category_personas:
                # Pick the first one (could be random or scored)
                persona = category_personas[0]
                selected.append(persona)
                selected_ids.add(persona.id)
                remaining_slots -= 1

    # Fill any remaining slots with diverse personas
    while remaining_slots > 0:
        unselected = [p for p in available if p.id not in selected_ids]
        if not unselected:
            break

        # Pick randomly from unselected
        persona = random.choice(unselected)
        selected.append(persona)
        selected_ids.add(persona.id)
        remaining_slots -= 1

    # Create assignments (branch_0, branch_1, etc.)
    assignments = {f"branch_{i}": persona for i, persona in enumerate(selected)}

    # Calculate coverage
    covered_categories = list(set(p.category for p in selected))
    all_categories = set(PersonaCategory)
    missing_categories = list(all_categories - set(covered_categories))

    # Generate rationale
    rationale_parts = []
    if required:
        rationale_parts.append(f"Required: {', '.join(required)}")
    if relevant_categories:
        rationale_parts.append(f"Relevant categories: {', '.join(c.value for c in relevant_categories[:3])}")
    if missing_categories:
        rationale_parts.append(f"Missing coverage: {', '.join(c.value for c in missing_categories)}")

    return AssignmentResult(
        assignments=assignments,
        coverage=covered_categories,
        missing_categories=missing_categories,
        rationale="; ".join(rationale_parts) if rationale_parts else "Default selection",
    )


# ============================================================================
# Missing Perspective Detection
# ============================================================================


def detect_missing_perspective(
    responses: list[PersonaResponse],
    library: Optional[PersonaLibrary] = None,
) -> Optional[PersonaCategory]:
    """Detect what perspective is missing from current responses.

    Analyzes the categories covered by current responses and identifies
    which important perspective is not represented.

    Args:
        responses: Current persona responses
        library: PersonaLibrary to reference

    Returns:
        Missing PersonaCategory, or None if well-covered
    """
    if library is None:
        library = PersonaLibrary()

    # Get categories of responding personas
    covered_categories: set[PersonaCategory] = set()
    for response in responses:
        persona = library.get(response.persona_id)
        if persona:
            covered_categories.add(persona.category)

    # Check what's missing
    all_categories = set(PersonaCategory)
    missing = all_categories - covered_categories

    if not missing:
        return None

    # Prioritize certain categories as more important to cover
    priority_order = [
        PersonaCategory.RISK,  # Always want risk perspective
        PersonaCategory.USER,  # User perspective is crucial
        PersonaCategory.OPPORTUNITY,  # Balance risk with opportunity
        PersonaCategory.TECHNICAL,  # Technical feasibility
        PersonaCategory.STRATEGIC,  # Strategic view
        PersonaCategory.ANALYTICAL,  # Deep analysis
    ]

    for category in priority_order:
        if category in missing:
            return category

    # Return any missing category
    return list(missing)[0] if missing else None


def detect_all_missing_perspectives(
    responses: list[PersonaResponse],
    library: Optional[PersonaLibrary] = None,
) -> list[PersonaCategory]:
    """Detect all perspectives missing from current responses.

    Args:
        responses: Current persona responses
        library: PersonaLibrary to reference

    Returns:
        List of all missing PersonaCategories
    """
    if library is None:
        library = PersonaLibrary()

    covered_categories: set[PersonaCategory] = set()
    for response in responses:
        persona = library.get(response.persona_id)
        if persona:
            covered_categories.add(persona.category)

    all_categories = set(PersonaCategory)
    return list(all_categories - covered_categories)


# ============================================================================
# Dynamic Spawning
# ============================================================================


def dynamic_spawn(
    missing_perspective: PersonaCategory,
    library: Optional[PersonaLibrary] = None,
    excluded_personas: Optional[list[str]] = None,
) -> Optional[Persona]:
    """Create a new branch for a missing perspective.

    Selects a persona from the missing category to spawn a new branch.

    Args:
        missing_perspective: The category that's missing
        library: PersonaLibrary to select from
        excluded_personas: Personas already in use

    Returns:
        Persona to spawn, or None if none available
    """
    if library is None:
        library = PersonaLibrary()

    excluded = set(excluded_personas or [])

    # Get personas in the missing category
    category_personas = [p for p in library.list_by_category(missing_perspective) if p.id not in excluded]

    if not category_personas:
        return None

    # Return first available (could be random or scored)
    return category_personas[0]


def suggest_additional_personas(
    problem: str,
    current_personas: list[str],
    max_suggestions: int = 3,
    library: Optional[PersonaLibrary] = None,
) -> list[Persona]:
    """Suggest additional personas that could add value.

    Based on problem analysis and current coverage, suggest
    personas that could provide unique perspectives.

    Args:
        problem: The problem description
        current_personas: IDs of currently assigned personas
        max_suggestions: Maximum number of suggestions
        library: PersonaLibrary to select from

    Returns:
        List of suggested personas
    """
    if library is None:
        library = PersonaLibrary()

    current_set = set(current_personas)
    suggestions: list[Persona] = []

    # Get current coverage
    current_categories: set[PersonaCategory] = set()
    for persona_id in current_personas:
        persona = library.get(persona_id)
        if persona:
            current_categories.add(persona.category)

    # Detect relevant categories for the problem
    relevant = detect_relevant_categories(problem)

    # Suggest from relevant but uncovered categories
    for category in relevant:
        if len(suggestions) >= max_suggestions:
            break

        if category not in current_categories:
            category_personas = [p for p in library.list_by_category(category) if p.id not in current_set]
            if category_personas:
                suggestions.append(category_personas[0])
                current_categories.add(category)

    # If still room, suggest contrasting personas
    if len(suggestions) < max_suggestions:
        # If we have pessimist, suggest optimist (and vice versa)
        contrasts = [
            ("pessimist", "optimist"),
            ("optimist", "pessimist"),
            ("novice", "veteran"),
            ("veteran", "novice"),
        ]
        for has, needs in contrasts:
            if len(suggestions) >= max_suggestions:
                break
            if has in current_set and needs not in current_set:
                persona = library.get(needs)
                if persona:
                    suggestions.append(persona)

    return suggestions


# ============================================================================
# Balanced Assignment
# ============================================================================


def assign_balanced(
    num_branches: int,
    library: Optional[PersonaLibrary] = None,
) -> AssignmentResult:
    """Assign personas with balanced category coverage.

    Ensures even distribution across categories regardless of problem.

    Args:
        num_branches: Number of branches to assign
        library: PersonaLibrary to select from

    Returns:
        AssignmentResult with balanced assignments
    """
    if library is None:
        library = PersonaLibrary()

    selected: list[Persona] = []
    selected_ids: set[str] = set()

    # Get all categories with personas
    categories = library.categories()
    category_index = 0

    # Round-robin through categories
    while len(selected) < num_branches:
        if not categories:
            break

        category = categories[category_index % len(categories)]
        category_personas = [p for p in library.list_by_category(category) if p.id not in selected_ids]

        if category_personas:
            persona = category_personas[0]
            selected.append(persona)
            selected_ids.add(persona.id)

        category_index += 1

        # Avoid infinite loop if we've exhausted all personas
        if category_index >= len(categories) * len(library):
            break

    assignments = {f"branch_{i}": persona for i, persona in enumerate(selected)}
    covered = list(set(p.category for p in selected))
    missing = list(set(PersonaCategory) - set(covered))

    return AssignmentResult(
        assignments=assignments,
        coverage=covered,
        missing_categories=missing,
        rationale="Balanced category coverage",
    )
