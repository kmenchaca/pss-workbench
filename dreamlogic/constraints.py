"""
Dream Logic Generator - Constraint Randomizer

Generates and applies random creative constraints to force unexpected directions.
"""

import random
from typing import Optional

from .types import Constraint, ConstraintType


# Constraint pools organized by type and difficulty
CONSTRAINT_POOLS: dict[ConstraintType, list[tuple[str, float]]] = {
    ConstraintType.MUST_INCLUDE: [
        # Easy (0.2-0.4)
        ("a mirror", 0.3),
        ("water in some form", 0.25),
        ("the number three", 0.3),
        ("something that glows", 0.35),
        ("a threshold or doorway", 0.3),
        # Medium (0.4-0.6)
        ("teeth that don't belong to a mouth", 0.5),
        ("a sound that has a color", 0.55),
        ("something that remembers", 0.45),
        ("gravity behaving wrong", 0.5),
        ("a message in an impossible language", 0.55),
        # Hard (0.6-0.8)
        ("the taste of a memory", 0.7),
        ("something that exists only in peripheral vision", 0.75),
        ("a conversation between two abstractions", 0.7),
        ("the moment between moments", 0.8),
        ("something that is simultaneously two things", 0.75),
    ],
    ConstraintType.MUST_EXCLUDE: [
        # Easy
        ("any mention of the sun", 0.25),
        ("the color blue", 0.3),
        ("human characters", 0.4),
        ("modern technology", 0.35),
        ("named locations", 0.3),
        # Medium
        ("linear time progression", 0.5),
        ("cause and effect relationships", 0.55),
        ("objects with clear boundaries", 0.5),
        ("emotions that have names", 0.55),
        ("physical laws working normally", 0.6),
        # Hard
        ("anything that makes sense", 0.75),
        ("any stable identity", 0.7),
        ("spatial consistency", 0.75),
        ("the ability to distinguish self from other", 0.8),
    ],
    ConstraintType.STYLE: [
        # Easy
        ("write as if underwater", 0.3),
        ("write as if very old", 0.25),
        ("write as if half-asleep", 0.35),
        ("write in fragments", 0.3),
        # Medium
        ("write as if the words themselves are uncertain", 0.5),
        ("write as if describing a photograph that keeps changing", 0.55),
        ("write as if translating from a language that doesn't exist", 0.5),
        ("write as if the narrator is forgetting mid-sentence", 0.6),
        # Hard
        ("write as if the medium is actively resisting meaning", 0.75),
        ("write as if words have mass and fall", 0.7),
        ("write as if grammar is a suggestion from another dimension", 0.75),
    ],
    ConstraintType.PERSPECTIVE: [
        # Easy
        ("from the viewpoint of an animal", 0.25),
        ("from far above", 0.3),
        ("from the perspective of a child", 0.3),
        ("from inside looking out", 0.35),
        # Medium
        ("from the viewpoint of an inanimate object", 0.5),
        ("from the perspective of time itself", 0.55),
        ("from the viewpoint of someone who doesn't exist yet", 0.55),
        ("from the perspective of a collective", 0.5),
        # Hard
        ("from the viewpoint of an abstract concept", 0.7),
        ("from nowhere and everywhere simultaneously", 0.75),
        ("from the perspective of the space between things", 0.8),
        ("from the viewpoint of pure potential", 0.75),
    ],
    ConstraintType.SCALE: [
        # Easy
        ("everything is giant", 0.25),
        ("everything is tiny", 0.25),
        ("human scale but stretched", 0.35),
        # Medium
        ("at the scale of cells and microbes", 0.5),
        ("at the scale of continents", 0.45),
        ("scale keeps shifting unpredictably", 0.55),
        ("at the scale of emotions", 0.5),
        # Hard
        ("at the scale of atoms", 0.65),
        ("at the scale of galaxies", 0.65),
        ("at the scale of probability waves", 0.75),
        ("at the scale of consciousness itself", 0.8),
    ],
    ConstraintType.TIME: [
        # Easy
        ("set in deep past", 0.25),
        ("set in far future", 0.3),
        ("at twilight", 0.25),
        ("during a storm", 0.3),
        # Medium
        ("before time existed", 0.5),
        ("after everything ends", 0.55),
        ("during a moment stretched infinitely", 0.55),
        ("in all times simultaneously", 0.6),
        # Hard
        ("in the moment of forgetting", 0.7),
        ("in the pause between heartbeats of the universe", 0.75),
        ("in the time that dreams happen", 0.65),
        ("at the precise instant of transformation", 0.7),
    ],
}


def random_constraint(
    difficulty_range: tuple[float, float] = (0.0, 1.0),
    constraint_type: Optional[ConstraintType] = None
) -> Constraint:
    """
    Generate a random creative constraint.

    Args:
        difficulty_range: Tuple of (min, max) difficulty to filter by.
        constraint_type: Optional specific type (random if not provided).

    Returns:
        A random constraint within the specified parameters.
    """
    # Select constraint type
    if constraint_type is None:
        constraint_type = random.choice(list(ConstraintType))

    # Get pool for this type
    pool = CONSTRAINT_POOLS[constraint_type]

    # Filter by difficulty range
    min_diff, max_diff = difficulty_range
    filtered = [(value, diff) for value, diff in pool
                if min_diff <= diff <= max_diff]

    # Fall back to full pool if nothing matches
    if not filtered:
        filtered = pool

    # Select random constraint
    value, difficulty = random.choice(filtered)

    return Constraint(
        type=constraint_type,
        value=value,
        difficulty=difficulty
    )


def chain_constraints(
    n: int,
    escalate_difficulty: bool = True,
    avoid_types: Optional[list[ConstraintType]] = None
) -> list[Constraint]:
    """
    Generate multiple random constraints, optionally with escalating difficulty.

    Args:
        n: Number of constraints to generate.
        escalate_difficulty: If True, each constraint is harder than the last.
        avoid_types: Types to exclude from generation.

    Returns:
        List of n constraints.
    """
    constraints = []
    available_types = list(ConstraintType)

    if avoid_types:
        available_types = [t for t in available_types if t not in avoid_types]

    if not available_types:
        available_types = list(ConstraintType)

    for i in range(n):
        if escalate_difficulty:
            # Start easy, get harder
            min_diff = i * (0.8 / n)
            max_diff = min(1.0, (i + 1) * (0.8 / n) + 0.3)
            diff_range = (min_diff, max_diff)
        else:
            diff_range = (0.0, 1.0)

        # Rotate through types to ensure variety
        constraint_type = available_types[i % len(available_types)]

        constraint = random_constraint(
            difficulty_range=diff_range,
            constraint_type=constraint_type
        )
        constraints.append(constraint)

    return constraints


def apply_constraint(content: str, constraint: Constraint) -> str:
    """
    Apply a constraint to content, adding directive or modifying it.

    Args:
        content: Original content.
        constraint: Constraint to apply.

    Returns:
        Modified content with constraint applied.
    """
    directives = {
        ConstraintType.MUST_INCLUDE: f"[Must include: {constraint.value}]\n\n{content}",
        ConstraintType.MUST_EXCLUDE: f"[Must exclude: {constraint.value}]\n\n{content}",
        ConstraintType.STYLE: f"[Style: {constraint.value}]\n\n{content}",
        ConstraintType.PERSPECTIVE: f"[Perspective: {constraint.value}]\n\n{content}",
        ConstraintType.SCALE: f"[Scale: {constraint.value}]\n\n{content}",
        ConstraintType.TIME: f"[Time: {constraint.value}]\n\n{content}",
    }

    return directives.get(constraint.type, content)


def constraint_to_prompt(constraint: Constraint) -> str:
    """
    Convert a constraint to a prompt instruction.

    Args:
        constraint: The constraint to convert.

    Returns:
        Prompt-ready instruction string.
    """
    templates = {
        ConstraintType.MUST_INCLUDE: "You must include {value} somewhere in your response.",
        ConstraintType.MUST_EXCLUDE: "You must not include or reference {value}.",
        ConstraintType.STYLE: "Write in this style: {value}.",
        ConstraintType.PERSPECTIVE: "Write {value}.",
        ConstraintType.SCALE: "Set everything {value}.",
        ConstraintType.TIME: "Set this {value}.",
    }

    template = templates.get(constraint.type, "{value}")
    return template.format(value=constraint.value)


def constraints_to_prompt(constraints: list[Constraint]) -> str:
    """
    Convert multiple constraints to a combined prompt.

    Args:
        constraints: List of constraints.

    Returns:
        Combined prompt instruction.
    """
    if not constraints:
        return ""

    instructions = [constraint_to_prompt(c) for c in constraints]
    return "CONSTRAINTS:\n" + "\n".join(f"- {i}" for i in instructions)


def check_constraint_satisfied(content: str, constraint: Constraint) -> bool:
    """
    Check if content satisfies a constraint.

    Args:
        content: Content to check.
        constraint: Constraint to verify.

    Returns:
        True if constraint appears satisfied.
    """
    content_lower = content.lower()
    value_lower = constraint.value.lower()

    if constraint.type == ConstraintType.MUST_INCLUDE:
        # Check if the key concept is present
        key_words = value_lower.split()
        # At least half of the key words should be present
        matches = sum(1 for word in key_words if word in content_lower)
        return matches >= len(key_words) / 2

    elif constraint.type == ConstraintType.MUST_EXCLUDE:
        # Check that the key concept is NOT present
        key_words = value_lower.split()
        # None of the significant key words should be present
        significant = [w for w in key_words if len(w) > 3]
        if significant:
            return not any(word in content_lower for word in significant)
        return True

    # For other constraint types, we can't easily verify programmatically
    # Assume satisfied unless obviously violated
    return True


def constraint_difficulty_level(difficulty: float) -> str:
    """
    Get a human-readable difficulty level name.

    Args:
        difficulty: Difficulty score from 0 to 1.

    Returns:
        Level name string.
    """
    if difficulty < 0.3:
        return "easy"
    elif difficulty < 0.5:
        return "moderate"
    elif difficulty < 0.7:
        return "challenging"
    else:
        return "extreme"


def get_random_by_difficulty(level: str) -> Constraint:
    """
    Get a random constraint at a specified difficulty level.

    Args:
        level: One of "easy", "moderate", "challenging", "extreme".

    Returns:
        A constraint at that difficulty level.
    """
    ranges = {
        "easy": (0.0, 0.35),
        "moderate": (0.35, 0.55),
        "challenging": (0.55, 0.75),
        "extreme": (0.75, 1.0),
    }

    diff_range = ranges.get(level, (0.0, 1.0))
    return random_constraint(difficulty_range=diff_range)


def all_constraint_types() -> list[ConstraintType]:
    """Return all available constraint types."""
    return list(ConstraintType)


def describe_constraint_type(constraint_type: ConstraintType) -> str:
    """
    Get a description of what a constraint type does.

    Args:
        constraint_type: The type to describe.

    Returns:
        Human-readable description.
    """
    descriptions = {
        ConstraintType.MUST_INCLUDE: "Forces inclusion of specific elements or concepts",
        ConstraintType.MUST_EXCLUDE: "Forbids certain elements, forcing creative alternatives",
        ConstraintType.STYLE: "Imposes a writing style or voice",
        ConstraintType.PERSPECTIVE: "Changes the viewpoint or observer",
        ConstraintType.SCALE: "Alters the size/scope of the setting",
        ConstraintType.TIME: "Modifies when/how time operates",
    }
    return descriptions.get(constraint_type, "Unknown constraint type")
