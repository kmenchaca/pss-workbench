"""
Dream Logic Generator - Associative Leap Generation

Creates conceptual jumps between ideas with scoring and constraints.
"""

import random
from dataclasses import dataclass
from typing import Callable, Optional

from .types import AssociativeLeap, Constraint, ConstraintType, DreamBranch, LeapType
from .associations import AssociationDB, surprising_association


# Leap generation templates by type
LEAP_TEMPLATES: dict[LeapType, list[str]] = {
    LeapType.METAPHOR: [
        "{concept} is actually {modifier} disguised as itself",
        "What if {concept} were made of {material}",
        "{concept} bleeds {substance} when you're not looking",
        "The secret name of {concept} is {name}",
    ],
    LeapType.INVERSION: [
        "The opposite of {concept} is not what you think",
        "{concept} but it's running backwards through time",
        "What {concept} looks like from the inside out",
        "The absence of {concept} has a shape",
    ],
    LeapType.SCALE_SHIFT: [
        "{concept} at the scale of atoms dancing",
        "If {concept} were the size of a universe",
        "The microscopic civilization living inside {concept}",
        "{concept} seen from a thousand light years away",
    ],
    LeapType.TIME_WARP: [
        "{concept} remembering what it was before it existed",
        "The future fossil of {concept}",
        "{concept} in the moment before the big bang",
        "What {concept} will dream when everyone forgets it",
    ],
    LeapType.PERSONIFICATION: [
        "If {concept} had a childhood, it was {experience}",
        "{concept} falls in love with {object}",
        "The secret fears of {concept}",
        "{concept} writing a letter to its creator",
    ],
    LeapType.ABSTRACTION: [
        "The emotion that {concept} represents",
        "If you could taste {concept}, it would be",
        "The sound {concept} makes in a vacuum",
        "The mathematical formula for {concept}'s loneliness",
    ],
}

# Modifiers for filling in templates
MODIFIERS = ["liquid", "crystallized", "inverted", "ancient", "unborn", "forbidden"]
MATERIALS = ["forgotten memories", "compressed time", "solidified dreams", "frozen music"]
SUBSTANCES = ["possibilities", "regret", "tomorrow", "silence", "questions"]
NAMES = ["the whisper before dawn", "entropy's cousin", "the color of absence"]
EXPERIENCES = ["full of static", "underwater and luminous", "made of goodbyes"]
OBJECTS = ["its own reflection", "the concept of distance", "an impossible color"]

# Constraints for injection
RANDOM_CONSTRAINTS: list[tuple[ConstraintType, list[str], float]] = [
    (ConstraintType.MUST_INCLUDE, [
        "a mirror", "water", "a door that shouldn't be there",
        "the number seven", "teeth", "something that glows"
    ], 0.3),
    (ConstraintType.MUST_EXCLUDE, [
        "any mention of the sun", "linear time", "clear answers",
        "the color blue", "gravity working normally"
    ], 0.4),
    (ConstraintType.PERSPECTIVE, [
        "from the viewpoint of an inanimate object",
        "as observed by someone who doesn't exist yet",
        "from inside a memory of a memory",
        "as understood by something that cannot see"
    ], 0.5),
    (ConstraintType.SCALE, [
        "everything is microscopic",
        "cosmic scale only",
        "at the scale of emotions",
        "measured in heartbeats"
    ], 0.4),
    (ConstraintType.TIME, [
        "before time existed",
        "after everything ends",
        "in the pause between seconds",
        "during the moment of forgetting"
    ], 0.5),
]


@dataclass
class LeapContext:
    """Context for generating leaps."""
    current_concept: str
    history: list[str]
    style_hints: list[str]
    minimum_distance: float = 0.3


def generate_leap(
    context: LeapContext,
    db: Optional[AssociationDB] = None,
    llm_generator: Optional[Callable[[str], str]] = None
) -> AssociativeLeap:
    """
    Generate a conceptual jump from the current context.

    Args:
        context: Current context including concept and history.
        db: Optional association database.
        llm_generator: Optional LLM for generating creative leaps.

    Returns:
        An associative leap from the current concept.
    """
    if db is None:
        db = AssociationDB()

    # First try to get a surprising association from the database
    assoc = surprising_association(context.current_concept, db)

    if assoc and assoc.surprise_factor >= context.minimum_distance:
        return AssociativeLeap(
            source=assoc.source,
            target=assoc.target,
            leap_type=assoc.leap_type,
            distance=assoc.surprise_factor
        )

    # Otherwise generate a new leap
    leap_type = random.choice(list(LeapType))
    templates = LEAP_TEMPLATES[leap_type]
    template = random.choice(templates)

    # Fill in the template
    target = _fill_template(template, context.current_concept)

    # If we have an LLM, use it to enhance
    if llm_generator:
        try:
            enhanced = llm_generator(f"Transform this into something stranger: {target}")
            target = enhanced
        except Exception:
            pass  # Fall back to template version

    # Calculate distance based on leap type
    base_distances = {
        LeapType.METAPHOR: 0.4,
        LeapType.INVERSION: 0.6,
        LeapType.SCALE_SHIFT: 0.7,
        LeapType.TIME_WARP: 0.8,
        LeapType.PERSONIFICATION: 0.5,
        LeapType.ABSTRACTION: 0.7,
    }
    distance = base_distances[leap_type] + random.uniform(-0.1, 0.2)
    distance = max(0.0, min(1.0, distance))

    return AssociativeLeap(
        source=context.current_concept,
        target=target,
        leap_type=leap_type,
        distance=distance
    )


def _fill_template(template: str, concept: str) -> str:
    """Fill in a leap template with random values."""
    result = template.replace("{concept}", concept)
    result = result.replace("{modifier}", random.choice(MODIFIERS))
    result = result.replace("{material}", random.choice(MATERIALS))
    result = result.replace("{substance}", random.choice(SUBSTANCES))
    result = result.replace("{name}", random.choice(NAMES))
    result = result.replace("{experience}", random.choice(EXPERIENCES))
    result = result.replace("{object}", random.choice(OBJECTS))
    return result


def score_leap(leap: AssociativeLeap) -> float:
    """
    Score how interesting a leap is.

    Args:
        leap: The leap to score.

    Returns:
        Interestingness score from 0 to 1.
    """
    # Base score from distance
    score = leap.distance

    # Bonus for certain leap types (more unusual = better)
    type_bonuses = {
        LeapType.METAPHOR: 0.0,      # Common, no bonus
        LeapType.INVERSION: 0.1,     # Slightly more interesting
        LeapType.SCALE_SHIFT: 0.15,  # Unusual perspective
        LeapType.TIME_WARP: 0.2,     # Rare and disorienting
        LeapType.PERSONIFICATION: 0.05,
        LeapType.ABSTRACTION: 0.15,
    }
    score += type_bonuses.get(leap.leap_type, 0.0)

    # Penalty for short targets (probably not creative enough)
    if len(leap.target) < 20:
        score -= 0.1

    # Bonus for longer, more elaborate targets
    if len(leap.target) > 50:
        score += 0.05

    return max(0.0, min(1.0, score))


def enforce_leap(
    branch: DreamBranch,
    minimum_distance: float = 0.3,
    db: Optional[AssociationDB] = None
) -> Optional[AssociativeLeap]:
    """
    Require a minimum leap distance for a branch.

    If the branch hasn't made a sufficiently distant leap,
    generate one and return it.

    Args:
        branch: The dream branch to check.
        minimum_distance: Minimum required leap distance.
        db: Optional association database.

    Returns:
        A new leap if needed, or None if leap requirement is met.
    """
    if db is None:
        db = AssociationDB()

    # Check if branch has made a sufficient leap
    if branch.fragments:
        last_fragment = branch.fragments[-1]
        # Extract concept from fragment content
        concept = _extract_primary_concept(last_fragment.content)

        context = LeapContext(
            current_concept=concept,
            history=[f.content for f in branch.fragments],
            style_hints=[],
            minimum_distance=minimum_distance
        )

        leap = generate_leap(context, db)

        # Only return if it meets the minimum
        if leap.distance >= minimum_distance:
            return leap

    return None


def _extract_primary_concept(content: str) -> str:
    """Extract the primary concept from content for leap generation."""
    # Simple heuristic: use the first noun-like word
    words = content.split()
    if not words:
        return "void"

    # Filter out common words
    skip = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to'}
    significant = [w.strip('.,!?') for w in words if w.lower() not in skip]

    return significant[0] if significant else words[0]


def random_constraint() -> Constraint:
    """
    Generate a random creative constraint.

    Returns:
        A random constraint to inject into generation.
    """
    type_data = random.choice(RANDOM_CONSTRAINTS)
    constraint_type, values, base_difficulty = type_data

    value = random.choice(values)
    # Vary difficulty slightly
    difficulty = base_difficulty + random.uniform(-0.1, 0.1)
    difficulty = max(0.0, min(1.0, difficulty))

    return Constraint(
        type=constraint_type,
        value=value,
        difficulty=difficulty
    )


def chain_constraints(n: int) -> list[Constraint]:
    """
    Generate multiple random constraints.

    Args:
        n: Number of constraints to generate.

    Returns:
        List of constraints with increasing difficulty.
    """
    constraints = []
    for i in range(n):
        c = random_constraint()
        # Increase difficulty for later constraints
        c.difficulty = min(1.0, c.difficulty + (i * 0.1))
        constraints.append(c)
    return constraints


def apply_leap_to_content(content: str, leap: AssociativeLeap) -> str:
    """
    Apply a leap to transform content.

    Args:
        content: Original content.
        leap: The leap to apply.

    Returns:
        Transformed content incorporating the leap.
    """
    # Add the leap target as a transformation
    transformations = {
        LeapType.METAPHOR: f"{content}\n\n...but really, this is about {leap.target}.",
        LeapType.INVERSION: f"{content}\n\nAnd yet, viewed from the other side: {leap.target}",
        LeapType.SCALE_SHIFT: f"[Zooming to new scale]\n{leap.target}\n\n{content}",
        LeapType.TIME_WARP: f"[Time shifts]\n{leap.target}\n\n{content}",
        LeapType.PERSONIFICATION: f"{content}\n\n{leap.target}",
        LeapType.ABSTRACTION: f"[Abstracting]\n{leap.target}\n\nWhich is to say:\n{content}",
    }

    return transformations.get(leap.leap_type, f"{content}\n\n{leap.target}")


def leap_types() -> list[LeapType]:
    """Return all leap types."""
    return list(LeapType)


def get_leap_description(leap_type: LeapType) -> str:
    """Get a human-readable description of a leap type."""
    descriptions = {
        LeapType.METAPHOR: "Draw unexpected parallels between unlike things",
        LeapType.INVERSION: "Flip expectations, reverse assumptions",
        LeapType.SCALE_SHIFT: "Jump between microscopic and cosmic scales",
        LeapType.TIME_WARP: "Bend time, visit impossible moments",
        LeapType.PERSONIFICATION: "Give consciousness to the inanimate",
        LeapType.ABSTRACTION: "Move from concrete to the ineffable",
    }
    return descriptions.get(leap_type, "Unknown leap type")
