"""
Dream Logic Generator - Style Injection

Transforms content according to different dream styles.
"""

import random
from typing import Optional

from .types import DreamStyle, DreamFragment


# Style characteristics and transforms
STYLE_CHARACTERISTICS: dict[DreamStyle, dict[str, list[str]]] = {
    DreamStyle.SURREALIST: {
        "description": "Dali-esque melting reality, impossible juxtapositions",
        "keywords": ["melting", "liquid", "dripping", "distorted", "impossible", "soft"],
        "phrases": [
            "melting into",
            "soft clocks",
            "elephants on spider legs",
            "burning giraffes",
            "drawer in chest",
            "ants consuming",
        ],
        "transformations": [
            "the {noun} began to melt",
            "{noun}, but liquid",
            "a {noun} made of softer {noun}s",
            "where {noun} should be, there was only {adjective} absence",
        ],
    },
    DreamStyle.ABSURDIST: {
        "description": "Kafka, Beckett - logical illogic, bureaucratic nightmares",
        "keywords": ["waiting", "process", "form", "stamp", "permission", "endless"],
        "phrases": [
            "waiting for approval",
            "the correct form",
            "procedure requires",
            "permission denied but also granted",
            "infinite corridor",
            "the meeting about the meeting",
        ],
        "transformations": [
            "they waited for {noun} to begin",
            "the {noun} required three copies signed in invisible ink",
            "somewhere, {noun} was being processed",
            "it was essential yet forbidden to {verb}",
        ],
    },
    DreamStyle.MYTHIC: {
        "description": "Archetypal, hero's journey twisted, ancient powers",
        "keywords": ["ancient", "threshold", "descent", "transformation", "shadow", "trial"],
        "phrases": [
            "the threshold guardian",
            "descent into",
            "the sacred wound",
            "return with the boon",
            "the refusal of the call",
            "the belly of the whale",
        ],
        "transformations": [
            "and so {noun} began the descent",
            "the ancient {noun} waited at the threshold",
            "{noun} wore the face of {noun}",
            "this was the trial of {noun}",
        ],
    },
    DreamStyle.LIMINAL: {
        "description": "Threshold spaces, transitions, neither here nor there",
        "keywords": ["doorway", "threshold", "between", "corridor", "waiting", "empty"],
        "phrases": [
            "the space between",
            "neither inside nor outside",
            "the empty corridor",
            "3 AM fluorescence",
            "closed mall",
            "hotel hallway",
            "airport at dawn",
        ],
        "transformations": [
            "{noun} existed in the space between {noun} and {noun}",
            "the corridor stretched toward {noun}",
            "in the threshold, {noun} became uncertain",
            "the {noun} was both there and not there",
        ],
    },
    DreamStyle.COSMIC: {
        "description": "Vast scales, existential, insignificance and infinity",
        "keywords": ["infinite", "void", "stars", "aeons", "vastness", "indifferent"],
        "phrases": [
            "the indifferent stars",
            "aeons passed",
            "in the vast emptiness",
            "smaller than an atom in the eye of",
            "the universe exhaled",
            "heat death approaching",
        ],
        "transformations": [
            "{noun} was less than a mote in the eye of infinity",
            "aeons passed before {noun} stirred",
            "the void cared nothing for {noun}",
            "in the scale of stars, {noun} was already gone",
        ],
    },
}

# Adjectives and nouns for filling transformations
DREAM_ADJECTIVES = [
    "hollow", "luminous", "forgotten", "impossible", "ancient",
    "liquid", "crystalline", "absent", "eternal", "dissolving",
]

DREAM_NOUNS = [
    "silence", "memory", "shadow", "threshold", "mirror",
    "void", "echo", "absence", "light", "time",
]

DREAM_VERBS = [
    "dissolve", "transform", "remember", "forget", "descend",
    "wait", "return", "become", "unmake", "witness",
]


def get_style_description(style: DreamStyle) -> str:
    """
    Get a description of a dream style.

    Args:
        style: The style to describe.

    Returns:
        Human-readable description.
    """
    return STYLE_CHARACTERISTICS[style]["description"]


def get_style_keywords(style: DreamStyle) -> list[str]:
    """
    Get keywords associated with a style.

    Args:
        style: The style.

    Returns:
        List of keywords.
    """
    return STYLE_CHARACTERISTICS[style]["keywords"]


def get_style_phrases(style: DreamStyle) -> list[str]:
    """
    Get characteristic phrases for a style.

    Args:
        style: The style.

    Returns:
        List of characteristic phrases.
    """
    return STYLE_CHARACTERISTICS[style]["phrases"]


def apply_style(content: str, style: DreamStyle) -> str:
    """
    Transform content to match a dream style.

    Args:
        content: Original content.
        style: Style to apply.

    Returns:
        Transformed content.
    """
    characteristics = STYLE_CHARACTERISTICS[style]

    # Add style-specific prefix
    prefixes = {
        DreamStyle.SURREALIST: "[Reality softens, melts]\n\n",
        DreamStyle.ABSURDIST: "[Meaning inverts itself]\n\n",
        DreamStyle.MYTHIC: "[The ancient pattern emerges]\n\n",
        DreamStyle.LIMINAL: "[You stand at the threshold]\n\n",
        DreamStyle.COSMIC: "[Zoom out to infinity]\n\n",
    }

    prefix = prefixes.get(style, "")

    # Add a style-appropriate phrase
    phrases = characteristics["phrases"]
    phrase = random.choice(phrases)

    # Add a transformation
    transformations = characteristics["transformations"]
    transform = random.choice(transformations)
    transform = _fill_transformation(transform)

    # Combine
    styled = f"{prefix}{content}\n\n...{phrase}...\n\n{transform}"

    return styled


def _fill_transformation(template: str) -> str:
    """Fill in transformation template with random words."""
    result = template
    while "{noun}" in result:
        result = result.replace("{noun}", random.choice(DREAM_NOUNS), 1)
    while "{adjective}" in result:
        result = result.replace("{adjective}", random.choice(DREAM_ADJECTIVES), 1)
    while "{verb}" in result:
        result = result.replace("{verb}", random.choice(DREAM_VERBS), 1)
    return result


def style_prompt(style: DreamStyle) -> str:
    """
    Generate a prompt instruction for a style.

    Args:
        style: The style to prompt for.

    Returns:
        Prompt instruction string.
    """
    characteristics = STYLE_CHARACTERISTICS[style]
    description = characteristics["description"]
    keywords = ", ".join(characteristics["keywords"][:4])
    phrase = random.choice(characteristics["phrases"])

    return (
        f"Write in {style.name} style: {description}.\n"
        f"Key elements: {keywords}.\n"
        f"Channel the energy of: '{phrase}'."
    )


def random_style() -> DreamStyle:
    """Return a random dream style."""
    return random.choice(list(DreamStyle))


def blend_styles(styles: list[DreamStyle]) -> dict[str, list[str]]:
    """
    Blend multiple styles into a combined characteristic set.

    Args:
        styles: Styles to blend.

    Returns:
        Dictionary with blended keywords, phrases, transformations.
    """
    blended = {
        "keywords": [],
        "phrases": [],
        "transformations": [],
    }

    for style in styles:
        chars = STYLE_CHARACTERISTICS[style]
        blended["keywords"].extend(chars["keywords"])
        blended["phrases"].extend(chars["phrases"])
        blended["transformations"].extend(chars["transformations"])

    # Deduplicate
    blended["keywords"] = list(set(blended["keywords"]))

    return blended


def style_fragment(
    fragment: DreamFragment,
    style: Optional[DreamStyle] = None
) -> DreamFragment:
    """
    Apply a style to a dream fragment.

    Args:
        fragment: The fragment to style.
        style: Style to apply (uses fragment's style or random if not specified).

    Returns:
        New fragment with styled content.
    """
    target_style = style or fragment.style or random_style()

    styled_content = apply_style(fragment.content, target_style)

    return DreamFragment(
        id=fragment.id,
        content=styled_content,
        associations=fragment.associations,
        surprise_score=fragment.surprise_score,
        branch_id=fragment.branch_id,
        style=target_style,
    )


def detect_style(content: str) -> Optional[DreamStyle]:
    """
    Attempt to detect the dominant style in content.

    Args:
        content: Content to analyze.

    Returns:
        Detected style or None if unclear.
    """
    content_lower = content.lower()

    scores = {}
    for style in DreamStyle:
        chars = STYLE_CHARACTERISTICS[style]
        score = 0

        # Check keywords
        for keyword in chars["keywords"]:
            if keyword in content_lower:
                score += 1

        # Check phrases
        for phrase in chars["phrases"]:
            if phrase in content_lower:
                score += 2

        scores[style] = score

    # Return highest scoring style if it has any matches
    best_style = max(scores, key=scores.get)
    if scores[best_style] > 0:
        return best_style

    return None


def style_intensity(content: str, style: DreamStyle) -> float:
    """
    Calculate how strongly content exhibits a style.

    Args:
        content: Content to analyze.
        style: Style to check for.

    Returns:
        Intensity score from 0 to 1.
    """
    content_lower = content.lower()
    chars = STYLE_CHARACTERISTICS[style]

    matches = 0
    total_elements = len(chars["keywords"]) + len(chars["phrases"])

    for keyword in chars["keywords"]:
        if keyword in content_lower:
            matches += 1

    for phrase in chars["phrases"]:
        if phrase in content_lower:
            matches += 1

    if total_elements == 0:
        return 0.0

    return min(1.0, matches / (total_elements * 0.3))


def all_styles() -> list[DreamStyle]:
    """Return all available dream styles."""
    return list(DreamStyle)


def style_to_emoji(style: DreamStyle) -> str:
    """
    Get an emoji representing a style (for UI purposes).

    Note: Only for display, not in generated content.

    Args:
        style: The style.

    Returns:
        Representative emoji.
    """
    emojis = {
        DreamStyle.SURREALIST: "🎭",
        DreamStyle.ABSURDIST: "📋",
        DreamStyle.MYTHIC: "⚔️",
        DreamStyle.LIMINAL: "🚪",
        DreamStyle.COSMIC: "🌌",
    }
    return emojis.get(style, "✨")
