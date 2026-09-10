"""
Dream Logic Generator - Output Formats

Transforms dream collages into various output formats:
- Story narrative
- Image generation prompts
- Game design concepts
- Poetry
- Raw unprocessed fragments
"""

import random
from typing import Optional

from .types import DreamCollage, DreamFragment, DreamStyle


def to_story(collage: DreamCollage, smooth: bool = False) -> str:
    """
    Convert a dream collage to narrative story format.

    By default, preserves the jarring dream logic rather than
    smoothing into conventional narrative.

    Args:
        collage: The dream collage to convert.
        smooth: If True, add narrative transitions.

    Returns:
        Story-format text.
    """
    if not collage.fragments:
        return "The dream contained nothing, yet everything."

    parts = []

    # Opening based on style
    openings = {
        DreamStyle.SURREALIST: "Reality began to soften around the edges.",
        DreamStyle.ABSURDIST: "The proper forms had been filed, in triplicate.",
        DreamStyle.MYTHIC: "In the time before memory, this was already happening.",
        DreamStyle.LIMINAL: "You are standing in a doorway. You have always been standing here.",
        DreamStyle.COSMIC: "From the perspective of stars, this had already ended.",
    }

    if collage.style:
        parts.append(openings.get(collage.style, "It began."))
    else:
        parts.append("The dream opened.")

    parts.append("")

    # Add fragments
    if smooth:
        transitions = [
            "And then,",
            "Which led to",
            "Meanwhile,",
            "Underneath that,",
            "Or perhaps,",
            "Simultaneously,",
        ]
        for i, fragment in enumerate(collage.fragments):
            if i > 0:
                parts.append(random.choice(transitions))
            parts.append(fragment.content)
            parts.append("")
    else:
        # Dream-logic transitions (or none at all)
        dream_transitions = [
            "",  # No transition
            "[shift]",
            "[cut]",
            "...",
            "///",
            "[elsewhere]",
            "[or was it]",
        ]
        for i, fragment in enumerate(collage.fragments):
            if i > 0:
                parts.append(random.choice(dream_transitions))
            parts.append(fragment.content)
            parts.append("")

    # Add emergent themes as epilogue
    if collage.emergent_themes:
        parts.append("---")
        parts.append("(What remained:)")
        for theme in collage.emergent_themes[:3]:
            parts.append(f"  - {theme}")

    return "\n".join(parts)


def to_image_prompt(
    collage: DreamCollage,
    style_hints: Optional[list[str]] = None
) -> str:
    """
    Convert a dream collage to an image generation prompt.

    Extracts visual elements and creates a prompt suitable for
    image generation models.

    Args:
        collage: The dream collage.
        style_hints: Optional additional style hints.

    Returns:
        Image generation prompt.
    """
    if not collage.fragments:
        return "An empty void that somehow contains everything"

    parts = []

    # Extract visual elements from fragments
    visual_keywords = []
    for fragment in collage.fragments:
        # Extract nouns and adjectives (simple heuristic)
        words = fragment.content.lower().split()
        for word in words:
            clean = word.strip('.,!?;:"\'-')
            if len(clean) > 4:
                visual_keywords.append(clean)

    # Deduplicate and limit
    unique_keywords = list(set(visual_keywords))[:10]

    # Style prefix based on dream style
    style_prefixes = {
        DreamStyle.SURREALIST: "A surrealist dreamscape, Dali-inspired, melting forms,",
        DreamStyle.ABSURDIST: "An absurdist scene, Magritte-like, impossible geometry,",
        DreamStyle.MYTHIC: "A mythic vision, ancient archetypes, golden light,",
        DreamStyle.LIMINAL: "A liminal space, empty corridors, flickering fluorescent,",
        DreamStyle.COSMIC: "A cosmic vista, nebulae, vast emptiness,",
    }

    if collage.style:
        parts.append(style_prefixes.get(collage.style, "A dreamlike scene,"))
    else:
        parts.append("A surreal dreamscape,")

    # Add key visual elements
    parts.append(" ".join(unique_keywords[:5]))

    # Add connection imagery
    if collage.connections:
        conn = collage.connections[0]
        parts.append(f", {conn.source} transforming into {conn.target}")

    # Add style hints
    if style_hints:
        parts.append(", " + ", ".join(style_hints))

    # Add quality suffixes
    parts.append(", highly detailed, ethereal lighting, dreamlike atmosphere")

    return " ".join(parts)


def to_game_concept(collage: DreamCollage) -> str:
    """
    Convert a dream collage to a game design concept.

    Extracts mechanics, themes, and atmosphere for game design.

    Args:
        collage: The dream collage.

    Returns:
        Game concept document.
    """
    if not collage.fragments:
        return "Game Concept: The Empty Dream\n\nA game about nothing that matters."

    parts = []

    parts.append("GAME CONCEPT: Dream Logic")
    parts.append("=" * 50)
    parts.append("")

    # Genre based on style
    genres = {
        DreamStyle.SURREALIST: "Puzzle / Walking Simulator",
        DreamStyle.ABSURDIST: "Adventure / Dark Comedy",
        DreamStyle.MYTHIC: "Action RPG / Roguelike",
        DreamStyle.LIMINAL: "Horror / Exploration",
        DreamStyle.COSMIC: "Strategy / Meditation",
    }

    if collage.style:
        parts.append(f"Genre: {genres.get(collage.style, 'Experimental')}")
    else:
        parts.append("Genre: Experimental / Dream Simulator")
    parts.append("")

    # Core mechanic from connections
    parts.append("CORE MECHANICS:")
    if collage.connections:
        for conn in collage.connections[:3]:
            mechanic = f"  - Transform {conn.source} into {conn.target}"
            parts.append(mechanic)
    else:
        parts.append("  - Navigate through shifting dreamscapes")
        parts.append("  - Collect fragments of meaning")
    parts.append("")

    # Atmosphere from fragments
    parts.append("ATMOSPHERE:")
    for fragment in collage.fragments[:3]:
        # Extract first sentence or 100 chars
        excerpt = fragment.content[:100].split('.')[0]
        parts.append(f"  \"{excerpt}...\"")
    parts.append("")

    # Themes
    parts.append("THEMES:")
    if collage.emergent_themes:
        for theme in collage.emergent_themes:
            parts.append(f"  - {theme}")
    else:
        parts.append("  - The nature of dreams")
        parts.append("  - Logic vs. feeling")
    parts.append("")

    # Win condition
    parts.append("OBJECTIVE:")
    parts.append("  There is no winning. There is only dreaming.")
    parts.append("  Progress is measured in strangeness accumulated.")

    return "\n".join(parts)


def to_poem(collage: DreamCollage, form: str = "free") -> str:
    """
    Convert a dream collage to poetic format.

    Args:
        collage: The dream collage.
        form: Poem form ("free", "haiku_chain", "fragments").

    Returns:
        Poem text.
    """
    if not collage.fragments:
        return "nothing\nand nothing\nand nothing again"

    if form == "haiku_chain":
        return _to_haiku_chain(collage)
    elif form == "fragments":
        return _to_fragment_poem(collage)
    else:
        return _to_free_verse(collage)


def _to_free_verse(collage: DreamCollage) -> str:
    """Convert to free verse poem."""
    lines = []

    for fragment in collage.fragments:
        # Break into lines at natural points
        content = fragment.content
        sentences = content.replace('...', '.').split('.')

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Break long sentences into lines
            words = sentence.split()
            if len(words) > 8:
                mid = len(words) // 2
                lines.append(' '.join(words[:mid]))
                lines.append(' '.join(words[mid:]))
            else:
                lines.append(sentence)

        lines.append("")  # Stanza break

    # Add coda from themes
    if collage.emergent_themes:
        lines.append("---")
        theme = random.choice(collage.emergent_themes)
        lines.append(theme.lower())

    return "\n".join(lines)


def _to_haiku_chain(collage: DreamCollage) -> str:
    """Convert to a chain of haiku-like stanzas (not strict syllables)."""
    haikus = []

    for fragment in collage.fragments:
        words = fragment.content.split()[:17]  # Rough haiku length

        # Split into three lines (5-7-5 word approximation)
        if len(words) >= 10:
            line1 = ' '.join(words[:4])
            line2 = ' '.join(words[4:9])
            line3 = ' '.join(words[9:13])
            haikus.append(f"{line1}\n{line2}\n{line3}")
        elif len(words) >= 5:
            line1 = ' '.join(words[:2])
            line2 = ' '.join(words[2:4])
            line3 = ' '.join(words[4:])
            haikus.append(f"{line1}\n{line2}\n{line3}")

    return "\n\n".join(haikus) if haikus else "dream\nwithout words\nstill dreaming"


def _to_fragment_poem(collage: DreamCollage) -> str:
    """Convert to fragmented/concrete poetry."""
    lines = []

    for fragment in collage.fragments:
        words = fragment.content.split()

        # Scatter words across lines
        for i, word in enumerate(words[:15]):
            clean = word.strip('.,!?;:"\'-')
            if clean:
                # Random indentation
                indent = " " * random.randint(0, 20)
                lines.append(f"{indent}{clean}")

        lines.append("")

    return "\n".join(lines)


def to_raw(collage: DreamCollage) -> str:
    """
    Return unprocessed fragments.

    Args:
        collage: The dream collage.

    Returns:
        Raw fragment text.
    """
    if not collage.fragments:
        return ""

    parts = []

    for i, fragment in enumerate(collage.fragments):
        parts.append(f"[{i + 1}] {fragment.id}")
        parts.append(fragment.content)
        parts.append("")

    if collage.emergent_themes:
        parts.append("---")
        parts.append("Emergent themes:")
        for theme in collage.emergent_themes:
            parts.append(f"  - {theme}")

    if collage.connections:
        parts.append("")
        parts.append("Connections:")
        for conn in collage.connections:
            parts.append(f"  {conn.source} -> {conn.target} ({conn.leap_type.name})")

    return "\n".join(parts)


def to_json_serializable(collage: DreamCollage) -> dict:
    """
    Convert collage to JSON-serializable dictionary.

    Args:
        collage: The dream collage.

    Returns:
        Dictionary representation.
    """
    return {
        "fragments": [
            {
                "id": f.id,
                "content": f.content,
                "associations": f.associations,
                "surprise_score": f.surprise_score,
                "branch_id": f.branch_id,
                "style": f.style.name if f.style else None,
            }
            for f in collage.fragments
        ],
        "connections": [
            {
                "source": c.source,
                "target": c.target,
                "leap_type": c.leap_type.name,
                "distance": c.distance,
            }
            for c in collage.connections
        ],
        "emergent_themes": collage.emergent_themes,
        "style": collage.style.name if collage.style else None,
        "total_interestingness": collage.total_interestingness,
    }


def all_output_formats() -> list[str]:
    """Return all available output format names."""
    return ["story", "image_prompt", "game_concept", "poem", "raw", "json"]
