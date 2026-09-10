"""
Dream Logic Generator - Synthesis as Amplification

Unlike traditional synthesis that reconciles differences, dream synthesis
amplifies the weird, finds unexpected resonances, and creates collages
that embrace contradictions.
"""

import random
from typing import Optional

from .types import (
    DreamFragment,
    DreamCollage,
    AssociativeLeap,
    LeapType,
    DreamStyle,
)
from .interestingness import score_interestingness


def amplify(fragments: list[DreamFragment]) -> DreamCollage:
    """
    Intensify the weird across fragments rather than smooth it out.

    Args:
        fragments: Dream fragments to amplify.

    Returns:
        DreamCollage with amplified weirdness.
    """
    if not fragments:
        return DreamCollage()

    collage = DreamCollage()

    # Score fragments and sort by interestingness
    scored = []
    for fragment in fragments:
        score = score_interestingness(fragment.content)
        scored.append((fragment, score.total))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep the weirdest content, amplify it
    for fragment, score in scored:
        # Create amplified version
        amplified = _amplify_fragment(fragment, score)
        collage.add_fragment(amplified)

    # Find connections between fragments
    connections = find_resonances(fragments)
    for conn in connections:
        collage.add_connection(conn)

    # Extract emergent themes
    themes = emergent_themes(fragments)
    for theme in themes:
        collage.add_theme(theme)

    # Set overall style to most common among fragments
    style_counts: dict[DreamStyle, int] = {}
    for f in fragments:
        if f.style:
            style_counts[f.style] = style_counts.get(f.style, 0) + 1
    if style_counts:
        collage.style = max(style_counts, key=style_counts.get)

    return collage


def _amplify_fragment(fragment: DreamFragment, score: float) -> DreamFragment:
    """Create an amplified version of a fragment."""
    # The weirder it already is, the less we need to amplify
    amplification_needed = 1.0 - score

    content = fragment.content

    if amplification_needed > 0.5:
        # Add amplifying transformations
        amplifications = [
            f"\n[This intensifies]\n{content}",
            f"{content}\n[Now stranger:]",
            f"[Zooming into the weird]\n{content}",
        ]
        content = random.choice(amplifications)

    # Boost surprise score slightly
    new_score = min(1.0, fragment.surprise_score + amplification_needed * 0.2)

    return DreamFragment(
        id=fragment.id,
        content=content,
        associations=fragment.associations,
        surprise_score=new_score,
        branch_id=fragment.branch_id,
        style=fragment.style,
    )


def find_resonances(fragments: list[DreamFragment]) -> list[AssociativeLeap]:
    """
    Find unexpected connections between fragments.

    Rather than looking for logical connections, we look for
    surprising resonances - words or concepts that echo across
    fragments in unexpected ways.

    Args:
        fragments: Fragments to analyze.

    Returns:
        List of associative leaps connecting fragments.
    """
    if len(fragments) < 2:
        return []

    connections = []

    # Extract key concepts from each fragment
    fragment_concepts: dict[str, set[str]] = {}
    for fragment in fragments:
        concepts = _extract_concepts(fragment.content)
        fragment_concepts[fragment.id] = concepts

    # Look for surprising overlaps
    processed_pairs: set[tuple[str, str]] = set()

    for i, frag1 in enumerate(fragments):
        for j, frag2 in enumerate(fragments):
            if i >= j:
                continue

            pair = (frag1.id, frag2.id)
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            concepts1 = fragment_concepts[frag1.id]
            concepts2 = fragment_concepts[frag2.id]

            # Find overlapping concepts
            overlap = concepts1 & concepts2

            # Filter out common words
            common = {'the', 'and', 'but', 'with', 'from', 'into', 'that', 'this'}
            meaningful_overlap = overlap - common

            for concept in meaningful_overlap:
                # Create a connection
                leap = AssociativeLeap(
                    source=f"{frag1.id}:{concept}",
                    target=f"{frag2.id}:{concept}",
                    leap_type=LeapType.METAPHOR,
                    distance=0.6,  # Resonance connections are moderately distant
                )
                connections.append(leap)

    # Also create some random leaps between fragments
    if len(fragments) >= 2:
        num_random = min(3, len(fragments) - 1)
        for _ in range(num_random):
            frag1, frag2 = random.sample(fragments, 2)
            concept1 = random.choice(list(fragment_concepts.get(frag1.id, {"void"})))
            concept2 = random.choice(list(fragment_concepts.get(frag2.id, {"void"})))

            leap = AssociativeLeap(
                source=concept1,
                target=concept2,
                leap_type=random.choice(list(LeapType)),
                distance=random.uniform(0.5, 0.9),
            )
            connections.append(leap)

    return connections


def _extract_concepts(content: str) -> set[str]:
    """Extract key concepts from content."""
    words = content.lower().split()
    # Filter to meaningful words
    skip = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'in', 'on', 'at', 'to', 'of', 'for', 'with', 'and', 'but',
        'or', 'that', 'this', 'it', 'its', 'as', 'by', 'from', 'into'
    }
    concepts = set()
    for word in words:
        clean = word.strip('.,!?;:"\'-')
        if len(clean) > 3 and clean not in skip:
            concepts.add(clean)
    return concepts


def collage(fragments: list[DreamFragment], smooth: bool = False) -> str:
    """
    Arrange fragments into a collage.

    By default, does NOT smooth transitions - preserves jarring edges.

    Args:
        fragments: Fragments to arrange.
        smooth: If True, add minimal transitions (usually False for dream logic).

    Returns:
        Combined text collage.
    """
    if not fragments:
        return ""

    if smooth:
        # Minimal smoothing
        parts = []
        for i, fragment in enumerate(fragments):
            if i > 0:
                parts.append("\n\n---\n\n")
            parts.append(fragment.content)
        return "".join(parts)

    # No smoothing - raw juxtaposition
    parts = []
    for fragment in fragments:
        parts.append(fragment.content)

    # Random arrangement order to increase disorientation
    random.shuffle(parts)

    # Use various separators
    separators = [
        "\n\n",
        "\n\n[]\n\n",
        "\n\n///\n\n",
        "\n\n. . .\n\n",
        "\n\n[shift]\n\n",
    ]

    result_parts = []
    for i, part in enumerate(parts):
        if i > 0:
            sep = random.choice(separators)
            result_parts.append(sep)
        result_parts.append(part)

    return "".join(result_parts)


def emergent_themes(fragments: list[DreamFragment]) -> list[str]:
    """
    Extract patterns from chaos - themes that emerge across fragments.

    These aren't necessarily logical themes, but patterns in the madness.

    Args:
        fragments: Fragments to analyze.

    Returns:
        List of emergent theme descriptions.
    """
    if not fragments:
        return []

    # Collect all concepts
    all_concepts: dict[str, int] = {}
    for fragment in fragments:
        concepts = _extract_concepts(fragment.content)
        for concept in concepts:
            all_concepts[concept] = all_concepts.get(concept, 0) + 1

    # Find recurring concepts (appear in multiple fragments)
    recurring = [c for c, count in all_concepts.items() if count >= 2]

    themes = []

    # Create theme descriptions from recurring concepts
    if recurring:
        # Pattern: "The X of Y" themes
        if len(recurring) >= 2:
            c1, c2 = random.sample(recurring[:6], min(2, len(recurring)))
            themes.append(f"The {c1} of {c2}")

        # Pattern: "X becoming Y"
        if len(recurring) >= 2:
            c1, c2 = random.sample(recurring[:6], min(2, len(recurring)))
            themes.append(f"{c1} becoming {c2}")

        # Single concept themes
        for concept in recurring[:3]:
            theme_templates = [
                f"The persistence of {concept}",
                f"{concept} echoes throughout",
                f"Everything leads to {concept}",
            ]
            themes.append(random.choice(theme_templates))

    # Add abstract themes based on fragment characteristics
    abstract_themes = _detect_abstract_themes(fragments)
    themes.extend(abstract_themes)

    return themes[:5]  # Limit themes


def _detect_abstract_themes(fragments: list[DreamFragment]) -> list[str]:
    """Detect abstract thematic patterns."""
    themes = []

    # Check for style patterns
    styles = [f.style for f in fragments if f.style]
    if styles:
        style_set = set(styles)
        if len(style_set) == 1:
            themes.append(f"Unified {styles[0].name.lower()} vision")
        elif len(style_set) > 2:
            themes.append("Style dissolution - boundaries collapse")

    # Check for high surprise scores
    high_surprise = sum(1 for f in fragments if f.surprise_score > 0.7)
    if high_surprise > len(fragments) / 2:
        themes.append("Sustained strangeness")

    # Check for associations
    association_count = sum(len(f.associations) for f in fragments)
    if association_count > len(fragments) * 2:
        themes.append("Dense web of associations")

    return themes


def embrace_contradictions(fragments: list[DreamFragment]) -> str:
    """
    Create output that explicitly embraces contradictions between fragments.

    Args:
        fragments: Fragments that may contradict each other.

    Returns:
        Text that highlights and celebrates contradictions.
    """
    if not fragments:
        return ""

    # Find potential contradictions by looking for opposing concepts
    opposites = [
        ("light", "dark"),
        ("up", "down"),
        ("in", "out"),
        ("alive", "dead"),
        ("past", "future"),
        ("hot", "cold"),
        ("large", "small"),
    ]

    contradiction_notes = []

    for frag1 in fragments:
        content1 = frag1.content.lower()
        for frag2 in fragments:
            if frag1.id == frag2.id:
                continue
            content2 = frag2.content.lower()

            for word1, word2 in opposites:
                if word1 in content1 and word2 in content2:
                    contradiction_notes.append(
                        f"[{word1} and {word2} coexist without resolution]"
                    )

    # Build output
    parts = [f.content for f in fragments]
    result = "\n\n".join(parts)

    if contradiction_notes:
        result += "\n\n---\n\n"
        result += "[These contradictions remain unresolved, as they should:]\n"
        result += "\n".join(contradiction_notes[:5])

    return result


def synthesis_prompt(fragments: list[DreamFragment]) -> str:
    """
    Generate a prompt for further synthesis/amplification.

    Args:
        fragments: Current fragments.

    Returns:
        Prompt for generating more dream content.
    """
    themes = emergent_themes(fragments)
    connections = find_resonances(fragments)

    prompt_parts = [
        "Given these dream fragments, amplify the strangeness.",
        "",
        "Current themes emerging:",
    ]

    for theme in themes[:3]:
        prompt_parts.append(f"  - {theme}")

    if connections:
        prompt_parts.append("")
        prompt_parts.append("Unexpected resonances found:")
        for conn in connections[:3]:
            prompt_parts.append(f"  - {conn.source} <-> {conn.target}")

    prompt_parts.append("")
    prompt_parts.append("Do NOT resolve contradictions. Amplify them.")
    prompt_parts.append("Do NOT explain. Intensify.")
    prompt_parts.append("Make it weirder.")

    return "\n".join(prompt_parts)


def create_collage(
    fragments: list[DreamFragment],
    style: Optional[DreamStyle] = None
) -> DreamCollage:
    """
    Create a full dream collage from fragments.

    Convenience function combining amplification, connections, and themes.

    Args:
        fragments: Fragments to combine.
        style: Optional overall style to apply.

    Returns:
        Complete DreamCollage.
    """
    collage_result = amplify(fragments)

    if style:
        collage_result.style = style

    return collage_result
