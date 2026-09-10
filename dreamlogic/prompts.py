"""
Dream Logic Generator - Dream Prompts

Prompts that replace logical thinking with associative, dreamlike exploration.
These prompts trigger associations, inversions, mythologizing, and scale shifts.
"""

import random
from typing import Optional

from .types import DreamStyle


# Association trigger prompts
ASSOCIATION_PROMPTS = [
    "What does this remind you of that makes no sense?",
    "If this were a smell, what would it be?",
    "What color is this feeling?",
    "What would this sound like in an empty cathedral?",
    "If you could taste this idea, what would it be?",
    "What texture does this concept have?",
    "What does this look like from inside a dream?",
    "If this were weather, what would it be?",
    "What animal does this remind you of, and why is it wrong?",
    "What childhood memory does this invoke that you've never actually had?",
]

# Inversion prompts
INVERSION_PROMPTS = [
    "What is the opposite of this, and why is it the same?",
    "If everything about this were inverted, what remains unchanged?",
    "What would this look like running backwards through time?",
    "If this were true, what impossible thing would also be true?",
    "What is this from the perspective of what it destroys?",
    "How would this appear if gravity worked in reverse?",
    "What is the shadow of this idea?",
    "If this were a lie, what truth would it contain?",
    "What would happen if this existed in negative space?",
    "How does this look from underneath itself?",
]

# Mythologizing prompts
MYTHIC_PROMPTS = [
    "If this were an ancient myth, who would be the hero?",
    "What threshold does this guard?",
    "What descent does this require?",
    "If this were a sacred wound, how would it be healed?",
    "What trial does this represent?",
    "What is the belly of this whale?",
    "What boon does this return with?",
    "If this were an archetype, which mask does it wear?",
    "What ancient pattern is this echoing?",
    "What is the shadow side of this journey?",
]

# Scale shift prompts
SCALE_PROMPTS = [
    "What does this look like at the scale of atoms?",
    "If this were the size of a galaxy, what would you notice?",
    "What is this from the perspective of something microscopic living inside it?",
    "How does this appear from a thousand light years away?",
    "If this were the size of a heartbeat, what would it contain?",
    "What does this look like at the scale of geological time?",
    "If you could zoom into the space between its molecules, what would you find?",
    "What is this from the viewpoint of the universe?",
    "If this were measured in lifetimes, how long would it be?",
    "What does this look like at the scale of a thought?",
]

# Time shift prompts
TIME_PROMPTS = [
    "What was this before it existed?",
    "What will this become after it is forgotten?",
    "What does this look like in the moment of its creation?",
    "How does this appear at the instant of its ending?",
    "What is this in the pause between heartbeats?",
    "If this were happening yesterday and tomorrow simultaneously, what would change?",
    "What memory does this have of its own future?",
    "What is this in the time before time?",
    "How does this exist in the eternal now?",
    "What does this look like from the end of the universe looking back?",
]

# Personification prompts
PERSONIFICATION_PROMPTS = [
    "If this could speak, what secret would it tell?",
    "What does this dream about when no one is watching?",
    "What is this afraid of?",
    "If this could love, what would it choose?",
    "What grief does this carry?",
    "If this had a childhood, what trauma did it experience?",
    "What does this want that it cannot have?",
    "If this could write a letter, who would it address?",
    "What is the one thing this would never tell anyone?",
    "If this could die, how would it choose to go?",
]


def what_does_this_remind_you_of(context: str) -> str:
    """
    Generate an association-triggering prompt.

    Args:
        context: The current context/content.

    Returns:
        Prompt to trigger associations.
    """
    base = random.choice(ASSOCIATION_PROMPTS)
    return f"Given: {context[:200]}...\n\n{base}"


def what_is_the_opposite(context: str) -> str:
    """
    Generate an inversion prompt.

    Args:
        context: The current context/content.

    Returns:
        Prompt to explore inversions.
    """
    base = random.choice(INVERSION_PROMPTS)
    return f"Consider: {context[:200]}...\n\n{base}"


def if_this_were_a_myth(context: str) -> str:
    """
    Generate a mythologizing prompt.

    Args:
        context: The current context/content.

    Returns:
        Prompt to mythologize the content.
    """
    base = random.choice(MYTHIC_PROMPTS)
    return f"The story so far: {context[:200]}...\n\n{base}"


def zoom_to_scale(context: str, scale: str = "random") -> str:
    """
    Generate a scale-shift prompt.

    Args:
        context: The current context/content.
        scale: Scale direction ("micro", "cosmic", or "random").

    Returns:
        Prompt to shift scale.
    """
    if scale == "micro":
        prompts = [p for p in SCALE_PROMPTS if "atom" in p or "microscopic" in p or "molecule" in p]
    elif scale == "cosmic":
        prompts = [p for p in SCALE_PROMPTS if "galaxy" in p or "universe" in p or "light year" in p]
    else:
        prompts = SCALE_PROMPTS

    base = random.choice(prompts) if prompts else random.choice(SCALE_PROMPTS)
    return f"Observe: {context[:200]}...\n\n{base}"


def time_shift(context: str, direction: str = "random") -> str:
    """
    Generate a time-shift prompt.

    Args:
        context: The current context/content.
        direction: "past", "future", or "random".

    Returns:
        Prompt to shift time perspective.
    """
    if direction == "past":
        prompts = [p for p in TIME_PROMPTS if "before" in p or "was" in p or "memory" in p]
    elif direction == "future":
        prompts = [p for p in TIME_PROMPTS if "will" in p or "after" in p or "end" in p]
    else:
        prompts = TIME_PROMPTS

    base = random.choice(prompts) if prompts else random.choice(TIME_PROMPTS)
    return f"In this moment: {context[:200]}...\n\n{base}"


def personify(context: str) -> str:
    """
    Generate a personification prompt.

    Args:
        context: The current context/content.

    Returns:
        Prompt to personify the content.
    """
    base = random.choice(PERSONIFICATION_PROMPTS)
    return f"This exists: {context[:200]}...\n\n{base}"


def style_specific_prompt(context: str, style: DreamStyle) -> str:
    """
    Generate a prompt appropriate for a specific dream style.

    Args:
        context: The current context/content.
        style: The dream style.

    Returns:
        Style-appropriate prompt.
    """
    style_prompts = {
        DreamStyle.SURREALIST: [
            "What is melting here that shouldn't melt?",
            "What impossible object appears in the corner of your vision?",
            "If space bent around this, what shape would it make?",
            "What soft thing here should be hard? What hard thing should be soft?",
        ],
        DreamStyle.ABSURDIST: [
            "What bureaucratic process governs this?",
            "What form must be filled out in triplicate?",
            "What is the logical conclusion that makes no sense?",
            "Who is waiting for permission that will never come?",
        ],
        DreamStyle.MYTHIC: [
            "What threshold must be crossed?",
            "What is the boon that must be returned?",
            "What ancient pattern is being repeated?",
            "What mask does the guide wear?",
        ],
        DreamStyle.LIMINAL: [
            "What exists in the space between?",
            "What threshold is this that leads nowhere?",
            "What is neither inside nor outside?",
            "What time is it in this empty corridor?",
        ],
        DreamStyle.COSMIC: [
            "From the scale of galaxies, what does this amount to?",
            "In the lifetime of a star, how long does this last?",
            "What is this to the indifferent void?",
            "When everything ends, what echo of this remains?",
        ],
    }

    prompts = style_prompts.get(style, ASSOCIATION_PROMPTS)
    base = random.choice(prompts)
    return f"[{style.name}]\n{context[:200]}...\n\n{base}"


def random_dream_prompt(context: str) -> str:
    """
    Generate a random dream prompt of any type.

    Args:
        context: The current context/content.

    Returns:
        Random dream prompt.
    """
    all_prompts = (
        ASSOCIATION_PROMPTS +
        INVERSION_PROMPTS +
        MYTHIC_PROMPTS +
        SCALE_PROMPTS +
        TIME_PROMPTS +
        PERSONIFICATION_PROMPTS
    )
    base = random.choice(all_prompts)
    return f"{context[:200]}...\n\n{base}"


def chain_prompts(context: str, n: int = 3) -> list[str]:
    """
    Generate a chain of prompts for sequential exploration.

    Args:
        context: The current context/content.
        n: Number of prompts in the chain.

    Returns:
        List of prompts.
    """
    prompt_generators = [
        what_does_this_remind_you_of,
        what_is_the_opposite,
        if_this_were_a_myth,
        zoom_to_scale,
        time_shift,
        personify,
    ]

    selected = random.sample(prompt_generators, min(n, len(prompt_generators)))
    return [gen(context) for gen in selected]


def meta_prompt(context: str) -> str:
    """
    Generate a meta-prompt that asks about the nature of the exploration itself.

    Args:
        context: The current context/content.

    Returns:
        Meta-level prompt.
    """
    meta_prompts = [
        "What is this exploration avoiding?",
        "What question is this really asking?",
        "What would this look like if it stopped making sense entirely?",
        "What is the dream dreaming about itself?",
        "If this exploration were a creature, what would it eat?",
        "What is hiding in the spaces between these words?",
        "What would happen if we followed the wrong thread deliberately?",
        "What is the question that cannot be asked here?",
    ]
    base = random.choice(meta_prompts)
    return f"[META]\n{context[:200]}...\n\n{base}"


def constraint_prompt(context: str, constraint_text: str) -> str:
    """
    Generate a prompt that incorporates a specific constraint.

    Args:
        context: The current context/content.
        constraint_text: The constraint to incorporate.

    Returns:
        Constrained prompt.
    """
    return (
        f"Given this: {context[:200]}...\n\n"
        f"CONSTRAINT: {constraint_text}\n\n"
        "Continue, but weirder."
    )


def all_prompt_types() -> list[str]:
    """Return names of all prompt types available."""
    return [
        "association",
        "inversion",
        "mythic",
        "scale",
        "time",
        "personification",
        "meta",
        "style_specific",
    ]
