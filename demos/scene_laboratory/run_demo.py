#!/usr/bin/env python3
"""Scene Laboratory: PSS writes the same scene 5 radically different ways.

Not "explore themes." Actual prose.

A writer gives us: "Write the scene where she tells him it's over."

PSS produces:
- Draft 1: Cold, clinical, third person distant
- Draft 2: Raw anger, first person stream of consciousness
- Draft 3: Dark comedy, absurdist details
- Draft 4: From his perspective, devastated
- Draft 5: Told through physical sensations only

The writer gets 5 actual drafts they can use, combine, or steal from.
THAT'S a superpower.
"""

import os
import sys
import json
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pss.config import PSSConfig
from pss.providers import create_provider
from pss.harness import run_pss


# Enforced stylistic diversity - branches MUST be different along these axes
STYLE_AXES = [
    "cold_clinical",      # Detached, observational, third person distant
    "raw_emotional",      # First person, stream of consciousness, messy
    "dark_comedy",        # Absurdist, finding humor in pain
    "other_perspective",  # From the other person's POV
    "sensory_physical",   # No internal thoughts, only body sensations
]

STYLE_PROMPTS = {
    "cold_clinical": """Write this scene COLD and CLINICAL.
- Third person distant, like a camera watching
- No access to internal thoughts
- Precise, observational language
- The emotion is in what's NOT said
- Think: Hemingway, Cormac McCarthy""",

    "raw_emotional": """Write this scene as RAW EMOTIONAL FIRST PERSON.
- Deep first person, messy stream of consciousness
- Run-on sentences when overwhelmed
- Contradictions, backtracking, denial
- The prose itself should feel unhinged
- Think: Sylvia Plath, Rachel Cusk""",

    "dark_comedy": """Write this scene as DARK COMEDY.
- Find the absurd in the tragic
- Inappropriate details that land weird
- The character notices the wrong things
- Funny but it makes you uncomfortable
- Think: Lorrie Moore, George Saunders""",

    "other_perspective": """Write this scene from THE OTHER PERSON'S perspective.
- They're receiving this news, not delivering it
- Their internal experience of the moment
- What they notice, what they miss
- The same scene, totally different meaning
- Flip the power dynamic""",

    "sensory_physical": """Write this scene through PHYSICAL SENSATION ONLY.
- No thoughts, no dialogue tags beyond "said"
- Only what the body experiences
- Temperature, texture, pressure, movement
- The emotion lives in the body
- Think: visceral, somatic, embodied""",
}


def create_scene_prompt(scene_description: str, style: str) -> str:
    """Create a prompt for a specific style."""
    style_instruction = STYLE_PROMPTS.get(style, "")

    return f"""Write this scene as FICTION. Output ONLY the scene itself - no commentary, no analysis, no suggestions.

SCENE: {scene_description}

STYLE:
{style_instruction}

IMPORTANT:
- Write the actual PROSE. Start with a sentence of narrative.
- Do NOT give writing advice or analysis.
- Do NOT say "consider" or "you could" or "here's how".
- Just write the scene. 200-300 words.

---

"""


def main():
    print("=" * 70)
    print("SCENE LABORATORY")
    print("Same scene. Five radically different drafts.")
    print("=" * 70)
    print()

    # The scene to explore
    scene = """She tells him the relationship is over.
They're in his apartment. It's 2am. She's been working up to this for months.
He didn't see it coming."""

    print("SCENE:")
    print(f"  {scene}")
    print()
    print("DRAFTS TO GENERATE:")
    for i, style in enumerate(STYLE_AXES, 1):
        print(f"  {i}. {style.replace('_', ' ').title()}")
    print()

    provider = create_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")

    drafts = {}
    total_tokens = 0

    # Generate each draft with enforced style
    for style in STYLE_AXES:
        print(f"\n{'='*70}")
        print(f"GENERATING: {style.replace('_', ' ').upper()}")
        print("=" * 70)

        prompt = create_scene_prompt(scene, style)

        config = PSSConfig(
            provider="openrouter",
            model="meta-llama/llama-3.1-8b-instruct",
            convergence_enabled=False,
            max_contexts=1,
            per_context_max=3000,
            total_max=3000,
        )

        result = run_pss(prompt, config, provider)
        output = result.leaves[0].output if result.leaves else ""

        # Clean up the output
        draft = output.strip()
        if draft.startswith("BEGIN THE SCENE:"):
            draft = draft[len("BEGIN THE SCENE:"):].strip()

        drafts[style] = draft
        total_tokens += result.total_usage.total_tokens

        # Show preview
        preview = draft[:500] + "..." if len(draft) > 500 else draft
        print(f"\n{preview}")
        print(f"\n[{result.total_usage.total_tokens} tokens]")

    # Output all drafts for the writer
    print()
    print("=" * 70)
    print("ALL DRAFTS")
    print("=" * 70)

    for style, draft in drafts.items():
        print(f"\n{'='*70}")
        print(f"DRAFT: {style.replace('_', ' ').upper()}")
        print("=" * 70)
        print()
        print(draft)

    # Save to file for the writer
    output_data = {
        "scene": scene,
        "generated_at": datetime.now().isoformat(),
        "total_tokens": total_tokens,
        "drafts": drafts,
    }

    output_path = os.path.join(os.path.dirname(__file__), "drafts_output.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated {len(drafts)} radically different drafts")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Drafts saved to: {output_path}")
    print()
    print("USE THESE DRAFTS:")
    print("  - Pick the one that resonates")
    print("  - Combine elements from multiple")
    print("  - Use them to discover what your scene COULD be")
    print("  - The cold clinical draft might reveal subtext")
    print("  - The dark comedy might find the truth in absurdity")


if __name__ == "__main__":
    main()
