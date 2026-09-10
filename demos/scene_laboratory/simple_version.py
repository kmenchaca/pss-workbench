#!/usr/bin/env python3
"""Scene Laboratory - Simple Version

Direct API calls, no PSS overhead. Just produce 5 different drafts.
"""

import os
import sys
import json
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pss.providers import create_provider

STYLES = {
    "cold_clinical": "Third person distant. Camera watching. No internal thoughts. Hemingway-style.",
    "raw_emotional": "First person stream of consciousness. Messy. Run-on sentences. Unhinged.",
    "dark_comedy": "Find the absurd in the tragic. Wrong details noticed. Uncomfortable humor.",
    "other_pov": "From HIS perspective. He's receiving this. His confusion, his denial.",
    "sensory_only": "Physical sensations only. No thoughts. Temperature, pressure, movement.",
}


def generate_draft(provider, scene: str, style_name: str, style_desc: str) -> str:
    """Generate one draft via provider."""
    messages = [
        {"role": "user", "content": f"""Write a 200-word scene. ONLY output the prose itself, nothing else.

SCENE: {scene}

STYLE: {style_desc}

Start the scene now. First word should be narrative prose:"""},
    ]

    response, tool_calls, tokens, usage = provider.chat(messages)
    return response


def main():
    scene = """She tells him the relationship is over.
His apartment. 2am. She's been working up to this for months.
He didn't see it coming."""

    print("=" * 60)
    print("SCENE LABORATORY")
    print("=" * 60)
    print()
    print("SCENE:")
    print(scene)
    print()

    provider = create_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")
    drafts = {}

    for style_name, style_desc in STYLES.items():
        print(f"\n{'='*60}")
        print(f"{style_name.upper().replace('_', ' ')}")
        print("=" * 60)

        try:
            draft = generate_draft(provider, scene, style_name, style_desc)
            drafts[style_name] = draft
            print(draft)
        except Exception as e:
            print(f"Error: {e}")
            drafts[style_name] = f"Error: {e}"

    # Save all drafts
    output = {
        "scene": scene,
        "timestamp": datetime.now().isoformat(),
        "drafts": drafts,
    }

    with open("drafts.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Saved {len(drafts)} drafts to drafts.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
