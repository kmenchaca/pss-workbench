#!/usr/bin/env python
"""Run a single preset test for quick experimentation."""

import subprocess
import sys
from pathlib import Path

# Quick test prompts - one for each preset
QUICK_TESTS = {
    "explore": "What are the different approaches to learning a new programming language?",
    "write": "Write the opening paragraph of a story about finding an old letter in a book.",
    "research": "How do recommendation algorithms work?",
    "draft": "Write a brief product announcement for a new task management app.",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_single.py <preset> [custom_prompt]")
        print(f"Available presets: {', '.join(QUICK_TESTS.keys())}")
        print("\nExamples:")
        print("  python run_single.py explore")
        print('  python run_single.py write "A poem about debugging"')
        return 1

    preset = sys.argv[1]
    if preset not in QUICK_TESTS and preset not in ["explore", "write", "research", "draft"]:
        print(f"Unknown preset: {preset}")
        print(f"Available: {', '.join(QUICK_TESTS.keys())}")
        return 1

    # Use custom prompt if provided, otherwise use default
    prompt = sys.argv[2] if len(sys.argv) > 2 else QUICK_TESTS.get(preset, "Hello")

    print(f"Running: pss {preset}")
    print(f"Prompt: {prompt[:80]}...")
    print()

    cmd = [
        sys.executable,
        "-m",
        "pss",
        preset,
        prompt,
        "--max-tokens",
        "20000",
        "--verbose",
    ]

    subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
