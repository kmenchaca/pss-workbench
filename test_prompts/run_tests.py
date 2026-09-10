#!/usr/bin/env python
"""Run preset test prompts and capture output for analysis."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Test prompts for each preset
EXPLORE_PROMPTS = [
    (
        "tech_architecture",
        "What are the different architectural approaches to building a real-time collaborative document editor like Google Docs?",
    ),
    (
        "career_pivot",
        "What are the different paths someone with 10 years of experience in marketing could take to transition into tech?",
    ),
    (
        "housing_policy",
        "What are fundamentally different ways a city could address housing affordability?",
    ),
    (
        "sleep_theories",
        "What are the competing theories about why humans need sleep?",
    ),
    (
        "indie_games",
        "What are the different strategies a small indie game studio could use to compete against AAA publishers?",
    ),
]

WRITE_PROMPTS = [
    (
        "short_fiction",
        "Write the opening scene of a story about someone who discovers their childhood home is being demolished.",
    ),
    (
        "product_delay_email",
        "Write an email from a product manager explaining to stakeholders why a major feature needs to be delayed by 3 months.",
    ),
    (
        "learning_essay",
        "Write a short personal essay about the experience of learning something difficult as an adult.",
    ),
    (
        "smart_home_copy",
        "Write product copy for a new smart home device that helps people reduce their energy consumption.",
    ),
    (
        "er_nurse",
        "Write the opening of a piece about a day in the life of a night shift emergency room nurse.",
    ),
]


def run_test(preset: str, name: str, prompt: str, output_dir: Path) -> dict:
    """Run a single test and capture output."""
    print(f"\n{'='*60}")
    print(f"Running: {preset}/{name}")
    print(f"{'='*60}")

    output_file = output_dir / f"{preset}_{name}.txt"

    # Build command - use reasonable token budgets for testing
    cmd = [
        sys.executable,
        "-m",
        "pss",
        preset,
        prompt,
        "--max-tokens",
        "30000",  # Enough for decent exploration
        "--verbose",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=Path(__file__).parent.parent,
        )

        # Combine stdout and stderr
        full_output = f"COMMAND: {' '.join(cmd)}\n\n"
        full_output += f"PROMPT: {prompt}\n\n"
        full_output += "=" * 60 + "\n"
        full_output += "STDOUT:\n"
        full_output += "=" * 60 + "\n"
        full_output += result.stdout
        full_output += "\n" + "=" * 60 + "\n"
        full_output += "STDERR:\n"
        full_output += "=" * 60 + "\n"
        full_output += result.stderr
        full_output += f"\n\nEXIT CODE: {result.returncode}\n"

        # Save to file
        output_file.write_text(full_output, encoding="utf-8")
        print(f"Saved to: {output_file}")

        return {
            "name": name,
            "preset": preset,
            "exit_code": result.returncode,
            "output_file": str(output_file),
        }

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name}")
        return {
            "name": name,
            "preset": preset,
            "exit_code": -1,
            "error": "timeout",
        }
    except Exception as e:
        print(f"ERROR: {name} - {e}")
        return {
            "name": name,
            "preset": preset,
            "exit_code": -1,
            "error": str(e),
        }


def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / f"results_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")

    results = []

    # Run explore tests
    print("\n" + "=" * 60)
    print("EXPLORE PRESET TESTS")
    print("=" * 60)
    for name, prompt in EXPLORE_PROMPTS:
        result = run_test("explore", name, prompt, output_dir)
        results.append(result)

    # Run write tests
    print("\n" + "=" * 60)
    print("WRITE PRESET TESTS")
    print("=" * 60)
    for name, prompt in WRITE_PROMPTS:
        result = run_test("write", name, prompt, output_dir)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "✓" if r.get("exit_code") == 0 else "✗"
        print(f"  {status} {r['preset']}/{r['name']}")

    print(f"\nResults saved to: {output_dir}")

    # Create summary file
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"PSS Preset Test Results\n")
        f.write(f"Run at: {timestamp}\n\n")
        for r in results:
            f.write(f"{r['preset']}/{r['name']}: exit_code={r.get('exit_code')}\n")
            if "output_file" in r:
                f.write(f"  -> {r['output_file']}\n")
            if "error" in r:
                f.write(f"  -> ERROR: {r['error']}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
