#!/usr/bin/env python3
"""Live fire test for Learner-Controlled Harness.

Demonstrates the meta-learner controlling system selection and learning
from outcomes over time.

Usage:
    python demos/live_learner.py "Your question or task here"
    python demos/live_learner.py --multi "Task 1" "Task 2" "Task 3"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified import (
    LearnerControlledHarness,
    LearnerResult,
    classify_for_routing,
)


def print_classification(prompt: str):
    """Print classification info for a prompt."""
    info = classify_for_routing(prompt)

    print(f"\n[Classification]")
    print(f"  Task: {prompt[:60]}...")
    print(f"  Type: {info['classification']}")
    print(f"  Primary: {info['primary_system']}")
    print(f"  Systems: {', '.join(info['recommended_systems'])}")


def print_result(result: LearnerResult):
    """Print a learner result."""
    print(f"\n{'=' * 60}")
    print(f"RESULT")
    print(f"{'=' * 60}")

    print(f"\n[Classification]: {result.classification}")
    print(f"[Strategy]: {result.strategy_used}")
    print(f"[Confidence]: {result.confidence:.0%}")
    print(f"[Learning Recorded]: {result.learning_recorded}")

    if result.result.success:
        print(f"\n[Output]")
        output = result.result.output
        if isinstance(output, dict):
            for key, value in output.items():
                text = str(value)
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"  {key}: {text}")
        else:
            text = str(output)
            if len(text) > 500:
                text = text[:500] + "..."
            print(text)
    else:
        print(f"\n[Error]: {result.result.error}")

    print(f"\n[Stats]")
    print(f"  Elapsed: {result.result.elapsed_seconds:.2f}s")


async def run_single(harness: LearnerControlledHarness, prompt: str):
    """Run a single task."""
    print(f"\n[Processing]: {prompt[:60]}...")

    # Show classification
    print_classification(prompt)

    # Run with learner
    result = await harness.solve(prompt)

    # Print result
    print_result(result)

    return result


async def run_multiple(harness: LearnerControlledHarness, prompts: list[str]):
    """Run multiple tasks and show learning."""
    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'#' * 60}")
        print(f"# TASK {i}/{len(prompts)}")
        print(f"{'#' * 60}")

        result = await run_single(harness, prompt)
        results.append(result)

        # Show learning stats after each task
        stats = harness.get_stats()
        print(f"\n[Learning Stats]")
        print(f"  Total tasks: {stats.total_tasks}")
        print(f"  Success rate: {stats.success_rate:.0%}")
        if stats.system_performance:
            print(f"  System performance:")
            for sys, perf in stats.system_performance.items():
                print(f"    {sys}: {perf:.0%}")

    # Final suggestions
    print(f"\n{'=' * 60}")
    print("IMPROVEMENT SUGGESTIONS")
    print(f"{'=' * 60}")
    for suggestion in harness.suggest_improvements():
        print(f"  - {suggestion}")

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Run Learner-Controlled Harness live fire test"
    )
    parser.add_argument(
        "prompts",
        nargs="+",
        help="Task(s) to process"
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=["anthropic", "openrouter"],
        help="LLM provider"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/llama-3.1-8b-instruct",
        help="Model to use"
    )
    parser.add_argument(
        "--no-learn",
        action="store_true",
        help="Disable learning (just classify and route)"
    )

    args = parser.parse_args()

    print(f"[Config]")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Learning: {'disabled' if args.no_learn else 'enabled'}")
    print(f"  Tasks: {len(args.prompts)}")

    # Create harness
    harness = LearnerControlledHarness(
        provider=args.provider,
        model=args.model,
        learning_enabled=not args.no_learn,
    )

    # Run tasks
    if len(args.prompts) == 1:
        await run_single(harness, args.prompts[0])
    else:
        await run_multiple(harness, args.prompts)


if __name__ == "__main__":
    asyncio.run(main())
