#!/usr/bin/env python3
"""Command Line Interface for the Unified Experimental Systems.

Usage:
    python -m unified argswarm "AI will replace most jobs"
    python -m unified personas "Analyze the risks of launching feature X"
    python -m unified --list
    python -m unified --auto "What should we do about climate change?"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import list_all_systems, get_system_type
from .harness import UnifiedHarness, SystemResult
from .router import SystemRouter


def print_systems():
    """Print all available systems."""
    print("\nAvailable Experimental Systems:")
    print("=" * 60)

    for sys_info in list_all_systems():
        print(f"\n  {sys_info['id']}")
        print(f"    {sys_info['name']}")
        print(f"    {sys_info['description']}")
        if sys_info['best_for']:
            print(f"    Best for: {', '.join(sys_info['best_for'][:3])}")

    print("\n" + "=" * 60)


def print_result(result: SystemResult, verbose: bool = True):
    """Print a system result."""
    print("\n" + "=" * 60)
    print(f"SYSTEM: {result.system_type.value}")
    print("=" * 60)

    if not result.success:
        print(f"\n[ERROR] {result.error}")
        return

    output = result.output

    if isinstance(output, dict):
        # Pretty print dict output
        for key, value in output.items():
            print(f"\n[{key.upper()}]")
            if isinstance(value, list):
                for i, item in enumerate(value, 1):
                    if isinstance(item, dict):
                        for k, v in item.items():
                            text = str(v)[:500] + "..." if len(str(v)) > 500 else str(v)
                            print(f"  {k}: {text}")
                        print()
                    else:
                        print(f"  {i}. {item}")
            elif isinstance(value, str):
                # Truncate long strings
                if len(value) > 1000:
                    print(value[:1000] + "...")
                else:
                    print(value)
            else:
                print(value)
    else:
        print(output)

    if verbose:
        print("\n" + "-" * 40)
        print("[STATS]")
        print(f"  Elapsed: {result.elapsed_seconds:.2f}s")
        print(f"  Tokens: ~{result.tokens_used}")
        if result.metadata:
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Interface for PSS Experimental Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  unified argswarm "AI will replace most jobs"
  unified personas "Analyze the risks of feature X"
  unified --auto "How should we approach this problem?"
  unified --list

Systems:
  argswarm   - Debate system with FOR/AGAINST positions
  personas   - Multi-persona perspective analysis
  redteam    - Security vulnerability analysis
  evolution  - Evolutionary code optimization
  temporal   - Future scenario planning
  knowledge  - Knowledge graph construction
  embodied   - Physical action planning
  metalearner - Strategy optimization
  dreamlogic - Creative association generation
  negotiate  - Multi-party negotiation simulation
"""
    )

    parser.add_argument(
        "system",
        nargs="?",
        help="System to run (e.g., argswarm, personas)"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to process"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available systems"
    )
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="Auto-route to best system for the prompt"
    )
    parser.add_argument(
        "--provider", "-p",
        default="openrouter",
        choices=["anthropic", "openrouter"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model", "-m",
        default="meta-llama/llama-3.1-8b-instruct",
        help="Model to use"
    )
    parser.add_argument(
        "--branches", "-b",
        type=int,
        default=4,
        help="Number of branches (where applicable)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--multi",
        nargs="+",
        help="Run multiple systems on the same prompt"
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        print_systems()
        return

    # Handle --auto with prompt as first positional
    if args.auto:
        if args.system:
            prompt = args.system + (" " + args.prompt if args.prompt else "")
        else:
            print("Error: --auto requires a prompt")
            sys.exit(1)

        print(f"\n[Auto-routing prompt...]")
        router = SystemRouter()
        decision = router.route(prompt)
        print(f"  Selected: {decision.system_type.value}")
        print(f"  Confidence: {decision.confidence:.0%}")
        print(f"  Reasoning: {decision.reasoning}")

        args.system = decision.system_type.value
        args.prompt = prompt

    # Validate args
    if not args.system:
        parser.print_help()
        sys.exit(1)

    if not args.prompt and not args.multi:
        print("Error: prompt is required")
        sys.exit(1)

    # Handle --multi
    if args.multi:
        systems = args.multi
        prompt = args.system  # First positional is prompt in this case
    else:
        systems = [args.system]
        prompt = args.prompt

    # Validate systems
    for sys_name in systems:
        if not get_system_type(sys_name):
            print(f"Error: Unknown system '{sys_name}'")
            print("Use --list to see available systems")
            sys.exit(1)

    # Run
    harness = UnifiedHarness(provider=args.provider, model=args.model)

    async def run():
        if len(systems) == 1:
            return [await harness.run(systems[0], prompt)]
        else:
            return await harness.run_multiple(systems, prompt)

    if not args.quiet:
        print(f"\n[Config]")
        print(f"  Provider: {args.provider}")
        print(f"  Model: {args.model}")
        print(f"  System(s): {', '.join(systems)}")
        print(f"  Prompt: {prompt[:60]}...")

    results = asyncio.run(run())

    # Output
    if args.json:
        output = []
        for result in results:
            output.append({
                "system": result.system_type.value,
                "success": result.success,
                "output": result.output,
                "metadata": result.metadata,
                "elapsed": result.elapsed_seconds,
                "error": result.error,
            })
        print(json.dumps(output, indent=2, default=str))
    else:
        for result in results:
            print_result(result, verbose=not args.quiet)


if __name__ == "__main__":
    main()
