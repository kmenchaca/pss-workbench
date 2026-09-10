#!/usr/bin/env python3
"""Live fire test for Persona Ensemble.

Runs the persona ensemble against a real problem with actual LLM calls.
Uses the PSS provider system for LLM integration.

Usage:
    python demos/live_personas.py "Should we adopt a 4-day work week?"
    python demos/live_personas.py --provider openrouter --model meta-llama/llama-3.1-8b-instruct "topic"
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pss.providers import Provider, create_provider
import re

from personas import (
    Persona,
    PersonaResponse,
    PersonaLibrary,
    assign_personas,
    synthesize_perspectives,
    multi_perspective_brief,
    check_consistency,
    PESSIMIST,
    OPTIMIST,
    SECURITY_ENGINEER,
    FIRST_PRINCIPLES,
    NOVICE,
    VETERAN,
)


# Simple text extraction helpers (the library functions expect PersonaResponse objects)
def extract_text_patterns(text: str, patterns: list[str]) -> list[str]:
    """Extract sentences matching patterns from text."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    matches = []
    combined = "|".join(f"({p})" for p in patterns)
    for s in sentences:
        if re.search(combined, s, re.IGNORECASE):
            matches.append(s)
    return matches[:5]


def extract_assumptions_from_text(text: str) -> list[str]:
    patterns = [r"assum", r"expect", r"presum", r"given that", r"if we"]
    return extract_text_patterns(text, patterns)


def extract_risks_from_text(text: str) -> list[str]:
    patterns = [r"risk", r"danger", r"fail", r"problem", r"concern", r"could go wrong"]
    return extract_text_patterns(text, patterns)


def extract_opportunities_from_text(text: str) -> list[str]:
    patterns = [r"opportunit", r"benefit", r"advantage", r"upside", r"potential"]
    return extract_text_patterns(text, patterns)


# ============================================================================
# LLM Integration
# ============================================================================


@dataclass
class EnrichedResponse:
    """Enriched response with additional metadata."""
    response: PersonaResponse
    persona_name: str
    tokens_used: int
    branch_id: str


async def run_persona_branch(
    provider: Provider,
    persona: Persona,
    problem: str,
    branch_id: str,
) -> EnrichedResponse:
    """Run a single persona branch with real LLM calls.

    Args:
        provider: LLM provider to use
        persona: The persona for this branch
        problem: The problem/question to analyze
        branch_id: Unique identifier for this branch

    Returns:
        EnrichedResponse with the persona's analysis and metadata
    """
    # Build messages with persona system prompt
    messages = [
        {"role": "system", "content": persona.system_prompt},
        {"role": "user", "content": f"""Analyze this question/problem from your perspective:

{problem}

Provide your analysis including:
1. Your key observations and insights
2. Assumptions you're making
3. Risks you see
4. Opportunities you identify
5. Your recommendation

Stay true to your persona's perspective and thinking style throughout."""}
    ]

    # Call LLM
    text, tool_calls, tokens_used, usage = await provider.chat_async(messages)

    # Extract worldview elements from the response text
    assumptions = extract_assumptions_from_text(text)
    risks = extract_risks_from_text(text)
    opportunities = extract_opportunities_from_text(text)

    response = PersonaResponse(
        persona_id=persona.id,
        content=text,
        assumptions=assumptions,
        risks_identified=risks,
        opportunities=opportunities,
        confidence=0.8,
        raw_response=text,
    )

    return EnrichedResponse(
        response=response,
        persona_name=persona.name,
        tokens_used=tokens_used,
        branch_id=branch_id,
    )


async def run_ensemble(
    provider: Provider,
    problem: str,
    num_personas: int = 4,
    persona_names: list[str] | None = None,
) -> dict:
    """Run the full persona ensemble.

    Args:
        provider: LLM provider to use
        problem: The problem/question to analyze
        num_personas: Number of personas to use (if not specified)
        persona_names: Specific persona names to use (optional)

    Returns:
        Dict with responses, synthesis, and stats
    """
    start_time = time.time()
    library = PersonaLibrary()

    # Get personas
    if persona_names:
        personas = []
        for name in persona_names:
            persona = library.get_persona(name.lower())
            if persona:
                personas.append(persona)
            else:
                print(f"Warning: Persona '{name}' not found, skipping")
        print(f"\n[Using Specified Personas]")
    else:
        # Use assign_personas to select appropriate personas
        assignment = assign_personas(problem, num_personas, library)
        # assignments is a dict of branch_id -> persona
        personas = list(assignment.assignments.values())
        print(f"\n[Auto-Assigned Personas]")

    for p in personas:
        style = p.thinking_style.value if hasattr(p.thinking_style, 'value') else p.thinking_style
        print(f"  - {p.name}: {style}")

    # Run all personas in parallel
    print(f"\n[Running {len(personas)} branches in parallel...]")
    tasks = [
        run_persona_branch(provider, persona, problem, f"branch_{i}")
        for i, persona in enumerate(personas)
    ]
    enriched_responses = await asyncio.gather(*tasks)

    # Check consistency
    print(f"\n[Consistency Scores]")
    for enriched in enriched_responses:
        persona = next(p for p in personas if p.id == enriched.response.persona_id)
        score = check_consistency(persona, enriched.response)
        assessment = "consistent" if score.score > 0.7 else "needs reinforcement" if score.score > 0.4 else "drifted"
        print(f"  - {enriched.persona_name}: {score.score:.2f} ({assessment})")

    # Extract raw responses for synthesis
    raw_responses = [e.response for e in enriched_responses]

    # Synthesize perspectives
    print(f"\n[Synthesizing perspectives...]")
    synthesis = synthesize_perspectives(raw_responses)
    brief = multi_perspective_brief(raw_responses, library)  # Pass library, not synthesis

    elapsed = time.time() - start_time
    total_tokens = sum(e.tokens_used for e in enriched_responses)

    return {
        "problem": problem,
        "personas": personas,
        "enriched_responses": enriched_responses,
        "synthesis": synthesis,
        "brief": brief,
        "stats": {
            "elapsed_seconds": elapsed,
            "total_tokens": total_tokens,
            "num_branches": len(personas),
        }
    }


def safe_print(text: str):
    """Print text safely, handling encoding issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def print_results(results: dict):
    """Pretty print the ensemble results."""
    safe_print("\n" + "=" * 80)
    safe_print("PERSONA ENSEMBLE RESULTS")
    safe_print("=" * 80)

    safe_print(f"\n[Problem]\n{results['problem']}")

    safe_print(f"\n[Individual Responses]")
    for enriched in results["enriched_responses"]:
        response = enriched.response
        safe_print(f"\n{'-' * 40}")
        safe_print(f"[{enriched.persona_name}]")
        safe_print(f"{'-' * 40}")
        content = response.content
        safe_print(content[:600] + "..." if len(content) > 600 else content)
        if response.assumptions:
            safe_print(f"\nKey Assumptions: {response.assumptions[:3]}")
        if response.risks_identified:
            safe_print(f"Risks Identified: {response.risks_identified[:3]}")
        if response.opportunities:
            safe_print(f"Opportunities: {response.opportunities[:3]}")

    safe_print(f"\n{'=' * 40}")
    safe_print("[SYNTHESIS]")
    safe_print(f"{'=' * 40}")
    synthesis = results["synthesis"]
    # synthesis is a string
    safe_print(synthesis[:1500] + "..." if len(synthesis) > 1500 else synthesis)

    safe_print(f"\n{'=' * 40}")
    safe_print("[MULTI-PERSPECTIVE BRIEF]")
    safe_print(f"{'=' * 40}")
    brief = results["brief"]
    safe_print(brief[:2000] + "..." if len(brief) > 2000 else brief)

    safe_print(f"\n[Stats]")
    stats = results["stats"]
    safe_print(f"  Elapsed: {stats['elapsed_seconds']:.2f}s")
    safe_print(f"  Total Tokens: {stats['total_tokens']}")
    safe_print(f"  Branches: {stats['num_branches']}")


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description="Run Persona Ensemble live fire test")
    parser.add_argument("problem", help="The problem/question to analyze")
    parser.add_argument("--provider", default="openrouter", choices=["anthropic", "openrouter"],
                       help="LLM provider to use")
    parser.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct",
                       help="Model to use")
    parser.add_argument("--num-personas", type=int, default=4,
                       help="Number of personas to run")
    parser.add_argument("--personas", nargs="+",
                       help="Specific persona IDs to use (e.g., pessimist optimist security_engineer)")
    args = parser.parse_args()

    print(f"[Config]")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Problem: {args.problem[:60]}...")

    # Create provider
    provider = create_provider(args.provider, args.model)

    # Run ensemble
    results = await run_ensemble(
        provider=provider,
        problem=args.problem,
        num_personas=args.num_personas,
        persona_names=args.personas,
    )

    # Print results
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
