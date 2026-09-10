#!/usr/bin/env python3
"""Live fire test for Argumentative Swarm.

Runs the debate system against a real proposition with actual LLM calls.
Uses the PSS provider system for LLM integration.

Usage:
    python demos/live_argswarm.py "Remote work is better than office work"
    python demos/live_argswarm.py --provider openrouter --rounds 2 "proposition"
"""

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pss.providers import Provider, create_provider
from argswarm import (
    Position,
    Stance,
    Argument,
    Refutation,
    Judgment,
    Crux,
    StrengthScore,
    assign_positions,
)


def estimate_strength(text: str) -> float:
    """Estimate argument strength from text heuristically."""
    score = 0.5  # Base score

    # Length bonus (longer arguments tend to be more detailed)
    if len(text) > 500:
        score += 0.1
    if len(text) > 1000:
        score += 0.1

    # Structure markers
    text_lower = text.lower()
    if "claim" in text_lower:
        score += 0.1
    if "evidence" in text_lower:
        score += 0.1
    if "because" in text_lower or "therefore" in text_lower:
        score += 0.05

    return min(score, 1.0)


# ============================================================================
# Helper Functions
# ============================================================================


def stance_to_str(stance: Stance) -> str:
    """Convert stance to readable string."""
    return {
        Stance.FOR: "FOR the proposition",
        Stance.AGAINST: "AGAINST the proposition",
        Stance.NEUTRAL: "NEUTRAL",
        Stance.STEELMAN_WEAK: "STEELMAN (strengthening weak arguments)",
        Stance.DEVILS_ADVOCATE: "DEVIL'S ADVOCATE",
        Stance.SYNTHESIS: "SYNTHESIS (finding common ground)",
    }.get(stance, str(stance))


# ============================================================================
# LLM Integration
# ============================================================================


async def generate_opening_argument(
    provider: Provider,
    position: Position,
    proposition: str,
) -> Argument:
    """Generate opening argument for a position."""
    stance_desc = stance_to_str(position.stance)

    messages = [
        {"role": "system", "content": f"""You are a skilled debater arguing {stance_desc}.

Your goal is to construct the strongest possible argument for your position.
Be persuasive, use evidence and reasoning, and anticipate counterarguments.

Structure your argument clearly with:
1. CLAIM: Your main assertion (1-2 sentences)
2. EVIDENCE: Supporting facts, data, or examples
3. WARRANT: Why the evidence supports your claim"""},
        {"role": "user", "content": f"""Proposition: {proposition}

Your position: {stance_desc}

Construct your opening argument. Be specific and compelling."""}
    ]

    text, _, tokens_used, _ = await provider.chat_async(messages)

    # Parse basic structure from response
    claim = text[:200] if len(text) > 200 else text
    if "CLAIM:" in text.upper():
        parts = text.upper().split("CLAIM:")
        if len(parts) > 1:
            claim_part = parts[1].split("EVIDENCE:")[0] if "EVIDENCE:" in parts[1] else parts[1][:200]
            claim = claim_part.strip()

    return Argument(
        id=f"arg_{position.id}_opening",
        position_id=position.id,
        claim=claim,
        evidence=text,  # Full text as evidence for now
        warrant="",
        strength=estimate_strength(text),
        round_number=1,
    )


async def generate_rebuttal(
    provider: Provider,
    position: Position,
    target_argument: Argument,
    all_positions: list[Position],
    proposition: str,
) -> Refutation:
    """Generate a rebuttal to an opponent's argument."""
    stance_desc = stance_to_str(position.stance)

    # Find target position name
    target_pos = next((p for p in all_positions if p.id == target_argument.position_id), None)
    target_desc = stance_to_str(target_pos.stance) if target_pos else "opponent"

    messages = [
        {"role": "system", "content": f"""You are a skilled debater arguing {stance_desc}.

Your opponent ({target_desc}) has made an argument. Your goal is to refute it effectively.
Attack the weakest points, challenge the evidence, or show why their reasoning fails.
Be respectful but incisive."""},
        {"role": "user", "content": f"""Proposition: {proposition}

Opponent's argument:
{target_argument.evidence[:800]}

Construct your rebuttal. Identify specific weaknesses and counter them."""}
    ]

    text, _, tokens_used, _ = await provider.chat_async(messages)

    return Refutation(
        id=f"refute_{uuid.uuid4().hex[:8]}",
        target_argument_id=target_argument.id,
        counter_claim=text[:300],
        evidence=text,
        refutation_type="rebut",
        strength=estimate_strength(text),
        position_id=position.id,
        round_number=2,
    )


async def generate_closing(
    provider: Provider,
    position: Position,
    my_arguments: list[Argument],
    refutations_against_me: list[Refutation],
    proposition: str,
) -> str:
    """Generate closing statement."""
    stance_desc = stance_to_str(position.stance)

    my_arg_text = "\n".join(a.evidence[:300] + "..." for a in my_arguments) if my_arguments else "None"
    ref_text = "\n".join(r.evidence[:300] + "..." for r in refutations_against_me) if refutations_against_me else "None"

    messages = [
        {"role": "system", "content": f"""You are a skilled debater arguing {stance_desc}.

This is your closing statement. Summarize your strongest points, address any
criticisms that were raised, and make a compelling final appeal."""},
        {"role": "user", "content": f"""Proposition: {proposition}

Your arguments so far:
{my_arg_text}

Criticisms raised against you:
{ref_text}

Deliver your closing statement (2-3 paragraphs)."""}
    ]

    text, _, _, _ = await provider.chat_async(messages)
    return text


async def run_adjudication(
    provider: Provider,
    proposition: str,
    positions: list[Position],
    arguments: list[Argument],
    refutations: list[Refutation],
    closings: dict[str, str],
) -> Judgment:
    """Run final adjudication on the debate."""

    # Format debate summary
    debate_parts = []
    for pos in positions:
        stance_desc = stance_to_str(pos.stance)
        pos_args = [a for a in arguments if a.position_id == pos.id]
        pos_refs = [r for r in refutations if r.position_id == pos.id]

        debate_parts.append(f"\n=== {stance_desc} ===")
        if pos_args:
            debate_parts.append("Arguments:")
            for arg in pos_args:
                debate_parts.append(f"  - {arg.evidence[:300]}...")
        if pos_refs:
            debate_parts.append("Rebuttals made:")
            for ref in pos_refs:
                debate_parts.append(f"  - {ref.evidence[:200]}...")
        if pos.id in closings:
            debate_parts.append(f"Closing: {closings[pos.id][:300]}...")

    debate_summary = "\n".join(debate_parts)

    messages = [
        {"role": "system", "content": """You are an impartial judge evaluating a debate.

Assess the arguments on their merits:
- Strength of evidence
- Logical validity
- Effective rebuttals
- Persuasiveness

Determine which side presented the stronger overall case.
Be specific about WHY one side won (or if it's a tie)."""},
        {"role": "user", "content": f"""Proposition: {proposition}

DEBATE TRANSCRIPT:
{debate_summary}

Render your judgment:
1. WINNER: State "FOR", "AGAINST", or "TIE"
2. KEY CRUXES: The central points of disagreement (bullet points)
3. REASONING: Explain your decision (2-3 paragraphs)"""}
    ]

    text, _, tokens_used, _ = await provider.chat_async(messages)

    # Parse winner from text
    winner_id = None
    text_upper = text.upper()
    if "WINNER: FOR" in text_upper or "WINNER:FOR" in text_upper:
        winner_id = next((p.id for p in positions if p.stance == Stance.FOR), None)
    elif "WINNER: AGAINST" in text_upper or "WINNER:AGAINST" in text_upper:
        winner_id = next((p.id for p in positions if p.stance == Stance.AGAINST), None)

    # Extract cruxes
    cruxes = []
    if "CRUX" in text_upper:
        crux_section = text.split("CRUX")[1] if len(text.split("CRUX")) > 1 else ""
        for line in crux_section.split("\n")[:5]:
            line = line.strip().lstrip("-•*")
            if line and len(line) > 10:
                cruxes.append(Crux(
                    id=f"crux_{uuid.uuid4().hex[:6]}",
                    description=line.strip(),
                    positions_involved=[p.id for p in positions],
                ))

    # Calculate strength scores
    strength_scores = []
    for pos in positions:
        pos_args = [a for a in arguments if a.position_id == pos.id]
        avg_strength = sum(a.strength for a in pos_args) / len(pos_args) if pos_args else 0.5
        strength_scores.append(StrengthScore(
            position_id=pos.id,
            score=avg_strength,
            argument_quality=avg_strength,
        ))

    return Judgment(
        winner=winner_id,
        reasoning=text,
        key_cruxes=cruxes,
        strength_scores=strength_scores,
    )


async def run_debate(
    provider: Provider,
    proposition: str,
    num_rounds: int = 2,
) -> dict:
    """Run a full debate.

    Args:
        provider: LLM provider to use
        proposition: The debate proposition
        num_rounds: Number of debate rounds

    Returns:
        Dict with positions, arguments, judgment, and stats
    """
    start_time = time.time()
    total_tokens = 0

    # Assign positions (FOR and AGAINST)
    positions = assign_positions(proposition, 2)
    print(f"\n[Positions Assigned]")
    for pos in positions:
        print(f"  - {pos.id}: {stance_to_str(pos.stance)}")

    all_arguments: list[Argument] = []
    all_refutations: list[Refutation] = []

    # Round 1: Opening arguments
    print(f"\n[Round 1: Opening Arguments]")
    opening_tasks = [
        generate_opening_argument(provider, pos, proposition)
        for pos in positions
    ]
    opening_arguments = await asyncio.gather(*opening_tasks)

    for arg in opening_arguments:
        all_arguments.append(arg)
        pos = next(p for p in positions if p.id == arg.position_id)
        print(f"  {stance_to_str(pos.stance)}: {arg.claim[:80]}...")

    # Round 2: Rebuttals
    if num_rounds >= 2:
        print(f"\n[Round 2: Rebuttals]")
        rebuttal_tasks = []

        for pos in positions:
            # Find opponent arguments to rebut
            opponent_args = [a for a in all_arguments if a.position_id != pos.id]
            if opponent_args:
                target = max(opponent_args, key=lambda a: a.strength)
                rebuttal_tasks.append(
                    generate_rebuttal(provider, pos, target, positions, proposition)
                )

        if rebuttal_tasks:
            rebuttals = await asyncio.gather(*rebuttal_tasks)
            for refutation in rebuttals:
                all_refutations.append(refutation)
                pos = next(p for p in positions if p.id == refutation.position_id)
                print(f"  {stance_to_str(pos.stance)} rebuts: {refutation.counter_claim[:80]}...")

    # Closing statements
    print(f"\n[Closing Statements]")
    closings = {}
    closing_tasks = []
    for pos in positions:
        my_args = [a for a in all_arguments if a.position_id == pos.id]
        refs_against = [r for r in all_refutations
                       if any(a.id == r.target_argument_id for a in my_args)]
        closing_tasks.append(generate_closing(provider, pos, my_args, refs_against, proposition))

    closing_texts = await asyncio.gather(*closing_tasks)
    for pos, closing in zip(positions, closing_texts):
        closings[pos.id] = closing
        print(f"  {stance_to_str(pos.stance)}: {closing[:80]}...")

    # Adjudication
    print(f"\n[Adjudication]")
    judgment = await run_adjudication(
        provider, proposition, positions, all_arguments, all_refutations, closings
    )

    elapsed = time.time() - start_time

    return {
        "proposition": proposition,
        "positions": positions,
        "arguments": all_arguments,
        "refutations": all_refutations,
        "closings": closings,
        "judgment": judgment,
        "stats": {
            "elapsed_seconds": elapsed,
            "num_rounds": num_rounds,
            "num_arguments": len(all_arguments),
            "num_refutations": len(all_refutations),
        }
    }


def print_results(results: dict):
    """Pretty print the debate results."""
    print("\n" + "=" * 80)
    print("ARGUMENTATIVE SWARM RESULTS")
    print("=" * 80)

    print(f"\n[Proposition]\n{results['proposition']}")

    print(f"\n[Arguments]")
    for arg in results["arguments"]:
        pos = next(p for p in results["positions"] if p.id == arg.position_id)
        print(f"\n{'-' * 40}")
        print(f"[{stance_to_str(pos.stance)}] (strength: {arg.strength:.2f})")
        print(f"{'-' * 40}")
        print(arg.evidence[:500] + "..." if len(arg.evidence) > 500 else arg.evidence)

    if results["refutations"]:
        print(f"\n[Refutations]")
        for ref in results["refutations"]:
            pos = next(p for p in results["positions"] if p.id == ref.position_id)
            print(f"\n{'-' * 40}")
            print(f"[{stance_to_str(pos.stance)} REBUTS] (strength: {ref.strength:.2f})")
            print(f"{'-' * 40}")
            print(ref.evidence[:400] + "..." if len(ref.evidence) > 400 else ref.evidence)

    print(f"\n{'=' * 40}")
    print("[JUDGMENT]")
    print(f"{'=' * 40}")
    judgment = results["judgment"]

    # Determine winner string
    if judgment.winner:
        winner_pos = next((p for p in results["positions"] if p.id == judgment.winner), None)
        winner_str = stance_to_str(winner_pos.stance) if winner_pos else "Unknown"
    else:
        winner_str = "TIE / NO CLEAR WINNER"

    print(f"\nWinner: {winner_str}")

    if judgment.key_cruxes:
        print(f"\nKey Cruxes:")
        for crux in judgment.key_cruxes[:5]:
            print(f"  - {crux.description}")

    print(f"\nReasoning:")
    print(judgment.reasoning[:1000] + "..." if len(judgment.reasoning) > 1000 else judgment.reasoning)

    print(f"\n[Stats]")
    stats = results["stats"]
    print(f"  Elapsed: {stats['elapsed_seconds']:.2f}s")
    print(f"  Rounds: {stats['num_rounds']}")
    print(f"  Arguments: {stats['num_arguments']}")
    print(f"  Refutations: {stats['num_refutations']}")


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description="Run Argumentative Swarm live fire test")
    parser.add_argument("proposition", help="The proposition to debate")
    parser.add_argument("--provider", default="openrouter", choices=["anthropic", "openrouter"],
                       help="LLM provider to use")
    parser.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct",
                       help="Model to use")
    parser.add_argument("--rounds", type=int, default=2,
                       help="Number of debate rounds")
    args = parser.parse_args()

    print(f"[Config]")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Proposition: {args.proposition[:60]}...")
    print(f"  Rounds: {args.rounds}")

    # Create provider
    provider = create_provider(args.provider, args.model)

    # Run debate
    results = await run_debate(
        provider=provider,
        proposition=args.proposition,
        num_rounds=args.rounds,
    )

    # Print results
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
