#!/usr/bin/env python3
"""Watch Mode - See the exploration tree grow in real-time.

Instead of just showing the final answer, watch mode shows:
- Branches spawning and exploring
- Each perspective developing
- Tensions surfacing between branches
- Synthesis happening

This is where the superpower becomes visible.
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pss.providers import Provider, create_provider


# =============================================================================
# Logging
# =============================================================================

class WatchLogger:
    """Logs watch mode runs to file."""

    def __init__(self, trace_dir: str = "traces/watch"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self.start_time = time.time()

    def log(self, event_type: str, data: dict):
        """Log an event."""
        self.events.append({
            "time": time.time() - self.start_time,
            "type": event_type,
            **data
        })

    def save(self, system: str, prompt: str) -> str:
        """Save the trace to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{system}_{timestamp}.json"
        filepath = self.trace_dir / filename

        trace = {
            "system": system,
            "prompt": prompt,
            "timestamp": timestamp,
            "duration": time.time() - self.start_time,
            "events": self.events,
        }

        with open(filepath, "w") as f:
            json.dump(trace, f, indent=2, default=str)

        return str(filepath)


# Global logger instance
_logger: Optional[WatchLogger] = None


def init_logger(trace_dir: str = "traces/watch"):
    """Initialize the logger."""
    global _logger
    _logger = WatchLogger(trace_dir)


def log_event(event_type: str, **data):
    """Log an event if logger is active."""
    if _logger:
        _logger.log(event_type, data)


def save_trace(system: str, prompt: str) -> Optional[str]:
    """Save trace if logger is active."""
    if _logger:
        return _logger.save(system, prompt)
    return None


# =============================================================================
# Visual Components
# =============================================================================

def clear_line():
    """Clear current line."""
    print("\r" + " " * 80 + "\r", end="")


def print_tree_header(prompt: str, system: str):
    """Print the tree header."""
    print()
    print("=" * 70)
    print(f"  EXPLORATION TREE: {system.upper()}")
    print("=" * 70)
    print(f"  Seed: {prompt[:60]}...")
    print("-" * 70)


def print_branch_spawn(branch_id: str, name: str, style: str):
    """Show a branch spawning."""
    print(f"  [+] Branch {branch_id}: {name} ({style})")
    log_event("branch_spawn", branch_id=branch_id, name=name, style=style)


def print_branch_exploring(branch_id: str, snippet: str):
    """Show a branch exploring."""
    # Truncate and clean snippet
    snippet = snippet.replace("\n", " ")[:50]
    print(f"      {branch_id} exploring: {snippet}...")


def print_branch_complete(branch_id: str, summary: str):
    """Show a branch completing."""
    summary_short = summary.replace("\n", " ")[:60]
    print(f"  [*] {branch_id} complete: {summary_short}...")
    log_event("branch_complete", branch_id=branch_id, summary=summary)


def print_tension(branch_a: str, branch_b: str, description: str):
    """Show a tension between branches."""
    print(f"  [!] TENSION: {branch_a} vs {branch_b}")
    print(f"      {description[:60]}...")
    log_event("tension", branch_a=branch_a, branch_b=branch_b, description=description)


def print_synthesis_start():
    """Show synthesis beginning."""
    print()
    print("-" * 70)
    print("  SYNTHESIS: Merging perspectives...")
    print("-" * 70)


def print_synthesis_step(step: str):
    """Show a synthesis step."""
    print(f"      > {step}")


def print_final_result(result: str):
    """Show the final result."""
    print()
    print("=" * 70)
    print("  RESULT")
    print("=" * 70)
    # Print wrapped
    words = result.split()
    line = "  "
    for word in words[:200]:  # Limit
        if len(line) + len(word) > 68:
            print(line)
            line = "  "
        line += word + " "
    if line.strip():
        print(line)
    if len(words) > 200:
        print("  ...")


# =============================================================================
# Watched Persona Ensemble
# =============================================================================

async def watch_personas(
    prompt: str,
    provider: Provider,
    num_branches: int = 4,
) -> dict[str, Any]:
    """Run persona ensemble with visible exploration."""
    from personas import PersonaLibrary, assign_personas

    print_tree_header(prompt, "persona ensemble")

    library = PersonaLibrary()

    # Phase 1: Spawn branches
    print("\n  Phase 1: SPAWNING BRANCHES")
    print("  " + "-" * 40)

    assignment = assign_personas(
        problem=prompt,
        num_branches=num_branches,
        library=library,
    )
    personas = list(assignment.assignments.values())

    for i, persona in enumerate(personas):
        print_branch_spawn(f"B{i+1}", persona.name, persona.thinking_style.value)
        await asyncio.sleep(0.3)  # Visual pause

    # Phase 2: Parallel exploration
    print("\n  Phase 2: PARALLEL EXPLORATION")
    print("  " + "-" * 40)

    responses = []
    tasks = []

    async def explore_branch(idx: int, persona):
        """Single branch exploration with progress."""
        messages = [
            {"role": "system", "content": persona.system_prompt},
            {"role": "user", "content": prompt},
        ]

        print(f"      B{idx+1} ({persona.name}) thinking...")

        response, _, _, _ = await provider.chat_async(messages)

        # Show completion
        first_sentence = response.split(".")[0] if "." in response else response[:50]
        print_branch_complete(f"B{idx+1}", first_sentence)

        return {"persona": persona, "response": response}

    # Run all branches in parallel
    results = await asyncio.gather(*[
        explore_branch(i, p) for i, p in enumerate(personas)
    ])
    responses = results

    # Phase 3: Surface tensions
    print("\n  Phase 3: SURFACING TENSIONS")
    print("  " + "-" * 40)

    tensions_found = []
    for i, r1 in enumerate(responses):
        for j, r2 in enumerate(responses):
            if i >= j:
                continue
            # Simple tension detection
            r1_lower = r1["response"].lower()
            r2_lower = r2["response"].lower()

            if ("risk" in r1_lower and "opportunity" in r2_lower) or \
               ("pessimist" in r1["persona"].name.lower() and "optimist" in r2["persona"].name.lower()):
                tension = f"{r1['persona'].name} sees risks, {r2['persona'].name} sees opportunities"
                print_tension(f"B{i+1}", f"B{j+1}", tension)
                tensions_found.append(tension)
                await asyncio.sleep(0.2)

    if not tensions_found:
        print("      No major tensions detected (perspectives aligned)")

    # Phase 4: Synthesis
    print_synthesis_start()

    print_synthesis_step("Collecting unique insights from each branch...")
    await asyncio.sleep(0.3)

    print_synthesis_step("Identifying common themes...")
    await asyncio.sleep(0.3)

    print_synthesis_step("Reconciling tensions...")
    await asyncio.sleep(0.3)

    # Build synthesis
    synthesis_prompt = f"Synthesize these perspectives on: {prompt}\n\n"
    for r in responses:
        synthesis_prompt += f"[{r['persona'].name}]: {r['response'][:400]}...\n\n"

    messages = [
        {"role": "system", "content": "Synthesize multiple perspectives into a balanced, actionable summary. Highlight where they agree and disagree."},
        {"role": "user", "content": synthesis_prompt},
    ]

    print_synthesis_step("Generating unified synthesis...")
    synthesis, _, _, _ = await provider.chat_async(messages)

    print_final_result(synthesis)

    return {
        "responses": responses,
        "tensions": tensions_found,
        "synthesis": synthesis,
    }


# =============================================================================
# Watched Argumentative Swarm
# =============================================================================

async def watch_argswarm(
    prompt: str,
    provider: Provider,
) -> dict[str, Any]:
    """Run argument swarm with visible debate."""
    from argswarm import assign_positions, Stance

    print_tree_header(prompt, "argumentative swarm")

    # Phase 1: Assign positions
    print("\n  Phase 1: ASSIGNING POSITIONS")
    print("  " + "-" * 40)

    positions = assign_positions(prompt, 2)
    for i, pos in enumerate(positions):
        stance = "FOR" if pos.stance == Stance.FOR else "AGAINST"
        print_branch_spawn(f"B{i+1}", stance, "adversarial")
        await asyncio.sleep(0.3)

    # Phase 2: Opening arguments
    print("\n  Phase 2: OPENING ARGUMENTS")
    print("  " + "-" * 40)

    arguments = []
    for i, pos in enumerate(positions):
        stance = "FOR" if pos.stance == Stance.FOR else "AGAINST"
        print(f"      B{i+1} ({stance}) constructing argument...")

        messages = [
            {"role": "system", "content": f"You are arguing {stance} the proposition. Make your strongest case."},
            {"role": "user", "content": f"Proposition: {prompt}\n\nPresent your opening argument."},
        ]
        response, _, _, _ = await provider.chat_async(messages)

        first_sentence = response.split(".")[0] if "." in response else response[:50]
        print_branch_complete(f"B{i+1}", first_sentence)

        arguments.append({"position": pos, "stance": stance, "argument": response})

    # Phase 3: Rebuttals
    print("\n  Phase 3: REBUTTALS")
    print("  " + "-" * 40)

    rebuttals = []
    for i, arg in enumerate(arguments):
        opponent = arguments[1 - i]  # The other one
        print(f"      B{i+1} ({arg['stance']}) rebutting {opponent['stance']}...")

        messages = [
            {"role": "system", "content": f"You are arguing {arg['stance']}. Attack your opponent's argument."},
            {"role": "user", "content": f"Opponent's argument:\n{opponent['argument'][:500]}\n\nRebut this."},
        ]
        response, _, _, _ = await provider.chat_async(messages)

        first_sentence = response.split(".")[0] if "." in response else response[:50]
        print_branch_complete(f"B{i+1}", f"Rebuts: {first_sentence}")

        rebuttals.append({"stance": arg["stance"], "rebuttal": response})

    # Phase 4: Show the clash
    print("\n  Phase 4: KEY CLASH POINTS")
    print("  " + "-" * 40)

    print_tension("B1 (FOR)", "B2 (AGAINST)", f"Core disagreement on: {prompt[:40]}...")

    # Phase 5: Adjudication
    print_synthesis_start()
    print("  (Judge evaluating arguments...)")
    print()

    debate_text = f"Proposition: {prompt}\n\n"
    for arg in arguments:
        debate_text += f"[{arg['stance']}]: {arg['argument'][:400]}...\n\n"
    for reb in rebuttals:
        debate_text += f"[{reb['stance']} REBUTS]: {reb['rebuttal'][:300]}...\n\n"

    messages = [
        {"role": "system", "content": "You are a debate judge. Evaluate the arguments and declare a winner with reasoning."},
        {"role": "user", "content": f"{debate_text}\n\nWho wins this debate and why?"},
    ]

    print_synthesis_step("Weighing arguments...")
    await asyncio.sleep(0.3)

    print_synthesis_step("Evaluating rebuttals...")
    await asyncio.sleep(0.3)

    print_synthesis_step("Rendering judgment...")
    judgment, _, _, _ = await provider.chat_async(messages)

    print_final_result(judgment)

    return {
        "arguments": arguments,
        "rebuttals": rebuttals,
        "judgment": judgment,
    }


# =============================================================================
# Watched Dream Logic
# =============================================================================

async def watch_dreamlogic(
    prompt: str,
    provider: Provider,
) -> dict[str, Any]:
    """Run dream logic with visible association chains."""

    print_tree_header(prompt, "dream logic")

    # Phase 1: Seed
    print("\n  Phase 1: PLANTING SEED")
    print("  " + "-" * 40)
    print(f"      Seed concept: {prompt}")
    await asyncio.sleep(0.5)

    # Phase 2: Free association
    print("\n  Phase 2: FREE ASSOCIATION (dream state)")
    print("  " + "-" * 40)

    messages = [
        {"role": "system", "content": "Think in dream logic. Make unexpected, surreal associations. Let one idea flow into another without conventional logic. Be creative and weird."},
        {"role": "user", "content": f"Starting from: {prompt}\n\nLet your mind wander freely. What strange connections emerge?"},
    ]

    print("      Entering dream state...")
    await asyncio.sleep(0.3)

    associations, _, _, _ = await provider.chat_async(messages)

    # Show associations emerging
    lines = associations.split("\n")[:8]
    for line in lines:
        if line.strip():
            print(f"      ~ {line.strip()[:60]}...")
            await asyncio.sleep(0.2)

    # Phase 3: Pattern recognition
    print("\n  Phase 3: PATTERN RECOGNITION (waking)")
    print("  " + "-" * 40)

    messages = [
        {"role": "system", "content": "You're waking from a creative dream. Extract the useful patterns and insights from the surreal associations. Ground them in reality."},
        {"role": "user", "content": f"Dream associations:\n{associations}\n\nWhat practical insights emerge from these strange connections?"},
    ]

    print("      Waking up...")
    await asyncio.sleep(0.3)

    print("      Recognizing patterns...")
    insights, _, _, _ = await provider.chat_async(messages)

    print_final_result(insights)

    return {
        "seed": prompt,
        "associations": associations,
        "insights": insights,
    }


# =============================================================================
# Main Watch Interface
# =============================================================================

async def watch(
    system: str,
    prompt: str,
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.1-8b-instruct",
    trace: bool = False,
    trace_dir: str = "traces/watch",
) -> dict[str, Any]:
    """Main watch interface.

    Args:
        system: System to run (personas, argswarm, dreamlogic)
        prompt: The prompt
        provider: LLM provider
        model: Model to use
        trace: Whether to save trace log
        trace_dir: Directory for trace files

    Returns:
        Result dictionary
    """
    # Initialize logging if tracing enabled
    if trace:
        init_logger(trace_dir)
        log_event("start", system=system, prompt=prompt, provider=provider, model=model)

    llm = create_provider(provider, model)

    if system == "personas":
        result = await watch_personas(prompt, llm)
    elif system == "argswarm":
        result = await watch_argswarm(prompt, llm)
    elif system == "dreamlogic":
        result = await watch_dreamlogic(prompt, llm)
    else:
        print(f"Watch mode not yet implemented for: {system}")
        print("Available: personas, argswarm, dreamlogic")
        return {}

    # Save trace if enabled
    if trace:
        log_event("complete", result_keys=list(result.keys()) if result else [])
        trace_path = save_trace(system, prompt)
        print(f"\n  [Trace saved to: {trace_path}]")

    return result


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Watch exploration trees grow in real-time"
    )
    parser.add_argument("system", choices=["personas", "argswarm", "dreamlogic"],
                       help="System to watch")
    parser.add_argument("prompt", help="The prompt to explore")
    parser.add_argument("--provider", "-p", default="openrouter",
                       help="LLM provider")
    parser.add_argument("--model", "-m", default="meta-llama/llama-3.1-8b-instruct",
                       help="Model to use")
    parser.add_argument("--trace", "-t", action="store_true",
                       help="Save trace log to traces/watch/")
    parser.add_argument("--trace-dir", default="traces/watch",
                       help="Directory for trace files")

    args = parser.parse_args()

    asyncio.run(watch(
        args.system,
        args.prompt,
        args.provider,
        args.model,
        trace=args.trace,
        trace_dir=args.trace_dir,
    ))


if __name__ == "__main__":
    main()
