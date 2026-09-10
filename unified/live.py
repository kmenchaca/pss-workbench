#!/usr/bin/env python3
"""Live Renderer - Watch the exploration tree grow in real-time.

This is where the superpower becomes visible.

Usage:
    python unified/live.py personas "Should we build or buy?"
    python unified/live.py argswarm "TypeScript is better than JavaScript"
    python unified/live.py dreamlogic "enterprise sales strategy"
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich import box

from pss.providers import Provider, create_provider


# =============================================================================
# State Management
# =============================================================================

@dataclass
class Branch:
    """A single branch in the exploration tree."""
    id: str
    name: str
    style: str
    spawned_at: float
    completed_at: Optional[float] = None
    summary: Optional[str] = None
    full_response: Optional[str] = None


@dataclass
class Tension:
    """A tension between branches."""
    branch_a: str
    branch_b: str
    description: str
    detected_at: float


@dataclass
class ExplorationState:
    """Current state of the exploration."""
    system: str
    prompt: str
    start_time: float = field(default_factory=time.time)
    branches: dict[str, Branch] = field(default_factory=dict)
    tensions: list[Tension] = field(default_factory=list)
    synthesis: Optional[str] = None
    phase: str = "initializing"
    phase_detail: str = ""

    def elapsed(self) -> float:
        return time.time() - self.start_time


# =============================================================================
# Live Renderer
# =============================================================================

class LiveRenderer:
    """Renders exploration state in real-time."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.state: Optional[ExplorationState] = None

    def render(self) -> Panel:
        """Render current state as a Rich Panel."""
        if not self.state:
            return Panel("Initializing...")

        content = Text()

        # Header
        content.append(f"Query: ", style="dim")
        content.append(f"{self.state.prompt}\n\n", style="bold")

        # Phase indicator
        phase_icons = {
            "initializing": "...",
            "spawning": "🌱",
            "exploring": "🔍",
            "tensions": "⚡",
            "synthesis": "🔮",
            "complete": "✓",
        }
        icon = phase_icons.get(self.state.phase, "•")
        content.append(f"{icon} {self.state.phase.upper()}", style="bold cyan")
        if self.state.phase_detail:
            content.append(f" — {self.state.phase_detail}", style="dim")
        content.append("\n\n")

        # Branches
        for branch_id, branch in self.state.branches.items():
            elapsed = branch.spawned_at - self.state.start_time

            if branch.completed_at:
                # Completed branch
                content.append(f"[{elapsed:5.1f}s] ", style="dim")
                content.append("✓ ", style="green")
                content.append(f"{branch.name}", style="bold")
                content.append(f" ({branch.style})\n", style="dim")

                if branch.summary:
                    # Wrap the summary nicely
                    summary = branch.summary[:200]
                    if len(branch.summary) > 200:
                        summary += "..."
                    lines = self._wrap_text(summary, 60)
                    for line in lines:
                        content.append(f"         {line}\n", style="italic")
                content.append("\n")
            else:
                # In-progress branch
                content.append(f"[{elapsed:5.1f}s] ", style="dim")
                content.append("◌ ", style="yellow")
                content.append(f"{branch.name}", style="bold yellow")
                content.append(f" ({branch.style}) ", style="dim")
                content.append("thinking...\n", style="dim italic")

        # Tensions
        if self.state.tensions:
            content.append("\n")
            for tension in self.state.tensions:
                elapsed = tension.detected_at - self.state.start_time
                content.append(f"[{elapsed:5.1f}s] ", style="dim")
                content.append("⚡ TENSION: ", style="bold red")
                content.append(f"{tension.branch_a} vs {tension.branch_b}\n", style="red")
                lines = self._wrap_text(tension.description, 55)
                for line in lines:
                    content.append(f"         {line}\n", style="italic red")

        # Synthesis
        if self.state.synthesis:
            content.append("\n")
            content.append("═" * 50 + "\n", style="bold blue")
            content.append("SYNTHESIS\n", style="bold blue")
            content.append("═" * 50 + "\n\n", style="bold blue")

            lines = self._wrap_text(self.state.synthesis, 50)
            for line in lines:
                content.append(f"{line}\n")

        # Footer with elapsed time
        content.append(f"\n[{self.state.elapsed():.1f}s elapsed]", style="dim")

        title = f"🌳 {self.state.system.upper()}"
        return Panel(content, title=title, border_style="blue", box=box.ROUNDED)

    def _wrap_text(self, text: str, width: int) -> list[str]:
        """Wrap text to specified width."""
        text = text.replace("\n", " ")
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines or [""]


# =============================================================================
# Live Persona Ensemble
# =============================================================================

async def live_personas(
    prompt: str,
    provider: Provider,
    renderer: LiveRenderer,
    live: Live,
) -> dict[str, Any]:
    """Run persona ensemble with live rendering."""
    from personas import PersonaLibrary, assign_personas

    state = ExplorationState(system="personas", prompt=prompt)
    renderer.state = state

    library = PersonaLibrary()

    # Phase 1: Spawn branches
    state.phase = "spawning"
    state.phase_detail = "Assigning personas to branches"
    live.refresh()

    assignment = assign_personas(
        problem=prompt,
        num_branches=4,
        library=library,
    )
    personas = list(assignment.assignments.values())

    for i, persona in enumerate(personas):
        branch = Branch(
            id=f"B{i+1}",
            name=persona.name,
            style=persona.thinking_style.value,
            spawned_at=time.time(),
        )
        state.branches[branch.id] = branch
        live.refresh()
        await asyncio.sleep(0.3)

    # Phase 2: Parallel exploration
    state.phase = "exploring"
    state.phase_detail = "Branches thinking in parallel"
    live.refresh()

    responses = []

    async def explore_branch(idx: int, persona):
        branch_id = f"B{idx+1}"
        messages = [
            {"role": "system", "content": persona.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response, _, _, _ = await provider.chat_async(messages)

        # Update branch state
        branch = state.branches[branch_id]
        branch.completed_at = time.time()
        branch.full_response = response
        branch.summary = response.split(".")[0] if "." in response else response[:100]
        live.refresh()

        return {"persona": persona, "response": response}

    results = await asyncio.gather(*[
        explore_branch(i, p) for i, p in enumerate(personas)
    ])
    responses = results

    # Phase 3: Surface tensions
    state.phase = "tensions"
    state.phase_detail = "Analyzing disagreements"
    live.refresh()

    for i, r1 in enumerate(responses):
        for j, r2 in enumerate(responses):
            if i >= j:
                continue
            r1_lower = r1["response"].lower()
            r2_lower = r2["response"].lower()

            if (("risk" in r1_lower and "opportunity" in r2_lower) or
                ("pessimist" in r1["persona"].name.lower() and
                 "optimist" in r2["persona"].name.lower())):
                tension = Tension(
                    branch_a=r1["persona"].name,
                    branch_b=r2["persona"].name,
                    description=f"{r1['persona'].name} sees risks where {r2['persona'].name} sees opportunities. This is the crux.",
                    detected_at=time.time(),
                )
                state.tensions.append(tension)
                live.refresh()
                await asyncio.sleep(0.3)

    if not state.tensions:
        state.phase_detail = "Perspectives aligned (no major tensions)"
        live.refresh()

    # Phase 4: Synthesis
    state.phase = "synthesis"
    state.phase_detail = "Merging perspectives"
    live.refresh()

    synthesis_prompt = f"Synthesize these perspectives on: {prompt}\n\n"
    for r in responses:
        synthesis_prompt += f"[{r['persona'].name}]: {r['response'][:400]}...\n\n"

    messages = [
        {"role": "system", "content": "Synthesize multiple perspectives into a balanced, actionable summary. Highlight where they agree and disagree. Be concise but insightful."},
        {"role": "user", "content": synthesis_prompt},
    ]

    synthesis, _, _, _ = await provider.chat_async(messages)
    state.synthesis = synthesis
    state.phase = "complete"
    state.phase_detail = ""
    live.refresh()

    return {
        "responses": responses,
        "tensions": state.tensions,
        "synthesis": synthesis,
    }


# =============================================================================
# Live Argumentative Swarm
# =============================================================================

async def live_argswarm(
    prompt: str,
    provider: Provider,
    renderer: LiveRenderer,
    live: Live,
) -> dict[str, Any]:
    """Run argument swarm with live rendering."""
    from argswarm import assign_positions, Stance

    state = ExplorationState(system="argswarm", prompt=prompt)
    renderer.state = state

    # Phase 1: Assign positions
    state.phase = "spawning"
    state.phase_detail = "Assigning debate positions"
    live.refresh()

    positions = assign_positions(prompt, 2)
    for i, pos in enumerate(positions):
        stance = "FOR" if pos.stance == Stance.FOR else "AGAINST"
        branch = Branch(
            id=f"B{i+1}",
            name=stance,
            style="adversarial",
            spawned_at=time.time(),
        )
        state.branches[branch.id] = branch
        live.refresh()
        await asyncio.sleep(0.3)

    # Phase 2: Opening arguments
    state.phase = "exploring"
    state.phase_detail = "Constructing opening arguments"
    live.refresh()

    arguments = []
    for i, pos in enumerate(positions):
        stance = "FOR" if pos.stance == Stance.FOR else "AGAINST"

        messages = [
            {"role": "system", "content": f"You are arguing {stance} the proposition. Make your strongest case."},
            {"role": "user", "content": f"Proposition: {prompt}\n\nPresent your opening argument."},
        ]
        response, _, _, _ = await provider.chat_async(messages)

        branch = state.branches[f"B{i+1}"]
        branch.completed_at = time.time()
        branch.full_response = response
        branch.summary = response.split(".")[0] if "." in response else response[:100]
        live.refresh()

        arguments.append({"position": pos, "stance": stance, "argument": response})

    # Phase 3: Rebuttals
    state.phase_detail = "Exchanging rebuttals"
    live.refresh()

    rebuttals = []
    for i, arg in enumerate(arguments):
        opponent = arguments[1 - i]

        messages = [
            {"role": "system", "content": f"You are arguing {arg['stance']}. Attack your opponent's argument."},
            {"role": "user", "content": f"Opponent's argument:\n{opponent['argument'][:500]}\n\nRebut this."},
        ]
        response, _, _, _ = await provider.chat_async(messages)

        # Update branch summary to show rebuttal
        branch = state.branches[f"B{i+1}"]
        branch.summary = f"[Rebuttal] {response.split('.')[0] if '.' in response else response[:80]}"
        live.refresh()

        rebuttals.append({"stance": arg["stance"], "rebuttal": response})

    # Phase 4: Tension
    state.phase = "tensions"
    state.phase_detail = "Core disagreement"

    tension = Tension(
        branch_a="FOR",
        branch_b="AGAINST",
        description=f"Fundamental disagreement on: {prompt}",
        detected_at=time.time(),
    )
    state.tensions.append(tension)
    live.refresh()

    # Phase 5: Judgment
    state.phase = "synthesis"
    state.phase_detail = "Judge rendering verdict"
    live.refresh()

    debate_text = f"Proposition: {prompt}\n\n"
    for arg in arguments:
        debate_text += f"[{arg['stance']}]: {arg['argument'][:400]}...\n\n"
    for reb in rebuttals:
        debate_text += f"[{reb['stance']} REBUTS]: {reb['rebuttal'][:300]}...\n\n"

    messages = [
        {"role": "system", "content": "You are a debate judge. Evaluate the arguments and declare a winner with reasoning. Be concise."},
        {"role": "user", "content": f"{debate_text}\n\nWho wins this debate and why?"},
    ]

    judgment, _, _, _ = await provider.chat_async(messages)
    state.synthesis = judgment
    state.phase = "complete"
    state.phase_detail = ""
    live.refresh()

    return {
        "arguments": arguments,
        "rebuttals": rebuttals,
        "judgment": judgment,
    }


# =============================================================================
# Live Dream Logic
# =============================================================================

async def live_dreamlogic(
    prompt: str,
    provider: Provider,
    renderer: LiveRenderer,
    live: Live,
) -> dict[str, Any]:
    """Run dream logic with live rendering."""
    state = ExplorationState(system="dreamlogic", prompt=prompt)
    renderer.state = state

    # Phase 1: Seed
    state.phase = "spawning"
    state.phase_detail = f"Planting seed: {prompt[:40]}..."
    live.refresh()

    # Create a single "dream" branch
    branch = Branch(
        id="D1",
        name="Dream State",
        style="surreal",
        spawned_at=time.time(),
    )
    state.branches["D1"] = branch
    live.refresh()
    await asyncio.sleep(0.5)

    # Phase 2: Free association
    state.phase = "exploring"
    state.phase_detail = "Entering dream state..."
    live.refresh()

    messages = [
        {"role": "system", "content": "Think in dream logic. Make unexpected, surreal associations. Let one idea flow into another without conventional logic. Be creative and weird."},
        {"role": "user", "content": f"Starting from: {prompt}\n\nLet your mind wander freely. What strange connections emerge?"},
    ]

    associations, _, _, _ = await provider.chat_async(messages)

    branch.summary = associations[:150].replace("\n", " ") + "..."
    branch.full_response = associations
    branch.completed_at = time.time()
    live.refresh()

    # Phase 3: Pattern recognition
    state.phase = "tensions"
    state.phase_detail = "Waking up... recognizing patterns"
    live.refresh()

    # Add the "tension" between dream and reality
    tension = Tension(
        branch_a="Dream",
        branch_b="Reality",
        description="Extracting practical insights from surreal associations",
        detected_at=time.time(),
    )
    state.tensions.append(tension)
    live.refresh()
    await asyncio.sleep(0.5)

    # Phase 4: Insights
    state.phase = "synthesis"
    state.phase_detail = "Grounding insights in reality"
    live.refresh()

    messages = [
        {"role": "system", "content": "You're waking from a creative dream. Extract the useful patterns and insights from the surreal associations. Ground them in reality. Be practical."},
        {"role": "user", "content": f"Dream associations:\n{associations}\n\nWhat practical insights emerge from these strange connections?"},
    ]

    insights, _, _, _ = await provider.chat_async(messages)
    state.synthesis = insights
    state.phase = "complete"
    state.phase_detail = ""
    live.refresh()

    return {
        "seed": prompt,
        "associations": associations,
        "insights": insights,
    }


# =============================================================================
# Main Interface
# =============================================================================

async def live_explore(
    system: str,
    prompt: str,
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.1-8b-instruct",
    simple: bool = False,
) -> dict[str, Any]:
    """Main live exploration interface."""
    console = Console()
    renderer = LiveRenderer(console)
    llm = create_provider(provider, model)

    if simple:
        # Simple mode - just print updates without Rich Live
        return await _simple_explore(system, prompt, llm, console)

    try:
        with Live(renderer.render(), console=console, refresh_per_second=4, transient=False) as live:
            def refresh():
                live.update(renderer.render())

            # Monkey-patch refresh onto live object
            live.refresh = refresh

            if system == "personas":
                result = await live_personas(prompt, llm, renderer, live)
            elif system == "argswarm":
                result = await live_argswarm(prompt, llm, renderer, live)
            elif system == "dreamlogic":
                result = await live_dreamlogic(prompt, llm, renderer, live)
            else:
                console.print(f"[red]Unknown system: {system}[/red]")
                console.print("Available: personas, argswarm, dreamlogic")
                return {}

        return result
    except Exception as e:
        console.print(f"[yellow]Live mode failed ({e}), falling back to simple mode[/yellow]")
        return await _simple_explore(system, prompt, llm, console)


async def _simple_explore(
    system: str,
    prompt: str,
    llm: Provider,
    console: Console,
) -> dict[str, Any]:
    """Simple fallback mode with print statements."""
    from rich.panel import Panel

    console.print(Panel(f"[bold]{system.upper()}[/bold]\n{prompt}", title="Query"))

    if system == "personas":
        return await _simple_personas(prompt, llm, console)
    elif system == "argswarm":
        return await _simple_argswarm(prompt, llm, console)
    elif system == "dreamlogic":
        return await _simple_dreamlogic(prompt, llm, console)
    else:
        console.print(f"[red]Unknown system: {system}[/red]")
        return {}


async def _simple_personas(prompt: str, llm: Provider, console: Console) -> dict[str, Any]:
    """Simple personas without live display."""
    from personas import PersonaLibrary, assign_personas

    console.print("\n[cyan]Phase 1: SPAWNING[/cyan]")

    library = PersonaLibrary()
    assignment = assign_personas(problem=prompt, num_branches=4, library=library)
    personas = list(assignment.assignments.values())

    for i, persona in enumerate(personas):
        console.print(f"  [green]+[/green] B{i+1}: {persona.name} ({persona.thinking_style.value})")

    console.print("\n[cyan]Phase 2: EXPLORING[/cyan]")
    responses = []

    async def explore(idx, persona):
        console.print(f"  [yellow]...[/yellow] B{idx+1} ({persona.name}) thinking...")
        messages = [
            {"role": "system", "content": persona.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response, _, _, _ = await llm.chat_async(messages)
        summary = response.split(".")[0] if "." in response else response[:80]
        console.print(f"  [green]OK[/green] B{idx+1}: {summary[:60]}...")
        return {"persona": persona, "response": response}

    results = await asyncio.gather(*[explore(i, p) for i, p in enumerate(personas)])
    responses = results

    console.print("\n[cyan]Phase 3: SYNTHESIS[/cyan]")
    synthesis_prompt = f"Synthesize these perspectives on: {prompt}\n\n"
    for r in responses:
        synthesis_prompt += f"[{r['persona'].name}]: {r['response'][:400]}...\n\n"

    messages = [
        {"role": "system", "content": "Synthesize multiple perspectives into a balanced summary."},
        {"role": "user", "content": synthesis_prompt},
    ]
    synthesis, _, _, _ = await llm.chat_async(messages)

    console.print(Panel(synthesis, title="[bold blue]SYNTHESIS[/bold blue]"))

    return {"responses": responses, "tensions": [], "synthesis": synthesis}


async def _simple_argswarm(prompt: str, llm: Provider, console: Console) -> dict[str, Any]:
    """Simple argswarm without live display."""
    from argswarm import assign_positions, Stance

    console.print("\n[cyan]Phase 1: POSITIONS[/cyan]")
    positions = assign_positions(prompt, 2)

    for i, pos in enumerate(positions):
        stance = "FOR" if pos.stance == Stance.FOR else "AGAINST"
        console.print(f"  [green]+[/green] B{i+1}: {stance}")

    console.print("\n[cyan]Phase 2: ARGUMENTS[/cyan]")
    arguments = []
    for i, pos in enumerate(positions):
        stance = "FOR" if pos.stance == Stance.FOR else "AGAINST"
        console.print(f"  [yellow]...[/yellow] {stance} constructing argument...")
        messages = [
            {"role": "system", "content": f"You are arguing {stance} the proposition. Make your strongest case."},
            {"role": "user", "content": f"Proposition: {prompt}\n\nPresent your opening argument."},
        ]
        response, _, _, _ = await llm.chat_async(messages)
        summary = response.split(".")[0] if "." in response else response[:80]
        console.print(f"  [green]OK[/green] {stance}: {summary[:60]}...")
        arguments.append({"stance": stance, "argument": response})

    console.print("\n[cyan]Phase 3: REBUTTALS[/cyan]")
    rebuttals = []
    for i, arg in enumerate(arguments):
        opponent = arguments[1 - i]
        console.print(f"  [yellow]...[/yellow] {arg['stance']} rebutting...")
        messages = [
            {"role": "system", "content": f"You are arguing {arg['stance']}. Attack your opponent's argument."},
            {"role": "user", "content": f"Opponent's argument:\n{opponent['argument'][:500]}\n\nRebut this."},
        ]
        response, _, _, _ = await llm.chat_async(messages)
        rebuttals.append({"stance": arg["stance"], "rebuttal": response})
        console.print(f"  [green]OK[/green] {arg['stance']} rebuttal complete")

    console.print("\n[cyan]Phase 4: JUDGMENT[/cyan]")
    debate_text = f"Proposition: {prompt}\n\n"
    for arg in arguments:
        debate_text += f"[{arg['stance']}]: {arg['argument'][:400]}...\n\n"

    messages = [
        {"role": "system", "content": "You are a debate judge. Evaluate and declare a winner."},
        {"role": "user", "content": f"{debate_text}\n\nWho wins this debate and why?"},
    ]
    judgment, _, _, _ = await llm.chat_async(messages)

    console.print(Panel(judgment, title="[bold blue]JUDGMENT[/bold blue]"))

    return {"arguments": arguments, "rebuttals": rebuttals, "judgment": judgment}


async def _simple_dreamlogic(prompt: str, llm: Provider, console: Console) -> dict[str, Any]:
    """Simple dreamlogic without live display."""
    console.print("\n[cyan]Phase 1: DREAM STATE[/cyan]")
    console.print(f"  Seed: {prompt}")

    messages = [
        {"role": "system", "content": "Think in dream logic. Make unexpected, surreal associations."},
        {"role": "user", "content": f"Starting from: {prompt}\n\nLet your mind wander freely."},
    ]
    console.print("  [yellow]...[/yellow] Dreaming...")
    associations, _, _, _ = await llm.chat_async(messages)
    console.print(f"  [green]OK[/green] Dream complete")

    console.print("\n[cyan]Phase 2: WAKING[/cyan]")
    messages = [
        {"role": "system", "content": "Extract useful patterns from surreal associations. Ground them in reality."},
        {"role": "user", "content": f"Dream associations:\n{associations}\n\nWhat practical insights emerge?"},
    ]
    console.print("  [yellow]...[/yellow] Extracting insights...")
    insights, _, _, _ = await llm.chat_async(messages)

    console.print(Panel(insights, title="[bold blue]INSIGHTS[/bold blue]"))

    return {"seed": prompt, "associations": associations, "insights": insights}


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Watch exploration trees grow in real-time"
    )
    parser.add_argument("system", choices=["personas", "argswarm", "dreamlogic"],
                       help="System to run")
    parser.add_argument("prompt", help="The prompt to explore")
    parser.add_argument("--provider", "-p", default="openrouter",
                       help="LLM provider")
    parser.add_argument("--model", "-m", default="meta-llama/llama-3.1-8b-instruct",
                       help="Model to use")
    parser.add_argument("--simple", "-s", action="store_true",
                       help="Use simple output (no live updating)")

    args = parser.parse_args()

    asyncio.run(live_explore(
        args.system,
        args.prompt,
        args.provider,
        args.model,
        simple=args.simple,
    ))


if __name__ == "__main__":
    main()
