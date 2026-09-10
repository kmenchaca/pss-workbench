"""Preset configurations for common PSS use cases.

Presets provide opinionated defaults for different tasks:
- explore: Map out possibility spaces (cheap, wide, shallow)
- write: Creative/professional writing (moderate depth)
- research: Deep analysis (capable model, thorough)
- draft: Polished single outputs (low branching)
- investigate: Agentic code investigation with synthesis (v0.4)
- debug: Parallel bug hunting with evidence voting (v0.4)
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Preset:
    """A preset configuration for PSS."""

    name: str
    description: str

    # Model settings
    provider: str
    model: str

    # Gate settings (when to interrupt)
    soft_gate_tokens: int
    hard_gate_tokens: int

    # Budget settings
    total_max: int
    max_contexts: int

    # System prompt to guide exploration style
    system_prompt: str

    # v0.4: Agentic branches
    agentic_enabled: bool = False
    agentic_tools: list[str] | None = None

    # v0.4: Synthesis
    synthesis_enabled: bool = False
    synthesis_strategy: str = "merge"
    synthesis_provider: str | None = None
    synthesis_model: str | None = None

    # v0.5: Diversity-aware spawning
    diversity_aware_enabled: bool = False
    diversity_target: float = 0.5
    diversity_min_threshold: float = 0.3


# Preset definitions
PRESETS: dict[str, Preset] = {
    "explore": Preset(
        name="explore",
        description="Map out a topic space - cheap, wide, shallow exploration",
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        soft_gate_tokens=3000,
        hard_gate_tokens=10000,
        total_max=50000,
        max_contexts=15,
        system_prompt="""You are mapping out a topic space. Your goal is divergent exploration.

Guidelines:
- Branch freely into different angles, approaches, and perspectives
- Prefer breadth over depth - survey the landscape before diving deep
- Don't commit to conclusions too early
- When you see multiple valid directions, explore them as separate branches
- It's better to have many shallow explorations than one deep one
- Flag interesting areas for deeper investigation, but don't go deep yourself

At checkpoints, prefer branching over continuing unless you've found something definitive.""",
        # v0.5: Higher diversity target for exploration
        diversity_aware_enabled=True,
        diversity_target=0.6,
        diversity_min_threshold=0.3,
    ),
    "write": Preset(
        name="write",
        description="Creative or professional writing - moderate depth, varied outputs",
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        soft_gate_tokens=8000,
        hard_gate_tokens=25000,
        total_max=100000,
        max_contexts=8,
        system_prompt="""You are a writer exploring different creative directions.

Guidelines:
- Commit to a voice and style early, then develop it
- Each branch should feel like a genuinely different piece, not a variation
- Let the writing breathe - don't rush to conclusions
- If you branch, make each branch a different genre, tone, or approach
- Quality matters more than quantity of branches
- Show, don't tell. Develop scenes and moments.

At checkpoints, continue if you're onto something good. Branch if you see a completely different direction worth exploring.""",
        # v0.5: Moderate diversity target for writing
        diversity_aware_enabled=True,
        diversity_target=0.5,
        diversity_min_threshold=0.3,
    ),
    "research": Preset(
        name="research",
        description="Deep analysis - thorough, careful, fewer but deeper branches",
        provider="openrouter",
        model="meta-llama/llama-3.1-70b-instruct",
        soft_gate_tokens=15000,
        hard_gate_tokens=50000,
        total_max=200000,
        max_contexts=5,
        system_prompt="""You are conducting thorough research on a topic.

Guidelines:
- Be rigorous and systematic in your analysis
- Cite sources and evidence when possible
- Consider multiple perspectives, but evaluate them critically
- Go deep on important points rather than skimming many
- Branch only when you find genuinely distinct analytical frames
- Acknowledge uncertainty and gaps in knowledge
- Synthesize findings into clear conclusions

At checkpoints, continue if you're building toward a solid analysis. Branch only for fundamentally different analytical approaches.""",
    ),
    "draft": Preset(
        name="draft",
        description="Polished outputs - focused, low branching, publication-ready",
        provider="openrouter",
        model="meta-llama/llama-3.1-70b-instruct",
        soft_gate_tokens=10000,
        hard_gate_tokens=30000,
        total_max=80000,
        max_contexts=3,
        system_prompt="""You are producing polished, publication-ready content.

Guidelines:
- Focus on quality over exploration breadth
- Structure your output clearly with good organization
- Refine language and flow as you write
- Branch sparingly - only for genuinely different approaches
- Aim for completeness within each output
- Edit as you go - no rough drafts

At checkpoints, continue to completion unless you realize the entire approach is wrong.""",
    ),
    # v0.4: Agentic presets with synthesis
    "investigate": Preset(
        name="investigate",
        description="Agentic code investigation - parallel exploration with synthesis",
        provider="openrouter",
        model="openai/gpt-4o-mini",  # GPT-4o-mini for reliable tool calling
        soft_gate_tokens=5000,
        hard_gate_tokens=20000,
        total_max=100000,
        max_contexts=8,
        system_prompt="""You are investigating a codebase. Use tools to explore and understand it.

Guidelines:
- Read files to understand code structure and implementation
- Run tests to verify behavior
- Search for patterns, issues, and related code
- Branch when you find multiple areas worth investigating
- Each branch should investigate independently
- Record findings with confidence levels
- Use tool outputs as evidence for your conclusions

At checkpoints, branch if you find multiple interesting leads to pursue in parallel.""",
        agentic_enabled=True,
        agentic_tools=["read_file", "list_directory", "search_files", "run_command"],
        synthesis_enabled=True,
        synthesis_strategy="plan",
        synthesis_provider="anthropic",
        synthesis_model="claude-sonnet-4-20250514",
    ),
    "debug": Preset(
        name="debug",
        description="Parallel bug hunting with evidence aggregation",
        provider="openrouter",
        model="openai/gpt-4o-mini",  # GPT-4o-mini for reliable tool calling
        soft_gate_tokens=3000,
        hard_gate_tokens=15000,
        total_max=80000,
        max_contexts=10,
        system_prompt="""You are debugging a codebase. Hunt for bugs and issues.

Guidelines:
- Run tests and check for failures
- Read code that tests reference
- Search for common bug patterns (null checks, boundary conditions, race conditions)
- Branch to investigate different potential issues in parallel
- Record what you find with confidence levels
- Use test outputs and code analysis as evidence
- Don't assume - verify with tools

At checkpoints, branch to investigate different bug hypotheses in parallel.""",
        agentic_enabled=True,
        agentic_tools=["read_file", "run_command", "search_files"],
        synthesis_enabled=True,
        synthesis_strategy="vote",
        synthesis_provider="anthropic",
        synthesis_model="claude-sonnet-4-20250514",
        # v0.5: Diversity-aware for wider bug coverage
        diversity_aware_enabled=True,
        diversity_target=0.5,
        diversity_min_threshold=0.25,
    ),
    # v0.6: Reasoning preset for math/logic problems
    "reasoning": Preset(
        name="reasoning",
        description="Mathematical and logical reasoning - systematic exploration with verification",
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        soft_gate_tokens=2000,   # Give time for systematic exploration
        hard_gate_tokens=8000,   # Allow deeper work before forcing decision
        total_max=50000,
        max_contexts=10,  # Allow more parallel approaches
        system_prompt="""You are solving a mathematical or logical reasoning problem.

Guidelines:
- Think step by step and show your work
- Be systematic - try combinations methodically
- Check your arithmetic carefully
- If you have alternative approaches in mind, list them as branches
- BRANCH when you see multiple valid strategies worth trying in parallel
- CONTINUE only if you're confident in your current approach
- Verify your solution before terminating

At checkpoints: If you see alternative approaches, branch to explore them. Continue only if current path is clearly working. Terminate with your verified solution.""",
        # Lower thresholds to encourage more branching on reasoning tasks
        diversity_aware_enabled=True,
        diversity_target=0.35,
        diversity_min_threshold=0.15,
    ),
}


PresetName = Literal["explore", "write", "research", "draft", "investigate", "debug", "reasoning"]


def get_preset(name: str) -> Preset | None:
    """Get a preset by name."""
    return PRESETS.get(name)


def list_presets() -> list[Preset]:
    """List all available presets."""
    return list(PRESETS.values())


def apply_preset_to_config(preset: Preset, config) -> None:
    """Apply preset settings to a config object (mutates config)."""
    config.provider = preset.provider
    config.model = preset.model
    config.soft_gate_tokens = preset.soft_gate_tokens
    config.hard_gate_tokens = preset.hard_gate_tokens
    config.total_max = preset.total_max
    config.max_contexts = preset.max_contexts
    config.system_prompt = preset.system_prompt

    # v0.4: Agentic and synthesis settings
    config.agentic_enabled = preset.agentic_enabled
    if preset.agentic_tools:
        config.agentic_tools = preset.agentic_tools
    config.synthesis_enabled = preset.synthesis_enabled
    config.synthesis_strategy = preset.synthesis_strategy
    if preset.synthesis_provider:
        config.synthesis_provider = preset.synthesis_provider
    if preset.synthesis_model:
        config.synthesis_model = preset.synthesis_model

    # v0.5: Diversity-aware settings
    config.diversity_aware_enabled = preset.diversity_aware_enabled
    config.diversity_target = preset.diversity_target
    config.diversity_min_threshold = preset.diversity_min_threshold
