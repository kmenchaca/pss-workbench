"""Configuration loading for PSS."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class PSSConfig:
    """Configuration for a PSS run."""

    # Provider settings
    provider: Literal["anthropic", "openrouter"] = "openrouter"
    model: str = "meta-llama/llama-3.1-8b-instruct"

    # Gate settings
    soft_gate_tokens: int = 15000
    hard_gate_tokens: int = 50000

    # Budget settings
    per_context_max: int = 50000
    total_max: int = 500000
    max_contexts: int = 20

    # Termination
    termination_strategy: Literal[
        "first_good_or_budget", "all_exhausted", "timeout"
    ] = "first_good_or_budget"
    good_enough_threshold: float = 0.8
    timeout_seconds: int | None = None

    # Verification
    verification_strategy: Literal["none", "programmatic", "checkbox", "reviewer"] = (
        "none"
    )
    test_command: str | None = (
        None  # For programmatic verification (e.g., "pytest", "npm test")
    )

    # Execution
    parallel: bool = False
    max_parallel: int = 5

    # Diversity analysis
    diversity_enabled: bool = True
    diversity_provider: Literal["openai", "none"] = "openai"
    diversity_model: str = "text-embedding-3-small"
    diversity_show_matrix: bool = False

    # System prompt (set by presets, can be overridden)
    system_prompt: str | None = None

    # v0.4: Agentic branches
    agentic_enabled: bool = False
    agentic_tools: list[str] | None = None  # Default: ["read_file", "list_directory", "search_files"]
    agentic_tool_timeout: int = 30
    agentic_working_dir: str | None = None

    # v0.4: Synthesis phase
    synthesis_enabled: bool = False
    synthesis_strategy: Literal["merge", "vote", "deliberate", "plan"] = "merge"
    synthesis_provider: Literal["anthropic", "openrouter"] | None = None  # Override for synthesis
    synthesis_model: str | None = None  # Override model for synthesis
    synthesis_max_tokens: int = 8000

    # v0.5: Diversity-aware spawning
    diversity_aware_enabled: bool = False
    diversity_target: float = 0.5  # Stop spawning new branches when reached
    diversity_min_threshold: float = 0.3  # Encourage branching below this
    diversity_max_embedding_cost: float = 0.01  # USD cap for embeddings

    # v1.0: Failure propagation
    failure_propagation_enabled: bool = False
    failure_max_to_inject: int = 5  # Max failures to inject per branch
    failure_min_confidence: float = 0.5  # Min confidence to propagate
    failure_stale_after_branches: int = 5  # Branches until failure becomes stale
    failure_archive_after_branches: int = 15  # Branches until failure archived

    # v1.0: Adaptive gates
    adaptive_gates_enabled: bool = False
    adaptive_min_tokens: int = 5000  # Never gate before this
    adaptive_max_tokens: int = 50000  # Always gate by this
    adaptive_high_velocity_threshold: float = 0.7  # Above = delay gates
    adaptive_low_velocity_threshold: float = 0.4  # Below = accelerate gates
    adaptive_soft_gate_base_tokens: int = 15000  # Default soft gate position
    adaptive_hard_gate_base_tokens: int = 35000  # Default hard gate position
    adaptive_velocity_window_tokens: int = 2000  # Window for velocity measurement
    adaptive_measurement_interval_tokens: int = 500  # How often to measure

    # v1.1: Momentum-based early gates (empirically derived from trace analysis)
    adaptive_momentum_enabled: bool = True  # Enable momentum loss detection
    adaptive_momentum_loss_threshold: float = 0.1  # Velocity drop from peak to trigger early gate
    adaptive_conclusion_velocity: float = 0.55  # Branches converge to this when completing

    # v1.0: Bulletin board (cross-branch signals)
    bulletin_enabled: bool = False
    bulletin_max_signals_per_injection: int = 5  # Max signals per gate
    bulletin_delivery_strategy: str = "gate_batched"  # eager, gate_batched, relevance_filtered
    bulletin_min_confidence_to_post: float = 0.3  # Min confidence to post signal
    bulletin_min_confidence_to_deliver: float = 0.4  # Min confidence to deliver
    bulletin_signal_half_life_tokens: int = 10000  # Confidence decay half-life
    bulletin_dead_source_penalty: float = 0.5  # Multiplier when source died

    # v1.2: Convergent task detection (prompt-based)
    convergence_enabled: bool = False  # Enable convergent task detection
    convergence_branch_on_uncertain: bool = True  # Treat uncertain as divergent
    convergence_confidence_threshold: float = 0.7  # Min confidence to classify

    # v1.3: Lazy branching (response-based)
    lazy_branching_enabled: bool = False  # Check response for convergence
    lazy_branching_min_tokens: int = 20  # Min tokens before judging (short answers OK)
    lazy_branching_window_tokens: int = 2000  # Check convergence within this window
    lazy_branching_convergence_velocity: float = 0.5  # Below = converged
    lazy_branching_max_converged_length: int = 1000  # Converged responses under this length

    # v1.4: Sibling diversity enforcement
    sibling_diversity_enabled: bool = False  # Reject similar sibling branches
    sibling_similarity_threshold: float = 0.7  # Reject if similarity > this (0-1)
    sibling_diversity_include_existing: bool = True  # Compare against running siblings
    sibling_diversity_min_edits: int = 2  # Only check when >= this many edits

    # v1.5: Context compression at spawn
    context_compression: bool = False  # Summarize parent history when spawning branches
    compression_max_messages: int = 20  # Compress if message count exceeds this

    # v1.5: Branch priority scheduling
    priority_scheduling: bool = False  # Sort branches by confidence before processing
    priority_budget_threshold: float = 0.8  # Skip low-confidence branches above this % budget used

    # v1.5: Incremental synthesis
    incremental_synthesis: bool = False  # Check consensus as leaves arrive
    incremental_check_interval: int = 3  # Check every N leaves
    incremental_consensus_threshold: float = 0.8  # Early stop above this

    # v1.5: Path diversity tracking
    path_diversity_enabled: bool = False  # Track exploration trajectory diversity


def load_config(path: str | Path) -> PSSConfig:
    """Load config from a YAML file."""
    path = Path(path)
    if not path.exists():
        return PSSConfig()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Flatten nested config structure
    flat = {}

    if "provider" in data:
        if isinstance(data["provider"], dict):
            flat["provider"] = data["provider"].get("name")
            flat["model"] = data["provider"].get("model")
        else:
            # Flat format: provider is a string
            flat["provider"] = data["provider"]
    if "gates" in data:
        flat["soft_gate_tokens"] = data["gates"].get("soft_gate_tokens", 15000)
        flat["hard_gate_tokens"] = data["gates"].get("hard_gate_tokens", 50000)
    if "budget" in data:
        flat["per_context_max"] = data["budget"].get("per_context_max", 50000)
        flat["total_max"] = data["budget"].get("total_max", 500000)
        flat["max_contexts"] = data["budget"].get("max_contexts", 20)
    if "termination" in data:
        flat["termination_strategy"] = data["termination"].get(
            "strategy", "first_good_or_budget"
        )
        flat["good_enough_threshold"] = data["termination"].get(
            "good_enough_threshold", 0.8
        )
        flat["timeout_seconds"] = data["termination"].get("timeout_seconds")
    if "verification" in data:
        flat["verification_strategy"] = data["verification"].get("strategy", "none")
        flat["test_command"] = data["verification"].get("test_command")
    if "execution" in data:
        flat["parallel"] = data["execution"].get("parallel", False)
        flat["max_parallel"] = data["execution"].get("max_parallel", 5)
    if "diversity" in data:
        flat["diversity_enabled"] = data["diversity"].get("enabled", True)
        flat["diversity_provider"] = data["diversity"].get("provider", "openai")
        flat["diversity_model"] = data["diversity"].get(
            "model", "text-embedding-3-small"
        )
        flat["diversity_show_matrix"] = data["diversity"].get("show_matrix", False)

    # v0.4: Agentic branches
    if "agentic" in data:
        flat["agentic_enabled"] = data["agentic"].get("enabled", False)
        flat["agentic_tools"] = data["agentic"].get("tools")
        flat["agentic_tool_timeout"] = data["agentic"].get("tool_timeout", 30)
        flat["agentic_working_dir"] = data["agentic"].get("working_dir")

    # v0.4: Synthesis phase
    if "synthesis" in data:
        flat["synthesis_enabled"] = data["synthesis"].get("enabled", False)
        flat["synthesis_strategy"] = data["synthesis"].get("strategy", "merge")
        flat["synthesis_provider"] = data["synthesis"].get("provider")
        flat["synthesis_model"] = data["synthesis"].get("model")
        flat["synthesis_max_tokens"] = data["synthesis"].get("max_tokens", 8000)

    # v0.5: Diversity-aware spawning
    if "diversity_aware" in data:
        flat["diversity_aware_enabled"] = data["diversity_aware"].get("enabled", False)
        flat["diversity_target"] = data["diversity_aware"].get("target", 0.5)
        flat["diversity_min_threshold"] = data["diversity_aware"].get("min_threshold", 0.3)
        flat["diversity_max_embedding_cost"] = data["diversity_aware"].get("max_embedding_cost", 0.01)

    # v1.0: Failure propagation
    if "failure_propagation" in data:
        flat["failure_propagation_enabled"] = data["failure_propagation"].get("enabled", False)
        flat["failure_max_to_inject"] = data["failure_propagation"].get("max_to_inject", 5)
        flat["failure_min_confidence"] = data["failure_propagation"].get("min_confidence", 0.5)
        flat["failure_stale_after_branches"] = data["failure_propagation"].get("stale_after_branches", 5)
        flat["failure_archive_after_branches"] = data["failure_propagation"].get("archive_after_branches", 15)

    # v1.0: Adaptive gates
    if "adaptive_gates" in data:
        flat["adaptive_gates_enabled"] = data["adaptive_gates"].get("enabled", False)
        flat["adaptive_min_tokens"] = data["adaptive_gates"].get("min_tokens", 5000)
        flat["adaptive_max_tokens"] = data["adaptive_gates"].get("max_tokens", 50000)
        flat["adaptive_high_velocity_threshold"] = data["adaptive_gates"].get("high_velocity_threshold", 0.7)
        flat["adaptive_low_velocity_threshold"] = data["adaptive_gates"].get("low_velocity_threshold", 0.4)
        flat["adaptive_soft_gate_base_tokens"] = data["adaptive_gates"].get("soft_gate_base_tokens", 15000)
        flat["adaptive_hard_gate_base_tokens"] = data["adaptive_gates"].get("hard_gate_base_tokens", 35000)
        flat["adaptive_velocity_window_tokens"] = data["adaptive_gates"].get("velocity_window_tokens", 2000)
        flat["adaptive_measurement_interval_tokens"] = data["adaptive_gates"].get("measurement_interval_tokens", 500)
        # v1.1: Momentum detection
        flat["adaptive_momentum_enabled"] = data["adaptive_gates"].get("momentum_enabled", True)
        flat["adaptive_momentum_loss_threshold"] = data["adaptive_gates"].get("momentum_loss_threshold", 0.1)
        flat["adaptive_conclusion_velocity"] = data["adaptive_gates"].get("conclusion_velocity", 0.55)

    # v1.0: Bulletin board
    if "bulletin" in data:
        flat["bulletin_enabled"] = data["bulletin"].get("enabled", False)
        flat["bulletin_max_signals_per_injection"] = data["bulletin"].get("max_signals_per_injection", 5)
        flat["bulletin_delivery_strategy"] = data["bulletin"].get("delivery_strategy", "gate_batched")
        flat["bulletin_min_confidence_to_post"] = data["bulletin"].get("min_confidence_to_post", 0.3)
        flat["bulletin_min_confidence_to_deliver"] = data["bulletin"].get("min_confidence_to_deliver", 0.4)
        flat["bulletin_signal_half_life_tokens"] = data["bulletin"].get("signal_half_life_tokens", 10000)
        flat["bulletin_dead_source_penalty"] = data["bulletin"].get("dead_source_penalty", 0.5)

    # v1.2: Convergent task detection
    if "convergence" in data:
        flat["convergence_enabled"] = data["convergence"].get("enabled", False)
        flat["convergence_branch_on_uncertain"] = data["convergence"].get("branch_on_uncertain", True)
        flat["convergence_confidence_threshold"] = data["convergence"].get("confidence_threshold", 0.7)

    # v1.3: Lazy branching
    if "lazy_branching" in data:
        flat["lazy_branching_enabled"] = data["lazy_branching"].get("enabled", False)
        flat["lazy_branching_min_tokens"] = data["lazy_branching"].get("min_tokens", 200)
        flat["lazy_branching_window_tokens"] = data["lazy_branching"].get("window_tokens", 750)
        flat["lazy_branching_convergence_velocity"] = data["lazy_branching"].get("convergence_velocity", 0.5)
        flat["lazy_branching_max_converged_length"] = data["lazy_branching"].get("max_converged_length", 300)

    # v1.4: Sibling diversity enforcement
    if "sibling_diversity" in data:
        flat["sibling_diversity_enabled"] = data["sibling_diversity"].get("enabled", False)
        flat["sibling_similarity_threshold"] = data["sibling_diversity"].get("similarity_threshold", 0.7)
        flat["sibling_diversity_include_existing"] = data["sibling_diversity"].get("include_existing", True)
        flat["sibling_diversity_min_edits"] = data["sibling_diversity"].get("min_edits", 2)

    # v1.5: Context compression
    if "context_compression" in data:
        flat["context_compression"] = data["context_compression"].get("enabled", False)
        flat["compression_max_messages"] = data["context_compression"].get("max_messages", 20)

    # v1.5: Priority scheduling
    if "priority_scheduling" in data:
        flat["priority_scheduling"] = data["priority_scheduling"].get("enabled", False)
        flat["priority_budget_threshold"] = data["priority_scheduling"].get("budget_threshold", 0.8)

    # v1.5: Incremental synthesis
    if "incremental_synthesis" in data:
        flat["incremental_synthesis"] = data["incremental_synthesis"].get("enabled", False)
        flat["incremental_check_interval"] = data["incremental_synthesis"].get("check_interval", 3)
        flat["incremental_consensus_threshold"] = data["incremental_synthesis"].get("consensus_threshold", 0.8)

    # v1.5: Path diversity
    if "path_diversity" in data:
        flat["path_diversity_enabled"] = data["path_diversity"].get("enabled", False)

    # Also support flat keys at top level
    for key in [
        "provider",
        "model",
        "soft_gate_tokens",
        "hard_gate_tokens",
        "per_context_max",
        "total_max",
        "max_contexts",
        "termination_strategy",
        "good_enough_threshold",
        "timeout_seconds",
        "verification_strategy",
        "test_command",
        "parallel",
        "max_parallel",
        "diversity_enabled",
        "diversity_provider",
        "diversity_model",
        "diversity_show_matrix",
        "system_prompt",
        # v0.4 agentic
        "agentic_enabled",
        "agentic_tools",
        "agentic_tool_timeout",
        "agentic_working_dir",
        # v0.4 synthesis
        "synthesis_enabled",
        "synthesis_strategy",
        "synthesis_provider",
        "synthesis_model",
        "synthesis_max_tokens",
        # v0.5 diversity-aware
        "diversity_aware_enabled",
        "diversity_target",
        "diversity_min_threshold",
        "diversity_max_embedding_cost",
        # v1.0 failure propagation
        "failure_propagation_enabled",
        "failure_max_to_inject",
        "failure_min_confidence",
        "failure_stale_after_branches",
        "failure_archive_after_branches",
        # v1.0 adaptive gates
        "adaptive_gates_enabled",
        "adaptive_min_tokens",
        "adaptive_max_tokens",
        "adaptive_high_velocity_threshold",
        "adaptive_low_velocity_threshold",
        "adaptive_soft_gate_base_tokens",
        "adaptive_hard_gate_base_tokens",
        "adaptive_velocity_window_tokens",
        "adaptive_measurement_interval_tokens",
        # v1.1 momentum detection
        "adaptive_momentum_enabled",
        "adaptive_momentum_loss_threshold",
        "adaptive_conclusion_velocity",
        # v1.0 bulletin board
        "bulletin_enabled",
        "bulletin_max_signals_per_injection",
        "bulletin_delivery_strategy",
        "bulletin_min_confidence_to_post",
        "bulletin_min_confidence_to_deliver",
        "bulletin_signal_half_life_tokens",
        "bulletin_dead_source_penalty",
        # v1.2 convergent task detection
        "convergence_enabled",
        "convergence_branch_on_uncertain",
        "convergence_confidence_threshold",
        # v1.3 lazy branching
        "lazy_branching_enabled",
        "lazy_branching_min_tokens",
        "lazy_branching_window_tokens",
        "lazy_branching_convergence_velocity",
        "lazy_branching_max_converged_length",
        # v1.4 sibling diversity enforcement
        "sibling_diversity_enabled",
        "sibling_similarity_threshold",
        "sibling_diversity_include_existing",
        "sibling_diversity_min_edits",
        # v1.5 context compression
        "context_compression",
        "compression_max_messages",
        # v1.5 priority scheduling
        "priority_scheduling",
        "priority_budget_threshold",
        # v1.5 incremental synthesis
        "incremental_synthesis",
        "incremental_check_interval",
        "incremental_consensus_threshold",
        # v1.5 path diversity
        "path_diversity_enabled",
    ]:
        if key in data and key not in flat:
            flat[key] = data[key]

    # Filter to only valid fields
    valid_fields = {f for f in PSSConfig.__dataclass_fields__}
    filtered = {k: v for k, v in flat.items() if k in valid_fields and v is not None}

    return PSSConfig(**filtered)
