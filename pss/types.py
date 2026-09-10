"""Core data structures for Pepe Silvia Search."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class SpawnReason(Enum):
    """Why a branch was created."""

    INITIAL = "initial"  # root branch
    GATE_BRANCH = "gate_branch"  # normal branch at gate
    PROMOTION = "promotion"  # user promoted this direction
    FAILURE_RECOVERY = "failure_recovery"  # spawned to avoid known failure
    DIVERSITY_SPAWN = "diversity_spawn"  # spawned for diversity


class TerminationReason(Enum):
    """Why a branch terminated."""

    COMPLETED = "completed"  # reached satisfactory output
    STUCK = "stuck"  # self-terminated, no good path
    KILLED = "killed"  # user killed
    BUDGET = "budget"  # hit token limit
    MERGED = "merged"  # absorbed into another branch


@dataclass
class SpawnMetadata:
    """Metadata captured at branch spawn time."""

    spawn_reason: SpawnReason
    spawn_edit: str  # the "what if X" that created this branch
    spawn_gate_number: int  # which gate triggered the spawn
    spawn_confidence: float  # confidence from parent's gate decision
    parent_velocity: float | None = None  # velocity when spawned (for adaptive gates)


@dataclass
class VelocitySnapshot:
    """Velocity metrics at a point in time (for adaptive gates)."""

    timestamp: float
    tokens_in_window: int
    novelty_rate: float  # 0-1, new concepts per token
    assertion_rate: float  # 0-1, claims per token
    uncertainty_trend: float  # -1 to 1, negative = less certain
    repetition_score: float  # 0-1, higher = more repetitive
    question_density: float  # 0-1, higher = more questions

    @property
    def overall_velocity(self) -> float:
        """Composite score 0-1. Higher = more productive exploration."""
        return (
            self.novelty_rate * 0.3
            + self.assertion_rate * 0.25
            + (1 - self.repetition_score) * 0.25
            + (1 - self.question_density) * 0.1
            + (self.uncertainty_trend + 1) / 2 * 0.1  # normalize to 0-1
        )


@dataclass
class TokenUsage:
    """Token usage and cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0  # in USD

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost=self.cost + other.cost,
        )


@dataclass
class ToolTrace:
    """Record of a single tool call during exploration."""

    tool_name: str
    arguments: dict[str, Any]
    result: str
    success: bool
    timestamp: float
    tokens_used: int = 0  # Tokens consumed by this call + result in context


@dataclass
class Context:
    """A single exploration context in the search tree."""

    id: str
    parent_id: str | None
    messages: list[dict] = field(default_factory=list)  # conversation history
    state: dict = field(default_factory=dict)  # files, artifacts, scratchpad
    token_count: int = 0  # total tokens (input + output) for gate logic
    token_budget: int = 50000
    status: Literal["running", "terminated", "branched"] = "running"
    branch_reason: str | None = None  # edit that spawned this branch
    output: str | None = None  # final output if terminated with leaf
    gates_seen: int = 0
    # v0.2: detailed token/cost tracking
    usage: TokenUsage = field(default_factory=TokenUsage)
    # v0.4: tool traces for agentic branches
    tool_traces: list[ToolTrace] = field(default_factory=list)
    # v1.0: genealogy tracking
    spawn_metadata: SpawnMetadata | None = None
    termination_reason: TerminationReason | None = None
    velocity_history: list[VelocitySnapshot] = field(default_factory=list)


@dataclass
class GateDecision:
    """Decision made at a checkpoint gate."""

    action: Literal["continue", "branch", "terminate"]
    branches: list[str] = field(default_factory=list)  # edit prompts for new branches
    also_continue: bool = False  # should original path continue alongside branches?
    output: str | None = None  # if terminating with a leaf
    confidence: float | None = None  # self-reported, optional
    reasoning: str | None = None


@dataclass
class SearchTree:
    """The full tree of exploration contexts."""

    contexts: dict[str, Context] = field(default_factory=dict)
    root_id: str = "root"
    leaves: list[str] = field(
        default_factory=list
    )  # context IDs that terminated with output


@dataclass
class GateResult:
    """Result of running a context until it hits a gate."""

    gate_type: Literal["soft", "hard", "none"]
    tokens_used: int  # legacy: total tokens
    response_text: str
    tool_calls: list[dict] = field(default_factory=list)
    # v0.2: detailed tracking
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class SynthesisResult:
    """Result of the synthesis phase."""

    unified_output: str
    confidence: float
    evidence_summary: dict[str, int] = field(default_factory=dict)  # finding -> count
    dissenting_views: list[str] = field(default_factory=list)
    action_plan: list[str] | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class PSSResult:
    """Complete result of a PSS run."""

    leaves: list[Context]
    tree: "SearchTree"
    total_usage: TokenUsage
    synthesis: SynthesisResult | None = None
    # v1.1: Trace metadata (optional, populated when --trace enabled)
    _trace_data: dict | None = None  # Internal: raw data for trace building


@dataclass
class DiversityState:
    """Real-time diversity tracking during exploration (v0.5)."""

    current_score: float = 0.0
    interpretation: str = "unknown"  # low/medium/high/unknown
    num_leaves: int = 0
    cached_embeddings: dict[str, list[float]] = field(default_factory=dict)
    total_embedding_cost: float = 0.0
    last_computed_at: str | None = None  # leaf ID that triggered last computation

    def should_encourage_branching(self, min_threshold: float) -> bool:
        """Returns True if diversity is below threshold."""
        if self.num_leaves < 2:
            return False
        return self.current_score < min_threshold

    def is_target_reached(self, target: float) -> bool:
        """Returns True if diversity target is met."""
        if self.num_leaves < 2:
            return False
        return self.current_score >= target
