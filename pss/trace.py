"""Trace collection for PSS runs.

Captures comprehensive data about each exploration run for post-hoc analysis.
Use --trace flag to enable: pss run "prompt" --trace
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pss.config import PSSConfig
    from pss.failures import FailureSummary
    from pss.genealogy import GenealogyTree
    from pss.signals import BulletinBoard, Signal
    from pss.types import (
        Context,
        DiversityState,
        PSSResult,
        SearchTree,
        SynthesisResult,
        TokenUsage,
        VelocitySnapshot,
    )


@dataclass
class ExplorationTrace:
    """Complete trace of a PSS exploration run.

    Captures everything needed for post-hoc analysis:
    - What was explored (prompt, config)
    - How it was explored (tree structure, decisions, velocities)
    - What failed (failure registry snapshots)
    - What was discovered (signals, synthesis)
    - Outcome quality (for manual rating)
    """

    # Session metadata
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    prompt: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    # Tree structure
    tree: dict[str, Any] = field(default_factory=dict)  # serialized SearchTree
    leaves: list[dict[str, Any]] = field(default_factory=list)  # serialized Contexts

    # v1.0 features
    genealogy: dict[str, Any] | None = None  # serialized GenealogyTree
    failures: list[dict[str, Any]] = field(default_factory=list)  # serialized FailureSummaries
    signals: list[dict[str, Any]] = field(default_factory=list)  # serialized Signals

    # Velocity snapshots per branch
    velocity_snapshots: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Synthesis
    synthesis: dict[str, Any] | None = None  # serialized SynthesisResult

    # Diversity
    diversity_state: dict[str, Any] | None = None

    # Aggregates
    total_usage: dict[str, Any] = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "cost": 0.0
    })
    elapsed_seconds: float = 0.0

    # Counts
    total_contexts_created: int = 0
    total_gates_seen: int = 0
    total_branches_spawned: int = 0
    num_leaves: int = 0

    # Termination breakdown
    leaf_termination_reasons: dict[str, int] = field(default_factory=dict)

    # Human rating (filled in later during analysis)
    human_rating: int | None = None  # 1-5 scale
    human_notes: str | None = None

    # Execution events (timestamped log)
    events: list[dict[str, Any]] = field(default_factory=list)

    def add_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Add a timestamped event to the trace."""
        self.events.append({
            "timestamp": time.time(),
            "type": event_type,
            "data": data or {},
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to JSON-serializable dict."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save trace to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "ExplorationTrace":
        """Load trace from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def serialize_context(ctx: "Context") -> dict[str, Any]:
    """Serialize a Context for trace storage."""
    from pss.types import TerminationReason, SpawnReason

    result = {
        "id": ctx.id,
        "parent_id": ctx.parent_id,
        "status": ctx.status,
        "branch_reason": ctx.branch_reason,
        "output": ctx.output,
        "gates_seen": ctx.gates_seen,
        "token_count": ctx.token_count,
        "token_budget": ctx.token_budget,
        "usage": {
            "input_tokens": ctx.usage.input_tokens,
            "output_tokens": ctx.usage.output_tokens,
            "cost": ctx.usage.cost,
        },
        "termination_reason": ctx.termination_reason.value if ctx.termination_reason else None,
    }

    # Serialize spawn metadata
    if ctx.spawn_metadata:
        result["spawn_metadata"] = {
            "spawn_reason": ctx.spawn_metadata.spawn_reason.value,
            "spawn_edit": ctx.spawn_metadata.spawn_edit,
            "spawn_gate_number": ctx.spawn_metadata.spawn_gate_number,
            "spawn_confidence": ctx.spawn_metadata.spawn_confidence,
            "parent_velocity": ctx.spawn_metadata.parent_velocity,
        }

    # Serialize velocity history
    if ctx.velocity_history:
        result["velocity_history"] = [
            serialize_velocity_snapshot(v) for v in ctx.velocity_history
        ]

    # Serialize tool traces
    if ctx.tool_traces:
        result["tool_traces"] = [
            {
                "tool_name": t.tool_name,
                "arguments": t.arguments,
                "result": t.result[:500] if len(t.result) > 500 else t.result,  # truncate
                "success": t.success,
                "timestamp": t.timestamp,
                "tokens_used": t.tokens_used,
            }
            for t in ctx.tool_traces
        ]

    # Don't serialize full message history (too large)
    # Instead, capture message count and last assistant message
    result["message_count"] = len(ctx.messages)
    for msg in reversed(ctx.messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            content = msg["content"]
            result["last_assistant_message"] = content[:1000] if len(content) > 1000 else content
            break

    return result


def serialize_velocity_snapshot(v: "VelocitySnapshot") -> dict[str, Any]:
    """Serialize a VelocitySnapshot."""
    return {
        "timestamp": v.timestamp,
        "tokens_in_window": v.tokens_in_window,
        "novelty_rate": v.novelty_rate,
        "assertion_rate": v.assertion_rate,
        "uncertainty_trend": v.uncertainty_trend,
        "repetition_score": v.repetition_score,
        "question_density": v.question_density,
        "overall_velocity": v.overall_velocity,
    }


def serialize_search_tree(tree: "SearchTree") -> dict[str, Any]:
    """Serialize a SearchTree."""
    return {
        "root_id": tree.root_id,
        "leaves": tree.leaves,
        "context_ids": list(tree.contexts.keys()),
        "contexts": {
            ctx_id: serialize_context(ctx)
            for ctx_id, ctx in tree.contexts.items()
        },
    }


def serialize_failure(failure: "FailureSummary") -> dict[str, Any]:
    """Serialize a FailureSummary."""
    return {
        "branch_id": failure.branch_id,
        "failure_type": failure.failure_type.value,
        "scope": failure.scope.value,
        "what_was_attempted": failure.what_was_attempted,
        "why_it_failed": failure.why_it_failed,
        "evidence": failure.evidence,
        "suggested_avoidance": failure.suggested_avoidance,
        "confidence": failure.confidence,
        "branch_depth": failure.branch_depth,
        "branch_path": failure.branch_path,
        "sibling_outcomes": failure.sibling_outcomes,
        "age_branches": failure.age_branches,
        "status": failure.status,
    }


def serialize_signal(signal: "Signal") -> dict[str, Any]:
    """Serialize a Signal."""
    return {
        "id": signal.id,
        "source_branch": signal.source_branch,
        "timestamp": signal.timestamp,
        "signal_type": signal.signal_type.value,
        "content": signal.content,
        "confidence": signal.confidence,
        "tags": signal.tags,
        "source_weight": signal.source_weight,
        "created_at_total_tokens": signal.created_at_total_tokens,
        "source_terminated": signal.source_terminated,
        "source_termination_reason": signal.source_termination_reason,
        "delivered_to": signal.delivered_to,
    }


def serialize_synthesis(synthesis: "SynthesisResult") -> dict[str, Any]:
    """Serialize a SynthesisResult."""
    return {
        "unified_output": synthesis.unified_output,
        "confidence": synthesis.confidence,
        "evidence_summary": synthesis.evidence_summary,
        "dissenting_views": synthesis.dissenting_views,
        "action_plan": synthesis.action_plan,
        "usage": {
            "input_tokens": synthesis.usage.input_tokens,
            "output_tokens": synthesis.usage.output_tokens,
            "cost": synthesis.usage.cost,
        },
    }


def serialize_diversity_state(state: "DiversityState") -> dict[str, Any]:
    """Serialize a DiversityState (without embeddings - too large)."""
    return {
        "current_score": state.current_score,
        "interpretation": state.interpretation,
        "num_leaves": state.num_leaves,
        "total_embedding_cost": state.total_embedding_cost,
        "last_computed_at": state.last_computed_at,
    }


def serialize_genealogy(genealogy: "GenealogyTree") -> dict[str, Any]:
    """Serialize a GenealogyTree."""
    nodes = {}
    for node_id, node in genealogy.nodes.items():
        nodes[node_id] = {
            "id": node.id,
            "parent_id": node.parent_id,
            "children_ids": node.children_ids,
            "spawn_reason": node.spawn_reason.value if node.spawn_reason else None,
            "spawn_edit": node.spawn_edit,
            "spawn_gate_number": node.spawn_gate_number,
            "spawn_confidence": node.spawn_confidence,
            "parent_velocity": node.parent_velocity,
            "is_leaf": node.is_leaf,
            "leaf_confidence": node.leaf_confidence,
            "termination_reason": node.termination_reason.value if node.termination_reason else None,
            "signals_posted": node.signals_posted,
            "signals_received": node.signals_received,
        }
    return {
        "root_id": genealogy.root_id,
        "nodes": nodes,
    }


def serialize_config(config: "PSSConfig") -> dict[str, Any]:
    """Serialize relevant config fields for trace."""
    return {
        "provider": config.provider,
        "model": config.model,
        "soft_gate_tokens": config.soft_gate_tokens,
        "hard_gate_tokens": config.hard_gate_tokens,
        "per_context_max": config.per_context_max,
        "total_max": config.total_max,
        "max_contexts": config.max_contexts,
        "parallel": config.parallel,
        "max_parallel": config.max_parallel,
        "agentic_enabled": config.agentic_enabled,
        "agentic_tools": config.agentic_tools,
        "synthesis_enabled": config.synthesis_enabled,
        "synthesis_strategy": config.synthesis_strategy,
        "diversity_aware_enabled": config.diversity_aware_enabled,
        "diversity_target": config.diversity_target,
        "diversity_min_threshold": config.diversity_min_threshold,
        "failure_propagation_enabled": config.failure_propagation_enabled,
        "bulletin_enabled": config.bulletin_enabled,
        "adaptive_gates_enabled": config.adaptive_gates_enabled,
    }


def build_trace_from_result(
    prompt: str,
    config: "PSSConfig",
    result: "PSSResult | list[Context]",
    elapsed_seconds: float,
    tree: "SearchTree | None" = None,
    genealogy: "GenealogyTree | None" = None,
    failure_registry: Any = None,
    bulletin: "BulletinBoard | None" = None,
    diversity_state: "DiversityState | None" = None,
) -> ExplorationTrace:
    """Build an ExplorationTrace from a completed PSS run.

    This is the main entry point for trace collection. Call this at the end
    of a PSS run to capture all the data.
    """
    from pss.types import PSSResult, TerminationReason

    trace = ExplorationTrace(
        prompt=prompt,
        config_snapshot=serialize_config(config),
        elapsed_seconds=elapsed_seconds,
    )

    # Handle both PSSResult and list[Context]
    if isinstance(result, PSSResult):
        leaves = result.leaves
        if result.synthesis:
            trace.synthesis = serialize_synthesis(result.synthesis)
        trace.total_usage = {
            "input_tokens": result.total_usage.input_tokens,
            "output_tokens": result.total_usage.output_tokens,
            "cost": result.total_usage.cost,
        }
        tree = result.tree
    else:
        leaves = result

    # Serialize leaves
    trace.leaves = [serialize_context(ctx) for ctx in leaves]
    trace.num_leaves = len(leaves)

    # Termination reason breakdown
    termination_counts: dict[str, int] = {}
    for ctx in leaves:
        reason = ctx.termination_reason.value if ctx.termination_reason else "unknown"
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    trace.leaf_termination_reasons = termination_counts

    # Velocity snapshots
    for ctx in leaves:
        if ctx.velocity_history:
            trace.velocity_snapshots[ctx.id] = [
                serialize_velocity_snapshot(v) for v in ctx.velocity_history
            ]

    # Tree structure
    if tree:
        trace.tree = serialize_search_tree(tree)
        trace.total_contexts_created = len(tree.contexts)
        trace.total_gates_seen = sum(c.gates_seen for c in tree.contexts.values())
        trace.total_branches_spawned = len(tree.contexts) - 1  # exclude root

    # Genealogy
    if genealogy:
        trace.genealogy = serialize_genealogy(genealogy)

    # Failures
    if failure_registry:
        trace.failures = [serialize_failure(f) for f in failure_registry.failures]

    # Signals
    if bulletin:
        trace.signals = [serialize_signal(s) for s in bulletin.signals]

    # Diversity
    if diversity_state:
        trace.diversity_state = serialize_diversity_state(diversity_state)

    # Calculate total usage if not from PSSResult
    if not isinstance(result, PSSResult):
        total_input = sum(ctx.usage.input_tokens for ctx in leaves)
        total_output = sum(ctx.usage.output_tokens for ctx in leaves)
        total_cost = sum(ctx.usage.cost for ctx in leaves)
        trace.total_usage = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost": total_cost,
        }

    return trace


def get_trace_dir() -> Path:
    """Get the default trace directory."""
    return Path("traces")


def save_trace(trace: ExplorationTrace, trace_dir: Path | None = None) -> Path:
    """Save a trace to the default or specified directory.

    Returns the path where the trace was saved.
    """
    if trace_dir is None:
        trace_dir = get_trace_dir()

    # Use timestamp and ID for filename
    from datetime import datetime
    ts = datetime.fromtimestamp(trace.timestamp)
    filename = f"{ts.strftime('%Y%m%d_%H%M%S')}_{trace.id}.json"
    path = trace_dir / filename

    trace.save(path)
    return path


def list_traces(trace_dir: Path | None = None) -> list[Path]:
    """List all trace files in the trace directory."""
    if trace_dir is None:
        trace_dir = get_trace_dir()

    if not trace_dir.exists():
        return []

    return sorted(trace_dir.glob("*.json"), reverse=True)


def load_traces(trace_dir: Path | None = None, limit: int = 100) -> list[ExplorationTrace]:
    """Load traces from the trace directory."""
    paths = list_traces(trace_dir)[:limit]
    return [ExplorationTrace.load(p) for p in paths]


# Analysis utilities

def analyze_velocity_patterns(trace: ExplorationTrace) -> dict[str, Any]:
    """Analyze velocity patterns in a trace.

    Returns insights about velocity trends and their correlation with outcomes.
    """
    results: dict[str, Any] = {
        "branches_with_velocity": 0,
        "avg_final_velocity": 0.0,
        "successful_branch_avg_velocity": 0.0,
        "stuck_branch_avg_velocity": 0.0,
        "velocity_trend_by_outcome": {},
    }

    successful_velocities = []
    stuck_velocities = []
    all_final_velocities = []

    for leaf in trace.leaves:
        branch_id = leaf["id"]
        if branch_id in trace.velocity_snapshots:
            snapshots = trace.velocity_snapshots[branch_id]
            if snapshots:
                results["branches_with_velocity"] += 1
                final_vel = snapshots[-1]["overall_velocity"]
                all_final_velocities.append(final_vel)

                termination = leaf.get("termination_reason", "unknown")
                if termination == "completed":
                    successful_velocities.append(final_vel)
                elif termination == "stuck":
                    stuck_velocities.append(final_vel)

    if all_final_velocities:
        results["avg_final_velocity"] = sum(all_final_velocities) / len(all_final_velocities)
    if successful_velocities:
        results["successful_branch_avg_velocity"] = sum(successful_velocities) / len(successful_velocities)
    if stuck_velocities:
        results["stuck_branch_avg_velocity"] = sum(stuck_velocities) / len(stuck_velocities)

    return results


def analyze_failure_effectiveness(trace: ExplorationTrace) -> dict[str, Any]:
    """Analyze whether failure propagation was effective.

    Looks at whether branches that received failure warnings avoided those paths.
    """
    results: dict[str, Any] = {
        "total_failures_extracted": len(trace.failures),
        "failures_by_type": {},
        "failures_by_scope": {},
    }

    for failure in trace.failures:
        ftype = failure.get("failure_type", "unknown")
        scope = failure.get("scope", "unknown")
        results["failures_by_type"][ftype] = results["failures_by_type"].get(ftype, 0) + 1
        results["failures_by_scope"][scope] = results["failures_by_scope"].get(scope, 0) + 1

    return results


def analyze_signal_utility(trace: ExplorationTrace) -> dict[str, Any]:
    """Analyze signal utility - which signal types were useful."""
    results: dict[str, Any] = {
        "total_signals": len(trace.signals),
        "signals_by_type": {},
        "delivery_rate": 0.0,
    }

    delivered_count = 0
    for signal in trace.signals:
        stype = signal.get("signal_type", "unknown")
        results["signals_by_type"][stype] = results["signals_by_type"].get(stype, 0) + 1
        if signal.get("delivered_to"):
            delivered_count += 1

    if trace.signals:
        results["delivery_rate"] = delivered_count / len(trace.signals)

    return results


def analyze_spawn_confidence(trace: ExplorationTrace) -> dict[str, Any]:
    """Analyze whether spawn confidence predicts branch success."""
    results: dict[str, Any] = {
        "branches_with_confidence": 0,
        "successful_avg_confidence": 0.0,
        "stuck_avg_confidence": 0.0,
        "confidence_by_outcome": {},
    }

    successful_confs = []
    stuck_confs = []

    for leaf in trace.leaves:
        spawn_meta = leaf.get("spawn_metadata")
        if spawn_meta and spawn_meta.get("spawn_confidence") is not None:
            results["branches_with_confidence"] += 1
            conf = spawn_meta["spawn_confidence"]

            termination = leaf.get("termination_reason", "unknown")
            if termination == "completed":
                successful_confs.append(conf)
            elif termination == "stuck":
                stuck_confs.append(conf)

    if successful_confs:
        results["successful_avg_confidence"] = sum(successful_confs) / len(successful_confs)
    if stuck_confs:
        results["stuck_avg_confidence"] = sum(stuck_confs) / len(stuck_confs)

    return results


def generate_trace_summary(trace: ExplorationTrace) -> str:
    """Generate a human-readable summary of a trace."""
    lines = [
        f"Trace: {trace.id}",
        f"Prompt: {trace.prompt[:100]}..." if len(trace.prompt) > 100 else f"Prompt: {trace.prompt}",
        f"Duration: {trace.elapsed_seconds:.1f}s",
        f"Cost: ${trace.total_usage['cost']:.4f}",
        f"Tokens: {trace.total_usage['input_tokens'] + trace.total_usage['output_tokens']:,}",
        f"Contexts: {trace.total_contexts_created}",
        f"Leaves: {trace.num_leaves}",
        "",
        "Termination breakdown:",
    ]

    for reason, count in trace.leaf_termination_reasons.items():
        lines.append(f"  {reason}: {count}")

    if trace.human_rating:
        lines.append(f"\nHuman rating: {trace.human_rating}/5")
    if trace.human_notes:
        lines.append(f"Notes: {trace.human_notes}")

    return "\n".join(lines)
