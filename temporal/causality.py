"""Causal chain tracking for the Temporal Scenario Planner.

This module provides causal graph construction, traversal, and analysis
to understand how decisions lead to outcomes.
"""

from dataclasses import dataclass, field
from typing import Any

from .types import (
    CausalNode,
    Decision,
    Event,
    LeveragePoint,
    Timeline,
)


@dataclass
class CausalGraph:
    """Directed acyclic graph of decisions and consequences.

    Attributes:
        nodes: All nodes in the graph (events and decisions).
        edges: Edges from cause to effect.
        root_nodes: Nodes with no causes.
        leaf_nodes: Nodes with no effects.
    """
    nodes: dict[str, CausalNode] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)
    reverse_edges: dict[str, list[str]] = field(default_factory=dict)

    @property
    def root_nodes(self) -> list[str]:
        """Get nodes with no causes."""
        return [
            node_id for node_id in self.nodes
            if node_id not in self.reverse_edges or not self.reverse_edges[node_id]
        ]

    @property
    def leaf_nodes(self) -> list[str]:
        """Get nodes with no effects."""
        return [
            node_id for node_id in self.nodes
            if node_id not in self.edges or not self.edges[node_id]
        ]

    def add_node(self, node: CausalNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []
        if node.id not in self.reverse_edges:
            self.reverse_edges[node.id] = []

    def add_edge(self, from_id: str, to_id: str, strength: float = 1.0) -> None:
        """Add a causal edge."""
        if from_id not in self.edges:
            self.edges[from_id] = []
        if to_id not in self.reverse_edges:
            self.reverse_edges[to_id] = []

        if to_id not in self.edges[from_id]:
            self.edges[from_id].append(to_id)
        if from_id not in self.reverse_edges[to_id]:
            self.reverse_edges[to_id].append(from_id)

        # Update node cause/effect lists
        if from_id in self.nodes:
            if to_id not in self.nodes[from_id].effects:
                self.nodes[from_id].effects.append(to_id)
        if to_id in self.nodes:
            if from_id not in self.nodes[to_id].causes:
                self.nodes[to_id].causes.append(from_id)

    def get_descendants(self, node_id: str) -> set[str]:
        """Get all descendants of a node (transitive closure)."""
        descendants: set[str] = set()
        to_visit = list(self.edges.get(node_id, []))

        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.edges.get(current, []))

        return descendants

    def get_ancestors(self, node_id: str) -> set[str]:
        """Get all ancestors of a node (transitive closure)."""
        ancestors: set[str] = set()
        to_visit = list(self.reverse_edges.get(node_id, []))

        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.reverse_edges.get(current, []))

        return ancestors


def build_causal_graph(timeline: Timeline) -> CausalGraph:
    """Build a causal graph from a timeline.

    Args:
        timeline: Timeline to analyze.

    Returns:
        CausalGraph representing the timeline's causal structure.
    """
    graph = CausalGraph()

    # Add decision nodes
    for decision in timeline.decisions:
        node = CausalNode(
            id=decision.id,
            node_type="decision",
            description=f"{decision.description}: {decision.chosen}",
            causes=[],
            effects=[],
            strength=1.0,
        )
        graph.add_node(node)

    # Add event nodes and connect causes
    for event in timeline.events:
        node = CausalNode(
            id=event.id,
            node_type="event",
            description=event.description,
            causes=[event.cause] if event.cause else [],
            effects=[],
            strength=abs(event.magnitude) / 10.0,
        )
        graph.add_node(node)

        # Connect to cause
        if event.cause and event.cause in graph.nodes:
            graph.add_edge(event.cause, event.id)

    # Infer additional causal links from temporal proximity and effects
    _infer_causal_links(graph, timeline)

    return graph


def _infer_causal_links(graph: CausalGraph, timeline: Timeline) -> None:
    """Infer causal links not explicitly stated."""
    # Decisions that happen before events might cause them
    decisions_by_date = sorted(timeline.decisions, key=lambda d: d.date)
    events_by_date = sorted(timeline.events, key=lambda e: e.date)

    for event in events_by_date:
        if event.cause:
            continue

        # Find most recent decision
        recent_decisions = [
            d for d in decisions_by_date
            if d.date < event.date
        ]

        if recent_decisions:
            most_recent = recent_decisions[-1]
            # Weak causal link
            if most_recent.id in graph.nodes and event.id in graph.nodes:
                graph.add_edge(most_recent.id, event.id)
                graph.nodes[event.id].strength = 0.3


def trace_cause(graph: CausalGraph, node_id: str) -> list[list[str]]:
    """Trace all causal chains leading to a node.

    Args:
        graph: Causal graph.
        node_id: Node to trace back from.

    Returns:
        List of causal chains (each chain is a list of node IDs).
    """
    if node_id not in graph.nodes:
        return []

    chains: list[list[str]] = []

    def _trace_back(current_id: str, current_chain: list[str]) -> None:
        causes = graph.reverse_edges.get(current_id, [])
        if not causes:
            # Reached a root node
            chains.append(list(reversed(current_chain)))
            return

        for cause_id in causes:
            if cause_id not in current_chain:  # Avoid cycles
                _trace_back(cause_id, current_chain + [cause_id])

    _trace_back(node_id, [node_id])
    return chains


def trace_effects(graph: CausalGraph, node_id: str) -> list[list[str]]:
    """Trace all consequence chains from a node.

    Args:
        graph: Causal graph.
        node_id: Node to trace forward from.

    Returns:
        List of effect chains (each chain is a list of node IDs).
    """
    if node_id not in graph.nodes:
        return []

    chains: list[list[str]] = []

    def _trace_forward(current_id: str, current_chain: list[str]) -> None:
        effects = graph.edges.get(current_id, [])
        if not effects:
            # Reached a leaf node
            chains.append(current_chain)
            return

        for effect_id in effects:
            if effect_id not in current_chain:  # Avoid cycles
                _trace_forward(effect_id, current_chain + [effect_id])

    _trace_forward(node_id, [node_id])
    return chains


def find_leverage_points(graph: CausalGraph) -> list[LeveragePoint]:
    """Find decisions with outsized impact.

    Leverage points are decisions that lead to many downstream effects
    or have high-strength connections.

    Args:
        graph: Causal graph to analyze.

    Returns:
        List of leverage points, sorted by impact.
    """
    leverage_points: list[LeveragePoint] = []

    for node_id, node in graph.nodes.items():
        if node.node_type != "decision":
            continue

        # Count downstream nodes
        descendants = graph.get_descendants(node_id)
        downstream_count = len(descendants)

        # Calculate impact score
        # Impact = direct effects * strength + downstream reach
        direct_effects = len(graph.edges.get(node_id, []))
        impact_score = (direct_effects * node.strength) + (downstream_count * 0.5)

        # Check if effects are reversible (simplistic heuristic)
        reversibility = _estimate_reversibility(graph, node_id)

        # Timing sensitivity based on how early in the chain
        depth = _calculate_depth(graph, node_id)
        timing_sensitivity = 1.0 / (depth + 1)

        leverage_points.append(LeveragePoint(
            node_id=node_id,
            impact_score=impact_score,
            downstream_count=downstream_count,
            reversibility=reversibility,
            timing_sensitivity=timing_sensitivity,
        ))

    # Sort by impact
    leverage_points.sort(key=lambda lp: lp.impact_score, reverse=True)
    return leverage_points


def _estimate_reversibility(graph: CausalGraph, node_id: str) -> float:
    """Estimate how reversible a decision's effects are."""
    descendants = graph.get_descendants(node_id)
    if not descendants:
        return 1.0  # No effects = fully reversible

    # More downstream effects = less reversible
    reversibility = 1.0 / (1.0 + len(descendants) * 0.2)
    return reversibility


def _calculate_depth(graph: CausalGraph, node_id: str) -> int:
    """Calculate depth of a node from root."""
    ancestors = graph.get_ancestors(node_id)
    if not ancestors:
        return 0

    # Find maximum path length to any root
    max_depth = 0
    for root in graph.root_nodes:
        if root in ancestors or root == node_id:
            paths = trace_cause(graph, node_id)
            for path in paths:
                if path[0] == root:
                    max_depth = max(max_depth, len(path) - 1)
    return max_depth


def critical_path(graph: CausalGraph) -> list[str]:
    """Find the critical path through the causal graph.

    The critical path is the longest chain from any root to any leaf,
    representing the most significant causal sequence.

    Args:
        graph: Causal graph.

    Returns:
        List of node IDs representing the critical path.
    """
    longest_path: list[str] = []

    for root in graph.root_nodes:
        paths = trace_effects(graph, root)
        for path in paths:
            if len(path) > len(longest_path):
                longest_path = path

    return longest_path


def counterfactual_analysis(
    graph: CausalGraph,
    decision_id: str,
    alternate_choice: str,
) -> dict[str, Any]:
    """Analyze what would happen with a different decision.

    Args:
        graph: Causal graph.
        decision_id: Decision to vary.
        alternate_choice: What would have been chosen instead.

    Returns:
        Analysis of counterfactual scenario.
    """
    if decision_id not in graph.nodes:
        return {"error": "Decision not found"}

    node = graph.nodes[decision_id]
    if node.node_type != "decision":
        return {"error": "Node is not a decision"}

    # Get all downstream effects
    affected_nodes = graph.get_descendants(decision_id)

    # Calculate impact
    analysis = {
        "decision": node.description,
        "alternate_choice": alternate_choice,
        "affected_nodes_count": len(affected_nodes),
        "affected_events": [],
        "cascade_depth": 0,
        "estimated_impact": "unknown",
    }

    # List affected events
    for affected_id in affected_nodes:
        if affected_id in graph.nodes:
            affected_node = graph.nodes[affected_id]
            if affected_node.node_type == "event":
                analysis["affected_events"].append(affected_node.description)

    # Calculate cascade depth
    chains = trace_effects(graph, decision_id)
    if chains:
        analysis["cascade_depth"] = max(len(chain) for chain in chains)

    # Estimate impact magnitude
    leverage_points = find_leverage_points(graph)
    for lp in leverage_points:
        if lp.node_id == decision_id:
            if lp.impact_score > 5:
                analysis["estimated_impact"] = "high"
            elif lp.impact_score > 2:
                analysis["estimated_impact"] = "medium"
            else:
                analysis["estimated_impact"] = "low"
            break

    return analysis


def visualize_graph(graph: CausalGraph, max_nodes: int = 20) -> str:
    """Create a text visualization of the causal graph.

    Args:
        graph: Graph to visualize.
        max_nodes: Maximum nodes to show.

    Returns:
        ASCII art representation.
    """
    lines = ["Causal Graph Visualization", "=" * 40]

    # Show roots first
    roots = graph.root_nodes[:max_nodes // 2]
    lines.append("\nRoot causes:")
    for root_id in roots:
        node = graph.nodes.get(root_id)
        if node:
            lines.append(f"  [{node.node_type[0].upper()}] {node.description[:50]}")

    # Show key paths
    lines.append("\nKey causal chains:")
    shown = 0
    for root in roots:
        if shown >= 3:
            break
        chains = trace_effects(graph, root)
        if chains:
            longest = max(chains, key=len)
            if len(longest) > 1:
                lines.append(f"\n  Chain from {root}:")
                for i, node_id in enumerate(longest[:5]):
                    node = graph.nodes.get(node_id)
                    if node:
                        prefix = "    " + "->" * i
                        lines.append(f"{prefix} {node.description[:40]}")
                shown += 1

    # Show leverage points
    leverage = find_leverage_points(graph)[:3]
    if leverage:
        lines.append("\nTop leverage points:")
        for lp in leverage:
            node = graph.nodes.get(lp.node_id)
            if node:
                lines.append(f"  - {node.description[:40]} (impact: {lp.impact_score:.1f})")

    return "\n".join(lines)


def merge_graphs(graphs: list[CausalGraph]) -> CausalGraph:
    """Merge multiple causal graphs.

    Args:
        graphs: Graphs to merge.

    Returns:
        Merged graph with all nodes and edges.
    """
    merged = CausalGraph()

    for graph in graphs:
        for node_id, node in graph.nodes.items():
            if node_id not in merged.nodes:
                merged.add_node(node)

        for from_id, to_ids in graph.edges.items():
            for to_id in to_ids:
                merged.add_edge(from_id, to_id)

    return merged


def find_common_causes(graph: CausalGraph, node_ids: list[str]) -> list[str]:
    """Find common causes of multiple nodes.

    Args:
        graph: Causal graph.
        node_ids: Nodes to find common causes for.

    Returns:
        List of common ancestor node IDs.
    """
    if not node_ids:
        return []

    ancestor_sets = [graph.get_ancestors(node_id) for node_id in node_ids]
    if not ancestor_sets:
        return []

    common = ancestor_sets[0]
    for ancestors in ancestor_sets[1:]:
        common = common & ancestors

    return list(common)


def find_divergence_point(
    graph: CausalGraph,
    node_a: str,
    node_b: str,
) -> str | None:
    """Find where two causal chains diverged.

    Args:
        graph: Causal graph.
        node_a: First node.
        node_b: Second node.

    Returns:
        Node ID where chains diverged, or None.
    """
    ancestors_a = graph.get_ancestors(node_a)
    ancestors_b = graph.get_ancestors(node_b)

    common = ancestors_a & ancestors_b
    if not common:
        return None

    # Find the most recent common ancestor
    # (the one that's closest to both nodes)
    best_divergence = None
    best_distance = float("inf")

    for ancestor_id in common:
        # Calculate distance to both nodes
        dist_a = len(trace_effects(graph, ancestor_id))
        dist_b = len(trace_effects(graph, ancestor_id))
        total_dist = dist_a + dist_b

        if total_dist < best_distance:
            best_distance = total_dist
            best_divergence = ancestor_id

    return best_divergence
