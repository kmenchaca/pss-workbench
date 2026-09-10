"""Branch genealogy tracking for PSS v1.0.

This module tracks the full lineage of branches—who spawned whom, at what gates,
with what confidence—and provides traversal methods and weight computation for
synthesis.

GenealogyTree wraps SearchTree rather than replacing it, preserving backward
compatibility with existing tests and code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pss.types import (
    Context,
    SearchTree,
    SpawnMetadata,
    SpawnReason,
    TerminationReason,
    VelocitySnapshot,
)

if TYPE_CHECKING:
    pass


@dataclass
class BranchNode:
    """Rich node in the genealogy tree.

    This captures genealogy-specific data that Context doesn't need to know about.
    It references the Context by ID rather than holding a copy.
    """

    id: str
    parent_id: str | None
    children_ids: list[str] = field(default_factory=list)

    # Spawn metadata (copied from Context.spawn_metadata for convenience)
    spawn_reason: SpawnReason = SpawnReason.INITIAL
    spawn_edit: str = ""
    spawn_gate_number: int = 0
    spawn_confidence: float = 0.5
    parent_velocity: float | None = None

    # Terminal state
    is_leaf: bool = False
    leaf_confidence: float | None = None
    termination_reason: TerminationReason | None = None

    # Signals (for bulletin board integration)
    signals_posted: list[str] = field(default_factory=list)
    signals_received: list[str] = field(default_factory=list)


@dataclass
class GenealogyTree:
    """Full genealogy with traversal methods.

    Wraps SearchTree rather than replacing it. Call sync_from_search_tree()
    after the SearchTree changes to update genealogy state.
    """

    nodes: dict[str, BranchNode] = field(default_factory=dict)
    root_id: str = "root"
    _search_tree: SearchTree | None = None

    @classmethod
    def from_search_tree(cls, tree: SearchTree) -> GenealogyTree:
        """Build genealogy from existing SearchTree."""
        genealogy = cls(_search_tree=tree, root_id=tree.root_id)
        genealogy.sync_from_search_tree()
        return genealogy

    def sync_from_search_tree(self) -> None:
        """Update genealogy nodes from SearchTree state.

        Call this after any tree modification (spawn, terminate, etc.).
        """
        if self._search_tree is None:
            return

        tree = self._search_tree

        # Build/update nodes for all contexts
        for ctx_id, ctx in tree.contexts.items():
            if ctx_id not in self.nodes:
                self.nodes[ctx_id] = self._create_node_from_context(ctx)
            else:
                self._update_node_from_context(self.nodes[ctx_id], ctx)

        # Rebuild children lists
        for node in self.nodes.values():
            node.children_ids = []

        for node in self.nodes.values():
            if node.parent_id and node.parent_id in self.nodes:
                self.nodes[node.parent_id].children_ids.append(node.id)

        # Mark leaves
        for leaf_id in tree.leaves:
            if leaf_id in self.nodes:
                self.nodes[leaf_id].is_leaf = True

    def _create_node_from_context(self, ctx: Context) -> BranchNode:
        """Create a BranchNode from a Context."""
        node = BranchNode(
            id=ctx.id,
            parent_id=ctx.parent_id,
        )

        if ctx.spawn_metadata:
            node.spawn_reason = ctx.spawn_metadata.spawn_reason
            node.spawn_edit = ctx.spawn_metadata.spawn_edit
            node.spawn_gate_number = ctx.spawn_metadata.spawn_gate_number
            node.spawn_confidence = ctx.spawn_metadata.spawn_confidence
            node.parent_velocity = ctx.spawn_metadata.parent_velocity
        elif ctx.parent_id is None:
            # Root node
            node.spawn_reason = SpawnReason.INITIAL
            node.spawn_confidence = 1.0

        node.termination_reason = ctx.termination_reason
        node.is_leaf = ctx.output is not None

        return node

    def _update_node_from_context(self, node: BranchNode, ctx: Context) -> None:
        """Update an existing node from Context state."""
        node.termination_reason = ctx.termination_reason
        node.is_leaf = ctx.output is not None

        # Update leaf confidence if available
        # (would come from gate decision, stored elsewhere)

    def get_ancestors(self, node_id: str) -> list[BranchNode]:
        """Return path from root to node (excluding the node itself)."""
        ancestors = []
        current_id = node_id

        while current_id in self.nodes:
            node = self.nodes[current_id]
            if node.parent_id is None:
                break
            if node.parent_id not in self.nodes:
                break
            ancestors.append(self.nodes[node.parent_id])
            current_id = node.parent_id

        ancestors.reverse()
        return ancestors

    def get_descendants(self, node_id: str) -> list[BranchNode]:
        """Return all nodes spawned from this one (breadth-first)."""
        if node_id not in self.nodes:
            return []

        descendants = []
        queue = list(self.nodes[node_id].children_ids)

        while queue:
            child_id = queue.pop(0)
            if child_id in self.nodes:
                child = self.nodes[child_id]
                descendants.append(child)
                queue.extend(child.children_ids)

        return descendants

    def get_siblings(self, node_id: str) -> list[BranchNode]:
        """Return nodes with same parent (excluding the node itself)."""
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        if node.parent_id is None or node.parent_id not in self.nodes:
            return []

        parent = self.nodes[node.parent_id]
        return [
            self.nodes[child_id]
            for child_id in parent.children_ids
            if child_id != node_id and child_id in self.nodes
        ]

    def get_leaves(self) -> list[BranchNode]:
        """Return all terminal nodes."""
        return [node for node in self.nodes.values() if node.is_leaf]

    def get_depth(self, node_id: str) -> int:
        """How many generations from root (root = 0)."""
        return len(self.get_ancestors(node_id))

    def get_branch_path(self, node_id: str) -> list[str]:
        """Return the edit chain that led to this branch.

        E.g., ["explore auth", "focus on OAuth", "check token refresh"]
        """
        ancestors = self.get_ancestors(node_id)

        # Include edits from ancestors and the node itself
        path = [a.spawn_edit for a in ancestors if a.spawn_edit]

        if node_id in self.nodes and self.nodes[node_id].spawn_edit:
            path.append(self.nodes[node_id].spawn_edit)

        return path

    def compute_leaf_weight(self, leaf_id: str) -> float:
        """Compute synthesis weight for a leaf based on its lineage.

        Factors:
        - Depth: deeper = slight bonus (survived more gates)
        - Spawn confidence: higher = bonus (parent thought promising)
        - Sibling survival rate: lower = bonus (found the good path)
        - Termination reason: completed > stuck

        Returns weight in range [0.15, 1.0].
        """
        if leaf_id not in self.nodes:
            return 0.15  # minimum weight

        node = self.nodes[leaf_id]

        # Base weight
        weight = 0.15

        # Depth factor: max 15% bonus for depth >= 5
        depth = self.get_depth(leaf_id)
        depth_factor = min(depth / 5, 1.0) * 0.15
        weight += depth_factor

        # Confidence factor: up to 25% based on spawn confidence
        confidence_factor = node.spawn_confidence * 0.25
        weight += confidence_factor

        # Sibling survival factor: up to 20% if siblings died
        siblings = self.get_siblings(leaf_id)
        if siblings:
            dead_siblings = [
                s
                for s in siblings
                if s.termination_reason
                in (TerminationReason.STUCK, TerminationReason.KILLED)
            ]
            survival_factor = (len(dead_siblings) / len(siblings)) * 0.2
            weight += survival_factor

        # Termination reason factor: up to 15% for clean completion
        if node.termination_reason == TerminationReason.COMPLETED:
            weight += 0.15
        elif node.termination_reason == TerminationReason.BUDGET:
            weight += 0.05  # partial credit for budget exhaustion

        # Velocity factor would go here once adaptive gates are implemented
        # For now, skip velocity weighting

        return min(weight, 1.0)

    def format_for_synthesis(self, include_weights: bool = True) -> str:
        """Format the tree structure for synthesis prompts.

        Returns a text visualization of the genealogy.
        """
        lines = ["Branch Genealogy:"]

        def format_node(node_id: str, prefix: str, is_last: bool) -> list[str]:
            if node_id not in self.nodes:
                return []

            node = self.nodes[node_id]
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            # Node status
            status = ""
            if node.is_leaf:
                weight = self.compute_leaf_weight(node_id) if include_weights else None
                weight_str = f" [weight: {weight:.2f}]" if weight else ""
                if node.termination_reason == TerminationReason.COMPLETED:
                    status = f"✓ LEAF{weight_str}"
                elif node.termination_reason == TerminationReason.BUDGET:
                    status = f"~ PARTIAL{weight_str}"
                else:
                    status = f"? LEAF{weight_str}"
            elif node.termination_reason == TerminationReason.STUCK:
                status = "✗ STUCK"
            elif node.termination_reason == TerminationReason.KILLED:
                status = "✗ KILLED"

            # Format the edit
            edit_str = f' "{node.spawn_edit[:30]}..."' if len(node.spawn_edit) > 30 else f' "{node.spawn_edit}"' if node.spawn_edit else ""

            result = [f"{prefix}{connector}{node_id}{edit_str} {status}".rstrip()]

            # Recurse to children
            children = node.children_ids
            for i, child_id in enumerate(children):
                is_child_last = i == len(children) - 1
                result.extend(format_node(child_id, child_prefix, is_child_last))

            return result

        # Start from root
        if self.root_id in self.nodes:
            root = self.nodes[self.root_id]
            lines.append(f"├── {self.root_id} (initial)")
            for i, child_id in enumerate(root.children_ids):
                is_last = i == len(root.children_ids) - 1
                lines.extend(format_node(child_id, "│   ", is_last))

        return "\n".join(lines)
