"""Tests for pss.genealogy module."""

import pytest

from pss.genealogy import BranchNode, GenealogyTree
from pss.types import (
    Context,
    SearchTree,
    SpawnMetadata,
    SpawnReason,
    TerminationReason,
)


def make_context(
    id: str,
    parent_id: str | None = None,
    spawn_metadata: SpawnMetadata | None = None,
    output: str | None = None,
    termination_reason: TerminationReason | None = None,
) -> Context:
    """Helper to create a Context for testing."""
    return Context(
        id=id,
        parent_id=parent_id,
        spawn_metadata=spawn_metadata,
        output=output,
        termination_reason=termination_reason,
    )


def make_spawn_metadata(
    reason: SpawnReason = SpawnReason.GATE_BRANCH,
    edit: str = "test edit",
    gate_number: int = 1,
    confidence: float = 0.5,
) -> SpawnMetadata:
    """Helper to create SpawnMetadata for testing."""
    return SpawnMetadata(
        spawn_reason=reason,
        spawn_edit=edit,
        spawn_gate_number=gate_number,
        spawn_confidence=confidence,
    )


class TestGenealogyTreeFromSearchTree:
    """Tests for GenealogyTree.from_search_tree()."""

    def test_creates_genealogy_from_empty_tree(self):
        tree = SearchTree()
        tree.root_id = "root"

        genealogy = GenealogyTree.from_search_tree(tree)

        assert genealogy.root_id == "root"
        assert genealogy._search_tree is tree
        assert len(genealogy.nodes) == 0

    def test_creates_genealogy_from_single_root(self):
        tree = SearchTree()
        tree.root_id = "root"
        tree.contexts["root"] = make_context("root")

        genealogy = GenealogyTree.from_search_tree(tree)

        assert "root" in genealogy.nodes
        assert genealogy.nodes["root"].parent_id is None
        assert genealogy.nodes["root"].spawn_reason == SpawnReason.INITIAL

    def test_creates_genealogy_with_branches(self):
        tree = SearchTree()
        tree.root_id = "root"
        tree.contexts["root"] = make_context("root")
        tree.contexts["root->a"] = make_context(
            "root->a",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(
                reason=SpawnReason.GATE_BRANCH,
                edit="explore option A",
                confidence=0.8,
            ),
        )
        tree.contexts["root->b"] = make_context(
            "root->b",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(
                reason=SpawnReason.GATE_BRANCH,
                edit="explore option B",
                confidence=0.6,
            ),
        )

        genealogy = GenealogyTree.from_search_tree(tree)

        assert len(genealogy.nodes) == 3
        assert genealogy.nodes["root->a"].spawn_edit == "explore option A"
        assert genealogy.nodes["root->a"].spawn_confidence == 0.8
        assert genealogy.nodes["root->b"].spawn_edit == "explore option B"
        # Check children are tracked
        assert "root->a" in genealogy.nodes["root"].children_ids
        assert "root->b" in genealogy.nodes["root"].children_ids


class TestGenealogyTreeTraversal:
    """Tests for tree traversal methods."""

    @pytest.fixture
    def deep_tree(self) -> GenealogyTree:
        """Create a tree with depth 4:

        root
        ├── a
        │   ├── a1 (leaf, completed)
        │   └── a2 (leaf, stuck)
        └── b
            └── b1 (leaf, completed)
        """
        tree = SearchTree()
        tree.root_id = "root"

        tree.contexts["root"] = make_context("root")
        tree.contexts["a"] = make_context(
            "a",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(edit="path A", confidence=0.8),
        )
        tree.contexts["b"] = make_context(
            "b",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(edit="path B", confidence=0.6),
        )
        tree.contexts["a1"] = make_context(
            "a1",
            parent_id="a",
            spawn_metadata=make_spawn_metadata(edit="refine A1", confidence=0.9),
            output="Result A1",
            termination_reason=TerminationReason.COMPLETED,
        )
        tree.contexts["a2"] = make_context(
            "a2",
            parent_id="a",
            spawn_metadata=make_spawn_metadata(edit="refine A2", confidence=0.4),
            termination_reason=TerminationReason.STUCK,
        )
        tree.contexts["b1"] = make_context(
            "b1",
            parent_id="b",
            spawn_metadata=make_spawn_metadata(edit="refine B1", confidence=0.7),
            output="Result B1",
            termination_reason=TerminationReason.COMPLETED,
        )

        tree.leaves = ["a1", "b1"]  # a2 is stuck, not a successful leaf

        return GenealogyTree.from_search_tree(tree)

    def test_get_ancestors_root(self, deep_tree: GenealogyTree):
        """Root has no ancestors."""
        ancestors = deep_tree.get_ancestors("root")
        assert ancestors == []

    def test_get_ancestors_first_level(self, deep_tree: GenealogyTree):
        """First level child has root as ancestor."""
        ancestors = deep_tree.get_ancestors("a")
        assert len(ancestors) == 1
        assert ancestors[0].id == "root"

    def test_get_ancestors_deep(self, deep_tree: GenealogyTree):
        """Deep node has all ancestors in order."""
        ancestors = deep_tree.get_ancestors("a1")
        assert len(ancestors) == 2
        assert ancestors[0].id == "root"
        assert ancestors[1].id == "a"

    def test_get_descendants_leaf(self, deep_tree: GenealogyTree):
        """Leaf has no descendants."""
        descendants = deep_tree.get_descendants("a1")
        assert descendants == []

    def test_get_descendants_root(self, deep_tree: GenealogyTree):
        """Root has all other nodes as descendants."""
        descendants = deep_tree.get_descendants("root")
        assert len(descendants) == 5
        descendant_ids = [d.id for d in descendants]
        assert "a" in descendant_ids
        assert "b" in descendant_ids
        assert "a1" in descendant_ids
        assert "a2" in descendant_ids
        assert "b1" in descendant_ids

    def test_get_descendants_middle(self, deep_tree: GenealogyTree):
        """Middle node has its children as descendants."""
        descendants = deep_tree.get_descendants("a")
        assert len(descendants) == 2
        descendant_ids = [d.id for d in descendants]
        assert "a1" in descendant_ids
        assert "a2" in descendant_ids

    def test_get_siblings_root(self, deep_tree: GenealogyTree):
        """Root has no siblings."""
        siblings = deep_tree.get_siblings("root")
        assert siblings == []

    def test_get_siblings_first_level(self, deep_tree: GenealogyTree):
        """First level nodes are siblings of each other."""
        siblings = deep_tree.get_siblings("a")
        assert len(siblings) == 1
        assert siblings[0].id == "b"

    def test_get_siblings_excludes_self(self, deep_tree: GenealogyTree):
        """Siblings list excludes the node itself."""
        siblings = deep_tree.get_siblings("a1")
        sibling_ids = [s.id for s in siblings]
        assert "a1" not in sibling_ids
        assert "a2" in sibling_ids

    def test_get_depth_root(self, deep_tree: GenealogyTree):
        """Root is at depth 0."""
        assert deep_tree.get_depth("root") == 0

    def test_get_depth_first_level(self, deep_tree: GenealogyTree):
        """First level children are at depth 1."""
        assert deep_tree.get_depth("a") == 1
        assert deep_tree.get_depth("b") == 1

    def test_get_depth_second_level(self, deep_tree: GenealogyTree):
        """Second level children are at depth 2."""
        assert deep_tree.get_depth("a1") == 2
        assert deep_tree.get_depth("b1") == 2

    def test_get_branch_path_root(self, deep_tree: GenealogyTree):
        """Root has empty branch path."""
        path = deep_tree.get_branch_path("root")
        assert path == []

    def test_get_branch_path_deep(self, deep_tree: GenealogyTree):
        """Deep node has full edit chain."""
        path = deep_tree.get_branch_path("a1")
        assert path == ["path A", "refine A1"]

    def test_get_leaves(self, deep_tree: GenealogyTree):
        """Get all leaf nodes."""
        leaves = deep_tree.get_leaves()
        leaf_ids = [l.id for l in leaves]
        assert "a1" in leaf_ids
        assert "b1" in leaf_ids


class TestLeafWeightComputation:
    """Tests for compute_leaf_weight()."""

    def test_minimum_weight_for_unknown_node(self):
        """Unknown node gets minimum weight."""
        genealogy = GenealogyTree()
        weight = genealogy.compute_leaf_weight("nonexistent")
        assert weight == 0.15

    def test_weight_includes_depth_bonus(self):
        """Deeper nodes get depth bonus."""
        tree = SearchTree()
        tree.root_id = "root"

        # Create a deep chain: root -> a -> b -> c -> d -> leaf
        tree.contexts["root"] = make_context("root")
        for i, (child, parent) in enumerate([
            ("a", "root"), ("b", "a"), ("c", "b"), ("d", "c"), ("leaf", "d")
        ]):
            tree.contexts[child] = make_context(
                child,
                parent_id=parent,
                spawn_metadata=make_spawn_metadata(confidence=0.5),
                output="output" if child == "leaf" else None,
                termination_reason=TerminationReason.COMPLETED if child == "leaf" else None,
            )
        tree.leaves = ["leaf"]

        genealogy = GenealogyTree.from_search_tree(tree)

        # Shallow leaf (depth 1)
        tree2 = SearchTree()
        tree2.root_id = "root"
        tree2.contexts["root"] = make_context("root")
        tree2.contexts["shallow"] = make_context(
            "shallow",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(confidence=0.5),
            output="output",
            termination_reason=TerminationReason.COMPLETED,
        )
        tree2.leaves = ["shallow"]
        genealogy2 = GenealogyTree.from_search_tree(tree2)

        deep_weight = genealogy.compute_leaf_weight("leaf")
        shallow_weight = genealogy2.compute_leaf_weight("shallow")

        # Deep node (depth 5) should have more weight than shallow (depth 1)
        assert deep_weight > shallow_weight

    def test_weight_includes_confidence_bonus(self):
        """Higher spawn confidence gives higher weight."""
        tree = SearchTree()
        tree.root_id = "root"
        tree.contexts["root"] = make_context("root")

        # High confidence branch
        tree.contexts["high_conf"] = make_context(
            "high_conf",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(confidence=0.9),
            output="output",
            termination_reason=TerminationReason.COMPLETED,
        )
        # Low confidence branch
        tree.contexts["low_conf"] = make_context(
            "low_conf",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(confidence=0.2),
            output="output",
            termination_reason=TerminationReason.COMPLETED,
        )
        tree.leaves = ["high_conf", "low_conf"]

        genealogy = GenealogyTree.from_search_tree(tree)

        high_weight = genealogy.compute_leaf_weight("high_conf")
        low_weight = genealogy.compute_leaf_weight("low_conf")

        assert high_weight > low_weight

    def test_weight_includes_sibling_survival_bonus(self):
        """Node with dead siblings gets survival bonus."""
        tree = SearchTree()
        tree.root_id = "root"
        tree.contexts["root"] = make_context("root")

        # Survivor with dead sibling
        tree.contexts["survivor"] = make_context(
            "survivor",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(confidence=0.5),
            output="output",
            termination_reason=TerminationReason.COMPLETED,
        )
        tree.contexts["dead_sibling"] = make_context(
            "dead_sibling",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(confidence=0.5),
            termination_reason=TerminationReason.STUCK,
        )
        tree.leaves = ["survivor"]

        genealogy = GenealogyTree.from_search_tree(tree)

        # Lone survivor (no siblings for comparison)
        tree2 = SearchTree()
        tree2.root_id = "root"
        tree2.contexts["root"] = make_context("root")
        tree2.contexts["lone"] = make_context(
            "lone",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(confidence=0.5),
            output="output",
            termination_reason=TerminationReason.COMPLETED,
        )
        tree2.leaves = ["lone"]
        genealogy2 = GenealogyTree.from_search_tree(tree2)

        survivor_weight = genealogy.compute_leaf_weight("survivor")
        lone_weight = genealogy2.compute_leaf_weight("lone")

        # Survivor with dead sibling should have more weight
        assert survivor_weight > lone_weight

    def test_weight_includes_completion_bonus(self):
        """COMPLETED termination gives bonus over BUDGET."""
        tree = SearchTree()
        tree.root_id = "root"
        tree.contexts["root"] = make_context("root")

        tree.contexts["completed"] = make_context(
            "completed",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(confidence=0.5),
            output="output",
            termination_reason=TerminationReason.COMPLETED,
        )
        tree.contexts["budget"] = make_context(
            "budget",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(confidence=0.5),
            output="partial output",
            termination_reason=TerminationReason.BUDGET,
        )
        tree.leaves = ["completed", "budget"]

        genealogy = GenealogyTree.from_search_tree(tree)

        completed_weight = genealogy.compute_leaf_weight("completed")
        budget_weight = genealogy.compute_leaf_weight("budget")

        assert completed_weight > budget_weight

    def test_weight_capped_at_one(self):
        """Weight is capped at 1.0 even with all bonuses."""
        tree = SearchTree()
        tree.root_id = "root"
        tree.contexts["root"] = make_context("root")

        # Create a perfect scenario: deep, high confidence, dead siblings, completed
        prev = "root"
        for i in range(6):
            name = f"level{i}"
            tree.contexts[name] = make_context(
                name,
                parent_id=prev,
                spawn_metadata=make_spawn_metadata(confidence=1.0),
            )
            prev = name

        tree.contexts["perfect"] = make_context(
            "perfect",
            parent_id=prev,
            spawn_metadata=make_spawn_metadata(confidence=1.0),
            output="perfect output",
            termination_reason=TerminationReason.COMPLETED,
        )
        tree.contexts["dead_sibling"] = make_context(
            "dead_sibling",
            parent_id=prev,
            spawn_metadata=make_spawn_metadata(confidence=0.5),
            termination_reason=TerminationReason.STUCK,
        )
        tree.leaves = ["perfect"]

        genealogy = GenealogyTree.from_search_tree(tree)
        weight = genealogy.compute_leaf_weight("perfect")

        assert weight <= 1.0


class TestFormatForSynthesis:
    """Tests for format_for_synthesis()."""

    def test_format_empty_tree(self):
        """Empty tree formats without error."""
        genealogy = GenealogyTree()
        output = genealogy.format_for_synthesis()
        assert "Branch Genealogy:" in output

    def test_format_includes_leaf_markers(self):
        """Formatted output includes leaf status markers."""
        tree = SearchTree()
        tree.root_id = "root"
        tree.contexts["root"] = make_context("root")
        tree.contexts["completed"] = make_context(
            "completed",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(edit="test edit"),
            output="output",
            termination_reason=TerminationReason.COMPLETED,
        )
        tree.contexts["stuck"] = make_context(
            "stuck",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(edit="bad path"),
            termination_reason=TerminationReason.STUCK,
        )
        tree.leaves = ["completed"]

        genealogy = GenealogyTree.from_search_tree(tree)
        output = genealogy.format_for_synthesis()

        assert "✓ LEAF" in output
        assert "✗ STUCK" in output

    def test_format_includes_weights(self):
        """Formatted output includes weights when requested."""
        tree = SearchTree()
        tree.root_id = "root"
        tree.contexts["root"] = make_context("root")
        tree.contexts["leaf"] = make_context(
            "leaf",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(confidence=0.8),
            output="output",
            termination_reason=TerminationReason.COMPLETED,
        )
        tree.leaves = ["leaf"]

        genealogy = GenealogyTree.from_search_tree(tree)
        output = genealogy.format_for_synthesis(include_weights=True)

        assert "weight:" in output

    def test_format_without_weights(self):
        """Formatted output excludes weights when not requested."""
        tree = SearchTree()
        tree.root_id = "root"
        tree.contexts["root"] = make_context("root")
        tree.contexts["leaf"] = make_context(
            "leaf",
            parent_id="root",
            spawn_metadata=make_spawn_metadata(),
            output="output",
            termination_reason=TerminationReason.COMPLETED,
        )
        tree.leaves = ["leaf"]

        genealogy = GenealogyTree.from_search_tree(tree)
        output = genealogy.format_for_synthesis(include_weights=False)

        assert "weight:" not in output


class TestBranchNode:
    """Tests for BranchNode dataclass."""

    def test_default_values(self):
        """BranchNode has sensible defaults."""
        node = BranchNode(id="test", parent_id=None)

        assert node.spawn_reason == SpawnReason.INITIAL
        assert node.spawn_edit == ""
        assert node.spawn_confidence == 0.5
        assert node.children_ids == []
        assert node.is_leaf is False
        assert node.termination_reason is None

    def test_with_spawn_data(self):
        """BranchNode can be created with spawn data."""
        node = BranchNode(
            id="test",
            parent_id="parent",
            spawn_reason=SpawnReason.GATE_BRANCH,
            spawn_edit="explore alternative",
            spawn_confidence=0.75,
        )

        assert node.spawn_reason == SpawnReason.GATE_BRANCH
        assert node.spawn_edit == "explore alternative"
        assert node.spawn_confidence == 0.75
