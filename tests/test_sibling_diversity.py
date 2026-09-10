"""Tests for sibling diversity enforcement (v1.4)."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from pss.sibling_diversity import (
    SiblingDiversityResult,
    compute_edit_similarity,
    check_sibling_diversity,
    format_diversity_rejection_log,
    format_retry_guidance,
)


class TestSiblingDiversityResult:
    """Tests for the SiblingDiversityResult dataclass."""

    def test_result_structure(self):
        """Test that result has correct fields."""
        result = SiblingDiversityResult(
            accepted_edits=["edit1", "edit2"],
            rejected_edits=["edit3"],
            rejection_reasons={"edit3": "85% similar to 'edit1'"},
            pairwise_distances=[("edit1", "edit2", 0.4)],
            min_distance=0.4,
            embedding_cost=0.001,
        )

        assert result.accepted_edits == ["edit1", "edit2"]
        assert result.rejected_edits == ["edit3"]
        assert "edit3" in result.rejection_reasons
        assert result.min_distance == 0.4
        assert result.embedding_cost == 0.001


class TestComputeEditSimilarity:
    """Tests for compute_edit_similarity function."""

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_identical_texts_high_similarity(self, mock_embeddings):
        """Identical texts should have high similarity."""
        # Mock embeddings - identical vectors
        mock_embeddings.return_value = (
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            0.0001,
        )

        similarity, cost = compute_edit_similarity("try OAuth", "try OAuth")

        assert similarity == 1.0  # Identical = 1.0
        assert cost == 0.0001

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_orthogonal_texts_low_similarity(self, mock_embeddings):
        """Orthogonal texts should have low similarity."""
        # Mock embeddings - orthogonal vectors
        mock_embeddings.return_value = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            0.0001,
        )

        similarity, cost = compute_edit_similarity("try OAuth", "use API keys")

        assert similarity == 0.0  # Orthogonal = 0.0
        assert cost == 0.0001

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_similar_texts_partial_similarity(self, mock_embeddings):
        """Similar texts should have partial similarity."""
        # Mock embeddings - somewhat similar vectors
        # cos(45°) ≈ 0.707
        mock_embeddings.return_value = (
            [[1.0, 0.0], [0.707, 0.707]],
            0.0001,
        )

        similarity, cost = compute_edit_similarity("try OAuth", "implement OAuth flow")

        assert 0.5 < similarity < 0.8
        assert cost == 0.0001


class TestCheckSiblingDiversity:
    """Tests for check_sibling_diversity function."""

    def test_empty_edits(self):
        """Empty edit list should return empty result."""
        result = check_sibling_diversity([])

        assert result.accepted_edits == []
        assert result.rejected_edits == []
        assert result.min_distance == 1.0
        assert result.embedding_cost == 0.0

    def test_single_edit_no_existing(self):
        """Single edit with no existing siblings should always pass."""
        result = check_sibling_diversity(["try OAuth"])

        assert result.accepted_edits == ["try OAuth"]
        assert result.rejected_edits == []
        assert result.embedding_cost == 0.0

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_two_different_edits_both_accepted(self, mock_embeddings):
        """Two different edits should both be accepted."""
        # Mock embeddings - orthogonal vectors
        mock_embeddings.return_value = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            0.0002,
        )

        result = check_sibling_diversity(
            ["try OAuth", "use API keys"],
            similarity_threshold=0.7,
        )

        assert len(result.accepted_edits) == 2
        assert result.rejected_edits == []
        assert "try OAuth" in result.accepted_edits
        assert "use API keys" in result.accepted_edits

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_two_similar_edits_one_rejected(self, mock_embeddings):
        """Two similar edits should have one rejected."""
        # Mock embeddings - nearly identical vectors (cosine similarity = 0.99)
        mock_embeddings.return_value = (
            [[1.0, 0.0, 0.0], [0.99, 0.1, 0.0]],
            0.0002,
        )

        result = check_sibling_diversity(
            ["try OAuth", "implement OAuth flow"],
            similarity_threshold=0.7,
        )

        # First edit accepted, second rejected
        assert len(result.accepted_edits) == 1
        assert len(result.rejected_edits) == 1
        assert result.accepted_edits[0] == "try OAuth"
        assert result.rejected_edits[0] == "implement OAuth flow"
        assert "implement OAuth flow" in result.rejection_reasons

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_three_edits_mixed_acceptance(self, mock_embeddings):
        """Three edits with mixed similarity."""
        # Mock: edit1 and edit2 similar, edit3 different
        mock_embeddings.return_value = (
            [[1.0, 0.0, 0.0], [0.95, 0.1, 0.0], [0.0, 1.0, 0.0]],
            0.0003,
        )

        result = check_sibling_diversity(
            ["try OAuth", "implement OAuth flow", "use API keys"],
            similarity_threshold=0.7,
        )

        assert len(result.accepted_edits) == 2
        assert len(result.rejected_edits) == 1
        assert "try OAuth" in result.accepted_edits
        assert "use API keys" in result.accepted_edits
        assert "implement OAuth flow" in result.rejected_edits

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_existing_siblings_considered(self, mock_embeddings):
        """Existing siblings should be checked against."""
        # Mock: new edit similar to existing sibling
        mock_embeddings.return_value = (
            [[0.95, 0.1, 0.0], [1.0, 0.0, 0.0]],
            0.0002,
        )

        result = check_sibling_diversity(
            ["implement OAuth flow"],
            similarity_threshold=0.7,
            existing_siblings=["try OAuth"],
        )

        # Should be rejected due to similarity to existing
        assert len(result.accepted_edits) == 0
        assert len(result.rejected_edits) == 1
        assert "implement OAuth flow" in result.rejected_edits

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_greedy_acceptance_order_matters(self, mock_embeddings):
        """Greedy acceptance processes in order - first similar edit wins."""
        # Mock: edit1 and edit2 are similar to each other
        mock_embeddings.return_value = (
            [[1.0, 0.0, 0.0], [0.98, 0.1, 0.0]],
            0.0002,
        )

        result = check_sibling_diversity(
            ["implement OAuth flow", "try OAuth"],  # Swapped order
            similarity_threshold=0.7,
        )

        # First one accepted, second rejected
        assert result.accepted_edits[0] == "implement OAuth flow"
        assert result.rejected_edits[0] == "try OAuth"

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_threshold_boundary(self, mock_embeddings):
        """Test behavior at threshold boundary."""
        # Mock: similarity just under threshold (0.69)
        # Vector with cos_similarity = 0.69 will pass threshold of 0.7
        mock_embeddings.return_value = (
            [[1.0, 0.0], [0.69, 0.724]],  # ~0.69 similarity (just under 0.7)
            0.0002,
        )

        result = check_sibling_diversity(
            ["edit1", "edit2"],
            similarity_threshold=0.7,
        )

        # Just under threshold should be accepted
        assert len(result.accepted_edits) == 2

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_pairwise_distances_populated(self, mock_embeddings):
        """Pairwise distances should be tracked."""
        mock_embeddings.return_value = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            0.0002,
        )

        result = check_sibling_diversity(
            ["edit1", "edit2"],
            similarity_threshold=0.7,
        )

        assert len(result.pairwise_distances) > 0
        # Check structure: (edit1, edit2, distance)
        assert len(result.pairwise_distances[0]) == 3


class TestFormatDiversityRejectionLog:
    """Tests for format_diversity_rejection_log function."""

    def test_short_edit_not_truncated(self):
        """Short edit strings should not be truncated."""
        log = format_diversity_rejection_log("try OAuth", "85% similar to 'edit1'")

        assert "try OAuth" in log
        assert "85% similar" in log
        assert "[PSS]" in log

    def test_long_edit_truncated(self):
        """Long edit strings should be truncated."""
        long_edit = "This is a very long edit string that exceeds the fifty character limit for display"
        log = format_diversity_rejection_log(long_edit, "too similar")

        assert "..." in log
        assert len(log) < len(long_edit) + 100  # Should be truncated


class TestFormatRetryGuidance:
    """Tests for format_retry_guidance function."""

    def test_empty_rejected_returns_empty(self):
        """Empty rejected list should return empty string."""
        guidance = format_retry_guidance([], {})
        assert guidance == ""

    def test_rejected_edits_listed(self):
        """Rejected edits should be listed in guidance."""
        guidance = format_retry_guidance(
            ["implement OAuth flow"],
            {"implement OAuth flow": "85% similar to 'try OAuth'"},
        )

        assert "implement OAuth" in guidance
        assert "85% similar" in guidance
        assert "genuinely different" in guidance

    def test_multiple_rejected_edits(self):
        """Multiple rejected edits should all be listed."""
        guidance = format_retry_guidance(
            ["edit1", "edit2"],
            {"edit1": "too similar", "edit2": "too similar"},
        )

        assert "edit1" in guidance
        assert "edit2" in guidance


class TestConfigIntegration:
    """Tests for config options integration."""

    def test_config_has_sibling_diversity_options(self):
        """Config should have sibling diversity options."""
        from pss.config import PSSConfig

        config = PSSConfig()

        assert hasattr(config, "sibling_diversity_enabled")
        assert hasattr(config, "sibling_similarity_threshold")
        assert hasattr(config, "sibling_diversity_include_existing")
        assert hasattr(config, "sibling_diversity_min_edits")

    def test_config_default_values(self):
        """Check default values are reasonable."""
        from pss.config import PSSConfig

        config = PSSConfig()

        assert config.sibling_diversity_enabled is False  # Off by default
        assert 0.5 <= config.sibling_similarity_threshold <= 0.9
        assert config.sibling_diversity_include_existing is True
        assert config.sibling_diversity_min_edits >= 2

    def test_config_yaml_loading(self):
        """Config should load from YAML correctly."""
        from pss.config import load_config
        import tempfile
        import os

        yaml_content = """
sibling_diversity:
  enabled: true
  similarity_threshold: 0.8
  include_existing: false
  min_edits: 3
"""
        # Create temp file, close it, then use path
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        temp_path = f.name
        f.write(yaml_content)
        f.close()

        try:
            config = load_config(temp_path)

            assert config.sibling_diversity_enabled is True
            assert config.sibling_similarity_threshold == 0.8
            assert config.sibling_diversity_include_existing is False
            assert config.sibling_diversity_min_edits == 3
        finally:
            os.unlink(temp_path)


class TestHarnessIntegration:
    """Tests for harness integration."""

    def test_get_existing_sibling_edits(self):
        """Test helper function for getting existing siblings."""
        from pss.harness import _get_existing_sibling_edits
        from pss.types import SearchTree, Context

        tree = SearchTree()

        # Parent context
        parent = Context(id="root", parent_id=None, messages=[])
        tree.contexts["root"] = parent

        # Two running siblings
        sibling1 = Context(
            id="root->a",
            parent_id="root",
            messages=[],
            branch_reason="try OAuth",
            status="running",
        )
        sibling2 = Context(
            id="root->b",
            parent_id="root",
            messages=[],
            branch_reason="use API keys",
            status="running",
        )
        # One terminated sibling (shouldn't be included)
        sibling3 = Context(
            id="root->c",
            parent_id="root",
            messages=[],
            branch_reason="old approach",
            status="terminated",
        )
        tree.contexts["root->a"] = sibling1
        tree.contexts["root->b"] = sibling2
        tree.contexts["root->c"] = sibling3

        edits = _get_existing_sibling_edits(tree, "root")

        assert len(edits) == 2
        assert "try OAuth" in edits
        assert "use API keys" in edits
        assert "old approach" not in edits

    def test_get_existing_sibling_edits_different_parent(self):
        """Only siblings with same parent should be returned."""
        from pss.harness import _get_existing_sibling_edits
        from pss.types import SearchTree, Context

        tree = SearchTree()

        # Two different parents
        parent1 = Context(id="root", parent_id=None, messages=[])
        parent2 = Context(id="other", parent_id=None, messages=[])
        tree.contexts["root"] = parent1
        tree.contexts["other"] = parent2

        # Child of root
        child1 = Context(
            id="root->a",
            parent_id="root",
            messages=[],
            branch_reason="root child",
            status="running",
        )
        # Child of other
        child2 = Context(
            id="other->a",
            parent_id="other",
            messages=[],
            branch_reason="other child",
            status="running",
        )
        tree.contexts["root->a"] = child1
        tree.contexts["other->a"] = child2

        edits = _get_existing_sibling_edits(tree, "root")

        assert len(edits) == 1
        assert "root child" in edits
        assert "other child" not in edits


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_handles_embedding_error(self, mock_embeddings):
        """Should handle embedding API errors gracefully."""
        mock_embeddings.side_effect = Exception("API error")

        # Should raise since we're calling directly
        with pytest.raises(Exception):
            check_sibling_diversity(["edit1", "edit2"])

    def test_single_edit_with_existing_siblings(self):
        """Single edit with existing siblings should still check."""
        # Even with 1 proposed edit, if there are existing siblings,
        # we need to check (but the function will embed and compare)
        # This test verifies the logic path exists
        pass  # Covered by test_existing_siblings_considered

    @patch("pss.sibling_diversity.get_embeddings_openai")
    def test_all_edits_rejected(self, mock_embeddings):
        """All edits being rejected should work correctly."""
        # All proposed edits similar to existing
        mock_embeddings.return_value = (
            [[0.99, 0.1, 0.0], [0.98, 0.1, 0.0], [1.0, 0.0, 0.0]],
            0.0003,
        )

        result = check_sibling_diversity(
            ["edit1", "edit2"],
            similarity_threshold=0.7,
            existing_siblings=["existing"],
        )

        # All rejected due to similarity to existing
        assert len(result.rejected_edits) == 2
        assert len(result.accepted_edits) == 0
