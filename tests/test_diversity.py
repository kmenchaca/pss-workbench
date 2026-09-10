"""Tests for diversity scoring."""

import numpy as np
from unittest.mock import patch

from pss.diversity import (
    cosine_distance,
    compute_diversity,
    compute_diversity_from_texts,
    interpret_diversity,
    get_leaf_text,
    truncate_for_embedding,
    format_diversity_report,
    DiversityResult,
    update_diversity_incremental,
    create_diversity_guidance,
)
from pss.types import Context, DiversityState
from pss.config import PSSConfig


def make_leaf(id: str, output: str) -> Context:
    """Create a minimal leaf context for testing."""
    return Context(
        id=id,
        parent_id=None,
        output=output,
    )


class TestCosineDistance:
    """Tests for cosine distance calculation."""

    def test_identical_vectors_zero_distance(self):
        """Identical vectors should have distance 0."""
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(v, v) == 0.0

    def test_orthogonal_vectors_distance_one(self):
        """Orthogonal vectors should have distance 1."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(cosine_distance(v1, v2) - 1.0) < 0.0001

    def test_opposite_vectors_max_distance(self):
        """Opposite vectors should have distance close to 1 (clamped)."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        # Distance is clamped to [0, 1], so opposite = 1.0
        assert cosine_distance(v1, v2) == 1.0

    def test_similar_vectors_low_distance(self):
        """Similar vectors should have low distance."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.1, 2.1, 3.1])
        dist = cosine_distance(v1, v2)
        assert dist < 0.01  # Very similar

    def test_zero_vector_returns_one(self):
        """Zero vector should return distance 1."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 2.0])
        assert cosine_distance(v1, v2) == 1.0


class TestInterpretDiversity:
    """Tests for diversity score interpretation."""

    def test_low_diversity(self):
        """Scores below 0.3 should be 'low'."""
        assert interpret_diversity(0.0) == "low"
        assert interpret_diversity(0.15) == "low"
        assert interpret_diversity(0.29) == "low"

    def test_medium_diversity(self):
        """Scores 0.3-0.6 should be 'medium'."""
        assert interpret_diversity(0.3) == "medium"
        assert interpret_diversity(0.45) == "medium"
        assert interpret_diversity(0.59) == "medium"

    def test_high_diversity(self):
        """Scores 0.6+ should be 'high'."""
        assert interpret_diversity(0.6) == "high"
        assert interpret_diversity(0.75) == "high"
        assert interpret_diversity(1.0) == "high"


class TestGetLeafText:
    """Tests for extracting text from leaf contexts."""

    def test_uses_output_if_present(self):
        """Should use leaf.output if available."""
        leaf = make_leaf("test", "This is the output")
        assert get_leaf_text(leaf) == "This is the output"

    def test_falls_back_to_last_assistant_message(self):
        """Should fall back to last assistant message if no output."""
        leaf = Context(
            id="test",
            parent_id=None,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Continue"},
                {"role": "assistant", "content": "Final response"},
            ],
        )
        assert get_leaf_text(leaf) == "Final response"

    def test_returns_empty_if_no_content(self):
        """Should return empty string if no output or messages."""
        leaf = Context(id="test", parent_id=None, messages=[])
        assert get_leaf_text(leaf) == ""


class TestTruncateForEmbedding:
    """Tests for text truncation."""

    def test_short_text_unchanged(self):
        """Short text should not be truncated."""
        text = "Short text"
        result = truncate_for_embedding(text, "text-embedding-3-small")
        assert result == text

    def test_long_text_truncated(self):
        """Long text should be truncated."""
        text = "x" * 100000  # Very long
        result = truncate_for_embedding(text, "text-embedding-3-small")
        assert len(result) < len(text)
        assert len(result) < 35000  # Should be under limit


class TestComputeDiversity:
    """Tests for the main diversity computation."""

    def test_single_leaf_returns_none(self):
        """Cannot compute diversity with only one leaf."""
        leaves = [make_leaf("root", "Only output")]
        result = compute_diversity(leaves, provider="none")
        assert result is None

    def test_provider_none_returns_none(self):
        """Provider 'none' should skip computation."""
        leaves = [
            make_leaf("leaf1", "Output 1"),
            make_leaf("leaf2", "Output 2"),
        ]
        result = compute_diversity(leaves, provider="none")
        assert result is None

    @patch("pss.diversity.get_embeddings_openai")
    def test_identical_outputs_low_diversity(self, mock_embeddings):
        """Identical outputs should have diversity ~0."""
        # Return identical embeddings
        mock_embeddings.return_value = (
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            0.0001,
        )

        leaves = [
            make_leaf("leaf1", "Same text"),
            make_leaf("leaf2", "Same text"),
        ]
        result = compute_diversity(leaves, provider="openai")

        assert result is not None
        assert result.score == 0.0
        assert result.interpretation == "low"

    @patch("pss.diversity.get_embeddings_openai")
    def test_orthogonal_outputs_high_diversity(self, mock_embeddings):
        """Orthogonal embeddings should have high diversity."""
        # Return orthogonal embeddings
        mock_embeddings.return_value = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            0.0001,
        )

        leaves = [
            make_leaf("leaf1", "Topic A"),
            make_leaf("leaf2", "Topic B"),
        ]
        result = compute_diversity(leaves, provider="openai")

        assert result is not None
        assert result.score == 1.0
        assert result.interpretation == "high"

    @patch("pss.diversity.get_embeddings_openai")
    def test_leaf_pairs_sorted_ascending(self, mock_embeddings):
        """Leaf pairs should be sorted by distance ascending."""
        # Return embeddings with known distances
        mock_embeddings.return_value = (
            [
                [1.0, 0.0, 0.0],  # leaf1
                [0.9, 0.1, 0.0],  # leaf2 - similar to leaf1
                [0.0, 1.0, 0.0],  # leaf3 - orthogonal to leaf1
            ],
            0.0001,
        )

        leaves = [
            make_leaf("leaf1", "A"),
            make_leaf("leaf2", "B"),
            make_leaf("leaf3", "C"),
        ]
        result = compute_diversity(leaves, provider="openai")

        assert result is not None
        # First pair should be the most similar (leaf1, leaf2)
        assert result.leaf_pairs[0][0] == "leaf1"
        assert result.leaf_pairs[0][1] == "leaf2"
        # Distance should be increasing
        distances = [p[2] for p in result.leaf_pairs]
        assert distances == sorted(distances)

    @patch("pss.diversity.get_embeddings_openai")
    def test_includes_pairwise_matrix_when_requested(self, mock_embeddings):
        """Should include matrix when include_matrix=True."""
        mock_embeddings.return_value = (
            [[1.0, 0.0], [0.0, 1.0]],
            0.0001,
        )

        leaves = [make_leaf("a", "A"), make_leaf("b", "B")]

        # Without matrix
        result1 = compute_diversity(leaves, provider="openai", include_matrix=False)
        assert result1.pairwise_matrix is None

        # With matrix
        result2 = compute_diversity(leaves, provider="openai", include_matrix=True)
        assert result2.pairwise_matrix is not None
        assert len(result2.pairwise_matrix) == 2

    @patch("pss.diversity.get_embeddings_openai")
    def test_caches_embeddings(self, mock_embeddings):
        """Should cache embeddings in result for re-analysis."""
        mock_embeddings.return_value = (
            [[1.0, 2.0], [3.0, 4.0]],
            0.0001,
        )

        leaves = [make_leaf("a", "A"), make_leaf("b", "B")]
        result = compute_diversity(leaves, provider="openai")

        assert "a" in result.embeddings
        assert "b" in result.embeddings
        assert result.embeddings["a"] == [1.0, 2.0]


class TestComputeDiversityFromTexts:
    """Tests for diversity computation from raw text strings."""

    def test_single_text_returns_none(self):
        """Cannot compute diversity with only one text."""
        result = compute_diversity_from_texts(["Only one"], provider="none")
        assert result is None

    def test_provider_none_returns_none(self):
        """Provider 'none' should skip computation."""
        result = compute_diversity_from_texts(["A", "B"], provider="none")
        assert result is None

    @patch("pss.diversity.get_embeddings_openai")
    def test_computes_diversity_for_text_list(self, mock_embeddings):
        """Should compute diversity for a list of text strings."""
        mock_embeddings.return_value = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            0.0001,
        )

        texts = ["First output", "Second output", "Third output"]
        result = compute_diversity_from_texts(texts, provider="openai")

        assert result is not None
        assert result.score == 1.0  # All orthogonal
        assert result.interpretation == "high"
        assert len(result.leaf_pairs) == 3  # 3 pairs for 3 texts

    @patch("pss.diversity.get_embeddings_openai")
    def test_generates_text_ids(self, mock_embeddings):
        """Should generate text_1, text_2, etc. IDs."""
        mock_embeddings.return_value = (
            [[1.0, 0.0], [0.0, 1.0]],
            0.0001,
        )

        texts = ["A", "B"]
        result = compute_diversity_from_texts(texts, provider="openai")

        assert "text_1" in result.embeddings
        assert "text_2" in result.embeddings


class TestFormatDiversityReport:
    """Tests for report formatting."""

    def test_basic_report_format(self):
        """Should format basic diversity result."""
        result = DiversityResult(
            score=0.65,
            interpretation="high",
            min_distance=0.45,
            max_distance=0.85,
            std_dev=0.12,
            leaf_pairs=[("leaf1", "leaf2", 0.45), ("leaf1", "leaf3", 0.85)],
            embedding_cost=0.00002,
        )
        leaves = [
            make_leaf("leaf1", "A"),
            make_leaf("leaf2", "B"),
            make_leaf("leaf3", "C"),
        ]

        report = format_diversity_report(result, leaves)

        assert "0.65" in report
        assert "high" in report
        assert "0.45" in report
        assert "Most similar pair" in report


# v0.5: Diversity-aware spawning tests


class TestDiversityState:
    """Tests for DiversityState dataclass."""

    def test_should_encourage_branching_below_threshold(self):
        """Should encourage branching when below threshold."""
        state = DiversityState(
            current_score=0.2,
            num_leaves=3,
        )
        assert state.should_encourage_branching(min_threshold=0.3) is True

    def test_should_not_encourage_branching_above_threshold(self):
        """Should not encourage branching when above threshold."""
        state = DiversityState(
            current_score=0.5,
            num_leaves=3,
        )
        assert state.should_encourage_branching(min_threshold=0.3) is False

    def test_should_not_encourage_branching_with_few_leaves(self):
        """Should not encourage branching with fewer than 2 leaves."""
        state = DiversityState(
            current_score=0.1,
            num_leaves=1,
        )
        assert state.should_encourage_branching(min_threshold=0.3) is False

    def test_is_target_reached_when_above(self):
        """Should report target reached when score >= target."""
        state = DiversityState(
            current_score=0.6,
            num_leaves=3,
        )
        assert state.is_target_reached(target=0.5) is True

    def test_is_target_not_reached_when_below(self):
        """Should not report target reached when score < target."""
        state = DiversityState(
            current_score=0.4,
            num_leaves=3,
        )
        assert state.is_target_reached(target=0.5) is False

    def test_is_target_not_reached_with_few_leaves(self):
        """Should not report target reached with fewer than 2 leaves."""
        state = DiversityState(
            current_score=0.8,
            num_leaves=1,
        )
        assert state.is_target_reached(target=0.5) is False


class TestUpdateDiversityIncremental:
    """Tests for incremental diversity updates."""

    @patch("pss.diversity.get_embeddings_openai")
    def test_updates_state_with_new_leaf(self, mock_embeddings):
        """Should update diversity state when new leaf is added."""
        # First call for first leaf, second for new leaf
        mock_embeddings.side_effect = [
            ([[1.0, 0.0, 0.0]], 0.0001),
            ([[0.0, 1.0, 0.0]], 0.0001),
        ]

        config = PSSConfig(
            diversity_aware_enabled=True,
            diversity_max_embedding_cost=0.01,
        )

        # Create state with one existing leaf
        state = DiversityState()
        leaf1 = make_leaf("leaf1", "First output")
        update_diversity_incremental(leaf1, state, config)

        # Add second leaf
        leaf2 = make_leaf("leaf2", "Second output")
        result = update_diversity_incremental(leaf2, state, config)

        assert result is True
        assert state.num_leaves == 2
        assert state.current_score == 1.0  # Orthogonal vectors
        assert state.interpretation == "high"
        assert "leaf1" in state.cached_embeddings
        assert "leaf2" in state.cached_embeddings

    @patch("pss.diversity.get_embeddings_openai")
    def test_respects_cost_cap(self, mock_embeddings):
        """Should return False when cost cap is exceeded."""
        config = PSSConfig(
            diversity_aware_enabled=True,
            diversity_max_embedding_cost=0.001,
        )

        state = DiversityState(
            total_embedding_cost=0.002,  # Already over cap
        )

        leaf = make_leaf("leaf1", "Output")
        result = update_diversity_incremental(leaf, state, config)

        assert result is False
        mock_embeddings.assert_not_called()

    def test_returns_false_for_empty_output(self):
        """Should return False when leaf has no output."""
        config = PSSConfig(diversity_aware_enabled=True)
        state = DiversityState()

        leaf = Context(id="empty", parent_id=None, messages=[])
        result = update_diversity_incremental(leaf, state, config)

        assert result is False

    @patch("pss.diversity.get_embeddings_openai")
    def test_handles_single_leaf(self, mock_embeddings):
        """Should handle first leaf correctly (no pairwise distances yet)."""
        mock_embeddings.return_value = ([[1.0, 2.0, 3.0]], 0.0001)

        config = PSSConfig(diversity_aware_enabled=True)
        state = DiversityState()

        leaf = make_leaf("leaf1", "First output")
        result = update_diversity_incremental(leaf, state, config)

        assert result is True
        assert state.num_leaves == 1
        assert "leaf1" in state.cached_embeddings
        # Score should still be 0 with only one leaf
        assert state.current_score == 0.0


class TestCreateDiversityGuidance:
    """Tests for diversity guidance generation."""

    def test_no_guidance_with_few_leaves(self):
        """Should return empty string with fewer than 2 leaves."""
        config = PSSConfig(
            diversity_aware_enabled=True,
            diversity_min_threshold=0.3,
            diversity_target=0.5,
        )
        state = DiversityState(num_leaves=1)

        guidance = create_diversity_guidance(state, config)
        assert guidance == ""

    def test_encourages_branching_when_below_threshold(self):
        """Should encourage branching when diversity is low."""
        config = PSSConfig(
            diversity_aware_enabled=True,
            diversity_min_threshold=0.3,
            diversity_target=0.5,
        )
        state = DiversityState(
            current_score=0.2,
            interpretation="low",
            num_leaves=3,
        )

        guidance = create_diversity_guidance(state, config)
        assert "Diversity Alert" in guidance
        assert "too similar" in guidance
        assert "genuinely different" in guidance

    def test_discourages_branching_when_target_reached(self):
        """Should discourage branching when target is reached."""
        config = PSSConfig(
            diversity_aware_enabled=True,
            diversity_min_threshold=0.3,
            diversity_target=0.5,
        )
        state = DiversityState(
            current_score=0.6,
            interpretation="high",
            num_leaves=3,
        )

        guidance = create_diversity_guidance(state, config)
        assert "Good coverage" in guidance
        assert "terminating" in guidance

    def test_neutral_guidance_in_middle_range(self):
        """Should provide neutral guidance in middle range."""
        config = PSSConfig(
            diversity_aware_enabled=True,
            diversity_min_threshold=0.3,
            diversity_target=0.5,
        )
        state = DiversityState(
            current_score=0.4,
            interpretation="medium",
            num_leaves=3,
        )

        guidance = create_diversity_guidance(state, config)
        assert "Diversity Note" in guidance
        assert "0.40" in guidance
        # Should not have the strong language
        assert "too similar" not in guidance
        assert "Good coverage" not in guidance
