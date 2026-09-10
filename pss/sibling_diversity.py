"""Sibling diversity enforcement for PSS v1.4.

Prevents similar branches from spawning by comparing edit strings
using embeddings. When a gate decision proposes multiple branches,
checks their edit strings for similarity and rejects those that
are too close to already-accepted edits or existing running siblings.

Key insight: "PSS doesn't just spawn branches—it enforces that siblings
are incompatible."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pss.diversity import cosine_distance, get_embeddings_openai


@dataclass
class SiblingDiversityResult:
    """Result of sibling diversity check."""

    accepted_edits: list[str]  # Edits that passed diversity check
    rejected_edits: list[str]  # Edits rejected for being too similar
    rejection_reasons: dict[str, str]  # edit -> reason for rejection
    pairwise_distances: list[tuple[str, str, float]]  # (edit1, edit2, distance)
    min_distance: float  # Minimum distance among accepted
    embedding_cost: float  # USD spent on embeddings


def compute_edit_similarity(
    edit1: str,
    edit2: str,
    embedding_model: str = "text-embedding-3-small",
) -> tuple[float, float]:
    """
    Compute similarity between two edit strings.

    Args:
        edit1: First edit string
        edit2: Second edit string
        embedding_model: Model for embeddings

    Returns:
        Tuple of (similarity, cost) where similarity is 0-1 (1 = identical)
    """
    embeddings, cost = get_embeddings_openai([edit1, edit2], embedding_model)

    emb1 = np.array(embeddings[0])
    emb2 = np.array(embeddings[1])

    # Cosine distance: 0 = identical, 1 = orthogonal
    distance = cosine_distance(emb1, emb2)

    # Convert to similarity (1 = identical, 0 = orthogonal)
    similarity = 1.0 - distance

    return similarity, cost


def check_sibling_diversity(
    proposed_edits: list[str],
    similarity_threshold: float = 0.7,
    existing_siblings: list[str] | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> SiblingDiversityResult:
    """
    Check diversity among proposed sibling branches.

    Uses greedy acceptance: processes edits in order, accepting each
    only if it's sufficiently different from all already-accepted edits
    and existing running siblings.

    Args:
        proposed_edits: List of edit strings from gate decision
        similarity_threshold: Reject if similarity > this (0-1)
        existing_siblings: Edit strings from already-running siblings
        embedding_model: Model for embeddings

    Returns:
        SiblingDiversityResult with accepted/rejected edits and reasons
    """
    if not proposed_edits:
        return SiblingDiversityResult(
            accepted_edits=[],
            rejected_edits=[],
            rejection_reasons={},
            pairwise_distances=[],
            min_distance=1.0,
            embedding_cost=0.0,
        )

    # Single edit always passes
    if len(proposed_edits) == 1 and not existing_siblings:
        return SiblingDiversityResult(
            accepted_edits=proposed_edits.copy(),
            rejected_edits=[],
            rejection_reasons={},
            pairwise_distances=[],
            min_distance=1.0,
            embedding_cost=0.0,
        )

    existing_siblings = existing_siblings or []

    # Get embeddings for all edits (proposed + existing) in one batch
    all_edits = proposed_edits + existing_siblings
    embeddings, cost = get_embeddings_openai(all_edits, embedding_model)

    # Split embeddings back
    proposed_embeddings = [np.array(e) for e in embeddings[: len(proposed_edits)]]
    existing_embeddings = [np.array(e) for e in embeddings[len(proposed_edits) :]]

    # Greedy acceptance algorithm
    accepted_edits: list[str] = []
    accepted_embeddings: list[np.ndarray] = []
    rejected_edits: list[str] = []
    rejection_reasons: dict[str, str] = {}
    pairwise_distances: list[tuple[str, str, float]] = []
    all_distances: list[float] = []

    # Start comparison set with existing siblings
    comparison_edits = existing_siblings.copy()
    comparison_embeddings = existing_embeddings.copy()

    for i, edit in enumerate(proposed_edits):
        emb = proposed_embeddings[i]
        too_similar = False
        similar_to: str | None = None
        max_similarity = 0.0

        # Check against all comparison edits
        for j, comp_emb in enumerate(comparison_embeddings):
            distance = cosine_distance(emb, comp_emb)
            similarity = 1.0 - distance
            comp_edit = comparison_edits[j]

            pairwise_distances.append((edit, comp_edit, distance))

            if similarity > max_similarity:
                max_similarity = similarity

            if similarity > similarity_threshold:
                too_similar = True
                similar_to = comp_edit
                break

        if too_similar:
            rejected_edits.append(edit)
            # Truncate similar_to for readability
            similar_display = similar_to[:40] + "..." if len(similar_to or "") > 40 else similar_to
            rejection_reasons[edit] = (
                f"{max_similarity:.0%} similar to '{similar_display}'"
            )
        else:
            accepted_edits.append(edit)
            accepted_embeddings.append(emb)
            comparison_edits.append(edit)
            comparison_embeddings.append(emb)

            # Track distances between accepted edits
            if len(accepted_embeddings) > 1:
                for prev_emb in accepted_embeddings[:-1]:
                    dist = cosine_distance(emb, prev_emb)
                    all_distances.append(dist)

    # Calculate min distance among accepted
    min_distance = min(all_distances) if all_distances else 1.0

    return SiblingDiversityResult(
        accepted_edits=accepted_edits,
        rejected_edits=rejected_edits,
        rejection_reasons=rejection_reasons,
        pairwise_distances=pairwise_distances,
        min_distance=min_distance,
        embedding_cost=cost,
    )


def format_diversity_rejection_log(
    edit: str,
    reason: str,
) -> str:
    """Format a rejection message for logging."""
    # Truncate edit for display
    edit_display = edit[:50] + "..." if len(edit) > 50 else edit
    return f"[PSS] Rejected similar sibling: {edit_display} ({reason})"


def format_retry_guidance(
    rejected_edits: list[str],
    rejection_reasons: dict[str, str],
) -> str:
    """
    Format guidance for retry with different edits.

    This can be used in future versions to prompt the LLM
    to propose genuinely different alternatives.
    """
    if not rejected_edits:
        return ""

    lines = [
        "The following proposed branches were rejected for being too similar:",
        "",
    ]

    for edit in rejected_edits:
        reason = rejection_reasons.get(edit, "too similar")
        lines.append(f"  - '{edit[:40]}...' ({reason})")

    lines.extend([
        "",
        "If branching again, propose directions that are genuinely different",
        "from accepted branches. Avoid rephrasing the same idea.",
    ])

    return "\n".join(lines)
