"""Diversity scoring for PSS leaf outputs using embeddings."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pss.types import Context

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pss.config import PSSConfig
    from pss.types import DiversityState


# Embedding model token limits
EMBEDDING_TOKEN_LIMITS = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}
DEFAULT_TOKEN_LIMIT = 8191


# Diversity interpretation thresholds (calibrate from real runs)
DIVERSITY_THRESHOLDS = {
    "low": 0.3,  # 0.0-0.3: variations on a theme
    "medium": 0.6,  # 0.3-0.6: different approaches
    # 0.6-1.0: genuinely different directions (high)
}


@dataclass
class DiversityResult:
    """Result of diversity analysis."""

    score: float  # 0-1, higher = more diverse
    interpretation: str  # "low", "medium", "high"
    min_distance: float
    max_distance: float
    std_dev: float
    leaf_pairs: list[tuple[str, str, float]]  # (id1, id2, distance), sorted ascending
    pairwise_matrix: list[list[float]] | None = None
    embeddings: dict[str, list[float]] = field(
        default_factory=dict
    )  # cached for re-analysis
    embedding_cost: float = 0.0  # USD spent on embeddings


def interpret_diversity(score: float) -> str:
    """Convert diversity score to human-readable interpretation."""
    if score < DIVERSITY_THRESHOLDS["low"]:
        return "low"
    elif score < DIVERSITY_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "high"


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    Returns 0 for identical vectors, 1 for orthogonal, 2 for opposite.
    We use 1 - cosine_similarity, clamped to [0, 1].
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 1.0  # Treat zero vectors as maximally distant

    similarity = dot / (norm1 * norm2)
    # Clamp to handle floating point errors
    similarity = max(-1.0, min(1.0, similarity))
    distance = 1.0 - similarity
    # Clamp distance to [0, 1] range
    return max(0.0, min(1.0, distance))


def get_leaf_text(leaf: Context) -> str:
    """Extract the text content from a leaf for embedding."""
    if leaf.output:
        return leaf.output

    # Fall back to last assistant message
    for msg in reversed(leaf.messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]

    return ""


def truncate_for_embedding(text: str, model: str, chars_per_token: float = 4.0) -> str:
    """
    Truncate text to fit within embedding model's token limit.

    Uses a conservative chars-per-token estimate. Better to under-count
    than hit the API limit.
    """
    token_limit = EMBEDDING_TOKEN_LIMITS.get(model, DEFAULT_TOKEN_LIMIT)
    # Leave some headroom
    safe_limit = int(token_limit * 0.95)
    char_limit = int(safe_limit * chars_per_token)

    if len(text) <= char_limit:
        return text

    return text[:char_limit]


def get_embeddings_openai(
    texts: list[str],
    model: str = "text-embedding-3-small",
) -> tuple[list[list[float]], float]:
    """
    Get embeddings from OpenAI API.

    Returns (embeddings, cost_usd).
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Set it in .env or environment, "
            "or use --no-diversity to skip diversity analysis."
        )

    client = OpenAI(api_key=api_key)

    # Truncate texts to fit model limits
    truncated = [truncate_for_embedding(t, model) for t in texts]

    # Filter out empty texts
    non_empty_indices = [i for i, t in enumerate(truncated) if t.strip()]
    non_empty_texts = [truncated[i] for i in non_empty_indices]

    if not non_empty_texts:
        # All texts are empty - return zero vectors
        dim = 1536 if "small" in model else 3072  # OpenAI embedding dimensions
        return [[0.0] * dim for _ in texts], 0.0

    response = client.embeddings.create(
        model=model,
        input=non_empty_texts,
    )

    # Calculate cost (text-embedding-3-small: $0.02/1M tokens)
    total_tokens = response.usage.total_tokens
    if "small" in model:
        cost = (total_tokens / 1_000_000) * 0.02
    elif "large" in model:
        cost = (total_tokens / 1_000_000) * 0.13
    else:
        cost = (total_tokens / 1_000_000) * 0.02  # Default to small pricing

    # Build result list with zero vectors for empty texts
    dim = len(response.data[0].embedding)
    result = [[0.0] * dim for _ in texts]
    for idx, emb_data in zip(non_empty_indices, response.data):
        result[idx] = emb_data.embedding

    return result, cost


def compute_diversity(
    leaves: list[Context],
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    include_matrix: bool = False,
) -> DiversityResult | None:
    """
    Compute diversity score for leaf outputs.

    Args:
        leaves: List of leaf contexts to analyze
        provider: Embedding provider ("openai" or "none")
        model: Embedding model name
        include_matrix: Whether to include full pairwise distance matrix

    Returns:
        DiversityResult with scores. Returns None if diversity cannot be computed.
    """
    if len(leaves) < 2:
        # Can't compute diversity with fewer than 2 items
        return None

    if provider == "none":
        return None

    # Extract text from each leaf
    texts = [get_leaf_text(leaf) for leaf in leaves]
    leaf_ids = [leaf.id for leaf in leaves]

    # Get embeddings
    if provider == "openai":
        embeddings, cost = get_embeddings_openai(texts, model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    # Convert to numpy arrays
    embedding_arrays = [np.array(e) for e in embeddings]

    # Compute pairwise distances
    n = len(leaves)
    distances = []
    leaf_pairs = []
    matrix = [[0.0] * n for _ in range(n)] if include_matrix else None

    for i in range(n):
        for j in range(i + 1, n):
            dist = cosine_distance(embedding_arrays[i], embedding_arrays[j])
            distances.append(dist)
            leaf_pairs.append((leaf_ids[i], leaf_ids[j], dist))

            if matrix is not None:
                matrix[i][j] = dist
                matrix[j][i] = dist

    # Sort pairs by distance ascending (most similar first)
    leaf_pairs.sort(key=lambda x: x[2])

    # Compute statistics
    if distances:
        score = float(np.mean(distances))
        min_dist = float(np.min(distances))
        max_dist = float(np.max(distances))
        std = float(np.std(distances))
    else:
        score = 0.0
        min_dist = 0.0
        max_dist = 0.0
        std = 0.0

    # Cache embeddings for potential re-analysis
    cached_embeddings = {leaf_id: emb for leaf_id, emb in zip(leaf_ids, embeddings)}

    return DiversityResult(
        score=score,
        interpretation=interpret_diversity(score),
        min_distance=min_dist,
        max_distance=max_dist,
        std_dev=std,
        leaf_pairs=leaf_pairs,
        pairwise_matrix=matrix,
        embeddings=cached_embeddings,
        embedding_cost=cost,
    )


def compute_diversity_from_texts(
    texts: list[str],
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    include_matrix: bool = False,
) -> DiversityResult | None:
    """
    Compute diversity score for a list of text strings.

    This is a convenience function for baseline comparison where
    we have raw text outputs instead of Context objects.

    Args:
        texts: List of text strings to analyze
        provider: Embedding provider ("openai" or "none")
        model: Embedding model name
        include_matrix: Whether to include full pairwise distance matrix

    Returns:
        DiversityResult with scores. Returns None if diversity cannot be computed.
    """
    if len(texts) < 2:
        return None

    if provider == "none":
        return None

    # Generate IDs for the texts
    text_ids = [f"text_{i + 1}" for i in range(len(texts))]

    # Get embeddings
    if provider == "openai":
        embeddings, cost = get_embeddings_openai(texts, model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    # Convert to numpy arrays
    embedding_arrays = [np.array(e) for e in embeddings]

    # Compute pairwise distances
    n = len(texts)
    distances = []
    text_pairs = []
    matrix = [[0.0] * n for _ in range(n)] if include_matrix else None

    for i in range(n):
        for j in range(i + 1, n):
            dist = cosine_distance(embedding_arrays[i], embedding_arrays[j])
            distances.append(dist)
            text_pairs.append((text_ids[i], text_ids[j], dist))

            if matrix is not None:
                matrix[i][j] = dist
                matrix[j][i] = dist

    # Sort pairs by distance ascending (most similar first)
    text_pairs.sort(key=lambda x: x[2])

    # Compute statistics
    if distances:
        score = float(np.mean(distances))
        min_dist = float(np.min(distances))
        max_dist = float(np.max(distances))
        std = float(np.std(distances))
    else:
        score = 0.0
        min_dist = 0.0
        max_dist = 0.0
        std = 0.0

    # Cache embeddings
    cached_embeddings = {text_id: emb for text_id, emb in zip(text_ids, embeddings)}

    return DiversityResult(
        score=score,
        interpretation=interpret_diversity(score),
        min_distance=min_dist,
        max_distance=max_dist,
        std_dev=std,
        leaf_pairs=text_pairs,
        pairwise_matrix=matrix,
        embeddings=cached_embeddings,
        embedding_cost=cost,
    )


def format_diversity_report(result: DiversityResult, leaves: list[Context]) -> str:
    """Format diversity result as a human-readable report."""
    lines = []

    # Interpretation labels
    interp_labels = {
        "low": "low (variations on a theme)",
        "medium": "medium (different approaches)",
        "high": "high (genuinely different directions)",
    }

    lines.append(
        f"Diversity score: {result.score:.2f} - {interp_labels[result.interpretation]}"
    )
    lines.append(f"  Min pairwise distance: {result.min_distance:.2f}")
    lines.append(f"  Max pairwise distance: {result.max_distance:.2f}")
    lines.append(f"  Std dev: {result.std_dev:.2f}")
    lines.append(f"  Embedding cost: ${result.embedding_cost:.6f}")

    # Show most similar pair (potential duplicates)
    if result.leaf_pairs:
        most_similar = result.leaf_pairs[0]
        lines.append("")
        lines.append(f"Most similar pair (distance {most_similar[2]:.2f}):")
        lines.append(f"  {most_similar[0]}")
        lines.append(f"  {most_similar[1]}")

    # Show pairwise matrix if included and small enough
    if result.pairwise_matrix and len(leaves) <= 6:
        lines.append("")
        lines.append("Pairwise distances:")

        # Build short IDs for header
        short_ids = []
        for leaf in leaves:
            parts = leaf.id.split("->")
            short_id = parts[-1][:6] if len(parts) > 1 else leaf.id[:6]
            short_ids.append(short_id)

        # Header row
        header = "       " + "  ".join(f"{sid:>6}" for sid in short_ids)
        lines.append(header)

        # Data rows
        for i, (sid, row) in enumerate(zip(short_ids, result.pairwise_matrix)):
            row_str = f"{sid:>6} "
            for j, val in enumerate(row):
                if i == j:
                    row_str += "    -- "
                else:
                    row_str += f" {val:>5.2f} "
            lines.append(row_str)

    return "\n".join(lines)


# v0.5: Diversity-aware spawning functions


def update_diversity_incremental(
    new_leaf: Context,
    diversity_state: "DiversityState",
    config: "PSSConfig",
) -> bool:
    """
    Incrementally update diversity with a new leaf.

    Uses cached embeddings to avoid re-embedding all leaves.
    Returns True if update successful, False if cost cap hit or error.
    """
    # Check cost cap
    if diversity_state.total_embedding_cost >= config.diversity_max_embedding_cost:
        return False

    # Get new leaf's text
    new_text = get_leaf_text(new_leaf)
    if not new_text:
        return False

    # Get embedding for the new leaf
    try:
        embeddings, cost = get_embeddings_openai([new_text], config.diversity_model)
    except Exception as e:
        logger.warning(f"[PSS] Embedding API failed for incremental diversity: {type(e).__name__}: {e}")
        return False

    diversity_state.total_embedding_cost += cost

    if not embeddings or not embeddings[0]:
        return False

    # Cache the new embedding
    diversity_state.cached_embeddings[new_leaf.id] = embeddings[0]

    # Compute all pairwise distances with cached embeddings
    leaf_ids = list(diversity_state.cached_embeddings.keys())
    if len(leaf_ids) < 2:
        diversity_state.num_leaves = len(leaf_ids)
        diversity_state.last_computed_at = new_leaf.id
        return True

    all_distances = []
    for i in range(len(leaf_ids)):
        for j in range(i + 1, len(leaf_ids)):
            emb_i = np.array(diversity_state.cached_embeddings[leaf_ids[i]])
            emb_j = np.array(diversity_state.cached_embeddings[leaf_ids[j]])
            dist = cosine_distance(emb_i, emb_j)
            all_distances.append(dist)

    # Update state
    diversity_state.current_score = float(np.mean(all_distances))
    diversity_state.interpretation = interpret_diversity(diversity_state.current_score)
    diversity_state.num_leaves = len(leaf_ids)
    diversity_state.last_computed_at = new_leaf.id

    return True


def get_path_text(leaf: Context, tree_contexts: dict[str, "Context"] | None = None) -> str:
    """Extract the exploration trajectory (not output) from a leaf.

    Concatenates branch reasons from root to leaf, plus gate decisions,
    to capture the *path taken* rather than the *result produced*.
    """
    parts = []

    # Walk up the tree via parent_id to collect branch reasons
    if tree_contexts:
        chain = []
        ctx = leaf
        while ctx:
            if ctx.branch_reason:
                chain.append(ctx.branch_reason)
            parent = tree_contexts.get(ctx.parent_id or "")
            ctx = parent
        parts.extend(reversed(chain))
    elif leaf.branch_reason:
        parts.append(leaf.branch_reason)

    # Add gate decision summaries from messages
    for msg in leaf.messages:
        content = str(msg.get("content", ""))
        if "---CHECKPOINT---" in content:
            parts.append(content[:300])
        elif "---BRANCH---" in content:
            parts.append(content[:300])

    return "\n".join(parts) if parts else get_leaf_text(leaf)


@dataclass
class PathDiversityResult:
    """Combined output + path diversity analysis."""

    output_diversity: DiversityResult | None
    path_diversity: DiversityResult | None
    convergence_signal: float  # 0-1: high = different paths, similar outputs (convergence)
    noise_signal: float  # 0-1: high = similar paths, different outputs (noise)


def compute_path_diversity(
    leaves: list[Context],
    tree_contexts: dict[str, "Context"] | None = None,
    provider: str = "openai",
    model: str = "text-embedding-3-small",
) -> PathDiversityResult | None:
    """Compute both output diversity and path diversity.

    Two branches with different paths but similar outputs = convergence (high confidence).
    Two branches with similar paths but different outputs = noise (low confidence).
    """
    if len(leaves) < 2 or provider == "none":
        return None

    # Compute output diversity (existing)
    output_div = compute_diversity(leaves, provider, model)

    # Compute path diversity
    path_texts = [get_path_text(leaf, tree_contexts) for leaf in leaves]
    path_ids = [leaf.id for leaf in leaves]

    try:
        path_embeddings, cost = get_embeddings_openai(path_texts, model)
    except Exception as e:
        logger.warning(f"[PSS] Path diversity embedding failed: {e}")
        return PathDiversityResult(
            output_diversity=output_div,
            path_diversity=None,
            convergence_signal=0.0,
            noise_signal=0.0,
        )

    # Compute pairwise path distances
    path_arrays = [np.array(e) for e in path_embeddings]
    path_distances = []
    path_pairs = []
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            dist = cosine_distance(path_arrays[i], path_arrays[j])
            path_distances.append(dist)
            path_pairs.append((path_ids[i], path_ids[j], dist))

    path_pairs.sort(key=lambda x: x[2])

    if path_distances:
        path_score = float(np.mean(path_distances))
        path_min = float(np.min(path_distances))
        path_max = float(np.max(path_distances))
        path_std = float(np.std(path_distances))
    else:
        path_score = path_min = path_max = path_std = 0.0

    path_div = DiversityResult(
        score=path_score,
        interpretation=interpret_diversity(path_score),
        min_distance=path_min,
        max_distance=path_max,
        std_dev=path_std,
        leaf_pairs=path_pairs,
        embedding_cost=cost,
    )

    # Compute convergence and noise signals
    output_score = output_div.score if output_div else 0.5
    convergence_signal = max(0.0, path_score - output_score)  # high path div, low output div
    noise_signal = max(0.0, output_score - path_score)  # low path div, high output div

    return PathDiversityResult(
        output_diversity=output_div,
        path_diversity=path_div,
        convergence_signal=convergence_signal,
        noise_signal=noise_signal,
    )


def create_diversity_guidance(
    diversity_state: "DiversityState",
    config: "PSSConfig",
) -> str:
    """
    Generate guidance text based on current diversity state.

    This is appended to gate prompts (after the first gate) to help
    the model make better branching decisions.
    """
    if diversity_state.num_leaves < 2:
        return ""

    score = diversity_state.current_score
    interp = diversity_state.interpretation

    if score < config.diversity_min_threshold:
        return f"""
**Diversity Alert**: Current diversity is {interp} ({score:.2f}).
Existing branches are too similar. If branching, propose **genuinely different**
directions, not variations. Branching is encouraged.
"""
    elif score >= config.diversity_target:
        return f"""
**Diversity Note**: Current diversity is {interp} ({score:.2f}).
Good coverage achieved. Prefer continuing or terminating over new branches.
"""
    else:
        return f"""
**Diversity Note**: Current diversity is {interp} ({score:.2f}).
"""
