"""Entity resolution (deduplication) for the Emergent Knowledge Graph system.

This module provides functions for finding and merging duplicate entities,
resolving conflicts, and computing entity similarity.
"""

from __future__ import annotations

import uuid
from difflib import SequenceMatcher
from typing import Any, Callable

from .types import (
    ConflictType,
    Entity,
    KnowledgeGraph,
    MergeConflict,
    Relation,
    ResolutionStrategy,
)


# Type alias for embedding function
EmbeddingFn = Callable[[str], list[float]]


def string_similarity(s1: str, s2: str) -> float:
    """Compute string similarity using SequenceMatcher.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Similarity score from 0.0 to 1.0.
    """
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Cosine similarity from -1.0 to 1.0.
    """
    if len(v1) != len(v2) or not v1:
        return 0.0

    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def entity_similarity(
    e1: Entity,
    e2: Entity,
    embedding_fn: EmbeddingFn | None = None,
) -> float:
    """Compute similarity between two entities.

    Uses name similarity, type matching, and optionally embeddings.

    Args:
        e1: First entity.
        e2: Second entity.
        embedding_fn: Optional function to compute embeddings.

    Returns:
        Similarity score from 0.0 to 1.0.
    """
    scores = []

    # Name similarity (highest weight)
    name_sim = string_similarity(e1.name, e2.name)
    scores.append(("name", name_sim, 0.4))

    # Check aliases
    alias_sim = 0.0
    for alias1 in [e1.name] + e1.aliases:
        for alias2 in [e2.name] + e2.aliases:
            alias_sim = max(alias_sim, string_similarity(alias1, alias2))
    scores.append(("alias", alias_sim, 0.2))

    # Type similarity
    type_sim = 1.0 if e1.type == e2.type else 0.3
    scores.append(("type", type_sim, 0.2))

    # Property overlap
    if e1.properties and e2.properties:
        common_keys = set(e1.properties.keys()) & set(e2.properties.keys())
        if common_keys:
            matching = sum(
                1 for k in common_keys
                if e1.properties[k] == e2.properties[k]
            )
            prop_sim = matching / len(common_keys)
        else:
            prop_sim = 0.0
    else:
        prop_sim = 0.0
    scores.append(("properties", prop_sim, 0.1))

    # Embedding similarity (if available)
    if embedding_fn:
        try:
            emb1 = embedding_fn(e1.name)
            emb2 = embedding_fn(e2.name)
            emb_sim = cosine_similarity(emb1, emb2)
            scores.append(("embedding", emb_sim, 0.1))
        except Exception:
            pass

    # Weighted average
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(score * weight for _, score, weight in scores)

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def find_duplicates(
    graph: KnowledgeGraph,
    threshold: float = 0.8,
    embedding_fn: EmbeddingFn | None = None,
) -> list[tuple[Entity, Entity, float]]:
    """Find potential duplicate entities in the graph.

    Args:
        graph: The graph to search.
        threshold: Minimum similarity to consider as duplicate.
        embedding_fn: Optional function to compute embeddings.

    Returns:
        List of (entity1, entity2, similarity) tuples.
    """
    duplicates = []
    entities = list(graph.entities.values())

    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1:]:
            sim = entity_similarity(e1, e2, embedding_fn)
            if sim >= threshold:
                duplicates.append((e1, e2, sim))

    # Sort by similarity (highest first)
    duplicates.sort(key=lambda x: x[2], reverse=True)

    return duplicates


def find_duplicates_for_entity(
    entity: Entity,
    graph: KnowledgeGraph,
    threshold: float = 0.8,
    embedding_fn: EmbeddingFn | None = None,
) -> list[tuple[Entity, float]]:
    """Find potential duplicates for a specific entity.

    Args:
        entity: The entity to find duplicates for.
        graph: The graph to search.
        threshold: Minimum similarity to consider as duplicate.
        embedding_fn: Optional function to compute embeddings.

    Returns:
        List of (matching_entity, similarity) tuples.
    """
    matches = []

    for existing in graph.entities.values():
        if existing.id == entity.id:
            continue

        sim = entity_similarity(entity, existing, embedding_fn)
        if sim >= threshold:
            matches.append((existing, sim))

    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)

    return matches


def merge_entities(
    e1: Entity,
    e2: Entity,
    strategy: ResolutionStrategy = ResolutionStrategy.MERGE,
) -> tuple[Entity, MergeConflict | None]:
    """Merge two entities into one.

    Args:
        e1: First entity (preferred in case of conflicts).
        e2: Second entity.
        strategy: How to resolve conflicts.

    Returns:
        Tuple of (merged entity, optional conflict record).
    """
    conflict = None

    if strategy == ResolutionStrategy.KEEP_FIRST:
        return e1, None

    if strategy == ResolutionStrategy.KEEP_SECOND:
        return e2, None

    # Merge strategy
    merged = Entity(
        id=e1.id,  # Keep first entity's ID
        name=e1.name if e1.confidence >= e2.confidence else e2.name,
        type=e1.type if e1.confidence >= e2.confidence else e2.type,
        source_branch=e1.source_branch,
        confidence=max(e1.confidence, e2.confidence),
        created_at=min(e1.created_at, e2.created_at),
    )

    # Merge aliases
    all_aliases = set(e1.aliases + e2.aliases)
    if e1.name != merged.name:
        all_aliases.add(e1.name)
    if e2.name != merged.name:
        all_aliases.add(e2.name)
    all_aliases.discard(merged.name)
    merged.aliases = list(all_aliases)

    # Merge properties
    merged.properties = dict(e2.properties)  # Start with e2
    merged.properties.update(e1.properties)  # Override with e1

    # Check for property conflicts
    conflicting_props = {}
    for key in set(e1.properties.keys()) & set(e2.properties.keys()):
        if e1.properties[key] != e2.properties[key]:
            conflicting_props[key] = {
                "e1_value": e1.properties[key],
                "e2_value": e2.properties[key],
            }

    if conflicting_props:
        conflict = MergeConflict(
            entity_a=e1,
            entity_b=e2,
            conflict_type=ConflictType.PROPERTY_CONFLICT,
            resolution=strategy,
            details={"conflicting_properties": conflicting_props},
        )

    return merged, conflict


def resolve_conflicts(
    e1: Entity,
    e2: Entity,
    conflict_type: ConflictType,
    strategy: ResolutionStrategy = ResolutionStrategy.MERGE,
) -> tuple[Entity | None, MergeConflict]:
    """Resolve conflicts between two entities.

    Args:
        e1: First entity.
        e2: Second entity.
        conflict_type: Type of conflict to resolve.
        strategy: Resolution strategy.

    Returns:
        Tuple of (resolved entity or None, conflict record).
    """
    conflict = MergeConflict(
        entity_a=e1,
        entity_b=e2,
        conflict_type=conflict_type,
        resolution=strategy,
    )

    if strategy == ResolutionStrategy.KEEP_FIRST:
        return e1, conflict

    if strategy == ResolutionStrategy.KEEP_SECOND:
        return e2, conflict

    if strategy == ResolutionStrategy.DISCARD_BOTH:
        return None, conflict

    if strategy == ResolutionStrategy.KEEP_BOTH:
        conflict.details["note"] = "Both entities kept as separate"
        return e1, conflict  # Return e1, but both should remain

    # Default to merge
    merged, merge_conflict = merge_entities(e1, e2, ResolutionStrategy.MERGE)
    if merge_conflict:
        conflict.details.update(merge_conflict.details)

    return merged, conflict


def deduplicate_graph(
    graph: KnowledgeGraph,
    threshold: float = 0.8,
    embedding_fn: EmbeddingFn | None = None,
    strategy: ResolutionStrategy = ResolutionStrategy.MERGE,
) -> tuple[KnowledgeGraph, list[MergeConflict]]:
    """Remove duplicate entities from a graph.

    Args:
        graph: The graph to deduplicate.
        threshold: Minimum similarity to merge.
        embedding_fn: Optional function to compute embeddings.
        strategy: How to resolve duplicates.

    Returns:
        Tuple of (deduplicated graph, list of conflicts).
    """
    conflicts = []
    entity_mapping: dict[str, str] = {}  # old_id -> new_id

    # Find and merge duplicates
    duplicates = find_duplicates(graph, threshold, embedding_fn)

    # Process duplicates, keeping track of what's been merged
    merged_into: dict[str, str] = {}  # Maps merged entity ID to survivor ID

    for e1, e2, _sim in duplicates:
        # Find the current representative for each entity
        id1 = e1.id
        while id1 in merged_into:
            id1 = merged_into[id1]

        id2 = e2.id
        while id2 in merged_into:
            id2 = merged_into[id2]

        if id1 == id2:
            # Already merged
            continue

        # Get current entities
        current_e1 = graph.entities.get(id1)
        current_e2 = graph.entities.get(id2)

        if not current_e1 or not current_e2:
            continue

        # Merge
        merged, conflict = merge_entities(current_e1, current_e2, strategy)
        if conflict:
            conflicts.append(conflict)

        # Update graph
        graph.entities[id1] = merged
        del graph.entities[id2]
        merged_into[id2] = id1
        entity_mapping[id2] = id1

    # Update relation references
    for relation in graph.relations.values():
        while relation.source_entity in entity_mapping:
            relation.source_entity = entity_mapping[relation.source_entity]
        while relation.target_entity in entity_mapping:
            relation.target_entity = entity_mapping[relation.target_entity]

    # Remove self-referential relations that might have been created
    to_remove = [
        rid for rid, rel in graph.relations.items()
        if rel.source_entity == rel.target_entity
    ]
    for rid in to_remove:
        del graph.relations[rid]

    return graph, conflicts


def cluster_similar_entities(
    graph: KnowledgeGraph,
    threshold: float = 0.7,
    embedding_fn: EmbeddingFn | None = None,
) -> list[list[Entity]]:
    """Cluster similar entities together.

    Args:
        graph: The graph to analyze.
        threshold: Minimum similarity to cluster together.
        embedding_fn: Optional function to compute embeddings.

    Returns:
        List of entity clusters.
    """
    entities = list(graph.entities.values())

    # Simple union-find for clustering
    parent: dict[str, str] = {e.id: e.id for e in entities}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Group similar entities
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1:]:
            sim = entity_similarity(e1, e2, embedding_fn)
            if sim >= threshold:
                union(e1.id, e2.id)

    # Build clusters
    clusters: dict[str, list[Entity]] = {}
    for entity in entities:
        root = find(entity.id)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(entity)

    return list(clusters.values())


def suggest_canonical_name(entities: list[Entity]) -> str:
    """Suggest a canonical name for a group of similar entities.

    Args:
        entities: List of similar entities.

    Returns:
        Suggested canonical name.
    """
    if not entities:
        return ""

    # Use the name from the highest confidence entity
    best = max(entities, key=lambda e: e.confidence)
    return best.name


def update_relations_for_merge(
    graph: KnowledgeGraph,
    old_entity_id: str,
    new_entity_id: str,
) -> list[Relation]:
    """Update relations when an entity is merged.

    Args:
        graph: The graph to update.
        old_entity_id: ID of the entity being removed.
        new_entity_id: ID of the entity it's merged into.

    Returns:
        List of updated relations.
    """
    updated = []

    for relation in graph.relations.values():
        changed = False

        if relation.source_entity == old_entity_id:
            relation.source_entity = new_entity_id
            changed = True

        if relation.target_entity == old_entity_id:
            relation.target_entity = new_entity_id
            changed = True

        if changed:
            updated.append(relation)

    return updated


def resolve_by_confidence(e1: Entity, e2: Entity) -> Entity:
    """Resolve conflict by keeping higher confidence entity.

    Args:
        e1: First entity.
        e2: Second entity.

    Returns:
        The entity with higher confidence.
    """
    return e1 if e1.confidence >= e2.confidence else e2


def resolve_by_recency(e1: Entity, e2: Entity) -> Entity:
    """Resolve conflict by keeping more recent entity.

    Args:
        e1: First entity.
        e2: Second entity.

    Returns:
        The more recently created entity.
    """
    return e1 if e1.created_at >= e2.created_at else e2


def resolve_by_source(
    e1: Entity,
    e2: Entity,
    trusted_sources: list[str],
) -> Entity:
    """Resolve conflict by preferring trusted sources.

    Args:
        e1: First entity.
        e2: Second entity.
        trusted_sources: List of trusted branch IDs (in order of trust).

    Returns:
        The entity from the more trusted source.
    """
    def get_trust_score(e: Entity) -> int:
        if e.source_branch in trusted_sources:
            return len(trusted_sources) - trusted_sources.index(e.source_branch)
        return 0

    return e1 if get_trust_score(e1) >= get_trust_score(e2) else e2
