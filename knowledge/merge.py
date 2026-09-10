"""Graph merging for the Emergent Knowledge Graph system.

This module provides functions for merging knowledge graphs,
handling conflicts, and tracking provenance during merges.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from .graph import add_entity, add_relation, copy_graph
from .resolution import (
    entity_similarity,
    find_duplicates_for_entity,
    merge_entities,
)
from .types import (
    ConflictType,
    Entity,
    KnowledgeGraph,
    MergeConflict,
    ProvenanceRecord,
    Relation,
    ResolutionStrategy,
)


# Type alias for embedding function
EmbeddingFn = Callable[[str], list[float]]


def merge_graphs(
    g1: KnowledgeGraph,
    g2: KnowledgeGraph,
    threshold: float = 0.8,
    embedding_fn: EmbeddingFn | None = None,
) -> tuple[KnowledgeGraph, list[MergeConflict]]:
    """Merge two knowledge graphs into one.

    Args:
        g1: First graph (base).
        g2: Second graph (to merge in).
        threshold: Similarity threshold for entity matching.
        embedding_fn: Optional function to compute embeddings.

    Returns:
        Tuple of (merged graph, list of conflicts).
    """
    merged = copy_graph(g1)
    conflicts: list[MergeConflict] = []
    entity_mapping: dict[str, str] = {}  # g2_id -> merged_id

    # Process entities from g2
    for entity in g2.entities.values():
        # Find matching entities in merged graph
        matches = find_duplicates_for_entity(
            entity, merged, threshold, embedding_fn
        )

        if matches:
            # Merge with best match
            best_match, _sim = matches[0]
            merged_entity, conflict = merge_entities(
                best_match, entity, ResolutionStrategy.MERGE
            )
            if conflict:
                conflicts.append(conflict)

            merged.entities[best_match.id] = merged_entity
            entity_mapping[entity.id] = best_match.id
        else:
            # Add as new entity
            merged.entities[entity.id] = entity
            entity_mapping[entity.id] = entity.id

    # Process relations from g2
    for relation in g2.relations.values():
        # Map entity references
        new_source = entity_mapping.get(relation.source_entity, relation.source_entity)
        new_target = entity_mapping.get(relation.target_entity, relation.target_entity)

        # Check if both endpoints exist
        if new_source not in merged.entities or new_target not in merged.entities:
            continue

        # Check for conflicting relations
        existing_conflict = find_conflicting_relation(
            merged, new_source, new_target, relation.relation_type
        )

        if existing_conflict:
            # Handle relation conflict
            resolved, conflict = edge_conflict_resolution(
                existing_conflict, relation, ResolutionStrategy.MERGE
            )
            if conflict:
                conflicts.append(conflict)
            if resolved:
                merged.relations[existing_conflict.id] = resolved
        else:
            # Add new relation with updated references
            new_relation = Relation(
                id=str(uuid.uuid4()),
                source_entity=new_source,
                target_entity=new_target,
                relation_type=relation.relation_type,
                properties=dict(relation.properties),
                source_branch=relation.source_branch,
                confidence=relation.confidence,
                bidirectional=relation.bidirectional,
            )
            merged.relations[new_relation.id] = new_relation

    return merged, conflicts


def find_conflicting_relation(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
    relation_type: str,
) -> Relation | None:
    """Find an existing relation that might conflict with a new one.

    Args:
        graph: The graph to search.
        source_id: Source entity ID.
        target_id: Target entity ID.
        relation_type: Type of relation.

    Returns:
        Conflicting relation if found, None otherwise.
    """
    for relation in graph.relations.values():
        if (relation.source_entity == source_id and
            relation.target_entity == target_id and
            relation.relation_type == relation_type):
            return relation
    return None


def edge_conflict_resolution(
    r1: Relation,
    r2: Relation,
    strategy: ResolutionStrategy = ResolutionStrategy.MERGE,
) -> tuple[Relation | None, MergeConflict | None]:
    """Resolve conflict between two relations.

    Args:
        r1: First relation (existing).
        r2: Second relation (new).
        strategy: How to resolve the conflict.

    Returns:
        Tuple of (resolved relation, optional conflict record).
    """
    # Check if relations have conflicting properties
    conflicting_props = {}
    for key in set(r1.properties.keys()) & set(r2.properties.keys()):
        if r1.properties[key] != r2.properties[key]:
            conflicting_props[key] = {
                "r1_value": r1.properties[key],
                "r2_value": r2.properties[key],
            }

    conflict = None
    if conflicting_props:
        conflict = MergeConflict(
            relation_a=r1,
            relation_b=r2,
            conflict_type=ConflictType.CONTRADICTORY_RELATION,
            resolution=strategy,
            details={"conflicting_properties": conflicting_props},
        )

    if strategy == ResolutionStrategy.KEEP_FIRST:
        return r1, conflict

    if strategy == ResolutionStrategy.KEEP_SECOND:
        return r2, conflict

    if strategy == ResolutionStrategy.DISCARD_BOTH:
        return None, conflict

    # Merge strategy: combine properties, use higher confidence
    merged = Relation(
        id=r1.id,
        source_entity=r1.source_entity,
        target_entity=r1.target_entity,
        relation_type=r1.relation_type,
        properties=dict(r2.properties),  # Start with r2
        source_branch=r1.source_branch,
        confidence=max(r1.confidence, r2.confidence),
        bidirectional=r1.bidirectional or r2.bidirectional,
    )
    merged.properties.update(r1.properties)  # Override with r1

    return merged, conflict


def incremental_merge(
    base: KnowledgeGraph,
    additions: KnowledgeGraph,
    threshold: float = 0.8,
    embedding_fn: EmbeddingFn | None = None,
) -> tuple[KnowledgeGraph, list[MergeConflict]]:
    """Add new knowledge to an existing graph incrementally.

    This is optimized for adding small amounts of new knowledge
    to a large existing graph.

    Args:
        base: Existing graph to add to.
        additions: New knowledge to add.
        threshold: Similarity threshold for entity matching.
        embedding_fn: Optional function to compute embeddings.

    Returns:
        Tuple of (updated graph, list of conflicts).
    """
    # For now, same as merge_graphs but could be optimized
    return merge_graphs(base, additions, threshold, embedding_fn)


def merge_with_provenance(
    graphs: list[KnowledgeGraph],
    source_names: list[str] | None = None,
    threshold: float = 0.8,
    embedding_fn: EmbeddingFn | None = None,
) -> tuple[KnowledgeGraph, list[MergeConflict], dict[str, ProvenanceRecord]]:
    """Merge multiple graphs while tracking where each fact came from.

    Args:
        graphs: List of graphs to merge.
        source_names: Names for each source graph.
        threshold: Similarity threshold for entity matching.
        embedding_fn: Optional function to compute embeddings.

    Returns:
        Tuple of (merged graph, conflicts, provenance records).
    """
    if not graphs:
        return KnowledgeGraph(), [], {}

    if source_names is None:
        source_names = [f"source_{i}" for i in range(len(graphs))]

    # Initialize provenance tracking
    provenance: dict[str, ProvenanceRecord] = {}

    # Start with first graph
    merged = copy_graph(graphs[0])

    # Record provenance for initial entities
    for entity in merged.entities.values():
        provenance[entity.id] = ProvenanceRecord(
            item_id=entity.id,
            item_type="entity",
            sources=[source_names[0]],
            branch_history=[entity.source_branch] if entity.source_branch else [],
            confidence_history=[entity.confidence],
        )

    # Record provenance for initial relations
    for relation in merged.relations.values():
        provenance[relation.id] = ProvenanceRecord(
            item_id=relation.id,
            item_type="relation",
            sources=[source_names[0]],
            branch_history=[relation.source_branch] if relation.source_branch else [],
            confidence_history=[relation.confidence],
        )

    all_conflicts: list[MergeConflict] = []

    # Merge remaining graphs
    for i, graph in enumerate(graphs[1:], start=1):
        source_name = source_names[i]
        entity_mapping: dict[str, str] = {}

        # Process entities
        for entity in graph.entities.values():
            matches = find_duplicates_for_entity(
                entity, merged, threshold, embedding_fn
            )

            if matches:
                best_match, _sim = matches[0]
                merged_entity, conflict = merge_entities(
                    best_match, entity, ResolutionStrategy.MERGE
                )
                if conflict:
                    all_conflicts.append(conflict)

                merged.entities[best_match.id] = merged_entity
                entity_mapping[entity.id] = best_match.id

                # Update provenance
                if best_match.id in provenance:
                    provenance[best_match.id].sources.append(source_name)
                    if entity.source_branch:
                        provenance[best_match.id].branch_history.append(entity.source_branch)
                    provenance[best_match.id].confidence_history.append(entity.confidence)
            else:
                merged.entities[entity.id] = entity
                entity_mapping[entity.id] = entity.id

                # Create provenance record
                provenance[entity.id] = ProvenanceRecord(
                    item_id=entity.id,
                    item_type="entity",
                    sources=[source_name],
                    branch_history=[entity.source_branch] if entity.source_branch else [],
                    confidence_history=[entity.confidence],
                )

        # Process relations
        for relation in graph.relations.values():
            new_source = entity_mapping.get(relation.source_entity, relation.source_entity)
            new_target = entity_mapping.get(relation.target_entity, relation.target_entity)

            if new_source not in merged.entities or new_target not in merged.entities:
                continue

            existing = find_conflicting_relation(
                merged, new_source, new_target, relation.relation_type
            )

            if existing:
                resolved, conflict = edge_conflict_resolution(
                    existing, relation, ResolutionStrategy.MERGE
                )
                if conflict:
                    all_conflicts.append(conflict)
                if resolved:
                    merged.relations[existing.id] = resolved

                    # Update provenance
                    if existing.id in provenance:
                        provenance[existing.id].sources.append(source_name)
                        if relation.source_branch:
                            provenance[existing.id].branch_history.append(relation.source_branch)
                        provenance[existing.id].confidence_history.append(relation.confidence)
            else:
                new_relation = Relation(
                    id=str(uuid.uuid4()),
                    source_entity=new_source,
                    target_entity=new_target,
                    relation_type=relation.relation_type,
                    properties=dict(relation.properties),
                    source_branch=relation.source_branch,
                    confidence=relation.confidence,
                    bidirectional=relation.bidirectional,
                )
                merged.relations[new_relation.id] = new_relation

                # Create provenance record
                provenance[new_relation.id] = ProvenanceRecord(
                    item_id=new_relation.id,
                    item_type="relation",
                    sources=[source_name],
                    branch_history=[relation.source_branch] if relation.source_branch else [],
                    confidence_history=[relation.confidence],
                )

    return merged, all_conflicts, provenance


def merge_with_voting(
    graphs: list[KnowledgeGraph],
    min_votes: int = 2,
    threshold: float = 0.8,
    embedding_fn: EmbeddingFn | None = None,
) -> KnowledgeGraph:
    """Merge graphs using voting - only keep facts that appear in multiple sources.

    Args:
        graphs: List of graphs to merge.
        min_votes: Minimum number of graphs that must contain a fact.
        threshold: Similarity threshold for entity matching.
        embedding_fn: Optional function to compute embeddings.

    Returns:
        Merged graph with only multiply-attested facts.
    """
    if len(graphs) < min_votes:
        # Not enough sources to meet threshold
        return KnowledgeGraph()

    # Track entity votes
    entity_votes: dict[str, list[Entity]] = {}

    # Group similar entities across graphs
    for graph in graphs:
        for entity in graph.entities.values():
            # Find matching group
            matched = False
            for key, group in entity_votes.items():
                # Check similarity with first entity in group
                sim = entity_similarity(entity, group[0], embedding_fn)
                if sim >= threshold:
                    group.append(entity)
                    matched = True
                    break

            if not matched:
                # Create new group
                entity_votes[entity.id] = [entity]

    # Create merged graph with entities that have enough votes
    merged = KnowledgeGraph()
    entity_mapping: dict[str, str] = {}  # Original ID -> merged ID

    for group_id, group in entity_votes.items():
        if len(group) >= min_votes:
            # Use highest confidence entity as canonical
            canonical = max(group, key=lambda e: e.confidence)
            merged.entities[canonical.id] = canonical

            # Map all original IDs to canonical ID
            for entity in group:
                entity_mapping[entity.id] = canonical.id

    # Track relation votes
    relation_votes: dict[tuple[str, str, str], list[Relation]] = {}

    for graph in graphs:
        for relation in graph.relations.values():
            # Map to canonical entity IDs
            source = entity_mapping.get(relation.source_entity)
            target = entity_mapping.get(relation.target_entity)

            if source and target:
                key = (source, target, relation.relation_type)
                if key not in relation_votes:
                    relation_votes[key] = []
                relation_votes[key].append(relation)

    # Add relations with enough votes
    for (source, target, rtype), group in relation_votes.items():
        if len(group) >= min_votes:
            # Use highest confidence relation
            canonical = max(group, key=lambda r: r.confidence)
            new_relation = Relation(
                id=str(uuid.uuid4()),
                source_entity=source,
                target_entity=target,
                relation_type=rtype,
                properties=dict(canonical.properties),
                confidence=canonical.confidence,
                bidirectional=canonical.bidirectional,
            )
            merged.relations[new_relation.id] = new_relation

    return merged


def diff_graphs(
    g1: KnowledgeGraph,
    g2: KnowledgeGraph,
    threshold: float = 0.8,
    embedding_fn: EmbeddingFn | None = None,
) -> dict[str, Any]:
    """Compute the difference between two graphs.

    Args:
        g1: First graph.
        g2: Second graph.
        threshold: Similarity threshold for entity matching.
        embedding_fn: Optional function to compute embeddings.

    Returns:
        Dictionary with added, removed, and modified items.
    """
    diff: dict[str, Any] = {
        "entities_added": [],
        "entities_removed": [],
        "entities_modified": [],
        "relations_added": [],
        "relations_removed": [],
        "relations_modified": [],
    }

    # Map g1 entities to g2 entities
    g1_to_g2: dict[str, str] = {}
    g2_matched: set[str] = set()

    for e1 in g1.entities.values():
        matches = find_duplicates_for_entity(e1, g2, threshold, embedding_fn)
        if matches:
            best_match, _sim = matches[0]
            g1_to_g2[e1.id] = best_match.id
            g2_matched.add(best_match.id)

            # Check if modified
            if e1.properties != best_match.properties or e1.name != best_match.name:
                diff["entities_modified"].append({
                    "original": e1,
                    "modified": best_match,
                })
        else:
            diff["entities_removed"].append(e1)

    # Find added entities
    for e2 in g2.entities.values():
        if e2.id not in g2_matched:
            diff["entities_added"].append(e2)

    # Compare relations
    g2_relations_matched: set[str] = set()

    for r1 in g1.relations.values():
        # Find matching relation in g2
        matched = False
        for r2 in g2.relations.values():
            if r2.id in g2_relations_matched:
                continue

            # Check if endpoints match
            r1_source_mapped = g1_to_g2.get(r1.source_entity, r1.source_entity)
            r1_target_mapped = g1_to_g2.get(r1.target_entity, r1.target_entity)

            if (r2.source_entity == r1_source_mapped and
                r2.target_entity == r1_target_mapped and
                r2.relation_type == r1.relation_type):
                g2_relations_matched.add(r2.id)
                matched = True

                # Check if modified
                if r1.properties != r2.properties:
                    diff["relations_modified"].append({
                        "original": r1,
                        "modified": r2,
                    })
                break

        if not matched:
            diff["relations_removed"].append(r1)

    # Find added relations
    for r2 in g2.relations.values():
        if r2.id not in g2_relations_matched:
            diff["relations_added"].append(r2)

    return diff
