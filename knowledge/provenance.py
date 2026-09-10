"""Provenance tracking for the Emergent Knowledge Graph system.

This module provides functions for tracking the source and history
of knowledge, including confidence propagation and source reliability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .types import Entity, KnowledgeGraph, ProvenanceRecord, Relation


@dataclass
class SourceReliability:
    """Tracks reliability metrics for a knowledge source.

    Attributes:
        source_id: Identifier for the source.
        total_contributions: Number of facts contributed.
        verified_contributions: Number of facts verified correct.
        contradicted_contributions: Number of facts found incorrect.
        reliability_score: Computed reliability (0.0 to 1.0).
    """

    source_id: str = ""
    total_contributions: int = 0
    verified_contributions: int = 0
    contradicted_contributions: int = 0
    reliability_score: float = 1.0

    def update_score(self) -> None:
        """Recalculate reliability score based on contributions."""
        if self.total_contributions == 0:
            self.reliability_score = 1.0
        else:
            # Score based on verified vs contradicted, with a base assumption
            verified_weight = self.verified_contributions * 1.0
            contradicted_weight = self.contradicted_contributions * 2.0  # Penalize more
            neutral_weight = (
                self.total_contributions
                - self.verified_contributions
                - self.contradicted_contributions
            ) * 0.5

            total_weight = verified_weight + neutral_weight
            negative_weight = contradicted_weight

            self.reliability_score = max(
                0.0,
                min(1.0, total_weight / (total_weight + negative_weight + 0.1))
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "total_contributions": self.total_contributions,
            "verified_contributions": self.verified_contributions,
            "contradicted_contributions": self.contradicted_contributions,
            "reliability_score": self.reliability_score,
        }


@dataclass
class ProvenanceTracker:
    """Manages provenance records for a knowledge graph.

    Attributes:
        records: Map of item ID to provenance record.
        source_reliability: Map of source ID to reliability metrics.
    """

    records: dict[str, ProvenanceRecord] = field(default_factory=dict)
    source_reliability: dict[str, SourceReliability] = field(default_factory=dict)

    def tag_provenance(
        self,
        item_id: str,
        item_type: str,
        source: str,
        branch_id: str | None = None,
        confidence: float = 1.0,
    ) -> ProvenanceRecord:
        """Tag an item with provenance information.

        Args:
            item_id: ID of the entity or relation.
            item_type: "entity" or "relation".
            source: Source identifier.
            branch_id: Optional branch ID.
            confidence: Confidence score.

        Returns:
            The created or updated ProvenanceRecord.
        """
        if item_id in self.records:
            record = self.records[item_id]
            record.sources.append(source)
            record.timestamps.append(datetime.now())
            record.confidence_history.append(confidence)
            if branch_id:
                record.branch_history.append(branch_id)
        else:
            record = ProvenanceRecord(
                item_id=item_id,
                item_type=item_type,
                sources=[source],
                branch_history=[branch_id] if branch_id else [],
                timestamps=[datetime.now()],
                confidence_history=[confidence],
            )
            self.records[item_id] = record

        # Update source reliability tracking
        if source not in self.source_reliability:
            self.source_reliability[source] = SourceReliability(source_id=source)
        self.source_reliability[source].total_contributions += 1

        return record

    def trace_provenance(self, item_id: str) -> ProvenanceRecord | None:
        """Get the provenance record for an item.

        Args:
            item_id: ID of the item.

        Returns:
            The ProvenanceRecord, or None if not tracked.
        """
        return self.records.get(item_id)

    def get_sources(self, item_id: str) -> list[str]:
        """Get all sources that contributed to an item.

        Args:
            item_id: ID of the item.

        Returns:
            List of source identifiers.
        """
        record = self.records.get(item_id)
        return record.sources if record else []

    def verify_contribution(self, item_id: str, source: str) -> None:
        """Mark a contribution as verified correct.

        Args:
            item_id: ID of the item.
            source: Source that contributed it.
        """
        if source in self.source_reliability:
            self.source_reliability[source].verified_contributions += 1
            self.source_reliability[source].update_score()

    def contradict_contribution(self, item_id: str, source: str) -> None:
        """Mark a contribution as contradicted/incorrect.

        Args:
            item_id: ID of the item.
            source: Source that contributed it.
        """
        if source in self.source_reliability:
            self.source_reliability[source].contradicted_contributions += 1
            self.source_reliability[source].update_score()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "records": {k: v.to_dict() for k, v in self.records.items()},
            "source_reliability": {
                k: v.to_dict() for k, v in self.source_reliability.items()
            },
        }


def tag_provenance(
    item: Entity | Relation,
    source: str,
    tracker: ProvenanceTracker,
) -> ProvenanceRecord:
    """Tag an entity or relation with provenance.

    Args:
        item: The entity or relation.
        source: Source identifier.
        tracker: The provenance tracker.

    Returns:
        The ProvenanceRecord.
    """
    item_type = "entity" if isinstance(item, Entity) else "relation"
    confidence = item.confidence

    return tracker.tag_provenance(
        item_id=item.id,
        item_type=item_type,
        source=source,
        branch_id=item.source_branch,
        confidence=confidence,
    )


def trace_provenance(
    item_id: str,
    tracker: ProvenanceTracker,
) -> dict[str, Any]:
    """Get detailed provenance information for an item.

    Args:
        item_id: ID of the item.
        tracker: The provenance tracker.

    Returns:
        Dictionary with provenance details.
    """
    record = tracker.trace_provenance(item_id)
    if not record:
        return {"found": False, "item_id": item_id}

    # Calculate aggregate confidence
    if record.confidence_history:
        avg_confidence = sum(record.confidence_history) / len(record.confidence_history)
        min_confidence = min(record.confidence_history)
        max_confidence = max(record.confidence_history)
    else:
        avg_confidence = min_confidence = max_confidence = 0.0

    # Get source reliability scores
    source_scores = {}
    for source in set(record.sources):
        if source in tracker.source_reliability:
            source_scores[source] = tracker.source_reliability[source].reliability_score
        else:
            source_scores[source] = 1.0

    return {
        "found": True,
        "item_id": item_id,
        "item_type": record.item_type,
        "sources": list(set(record.sources)),
        "source_count": len(set(record.sources)),
        "contribution_count": len(record.sources),
        "branches": list(set(record.branch_history)),
        "first_seen": record.timestamps[0].isoformat() if record.timestamps else None,
        "last_updated": record.timestamps[-1].isoformat() if record.timestamps else None,
        "confidence": {
            "average": avg_confidence,
            "min": min_confidence,
            "max": max_confidence,
            "history": record.confidence_history,
        },
        "source_reliability": source_scores,
    }


def confidence_propagation(
    graph: KnowledgeGraph,
    tracker: ProvenanceTracker,
) -> dict[str, float]:
    """Propagate confidence scores through the graph.

    Entities connected to high-confidence entities get a boost,
    while those connected only to low-confidence entities get reduced.

    Args:
        graph: The knowledge graph.
        tracker: The provenance tracker.

    Returns:
        Map of item ID to propagated confidence score.
    """
    propagated: dict[str, float] = {}

    # Start with base confidences
    for entity in graph.entities.values():
        propagated[entity.id] = entity.confidence

    for relation in graph.relations.values():
        propagated[relation.id] = relation.confidence

    # Iterative propagation (simplified PageRank-like)
    damping = 0.85
    iterations = 5

    for _ in range(iterations):
        new_scores: dict[str, float] = {}

        for entity in graph.entities.values():
            # Get incoming confidence from relations
            incoming_conf = []
            for relation in graph.relations.values():
                if relation.target_entity == entity.id:
                    source_conf = propagated.get(relation.source_entity, 0.5)
                    rel_conf = propagated.get(relation.id, 0.5)
                    incoming_conf.append(source_conf * rel_conf)

                if relation.bidirectional and relation.source_entity == entity.id:
                    target_conf = propagated.get(relation.target_entity, 0.5)
                    rel_conf = propagated.get(relation.id, 0.5)
                    incoming_conf.append(target_conf * rel_conf)

            if incoming_conf:
                avg_incoming = sum(incoming_conf) / len(incoming_conf)
                base_conf = entity.confidence
                new_scores[entity.id] = (1 - damping) * base_conf + damping * avg_incoming
            else:
                new_scores[entity.id] = propagated[entity.id]

        # Update relation scores based on endpoint confidence
        for relation in graph.relations.values():
            source_conf = propagated.get(relation.source_entity, 0.5)
            target_conf = propagated.get(relation.target_entity, 0.5)
            base_conf = relation.confidence

            # Relation confidence influenced by endpoints
            endpoint_influence = (source_conf + target_conf) / 2
            new_scores[relation.id] = (
                0.7 * base_conf + 0.3 * endpoint_influence
            )

        propagated.update(new_scores)

    return propagated


def source_reliability(
    sources: list[str],
    tracker: ProvenanceTracker,
) -> dict[str, float]:
    """Get reliability scores for multiple sources.

    Args:
        sources: List of source identifiers.
        tracker: The provenance tracker.

    Returns:
        Map of source ID to reliability score.
    """
    result = {}
    for source in sources:
        if source in tracker.source_reliability:
            result[source] = tracker.source_reliability[source].reliability_score
        else:
            result[source] = 1.0  # Default to fully trusted
    return result


def weighted_confidence(
    item_id: str,
    tracker: ProvenanceTracker,
) -> float:
    """Calculate confidence weighted by source reliability.

    Args:
        item_id: ID of the item.
        tracker: The provenance tracker.

    Returns:
        Weighted confidence score.
    """
    record = tracker.trace_provenance(item_id)
    if not record or not record.sources:
        return 0.5

    # Weight each confidence contribution by source reliability
    weighted_sum = 0.0
    weight_sum = 0.0

    for source, confidence in zip(record.sources, record.confidence_history):
        reliability = 1.0
        if source in tracker.source_reliability:
            reliability = tracker.source_reliability[source].reliability_score

        weighted_sum += confidence * reliability
        weight_sum += reliability

    return weighted_sum / weight_sum if weight_sum > 0 else 0.5


def merge_provenance(
    p1: ProvenanceRecord,
    p2: ProvenanceRecord,
) -> ProvenanceRecord:
    """Merge two provenance records.

    Args:
        p1: First provenance record.
        p2: Second provenance record.

    Returns:
        Merged provenance record.
    """
    return ProvenanceRecord(
        item_id=p1.item_id,
        item_type=p1.item_type,
        sources=p1.sources + p2.sources,
        branch_history=p1.branch_history + p2.branch_history,
        timestamps=p1.timestamps + p2.timestamps,
        confidence_history=p1.confidence_history + p2.confidence_history,
    )


def filter_by_source(
    graph: KnowledgeGraph,
    tracker: ProvenanceTracker,
    sources: list[str],
    require_all: bool = False,
) -> KnowledgeGraph:
    """Filter graph to items from specific sources.

    Args:
        graph: The knowledge graph.
        tracker: The provenance tracker.
        sources: Sources to filter by.
        require_all: If True, require all sources; if False, require any.

    Returns:
        Filtered graph.
    """
    filtered = KnowledgeGraph()
    source_set = set(sources)

    for entity in graph.entities.values():
        record = tracker.trace_provenance(entity.id)
        if record:
            entity_sources = set(record.sources)
            if require_all:
                if source_set <= entity_sources:
                    filtered.entities[entity.id] = entity
            else:
                if source_set & entity_sources:
                    filtered.entities[entity.id] = entity

    for relation in graph.relations.values():
        # Only include relations where both endpoints are in filtered graph
        if relation.source_entity in filtered.entities and \
           relation.target_entity in filtered.entities:
            record = tracker.trace_provenance(relation.id)
            if record:
                rel_sources = set(record.sources)
                if require_all:
                    if source_set <= rel_sources:
                        filtered.relations[relation.id] = relation
                else:
                    if source_set & rel_sources:
                        filtered.relations[relation.id] = relation

    return filtered


def filter_by_confidence(
    graph: KnowledgeGraph,
    tracker: ProvenanceTracker,
    min_confidence: float = 0.5,
    use_weighted: bool = True,
) -> KnowledgeGraph:
    """Filter graph to items above a confidence threshold.

    Args:
        graph: The knowledge graph.
        tracker: The provenance tracker.
        min_confidence: Minimum confidence required.
        use_weighted: If True, use source-weighted confidence.

    Returns:
        Filtered graph.
    """
    filtered = KnowledgeGraph()

    for entity in graph.entities.values():
        if use_weighted:
            conf = weighted_confidence(entity.id, tracker)
        else:
            conf = entity.confidence

        if conf >= min_confidence:
            filtered.entities[entity.id] = entity

    for relation in graph.relations.values():
        if relation.source_entity in filtered.entities and \
           relation.target_entity in filtered.entities:
            if use_weighted:
                conf = weighted_confidence(relation.id, tracker)
            else:
                conf = relation.confidence

            if conf >= min_confidence:
                filtered.relations[relation.id] = relation

    return filtered


def audit_trail(
    item_id: str,
    tracker: ProvenanceTracker,
) -> list[dict[str, Any]]:
    """Generate a detailed audit trail for an item.

    Args:
        item_id: ID of the item.
        tracker: The provenance tracker.

    Returns:
        List of audit events in chronological order.
    """
    record = tracker.trace_provenance(item_id)
    if not record:
        return []

    trail = []
    for i, (source, timestamp, confidence) in enumerate(
        zip(record.sources, record.timestamps, record.confidence_history)
    ):
        branch = record.branch_history[i] if i < len(record.branch_history) else None
        reliability = 1.0
        if source in tracker.source_reliability:
            reliability = tracker.source_reliability[source].reliability_score

        trail.append({
            "index": i,
            "timestamp": timestamp.isoformat(),
            "source": source,
            "branch": branch,
            "confidence": confidence,
            "source_reliability": reliability,
        })

    return trail
