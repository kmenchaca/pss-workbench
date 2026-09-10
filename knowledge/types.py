"""Core dataclasses for the Emergent Knowledge Graph system.

This module defines the fundamental data structures used throughout
the knowledge graph system, including entities, relations, graphs,
and supporting types for extraction, merging, and conflict resolution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ConflictType(Enum):
    """Types of conflicts that can occur during graph merging."""

    DUPLICATE_ENTITY = "duplicate_entity"
    CONTRADICTORY_RELATION = "contradictory_relation"
    TYPE_MISMATCH = "type_mismatch"
    PROPERTY_CONFLICT = "property_conflict"
    CONFIDENCE_DISAGREEMENT = "confidence_disagreement"


class ResolutionStrategy(Enum):
    """Strategies for resolving merge conflicts."""

    KEEP_FIRST = "keep_first"
    KEEP_SECOND = "keep_second"
    MERGE = "merge"
    KEEP_BOTH = "keep_both"
    DISCARD_BOTH = "discard_both"
    MANUAL = "manual"


@dataclass
class Entity:
    """A node in the knowledge graph representing a concept or thing.

    Attributes:
        id: Unique identifier for the entity.
        name: Human-readable name.
        type: Entity type (e.g., "Person", "Organization", "Concept").
        properties: Key-value attributes of the entity.
        source_branch: ID of the branch that extracted this entity.
        confidence: Confidence score (0.0 to 1.0) in this entity's existence.
        created_at: Timestamp when entity was created.
        aliases: Alternative names for this entity.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = "Entity"
    properties: dict[str, Any] = field(default_factory=dict)
    source_branch: str | None = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    aliases: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.id == other.id

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties,
            "source_branch": self.source_branch,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "aliases": self.aliases,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Create entity from dictionary representation."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            type=data.get("type", "Entity"),
            properties=data.get("properties", {}),
            source_branch=data.get("source_branch"),
            confidence=data.get("confidence", 1.0),
            created_at=created_at,
            aliases=data.get("aliases", []),
        )


@dataclass
class Relation:
    """An edge in the knowledge graph representing a relationship.

    Attributes:
        id: Unique identifier for the relation.
        source_entity: ID of the source entity.
        target_entity: ID of the target entity.
        relation_type: Type of relationship (e.g., "works_for", "is_a").
        properties: Key-value attributes of the relation.
        source_branch: ID of the branch that extracted this relation.
        confidence: Confidence score (0.0 to 1.0) in this relation.
        bidirectional: Whether the relation applies in both directions.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity: str = ""
    target_entity: str = ""
    relation_type: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    source_branch: str | None = None
    confidence: float = 1.0
    bidirectional: bool = False

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return False
        return self.id == other.id

    def to_dict(self) -> dict[str, Any]:
        """Convert relation to dictionary representation."""
        return {
            "id": self.id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relation_type": self.relation_type,
            "properties": self.properties,
            "source_branch": self.source_branch,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relation:
        """Create relation from dictionary representation."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_entity=data.get("source_entity", ""),
            target_entity=data.get("target_entity", ""),
            relation_type=data.get("relation_type", ""),
            properties=data.get("properties", {}),
            source_branch=data.get("source_branch"),
            confidence=data.get("confidence", 1.0),
            bidirectional=data.get("bidirectional", False),
        )


@dataclass
class KnowledgeGraph:
    """A collection of entities and relations forming a knowledge graph.

    Attributes:
        id: Unique identifier for the graph.
        entities: Map of entity ID to Entity.
        relations: Map of relation ID to Relation.
        metadata: Additional graph metadata.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def entity_count(self) -> int:
        """Return number of entities in the graph."""
        return len(self.entities)

    def relation_count(self) -> int:
        """Return number of relations in the graph."""
        return len(self.relations)

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "id": self.id,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": {k: v.to_dict() for k, v in self.relations.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        """Create graph from dictionary representation."""
        entities = {
            k: Entity.from_dict(v)
            for k, v in data.get("entities", {}).items()
        }
        relations = {
            k: Relation.from_dict(v)
            for k, v in data.get("relations", {}).items()
        }
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            entities=entities,
            relations=relations,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExtractionResult:
    """Output from a branch's extraction process.

    Attributes:
        entities: List of extracted entities.
        relations: List of extracted relations.
        provenance: Information about extraction source.
        branch_id: ID of the branch that performed extraction.
        raw_text: Original text that was processed.
        extraction_time: When extraction occurred.
        confidence: Overall confidence in extraction quality.
    """

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    branch_id: str | None = None
    raw_text: str = ""
    extraction_time: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert extraction result to dictionary representation."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "provenance": self.provenance,
            "branch_id": self.branch_id,
            "raw_text": self.raw_text,
            "extraction_time": self.extraction_time.isoformat(),
            "confidence": self.confidence,
        }


@dataclass
class MergeConflict:
    """Represents a conflict encountered during graph merging.

    Attributes:
        entity_a: First conflicting entity (or None for relation conflicts).
        entity_b: Second conflicting entity (or None for relation conflicts).
        relation_a: First conflicting relation (or None for entity conflicts).
        relation_b: Second conflicting relation (or None for entity conflicts).
        conflict_type: Type of conflict encountered.
        resolution: How the conflict was resolved.
        details: Additional details about the conflict.
    """

    entity_a: Entity | None = None
    entity_b: Entity | None = None
    relation_a: Relation | None = None
    relation_b: Relation | None = None
    conflict_type: ConflictType = ConflictType.DUPLICATE_ENTITY
    resolution: ResolutionStrategy = ResolutionStrategy.MANUAL
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert conflict to dictionary representation."""
        return {
            "entity_a": self.entity_a.to_dict() if self.entity_a else None,
            "entity_b": self.entity_b.to_dict() if self.entity_b else None,
            "relation_a": self.relation_a.to_dict() if self.relation_a else None,
            "relation_b": self.relation_b.to_dict() if self.relation_b else None,
            "conflict_type": self.conflict_type.value,
            "resolution": self.resolution.value,
            "details": self.details,
        }


@dataclass
class NodeType:
    """Definition of a valid node type in a schema.

    Attributes:
        name: Name of the node type.
        parent: Parent type (for inheritance).
        required_properties: Properties that must be present.
        optional_properties: Properties that may be present.
        description: Human-readable description.
    """

    name: str = ""
    parent: str | None = None
    required_properties: list[str] = field(default_factory=list)
    optional_properties: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert node type to dictionary representation."""
        return {
            "name": self.name,
            "parent": self.parent,
            "required_properties": self.required_properties,
            "optional_properties": self.optional_properties,
            "description": self.description,
        }


@dataclass
class EdgeType:
    """Definition of a valid edge type in a schema.

    Attributes:
        name: Name of the edge type.
        source_types: Valid source entity types.
        target_types: Valid target entity types.
        required_properties: Properties that must be present.
        optional_properties: Properties that may be present.
        description: Human-readable description.
        bidirectional: Whether edges of this type are bidirectional.
    """

    name: str = ""
    source_types: list[str] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)
    required_properties: list[str] = field(default_factory=list)
    optional_properties: list[str] = field(default_factory=list)
    description: str = ""
    bidirectional: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert edge type to dictionary representation."""
        return {
            "name": self.name,
            "source_types": self.source_types,
            "target_types": self.target_types,
            "required_properties": self.required_properties,
            "optional_properties": self.optional_properties,
            "description": self.description,
            "bidirectional": self.bidirectional,
        }


@dataclass
class SchemaConstraint:
    """A constraint on the knowledge graph structure.

    Attributes:
        name: Name of the constraint.
        constraint_type: Type of constraint (cardinality, uniqueness, etc.).
        target: What the constraint applies to.
        parameters: Constraint parameters.
    """

    name: str = ""
    constraint_type: str = ""
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert constraint to dictionary representation."""
        return {
            "name": self.name,
            "constraint_type": self.constraint_type,
            "target": self.target,
            "parameters": self.parameters,
        }


@dataclass
class GraphSchema:
    """Schema defining valid node/edge types and constraints.

    Attributes:
        name: Name of the schema.
        node_types: Map of node type name to NodeType.
        edge_types: Map of edge type name to EdgeType.
        constraints: List of schema constraints.
        version: Schema version.
    """

    name: str = "default"
    node_types: dict[str, NodeType] = field(default_factory=dict)
    edge_types: dict[str, EdgeType] = field(default_factory=dict)
    constraints: list[SchemaConstraint] = field(default_factory=list)
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            "name": self.name,
            "node_types": {k: v.to_dict() for k, v in self.node_types.items()},
            "edge_types": {k: v.to_dict() for k, v in self.edge_types.items()},
            "constraints": [c.to_dict() for c in self.constraints],
            "version": self.version,
        }


@dataclass
class ProvenanceRecord:
    """Records the source and history of a piece of knowledge.

    Attributes:
        item_id: ID of the entity or relation.
        item_type: "entity" or "relation".
        sources: List of source identifiers.
        branch_history: Branches that contributed to this item.
        timestamps: When each source contributed.
        confidence_history: How confidence changed over time.
    """

    item_id: str = ""
    item_type: str = ""
    sources: list[str] = field(default_factory=list)
    branch_history: list[str] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance record to dictionary representation."""
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "sources": self.sources,
            "branch_history": self.branch_history,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "confidence_history": self.confidence_history,
        }


@dataclass
class QueryResult:
    """Result of a query against the knowledge graph.

    Attributes:
        entities: Matching entities.
        relations: Matching relations.
        subgraphs: Matching subgraphs (lists of entity/relation IDs).
        paths: Matching paths.
        aggregates: Computed aggregate values.
    """

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    subgraphs: list[list[str]] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)
    aggregates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert query result to dictionary representation."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "subgraphs": self.subgraphs,
            "paths": self.paths,
            "aggregates": self.aggregates,
        }
