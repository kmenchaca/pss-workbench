"""Graph consistency checking for the Emergent Knowledge Graph system.

This module provides functions for validating graph consistency,
finding contradictions, checking reachability, and type validation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .schema import (
    get_type_ancestors,
    is_subtype_of,
    validate_entity,
    validate_relation,
    validate_relation_endpoints,
)
from .types import Entity, GraphSchema, KnowledgeGraph, Relation


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue found in the graph.

    Attributes:
        issue_type: Type of issue (error, warning, info).
        category: Category of issue (schema, logic, structure).
        message: Human-readable description.
        entity_ids: Affected entity IDs.
        relation_ids: Affected relation IDs.
        details: Additional details.
    """

    issue_type: str = "error"  # error, warning, info
    category: str = ""  # schema, logic, structure
    message: str = ""
    entity_ids: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "issue_type": self.issue_type,
            "category": self.category,
            "message": self.message,
            "entity_ids": self.entity_ids,
            "relation_ids": self.relation_ids,
            "details": self.details,
        }


@dataclass
class ConsistencyReport:
    """Report from a consistency check.

    Attributes:
        is_consistent: Whether the graph is fully consistent.
        issues: List of consistency issues found.
        stats: Summary statistics.
    """

    is_consistent: bool = True
    issues: list[ConsistencyIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: ConsistencyIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
        if issue.issue_type == "error":
            self.is_consistent = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_consistent": self.is_consistent,
            "issues": [i.to_dict() for i in self.issues],
            "stats": self.stats,
        }


def check_consistency(
    graph: KnowledgeGraph,
    schema: GraphSchema,
) -> ConsistencyReport:
    """Validate a graph against a schema.

    Args:
        graph: The graph to validate.
        schema: The schema to validate against.

    Returns:
        ConsistencyReport with any issues found.
    """
    report = ConsistencyReport()
    report.stats = {
        "entities_checked": 0,
        "relations_checked": 0,
        "schema_errors": 0,
    }

    # Validate entities
    for entity in graph.entities.values():
        report.stats["entities_checked"] += 1
        errors = validate_entity(entity, schema)
        for error in errors:
            report.add_issue(ConsistencyIssue(
                issue_type="error",
                category="schema",
                message=error,
                entity_ids=[entity.id],
            ))
            report.stats["schema_errors"] += 1

    # Validate relations
    for relation in graph.relations.values():
        report.stats["relations_checked"] += 1

        # Basic relation validation
        errors = validate_relation(relation, schema)
        for error in errors:
            report.add_issue(ConsistencyIssue(
                issue_type="error",
                category="schema",
                message=error,
                relation_ids=[relation.id],
            ))
            report.stats["schema_errors"] += 1

        # Endpoint validation
        source_entity = graph.entities.get(relation.source_entity)
        target_entity = graph.entities.get(relation.target_entity)

        if source_entity and target_entity:
            errors = validate_relation_endpoints(
                relation, source_entity, target_entity, schema
            )
            for error in errors:
                report.add_issue(ConsistencyIssue(
                    issue_type="error",
                    category="schema",
                    message=error,
                    relation_ids=[relation.id],
                    entity_ids=[source_entity.id, target_entity.id],
                ))
                report.stats["schema_errors"] += 1

    return report


def find_contradictions(graph: KnowledgeGraph) -> list[ConsistencyIssue]:
    """Find logical contradictions in the graph.

    Looks for:
    - Conflicting relations (e.g., A is_child_of B and A is_parent_of B)
    - Self-contradictory properties
    - Circular hierarchies in hierarchical relations

    Args:
        graph: The graph to analyze.

    Returns:
        List of contradiction issues.
    """
    issues = []

    # Define contradictory relation pairs
    contradictory_pairs = [
        ("is_parent_of", "is_child_of"),
        ("before", "after"),
        ("causes", "caused_by"),
        ("greater_than", "less_than"),
        ("supports", "contradicts"),
    ]

    # Build relation index
    relations_by_endpoints: dict[tuple[str, str], list[Relation]] = {}
    for relation in graph.relations.values():
        key = (relation.source_entity, relation.target_entity)
        if key not in relations_by_endpoints:
            relations_by_endpoints[key] = []
        relations_by_endpoints[key].append(relation)

    # Check for contradictory relations
    for (source, target), relations in relations_by_endpoints.items():
        # Check reverse direction
        reverse_key = (target, source)
        reverse_relations = relations_by_endpoints.get(reverse_key, [])

        for r1 in relations:
            for r2 in reverse_relations:
                for pair in contradictory_pairs:
                    if (r1.relation_type == pair[0] and r2.relation_type == pair[1]) or \
                       (r1.relation_type == pair[1] and r2.relation_type == pair[0]):
                        issues.append(ConsistencyIssue(
                            issue_type="error",
                            category="logic",
                            message=f"Contradictory relations: {r1.relation_type} and {r2.relation_type}",
                            relation_ids=[r1.id, r2.id],
                            entity_ids=[source, target],
                        ))

    # Check for circular hierarchies in is_a and part_of relations
    hierarchical_types = ["is_a", "part_of", "subclass_of", "instance_of"]

    for rtype in hierarchical_types:
        # Build adjacency list for this relation type
        adjacency: dict[str, list[str]] = {}
        for relation in graph.relations.values():
            if relation.relation_type == rtype:
                if relation.source_entity not in adjacency:
                    adjacency[relation.source_entity] = []
                adjacency[relation.source_entity].append(relation.target_entity)

        # Find cycles using DFS
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(node: str, path: list[str]) -> list[str] | None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    cycle = has_cycle(neighbor, path)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]

            path.pop()
            rec_stack.remove(node)
            return None

        for start in adjacency:
            if start not in visited:
                cycle = has_cycle(start, [])
                if cycle:
                    issues.append(ConsistencyIssue(
                        issue_type="error",
                        category="logic",
                        message=f"Circular hierarchy in {rtype} relations",
                        entity_ids=cycle,
                        details={"relation_type": rtype, "cycle": cycle},
                    ))

    return issues


def check_reachability(graph: KnowledgeGraph) -> ConsistencyReport:
    """Check for orphaned nodes (entities not connected to anything).

    Args:
        graph: The graph to analyze.

    Returns:
        ConsistencyReport with reachability issues.
    """
    report = ConsistencyReport()

    # Find connected entities
    connected: set[str] = set()
    for relation in graph.relations.values():
        connected.add(relation.source_entity)
        connected.add(relation.target_entity)

    # Find orphans
    orphans = []
    for entity_id in graph.entities:
        if entity_id not in connected:
            orphans.append(entity_id)

    if orphans:
        report.add_issue(ConsistencyIssue(
            issue_type="warning",
            category="structure",
            message=f"Found {len(orphans)} orphaned entities with no relations",
            entity_ids=orphans,
        ))

    # Find connected components
    parent: dict[str, str] = {eid: eid for eid in graph.entities}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for relation in graph.relations.values():
        if relation.source_entity in parent and relation.target_entity in parent:
            union(relation.source_entity, relation.target_entity)

    # Count components
    components: dict[str, list[str]] = {}
    for entity_id in graph.entities:
        root = find(entity_id)
        if root not in components:
            components[root] = []
        components[root].append(entity_id)

    if len(components) > 1:
        report.add_issue(ConsistencyIssue(
            issue_type="info",
            category="structure",
            message=f"Graph has {len(components)} disconnected components",
            details={
                "component_sizes": [len(c) for c in components.values()],
            },
        ))

    report.stats = {
        "total_entities": len(graph.entities),
        "connected_entities": len(connected),
        "orphaned_entities": len(orphans),
        "components": len(components),
    }

    return report


def type_check(graph: KnowledgeGraph, schema: GraphSchema) -> ConsistencyReport:
    """Validate that all entities and relations have proper types.

    Args:
        graph: The graph to check.
        schema: The schema defining valid types.

    Returns:
        ConsistencyReport with type errors.
    """
    report = ConsistencyReport()
    report.stats = {
        "valid_entity_types": 0,
        "invalid_entity_types": 0,
        "valid_relation_types": 0,
        "invalid_relation_types": 0,
    }

    # Check entity types
    for entity in graph.entities.values():
        if entity.type in schema.node_types:
            report.stats["valid_entity_types"] += 1
        else:
            report.stats["invalid_entity_types"] += 1
            report.add_issue(ConsistencyIssue(
                issue_type="error",
                category="schema",
                message=f"Unknown entity type: {entity.type}",
                entity_ids=[entity.id],
                details={"valid_types": list(schema.node_types.keys())},
            ))

    # Check relation types
    for relation in graph.relations.values():
        if relation.relation_type in schema.edge_types:
            report.stats["valid_relation_types"] += 1
        else:
            report.stats["invalid_relation_types"] += 1
            report.add_issue(ConsistencyIssue(
                issue_type="error",
                category="schema",
                message=f"Unknown relation type: {relation.relation_type}",
                relation_ids=[relation.id],
                details={"valid_types": list(schema.edge_types.keys())},
            ))

    return report


def check_required_relations(
    graph: KnowledgeGraph,
    schema: GraphSchema,
    requirements: dict[str, list[str]],
) -> ConsistencyReport:
    """Check that entities have required outgoing relations.

    Args:
        graph: The graph to check.
        schema: The schema for validation.
        requirements: Map of entity type to required relation types.

    Returns:
        ConsistencyReport with missing relation issues.
    """
    report = ConsistencyReport()

    # Build index of outgoing relations per entity
    outgoing: dict[str, set[str]] = {}
    for relation in graph.relations.values():
        if relation.source_entity not in outgoing:
            outgoing[relation.source_entity] = set()
        outgoing[relation.source_entity].add(relation.relation_type)

    # Check requirements
    for entity in graph.entities.values():
        required = requirements.get(entity.type, [])
        entity_relations = outgoing.get(entity.id, set())

        for req_type in required:
            if req_type not in entity_relations:
                report.add_issue(ConsistencyIssue(
                    issue_type="warning",
                    category="schema",
                    message=f"Entity of type {entity.type} missing required relation: {req_type}",
                    entity_ids=[entity.id],
                    details={"required_relation": req_type},
                ))

    return report


def check_cardinality(
    graph: KnowledgeGraph,
    constraints: dict[str, tuple[int, int | None]],
) -> ConsistencyReport:
    """Check relation cardinality constraints.

    Args:
        graph: The graph to check.
        constraints: Map of relation type to (min, max) cardinality.
                    None for max means unlimited.

    Returns:
        ConsistencyReport with cardinality violations.
    """
    report = ConsistencyReport()

    # Count relations per source entity per type
    counts: dict[tuple[str, str], int] = {}
    for relation in graph.relations.values():
        key = (relation.source_entity, relation.relation_type)
        counts[key] = counts.get(key, 0) + 1

    # Check constraints
    for (entity_id, rtype), count in counts.items():
        if rtype in constraints:
            min_card, max_card = constraints[rtype]

            if count < min_card:
                report.add_issue(ConsistencyIssue(
                    issue_type="error",
                    category="schema",
                    message=f"Cardinality violation: {rtype} requires at least {min_card} relations",
                    entity_ids=[entity_id],
                    details={"actual": count, "minimum": min_card},
                ))

            if max_card is not None and count > max_card:
                report.add_issue(ConsistencyIssue(
                    issue_type="error",
                    category="schema",
                    message=f"Cardinality violation: {rtype} allows at most {max_card} relations",
                    entity_ids=[entity_id],
                    details={"actual": count, "maximum": max_card},
                ))

    return report


def check_uniqueness(
    graph: KnowledgeGraph,
    unique_properties: dict[str, list[str]],
) -> ConsistencyReport:
    """Check that specified properties are unique within entity types.

    Args:
        graph: The graph to check.
        unique_properties: Map of entity type to list of unique property names.

    Returns:
        ConsistencyReport with uniqueness violations.
    """
    report = ConsistencyReport()

    # Group entities by type
    by_type: dict[str, list[Entity]] = {}
    for entity in graph.entities.values():
        if entity.type not in by_type:
            by_type[entity.type] = []
        by_type[entity.type].append(entity)

    # Check uniqueness
    for etype, props in unique_properties.items():
        entities = by_type.get(etype, [])

        for prop in props:
            seen_values: dict[Any, str] = {}

            for entity in entities:
                value = entity.properties.get(prop)
                if value is not None:
                    if value in seen_values:
                        report.add_issue(ConsistencyIssue(
                            issue_type="error",
                            category="schema",
                            message=f"Uniqueness violation: {prop} value '{value}' is not unique",
                            entity_ids=[seen_values[value], entity.id],
                            details={"property": prop, "value": value},
                        ))
                    else:
                        seen_values[value] = entity.id

    return report


def full_consistency_check(
    graph: KnowledgeGraph,
    schema: GraphSchema,
) -> ConsistencyReport:
    """Perform a comprehensive consistency check.

    Args:
        graph: The graph to check.
        schema: The schema to validate against.

    Returns:
        Combined ConsistencyReport.
    """
    report = ConsistencyReport()

    # Schema validation
    schema_report = check_consistency(graph, schema)
    report.issues.extend(schema_report.issues)
    report.stats.update(schema_report.stats)

    # Logical contradictions
    contradictions = find_contradictions(graph)
    report.issues.extend(contradictions)

    # Reachability
    reach_report = check_reachability(graph)
    report.issues.extend(reach_report.issues)
    report.stats.update(reach_report.stats)

    # Type checking
    type_report = type_check(graph, schema)
    report.issues.extend(type_report.issues)
    report.stats.update(type_report.stats)

    # Update overall consistency
    report.is_consistent = not any(
        issue.issue_type == "error" for issue in report.issues
    )

    return report
