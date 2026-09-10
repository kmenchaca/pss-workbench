"""Knowledge graph operations for the Emergent Knowledge Graph system.

This module provides the KnowledgeGraph class with methods for
adding, querying, and manipulating graph data.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Iterator

from .types import Entity, KnowledgeGraph, QueryResult, Relation


def add_entity(graph: KnowledgeGraph, entity: Entity) -> KnowledgeGraph:
    """Add an entity to the graph.

    Args:
        graph: The graph to add to.
        entity: The entity to add.

    Returns:
        The modified graph.
    """
    graph.entities[entity.id] = entity
    return graph


def add_relation(graph: KnowledgeGraph, relation: Relation) -> KnowledgeGraph:
    """Add a relation to the graph.

    Args:
        graph: The graph to add to.
        relation: The relation to add.

    Returns:
        The modified graph.

    Raises:
        ValueError: If source or target entity doesn't exist.
    """
    if relation.source_entity not in graph.entities:
        raise ValueError(f"Source entity {relation.source_entity} not found")
    if relation.target_entity not in graph.entities:
        raise ValueError(f"Target entity {relation.target_entity} not found")

    graph.relations[relation.id] = relation
    return graph


def remove_entity(graph: KnowledgeGraph, entity_id: str) -> KnowledgeGraph:
    """Remove an entity and its relations from the graph.

    Args:
        graph: The graph to remove from.
        entity_id: ID of the entity to remove.

    Returns:
        The modified graph.
    """
    if entity_id not in graph.entities:
        return graph

    # Remove the entity
    del graph.entities[entity_id]

    # Remove relations involving this entity
    relations_to_remove = [
        r_id for r_id, r in graph.relations.items()
        if r.source_entity == entity_id or r.target_entity == entity_id
    ]
    for r_id in relations_to_remove:
        del graph.relations[r_id]

    return graph


def remove_relation(graph: KnowledgeGraph, relation_id: str) -> KnowledgeGraph:
    """Remove a relation from the graph.

    Args:
        graph: The graph to remove from.
        relation_id: ID of the relation to remove.

    Returns:
        The modified graph.
    """
    if relation_id in graph.relations:
        del graph.relations[relation_id]
    return graph


def get_entity(graph: KnowledgeGraph, entity_id: str) -> Entity | None:
    """Get an entity by ID.

    Args:
        graph: The graph to search.
        entity_id: ID of the entity.

    Returns:
        The entity, or None if not found.
    """
    return graph.entities.get(entity_id)


def get_relation(graph: KnowledgeGraph, relation_id: str) -> Relation | None:
    """Get a relation by ID.

    Args:
        graph: The graph to search.
        relation_id: ID of the relation.

    Returns:
        The relation, or None if not found.
    """
    return graph.relations.get(relation_id)


def find_entities_by_name(
    graph: KnowledgeGraph,
    name: str,
    exact: bool = False,
) -> list[Entity]:
    """Find entities by name.

    Args:
        graph: The graph to search.
        name: Name to search for.
        exact: If True, require exact match; otherwise substring.

    Returns:
        List of matching entities.
    """
    results = []
    name_lower = name.lower()

    for entity in graph.entities.values():
        if exact:
            if entity.name == name or name in entity.aliases:
                results.append(entity)
        else:
            if name_lower in entity.name.lower():
                results.append(entity)
            elif any(name_lower in alias.lower() for alias in entity.aliases):
                results.append(entity)

    return results


def find_entities_by_type(
    graph: KnowledgeGraph,
    entity_type: str,
) -> list[Entity]:
    """Find entities by type.

    Args:
        graph: The graph to search.
        entity_type: Type to search for.

    Returns:
        List of matching entities.
    """
    return [e for e in graph.entities.values() if e.type == entity_type]


def neighbors(
    graph: KnowledgeGraph,
    entity_id: str,
    relation_type: str | None = None,
    direction: str = "both",
) -> list[Entity]:
    """Get entities connected to the given entity.

    Args:
        graph: The graph to search.
        entity_id: ID of the entity.
        relation_type: Optional filter by relation type.
        direction: "outgoing", "incoming", or "both".

    Returns:
        List of connected entities.
    """
    if entity_id not in graph.entities:
        return []

    neighbor_ids: set[str] = set()

    for relation in graph.relations.values():
        # Filter by relation type if specified
        if relation_type and relation.relation_type != relation_type:
            continue

        # Check outgoing relations
        if direction in ("outgoing", "both"):
            if relation.source_entity == entity_id:
                neighbor_ids.add(relation.target_entity)

        # Check incoming relations
        if direction in ("incoming", "both"):
            if relation.target_entity == entity_id:
                neighbor_ids.add(relation.source_entity)

        # Handle bidirectional relations
        if relation.bidirectional and direction == "both":
            if relation.source_entity == entity_id:
                neighbor_ids.add(relation.target_entity)
            elif relation.target_entity == entity_id:
                neighbor_ids.add(relation.source_entity)

    return [graph.entities[eid] for eid in neighbor_ids if eid in graph.entities]


def get_relations_for_entity(
    graph: KnowledgeGraph,
    entity_id: str,
    direction: str = "both",
) -> list[Relation]:
    """Get all relations involving an entity.

    Args:
        graph: The graph to search.
        entity_id: ID of the entity.
        direction: "outgoing", "incoming", or "both".

    Returns:
        List of relations.
    """
    results = []

    for relation in graph.relations.values():
        if direction in ("outgoing", "both") and relation.source_entity == entity_id:
            results.append(relation)
        elif direction in ("incoming", "both") and relation.target_entity == entity_id:
            results.append(relation)

    return results


def shortest_path(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
    max_depth: int = 10,
) -> list[str] | None:
    """Find the shortest path between two entities.

    Args:
        graph: The graph to search.
        source_id: ID of the source entity.
        target_id: ID of the target entity.
        max_depth: Maximum path length to search.

    Returns:
        List of entity IDs forming the path, or None if no path exists.
    """
    if source_id not in graph.entities or target_id not in graph.entities:
        return None

    if source_id == target_id:
        return [source_id]

    # BFS for shortest path
    queue: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])
    visited: set[str] = {source_id}

    while queue:
        current_id, path = queue.popleft()

        if len(path) > max_depth:
            continue

        # Get neighbors
        for neighbor in neighbors(graph, current_id):
            if neighbor.id == target_id:
                return path + [neighbor.id]

            if neighbor.id not in visited:
                visited.add(neighbor.id)
                queue.append((neighbor.id, path + [neighbor.id]))

    return None


def all_paths(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
    max_depth: int = 5,
) -> list[list[str]]:
    """Find all paths between two entities up to max depth.

    Args:
        graph: The graph to search.
        source_id: ID of the source entity.
        target_id: ID of the target entity.
        max_depth: Maximum path length.

    Returns:
        List of paths (each path is a list of entity IDs).
    """
    if source_id not in graph.entities or target_id not in graph.entities:
        return []

    paths: list[list[str]] = []

    def dfs(current: str, path: list[str], visited: set[str]) -> None:
        if len(path) > max_depth:
            return

        if current == target_id:
            paths.append(path[:])
            return

        for neighbor in neighbors(graph, current):
            if neighbor.id not in visited:
                visited.add(neighbor.id)
                path.append(neighbor.id)
                dfs(neighbor.id, path, visited)
                path.pop()
                visited.remove(neighbor.id)

    dfs(source_id, [source_id], {source_id})
    return paths


def subgraph(
    graph: KnowledgeGraph,
    entity_ids: list[str],
    include_connecting: bool = True,
) -> KnowledgeGraph:
    """Extract a subgraph containing specified entities.

    Args:
        graph: The source graph.
        entity_ids: IDs of entities to include.
        include_connecting: If True, include relations between entities.

    Returns:
        A new KnowledgeGraph containing the subgraph.
    """
    entity_set = set(entity_ids)

    # Create new graph with selected entities
    new_graph = KnowledgeGraph()
    for eid in entity_ids:
        if eid in graph.entities:
            new_graph.entities[eid] = graph.entities[eid]

    # Include relations between the selected entities
    if include_connecting:
        for relation in graph.relations.values():
            if (relation.source_entity in entity_set and
                relation.target_entity in entity_set):
                new_graph.relations[relation.id] = relation

    return new_graph


def query(
    graph: KnowledgeGraph,
    entity_type: str | None = None,
    relation_type: str | None = None,
    properties: dict[str, Any] | None = None,
) -> QueryResult:
    """Query the graph for matching entities and relations.

    Args:
        graph: The graph to query.
        entity_type: Filter entities by type.
        relation_type: Filter relations by type.
        properties: Filter by property values.

    Returns:
        QueryResult with matching entities and relations.
    """
    result = QueryResult()

    # Query entities
    for entity in graph.entities.values():
        if entity_type and entity.type != entity_type:
            continue

        if properties:
            match = True
            for key, value in properties.items():
                if entity.properties.get(key) != value:
                    match = False
                    break
            if not match:
                continue

        result.entities.append(entity)

    # Query relations
    for relation in graph.relations.values():
        if relation_type and relation.relation_type != relation_type:
            continue

        if properties:
            match = True
            for key, value in properties.items():
                if relation.properties.get(key) != value:
                    match = False
                    break
            if not match:
                continue

        result.relations.append(relation)

    return result


def pattern_match(
    graph: KnowledgeGraph,
    pattern: dict[str, Any],
) -> QueryResult:
    """Match a pattern in the graph.

    Pattern format:
    {
        "entities": [
            {"var": "a", "type": "Person"},
            {"var": "b", "type": "Organization"}
        ],
        "relations": [
            {"source": "a", "target": "b", "type": "works_for"}
        ]
    }

    Args:
        graph: The graph to search.
        pattern: The pattern to match.

    Returns:
        QueryResult with matching subgraphs.
    """
    result = QueryResult()

    entity_patterns = pattern.get("entities", [])
    relation_patterns = pattern.get("relations", [])

    if not entity_patterns:
        return result

    # Find candidates for each entity variable
    candidates: dict[str, list[Entity]] = {}
    for ep in entity_patterns:
        var = ep.get("var", "")
        etype = ep.get("type")
        props = ep.get("properties", {})

        matches = []
        for entity in graph.entities.values():
            if etype and entity.type != etype:
                continue
            if props:
                match = all(entity.properties.get(k) == v for k, v in props.items())
                if not match:
                    continue
            matches.append(entity)
        candidates[var] = matches

    # Generate all combinations and check relation constraints
    def check_assignment(assignment: dict[str, Entity]) -> bool:
        for rp in relation_patterns:
            source_var = rp.get("source", "")
            target_var = rp.get("target", "")
            rtype = rp.get("type")

            if source_var not in assignment or target_var not in assignment:
                return False

            source_id = assignment[source_var].id
            target_id = assignment[target_var].id

            # Check if matching relation exists
            found = False
            for relation in graph.relations.values():
                if (relation.source_entity == source_id and
                    relation.target_entity == target_id):
                    if rtype is None or relation.relation_type == rtype:
                        found = True
                        break
            if not found:
                return False
        return True

    # Generate assignments (simplified for small patterns)
    def generate_assignments(
        vars: list[str],
        current: dict[str, Entity],
    ) -> Iterator[dict[str, Entity]]:
        if not vars:
            yield current
            return

        var = vars[0]
        remaining = vars[1:]
        for entity in candidates.get(var, []):
            current[var] = entity
            yield from generate_assignments(remaining, current)

    var_names = [ep.get("var", "") for ep in entity_patterns]
    for assignment in generate_assignments(var_names, {}):
        if check_assignment(assignment):
            entity_ids = [e.id for e in assignment.values()]
            result.subgraphs.append(entity_ids)
            result.entities.extend(assignment.values())

    # Deduplicate entities
    seen = set()
    unique_entities = []
    for e in result.entities:
        if e.id not in seen:
            seen.add(e.id)
            unique_entities.append(e)
    result.entities = unique_entities

    return result


def aggregate(
    graph: KnowledgeGraph,
    metric: str,
) -> dict[str, Any]:
    """Compute aggregate statistics on the graph.

    Args:
        graph: The graph to analyze.
        metric: The metric to compute:
            - "count": Entity and relation counts
            - "types": Count by type
            - "degree": Node degree statistics
            - "density": Graph density
            - "components": Connected component count

    Returns:
        Dictionary of computed values.
    """
    if metric == "count":
        return {
            "entities": len(graph.entities),
            "relations": len(graph.relations),
        }

    elif metric == "types":
        entity_types: dict[str, int] = {}
        for entity in graph.entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1

        relation_types: dict[str, int] = {}
        for relation in graph.relations.values():
            rtype = relation.relation_type
            relation_types[rtype] = relation_types.get(rtype, 0) + 1

        return {
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    elif metric == "degree":
        degrees: dict[str, int] = {eid: 0 for eid in graph.entities}
        for relation in graph.relations.values():
            if relation.source_entity in degrees:
                degrees[relation.source_entity] += 1
            if relation.target_entity in degrees:
                degrees[relation.target_entity] += 1

        if not degrees:
            return {"min": 0, "max": 0, "avg": 0.0, "distribution": {}}

        values = list(degrees.values())
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "distribution": {str(d): values.count(d) for d in set(values)},
        }

    elif metric == "density":
        n = len(graph.entities)
        m = len(graph.relations)
        if n <= 1:
            density = 0.0
        else:
            max_edges = n * (n - 1)
            density = m / max_edges if max_edges > 0 else 0.0
        return {"density": density, "nodes": n, "edges": m}

    elif metric == "components":
        # Find connected components using union-find
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

        components = len(set(find(eid) for eid in graph.entities))
        return {"components": components}

    return {}


def to_json(graph: KnowledgeGraph) -> str:
    """Serialize the graph to JSON.

    Args:
        graph: The graph to serialize.

    Returns:
        JSON string representation.
    """
    return json.dumps(graph.to_dict(), indent=2)


def from_json(json_str: str) -> KnowledgeGraph:
    """Deserialize a graph from JSON.

    Args:
        json_str: JSON string representation.

    Returns:
        The deserialized KnowledgeGraph.
    """
    data = json.loads(json_str)
    return KnowledgeGraph.from_dict(data)


def copy_graph(graph: KnowledgeGraph) -> KnowledgeGraph:
    """Create a deep copy of a graph.

    Args:
        graph: The graph to copy.

    Returns:
        A new KnowledgeGraph with copied data.
    """
    return from_json(to_json(graph))
