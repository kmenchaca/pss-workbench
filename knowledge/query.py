"""Query interface for the Emergent Knowledge Graph system.

This module provides functions for querying knowledge graphs,
including natural language queries, pattern matching, aggregation,
and export to various formats.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .graph import (
    all_paths,
    find_entities_by_name,
    find_entities_by_type,
    neighbors,
    pattern_match,
    query,
    shortest_path,
    subgraph,
)
from .types import Entity, KnowledgeGraph, QueryResult, Relation


# Type alias for LLM call function
LLMCallFn = Callable[[str, str | None], str]


def natural_language_query(
    graph: KnowledgeGraph,
    question: str,
    llm_call: LLMCallFn | None = None,
) -> QueryResult:
    """Answer a question about the knowledge graph.

    Uses an LLM to translate the question to structured queries,
    or falls back to keyword matching.

    Args:
        graph: The knowledge graph to query.
        question: Natural language question.
        llm_call: Optional LLM function for query translation.

    Returns:
        QueryResult with matching entities and relations.
    """
    result = QueryResult()

    if llm_call:
        # Use LLM to translate question to query
        result = _llm_translate_query(graph, question, llm_call)
    else:
        # Fallback to keyword matching
        result = _keyword_query(graph, question)

    return result


def _llm_translate_query(
    graph: KnowledgeGraph,
    question: str,
    llm_call: LLMCallFn,
) -> QueryResult:
    """Use LLM to translate natural language to structured query."""
    # Build context about available entity/relation types
    entity_types = list(set(e.type for e in graph.entities.values()))
    entity_names = [e.name for e in graph.entities.values()][:20]  # Sample
    relation_types = list(set(r.relation_type for r in graph.relations.values()))

    system_prompt = """You are a knowledge graph query translator.
Given a question and information about a knowledge graph, output a JSON query specification.

Query format:
{
  "query_type": "entity" | "relation" | "path" | "pattern",
  "entity_name": "name to search for" (for entity queries),
  "entity_type": "type to filter by" (optional),
  "relation_type": "type to filter by" (for relation queries),
  "source_name": "source entity name" (for path queries),
  "target_name": "target entity name" (for path queries),
  "pattern": {...} (for pattern queries)
}

Respond only with the JSON query."""

    prompt = f"""Translate this question into a knowledge graph query:

Question: {question}

Available entity types: {entity_types}
Sample entity names: {entity_names}
Relation types: {relation_types}

Output the query JSON:"""

    response = llm_call(prompt, system_prompt)

    # Parse response
    try:
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            query_spec = json.loads(json_match.group(0))
            return _execute_query_spec(graph, query_spec)
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback to keyword query
    return _keyword_query(graph, question)


def _keyword_query(graph: KnowledgeGraph, question: str) -> QueryResult:
    """Simple keyword-based query matching."""
    result = QueryResult()
    words = question.lower().split()

    # Search entities
    for entity in graph.entities.values():
        name_lower = entity.name.lower()
        if any(word in name_lower for word in words):
            result.entities.append(entity)
            continue

        # Check aliases
        for alias in entity.aliases:
            if any(word in alias.lower() for word in words):
                result.entities.append(entity)
                break

    # Search relations connected to found entities
    entity_ids = {e.id for e in result.entities}
    for relation in graph.relations.values():
        if relation.source_entity in entity_ids or relation.target_entity in entity_ids:
            result.relations.append(relation)

    return result


def _execute_query_spec(graph: KnowledgeGraph, spec: dict[str, Any]) -> QueryResult:
    """Execute a structured query specification."""
    result = QueryResult()
    query_type = spec.get("query_type", "entity")

    if query_type == "entity":
        entity_name = spec.get("entity_name")
        entity_type = spec.get("entity_type")

        if entity_name:
            result.entities = find_entities_by_name(graph, entity_name)
        elif entity_type:
            result.entities = find_entities_by_type(graph, entity_type)

    elif query_type == "relation":
        rel_type = spec.get("relation_type")
        if rel_type:
            result.relations = [
                r for r in graph.relations.values()
                if r.relation_type == rel_type
            ]

    elif query_type == "path":
        source_name = spec.get("source_name")
        target_name = spec.get("target_name")

        if source_name and target_name:
            sources = find_entities_by_name(graph, source_name)
            targets = find_entities_by_name(graph, target_name)

            if sources and targets:
                path = shortest_path(graph, sources[0].id, targets[0].id)
                if path:
                    result.paths.append(path)
                    result.entities = [graph.entities[eid] for eid in path]

    elif query_type == "pattern":
        pattern = spec.get("pattern", {})
        if pattern:
            result = pattern_match(graph, pattern)

    return result


def structured_query(
    graph: KnowledgeGraph,
    entity_type: str | None = None,
    relation_type: str | None = None,
    properties: dict[str, Any] | None = None,
    source_entity: str | None = None,
    target_entity: str | None = None,
) -> QueryResult:
    """Execute a structured query against the graph.

    Args:
        graph: The knowledge graph to query.
        entity_type: Filter entities by type.
        relation_type: Filter relations by type.
        properties: Filter by property values.
        source_entity: Filter relations by source.
        target_entity: Filter relations by target.

    Returns:
        QueryResult with matches.
    """
    result = query(graph, entity_type, relation_type, properties)

    # Additional filtering for relations
    if source_entity or target_entity:
        filtered_relations = []
        for rel in result.relations:
            if source_entity and rel.source_entity != source_entity:
                continue
            if target_entity and rel.target_entity != target_entity:
                continue
            filtered_relations.append(rel)
        result.relations = filtered_relations

    return result


def aggregate(
    graph: KnowledgeGraph,
    metric: str,
) -> dict[str, Any]:
    """Compute aggregate statistics on the graph.

    Args:
        graph: The knowledge graph.
        metric: Metric to compute:
            - "count": Entity and relation counts
            - "types": Distribution by type
            - "degree": Node degree distribution
            - "confidence": Confidence statistics
            - "centrality": Basic centrality measures

    Returns:
        Dictionary of computed values.
    """
    from .graph import aggregate as graph_aggregate

    if metric in ("count", "types", "degree", "density", "components"):
        return graph_aggregate(graph, metric)

    if metric == "confidence":
        entity_confs = [e.confidence for e in graph.entities.values()]
        relation_confs = [r.confidence for r in graph.relations.values()]

        def stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"min": 0, "max": 0, "avg": 0}
            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        return {
            "entity_confidence": stats(entity_confs),
            "relation_confidence": stats(relation_confs),
        }

    if metric == "centrality":
        # Compute simple degree centrality
        degree: dict[str, int] = {eid: 0 for eid in graph.entities}
        for rel in graph.relations.values():
            if rel.source_entity in degree:
                degree[rel.source_entity] += 1
            if rel.target_entity in degree:
                degree[rel.target_entity] += 1

        if not degree:
            return {"most_connected": [], "least_connected": []}

        max_degree = max(degree.values())
        min_degree = min(degree.values())

        most_connected = [
            graph.entities[eid].name
            for eid, d in degree.items()
            if d == max_degree
        ]
        least_connected = [
            graph.entities[eid].name
            for eid, d in degree.items()
            if d == min_degree
        ]

        return {
            "most_connected": most_connected,
            "least_connected": least_connected,
            "max_degree": max_degree,
            "min_degree": min_degree,
        }

    return {}


def export_cypher(graph: KnowledgeGraph) -> str:
    """Export the graph to Neo4j Cypher format.

    Args:
        graph: The knowledge graph.

    Returns:
        Cypher statements as a string.
    """
    statements = []

    # Create entities
    for entity in graph.entities.values():
        props = dict(entity.properties)
        props["name"] = entity.name
        props["_id"] = entity.id
        props["confidence"] = entity.confidence

        props_str = ", ".join(
            f"{k}: {json.dumps(v)}" for k, v in props.items()
        )
        label = entity.type.replace(" ", "_")
        statements.append(f"CREATE (n:{label} {{{props_str}}})")

    # Create relations
    for relation in graph.relations.values():
        props = dict(relation.properties)
        props["confidence"] = relation.confidence

        props_str = ""
        if props:
            props_str = " {" + ", ".join(
                f"{k}: {json.dumps(v)}" for k, v in props.items()
            ) + "}"

        rel_type = relation.relation_type.upper().replace(" ", "_")

        # Use MATCH to find nodes by _id property
        statements.append(
            f"MATCH (a {{_id: '{relation.source_entity}'}}), "
            f"(b {{_id: '{relation.target_entity}'}}) "
            f"CREATE (a)-[:{rel_type}{props_str}]->(b)"
        )

    return ";\n".join(statements) + ";"


def export_json(graph: KnowledgeGraph, pretty: bool = True) -> str:
    """Export the graph to JSON format.

    Args:
        graph: The knowledge graph.
        pretty: Whether to pretty-print the JSON.

    Returns:
        JSON string.
    """
    data = graph.to_dict()
    if pretty:
        return json.dumps(data, indent=2)
    return json.dumps(data)


def export_graphml(graph: KnowledgeGraph) -> str:
    """Export the graph to GraphML format.

    Args:
        graph: The knowledge graph.

    Returns:
        GraphML XML string.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="name" for="node" attr.name="name" attr.type="string"/>',
        '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="confidence" for="all" attr.name="confidence" attr.type="double"/>',
        '  <key id="relation_type" for="edge" attr.name="relation_type" attr.type="string"/>',
        '  <graph id="G" edgedefault="directed">',
    ]

    # Nodes
    for entity in graph.entities.values():
        lines.append(f'    <node id="{entity.id}">')
        lines.append(f'      <data key="name">{_xml_escape(entity.name)}</data>')
        lines.append(f'      <data key="type">{_xml_escape(entity.type)}</data>')
        lines.append(f'      <data key="confidence">{entity.confidence}</data>')
        lines.append('    </node>')

    # Edges
    for relation in graph.relations.values():
        lines.append(
            f'    <edge id="{relation.id}" '
            f'source="{relation.source_entity}" '
            f'target="{relation.target_entity}">'
        )
        lines.append(
            f'      <data key="relation_type">'
            f'{_xml_escape(relation.relation_type)}</data>'
        )
        lines.append(f'      <data key="confidence">{relation.confidence}</data>')
        lines.append('    </edge>')

    lines.append('  </graph>')
    lines.append('</graphml>')

    return '\n'.join(lines)


def export_rdf(graph: KnowledgeGraph, base_uri: str = "http://example.org/") -> str:
    """Export the graph to RDF Turtle format.

    Args:
        graph: The knowledge graph.
        base_uri: Base URI for identifiers.

    Returns:
        RDF Turtle string.
    """
    lines = [
        f"@prefix : <{base_uri}> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]

    # Entities
    for entity in graph.entities.values():
        entity_uri = f":entity_{entity.id.replace('-', '_')}"
        type_uri = f":{entity.type.replace(' ', '_')}"

        lines.append(f"{entity_uri}")
        lines.append(f"    rdf:type {type_uri} ;")
        lines.append(f'    rdfs:label "{_turtle_escape(entity.name)}" ;')
        lines.append(f"    :confidence {entity.confidence} .")
        lines.append("")

    # Relations
    for relation in graph.relations.values():
        source_uri = f":entity_{relation.source_entity.replace('-', '_')}"
        target_uri = f":entity_{relation.target_entity.replace('-', '_')}"
        rel_uri = f":{relation.relation_type.replace(' ', '_')}"

        lines.append(f"{source_uri} {rel_uri} {target_uri} .")

    return '\n'.join(lines)


def _xml_escape(s: str) -> str:
    """Escape special characters for XML."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _turtle_escape(s: str) -> str:
    """Escape special characters for RDF Turtle."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def find_related(
    graph: KnowledgeGraph,
    entity_name: str,
    max_depth: int = 2,
) -> QueryResult:
    """Find all entities related to a given entity.

    Args:
        graph: The knowledge graph.
        entity_name: Name of the starting entity.
        max_depth: Maximum distance to search.

    Returns:
        QueryResult with related entities and relations.
    """
    result = QueryResult()

    # Find starting entities
    starts = find_entities_by_name(graph, entity_name)
    if not starts:
        return result

    start_id = starts[0].id
    visited: set[str] = {start_id}
    frontier = [start_id]

    for _ in range(max_depth):
        new_frontier = []
        for entity_id in frontier:
            for neighbor in neighbors(graph, entity_id):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    new_frontier.append(neighbor.id)
                    result.entities.append(neighbor)
        frontier = new_frontier

    # Get relations between visited entities
    for relation in graph.relations.values():
        if relation.source_entity in visited and relation.target_entity in visited:
            result.relations.append(relation)

    return result


def find_path_query(
    graph: KnowledgeGraph,
    source_name: str,
    target_name: str,
    max_depth: int = 5,
) -> QueryResult:
    """Find paths between two entities.

    Args:
        graph: The knowledge graph.
        source_name: Name of source entity.
        target_name: Name of target entity.
        max_depth: Maximum path length.

    Returns:
        QueryResult with paths and involved entities.
    """
    result = QueryResult()

    sources = find_entities_by_name(graph, source_name)
    targets = find_entities_by_name(graph, target_name)

    if not sources or not targets:
        return result

    paths = all_paths(graph, sources[0].id, targets[0].id, max_depth)

    for path in paths:
        result.paths.append(path)
        for eid in path:
            entity = graph.entities.get(eid)
            if entity and entity not in result.entities:
                result.entities.append(entity)

    return result
