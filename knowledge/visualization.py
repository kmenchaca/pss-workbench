"""Graph visualization for the Emergent Knowledge Graph system.

This module provides functions for rendering knowledge graphs
in various formats including ASCII, GraphViz DOT, and D3.js JSON.
"""

from __future__ import annotations

import json
from typing import Any

from .graph import neighbors, shortest_path
from .types import Entity, KnowledgeGraph, Relation


def render_ascii(
    graph: KnowledgeGraph,
    max_entities: int = 20,
    show_relations: bool = True,
) -> str:
    """Render the graph as ASCII text.

    Args:
        graph: The knowledge graph to render.
        max_entities: Maximum number of entities to show.
        show_relations: Whether to show relations.

    Returns:
        ASCII representation of the graph.
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append(f"Knowledge Graph: {graph.id}")
    lines.append(f"Entities: {len(graph.entities)} | Relations: {len(graph.relations)}")
    lines.append("=" * 60)
    lines.append("")

    # Group entities by type
    by_type: dict[str, list[Entity]] = {}
    for entity in graph.entities.values():
        if entity.type not in by_type:
            by_type[entity.type] = []
        by_type[entity.type].append(entity)

    # Render entities by type
    entity_count = 0
    for etype, entities in sorted(by_type.items()):
        lines.append(f"[{etype}]")
        for entity in sorted(entities, key=lambda e: e.name)[:max_entities - entity_count]:
            conf_bar = _confidence_bar(entity.confidence)
            lines.append(f"  {conf_bar} {entity.name}")

            # Show key properties
            if entity.properties:
                props = ", ".join(
                    f"{k}={v}" for k, v in list(entity.properties.items())[:3]
                )
                lines.append(f"       {props}")

            entity_count += 1
            if entity_count >= max_entities:
                break
        lines.append("")

        if entity_count >= max_entities:
            remaining = len(graph.entities) - max_entities
            if remaining > 0:
                lines.append(f"  ... and {remaining} more entities")
            break

    # Render relations
    if show_relations and graph.relations:
        lines.append("")
        lines.append("-" * 60)
        lines.append("Relations:")
        lines.append("-" * 60)

        # Build entity name lookup
        names = {e.id: e.name for e in graph.entities.values()}

        for relation in list(graph.relations.values())[:max_entities]:
            source = names.get(relation.source_entity, "?")
            target = names.get(relation.target_entity, "?")
            arrow = "<-->" if relation.bidirectional else "-->"

            lines.append(
                f"  {source} {arrow}[{relation.relation_type}]{arrow} {target}"
            )

        remaining = len(graph.relations) - max_entities
        if remaining > 0:
            lines.append(f"  ... and {remaining} more relations")

    return "\n".join(lines)


def _confidence_bar(confidence: float, width: int = 5) -> str:
    """Create a visual confidence bar."""
    filled = int(confidence * width)
    empty = width - filled
    return "[" + "#" * filled + "-" * empty + "]"


def render_tree(
    graph: KnowledgeGraph,
    root_id: str,
    relation_types: list[str] | None = None,
    max_depth: int = 4,
) -> str:
    """Render the graph as a tree starting from a root entity.

    Args:
        graph: The knowledge graph.
        root_id: ID of the root entity.
        relation_types: Relation types to follow (None = all).
        max_depth: Maximum depth to render.

    Returns:
        ASCII tree representation.
    """
    if root_id not in graph.entities:
        return f"Entity {root_id} not found"

    lines: list[str] = []
    visited: set[str] = set()

    def render_node(entity_id: str, prefix: str, depth: int) -> None:
        if depth > max_depth or entity_id in visited:
            return

        visited.add(entity_id)
        entity = graph.entities.get(entity_id)
        if not entity:
            return

        lines.append(f"{prefix}{entity.name} ({entity.type})")

        # Get children
        children = []
        for relation in graph.relations.values():
            if relation.source_entity == entity_id:
                if relation_types is None or relation.relation_type in relation_types:
                    children.append((relation.target_entity, relation.relation_type))

        # Render children
        for i, (child_id, rel_type) in enumerate(children):
            is_last = i == len(children) - 1
            connector = "\\-- " if is_last else "|-- "
            child_prefix = prefix[:-4] + ("    " if is_last else "|   ") if prefix else ""

            if child_id in visited:
                child_name = graph.entities.get(child_id, Entity()).name
                lines.append(f"{prefix}{connector}[{rel_type}] {child_name} (circular)")
            else:
                lines.append(f"{prefix}{connector}[{rel_type}]")
                render_node(child_id, prefix + ("    " if is_last else "|   "), depth + 1)

    render_node(root_id, "", 0)
    return "\n".join(lines)


def to_dot(
    graph: KnowledgeGraph,
    title: str = "Knowledge Graph",
    show_properties: bool = False,
    color_by_type: bool = True,
) -> str:
    """Convert the graph to GraphViz DOT format.

    Args:
        graph: The knowledge graph.
        title: Title for the graph.
        show_properties: Whether to show entity properties.
        color_by_type: Whether to color nodes by type.

    Returns:
        DOT format string.
    """
    lines = [
        "digraph KnowledgeGraph {",
        f'    label="{title}";',
        "    rankdir=LR;",
        "    node [shape=box, style=filled];",
        "",
    ]

    # Generate colors for types
    colors = [
        "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9",
        "#BAE1FF", "#E8BAFF", "#FFB3DE", "#C4BAFF",
    ]
    type_colors: dict[str, str] = {}
    for entity in graph.entities.values():
        if entity.type not in type_colors:
            type_colors[entity.type] = colors[len(type_colors) % len(colors)]

    # Nodes
    for entity in graph.entities.values():
        label = entity.name
        if show_properties and entity.properties:
            props = "\\n".join(f"{k}: {v}" for k, v in list(entity.properties.items())[:3])
            label = f"{entity.name}\\n---\\n{props}"

        color = type_colors[entity.type] if color_by_type else "#FFFFFF"
        node_id = _dot_safe_id(entity.id)

        lines.append(
            f'    {node_id} [label="{label}", fillcolor="{color}", '
            f'tooltip="{entity.type}"];'
        )

    lines.append("")

    # Edges
    for relation in graph.relations.values():
        source = _dot_safe_id(relation.source_entity)
        target = _dot_safe_id(relation.target_entity)
        label = relation.relation_type

        style = 'dir="both"' if relation.bidirectional else ""

        lines.append(f'    {source} -> {target} [label="{label}" {style}];')

    lines.append("}")
    return "\n".join(lines)


def _dot_safe_id(id_str: str) -> str:
    """Convert an ID to a safe DOT identifier."""
    return "n" + id_str.replace("-", "_")


def to_d3_json(
    graph: KnowledgeGraph,
    include_confidence: bool = True,
) -> str:
    """Convert the graph to D3.js force-directed graph JSON format.

    Args:
        graph: The knowledge graph.
        include_confidence: Whether to include confidence values.

    Returns:
        JSON string for D3.js visualization.
    """
    # Build node index
    node_index = {entity.id: i for i, entity in enumerate(graph.entities.values())}

    nodes = []
    for entity in graph.entities.values():
        node: dict[str, Any] = {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "group": hash(entity.type) % 10,  # For coloring
        }
        if include_confidence:
            node["confidence"] = entity.confidence
        if entity.properties:
            node["properties"] = entity.properties
        nodes.append(node)

    links = []
    for relation in graph.relations.values():
        if relation.source_entity not in node_index or relation.target_entity not in node_index:
            continue

        link: dict[str, Any] = {
            "source": node_index[relation.source_entity],
            "target": node_index[relation.target_entity],
            "type": relation.relation_type,
        }
        if include_confidence:
            link["confidence"] = relation.confidence
        if relation.bidirectional:
            link["bidirectional"] = True
        links.append(link)

    return json.dumps({"nodes": nodes, "links": links}, indent=2)


def to_vis_network_json(graph: KnowledgeGraph) -> str:
    """Convert the graph to vis.js network JSON format.

    Args:
        graph: The knowledge graph.

    Returns:
        JSON string for vis.js visualization.
    """
    nodes = []
    for entity in graph.entities.values():
        nodes.append({
            "id": entity.id,
            "label": entity.name,
            "title": f"{entity.type}\\nConfidence: {entity.confidence:.2f}",
            "group": entity.type,
        })

    edges = []
    for relation in graph.relations.values():
        edge: dict[str, Any] = {
            "from": relation.source_entity,
            "to": relation.target_entity,
            "label": relation.relation_type,
            "title": f"Confidence: {relation.confidence:.2f}",
        }
        if relation.bidirectional:
            edge["arrows"] = "to, from"
        else:
            edge["arrows"] = "to"
        edges.append(edge)

    return json.dumps({"nodes": nodes, "edges": edges}, indent=2)


def highlight_path(
    graph: KnowledgeGraph,
    path: list[str],
    format: str = "ascii",
) -> str:
    """Visualize a specific path in the graph.

    Args:
        graph: The knowledge graph.
        path: List of entity IDs forming the path.
        format: Output format ("ascii", "dot").

    Returns:
        Visualization string with path highlighted.
    """
    if format == "ascii":
        return _highlight_path_ascii(graph, path)
    elif format == "dot":
        return _highlight_path_dot(graph, path)
    else:
        return _highlight_path_ascii(graph, path)


def _highlight_path_ascii(graph: KnowledgeGraph, path: list[str]) -> str:
    """Render a path in ASCII."""
    if not path:
        return "Empty path"

    lines = ["Path:"]
    path_set = set(path)

    for i, entity_id in enumerate(path):
        entity = graph.entities.get(entity_id)
        if not entity:
            continue

        lines.append(f"  [{i+1}] {entity.name} ({entity.type})")

        # Find relation to next entity
        if i < len(path) - 1:
            next_id = path[i + 1]
            for relation in graph.relations.values():
                if relation.source_entity == entity_id and relation.target_entity == next_id:
                    arrow = "<-->" if relation.bidirectional else "  |"
                    lines.append(f"      {arrow} [{relation.relation_type}]")
                    break
                elif relation.target_entity == entity_id and relation.source_entity == next_id:
                    lines.append(f"      | [{relation.relation_type}]")
                    break
            else:
                lines.append("      | (no direct relation)")

    return "\n".join(lines)


def _highlight_path_dot(graph: KnowledgeGraph, path: list[str]) -> str:
    """Render graph with path highlighted in DOT format."""
    path_set = set(path)
    path_edges = set()

    # Find edges in path
    for i in range(len(path) - 1):
        path_edges.add((path[i], path[i + 1]))
        path_edges.add((path[i + 1], path[i]))  # For bidirectional

    lines = [
        "digraph KnowledgeGraph {",
        '    label="Path Highlighted";',
        "    rankdir=LR;",
        "",
    ]

    # Nodes
    for entity in graph.entities.values():
        node_id = _dot_safe_id(entity.id)
        color = "#FFD700" if entity.id in path_set else "#FFFFFF"
        penwidth = "3" if entity.id in path_set else "1"

        lines.append(
            f'    {node_id} [label="{entity.name}", style=filled, '
            f'fillcolor="{color}", penwidth={penwidth}];'
        )

    lines.append("")

    # Edges
    for relation in graph.relations.values():
        source = _dot_safe_id(relation.source_entity)
        target = _dot_safe_id(relation.target_entity)

        is_path_edge = (
            (relation.source_entity, relation.target_entity) in path_edges or
            (relation.target_entity, relation.source_entity) in path_edges
        )

        color = "#FF0000" if is_path_edge else "#000000"
        penwidth = "3" if is_path_edge else "1"

        lines.append(
            f'    {source} -> {target} [label="{relation.relation_type}", '
            f'color="{color}", penwidth={penwidth}];'
        )

    lines.append("}")
    return "\n".join(lines)


def summary_stats(graph: KnowledgeGraph) -> str:
    """Generate a summary statistics report.

    Args:
        graph: The knowledge graph.

    Returns:
        Formatted summary string.
    """
    lines = [
        "Knowledge Graph Summary",
        "=" * 40,
        "",
        f"Total Entities: {len(graph.entities)}",
        f"Total Relations: {len(graph.relations)}",
        "",
    ]

    # Entity types
    type_counts: dict[str, int] = {}
    for entity in graph.entities.values():
        type_counts[entity.type] = type_counts.get(entity.type, 0) + 1

    lines.append("Entity Types:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {etype}: {count}")

    lines.append("")

    # Relation types
    rel_counts: dict[str, int] = {}
    for relation in graph.relations.values():
        rel_counts[relation.relation_type] = rel_counts.get(relation.relation_type, 0) + 1

    lines.append("Relation Types:")
    for rtype, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {rtype}: {count}")

    lines.append("")

    # Confidence stats
    entity_confs = [e.confidence for e in graph.entities.values()]
    if entity_confs:
        lines.append("Entity Confidence:")
        lines.append(f"  Min: {min(entity_confs):.2f}")
        lines.append(f"  Max: {max(entity_confs):.2f}")
        lines.append(f"  Avg: {sum(entity_confs)/len(entity_confs):.2f}")

    return "\n".join(lines)


def to_mermaid(graph: KnowledgeGraph, max_nodes: int = 30) -> str:
    """Convert the graph to Mermaid diagram format.

    Args:
        graph: The knowledge graph.
        max_nodes: Maximum number of nodes to include.

    Returns:
        Mermaid diagram string.
    """
    lines = ["graph LR"]

    # Limit nodes
    entities = list(graph.entities.values())[:max_nodes]
    entity_ids = {e.id for e in entities}

    # Create safe IDs
    def safe_id(id_str: str) -> str:
        return "n" + id_str.replace("-", "")[:8]

    # Nodes with styling
    type_styles: dict[str, str] = {}
    style_index = 0

    for entity in entities:
        nid = safe_id(entity.id)
        label = entity.name.replace('"', "'")
        lines.append(f'    {nid}["{label}"]')

        if entity.type not in type_styles:
            type_styles[entity.type] = f"style{style_index}"
            style_index += 1

    lines.append("")

    # Edges
    for relation in graph.relations.values():
        if relation.source_entity not in entity_ids or relation.target_entity not in entity_ids:
            continue

        source = safe_id(relation.source_entity)
        target = safe_id(relation.target_entity)
        label = relation.relation_type.replace(" ", "_")

        if relation.bidirectional:
            lines.append(f"    {source} <-->|{label}| {target}")
        else:
            lines.append(f"    {source} -->|{label}| {target}")

    return "\n".join(lines)
