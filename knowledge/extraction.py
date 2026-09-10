"""Entity and relation extraction for the Emergent Knowledge Graph system.

This module provides functions for extracting entities and relations
from text using LLM calls, with structured output parsing and
confidence scoring.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable

from .types import Entity, ExtractionResult, GraphSchema, Relation


# Type alias for LLM call function
LLMCallFn = Callable[[str, str | None], str]


def extract_entities(
    text: str,
    schema: GraphSchema,
    llm_call: LLMCallFn,
    branch_id: str | None = None,
) -> list[Entity]:
    """Extract entities from text using an LLM.

    Args:
        text: The text to extract entities from.
        schema: The graph schema defining valid entity types.
        llm_call: Function to call the LLM (prompt, system_prompt) -> response.
        branch_id: ID of the branch performing extraction.

    Returns:
        List of extracted entities.
    """
    # Build the prompt with schema information
    type_names = list(schema.node_types.keys())
    type_descriptions = []
    for name, node_type in schema.node_types.items():
        desc = f"- {name}: {node_type.description or 'No description'}"
        if node_type.required_properties:
            desc += f" (required: {', '.join(node_type.required_properties)})"
        type_descriptions.append(desc)

    system_prompt = """You are an entity extraction system. Extract entities from the given text.
For each entity, provide:
- name: The entity's name
- type: One of the allowed types
- properties: A JSON object of relevant properties
- confidence: A score from 0.0 to 1.0

Respond with a JSON array of entities. Example:
[
  {"name": "John Smith", "type": "Person", "properties": {"occupation": "engineer"}, "confidence": 0.95}
]"""

    prompt = f"""Extract entities from the following text.

Allowed entity types:
{chr(10).join(type_descriptions)}

Text:
{text}

Respond only with a JSON array of entities."""

    response = llm_call(prompt, system_prompt)
    entities = parse_entity_response(response, schema, branch_id)

    return entities


def parse_entity_response(
    response: str,
    schema: GraphSchema,
    branch_id: str | None = None,
) -> list[Entity]:
    """Parse an LLM response into Entity objects.

    Args:
        response: The LLM response text.
        schema: The graph schema for validation.
        branch_id: ID of the branch that performed extraction.

    Returns:
        List of parsed Entity objects.
    """
    entities = []

    # Try to extract JSON from the response
    json_str = extract_json_from_response(response)
    if not json_str:
        return entities

    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            data = [data]

        for item in data:
            if not isinstance(item, dict):
                continue

            entity_type = item.get("type", "Entity")
            # Validate type against schema
            if entity_type not in schema.node_types:
                entity_type = "Entity"

            entity = Entity(
                id=str(uuid.uuid4()),
                name=item.get("name", ""),
                type=entity_type,
                properties=item.get("properties", {}),
                source_branch=branch_id,
                confidence=float(item.get("confidence", 0.8)),
                aliases=item.get("aliases", []),
            )
            entities.append(entity)

    except json.JSONDecodeError:
        pass

    return entities


def extract_relations(
    text: str,
    entities: list[Entity],
    schema: GraphSchema,
    llm_call: LLMCallFn,
    branch_id: str | None = None,
) -> list[Relation]:
    """Extract relations between entities from text.

    Args:
        text: The text to extract relations from.
        entities: Previously extracted entities.
        schema: The graph schema defining valid relation types.
        llm_call: Function to call the LLM.
        branch_id: ID of the branch performing extraction.

    Returns:
        List of extracted relations.
    """
    if not entities:
        return []

    # Build entity reference list
    entity_list = []
    for i, e in enumerate(entities):
        entity_list.append(f"{i}: {e.name} ({e.type})")

    # Build relation type descriptions
    relation_descriptions = []
    for name, edge_type in schema.edge_types.items():
        desc = f"- {name}: {edge_type.description or 'No description'}"
        if edge_type.source_types:
            desc += f" (from: {', '.join(edge_type.source_types)})"
        if edge_type.target_types:
            desc += f" (to: {', '.join(edge_type.target_types)})"
        relation_descriptions.append(desc)

    system_prompt = """You are a relation extraction system. Extract relationships between entities.
For each relation, provide:
- source: Index of the source entity
- target: Index of the target entity
- type: One of the allowed relation types
- properties: A JSON object of relevant properties
- confidence: A score from 0.0 to 1.0

Respond with a JSON array of relations. Example:
[
  {"source": 0, "target": 1, "type": "works_for", "properties": {}, "confidence": 0.9}
]"""

    prompt = f"""Extract relationships between the following entities based on the text.

Entities:
{chr(10).join(entity_list)}

Allowed relation types:
{chr(10).join(relation_descriptions)}

Text:
{text}

Respond only with a JSON array of relations."""

    response = llm_call(prompt, system_prompt)
    relations = parse_relation_response(response, entities, schema, branch_id)

    return relations


def parse_relation_response(
    response: str,
    entities: list[Entity],
    schema: GraphSchema,
    branch_id: str | None = None,
) -> list[Relation]:
    """Parse an LLM response into Relation objects.

    Args:
        response: The LLM response text.
        entities: List of entities to reference by index.
        schema: The graph schema for validation.
        branch_id: ID of the branch that performed extraction.

    Returns:
        List of parsed Relation objects.
    """
    relations = []

    json_str = extract_json_from_response(response)
    if not json_str:
        return relations

    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            data = [data]

        for item in data:
            if not isinstance(item, dict):
                continue

            source_idx = item.get("source")
            target_idx = item.get("target")

            # Validate indices
            if not isinstance(source_idx, int) or not isinstance(target_idx, int):
                continue
            if source_idx < 0 or source_idx >= len(entities):
                continue
            if target_idx < 0 or target_idx >= len(entities):
                continue

            relation_type = item.get("type", "related_to")
            # Validate type against schema
            if relation_type not in schema.edge_types:
                relation_type = "related_to"

            # Check if bidirectional
            bidirectional = False
            if relation_type in schema.edge_types:
                bidirectional = schema.edge_types[relation_type].bidirectional

            relation = Relation(
                id=str(uuid.uuid4()),
                source_entity=entities[source_idx].id,
                target_entity=entities[target_idx].id,
                relation_type=relation_type,
                properties=item.get("properties", {}),
                source_branch=branch_id,
                confidence=float(item.get("confidence", 0.8)),
                bidirectional=bidirectional,
            )
            relations.append(relation)

    except json.JSONDecodeError:
        pass

    return relations


def extract_properties(
    entity: Entity,
    text: str,
    llm_call: LLMCallFn,
) -> dict[str, Any]:
    """Extract additional properties for an entity from text.

    Args:
        entity: The entity to find properties for.
        text: The text to search in.
        llm_call: Function to call the LLM.

    Returns:
        Dictionary of extracted properties.
    """
    system_prompt = """You are a property extraction system. Extract properties for the given entity.
Respond with a JSON object of property names to values. Example:
{"occupation": "engineer", "age": 35, "location": "New York"}"""

    prompt = f"""Extract properties for the entity "{entity.name}" ({entity.type}) from the following text.

Text:
{text}

Respond only with a JSON object of properties."""

    response = llm_call(prompt, system_prompt)

    json_str = extract_json_from_response(response)
    if not json_str:
        return {}

    try:
        properties = json.loads(json_str)
        if isinstance(properties, dict):
            return properties
    except json.JSONDecodeError:
        pass

    return {}


def confidence_score(extraction: ExtractionResult) -> float:
    """Calculate overall confidence score for an extraction result.

    Args:
        extraction: The extraction result to score.

    Returns:
        Aggregate confidence score (0.0 to 1.0).
    """
    if not extraction.entities and not extraction.relations:
        return 0.0

    scores = []

    # Entity confidences
    for entity in extraction.entities:
        scores.append(entity.confidence)

    # Relation confidences
    for relation in extraction.relations:
        scores.append(relation.confidence)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def extract_json_from_response(response: str) -> str | None:
    """Extract JSON from an LLM response that may contain other text.

    Args:
        response: The LLM response text.

    Returns:
        The extracted JSON string, or None if not found.
    """
    # Try to find JSON array
    array_match = re.search(r'\[[\s\S]*\]', response)
    if array_match:
        return array_match.group(0)

    # Try to find JSON object
    object_match = re.search(r'\{[\s\S]*\}', response)
    if object_match:
        return object_match.group(0)

    return None


def full_extraction(
    text: str,
    schema: GraphSchema,
    llm_call: LLMCallFn,
    branch_id: str | None = None,
) -> ExtractionResult:
    """Perform full entity and relation extraction from text.

    Args:
        text: The text to extract from.
        schema: The graph schema to use.
        llm_call: Function to call the LLM.
        branch_id: ID of the branch performing extraction.

    Returns:
        ExtractionResult with entities and relations.
    """
    # Extract entities first
    entities = extract_entities(text, schema, llm_call, branch_id)

    # Extract relations between entities
    relations = extract_relations(text, entities, schema, llm_call, branch_id)

    # Optionally enrich entity properties
    for entity in entities:
        additional_props = extract_properties(entity, text, llm_call)
        entity.properties.update(additional_props)

    result = ExtractionResult(
        entities=entities,
        relations=relations,
        provenance={"source_text_length": len(text)},
        branch_id=branch_id,
        raw_text=text,
    )
    result.confidence = confidence_score(result)

    return result


def extract_with_context(
    text: str,
    context: str,
    schema: GraphSchema,
    llm_call: LLMCallFn,
    branch_id: str | None = None,
) -> ExtractionResult:
    """Extract entities and relations with additional context.

    Args:
        text: The primary text to extract from.
        context: Additional context about the domain.
        schema: The graph schema to use.
        llm_call: Function to call the LLM.
        branch_id: ID of the branch performing extraction.

    Returns:
        ExtractionResult with entities and relations.
    """
    # Prepend context to the extraction
    combined_text = f"Context: {context}\n\nText to analyze:\n{text}"
    return full_extraction(combined_text, schema, llm_call, branch_id)


def batch_extract(
    texts: list[str],
    schema: GraphSchema,
    llm_call: LLMCallFn,
    branch_id: str | None = None,
) -> list[ExtractionResult]:
    """Extract from multiple texts.

    Args:
        texts: List of texts to extract from.
        schema: The graph schema to use.
        llm_call: Function to call the LLM.
        branch_id: ID of the branch performing extraction.

    Returns:
        List of ExtractionResults.
    """
    results = []
    for text in texts:
        result = full_extraction(text, schema, llm_call, branch_id)
        results.append(result)
    return results


def merge_extraction_results(results: list[ExtractionResult]) -> ExtractionResult:
    """Merge multiple extraction results into one.

    Note: This is a simple merge without deduplication.
    Use the resolution module for proper entity resolution.

    Args:
        results: List of ExtractionResults to merge.

    Returns:
        Combined ExtractionResult.
    """
    all_entities = []
    all_relations = []
    all_provenance: dict[str, Any] = {"sources": []}

    for result in results:
        all_entities.extend(result.entities)
        all_relations.extend(result.relations)
        if result.branch_id:
            all_provenance["sources"].append(result.branch_id)

    merged = ExtractionResult(
        entities=all_entities,
        relations=all_relations,
        provenance=all_provenance,
    )
    merged.confidence = confidence_score(merged)

    return merged
