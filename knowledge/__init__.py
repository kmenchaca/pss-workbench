"""Emergent Knowledge Graph system for structured knowledge extraction.

This module provides a complete system for building knowledge graphs
from text using branching exploration, entity extraction, graph merging,
and consistency checking.

Example usage:
    from knowledge import (
        KnowledgeHarness,
        create_harness,
        GENERAL,
        TECHNICAL,
    )

    # Create a harness with an LLM function
    harness = create_harness(
        schema="GENERAL",
        llm_call=my_llm_function,
    )

    # Run extraction
    result = harness.run(text, num_branches=3)

    # Access the unified graph
    graph = result.unified_graph
    print(f"Extracted {graph.entity_count()} entities")

    # Query the graph
    from knowledge import natural_language_query
    results = natural_language_query(graph, "Who works at Company X?")

    # Visualize
    from knowledge import render_ascii, to_dot
    print(render_ascii(graph))
"""

# Core types
from .types import (
    ConflictType,
    Entity,
    ExtractionResult,
    GraphSchema,
    KnowledgeGraph,
    MergeConflict,
    ProvenanceRecord,
    QueryResult,
    Relation,
    ResolutionStrategy,
)

# Schema
from .schema import (
    GENERAL,
    ORGANIZATIONAL,
    SCIENTIFIC,
    TECHNICAL,
    SchemaBuilder,
    get_schema,
    validate_entity,
    validate_relation,
)

# Graph operations
from .graph import (
    add_entity,
    add_relation,
    aggregate,
    all_paths,
    copy_graph,
    find_entities_by_name,
    find_entities_by_type,
    from_json,
    get_entity,
    get_relation,
    neighbors,
    pattern_match,
    query,
    remove_entity,
    remove_relation,
    shortest_path,
    subgraph,
    to_json,
)

# Extraction
from .extraction import (
    batch_extract,
    confidence_score,
    extract_entities,
    extract_properties,
    extract_relations,
    full_extraction,
    merge_extraction_results,
)

# Resolution
from .resolution import (
    cluster_similar_entities,
    deduplicate_graph,
    entity_similarity,
    find_duplicates,
    merge_entities,
    resolve_conflicts,
)

# Merging
from .merge import (
    diff_graphs,
    edge_conflict_resolution,
    incremental_merge,
    merge_graphs,
    merge_with_provenance,
    merge_with_voting,
)

# Consistency
from .consistency import (
    ConsistencyIssue,
    ConsistencyReport,
    check_cardinality,
    check_consistency,
    check_reachability,
    check_required_relations,
    check_uniqueness,
    find_contradictions,
    full_consistency_check,
    type_check,
)

# Provenance
from .provenance import (
    ProvenanceTracker,
    SourceReliability,
    audit_trail,
    confidence_propagation,
    filter_by_confidence,
    filter_by_source,
    source_reliability,
    tag_provenance,
    trace_provenance,
    weighted_confidence,
)

# Harness
from .harness import (
    BranchConfig,
    BranchResult,
    HarnessResult,
    KnowledgeHarness,
    create_harness,
    simple_extraction,
)

# Query
from .query import (
    aggregate as query_aggregate,
    export_cypher,
    export_graphml,
    export_json,
    export_rdf,
    find_path_query,
    find_related,
    natural_language_query,
    structured_query,
)

# Visualization
from .visualization import (
    highlight_path,
    render_ascii,
    render_tree,
    summary_stats,
    to_d3_json,
    to_dot,
    to_mermaid,
    to_vis_network_json,
)

__version__ = "1.0.0"

__all__ = [
    # Types
    "ConflictType",
    "Entity",
    "ExtractionResult",
    "GraphSchema",
    "KnowledgeGraph",
    "MergeConflict",
    "ProvenanceRecord",
    "QueryResult",
    "Relation",
    "ResolutionStrategy",
    # Schema
    "GENERAL",
    "ORGANIZATIONAL",
    "SCIENTIFIC",
    "TECHNICAL",
    "SchemaBuilder",
    "get_schema",
    "validate_entity",
    "validate_relation",
    # Graph operations
    "add_entity",
    "add_relation",
    "aggregate",
    "all_paths",
    "copy_graph",
    "find_entities_by_name",
    "find_entities_by_type",
    "from_json",
    "get_entity",
    "get_relation",
    "neighbors",
    "pattern_match",
    "query",
    "remove_entity",
    "remove_relation",
    "shortest_path",
    "subgraph",
    "to_json",
    # Extraction
    "batch_extract",
    "confidence_score",
    "extract_entities",
    "extract_properties",
    "extract_relations",
    "full_extraction",
    "merge_extraction_results",
    # Resolution
    "cluster_similar_entities",
    "deduplicate_graph",
    "entity_similarity",
    "find_duplicates",
    "merge_entities",
    "resolve_conflicts",
    # Merging
    "diff_graphs",
    "edge_conflict_resolution",
    "incremental_merge",
    "merge_graphs",
    "merge_with_provenance",
    "merge_with_voting",
    # Consistency
    "ConsistencyIssue",
    "ConsistencyReport",
    "check_cardinality",
    "check_consistency",
    "check_reachability",
    "check_required_relations",
    "check_uniqueness",
    "find_contradictions",
    "full_consistency_check",
    "type_check",
    # Provenance
    "ProvenanceTracker",
    "SourceReliability",
    "audit_trail",
    "confidence_propagation",
    "filter_by_confidence",
    "filter_by_source",
    "source_reliability",
    "tag_provenance",
    "trace_provenance",
    "weighted_confidence",
    # Harness
    "BranchConfig",
    "BranchResult",
    "HarnessResult",
    "KnowledgeHarness",
    "create_harness",
    "simple_extraction",
    # Query
    "query_aggregate",
    "export_cypher",
    "export_graphml",
    "export_json",
    "export_rdf",
    "find_path_query",
    "find_related",
    "natural_language_query",
    "structured_query",
    # Visualization
    "highlight_path",
    "render_ascii",
    "render_tree",
    "summary_stats",
    "to_d3_json",
    "to_dot",
    "to_mermaid",
    "to_vis_network_json",
]
