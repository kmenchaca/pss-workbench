"""Main knowledge harness for the Emergent Knowledge Graph system.

This module provides the KnowledgeHarness class that orchestrates
branching exploration for knowledge graph construction.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .consistency import full_consistency_check
from .extraction import full_extraction
from .graph import add_entity, add_relation
from .merge import merge_graphs, merge_with_provenance
from .provenance import ProvenanceTracker, tag_provenance
from .resolution import deduplicate_graph, entity_similarity
from .schema import GENERAL, GraphSchema
from .types import (
    Entity,
    ExtractionResult,
    KnowledgeGraph,
    MergeConflict,
    Relation,
)


# Type aliases
LLMCallFn = Callable[[str, str | None], str]
EmbeddingFn = Callable[[str], list[float]]


@dataclass
class BranchConfig:
    """Configuration for a knowledge extraction branch.

    Attributes:
        focus_area: What this branch should focus on.
        depth: How deep to explore (affects extraction prompts).
        entity_types: Entity types to prioritize.
        relation_types: Relation types to prioritize.
    """

    focus_area: str = ""
    depth: str = "normal"  # shallow, normal, deep
    entity_types: list[str] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)


@dataclass
class BranchResult:
    """Result from a single extraction branch.

    Attributes:
        branch_id: Unique identifier for the branch.
        config: Branch configuration used.
        extraction: Extraction result from this branch.
        graph: Knowledge graph built by this branch.
        metadata: Additional branch metadata.
    """

    branch_id: str = ""
    config: BranchConfig = field(default_factory=BranchConfig)
    extraction: ExtractionResult | None = None
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessResult:
    """Result from a complete knowledge harness run.

    Attributes:
        unified_graph: The merged knowledge graph.
        branch_results: Results from each branch.
        conflicts: Conflicts encountered during merging.
        provenance: Provenance tracking data.
        consistency_report: Consistency check results.
        metadata: Run metadata.
    """

    unified_graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    branch_results: list[BranchResult] = field(default_factory=list)
    conflicts: list[MergeConflict] = field(default_factory=list)
    provenance: ProvenanceTracker = field(default_factory=ProvenanceTracker)
    consistency_report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeHarness:
    """Orchestrates branching exploration for knowledge graph construction.

    The harness spawns multiple branches to explore different aspects of
    a text, extracts entities and relations in parallel, then merges
    the results into a unified knowledge graph.
    """

    def __init__(
        self,
        schema: GraphSchema | None = None,
        llm_call: LLMCallFn | None = None,
        embedding_fn: EmbeddingFn | None = None,
        similarity_threshold: float = 0.8,
        min_confidence: float = 0.3,
    ) -> None:
        """Initialize the knowledge harness.

        Args:
            schema: Graph schema to use (defaults to GENERAL).
            llm_call: Function to call the LLM.
            embedding_fn: Optional function for embeddings.
            similarity_threshold: Threshold for entity matching.
            min_confidence: Minimum confidence to include extractions.
        """
        self.schema = schema or GENERAL
        self.llm_call = llm_call
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence

        self._branches: list[BranchConfig] = []
        self._provenance = ProvenanceTracker()

    def configure_branches(self, text: str, num_branches: int = 3) -> list[BranchConfig]:
        """Configure branches for exploring a text.

        This creates branch configurations that ensure diverse coverage
        of the text's content.

        Args:
            text: The text to explore.
            num_branches: Number of branches to create.

        Returns:
            List of branch configurations.
        """
        branches = []

        # Default branch focuses
        focuses = [
            "entities and their properties",
            "relationships between entities",
            "hierarchical structures and categories",
            "temporal information and events",
            "causal relationships and processes",
        ]

        for i in range(min(num_branches, len(focuses))):
            branch = BranchConfig(
                focus_area=focuses[i],
                depth="normal" if i < num_branches - 1 else "deep",
            )
            branches.append(branch)

        self._branches = branches
        return branches

    def spawn_branch(
        self,
        text: str,
        config: BranchConfig,
    ) -> BranchResult:
        """Spawn a single extraction branch.

        Args:
            text: The text to extract from.
            config: Branch configuration.

        Returns:
            BranchResult with extraction and graph.
        """
        if not self.llm_call:
            raise ValueError("LLM call function not configured")

        branch_id = str(uuid.uuid4())[:8]

        # Create focused extraction prompt based on config
        focus_prompt = f"Focus on extracting {config.focus_area}."
        if config.entity_types:
            focus_prompt += f" Prioritize entity types: {', '.join(config.entity_types)}."
        if config.relation_types:
            focus_prompt += f" Prioritize relation types: {', '.join(config.relation_types)}."

        # Modify LLM call to include focus
        def focused_llm_call(prompt: str, system: str | None) -> str:
            enhanced_system = system or ""
            enhanced_system += f"\n\n{focus_prompt}"
            return self.llm_call(prompt, enhanced_system)

        # Perform extraction
        extraction = full_extraction(
            text=text,
            schema=self.schema,
            llm_call=focused_llm_call,
            branch_id=branch_id,
        )

        # Filter by minimum confidence
        extraction.entities = [
            e for e in extraction.entities
            if e.confidence >= self.min_confidence
        ]
        extraction.relations = [
            r for r in extraction.relations
            if r.confidence >= self.min_confidence
        ]

        # Build branch graph
        graph = KnowledgeGraph(id=f"branch_{branch_id}")
        for entity in extraction.entities:
            graph.entities[entity.id] = entity

        for relation in extraction.relations:
            # Validate relation endpoints exist
            if (relation.source_entity in graph.entities and
                relation.target_entity in graph.entities):
                graph.relations[relation.id] = relation

        # Track provenance
        for entity in extraction.entities:
            tag_provenance(entity, f"branch_{branch_id}", self._provenance)
        for relation in extraction.relations:
            if relation.id in graph.relations:
                tag_provenance(relation, f"branch_{branch_id}", self._provenance)

        return BranchResult(
            branch_id=branch_id,
            config=config,
            extraction=extraction,
            graph=graph,
            metadata={
                "entity_count": len(extraction.entities),
                "relation_count": len(extraction.relations),
                "extraction_confidence": extraction.confidence,
            },
        )

    def check_sibling_diversity(
        self,
        new_graph: KnowledgeGraph,
        existing_graphs: list[KnowledgeGraph],
        min_diversity: float = 0.3,
    ) -> bool:
        """Check if a new branch is sufficiently diverse from siblings.

        Args:
            new_graph: Graph from the new branch.
            existing_graphs: Graphs from existing branches.
            min_diversity: Minimum required diversity (1 - similarity).

        Returns:
            True if diverse enough, False if too similar.
        """
        if not existing_graphs:
            return True

        for existing in existing_graphs:
            # Compare entity overlap
            new_entities = set(e.name.lower() for e in new_graph.entities.values())
            existing_entities = set(e.name.lower() for e in existing.entities.values())

            if new_entities and existing_entities:
                overlap = len(new_entities & existing_entities)
                total = len(new_entities | existing_entities)
                similarity = overlap / total if total > 0 else 0

                if similarity > (1 - min_diversity):
                    return False

        return True

    def merge_branch_graphs(
        self,
        branch_results: list[BranchResult],
    ) -> tuple[KnowledgeGraph, list[MergeConflict]]:
        """Merge graphs from all branches.

        Args:
            branch_results: Results from all branches.

        Returns:
            Tuple of (merged graph, conflicts).
        """
        if not branch_results:
            return KnowledgeGraph(), []

        graphs = [br.graph for br in branch_results]
        source_names = [f"branch_{br.branch_id}" for br in branch_results]

        merged, conflicts, provenance = merge_with_provenance(
            graphs=graphs,
            source_names=source_names,
            threshold=self.similarity_threshold,
            embedding_fn=self.embedding_fn,
        )

        # Update provenance tracker
        for item_id, record in provenance.items():
            self._provenance.records[item_id] = record

        return merged, conflicts

    def resolve_and_deduplicate(
        self,
        graph: KnowledgeGraph,
    ) -> tuple[KnowledgeGraph, list[MergeConflict]]:
        """Resolve entities and deduplicate the graph.

        Args:
            graph: The graph to process.

        Returns:
            Tuple of (deduplicated graph, conflicts).
        """
        return deduplicate_graph(
            graph=graph,
            threshold=self.similarity_threshold,
            embedding_fn=self.embedding_fn,
        )

    def run(
        self,
        text: str,
        num_branches: int = 3,
        parallel: bool = False,
    ) -> HarnessResult:
        """Run the full knowledge extraction pipeline.

        Args:
            text: The text to extract knowledge from.
            num_branches: Number of branches to spawn.
            parallel: Whether to run branches in parallel (requires async).

        Returns:
            HarnessResult with unified graph and metadata.
        """
        start_time = datetime.now()

        # Configure branches
        configs = self.configure_branches(text, num_branches)

        # Run branches
        branch_results: list[BranchResult] = []
        existing_graphs: list[KnowledgeGraph] = []

        for config in configs:
            result = self.spawn_branch(text, config)

            # Check diversity before accepting
            if self.check_sibling_diversity(result.graph, existing_graphs):
                branch_results.append(result)
                existing_graphs.append(result.graph)

        # Merge branch results
        merged, merge_conflicts = self.merge_branch_graphs(branch_results)

        # Deduplicate
        deduplicated, dedup_conflicts = self.resolve_and_deduplicate(merged)

        # Consistency check
        consistency = full_consistency_check(deduplicated, self.schema)

        # Build result
        return HarnessResult(
            unified_graph=deduplicated,
            branch_results=branch_results,
            conflicts=merge_conflicts + dedup_conflicts,
            provenance=self._provenance,
            consistency_report=consistency.to_dict(),
            metadata={
                "text_length": len(text),
                "num_branches_configured": len(configs),
                "num_branches_accepted": len(branch_results),
                "total_entities": deduplicated.entity_count(),
                "total_relations": deduplicated.relation_count(),
                "conflict_count": len(merge_conflicts) + len(dedup_conflicts),
                "is_consistent": consistency.is_consistent,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            },
        )

    async def run_async(
        self,
        text: str,
        num_branches: int = 3,
    ) -> HarnessResult:
        """Run the knowledge extraction pipeline asynchronously.

        Args:
            text: The text to extract knowledge from.
            num_branches: Number of branches to spawn.

        Returns:
            HarnessResult with unified graph and metadata.
        """
        start_time = datetime.now()

        # Configure branches
        configs = self.configure_branches(text, num_branches)

        # Run branches in parallel
        async def run_branch(config: BranchConfig) -> BranchResult:
            # Run in thread pool since LLM calls might be blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.spawn_branch, text, config
            )

        tasks = [run_branch(config) for config in configs]
        results = await asyncio.gather(*tasks)

        # Filter by diversity
        branch_results: list[BranchResult] = []
        existing_graphs: list[KnowledgeGraph] = []

        for result in results:
            if self.check_sibling_diversity(result.graph, existing_graphs):
                branch_results.append(result)
                existing_graphs.append(result.graph)

        # Merge and deduplicate
        merged, merge_conflicts = self.merge_branch_graphs(branch_results)
        deduplicated, dedup_conflicts = self.resolve_and_deduplicate(merged)

        # Consistency check
        consistency = full_consistency_check(deduplicated, self.schema)

        return HarnessResult(
            unified_graph=deduplicated,
            branch_results=branch_results,
            conflicts=merge_conflicts + dedup_conflicts,
            provenance=self._provenance,
            consistency_report=consistency.to_dict(),
            metadata={
                "text_length": len(text),
                "num_branches_configured": len(configs),
                "num_branches_accepted": len(branch_results),
                "total_entities": deduplicated.entity_count(),
                "total_relations": deduplicated.relation_count(),
                "conflict_count": len(merge_conflicts) + len(dedup_conflicts),
                "is_consistent": consistency.is_consistent,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "mode": "async",
            },
        )


def create_harness(
    schema: str | GraphSchema = "GENERAL",
    llm_call: LLMCallFn | None = None,
    embedding_fn: EmbeddingFn | None = None,
) -> KnowledgeHarness:
    """Factory function to create a knowledge harness.

    Args:
        schema: Schema name or GraphSchema instance.
        llm_call: Function to call the LLM.
        embedding_fn: Optional function for embeddings.

    Returns:
        Configured KnowledgeHarness instance.
    """
    from .schema import get_schema

    if isinstance(schema, str):
        schema = get_schema(schema)

    return KnowledgeHarness(
        schema=schema,
        llm_call=llm_call,
        embedding_fn=embedding_fn,
    )


def simple_extraction(
    text: str,
    llm_call: LLMCallFn,
    schema: GraphSchema | None = None,
) -> KnowledgeGraph:
    """Simple one-shot extraction without branching.

    Args:
        text: The text to extract from.
        llm_call: Function to call the LLM.
        schema: Optional schema to use.

    Returns:
        Extracted KnowledgeGraph.
    """
    schema = schema or GENERAL

    extraction = full_extraction(
        text=text,
        schema=schema,
        llm_call=llm_call,
    )

    graph = KnowledgeGraph()
    for entity in extraction.entities:
        graph.entities[entity.id] = entity

    for relation in extraction.relations:
        if (relation.source_entity in graph.entities and
            relation.target_entity in graph.entities):
            graph.relations[relation.id] = relation

    return graph
