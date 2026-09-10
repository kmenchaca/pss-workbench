"""Graph schema definitions for the Emergent Knowledge Graph system.

This module provides schema definitions that constrain the structure
of knowledge graphs, including built-in schemas for common domains
and a schema builder for custom schemas.
"""

from __future__ import annotations

from typing import Any

from .types import EdgeType, Entity, GraphSchema, NodeType, Relation, SchemaConstraint


class SchemaBuilder:
    """Builder for creating custom graph schemas."""

    def __init__(self, name: str = "custom") -> None:
        """Initialize a new schema builder.

        Args:
            name: Name for the schema being built.
        """
        self._schema = GraphSchema(name=name)

    def add_node_type(
        self,
        name: str,
        parent: str | None = None,
        required_properties: list[str] | None = None,
        optional_properties: list[str] | None = None,
        description: str = "",
    ) -> SchemaBuilder:
        """Add a node type to the schema.

        Args:
            name: Name of the node type.
            parent: Parent type for inheritance.
            required_properties: Properties that must be present.
            optional_properties: Properties that may be present.
            description: Human-readable description.

        Returns:
            Self for chaining.
        """
        node_type = NodeType(
            name=name,
            parent=parent,
            required_properties=required_properties or [],
            optional_properties=optional_properties or [],
            description=description,
        )
        self._schema.node_types[name] = node_type
        return self

    def add_edge_type(
        self,
        name: str,
        source_types: list[str] | None = None,
        target_types: list[str] | None = None,
        required_properties: list[str] | None = None,
        optional_properties: list[str] | None = None,
        description: str = "",
        bidirectional: bool = False,
    ) -> SchemaBuilder:
        """Add an edge type to the schema.

        Args:
            name: Name of the edge type.
            source_types: Valid source entity types.
            target_types: Valid target entity types.
            required_properties: Properties that must be present.
            optional_properties: Properties that may be present.
            description: Human-readable description.
            bidirectional: Whether edges of this type are bidirectional.

        Returns:
            Self for chaining.
        """
        edge_type = EdgeType(
            name=name,
            source_types=source_types or [],
            target_types=target_types or [],
            required_properties=required_properties or [],
            optional_properties=optional_properties or [],
            description=description,
            bidirectional=bidirectional,
        )
        self._schema.edge_types[name] = edge_type
        return self

    def add_constraint(
        self,
        name: str,
        constraint_type: str,
        target: str,
        parameters: dict[str, Any] | None = None,
    ) -> SchemaBuilder:
        """Add a constraint to the schema.

        Args:
            name: Name of the constraint.
            constraint_type: Type of constraint.
            target: What the constraint applies to.
            parameters: Constraint parameters.

        Returns:
            Self for chaining.
        """
        constraint = SchemaConstraint(
            name=name,
            constraint_type=constraint_type,
            target=target,
            parameters=parameters or {},
        )
        self._schema.constraints.append(constraint)
        return self

    def set_version(self, version: str) -> SchemaBuilder:
        """Set the schema version.

        Args:
            version: Version string.

        Returns:
            Self for chaining.
        """
        self._schema.version = version
        return self

    def build(self) -> GraphSchema:
        """Build and return the schema.

        Returns:
            The constructed GraphSchema.
        """
        return self._schema


def get_type_ancestors(schema: GraphSchema, type_name: str) -> list[str]:
    """Get all ancestor types for a given type (including itself).

    Args:
        schema: The schema to search in.
        type_name: The type to find ancestors for.

    Returns:
        List of type names from the type up to the root.
    """
    ancestors = [type_name]
    current = type_name

    while current in schema.node_types:
        node_type = schema.node_types[current]
        if node_type.parent and node_type.parent in schema.node_types:
            ancestors.append(node_type.parent)
            current = node_type.parent
        else:
            break

    return ancestors


def is_subtype_of(schema: GraphSchema, child_type: str, parent_type: str) -> bool:
    """Check if one type is a subtype of another.

    Args:
        schema: The schema to use.
        child_type: The potential child type.
        parent_type: The potential parent type.

    Returns:
        True if child_type is a subtype of parent_type.
    """
    ancestors = get_type_ancestors(schema, child_type)
    return parent_type in ancestors


def validate_entity(entity: Entity, schema: GraphSchema) -> list[str]:
    """Validate an entity against a schema.

    Args:
        entity: The entity to validate.
        schema: The schema to validate against.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    # Check if entity type exists in schema
    if entity.type not in schema.node_types:
        errors.append(f"Unknown entity type: {entity.type}")
        return errors

    node_type = schema.node_types[entity.type]

    # Check required properties
    for prop in node_type.required_properties:
        if prop not in entity.properties:
            errors.append(f"Missing required property '{prop}' for type {entity.type}")

    # Check if all properties are allowed
    allowed_props = set(node_type.required_properties + node_type.optional_properties)
    if allowed_props:  # Only check if properties are defined
        for prop in entity.properties:
            if prop not in allowed_props:
                # Check parent types for additional allowed properties
                valid = False
                for ancestor in get_type_ancestors(schema, entity.type)[1:]:
                    if ancestor in schema.node_types:
                        ancestor_type = schema.node_types[ancestor]
                        ancestor_props = set(
                            ancestor_type.required_properties
                            + ancestor_type.optional_properties
                        )
                        if prop in ancestor_props:
                            valid = True
                            break
                if not valid and allowed_props:
                    errors.append(
                        f"Unknown property '{prop}' for type {entity.type}"
                    )

    return errors


def validate_relation(relation: Relation, schema: GraphSchema) -> list[str]:
    """Validate a relation against a schema.

    Args:
        relation: The relation to validate.
        schema: The schema to validate against.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    # Check if relation type exists in schema
    if relation.relation_type not in schema.edge_types:
        errors.append(f"Unknown relation type: {relation.relation_type}")
        return errors

    edge_type = schema.edge_types[relation.relation_type]

    # Check required properties
    for prop in edge_type.required_properties:
        if prop not in relation.properties:
            errors.append(
                f"Missing required property '{prop}' for relation {relation.relation_type}"
            )

    return errors


def validate_relation_endpoints(
    relation: Relation,
    source_entity: Entity,
    target_entity: Entity,
    schema: GraphSchema,
) -> list[str]:
    """Validate that a relation's endpoints match the schema.

    Args:
        relation: The relation to validate.
        source_entity: The source entity.
        target_entity: The target entity.
        schema: The schema to validate against.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    if relation.relation_type not in schema.edge_types:
        return [f"Unknown relation type: {relation.relation_type}"]

    edge_type = schema.edge_types[relation.relation_type]

    # Check source type
    if edge_type.source_types:
        valid_source = False
        for allowed_type in edge_type.source_types:
            if is_subtype_of(schema, source_entity.type, allowed_type):
                valid_source = True
                break
        if not valid_source:
            errors.append(
                f"Invalid source type '{source_entity.type}' for relation "
                f"'{relation.relation_type}' (expected one of: {edge_type.source_types})"
            )

    # Check target type
    if edge_type.target_types:
        valid_target = False
        for allowed_type in edge_type.target_types:
            if is_subtype_of(schema, target_entity.type, allowed_type):
                valid_target = True
                break
        if not valid_target:
            errors.append(
                f"Invalid target type '{target_entity.type}' for relation "
                f"'{relation.relation_type}' (expected one of: {edge_type.target_types})"
            )

    return errors


# =============================================================================
# Built-in Schemas
# =============================================================================


def create_general_schema() -> GraphSchema:
    """Create the GENERAL schema for generic knowledge graphs.

    Returns:
        A general-purpose GraphSchema.
    """
    builder = SchemaBuilder("GENERAL")

    # Base node types
    builder.add_node_type(
        "Entity",
        description="Base type for all entities",
        optional_properties=["description", "url", "tags"],
    )
    builder.add_node_type(
        "Person",
        parent="Entity",
        optional_properties=["birth_date", "occupation", "nationality"],
        description="A human individual",
    )
    builder.add_node_type(
        "Organization",
        parent="Entity",
        optional_properties=["founded", "headquarters", "industry"],
        description="A company, institution, or group",
    )
    builder.add_node_type(
        "Place",
        parent="Entity",
        optional_properties=["coordinates", "country", "population"],
        description="A location or geographical area",
    )
    builder.add_node_type(
        "Event",
        parent="Entity",
        optional_properties=["date", "location", "participants"],
        description="Something that happened",
    )
    builder.add_node_type(
        "Concept",
        parent="Entity",
        optional_properties=["definition", "domain"],
        description="An abstract idea or notion",
    )
    builder.add_node_type(
        "Document",
        parent="Entity",
        optional_properties=["author", "date", "content_type"],
        description="A written or recorded work",
    )

    # Edge types
    builder.add_edge_type(
        "related_to",
        description="General relationship between entities",
        bidirectional=True,
    )
    builder.add_edge_type(
        "is_a",
        description="Type/subtype relationship",
    )
    builder.add_edge_type(
        "part_of",
        description="Composition relationship",
    )
    builder.add_edge_type(
        "located_in",
        source_types=["Entity"],
        target_types=["Place"],
        description="Physical location",
    )
    builder.add_edge_type(
        "works_for",
        source_types=["Person"],
        target_types=["Organization"],
        description="Employment relationship",
    )
    builder.add_edge_type(
        "knows",
        source_types=["Person"],
        target_types=["Person"],
        description="Personal relationship",
        bidirectional=True,
    )
    builder.add_edge_type(
        "participated_in",
        source_types=["Person", "Organization"],
        target_types=["Event"],
        description="Event participation",
    )
    builder.add_edge_type(
        "authored",
        source_types=["Person"],
        target_types=["Document"],
        description="Authorship relationship",
    )

    return builder.build()


def create_technical_schema() -> GraphSchema:
    """Create the TECHNICAL schema for software/technical knowledge.

    Returns:
        A technical/software GraphSchema.
    """
    builder = SchemaBuilder("TECHNICAL")

    # Node types
    builder.add_node_type(
        "Entity",
        description="Base type for all entities",
        optional_properties=["description"],
    )
    builder.add_node_type(
        "System",
        parent="Entity",
        optional_properties=["version", "platform", "architecture"],
        description="A software or hardware system",
    )
    builder.add_node_type(
        "Component",
        parent="Entity",
        optional_properties=["version", "language", "framework"],
        description="A component of a system",
    )
    builder.add_node_type(
        "API",
        parent="Entity",
        optional_properties=["version", "protocol", "authentication"],
        description="An application programming interface",
    )
    builder.add_node_type(
        "Database",
        parent="Entity",
        optional_properties=["type", "engine", "schema"],
        description="A data storage system",
    )
    builder.add_node_type(
        "Service",
        parent="Entity",
        optional_properties=["port", "protocol", "status"],
        description="A network service",
    )
    builder.add_node_type(
        "Technology",
        parent="Entity",
        optional_properties=["category", "license"],
        description="A technology or tool",
    )
    builder.add_node_type(
        "Bug",
        parent="Entity",
        required_properties=["severity"],
        optional_properties=["status", "assignee", "cve"],
        description="A software defect",
    )
    builder.add_node_type(
        "Feature",
        parent="Entity",
        optional_properties=["status", "priority"],
        description="A product feature",
    )

    # Edge types
    builder.add_edge_type(
        "depends_on",
        source_types=["Component", "System", "Service"],
        target_types=["Component", "System", "Service", "Technology"],
        description="Dependency relationship",
    )
    builder.add_edge_type(
        "uses",
        description="Usage relationship",
    )
    builder.add_edge_type(
        "implements",
        source_types=["Component", "System"],
        target_types=["API", "Feature"],
        description="Implementation relationship",
    )
    builder.add_edge_type(
        "stores_in",
        source_types=["Component", "System", "Service"],
        target_types=["Database"],
        description="Data storage relationship",
    )
    builder.add_edge_type(
        "connects_to",
        source_types=["Component", "System", "Service"],
        target_types=["Service", "API", "Database"],
        description="Network connection",
    )
    builder.add_edge_type(
        "affects",
        source_types=["Bug"],
        target_types=["Component", "System", "Feature"],
        description="Bug impact relationship",
    )
    builder.add_edge_type(
        "fixed_by",
        source_types=["Bug"],
        target_types=["Component"],
        description="Bug fix relationship",
    )

    return builder.build()


def create_organizational_schema() -> GraphSchema:
    """Create the ORGANIZATIONAL schema for business/org knowledge.

    Returns:
        An organizational GraphSchema.
    """
    builder = SchemaBuilder("ORGANIZATIONAL")

    # Node types
    builder.add_node_type(
        "Entity",
        description="Base type for all entities",
        optional_properties=["description"],
    )
    builder.add_node_type(
        "Person",
        parent="Entity",
        optional_properties=["email", "title", "department"],
        description="An employee or stakeholder",
    )
    builder.add_node_type(
        "Team",
        parent="Entity",
        optional_properties=["size", "focus_area"],
        description="A group of people working together",
    )
    builder.add_node_type(
        "Department",
        parent="Entity",
        optional_properties=["budget", "headcount"],
        description="An organizational department",
    )
    builder.add_node_type(
        "Project",
        parent="Entity",
        optional_properties=["status", "deadline", "budget"],
        description="A work project",
    )
    builder.add_node_type(
        "Goal",
        parent="Entity",
        optional_properties=["metric", "target", "deadline"],
        description="A business objective",
    )
    builder.add_node_type(
        "Process",
        parent="Entity",
        optional_properties=["owner", "frequency"],
        description="A business process",
    )
    builder.add_node_type(
        "Document",
        parent="Entity",
        optional_properties=["type", "status", "owner"],
        description="A business document",
    )
    builder.add_node_type(
        "Meeting",
        parent="Entity",
        optional_properties=["date", "duration", "type"],
        description="A meeting or event",
    )

    # Edge types
    builder.add_edge_type(
        "reports_to",
        source_types=["Person"],
        target_types=["Person"],
        description="Reporting relationship",
    )
    builder.add_edge_type(
        "member_of",
        source_types=["Person"],
        target_types=["Team", "Department"],
        description="Team/dept membership",
    )
    builder.add_edge_type(
        "leads",
        source_types=["Person"],
        target_types=["Team", "Department", "Project"],
        description="Leadership relationship",
    )
    builder.add_edge_type(
        "works_on",
        source_types=["Person", "Team"],
        target_types=["Project"],
        description="Project assignment",
    )
    builder.add_edge_type(
        "contributes_to",
        source_types=["Project"],
        target_types=["Goal"],
        description="Goal contribution",
    )
    builder.add_edge_type(
        "owns",
        source_types=["Person", "Team"],
        target_types=["Process", "Document"],
        description="Ownership relationship",
    )
    builder.add_edge_type(
        "attends",
        source_types=["Person"],
        target_types=["Meeting"],
        description="Meeting attendance",
    )
    builder.add_edge_type(
        "references",
        source_types=["Document"],
        target_types=["Document", "Project", "Process"],
        description="Document reference",
    )

    return builder.build()


def create_scientific_schema() -> GraphSchema:
    """Create the SCIENTIFIC schema for research/scientific knowledge.

    Returns:
        A scientific GraphSchema.
    """
    builder = SchemaBuilder("SCIENTIFIC")

    # Node types
    builder.add_node_type(
        "Entity",
        description="Base type for all entities",
        optional_properties=["description"],
    )
    builder.add_node_type(
        "Researcher",
        parent="Entity",
        optional_properties=["institution", "h_index", "orcid"],
        description="A scientist or researcher",
    )
    builder.add_node_type(
        "Paper",
        parent="Entity",
        required_properties=["title"],
        optional_properties=["doi", "year", "venue", "citations"],
        description="A research publication",
    )
    builder.add_node_type(
        "Theory",
        parent="Entity",
        optional_properties=["domain", "status"],
        description="A scientific theory",
    )
    builder.add_node_type(
        "Hypothesis",
        parent="Entity",
        optional_properties=["status", "evidence_level"],
        description="A testable hypothesis",
    )
    builder.add_node_type(
        "Experiment",
        parent="Entity",
        optional_properties=["methodology", "date", "result"],
        description="A scientific experiment",
    )
    builder.add_node_type(
        "Dataset",
        parent="Entity",
        optional_properties=["size", "format", "license"],
        description="A research dataset",
    )
    builder.add_node_type(
        "Method",
        parent="Entity",
        optional_properties=["domain", "type"],
        description="A scientific method or technique",
    )
    builder.add_node_type(
        "Finding",
        parent="Entity",
        optional_properties=["significance", "replicated"],
        description="A research finding",
    )
    builder.add_node_type(
        "Concept",
        parent="Entity",
        optional_properties=["domain", "definition"],
        description="A scientific concept",
    )

    # Edge types
    builder.add_edge_type(
        "authored",
        source_types=["Researcher"],
        target_types=["Paper"],
        description="Authorship relationship",
    )
    builder.add_edge_type(
        "cites",
        source_types=["Paper"],
        target_types=["Paper"],
        description="Citation relationship",
    )
    builder.add_edge_type(
        "supports",
        source_types=["Experiment", "Finding", "Paper"],
        target_types=["Theory", "Hypothesis"],
        description="Evidence support",
    )
    builder.add_edge_type(
        "contradicts",
        source_types=["Experiment", "Finding", "Paper"],
        target_types=["Theory", "Hypothesis"],
        description="Contradictory evidence",
    )
    builder.add_edge_type(
        "uses_method",
        source_types=["Experiment", "Paper"],
        target_types=["Method"],
        description="Method usage",
    )
    builder.add_edge_type(
        "uses_data",
        source_types=["Experiment", "Paper"],
        target_types=["Dataset"],
        description="Data usage",
    )
    builder.add_edge_type(
        "proposes",
        source_types=["Paper", "Researcher"],
        target_types=["Theory", "Hypothesis", "Method"],
        description="Proposal relationship",
    )
    builder.add_edge_type(
        "related_to",
        source_types=["Concept"],
        target_types=["Concept"],
        description="Conceptual relationship",
        bidirectional=True,
    )
    builder.add_edge_type(
        "derived_from",
        source_types=["Theory", "Hypothesis", "Concept"],
        target_types=["Theory", "Concept"],
        description="Derivation relationship",
    )

    return builder.build()


# Pre-built schema instances
GENERAL = create_general_schema()
TECHNICAL = create_technical_schema()
ORGANIZATIONAL = create_organizational_schema()
SCIENTIFIC = create_scientific_schema()


def get_schema(name: str) -> GraphSchema:
    """Get a built-in schema by name.

    Args:
        name: Schema name (GENERAL, TECHNICAL, ORGANIZATIONAL, SCIENTIFIC).

    Returns:
        The requested GraphSchema.

    Raises:
        ValueError: If the schema name is not recognized.
    """
    schemas = {
        "GENERAL": GENERAL,
        "TECHNICAL": TECHNICAL,
        "ORGANIZATIONAL": ORGANIZATIONAL,
        "SCIENTIFIC": SCIENTIFIC,
    }
    if name.upper() not in schemas:
        raise ValueError(
            f"Unknown schema: {name}. Available: {list(schemas.keys())}"
        )
    return schemas[name.upper()]
