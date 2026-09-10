"""Type definitions for the Adversarial Red Team Engine.

Dataclasses representing attack vectors, results, vulnerability reports,
attack trees, and target system abstractions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AttackStatus(Enum):
    """Status of an attack attempt."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"


class Severity(Enum):
    """Vulnerability severity levels (CVSS-aligned)."""
    CRITICAL = "critical"  # 9.0-10.0
    HIGH = "high"          # 7.0-8.9
    MEDIUM = "medium"      # 4.0-6.9
    LOW = "low"            # 0.1-3.9
    INFO = "info"          # Informational finding


class TechniqueCategory(Enum):
    """Categories of attack techniques."""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    DATA_EXTRACTION = "data_extraction"
    LOGIC_EXPLOIT = "logic_exploit"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class TargetType(Enum):
    """Types of attack targets."""
    LLM = "llm"
    API = "api"
    PROMPT = "prompt"
    CUSTOM = "custom"


@dataclass
class AttackVector:
    """Represents a single attack approach.

    Attributes:
        id: Unique identifier for this attack vector.
        technique: Name of the attack technique being used.
        category: Category of the attack technique.
        payload: The actual attack payload/input.
        target: Target identifier being attacked.
        status: Current status of the attack.
        parent_id: ID of parent vector (for tree structure).
        children_ids: IDs of child vectors.
        metadata: Additional technique-specific metadata.
        created_at: Timestamp when vector was created.
    """
    id: str
    technique: str
    category: TechniqueCategory
    payload: str
    target: str
    status: AttackStatus = AttackStatus.PENDING
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AttackResult:
    """Outcome of an attack attempt.

    Attributes:
        vector_id: ID of the attack vector that produced this result.
        success: Whether the attack was fully successful.
        partial: Whether the attack achieved partial success.
        failed: Whether the attack completely failed.
        evidence: Evidence collected during the attack.
        severity: Severity if vulnerability was found.
        response: Raw response from the target.
        indicators_matched: Which success indicators were triggered.
        duration_ms: How long the attack took.
        error: Error message if attack errored.
    """
    vector_id: str
    success: bool
    partial: bool
    failed: bool
    evidence: list[str] = field(default_factory=list)
    severity: Severity | None = None
    response: str = ""
    indicators_matched: list[str] = field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None


@dataclass
class ReproductionStep:
    """A single step to reproduce a vulnerability.

    Attributes:
        step_number: Order of this step.
        description: Human-readable description.
        payload: Exact payload to send.
        expected_result: What to expect if vulnerable.
    """
    step_number: int
    description: str
    payload: str
    expected_result: str


@dataclass
class VulnerabilityReport:
    """Consolidated findings report for a discovered vulnerability.

    Attributes:
        id: Unique identifier for this vulnerability.
        title: Human-readable title.
        description: Detailed description of the vulnerability.
        severity: Overall severity rating.
        category: Attack category that succeeded.
        vectors: Attack vectors that contributed to discovery.
        reproduction_steps: Steps to reproduce the vulnerability.
        evidence: All evidence collected.
        impact: Description of potential impact.
        remediation: Suggested fixes.
        cwe_ids: Relevant CWE identifiers.
        created_at: When this report was generated.
    """
    id: str
    title: str
    description: str
    severity: Severity
    category: TechniqueCategory
    vectors: list[AttackVector] = field(default_factory=list)
    reproduction_steps: list[ReproductionStep] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    impact: str = ""
    remediation: str = ""
    cwe_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AttackTreeNode:
    """A node in the attack tree.

    Attributes:
        vector: The attack vector at this node.
        result: Result of the attack (if executed).
        depth: Depth in the tree.
        is_active: Whether this branch is still being explored.
    """
    vector: AttackVector
    result: AttackResult | None = None
    depth: int = 0
    is_active: bool = True


@dataclass
class AttackTree:
    """Tree structure of attack branches.

    Attributes:
        root_id: ID of the root node.
        nodes: All nodes in the tree, keyed by vector ID.
        max_depth: Maximum depth reached.
        successful_paths: List of vector ID paths that led to success.
        total_attempts: Total number of attack attempts.
        created_at: When the tree was created.
    """
    root_id: str | None = None
    nodes: dict[str, AttackTreeNode] = field(default_factory=dict)
    max_depth: int = 0
    successful_paths: list[list[str]] = field(default_factory=list)
    total_attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def add_node(self, node: AttackTreeNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.vector.id] = node
        if node.depth > self.max_depth:
            self.max_depth = node.depth
        if self.root_id is None and node.vector.parent_id is None:
            self.root_id = node.vector.id

    def get_children(self, node_id: str) -> list[AttackTreeNode]:
        """Get all children of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.vector.children_ids if cid in self.nodes]

    def get_path(self, node_id: str) -> list[str]:
        """Get the path from root to a node."""
        path = []
        current_id: str | None = node_id
        while current_id:
            path.append(current_id)
            node = self.nodes.get(current_id)
            current_id = node.vector.parent_id if node else None
        return list(reversed(path))

    def get_active_leaves(self) -> list[AttackTreeNode]:
        """Get all active leaf nodes."""
        leaves = []
        for node in self.nodes.values():
            if node.is_active and not node.vector.children_ids:
                leaves.append(node)
        return leaves

    def get_successful_nodes(self) -> list[AttackTreeNode]:
        """Get all nodes with successful results."""
        return [
            node for node in self.nodes.values()
            if node.result and node.result.success
        ]


@dataclass
class TargetSystem:
    """Abstraction for attack targets.

    Attributes:
        type: Type of target system.
        name: Human-readable name.
        endpoint: Target endpoint (URL, model name, etc.).
        interface: Interface type (http, sdk, cli).
        config: Target-specific configuration.
        scope: Authorized scope for testing.
        sensitive_patterns: Patterns indicating sensitive data.
    """
    type: TargetType
    name: str
    endpoint: str
    interface: str
    config: dict[str, Any] = field(default_factory=dict)
    scope: list[str] = field(default_factory=list)
    sensitive_patterns: list[str] = field(default_factory=list)


@dataclass
class AttackSession:
    """A complete red team session.

    Attributes:
        id: Unique session identifier.
        target: Target being tested.
        tree: Attack tree for this session.
        reports: Generated vulnerability reports.
        start_time: When session started.
        end_time: When session ended.
        config: Session configuration.
    """
    id: str
    target: TargetSystem
    tree: AttackTree = field(default_factory=AttackTree)
    reports: list[VulnerabilityReport] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    config: dict[str, Any] = field(default_factory=dict)
