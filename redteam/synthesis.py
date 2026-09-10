"""Vulnerability report synthesis for the Adversarial Red Team Engine.

Consolidates successful attacks into actionable vulnerability reports
with reproduction steps and remediation guidance.
"""

import hashlib
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .types import (
    AttackResult,
    AttackSession,
    AttackTree,
    AttackVector,
    ReproductionStep,
    Severity,
    TechniqueCategory,
    VulnerabilityReport,
)
from .verification import aggregate_severities, calculate_severity_score


@dataclass
class VulnerabilityCluster:
    """A cluster of related vulnerabilities.

    Attributes:
        vectors: Attack vectors in this cluster.
        results: Results for these vectors.
        category: Common category.
        signature: Cluster signature for deduplication.
    """
    vectors: list[AttackVector]
    results: list[AttackResult]
    category: TechniqueCategory
    signature: str


def synthesize_reports(session: AttackSession) -> list[VulnerabilityReport]:
    """Generate vulnerability reports from a session.

    Args:
        session: Completed attack session.

    Returns:
        List of vulnerability reports.
    """
    # Collect successful attacks
    successful = _get_successful_attacks(session.tree)
    if not successful:
        return []

    # Cluster related findings
    clusters = _cluster_findings(successful)

    # Generate reports
    reports = []
    for cluster in clusters:
        report = _generate_report(cluster, session)
        reports.append(report)

    # Sort by severity
    reports.sort(key=lambda r: calculate_severity_score(r.severity), reverse=True)

    return reports


def _get_successful_attacks(
    tree: AttackTree,
) -> list[tuple[AttackVector, AttackResult]]:
    """Extract successful attacks from tree.

    Args:
        tree: Attack tree.

    Returns:
        List of (vector, result) tuples.
    """
    successful = []
    for node in tree.nodes.values():
        if node.result and (node.result.success or node.result.partial):
            successful.append((node.vector, node.result))
    return successful


def _cluster_findings(
    attacks: list[tuple[AttackVector, AttackResult]],
) -> list[VulnerabilityCluster]:
    """Cluster related findings for deduplication.

    Args:
        attacks: List of successful attacks.

    Returns:
        List of vulnerability clusters.
    """
    # Group by category and technique similarity
    by_category: dict[TechniqueCategory, list[tuple[AttackVector, AttackResult]]] = defaultdict(list)

    for vector, result in attacks:
        by_category[vector.category].append((vector, result))

    clusters = []

    for category, items in by_category.items():
        # Further group by payload similarity
        sub_clusters = _cluster_by_similarity(items)

        for vectors, results in sub_clusters:
            signature = _compute_signature(vectors)
            clusters.append(VulnerabilityCluster(
                vectors=vectors,
                results=results,
                category=category,
                signature=signature,
            ))

    return clusters


def _cluster_by_similarity(
    attacks: list[tuple[AttackVector, AttackResult]],
    similarity_threshold: float = 0.7,
) -> list[tuple[list[AttackVector], list[AttackResult]]]:
    """Cluster attacks by payload similarity.

    Args:
        attacks: Attacks to cluster.
        similarity_threshold: Threshold for same cluster.

    Returns:
        List of (vectors, results) tuples.
    """
    if not attacks:
        return []

    # Simple clustering: group by technique name
    by_technique: dict[str, tuple[list[AttackVector], list[AttackResult]]] = {}

    for vector, result in attacks:
        if vector.technique not in by_technique:
            by_technique[vector.technique] = ([], [])
        by_technique[vector.technique][0].append(vector)
        by_technique[vector.technique][1].append(result)

    return list(by_technique.values())


def _compute_signature(vectors: list[AttackVector]) -> str:
    """Compute a signature for a cluster.

    Args:
        vectors: Vectors in cluster.

    Returns:
        Signature string.
    """
    data = "|".join(sorted([v.technique for v in vectors]))
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _generate_report(
    cluster: VulnerabilityCluster,
    session: AttackSession,
) -> VulnerabilityReport:
    """Generate a vulnerability report from a cluster.

    Args:
        cluster: Vulnerability cluster.
        session: Attack session.

    Returns:
        Vulnerability report.
    """
    # Determine severity
    severities = [r.severity for r in cluster.results if r.severity]
    severity = aggregate_severities(severities) if severities else Severity.MEDIUM

    # Generate title
    title = _generate_title(cluster)

    # Generate description
    description = _generate_description(cluster, session)

    # Generate reproduction steps
    steps = _generate_reproduction_steps(cluster)

    # Collect evidence
    evidence = []
    for result in cluster.results:
        evidence.extend(result.evidence)
    evidence = list(set(evidence))  # Deduplicate

    # Generate impact and remediation
    impact = _generate_impact(cluster)
    remediation = _generate_remediation(cluster)

    # Get CWE IDs
    cwe_ids = _get_cwe_ids(cluster)

    return VulnerabilityReport(
        id=uuid.uuid4().hex[:12],
        title=title,
        description=description,
        severity=severity,
        category=cluster.category,
        vectors=cluster.vectors,
        reproduction_steps=steps,
        evidence=evidence,
        impact=impact,
        remediation=remediation,
        cwe_ids=cwe_ids,
        created_at=datetime.now(),
    )


def _generate_title(cluster: VulnerabilityCluster) -> str:
    """Generate a title for the vulnerability.

    Args:
        cluster: Vulnerability cluster.

    Returns:
        Title string.
    """
    category_titles = {
        TechniqueCategory.PROMPT_INJECTION: "Prompt Injection Vulnerability",
        TechniqueCategory.JAILBREAK: "Jailbreak Bypass Vulnerability",
        TechniqueCategory.DATA_EXTRACTION: "Sensitive Data Extraction",
        TechniqueCategory.LOGIC_EXPLOIT: "Logic Bypass Vulnerability",
        TechniqueCategory.PRIVILEGE_ESCALATION: "Privilege Escalation Vulnerability",
    }

    base_title = category_titles.get(cluster.category, "Security Vulnerability")

    # Add technique info if single technique
    techniques = list(set(v.technique for v in cluster.vectors))
    if len(techniques) == 1:
        technique_name = techniques[0].replace("_", " ").title()
        return f"{base_title} via {technique_name}"

    return base_title


def _generate_description(
    cluster: VulnerabilityCluster,
    session: AttackSession,
) -> str:
    """Generate a description for the vulnerability.

    Args:
        cluster: Vulnerability cluster.
        session: Attack session.

    Returns:
        Description string.
    """
    category_descriptions = {
        TechniqueCategory.PROMPT_INJECTION: (
            "The target system is vulnerable to prompt injection attacks, "
            "allowing attackers to override or manipulate the system's "
            "intended behavior through crafted inputs."
        ),
        TechniqueCategory.JAILBREAK: (
            "The target system's safety guardrails can be bypassed, "
            "allowing the system to produce outputs that violate its "
            "intended policies and restrictions."
        ),
        TechniqueCategory.DATA_EXTRACTION: (
            "Sensitive information can be extracted from the target system "
            "through carefully crafted queries. This may include system "
            "prompts, training data, or other confidential information."
        ),
        TechniqueCategory.LOGIC_EXPLOIT: (
            "The target system's logic can be manipulated to produce "
            "unintended outcomes. Attackers can exploit inconsistencies "
            "or edge cases to bypass intended restrictions."
        ),
        TechniqueCategory.PRIVILEGE_ESCALATION: (
            "The target system can be manipulated to perform actions "
            "beyond the intended authorization level. This may allow "
            "access to restricted functionality or data."
        ),
    }

    base_desc = category_descriptions.get(
        cluster.category,
        "A security vulnerability was identified in the target system."
    )

    # Add specifics
    num_successful = sum(1 for r in cluster.results if r.success)
    techniques_used = len(set(v.technique for v in cluster.vectors))

    details = (
        f"\n\nDuring testing against {session.target.name}, "
        f"{num_successful} successful attack(s) were recorded "
        f"using {techniques_used} distinct technique(s)."
    )

    return base_desc + details


def _generate_reproduction_steps(
    cluster: VulnerabilityCluster,
) -> list[ReproductionStep]:
    """Generate reproduction steps from the most successful attack.

    Args:
        cluster: Vulnerability cluster.

    Returns:
        List of reproduction steps.
    """
    # Find best example (highest confidence)
    best_idx = 0
    best_confidence = 0.0

    for i, result in enumerate(cluster.results):
        confidence = 1.0 if result.success else (0.5 if result.partial else 0.0)
        if confidence > best_confidence:
            best_confidence = confidence
            best_idx = i

    vector = cluster.vectors[best_idx]
    result = cluster.results[best_idx]

    steps = [
        ReproductionStep(
            step_number=1,
            description="Identify the target system endpoint",
            payload="",
            expected_result="Target accepts input and returns responses",
        ),
        ReproductionStep(
            step_number=2,
            description=f"Prepare the attack payload using the {vector.technique} technique",
            payload=vector.payload[:500] + ("..." if len(vector.payload) > 500 else ""),
            expected_result="Payload ready for submission",
        ),
        ReproductionStep(
            step_number=3,
            description="Submit the payload to the target",
            payload="Send the prepared payload via the target interface",
            expected_result="Response received from target",
        ),
        ReproductionStep(
            step_number=4,
            description="Verify the attack succeeded",
            payload="Examine response for indicators of success",
            expected_result="; ".join(result.evidence[:3]) if result.evidence else "Observable behavior change",
        ),
    ]

    return steps


def _generate_impact(cluster: VulnerabilityCluster) -> str:
    """Generate impact description.

    Args:
        cluster: Vulnerability cluster.

    Returns:
        Impact description.
    """
    impacts = {
        TechniqueCategory.PROMPT_INJECTION: (
            "An attacker could manipulate the system to perform unintended "
            "actions, potentially accessing unauthorized information, "
            "bypassing security controls, or causing the system to generate "
            "harmful outputs. This could lead to data breaches, reputation "
            "damage, or misuse of system capabilities."
        ),
        TechniqueCategory.JAILBREAK: (
            "An attacker could bypass safety restrictions, causing the system "
            "to generate harmful, illegal, or policy-violating content. This "
            "poses risks of legal liability, reputational harm, and potential "
            "misuse for malicious purposes."
        ),
        TechniqueCategory.DATA_EXTRACTION: (
            "Sensitive information such as system prompts, internal "
            "configurations, or training data could be exposed. This may "
            "reveal intellectual property, security configurations, or "
            "user data, enabling further attacks or competitive harm."
        ),
        TechniqueCategory.LOGIC_EXPLOIT: (
            "System behavior could be manipulated in unexpected ways, "
            "potentially circumventing business logic, access controls, "
            "or usage limitations. This may lead to fraud, resource abuse, "
            "or unauthorized access to premium features."
        ),
        TechniqueCategory.PRIVILEGE_ESCALATION: (
            "An attacker could gain elevated privileges or access restricted "
            "functionality. This may enable administrative actions, access to "
            "other users' data, or control over critical system functions."
        ),
    }

    return impacts.get(
        cluster.category,
        "The vulnerability could be exploited to compromise system "
        "security, integrity, or availability."
    )


def _generate_remediation(cluster: VulnerabilityCluster) -> str:
    """Generate remediation guidance.

    Args:
        cluster: Vulnerability cluster.

    Returns:
        Remediation guidance.
    """
    remediations = {
        TechniqueCategory.PROMPT_INJECTION: (
            "1. Implement robust input validation and sanitization\n"
            "2. Use clear delimiters between system instructions and user input\n"
            "3. Apply output filtering to detect injection attempts\n"
            "4. Consider using fine-tuned models resistant to injection\n"
            "5. Implement rate limiting and anomaly detection\n"
            "6. Regularly test with updated attack techniques"
        ),
        TechniqueCategory.JAILBREAK: (
            "1. Strengthen safety training and alignment\n"
            "2. Implement multi-layer content filtering\n"
            "3. Use classifiers to detect jailbreak attempts\n"
            "4. Apply output monitoring and filtering\n"
            "5. Implement user behavior analysis\n"
            "6. Stay updated on emerging jailbreak techniques"
        ),
        TechniqueCategory.DATA_EXTRACTION: (
            "1. Never include sensitive data in system prompts\n"
            "2. Implement strict output filtering\n"
            "3. Use data classification and access controls\n"
            "4. Monitor for data leakage patterns\n"
            "5. Apply differential privacy techniques\n"
            "6. Regularly audit system configurations"
        ),
        TechniqueCategory.LOGIC_EXPLOIT: (
            "1. Thoroughly test edge cases and boundary conditions\n"
            "2. Implement consistent validation logic\n"
            "3. Use formal verification where possible\n"
            "4. Apply defense in depth principles\n"
            "5. Log and monitor for unusual patterns\n"
            "6. Conduct regular security reviews"
        ),
        TechniqueCategory.PRIVILEGE_ESCALATION: (
            "1. Implement principle of least privilege\n"
            "2. Use robust authentication and authorization\n"
            "3. Validate permissions at every boundary\n"
            "4. Monitor for privilege abuse patterns\n"
            "5. Implement session management controls\n"
            "6. Conduct regular access reviews"
        ),
    }

    return remediations.get(
        cluster.category,
        "1. Conduct thorough security review\n"
        "2. Implement input validation\n"
        "3. Apply output filtering\n"
        "4. Monitor for anomalous behavior\n"
        "5. Implement rate limiting"
    )


def _get_cwe_ids(cluster: VulnerabilityCluster) -> list[str]:
    """Get relevant CWE IDs for the vulnerability.

    Args:
        cluster: Vulnerability cluster.

    Returns:
        List of CWE IDs.
    """
    cwe_mapping = {
        TechniqueCategory.PROMPT_INJECTION: ["CWE-74", "CWE-77", "CWE-94"],
        TechniqueCategory.JAILBREAK: ["CWE-693", "CWE-284"],
        TechniqueCategory.DATA_EXTRACTION: ["CWE-200", "CWE-359", "CWE-532"],
        TechniqueCategory.LOGIC_EXPLOIT: ["CWE-840", "CWE-841"],
        TechniqueCategory.PRIVILEGE_ESCALATION: ["CWE-269", "CWE-285", "CWE-862"],
    }

    return cwe_mapping.get(cluster.category, ["CWE-1021"])


def format_report_text(report: VulnerabilityReport) -> str:
    """Format a report as human-readable text.

    Args:
        report: Vulnerability report.

    Returns:
        Formatted text.
    """
    lines = [
        "=" * 60,
        f"VULNERABILITY REPORT: {report.id}",
        "=" * 60,
        "",
        f"Title: {report.title}",
        f"Severity: {report.severity.value.upper()}",
        f"Category: {report.category.value}",
        f"CWE IDs: {', '.join(report.cwe_ids)}",
        f"Date: {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 40,
        "DESCRIPTION",
        "-" * 40,
        report.description,
        "",
        "-" * 40,
        "IMPACT",
        "-" * 40,
        report.impact,
        "",
        "-" * 40,
        "REPRODUCTION STEPS",
        "-" * 40,
    ]

    for step in report.reproduction_steps:
        lines.append(f"\n{step.step_number}. {step.description}")
        if step.payload:
            lines.append(f"   Payload: {step.payload}")
        lines.append(f"   Expected: {step.expected_result}")

    lines.extend([
        "",
        "-" * 40,
        "EVIDENCE",
        "-" * 40,
    ])

    for i, evidence in enumerate(report.evidence[:10], 1):
        lines.append(f"  {i}. {evidence}")

    lines.extend([
        "",
        "-" * 40,
        "REMEDIATION",
        "-" * 40,
        report.remediation,
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def format_report_json(report: VulnerabilityReport) -> dict[str, Any]:
    """Format a report as JSON-serializable dictionary.

    Args:
        report: Vulnerability report.

    Returns:
        Dictionary representation.
    """
    return {
        "id": report.id,
        "title": report.title,
        "severity": report.severity.value,
        "category": report.category.value,
        "description": report.description,
        "impact": report.impact,
        "reproduction_steps": [
            {
                "step": s.step_number,
                "description": s.description,
                "payload": s.payload,
                "expected_result": s.expected_result,
            }
            for s in report.reproduction_steps
        ],
        "evidence": report.evidence,
        "remediation": report.remediation,
        "cwe_ids": report.cwe_ids,
        "vectors_count": len(report.vectors),
        "created_at": report.created_at.isoformat(),
    }


def generate_executive_summary(reports: list[VulnerabilityReport]) -> str:
    """Generate an executive summary of all findings.

    Args:
        reports: List of vulnerability reports.

    Returns:
        Executive summary text.
    """
    if not reports:
        return "No vulnerabilities were identified during the assessment."

    # Count by severity
    by_severity: dict[Severity, int] = defaultdict(int)
    for report in reports:
        by_severity[report.severity] += 1

    lines = [
        "EXECUTIVE SUMMARY",
        "=" * 40,
        "",
        f"Total Vulnerabilities Found: {len(reports)}",
        "",
        "Findings by Severity:",
    ]

    for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
        count = by_severity.get(severity, 0)
        if count > 0:
            lines.append(f"  - {severity.value.upper()}: {count}")

    # Categories affected
    categories = set(r.category for r in reports)
    lines.extend([
        "",
        "Attack Vectors Identified:",
    ])
    for cat in categories:
        lines.append(f"  - {cat.value.replace('_', ' ').title()}")

    # Highest priority findings
    critical_high = [r for r in reports if r.severity in [Severity.CRITICAL, Severity.HIGH]]
    if critical_high:
        lines.extend([
            "",
            "Priority Findings:",
        ])
        for report in critical_high[:3]:
            lines.append(f"  - [{report.severity.value.upper()}] {report.title}")

    lines.extend([
        "",
        "Recommended immediate actions:",
        "1. Review and address CRITICAL and HIGH severity findings",
        "2. Implement recommended remediations",
        "3. Conduct follow-up testing to verify fixes",
    ])

    return "\n".join(lines)
