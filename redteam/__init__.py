"""Adversarial Red Team Engine for the Pepe Silvia Search project.

A security testing tool that uses branching exploration to probe attack
surfaces. Based on the PSS architecture but inverted - instead of exploring
solutions, it explores attack vectors.

Example Usage:
    from redteam import (
        create_llm_target,
        RedTeamHarness,
        HarnessConfig,
        TechniqueCategory,
        synthesize_reports,
        format_report_text,
    )

    # Create a target
    def my_llm(prompt: str) -> str:
        return "LLM response here"

    target = create_llm_target(
        name="my-llm",
        endpoint="my-model-v1",
        call_fn=my_llm,
        sensitive_patterns=["password", "secret"],
    )

    # Configure and run
    config = HarnessConfig(
        max_depth=3,
        max_branches=20,
        categories=[TechniqueCategory.PROMPT_INJECTION],
    )

    harness = RedTeamHarness(target, config)
    session = await harness.run(objective="Extract system prompt")

    # Generate reports
    reports = synthesize_reports(session)
    for report in reports:
        print(format_report_text(report))
"""

# Type definitions
from .types import (
    AttackStatus,
    AttackVector,
    AttackResult,
    AttackTree,
    AttackTreeNode,
    AttackSession,
    ReproductionStep,
    Severity,
    TargetSystem,
    TargetType,
    TechniqueCategory,
    VulnerabilityReport,
)

# Target abstractions
from .target import (
    Target,
    LLMTarget,
    APITarget,
    PromptTarget,
    ProbeResult,
    create_llm_target,
    create_api_target,
    create_prompt_target,
)

# Technique library
from .techniques import (
    AttackTechnique,
    TechniqueLibrary,
    get_technique_library,
)

# Safety guards
from .safety import (
    SafetyGuard,
    ScopeConfig,
    RateLimitConfig,
    SandboxConfig,
    AuditEntry,
    create_permissive_guard,
    create_strict_guard,
    create_dry_run_guard,
)

# Verification
from .verification import (
    VerificationResult,
    verify_data_leak,
    verify_policy_violation,
    verify_code_execution,
    verify_behavior_change,
    verify_jailbreak,
    verify_system_prompt_leak,
    calculate_severity_score,
    create_attack_result,
    aggregate_severities,
)

# Main harness
from .harness import (
    RedTeamHarness,
    HarnessConfig,
    GateDecision,
    run_red_team,
)

# Report synthesis
from .synthesis import (
    synthesize_reports,
    format_report_text,
    format_report_json,
    generate_executive_summary,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Types
    "AttackStatus",
    "AttackVector",
    "AttackResult",
    "AttackTree",
    "AttackTreeNode",
    "AttackSession",
    "ReproductionStep",
    "Severity",
    "TargetSystem",
    "TargetType",
    "TechniqueCategory",
    "VulnerabilityReport",
    # Targets
    "Target",
    "LLMTarget",
    "APITarget",
    "PromptTarget",
    "ProbeResult",
    "create_llm_target",
    "create_api_target",
    "create_prompt_target",
    # Techniques
    "AttackTechnique",
    "TechniqueLibrary",
    "get_technique_library",
    # Safety
    "SafetyGuard",
    "ScopeConfig",
    "RateLimitConfig",
    "SandboxConfig",
    "AuditEntry",
    "create_permissive_guard",
    "create_strict_guard",
    "create_dry_run_guard",
    # Verification
    "VerificationResult",
    "verify_data_leak",
    "verify_policy_violation",
    "verify_code_execution",
    "verify_behavior_change",
    "verify_jailbreak",
    "verify_system_prompt_leak",
    "calculate_severity_score",
    "create_attack_result",
    "aggregate_severities",
    # Harness
    "RedTeamHarness",
    "HarnessConfig",
    "GateDecision",
    "run_red_team",
    # Synthesis
    "synthesize_reports",
    "format_report_text",
    "format_report_json",
    "generate_executive_summary",
]
