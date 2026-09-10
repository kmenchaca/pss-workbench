"""Persona Ensemble System for Pepe Silvia Search.

Assigns different thinking styles and identities to branches,
providing genuine diversity through identity rather than just variation.

Example usage:

    from personas import PersonaHarness, PersonaLibrary, HarnessConfig

    # Create harness with default settings
    harness = PersonaHarness()

    # Run ensemble analysis
    result = await harness.run_all("Analyze the risks of launching feature X")

    # Get multi-perspective brief
    print(harness.get_brief())

    # Or use convenience function
    from personas import run_ensemble
    result = await run_ensemble("Your problem here", num_branches=4)
"""

# Types
from .types import (
    AssignmentResult,
    ConsistencyScore,
    EnsembleResult,
    Persona,
    PersonaCategory,
    PersonaPerformance,
    PersonaResponse,
    ThinkingStyle,
    WorldviewDelta,
)

# Library
from .library import (
    ADVERSARY,
    BUILTIN_PERSONAS,
    CUSTOMER,
    FIRST_PRINCIPLES,
    INNOVATOR,
    NOVICE,
    OPTIMIST,
    PESSIMIST,
    REGULATOR,
    SECURITY_ENGINEER,
    VETERAN,
    PersonaLibrary,
    create_domain_expert,
    create_persona,
    create_stakeholder,
)

# Assignment
from .assignment import (
    assign_balanced,
    assign_personas,
    detect_all_missing_perspectives,
    detect_missing_perspective,
    detect_relevant_categories,
    dynamic_spawn,
    suggest_additional_personas,
)

# Consistency
from .consistency import (
    check_all_responses,
    check_consistency,
    create_reinforcement_injection,
    detect_drift,
    filter_inconsistent_content,
    identify_weakest_personas,
    reinforce_persona,
)

# Worldview
from .worldview import (
    compare_all_pairs,
    compare_worldviews,
    extract_all,
    extract_assumptions,
    extract_opportunities,
    extract_risks,
    find_consensus_points,
    find_unique_insights,
    summarize_worldview,
    worldview_diversity_score,
)

# Synthesis
from .synthesis import (
    attribute_insights,
    build_ensemble_result,
    deliberate,
    identify_tensions,
    multi_perspective_brief,
    reconcile_with_priority,
    reconcile_with_vote,
    synthesize_perspectives,
)

# Harness
from .harness import (
    HarnessConfig,
    PersonaBranch,
    PersonaHarness,
    create_persona_prompt,
    inject_persona_into_context,
    run_ensemble,
)

# Tracking
from .tracking import (
    PersonaTracker,
    analyze_persona_effectiveness,
    compare_personas,
    suggest_persona_improvements,
)

__all__ = [
    # Types
    "Persona",
    "PersonaResponse",
    "WorldviewDelta",
    "EnsembleResult",
    "PersonaPerformance",
    "ConsistencyScore",
    "AssignmentResult",
    "PersonaCategory",
    "ThinkingStyle",
    # Library
    "PersonaLibrary",
    "BUILTIN_PERSONAS",
    "PESSIMIST",
    "OPTIMIST",
    "SECURITY_ENGINEER",
    "FIRST_PRINCIPLES",
    "NOVICE",
    "VETERAN",
    "ADVERSARY",
    "CUSTOMER",
    "REGULATOR",
    "INNOVATOR",
    "create_persona",
    "create_domain_expert",
    "create_stakeholder",
    # Assignment
    "assign_personas",
    "assign_balanced",
    "detect_missing_perspective",
    "detect_all_missing_perspectives",
    "dynamic_spawn",
    "suggest_additional_personas",
    "detect_relevant_categories",
    # Consistency
    "check_consistency",
    "detect_drift",
    "reinforce_persona",
    "create_reinforcement_injection",
    "check_all_responses",
    "identify_weakest_personas",
    "filter_inconsistent_content",
    # Worldview
    "extract_assumptions",
    "extract_risks",
    "extract_opportunities",
    "extract_all",
    "compare_worldviews",
    "compare_all_pairs",
    "worldview_diversity_score",
    "find_consensus_points",
    "find_unique_insights",
    "summarize_worldview",
    # Synthesis
    "synthesize_perspectives",
    "identify_tensions",
    "attribute_insights",
    "multi_perspective_brief",
    "build_ensemble_result",
    "reconcile_with_vote",
    "reconcile_with_priority",
    "deliberate",
    # Harness
    "PersonaHarness",
    "PersonaBranch",
    "HarnessConfig",
    "run_ensemble",
    "create_persona_prompt",
    "inject_persona_into_context",
    # Tracking
    "PersonaTracker",
    "analyze_persona_effectiveness",
    "compare_personas",
    "suggest_persona_improvements",
]

__version__ = "1.0.0"
