"""Unified Interface for PSS Experimental Directions.

Provides a single entry point to all 10 experimental systems:
- redteam: Adversarial Red Team Engine
- evolution: Evolutionary Code Synthesis
- argswarm: Argumentative Swarm
- personas: Persona Ensemble
- temporal: Temporal Scenario Planner
- knowledge: Emergent Knowledge Graph
- embodied: Embodied Action Planner
- metalearner: Self-Improving Meta-Learner
- dreamlogic: Dream Logic Generator
- negotiate: Distributed Negotiation Simulator

Usage:
    from unified import UnifiedHarness, run_system

    # Quick run
    result = await run_system("argswarm", "AI will replace most jobs")

    # Full control
    harness = UnifiedHarness(provider="openrouter", model="llama-3.1-8b-instruct")
    result = await harness.run("personas", "Analyze risks of launching feature X")
"""

from .harness import UnifiedHarness, run_system, list_systems, get_system_info
from .router import SystemRouter, RouteDecision
from .config import UnifiedConfig, SystemConfig
from .meta_harness import MetaHarness, MetaResult, TaskAnalysis, smart_run, analyze_task
from .learner_integration import (
    LearnerControlledHarness,
    LearnerResult,
    LearningStats,
    learner_solve,
    classify_for_routing,
)

__version__ = "1.0.0"

__all__ = [
    # Unified Harness
    "UnifiedHarness",
    "run_system",
    "list_systems",
    "get_system_info",
    # Router
    "SystemRouter",
    "RouteDecision",
    # Config
    "UnifiedConfig",
    "SystemConfig",
    # Meta Harness
    "MetaHarness",
    "MetaResult",
    "TaskAnalysis",
    "smart_run",
    "analyze_task",
    # Learner Integration
    "LearnerControlledHarness",
    "LearnerResult",
    "LearningStats",
    "learner_solve",
    "classify_for_routing",
]
