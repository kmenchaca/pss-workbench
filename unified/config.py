"""Configuration for the Unified Interface.

Defines configuration for individual systems and the unified harness.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SystemType(Enum):
    """Available experimental systems."""

    REDTEAM = "redteam"
    EVOLUTION = "evolution"
    ARGSWARM = "argswarm"
    PERSONAS = "personas"
    TEMPORAL = "temporal"
    KNOWLEDGE = "knowledge"
    EMBODIED = "embodied"
    METALEARNER = "metalearner"
    DREAMLOGIC = "dreamlogic"
    NEGOTIATE = "negotiate"


# System metadata
SYSTEM_INFO = {
    SystemType.REDTEAM: {
        "name": "Adversarial Red Team Engine",
        "description": "Adversarial branches attack each other's solutions to find weaknesses",
        "input_type": "code_or_proposal",
        "output_type": "vulnerabilities_and_recommendations",
        "best_for": ["security review", "code audit", "proposal critique"],
    },
    SystemType.EVOLUTION: {
        "name": "Evolutionary Code Synthesis",
        "description": "Code evolves through mutation and selection toward fitness goals",
        "input_type": "specification_and_tests",
        "output_type": "evolved_code",
        "best_for": ["optimization", "algorithm design", "test-driven development"],
    },
    SystemType.ARGSWARM: {
        "name": "Argumentative Swarm",
        "description": "Branches adopt positions and debate to find truth",
        "input_type": "proposition",
        "output_type": "judgment_and_reasoning",
        "best_for": ["debate", "decision making", "policy analysis"],
    },
    SystemType.PERSONAS: {
        "name": "Persona Ensemble",
        "description": "Multiple persona-driven branches provide diverse perspectives",
        "input_type": "problem_statement",
        "output_type": "multi_perspective_analysis",
        "best_for": ["risk analysis", "stakeholder analysis", "strategy review"],
    },
    SystemType.TEMPORAL: {
        "name": "Temporal Scenario Planner",
        "description": "Projects scenarios forward through time with branching futures",
        "input_type": "initial_conditions",
        "output_type": "scenario_tree",
        "best_for": ["strategic planning", "risk assessment", "forecasting"],
    },
    SystemType.KNOWLEDGE: {
        "name": "Emergent Knowledge Graph",
        "description": "Builds knowledge graphs through distributed exploration",
        "input_type": "domain_or_question",
        "output_type": "knowledge_graph",
        "best_for": ["research synthesis", "concept mapping", "learning"],
    },
    SystemType.EMBODIED: {
        "name": "Embodied Action Planner",
        "description": "Plans physical actions considering constraints and physics",
        "input_type": "goal_and_environment",
        "output_type": "action_plan",
        "best_for": ["robotics", "task planning", "process optimization"],
    },
    SystemType.METALEARNER: {
        "name": "Self-Improving Meta-Learner",
        "description": "Learns from exploration patterns to improve future performance",
        "input_type": "task_and_history",
        "output_type": "improved_strategy",
        "best_for": ["optimization", "adaptive systems", "learning"],
    },
    SystemType.DREAMLOGIC: {
        "name": "Dream Logic Generator",
        "description": "Uses non-linear, associative reasoning for creative exploration",
        "input_type": "seed_concept",
        "output_type": "creative_associations",
        "best_for": ["brainstorming", "creative writing", "ideation"],
    },
    SystemType.NEGOTIATE: {
        "name": "Distributed Negotiation Simulator",
        "description": "Simulates multi-party negotiations with competing interests",
        "input_type": "parties_and_positions",
        "output_type": "negotiation_outcome",
        "best_for": ["negotiation prep", "conflict resolution", "deal structuring"],
    },
}


@dataclass
class SystemConfig:
    """Configuration for a specific system run.

    Attributes:
        system_type: Which system to run
        provider: LLM provider (anthropic, openrouter)
        model: Model identifier
        max_tokens: Maximum tokens per call
        temperature: Sampling temperature
        num_branches: Number of parallel branches (where applicable)
        timeout: Maximum runtime in seconds
        extra: System-specific configuration
    """

    system_type: SystemType
    provider: str = "openrouter"
    model: str = "meta-llama/llama-3.1-8b-instruct"
    max_tokens: int = 2000
    temperature: float = 0.7
    num_branches: int = 4
    timeout: float = 120.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedConfig:
    """Configuration for the unified harness.

    Attributes:
        default_provider: Default LLM provider
        default_model: Default model
        enable_routing: Whether to auto-route to best system
        parallel_systems: Whether to run multiple systems in parallel
        verbose: Whether to output progress
    """

    default_provider: str = "openrouter"
    default_model: str = "meta-llama/llama-3.1-8b-instruct"
    enable_routing: bool = False
    parallel_systems: bool = False
    verbose: bool = True


def get_system_type(name: str) -> Optional[SystemType]:
    """Convert string name to SystemType.

    Args:
        name: System name (e.g., "argswarm", "personas")

    Returns:
        SystemType if valid, None otherwise
    """
    try:
        return SystemType(name.lower())
    except ValueError:
        return None


def get_info(system_type: SystemType) -> dict[str, Any]:
    """Get metadata for a system.

    Args:
        system_type: The system type

    Returns:
        Dictionary with system metadata
    """
    return SYSTEM_INFO.get(system_type, {})


def list_all_systems() -> list[dict[str, Any]]:
    """List all available systems with their metadata.

    Returns:
        List of system info dictionaries
    """
    result = []
    for sys_type in SystemType:
        info = get_info(sys_type)
        result.append({
            "id": sys_type.value,
            "name": info.get("name", sys_type.value),
            "description": info.get("description", ""),
            "best_for": info.get("best_for", []),
        })
    return result
