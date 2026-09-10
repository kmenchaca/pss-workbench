"""Strategy definitions for the meta-learner system.

This module defines built-in exploration strategies and provides
mechanisms for applying them to problems.
"""

import json
from pathlib import Path
from typing import Any, Callable

from .types import Strategy


# Built-in strategy definitions
CHAIN_OF_THOUGHT = Strategy(
    id="chain_of_thought",
    name="Chain of Thought",
    description="Step-by-step reasoning that breaks down the problem into sequential logical steps.",
    parameters={
        "step_prefix": "Step",
        "max_steps": 10,
        "require_conclusion": True,
    },
    prompt_template="""Think through this problem step by step.

Problem: {problem}

For each step:
1. State what you're considering
2. Explain your reasoning
3. Draw a conclusion for that step

After all steps, provide your final answer.""",
)

DECOMPOSITION = Strategy(
    id="decomposition",
    name="Decomposition",
    description="Break the problem into independent subproblems, solve each, then combine.",
    parameters={
        "max_subproblems": 5,
        "parallel_solve": True,
        "combination_strategy": "merge",
    },
    prompt_template="""Break this problem into smaller, manageable subproblems.

Problem: {problem}

1. First, identify the key subproblems (aim for 3-5)
2. Solve each subproblem independently
3. Combine the solutions into a final answer

Format:
## Subproblems
[List each subproblem]

## Solutions
[Solve each one]

## Combined Answer
[Final integrated solution]""",
)

TOOL_HEAVY = Strategy(
    id="tool_heavy",
    name="Tool Heavy",
    description="Maximize use of available tools to gather information and verify steps.",
    parameters={
        "prefer_tools": True,
        "verify_with_tools": True,
        "tool_first": True,
    },
    prompt_template="""Solve this problem by making maximum use of available tools.

Problem: {problem}

Approach:
1. Identify what tools might help
2. Use tools to gather information before reasoning
3. Verify your conclusions with tools when possible
4. Document which tools you used and why

Prioritize tool use over pure reasoning where applicable.""",
)

ANALOGICAL = Strategy(
    id="analogical",
    name="Analogical Reasoning",
    description="Find similar problems or situations and reason by analogy.",
    parameters={
        "num_analogies": 3,
        "require_mapping": True,
        "check_disanalogies": True,
    },
    prompt_template="""Solve this problem by finding and reasoning from analogies.

Problem: {problem}

1. Think of 2-3 similar problems or situations you know how to solve
2. For each analogy:
   - Describe the analogous situation
   - Map the elements to the current problem
   - Note what doesn't map (disanalogies)
3. Use the strongest analogy to guide your solution
4. Adjust for any disanalogies

Format:
## Analogies
[List analogous situations]

## Best Analogy Mapping
[Show correspondence]

## Solution Based on Analogy
[Apply the analogy]""",
)

ADVERSARIAL = Strategy(
    id="adversarial",
    name="Adversarial Thinking",
    description="Consider failure modes, edge cases, and counterarguments first.",
    parameters={
        "num_failure_modes": 3,
        "require_mitigations": True,
        "devil_advocate": True,
    },
    prompt_template="""Approach this problem by considering what could go wrong.

Problem: {problem}

1. What are the most likely failure modes?
2. What edge cases might break naive solutions?
3. What would a critic say about obvious approaches?
4. Design a solution that addresses these concerns

Format:
## Potential Failures
[List failure modes]

## Edge Cases
[Identify tricky scenarios]

## Robust Solution
[Solution that handles the above]""",
)

MINIMAL = Strategy(
    id="minimal",
    name="Minimal Approach",
    description="Find the simplest possible solution that works.",
    parameters={
        "max_complexity": "low",
        "prefer_simple": True,
        "iterate_if_needed": True,
    },
    prompt_template="""Find the simplest solution to this problem.

Problem: {problem}

Approach:
1. What is the absolute minimum needed to solve this?
2. Start with the simplest approach
3. Only add complexity if the simple approach fails
4. Justify any added complexity

Prefer:
- Fewer steps over more steps
- Simple logic over clever tricks
- Direct solutions over elegant ones""",
)

EXHAUSTIVE = Strategy(
    id="exhaustive",
    name="Exhaustive Exploration",
    description="Thoroughly explore all possibilities before deciding.",
    parameters={
        "explore_all": True,
        "document_rejected": True,
        "comparison_matrix": True,
    },
    prompt_template="""Thoroughly explore all approaches to this problem.

Problem: {problem}

1. List ALL possible approaches (aim for at least 5)
2. For each approach:
   - Describe how it would work
   - List pros and cons
   - Estimate effort and likelihood of success
3. Create a comparison matrix
4. Select the best approach with justification

Format:
## All Approaches
[List every approach]

## Analysis
[Pros/cons for each]

## Comparison Matrix
[Compare on key dimensions]

## Selected Approach
[Best choice with reasoning]""",
)


# Registry of all built-in strategies
BUILTIN_STRATEGIES: dict[str, Strategy] = {
    "chain_of_thought": CHAIN_OF_THOUGHT,
    "decomposition": DECOMPOSITION,
    "tool_heavy": TOOL_HEAVY,
    "analogical": ANALOGICAL,
    "adversarial": ADVERSARIAL,
    "minimal": MINIMAL,
    "exhaustive": EXHAUSTIVE,
}


class StrategyApplicator:
    """Applies strategies to problems and generates prompts."""

    def __init__(self, strategy: Strategy):
        """Initialize with a strategy.

        Args:
            strategy: The strategy to apply.
        """
        self.strategy = strategy

    def apply(self, problem: str, context: dict[str, Any] | None = None) -> str:
        """Apply the strategy to a problem to generate a prompt.

        Args:
            problem: The problem to solve.
            context: Additional context to include in the prompt.

        Returns:
            The generated prompt string.
        """
        template = self.strategy.prompt_template
        if not template:
            # Default template if none specified
            template = "Problem: {problem}\n\nSolve this problem."

        # Build format dict
        format_dict = {"problem": problem}
        if context:
            format_dict.update(context)

        # Add parameter values to format dict
        for key, value in self.strategy.parameters.items():
            if key not in format_dict:
                format_dict[key] = value

        # Apply formatting - use safe formatting that ignores missing keys
        result = template
        for key, value in format_dict.items():
            result = result.replace("{" + key + "}", str(value))

        return result

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a parameter value from the strategy.

        Args:
            key: Parameter name.
            default: Default value if not found.

        Returns:
            The parameter value.
        """
        return self.strategy.parameters.get(key, default)


def apply_strategy(strategy: Strategy, problem: str, context: dict[str, Any] | None = None) -> str:
    """Convenience function to apply a strategy to a problem.

    Args:
        strategy: The strategy to apply.
        problem: The problem to solve.
        context: Additional context.

    Returns:
        The generated prompt.
    """
    applicator = StrategyApplicator(strategy)
    return applicator.apply(problem, context)


def get_builtin_strategy(name: str) -> Strategy | None:
    """Get a built-in strategy by name.

    Args:
        name: Strategy name or ID.

    Returns:
        The strategy if found, None otherwise.
    """
    return BUILTIN_STRATEGIES.get(name)


def list_builtin_strategies() -> list[Strategy]:
    """Get all built-in strategies.

    Returns:
        List of all built-in strategies.
    """
    return list(BUILTIN_STRATEGIES.values())


def save_strategy(strategy: Strategy, path: Path | str) -> None:
    """Save a strategy to a JSON file.

    Args:
        strategy: The strategy to save.
        path: File path to save to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(strategy.to_dict(), f, indent=2)


def load_strategy(path: Path | str) -> Strategy:
    """Load a strategy from a JSON file.

    Args:
        path: File path to load from.

    Returns:
        The loaded strategy.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is invalid JSON.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return Strategy.from_dict(data)


def save_strategies(strategies: list[Strategy], path: Path | str) -> None:
    """Save multiple strategies to a JSON file.

    Args:
        strategies: List of strategies to save.
        path: File path to save to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([s.to_dict() for s in strategies], f, indent=2)


def load_strategies(path: Path | str) -> list[Strategy]:
    """Load multiple strategies from a JSON file.

    Args:
        path: File path to load from.

    Returns:
        List of loaded strategies.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is invalid JSON.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return [Strategy.from_dict(s) for s in data]


def create_custom_strategy(
    id: str,
    name: str,
    description: str,
    prompt_template: str,
    parameters: dict[str, Any] | None = None,
    parent_ids: list[str] | None = None,
) -> Strategy:
    """Create a custom strategy.

    Args:
        id: Unique identifier.
        name: Human-readable name.
        description: What the strategy does.
        prompt_template: Template for generating prompts.
        parameters: Configuration parameters.
        parent_ids: IDs of parent strategies (for genealogy).

    Returns:
        A new Strategy instance.
    """
    return Strategy(
        id=id,
        name=name,
        description=description,
        prompt_template=prompt_template,
        parameters=parameters or {},
        parent_ids=parent_ids or [],
    )


# Strategy composition helpers
def compose_strategies(
    strategies: list[Strategy],
    composition_type: str = "sequential",
) -> Strategy:
    """Compose multiple strategies into a single strategy.

    Args:
        strategies: Strategies to compose.
        composition_type: How to compose ("sequential", "parallel", "conditional").

    Returns:
        A new composed strategy.
    """
    if not strategies:
        raise ValueError("At least one strategy required")

    if len(strategies) == 1:
        return strategies[0]

    ids = [s.id for s in strategies]
    names = [s.name for s in strategies]
    composed_id = f"composed_{composition_type}_{'_'.join(ids[:3])}"

    if composition_type == "sequential":
        # Execute strategies in sequence
        template_parts = []
        for i, s in enumerate(strategies, 1):
            template_parts.append(f"## Phase {i}: {s.name}\n{s.prompt_template}")
        combined_template = "\n\n".join(template_parts)

    elif composition_type == "parallel":
        # Consider all strategies simultaneously
        template_parts = [f"Consider these {len(strategies)} approaches:\n"]
        for i, s in enumerate(strategies, 1):
            template_parts.append(f"### Approach {i}: {s.name}\n{s.description}")
        template_parts.append("\nProblem: {problem}\n\nApply insights from all approaches.")
        combined_template = "\n".join(template_parts)

    elif composition_type == "conditional":
        # Choose strategy based on problem
        template_parts = [
            "Based on the problem type, choose the most appropriate approach:\n"
        ]
        for s in strategies:
            template_parts.append(f"- {s.name}: {s.description}")
        template_parts.append("\nProblem: {problem}\n\nFirst select approach, then apply it.")
        combined_template = "\n".join(template_parts)

    else:
        raise ValueError(f"Unknown composition type: {composition_type}")

    # Merge parameters
    merged_params = {}
    for s in strategies:
        merged_params.update(s.parameters)
    merged_params["composition_type"] = composition_type

    return Strategy(
        id=composed_id,
        name=f"Composed: {' + '.join(names[:3])}{'...' if len(names) > 3 else ''}",
        description=f"{composition_type.title()} composition of {len(strategies)} strategies",
        parameters=merged_params,
        prompt_template=combined_template,
        parent_ids=ids,
    )
