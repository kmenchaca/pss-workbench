"""Strategy mutation for the meta-learner system.

This module provides mutation operators to create strategy variants,
enabling evolutionary improvement of strategies over time.
"""

import random
import uuid
from datetime import datetime
from typing import Any, Callable

from .types import MutationType, Strategy, StrategyMutation


# Mutation parameter ranges
PARAMETER_TWEAKS = {
    "max_steps": (1, 20),
    "max_subproblems": (2, 10),
    "num_analogies": (1, 5),
    "num_failure_modes": (1, 10),
}

# Prompt modification templates
PROMPT_ADDITIONS = [
    "\n\nBe concise and focus on the core solution.",
    "\n\nExplain your reasoning at each step.",
    "\n\nConsider multiple perspectives before answering.",
    "\n\nStart with the simplest approach, then add complexity if needed.",
    "\n\nIdentify potential issues before proceeding.",
    "\n\nUse specific examples to illustrate your points.",
    "\n\nPrioritize accuracy over completeness.",
    "\n\nStructure your response with clear sections.",
]

PROMPT_PREFIXES = [
    "Take a deep breath and ",
    "Think carefully: ",
    "Consider this systematically: ",
    "Before answering, consider edge cases. ",
    "Let's approach this methodically. ",
]


class StrategyMutator:
    """Creates mutations of strategies for evolutionary improvement."""

    def __init__(self, rng: random.Random | None = None):
        """Initialize the mutator.

        Args:
            rng: Random number generator for reproducibility.
        """
        self.rng = rng or random.Random()

    def generate_id(self) -> str:
        """Generate a unique ID for a mutated strategy."""
        return f"mutated_{uuid.uuid4().hex[:8]}"

    def mutate(
        self,
        strategy: Strategy,
        mutation_type: MutationType | None = None,
    ) -> StrategyMutation:
        """Create a mutated variant of a strategy.

        Args:
            strategy: The strategy to mutate.
            mutation_type: Specific mutation to apply, or None for random.

        Returns:
            A StrategyMutation containing the new strategy.
        """
        if mutation_type is None:
            mutation_type = self.rng.choice([
                MutationType.PARAMETER_TWEAK,
                MutationType.PROMPT_EDIT,
                MutationType.SIMPLIFY,
                MutationType.EXTEND,
            ])

        if mutation_type == MutationType.PARAMETER_TWEAK:
            return self._tweak_parameters(strategy)
        elif mutation_type == MutationType.PROMPT_EDIT:
            return self._edit_prompt(strategy)
        elif mutation_type == MutationType.SIMPLIFY:
            return self._simplify(strategy)
        elif mutation_type == MutationType.EXTEND:
            return self._extend(strategy)
        else:
            # Default to parameter tweak
            return self._tweak_parameters(strategy)

    def _tweak_parameters(self, strategy: Strategy) -> StrategyMutation:
        """Mutate by adjusting parameter values."""
        new_params = dict(strategy.parameters)

        # Find tweakable parameters
        tweakable = [
            k for k in new_params
            if k in PARAMETER_TWEAKS or isinstance(new_params[k], (int, float, bool))
        ]

        if tweakable:
            param = self.rng.choice(tweakable)
            old_value = new_params[param]

            if param in PARAMETER_TWEAKS:
                min_val, max_val = PARAMETER_TWEAKS[param]
                new_value = self.rng.randint(min_val, max_val)
            elif isinstance(old_value, bool):
                new_value = not old_value
            elif isinstance(old_value, int):
                delta = self.rng.randint(-3, 3)
                new_value = max(1, old_value + delta)
            elif isinstance(old_value, float):
                delta = self.rng.uniform(-0.2, 0.2)
                new_value = max(0.0, min(1.0, old_value + delta))
            else:
                new_value = old_value

            new_params[param] = new_value
            description = f"Changed {param} from {old_value} to {new_value}"
        else:
            # Add a new parameter if none are tweakable
            new_params["mutation_iteration"] = 1
            description = "Added mutation_iteration parameter"

        new_strategy = Strategy(
            id=self.generate_id(),
            name=f"{strategy.name} (tweaked)",
            description=f"Parameter-tweaked variant of {strategy.name}",
            parameters=new_params,
            prompt_template=strategy.prompt_template,
            created_at=datetime.now(),
            parent_ids=[strategy.id],
        )

        return StrategyMutation(
            original_id=strategy.id,
            mutation_type=MutationType.PARAMETER_TWEAK,
            new_strategy=new_strategy,
            description=description,
        )

    def _edit_prompt(self, strategy: Strategy) -> StrategyMutation:
        """Mutate by modifying the prompt template."""
        template = strategy.prompt_template

        # Choose modification type
        mod_type = self.rng.choice(["add_suffix", "add_prefix", "restructure"])

        if mod_type == "add_suffix":
            addition = self.rng.choice(PROMPT_ADDITIONS)
            new_template = template + addition
            description = f"Added suffix: {addition[:30]}..."
        elif mod_type == "add_prefix":
            prefix = self.rng.choice(PROMPT_PREFIXES)
            new_template = prefix + template
            description = f"Added prefix: {prefix}"
        else:
            # Restructure by adding section headers
            if "##" not in template:
                lines = template.split("\n\n")
                if len(lines) > 1:
                    new_lines = []
                    for i, line in enumerate(lines):
                        if line.strip():
                            new_lines.append(f"## Part {i + 1}\n{line}")
                    new_template = "\n\n".join(new_lines)
                    description = "Added section structure"
                else:
                    new_template = template + "\n\n## Response\n"
                    description = "Added response section"
            else:
                new_template = template
                description = "No restructure needed"

        new_strategy = Strategy(
            id=self.generate_id(),
            name=f"{strategy.name} (edited)",
            description=f"Prompt-edited variant of {strategy.name}",
            parameters=dict(strategy.parameters),
            prompt_template=new_template,
            created_at=datetime.now(),
            parent_ids=[strategy.id],
        )

        return StrategyMutation(
            original_id=strategy.id,
            mutation_type=MutationType.PROMPT_EDIT,
            new_strategy=new_strategy,
            description=description,
        )

    def _simplify(self, strategy: Strategy) -> StrategyMutation:
        """Mutate by simplifying the strategy."""
        # Reduce parameters
        new_params = dict(strategy.parameters)
        for key in list(new_params.keys()):
            if isinstance(new_params[key], int) and new_params[key] > 3:
                new_params[key] = max(1, new_params[key] // 2)
            elif isinstance(new_params[key], bool) and key.startswith("require_"):
                new_params[key] = False

        # Simplify prompt
        template = strategy.prompt_template
        lines = template.split("\n")

        # Remove lines that seem optional
        simplified_lines = []
        for line in lines:
            # Keep essential lines
            if line.strip() and not any(
                skip in line.lower()
                for skip in ["optional", "if possible", "consider also", "you may also"]
            ):
                simplified_lines.append(line)

        new_template = "\n".join(simplified_lines)

        new_strategy = Strategy(
            id=self.generate_id(),
            name=f"{strategy.name} (simplified)",
            description=f"Simplified variant of {strategy.name}",
            parameters=new_params,
            prompt_template=new_template,
            created_at=datetime.now(),
            parent_ids=[strategy.id],
        )

        return StrategyMutation(
            original_id=strategy.id,
            mutation_type=MutationType.SIMPLIFY,
            new_strategy=new_strategy,
            description="Reduced parameters and simplified prompt",
        )

    def _extend(self, strategy: Strategy) -> StrategyMutation:
        """Mutate by extending the strategy with more detail."""
        # Increase parameters
        new_params = dict(strategy.parameters)
        for key in list(new_params.keys()):
            if isinstance(new_params[key], int):
                new_params[key] = new_params[key] + self.rng.randint(1, 3)

        # Extend prompt
        template = strategy.prompt_template

        extensions = [
            "\n\n## Verification\nBefore finalizing, verify your answer is correct.",
            "\n\n## Alternatives\nBriefly consider at least one alternative approach.",
            "\n\n## Edge Cases\nMention any edge cases or limitations.",
            "\n\n## Summary\nProvide a brief summary of your solution.",
        ]

        extension = self.rng.choice(extensions)
        new_template = template + extension

        new_strategy = Strategy(
            id=self.generate_id(),
            name=f"{strategy.name} (extended)",
            description=f"Extended variant of {strategy.name}",
            parameters=new_params,
            prompt_template=new_template,
            created_at=datetime.now(),
            parent_ids=[strategy.id],
        )

        return StrategyMutation(
            original_id=strategy.id,
            mutation_type=MutationType.EXTEND,
            new_strategy=new_strategy,
            description=f"Extended with: {extension[:30]}...",
        )


def crossover_strategies(
    strategy1: Strategy,
    strategy2: Strategy,
    rng: random.Random | None = None,
) -> Strategy:
    """Combine two strategies to create an offspring.

    Args:
        strategy1: First parent strategy.
        strategy2: Second parent strategy.
        rng: Random number generator.

    Returns:
        A new strategy combining elements of both parents.
    """
    rng = rng or random.Random()

    # Combine parameters
    new_params = {}
    all_keys = set(strategy1.parameters.keys()) | set(strategy2.parameters.keys())
    for key in all_keys:
        if key in strategy1.parameters and key in strategy2.parameters:
            # Both have the parameter - randomly choose
            source = rng.choice([strategy1, strategy2])
            new_params[key] = source.parameters[key]
        elif key in strategy1.parameters:
            new_params[key] = strategy1.parameters[key]
        else:
            new_params[key] = strategy2.parameters[key]

    # Combine prompts by interleaving sections
    lines1 = strategy1.prompt_template.split("\n\n")
    lines2 = strategy2.prompt_template.split("\n\n")

    combined_lines = []
    max_len = max(len(lines1), len(lines2))

    for i in range(max_len):
        if i < len(lines1) and rng.random() < 0.6:
            combined_lines.append(lines1[i])
        if i < len(lines2) and rng.random() < 0.6:
            combined_lines.append(lines2[i])

    if not combined_lines:
        combined_lines = lines1 if rng.random() < 0.5 else lines2

    new_template = "\n\n".join(combined_lines)

    return Strategy(
        id=f"crossover_{uuid.uuid4().hex[:8]}",
        name=f"Crossover: {strategy1.name[:15]} x {strategy2.name[:15]}",
        description=f"Crossover of {strategy1.name} and {strategy2.name}",
        parameters=new_params,
        prompt_template=new_template,
        created_at=datetime.now(),
        parent_ids=[strategy1.id, strategy2.id],
    )


def random_strategy(rng: random.Random | None = None) -> Strategy:
    """Generate a completely novel random strategy.

    Args:
        rng: Random number generator.

    Returns:
        A new randomly generated strategy.
    """
    rng = rng or random.Random()

    # Random components
    approaches = [
        "step by step",
        "by analogy",
        "from first principles",
        "through decomposition",
        "iteratively",
        "exhaustively",
        "minimally",
    ]

    focuses = [
        "accuracy",
        "efficiency",
        "clarity",
        "completeness",
        "simplicity",
        "robustness",
    ]

    structures = [
        "numbered list",
        "bullet points",
        "prose paragraphs",
        "hierarchical outline",
        "Q&A format",
    ]

    approach = rng.choice(approaches)
    focus = rng.choice(focuses)
    structure = rng.choice(structures)

    template = f"""Solve this problem {approach}, prioritizing {focus}.

Problem: {{problem}}

Present your solution using {structure}.

Steps:
1. Understand the problem
2. Apply the {approach} method
3. Verify your answer focuses on {focus}
4. Format the response as {structure}"""

    parameters = {
        "approach": approach,
        "focus": focus,
        "structure": structure,
        "max_iterations": rng.randint(1, 10),
        "verify": rng.random() < 0.5,
    }

    return Strategy(
        id=f"random_{uuid.uuid4().hex[:8]}",
        name=f"Random: {approach.title()} + {focus.title()}",
        description=f"Randomly generated strategy: {approach} approach with {focus} focus",
        parameters=parameters,
        prompt_template=template,
        created_at=datetime.now(),
        parent_ids=[],
    )


async def llm_guided_mutation(
    strategy: Strategy,
    llm_call: Callable[[str], Any],
    feedback: str | None = None,
) -> StrategyMutation:
    """Use an LLM to intelligently mutate a strategy.

    Args:
        strategy: The strategy to mutate.
        llm_call: Async function to call the LLM.
        feedback: Optional feedback about strategy performance.

    Returns:
        A StrategyMutation with the LLM-designed changes.
    """
    feedback_section = ""
    if feedback:
        feedback_section = f"\n\nPerformance feedback:\n{feedback}"

    prompt = f"""Improve this exploration strategy by making ONE focused change.

Current Strategy:
Name: {strategy.name}
Description: {strategy.description}
Prompt Template:
{strategy.prompt_template}
{feedback_section}

Suggest ONE improvement. Be specific and actionable.

Respond with:
CHANGE_TYPE: [parameter_tweak | prompt_edit | simplify | extend]
CHANGE_DESCRIPTION: [what you're changing]
NEW_PROMPT_TEMPLATE: [complete new template, or UNCHANGED if not modifying]
NEW_PARAMETERS: [JSON of parameters, or UNCHANGED if not modifying]"""

    response = await llm_call(prompt)

    # Parse response
    lines = response.split("\n")
    change_type = "prompt_edit"
    description = "LLM-guided improvement"
    new_template = strategy.prompt_template
    new_params = dict(strategy.parameters)

    in_template = False
    template_lines = []

    for line in lines:
        if line.startswith("CHANGE_TYPE:"):
            change_type = line.split(":", 1)[1].strip().lower()
        elif line.startswith("CHANGE_DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()
        elif line.startswith("NEW_PROMPT_TEMPLATE:"):
            content = line.split(":", 1)[1].strip()
            if content != "UNCHANGED":
                in_template = True
                if content:
                    template_lines.append(content)
        elif line.startswith("NEW_PARAMETERS:"):
            in_template = False
            if template_lines:
                new_template = "\n".join(template_lines)
            content = line.split(":", 1)[1].strip()
            if content != "UNCHANGED":
                try:
                    import json
                    new_params = json.loads(content)
                except json.JSONDecodeError:
                    pass
        elif in_template:
            template_lines.append(line)

    if template_lines and in_template:
        new_template = "\n".join(template_lines)

    # Map change type
    mutation_map = {
        "parameter_tweak": MutationType.PARAMETER_TWEAK,
        "prompt_edit": MutationType.PROMPT_EDIT,
        "simplify": MutationType.SIMPLIFY,
        "extend": MutationType.EXTEND,
    }
    mutation_type = mutation_map.get(change_type, MutationType.PROMPT_EDIT)

    new_strategy = Strategy(
        id=f"llm_mutated_{uuid.uuid4().hex[:8]}",
        name=f"{strategy.name} (LLM improved)",
        description=f"LLM-improved variant: {description}",
        parameters=new_params,
        prompt_template=new_template,
        created_at=datetime.now(),
        parent_ids=[strategy.id],
    )

    return StrategyMutation(
        original_id=strategy.id,
        mutation_type=mutation_type,
        new_strategy=new_strategy,
        description=f"LLM-guided: {description}",
    )


# Convenience functions
_default_mutator: StrategyMutator | None = None


def get_mutator() -> StrategyMutator:
    """Get the default mutator instance."""
    global _default_mutator
    if _default_mutator is None:
        _default_mutator = StrategyMutator()
    return _default_mutator


def mutate_strategy(
    strategy: Strategy,
    mutation_type: MutationType | None = None,
) -> StrategyMutation:
    """Mutate a strategy using the default mutator.

    Args:
        strategy: The strategy to mutate.
        mutation_type: Optional specific mutation type.

    Returns:
        A StrategyMutation with the new strategy.
    """
    return get_mutator().mutate(strategy, mutation_type)
