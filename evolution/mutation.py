"""Mutation operators for evolutionary code synthesis.

Each mutation operator uses LLM to generate code variants that
explore different aspects of the solution space:
- Algorithm: Try fundamentally different approaches
- Optimize: Improve performance characteristics
- Simplify: Reduce complexity and improve clarity
- Generalize: Make code more reusable
- Random: LLM-guided exploration
"""

import random
import uuid
from typing import Any, Protocol

from .genome import create_genome, genome_hash, is_valid_code
from .types import Genome, MutationOp, MutationType


class LLMProvider(Protocol):
    """Protocol for LLM providers used in mutation."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt."""
        ...


def _build_mutation_prompt(
    genome: Genome, mutation_type: MutationType, task_prompt: str
) -> str:
    """Build the prompt for LLM-guided mutation.

    Args:
        genome: The genome to mutate
        mutation_type: Type of mutation to apply
        task_prompt: Original task description

    Returns:
        Prompt string for LLM
    """
    base_context = f"""You are mutating code as part of an evolutionary code synthesis system.

Original task: {task_prompt}

Current code:
```python
{genome.code}
```

"""

    mutation_instructions = {
        MutationType.ALGORITHM: """Generate a variant that uses a DIFFERENT algorithm or approach.
- Consider alternative data structures
- Try different algorithmic paradigms (iterative vs recursive, etc.)
- Explore fundamentally different solutions to the same problem
- Maintain the same input/output contract""",
        MutationType.OPTIMIZE: """Generate a variant optimized for PERFORMANCE.
- Reduce time complexity if possible
- Minimize memory usage
- Cache repeated computations
- Use more efficient data structures
- Maintain correctness while improving speed""",
        MutationType.SIMPLIFY: """Generate a SIMPLER variant of this code.
- Remove unnecessary complexity
- Improve readability
- Reduce lines of code while maintaining functionality
- Use clearer variable names
- Eliminate redundant operations""",
        MutationType.GENERALIZE: """Generate a MORE GENERAL variant of this code.
- Add parameters to make it more flexible
- Handle more edge cases
- Make it work with more input types
- Extract reusable components
- Improve extensibility""",
        MutationType.RANDOM: """Generate a CREATIVE variant of this code.
- Try something unexpected
- Combine ideas in new ways
- Explore unconventional approaches
- Maintain correctness but be inventive""",
    }

    return (
        base_context
        + mutation_instructions[mutation_type]
        + """

Return ONLY the Python code, no explanations. The code must be valid Python.
```python
"""
    )


def _extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response.

    Handles markdown code blocks and raw code.
    """
    # Try to extract from markdown code block
    if "```python" in response:
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    # Return as-is, trimmed
    return response.strip()


async def mutate_algorithm(
    genome: Genome, prompt: str, llm: LLMProvider
) -> Genome | None:
    """Try a different algorithm for the same problem.

    Uses LLM to generate a structurally different solution that
    maintains the same input/output contract.

    Args:
        genome: The genome to mutate
        prompt: Original task description
        llm: LLM provider for generation

    Returns:
        New genome with different algorithm, or None if mutation fails
    """
    mutation_prompt = _build_mutation_prompt(genome, MutationType.ALGORITHM, prompt)

    try:
        response = await llm.generate(mutation_prompt)
        new_code = _extract_code_from_response(response)

        if not is_valid_code(new_code):
            return None

        new_genome = create_genome(
            code=new_code,
            generation=genome.generation + 1,
            parent_ids=[genome.id],
            lineage=genome.lineage + [genome.id],
        )
        new_genome.mutation_history = genome.mutation_history + ["algorithm"]

        return new_genome

    except Exception:
        return None


async def mutate_optimize(
    genome: Genome, prompt: str, llm: LLMProvider
) -> Genome | None:
    """Optimize code for performance.

    Uses LLM to improve time/space complexity while maintaining
    correctness.

    Args:
        genome: The genome to mutate
        prompt: Original task description
        llm: LLM provider for generation

    Returns:
        Optimized genome, or None if mutation fails
    """
    mutation_prompt = _build_mutation_prompt(genome, MutationType.OPTIMIZE, prompt)

    try:
        response = await llm.generate(mutation_prompt)
        new_code = _extract_code_from_response(response)

        if not is_valid_code(new_code):
            return None

        new_genome = create_genome(
            code=new_code,
            generation=genome.generation + 1,
            parent_ids=[genome.id],
            lineage=genome.lineage + [genome.id],
        )
        new_genome.mutation_history = genome.mutation_history + ["optimize"]

        return new_genome

    except Exception:
        return None


async def mutate_simplify(
    genome: Genome, prompt: str, llm: LLMProvider
) -> Genome | None:
    """Reduce complexity and improve clarity.

    Uses LLM to simplify code while maintaining functionality.

    Args:
        genome: The genome to mutate
        prompt: Original task description
        llm: LLM provider for generation

    Returns:
        Simplified genome, or None if mutation fails
    """
    mutation_prompt = _build_mutation_prompt(genome, MutationType.SIMPLIFY, prompt)

    try:
        response = await llm.generate(mutation_prompt)
        new_code = _extract_code_from_response(response)

        if not is_valid_code(new_code):
            return None

        new_genome = create_genome(
            code=new_code,
            generation=genome.generation + 1,
            parent_ids=[genome.id],
            lineage=genome.lineage + [genome.id],
        )
        new_genome.mutation_history = genome.mutation_history + ["simplify"]

        return new_genome

    except Exception:
        return None


async def mutate_generalize(
    genome: Genome, prompt: str, llm: LLMProvider
) -> Genome | None:
    """Make code more generic and reusable.

    Uses LLM to add flexibility and handle more cases.

    Args:
        genome: The genome to mutate
        prompt: Original task description
        llm: LLM provider for generation

    Returns:
        Generalized genome, or None if mutation fails
    """
    mutation_prompt = _build_mutation_prompt(genome, MutationType.GENERALIZE, prompt)

    try:
        response = await llm.generate(mutation_prompt)
        new_code = _extract_code_from_response(response)

        if not is_valid_code(new_code):
            return None

        new_genome = create_genome(
            code=new_code,
            generation=genome.generation + 1,
            parent_ids=[genome.id],
            lineage=genome.lineage + [genome.id],
        )
        new_genome.mutation_history = genome.mutation_history + ["generalize"]

        return new_genome

    except Exception:
        return None


async def random_mutation(genome: Genome, llm: LLMProvider) -> Genome | None:
    """Apply LLM-guided random change.

    Encourages creative exploration without specific direction.

    Args:
        genome: The genome to mutate
        llm: LLM provider for generation

    Returns:
        Mutated genome, or None if mutation fails
    """
    mutation_prompt = _build_mutation_prompt(
        genome, MutationType.RANDOM, "Generate a creative variant"
    )

    try:
        response = await llm.generate(mutation_prompt)
        new_code = _extract_code_from_response(response)

        if not is_valid_code(new_code):
            return None

        new_genome = create_genome(
            code=new_code,
            generation=genome.generation + 1,
            parent_ids=[genome.id],
            lineage=genome.lineage + [genome.id],
        )
        new_genome.mutation_history = genome.mutation_history + ["random"]

        return new_genome

    except Exception:
        return None


def select_mutation_type(genome: Genome) -> MutationType:
    """Select which mutation type to apply.

    Uses history-aware selection to avoid repeating the same
    mutation type too often.

    Args:
        genome: The genome to mutate

    Returns:
        Selected mutation type
    """
    all_types = list(MutationType)

    # Weight against recently used types
    recent = genome.mutation_history[-3:] if genome.mutation_history else []
    weights = []

    for mt in all_types:
        weight = 1.0
        # Reduce weight for recently used types
        if mt.name.lower() in recent:
            weight *= 0.3
        weights.append(weight)

    return random.choices(all_types, weights=weights, k=1)[0]


async def apply_mutation(
    genome: Genome, prompt: str, llm: LLMProvider, mutation_type: MutationType | None = None
) -> tuple[Genome | None, MutationOp]:
    """Apply a mutation to a genome.

    Selects mutation type if not specified and applies it.

    Args:
        genome: The genome to mutate
        prompt: Original task description
        llm: LLM provider for generation
        mutation_type: Specific type to apply, or None to auto-select

    Returns:
        Tuple of (mutated genome or None, mutation operation record)
    """
    if mutation_type is None:
        mutation_type = select_mutation_type(genome)

    mutation_op = MutationOp(
        type=mutation_type,
        description=f"Apply {mutation_type.name.lower()} mutation",
        prompt=prompt,
        source_genome_id=genome.id,
    )

    mutation_functions = {
        MutationType.ALGORITHM: mutate_algorithm,
        MutationType.OPTIMIZE: mutate_optimize,
        MutationType.SIMPLIFY: mutate_simplify,
        MutationType.GENERALIZE: mutate_generalize,
        MutationType.RANDOM: random_mutation,
    }

    if mutation_type == MutationType.RANDOM:
        result = await random_mutation(genome, llm)
    else:
        mutation_func = mutation_functions[mutation_type]
        result = await mutation_func(genome, prompt, llm)

    if result is not None:
        mutation_op.result_genome_id = result.id
        mutation_op.success = True
    else:
        mutation_op.success = False

    return result, mutation_op


def point_mutation(genome: Genome, position: int, new_char: str) -> Genome | None:
    """Apply a simple point mutation at the character level.

    This is a low-level mutation for fine-grained changes.

    Args:
        genome: The genome to mutate
        position: Character position to mutate
        new_char: New character to insert

    Returns:
        Mutated genome, or None if position is invalid
    """
    if position < 0 or position >= len(genome.code):
        return None

    new_code = genome.code[:position] + new_char + genome.code[position + 1 :]

    if not is_valid_code(new_code):
        return None

    return create_genome(
        code=new_code,
        generation=genome.generation + 1,
        parent_ids=[genome.id],
        lineage=genome.lineage + [genome.id],
    )


def deletion_mutation(genome: Genome, start: int, end: int) -> Genome | None:
    """Delete a range of characters from the genome.

    Args:
        genome: The genome to mutate
        start: Start position (inclusive)
        end: End position (exclusive)

    Returns:
        Mutated genome, or None if range is invalid
    """
    if start < 0 or end > len(genome.code) or start >= end:
        return None

    new_code = genome.code[:start] + genome.code[end:]

    if not is_valid_code(new_code):
        return None

    return create_genome(
        code=new_code,
        generation=genome.generation + 1,
        parent_ids=[genome.id],
        lineage=genome.lineage + [genome.id],
    )


def insertion_mutation(genome: Genome, position: int, code_to_insert: str) -> Genome | None:
    """Insert code at a specific position.

    Args:
        genome: The genome to mutate
        position: Position to insert at
        code_to_insert: Code string to insert

    Returns:
        Mutated genome, or None if position is invalid
    """
    if position < 0 or position > len(genome.code):
        return None

    new_code = genome.code[:position] + code_to_insert + genome.code[position:]

    if not is_valid_code(new_code):
        return None

    return create_genome(
        code=new_code,
        generation=genome.generation + 1,
        parent_ids=[genome.id],
        lineage=genome.lineage + [genome.id],
    )
