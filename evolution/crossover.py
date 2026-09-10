"""Crossover breeding operations for evolutionary code synthesis.

Implements various crossover strategies for combining genetic material
from two parent genomes to produce offspring:
- Single-point: Swap at one point
- Uniform: Mix functions randomly
- Semantic: LLM combines best parts
- Gene transfer: Transplant specific patterns
"""

import ast
import random
from typing import Any, Protocol

from .genome import (
    ast_to_code,
    create_genome,
    extract_genes,
    get_functions,
    is_valid_code,
    merge_functions,
    parse_to_ast,
)
from .types import CrossoverResult, Gene, Genome


class LLMProvider(Protocol):
    """Protocol for LLM providers used in semantic crossover."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt."""
        ...


def single_point_crossover(g1: Genome, g2: Genome) -> CrossoverResult:
    """Swap code at one randomly selected point.

    Selects a crossover point in each parent and swaps the code
    after that point to create two offspring.

    Args:
        g1: First parent genome
        g2: Second parent genome

    Returns:
        CrossoverResult with offspring
    """
    # Find valid crossover points (line boundaries)
    lines1 = g1.code.split("\n")
    lines2 = g2.code.split("\n")

    if len(lines1) < 2 or len(lines2) < 2:
        return CrossoverResult(
            success=False, error="Parents too short for crossover"
        )

    # Select crossover points
    point1 = random.randint(1, len(lines1) - 1)
    point2 = random.randint(1, len(lines2) - 1)

    # Create offspring by swapping
    offspring1_code = "\n".join(lines1[:point1] + lines2[point2:])
    offspring2_code = "\n".join(lines2[:point2] + lines1[point1:])

    offspring = []
    contribution1 = point1 / len(lines1)
    contribution2 = point2 / len(lines2)

    # Create offspring if valid
    if is_valid_code(offspring1_code):
        offspring.append(
            create_genome(
                code=offspring1_code,
                generation=max(g1.generation, g2.generation) + 1,
                parent_ids=[g1.id, g2.id],
                lineage=list(set(g1.lineage + g2.lineage + [g1.id, g2.id])),
            )
        )

    if is_valid_code(offspring2_code):
        offspring.append(
            create_genome(
                code=offspring2_code,
                generation=max(g1.generation, g2.generation) + 1,
                parent_ids=[g1.id, g2.id],
                lineage=list(set(g1.lineage + g2.lineage + [g1.id, g2.id])),
            )
        )

    return CrossoverResult(
        offspring=offspring,
        crossover_point=f"line {point1} / line {point2}",
        parent1_contribution=contribution1,
        parent2_contribution=1 - contribution1,
        success=len(offspring) > 0,
        error="" if offspring else "No valid offspring produced",
    )


def uniform_crossover(g1: Genome, g2: Genome) -> CrossoverResult:
    """Mix functions randomly from both parents.

    Extracts functions from both parents and randomly selects
    which parent contributes each function to the offspring.

    Args:
        g1: First parent genome
        g2: Second parent genome

    Returns:
        CrossoverResult with offspring
    """
    funcs1 = get_functions(g1)
    funcs2 = get_functions(g2)

    if not funcs1 and not funcs2:
        return CrossoverResult(
            success=False, error="No functions found in parents"
        )

    # Build name to function map
    func_map: dict[str, list[ast.FunctionDef]] = {}

    for f in funcs1:
        if f.name not in func_map:
            func_map[f.name] = []
        func_map[f.name].append(f)

    for f in funcs2:
        if f.name not in func_map:
            func_map[f.name] = []
        func_map[f.name].append(f)

    # Randomly select one version of each function
    selected_funcs: list[ast.FunctionDef] = []
    from_parent1 = 0
    from_parent2 = 0

    for name, versions in func_map.items():
        chosen = random.choice(versions)
        selected_funcs.append(chosen)

        # Track contribution
        if any(f is chosen for f in funcs1):
            from_parent1 += 1
        else:
            from_parent2 += 1

    # Build offspring code
    offspring_code = merge_functions(selected_funcs, [])

    if not is_valid_code(offspring_code):
        return CrossoverResult(
            success=False, error="Merged code is invalid"
        )

    total = from_parent1 + from_parent2
    contribution1 = from_parent1 / total if total > 0 else 0.5

    offspring = create_genome(
        code=offspring_code,
        generation=max(g1.generation, g2.generation) + 1,
        parent_ids=[g1.id, g2.id],
        lineage=list(set(g1.lineage + g2.lineage + [g1.id, g2.id])),
    )

    return CrossoverResult(
        offspring=[offspring],
        crossover_point="uniform function selection",
        parent1_contribution=contribution1,
        parent2_contribution=1 - contribution1,
        success=True,
    )


def _extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response."""
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

    return response.strip()


async def semantic_crossover(
    g1: Genome, g2: Genome, prompt: str, llm: LLMProvider
) -> CrossoverResult:
    """LLM combines best parts from both parents.

    Uses LLM to intelligently merge the best aspects of both
    parent solutions based on the original task.

    Args:
        g1: First parent genome
        g2: Second parent genome
        prompt: Original task description
        llm: LLM provider for generation

    Returns:
        CrossoverResult with offspring
    """
    crossover_prompt = f"""You are performing semantic crossover in an evolutionary code synthesis system.

Original task: {prompt}

Parent 1 code:
```python
{g1.code}
```

Parent 2 code:
```python
{g2.code}
```

Your goal: Create an OFFSPRING that combines the BEST aspects of both parents.
- Identify what each parent does well
- Merge their strengths into a single solution
- The offspring should be at least as good as the better parent
- Maintain the same input/output contract as the parents

Return ONLY the Python code, no explanations.
```python
"""

    try:
        response = await llm.generate(crossover_prompt)
        offspring_code = _extract_code_from_response(response)

        if not is_valid_code(offspring_code):
            return CrossoverResult(
                success=False, error="LLM produced invalid code"
            )

        offspring = create_genome(
            code=offspring_code,
            generation=max(g1.generation, g2.generation) + 1,
            parent_ids=[g1.id, g2.id],
            lineage=list(set(g1.lineage + g2.lineage + [g1.id, g2.id])),
        )

        return CrossoverResult(
            offspring=[offspring],
            crossover_point="semantic merge by LLM",
            parent1_contribution=0.5,
            parent2_contribution=0.5,
            success=True,
        )

    except Exception as e:
        return CrossoverResult(
            success=False, error=f"Semantic crossover failed: {str(e)}"
        )


def gene_transfer(donor: Genome, recipient: Genome, gene: Gene) -> CrossoverResult:
    """Transplant a specific pattern from donor to recipient.

    Extracts a gene (code pattern) from the donor and inserts
    it into the recipient genome.

    Args:
        donor: Genome to take gene from
        recipient: Genome to receive gene
        gene: The specific gene to transfer

    Returns:
        CrossoverResult with modified recipient
    """
    # Parse recipient
    tree = parse_to_ast(recipient.code)
    if tree is None:
        return CrossoverResult(
            success=False, error="Could not parse recipient"
        )

    # Parse the gene pattern
    gene_tree = parse_to_ast(gene.pattern)
    if gene_tree is None:
        return CrossoverResult(
            success=False, error="Could not parse gene pattern"
        )

    # Get the actual node from gene tree
    gene_nodes = list(ast.walk(gene_tree))
    transferable = [n for n in gene_nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))]

    if not transferable:
        return CrossoverResult(
            success=False, error="No transferable nodes in gene"
        )

    gene_node = transferable[0]

    # Add gene to recipient's module body
    if isinstance(tree, ast.Module):
        # Check for name conflicts
        existing_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                existing_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                existing_names.add(node.name)

        # Rename if conflict
        if isinstance(gene_node, (ast.FunctionDef, ast.ClassDef)):
            if gene_node.name in existing_names:
                gene_node.name = f"{gene_node.name}_transferred"

        tree.body.append(gene_node)

    offspring_code = ast_to_code(tree)

    if not is_valid_code(offspring_code):
        return CrossoverResult(
            success=False, error="Gene transfer produced invalid code"
        )

    offspring = create_genome(
        code=offspring_code,
        generation=max(donor.generation, recipient.generation) + 1,
        parent_ids=[donor.id, recipient.id],
        lineage=list(set(donor.lineage + recipient.lineage + [donor.id, recipient.id])),
    )

    return CrossoverResult(
        offspring=[offspring],
        crossover_point=f"gene transfer: {gene.id}",
        parent1_contribution=0.1,  # Donor contributes just the gene
        parent2_contribution=0.9,  # Recipient provides base
        success=True,
    )


def function_swap(g1: Genome, g2: Genome, func_name: str) -> CrossoverResult:
    """Swap a specific function between two genomes.

    Args:
        g1: First parent genome
        g2: Second parent genome
        func_name: Name of function to swap

    Returns:
        CrossoverResult with two offspring (each with swapped function)
    """
    tree1 = parse_to_ast(g1.code)
    tree2 = parse_to_ast(g2.code)

    if tree1 is None or tree2 is None:
        return CrossoverResult(
            success=False, error="Could not parse one or both parents"
        )

    # Find the functions to swap
    func1 = None
    func2 = None
    func1_idx = -1
    func2_idx = -1

    if isinstance(tree1, ast.Module):
        for i, node in enumerate(tree1.body):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                func1 = node
                func1_idx = i
                break

    if isinstance(tree2, ast.Module):
        for i, node in enumerate(tree2.body):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                func2 = node
                func2_idx = i
                break

    if func1 is None or func2 is None:
        return CrossoverResult(
            success=False, error=f"Function '{func_name}' not found in both parents"
        )

    # Perform swap
    offspring = []

    # Offspring 1: g1 with func2
    if isinstance(tree1, ast.Module):
        tree1.body[func1_idx] = func2
        code1 = ast_to_code(tree1)
        if is_valid_code(code1):
            offspring.append(
                create_genome(
                    code=code1,
                    generation=max(g1.generation, g2.generation) + 1,
                    parent_ids=[g1.id, g2.id],
                    lineage=list(set(g1.lineage + g2.lineage + [g1.id, g2.id])),
                )
            )

    # Re-parse for second offspring
    tree1 = parse_to_ast(g1.code)
    tree2 = parse_to_ast(g2.code)

    # Offspring 2: g2 with func1 (need to re-find since we modified)
    if tree1 and tree2 and isinstance(tree1, ast.Module) and isinstance(tree2, ast.Module):
        for i, node in enumerate(tree1.body):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                func1 = node
                break

        tree2.body[func2_idx] = func1
        code2 = ast_to_code(tree2)
        if is_valid_code(code2):
            offspring.append(
                create_genome(
                    code=code2,
                    generation=max(g1.generation, g2.generation) + 1,
                    parent_ids=[g1.id, g2.id],
                    lineage=list(set(g1.lineage + g2.lineage + [g1.id, g2.id])),
                )
            )

    return CrossoverResult(
        offspring=offspring,
        crossover_point=f"function swap: {func_name}",
        parent1_contribution=0.5,
        parent2_contribution=0.5,
        success=len(offspring) > 0,
        error="" if offspring else "No valid offspring from swap",
    )


def select_crossover_parents(
    genomes: list[Genome], strategy: str = "fitness"
) -> tuple[Genome, Genome] | None:
    """Select two parents for crossover.

    Args:
        genomes: Pool of genomes to select from
        strategy: Selection strategy ('fitness', 'random', 'diverse')

    Returns:
        Tuple of two parent genomes, or None if not enough genomes
    """
    if len(genomes) < 2:
        return None

    if strategy == "random":
        return tuple(random.sample(genomes, 2))  # type: ignore

    if strategy == "fitness":
        # Select based on fitness (tournament selection)
        def tournament(k: int = 3) -> Genome:
            contestants = random.sample(genomes, min(k, len(genomes)))
            return max(contestants, key=lambda g: g.fitness)

        p1 = tournament()
        # Ensure p2 is different
        remaining = [g for g in genomes if g.id != p1.id]
        if not remaining:
            remaining = genomes
        p2 = tournament() if len(remaining) >= 3 else random.choice(remaining)
        return (p1, p2)

    if strategy == "diverse":
        # Select genomes that are genetically distant
        from .genome import genome_distance

        p1 = random.choice(genomes)
        distances = [(g, genome_distance(p1, g)) for g in genomes if g.id != p1.id]
        if not distances:
            return (p1, random.choice(genomes))
        # Pick the most distant
        p2 = max(distances, key=lambda x: x[1])[0]
        return (p1, p2)

    return tuple(random.sample(genomes, 2))  # type: ignore


async def apply_crossover(
    g1: Genome,
    g2: Genome,
    strategy: str = "uniform",
    prompt: str = "",
    llm: LLMProvider | None = None,
) -> CrossoverResult:
    """Apply crossover using the specified strategy.

    Args:
        g1: First parent
        g2: Second parent
        strategy: Crossover strategy ('single_point', 'uniform', 'semantic')
        prompt: Task prompt (required for semantic)
        llm: LLM provider (required for semantic)

    Returns:
        CrossoverResult with offspring
    """
    if strategy == "single_point":
        return single_point_crossover(g1, g2)
    elif strategy == "uniform":
        return uniform_crossover(g1, g2)
    elif strategy == "semantic":
        if llm is None:
            return CrossoverResult(
                success=False, error="LLM required for semantic crossover"
            )
        return await semantic_crossover(g1, g2, prompt, llm)
    else:
        return CrossoverResult(
            success=False, error=f"Unknown crossover strategy: {strategy}"
        )
