"""Evolutionary Code Synthesis system.

Treats code generation as genetic programming where:
- Branches are mutations
- Fitness functions determine survival
- Synthesis does crossover breeding

This module provides a complete evolutionary framework for
generating and optimizing code solutions.

Example usage:
    ```python
    from evolution import EvolutionHarness, EvolutionConfig

    # Define tests
    tests = [
        {"input": [1, 2], "expected": 3},
        {"input": [0, 0], "expected": 0},
    ]

    # Configure evolution
    config = EvolutionConfig(
        population_size=20,
        max_generations=50,
        mutation_rate=0.8,
    )

    # Run evolution
    harness = EvolutionHarness(llm=my_llm, config=config, tests=tests)
    result = await harness.run("Write a function that adds two numbers")

    # Get best solution
    print(result.best_genome.code)
    ```
"""

# Types
from .types import (
    CrossoverResult,
    EvolutionConfig,
    FitnessResult,
    Gene,
    GeneCategory,
    Genome,
    MutationOp,
    MutationType,
    Population,
    PopulationStats,
)

# Genome operations
from .genome import (
    ast_to_code,
    create_genome,
    extract_genes,
    genome_distance,
    genome_hash,
    get_classes,
    get_functions,
    is_valid_code,
    merge_functions,
    parse_to_ast,
)

# Mutation operators
from .mutation import (
    apply_mutation,
    deletion_mutation,
    insertion_mutation,
    mutate_algorithm,
    mutate_generalize,
    mutate_optimize,
    mutate_simplify,
    point_mutation,
    random_mutation,
    select_mutation_type,
)

# Crossover operations
from .crossover import (
    apply_crossover,
    function_swap,
    gene_transfer,
    select_crossover_parents,
    semantic_crossover,
    single_point_crossover,
    uniform_crossover,
)

# Fitness functions
from .fitness import (
    CompositeFitness,
    FitnessFunction,
    NoveltyFitness,
    PerformanceFitness,
    QualityFitness,
    TestCaseFitness,
    TestCaseFitness as TestFitness,  # Alias for backward compatibility
    create_default_fitness_suite,
    evaluate_genome,
    pareto_rank,
)

# Population management
from .population import (
    check_convergence,
    diversity_maintenance,
    elitism,
    generational_replacement,
    get_selection_function,
    initialize_population,
    rank_selection,
    record_generation_stats,
    roulette_selection,
    steady_state_replacement,
    tournament_selection,
    truncation_selection,
    update_archive,
    update_gene_pool,
)

# Main harness
from .harness import (
    EvolutionHarness,
    EvolutionResult,
    GenerationReport,
    evolve,
)

# Lethal gene tracking
from .lethal import (
    FailureRecord,
    LethalGeneDB,
    LethalPattern,
    extract_error_patterns,
    identify_failure_cause,
)

__all__ = [
    # Types
    "CrossoverResult",
    "EvolutionConfig",
    "FitnessResult",
    "Gene",
    "GeneCategory",
    "Genome",
    "MutationOp",
    "MutationType",
    "Population",
    "PopulationStats",
    # Genome
    "ast_to_code",
    "create_genome",
    "extract_genes",
    "genome_distance",
    "genome_hash",
    "get_classes",
    "get_functions",
    "is_valid_code",
    "merge_functions",
    "parse_to_ast",
    # Mutation
    "apply_mutation",
    "deletion_mutation",
    "insertion_mutation",
    "mutate_algorithm",
    "mutate_generalize",
    "mutate_optimize",
    "mutate_simplify",
    "point_mutation",
    "random_mutation",
    "select_mutation_type",
    # Crossover
    "apply_crossover",
    "function_swap",
    "gene_transfer",
    "select_crossover_parents",
    "semantic_crossover",
    "single_point_crossover",
    "uniform_crossover",
    # Fitness
    "CompositeFitness",
    "FitnessFunction",
    "NoveltyFitness",
    "PerformanceFitness",
    "QualityFitness",
    "TestCaseFitness",
    "TestFitness",  # Alias for TestCaseFitness
    "create_default_fitness_suite",
    "evaluate_genome",
    "pareto_rank",
    # Population
    "check_convergence",
    "diversity_maintenance",
    "elitism",
    "generational_replacement",
    "get_selection_function",
    "initialize_population",
    "rank_selection",
    "record_generation_stats",
    "roulette_selection",
    "steady_state_replacement",
    "tournament_selection",
    "truncation_selection",
    "update_archive",
    "update_gene_pool",
    # Harness
    "EvolutionHarness",
    "EvolutionResult",
    "GenerationReport",
    "evolve",
    # Lethal
    "FailureRecord",
    "LethalGeneDB",
    "LethalPattern",
    "extract_error_patterns",
    "identify_failure_cause",
]

__version__ = "0.1.0"
