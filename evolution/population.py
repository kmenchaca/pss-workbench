"""Population management for evolutionary code synthesis.

Provides population-level operations:
- Selection strategies (tournament, elitism)
- Diversity maintenance
- Generational vs steady-state modes
- Archive management for Pareto-optimal solutions
"""

import random
from typing import Any, Callable

from .genome import extract_genes, genome_distance, genome_hash
from .types import (
    EvolutionConfig,
    FitnessResult,
    Gene,
    Genome,
    Population,
    PopulationStats,
)


def tournament_selection(
    pop: Population, k: int, tournament_size: int = 3
) -> list[Genome]:
    """Select k genomes using tournament selection.

    Repeatedly runs tournaments where random contestants compete,
    with the fittest winning.

    Args:
        pop: Population to select from
        k: Number of genomes to select
        tournament_size: Number of contestants per tournament

    Returns:
        List of selected genomes
    """
    if not pop.genomes:
        return []

    selected: list[Genome] = []
    actual_tournament_size = min(tournament_size, len(pop.genomes))

    for _ in range(k):
        # Select random contestants
        contestants = random.sample(pop.genomes, actual_tournament_size)

        # Winner is the fittest
        winner = max(contestants, key=lambda g: g.fitness)
        selected.append(winner)

    return selected


def elitism(pop: Population, n: int) -> list[Genome]:
    """Preserve top n genomes.

    Selects the n fittest genomes to carry forward unchanged
    to the next generation.

    Args:
        pop: Population to select from
        n: Number of elite genomes to preserve

    Returns:
        List of elite genomes
    """
    if not pop.genomes:
        return []

    # Sort by fitness (descending)
    sorted_genomes = sorted(pop.genomes, key=lambda g: g.fitness, reverse=True)

    return sorted_genomes[:n]


def diversity_maintenance(pop: Population, min_diversity: float = 0.2) -> Population:
    """Prevent monoculture by maintaining minimum diversity.

    If population diversity falls below threshold, removes
    similar genomes to make room for diversity.

    Args:
        pop: Population to maintain
        min_diversity: Minimum diversity threshold (0-1)

    Returns:
        Modified population with diversity maintained
    """
    if len(pop.genomes) < 2:
        return pop

    current_diversity = pop._calculate_diversity()

    if current_diversity >= min_diversity:
        return pop  # Diversity is acceptable

    # Find clusters of similar genomes
    clusters = _cluster_genomes(pop.genomes)

    # Remove excess from large clusters
    for cluster in clusters:
        if len(cluster) > 1:
            # Keep only the fittest from each cluster
            cluster.sort(key=lambda g: g.fitness, reverse=True)
            keep = cluster[0]
            for genome in cluster[1:]:
                if pop._calculate_diversity() < min_diversity:
                    pop.remove(genome.id)
                else:
                    break

    return pop


def _cluster_genomes(genomes: list[Genome], threshold: float = 0.2) -> list[list[Genome]]:
    """Cluster genomes by genetic similarity.

    Args:
        genomes: Genomes to cluster
        threshold: Distance threshold for clustering

    Returns:
        List of clusters (each cluster is a list of genomes)
    """
    if not genomes:
        return []

    clusters: list[list[Genome]] = []
    assigned: set[str] = set()

    for genome in genomes:
        if genome.id in assigned:
            continue

        # Start new cluster
        cluster = [genome]
        assigned.add(genome.id)

        # Find similar genomes
        for other in genomes:
            if other.id in assigned:
                continue

            distance = genome_distance(genome, other)
            if distance < threshold:
                cluster.append(other)
                assigned.add(other.id)

        clusters.append(cluster)

    return clusters


def roulette_selection(pop: Population, k: int) -> list[Genome]:
    """Select k genomes using fitness-proportionate selection.

    Probability of selection is proportional to fitness.

    Args:
        pop: Population to select from
        k: Number of genomes to select

    Returns:
        List of selected genomes
    """
    if not pop.genomes:
        return []

    # Calculate selection probabilities
    fitnesses = [max(g.fitness, 0.001) for g in pop.genomes]  # Avoid zero
    total_fitness = sum(fitnesses)
    probs = [f / total_fitness for f in fitnesses]

    # Select k genomes
    selected = random.choices(pop.genomes, weights=probs, k=k)
    return list(selected)


def rank_selection(pop: Population, k: int) -> list[Genome]:
    """Select k genomes using rank-based selection.

    Selection probability based on rank rather than absolute fitness,
    reducing selection pressure.

    Args:
        pop: Population to select from
        k: Number of genomes to select

    Returns:
        List of selected genomes
    """
    if not pop.genomes:
        return []

    # Sort by fitness and assign ranks
    sorted_genomes = sorted(pop.genomes, key=lambda g: g.fitness)
    n = len(sorted_genomes)

    # Rank-based weights (rank 1 = lowest, rank n = highest)
    weights = [i + 1 for i in range(n)]

    selected = random.choices(sorted_genomes, weights=weights, k=k)
    return list(selected)


def truncation_selection(pop: Population, k: int, truncation_pct: float = 0.5) -> list[Genome]:
    """Select k genomes from top truncation_pct of population.

    Only considers the fittest portion of the population.

    Args:
        pop: Population to select from
        k: Number of genomes to select
        truncation_pct: Fraction of population to consider

    Returns:
        List of selected genomes
    """
    if not pop.genomes:
        return []

    # Sort by fitness
    sorted_genomes = sorted(pop.genomes, key=lambda g: g.fitness, reverse=True)

    # Truncate
    cutoff = max(1, int(len(sorted_genomes) * truncation_pct))
    candidates = sorted_genomes[:cutoff]

    # Random selection from candidates
    return random.choices(candidates, k=k)


def generational_replacement(
    pop: Population,
    offspring: list[Genome],
    elitism_count: int = 2,
) -> Population:
    """Replace population with offspring (generational model).

    The entire population is replaced except for elite individuals.

    Args:
        pop: Current population
        offspring: New offspring genomes
        elitism_count: Number of elites to preserve

    Returns:
        New population for next generation
    """
    # Preserve elites
    elites = elitism(pop, elitism_count)

    # Create new population
    new_genomes = elites + offspring

    # Trim to target size if needed
    target_size = len(pop.genomes)
    if len(new_genomes) > target_size:
        # Keep elites, then best offspring
        non_elite = [g for g in new_genomes if g not in elites]
        non_elite.sort(key=lambda g: g.fitness, reverse=True)
        new_genomes = elites + non_elite[: target_size - len(elites)]

    return Population(
        genomes=new_genomes,
        generation=pop.generation + 1,
        archive=pop.archive.copy(),
        gene_pool=pop.gene_pool.copy(),
        stats_history=pop.stats_history.copy(),
    )


def steady_state_replacement(
    pop: Population,
    offspring: list[Genome],
    replacement_strategy: str = "worst",
) -> Population:
    """Replace individuals one at a time (steady-state model).

    Offspring replace existing individuals immediately.

    Args:
        pop: Current population
        offspring: New offspring to insert
        replacement_strategy: How to choose who to replace
            ('worst', 'random', 'tournament')

    Returns:
        Modified population
    """
    new_genomes = pop.genomes.copy()

    for child in offspring:
        if len(new_genomes) < 1:
            new_genomes.append(child)
            continue

        # Choose who to replace
        if replacement_strategy == "worst":
            # Replace worst genome
            worst_idx = min(range(len(new_genomes)), key=lambda i: new_genomes[i].fitness)
            if child.fitness > new_genomes[worst_idx].fitness:
                new_genomes[worst_idx] = child

        elif replacement_strategy == "random":
            # Replace random genome
            idx = random.randint(0, len(new_genomes) - 1)
            new_genomes[idx] = child

        elif replacement_strategy == "tournament":
            # Tournament to find loser
            contestants_idx = random.sample(range(len(new_genomes)), min(3, len(new_genomes)))
            loser_idx = min(contestants_idx, key=lambda i: new_genomes[i].fitness)
            if child.fitness > new_genomes[loser_idx].fitness:
                new_genomes[loser_idx] = child

    return Population(
        genomes=new_genomes,
        generation=pop.generation,  # Same generation in steady-state
        archive=pop.archive.copy(),
        gene_pool=pop.gene_pool.copy(),
        stats_history=pop.stats_history.copy(),
    )


def update_archive(pop: Population) -> None:
    """Update Pareto archive with non-dominated solutions.

    Adds any genome that is not dominated by any archive member.

    Args:
        pop: Population to update archive from
    """
    for genome in pop.genomes:
        if genome.fitness_scores is None:
            continue

        # Check if genome is dominated by any archive member
        is_dominated = False
        dominated_archive = []

        for archived in pop.archive:
            if archived.fitness_scores is None:
                continue

            if archived.fitness_scores.dominates(genome.fitness_scores):
                is_dominated = True
                break

            if genome.fitness_scores.dominates(archived.fitness_scores):
                dominated_archive.append(archived)

        if not is_dominated:
            # Remove dominated archive members
            for dominated in dominated_archive:
                pop.archive.remove(dominated)

            # Add to archive if not already there
            if genome.id not in [a.id for a in pop.archive]:
                pop.archive.append(genome)


def update_gene_pool(pop: Population, max_genes: int = 100) -> None:
    """Extract and store useful genes from population.

    Identifies reusable code patterns from fit genomes and
    adds them to the shared gene pool.

    Args:
        pop: Population to extract genes from
        max_genes: Maximum genes to keep in pool
    """
    # Extract genes from top performers
    top_genomes = elitism(pop, min(5, len(pop.genomes)))

    for genome in top_genomes:
        genes = extract_genes(genome)
        for gene in genes:
            # Avoid duplicates
            if gene.pattern not in [g.pattern for g in pop.gene_pool]:
                pop.gene_pool.append(gene)

    # Trim if too large
    if len(pop.gene_pool) > max_genes:
        # Keep genes from most fit origins
        pop.gene_pool = pop.gene_pool[:max_genes]


def initialize_population(
    initial_genomes: list[Genome],
    target_size: int,
    seed_genome: Genome | None = None,
) -> Population:
    """Create initial population.

    Args:
        initial_genomes: Starting genomes
        target_size: Desired population size
        seed_genome: Optional genome to seed from

    Returns:
        Initialized population
    """
    genomes = initial_genomes.copy()

    if seed_genome and seed_genome not in genomes:
        genomes.insert(0, seed_genome)

    # Ensure unique AST hashes
    for genome in genomes:
        if not genome.ast_hash:
            genome.ast_hash = genome_hash(genome)

    return Population(
        genomes=genomes[:target_size],
        generation=0,
        archive=[],
        gene_pool=[],
        stats_history=[],
    )


def record_generation_stats(pop: Population) -> None:
    """Record statistics for the current generation.

    Adds current stats to history for tracking progress.

    Args:
        pop: Population to record stats from
    """
    stats = pop.get_stats()
    pop.stats_history.append(stats)


def check_convergence(
    pop: Population,
    threshold: float = 0.01,
    window: int = 5,
) -> bool:
    """Check if population has converged.

    Convergence is detected when fitness improvement plateaus.

    Args:
        pop: Population to check
        threshold: Minimum improvement to not be considered converged
        window: Number of generations to consider

    Returns:
        True if population has converged
    """
    if len(pop.stats_history) < window + 1:
        return False

    recent_stats = pop.stats_history[-window:]
    first_fitness = recent_stats[0].best_fitness
    last_fitness = recent_stats[-1].best_fitness

    improvement = abs(last_fitness - first_fitness)
    return improvement < threshold


def get_selection_function(
    strategy: str,
) -> Callable[[Population, int], list[Genome]]:
    """Get selection function by name.

    Args:
        strategy: Selection strategy name

    Returns:
        Selection function
    """
    strategies = {
        "tournament": tournament_selection,
        "roulette": roulette_selection,
        "rank": rank_selection,
        "truncation": truncation_selection,
    }
    return strategies.get(strategy, tournament_selection)
