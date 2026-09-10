"""Core dataclasses for evolutionary code synthesis.

Defines the fundamental types used throughout the evolution system:
- Genome: A complete code solution with lineage and fitness
- Gene: A reusable code pattern/fragment
- Population: Collection of genomes with statistics
- FitnessResult: Multi-objective fitness scores
- CrossoverResult: Result of breeding two genomes
- MutationOp: Description of a mutation operation
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class MutationType(Enum):
    """Types of mutations that can be applied to genomes."""

    ALGORITHM = auto()  # Try different algorithm
    OPTIMIZE = auto()  # Optimize for performance
    SIMPLIFY = auto()  # Reduce complexity
    GENERALIZE = auto()  # Make more generic
    RANDOM = auto()  # LLM-guided random change


class GeneCategory(Enum):
    """Categories of reusable code patterns."""

    LOOP = auto()
    RECURSION = auto()
    DATASTRUCTURE = auto()
    ALGORITHM = auto()
    UTILITY = auto()
    ERROR_HANDLING = auto()
    OPTIMIZATION = auto()


@dataclass
class Gene:
    """A reusable code pattern/fragment.

    Represents an extractable pattern from code that could be
    transplanted to other genomes.

    Attributes:
        id: Unique identifier for this gene
        pattern: The code pattern as a string
        category: Type of pattern (loop, recursion, etc.)
        origin: Genome ID this gene was extracted from
        ast_node_type: The AST node type (e.g., 'FunctionDef', 'For')
    """

    id: str
    pattern: str
    category: GeneCategory
    origin: str
    ast_node_type: str = ""


@dataclass
class FitnessResult:
    """Multi-objective fitness evaluation result.

    Stores fitness scores across multiple dimensions for
    Pareto-optimal selection.

    Attributes:
        tests_passed: Fraction of tests passing (0.0 to 1.0)
        performance: Performance score (higher is better, normalized)
        quality: Code quality score (complexity, style, etc.)
        novelty: Uniqueness compared to population
        raw_scores: Additional domain-specific scores
    """

    tests_passed: float = 0.0
    performance: float = 0.0
    quality: float = 0.0
    novelty: float = 0.0
    raw_scores: dict[str, float] = field(default_factory=dict)

    @property
    def composite(self) -> float:
        """Simple weighted composite score."""
        return (
            self.tests_passed * 0.4
            + self.performance * 0.2
            + self.quality * 0.2
            + self.novelty * 0.2
        )

    def dominates(self, other: "FitnessResult") -> bool:
        """Check if this result Pareto-dominates another.

        A result dominates if it's at least as good in all objectives
        and strictly better in at least one.
        """
        dominated_in_any = False
        better_in_any = False

        for attr in ["tests_passed", "performance", "quality", "novelty"]:
            self_val = getattr(self, attr)
            other_val = getattr(other, attr)
            if self_val < other_val:
                dominated_in_any = True
            if self_val > other_val:
                better_in_any = True

        return better_in_any and not dominated_in_any


@dataclass
class Genome:
    """Represents a code solution in the evolutionary system.

    A genome contains the actual code, its AST hash for quick
    comparison, lineage information, and fitness scores.

    Attributes:
        id: Unique identifier for this genome
        code: The actual code as a string
        ast_hash: Hash of the AST for structural comparison
        lineage: List of ancestor genome IDs
        fitness_scores: Multi-objective fitness evaluation
        generation: Which generation this genome was created in
        parent_ids: Immediate parent genome IDs (1 for mutation, 2 for crossover)
        mutation_history: List of mutations applied to create this genome
        metadata: Additional metadata (creation time, etc.)
    """

    id: str
    code: str
    ast_hash: str = ""
    lineage: list[str] = field(default_factory=list)
    fitness_scores: FitnessResult | None = None
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_evaluated(self) -> bool:
        """Check if this genome has been evaluated."""
        return self.fitness_scores is not None

    @property
    def fitness(self) -> float:
        """Get composite fitness score, or 0 if not evaluated."""
        if self.fitness_scores is None:
            return 0.0
        return self.fitness_scores.composite


@dataclass
class MutationOp:
    """Describes a mutation operation applied to a genome.

    Attributes:
        type: The type of mutation
        description: Human-readable description
        prompt: The prompt used for LLM-guided mutation
        source_genome_id: ID of the genome being mutated
        result_genome_id: ID of the resulting genome (if successful)
        success: Whether the mutation produced valid code
    """

    type: MutationType
    description: str
    prompt: str = ""
    source_genome_id: str = ""
    result_genome_id: str = ""
    success: bool = False


@dataclass
class CrossoverResult:
    """Result of breeding two genomes.

    Attributes:
        offspring: The resulting genome(s)
        crossover_point: Description of where crossover occurred
        parent1_contribution: Fraction of code from parent 1
        parent2_contribution: Fraction of code from parent 2
        success: Whether crossover produced valid code
        error: Error message if crossover failed
    """

    offspring: list[Genome] = field(default_factory=list)
    crossover_point: str = ""
    parent1_contribution: float = 0.5
    parent2_contribution: float = 0.5
    success: bool = False
    error: str = ""


@dataclass
class PopulationStats:
    """Statistics about a population.

    Attributes:
        size: Number of genomes in population
        generation: Current generation number
        best_fitness: Highest fitness in population
        avg_fitness: Average fitness
        diversity: Diversity metric (0 to 1)
        convergence: Convergence metric (how similar genomes are)
    """

    size: int = 0
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity: float = 0.0
    convergence: float = 0.0


@dataclass
class Population:
    """Collection of genomes with population-level statistics.

    Attributes:
        genomes: List of genomes in the population
        generation: Current generation number
        archive: Pareto-optimal solutions found so far
        gene_pool: Extracted genes available for transfer
        stats_history: Statistics from each generation
    """

    genomes: list[Genome] = field(default_factory=list)
    generation: int = 0
    archive: list[Genome] = field(default_factory=list)
    gene_pool: list[Gene] = field(default_factory=list)
    stats_history: list[PopulationStats] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Current population size."""
        return len(self.genomes)

    @property
    def best_genome(self) -> Genome | None:
        """Get the genome with highest fitness."""
        if not self.genomes:
            return None
        evaluated = [g for g in self.genomes if g.is_evaluated]
        if not evaluated:
            return self.genomes[0]
        return max(evaluated, key=lambda g: g.fitness)

    def get_stats(self) -> PopulationStats:
        """Calculate current population statistics."""
        if not self.genomes:
            return PopulationStats()

        evaluated = [g for g in self.genomes if g.is_evaluated]
        fitnesses = [g.fitness for g in evaluated] if evaluated else [0.0]

        return PopulationStats(
            size=len(self.genomes),
            generation=self.generation,
            best_fitness=max(fitnesses),
            avg_fitness=sum(fitnesses) / len(fitnesses),
            diversity=self._calculate_diversity(),
            convergence=self._calculate_convergence(),
        )

    def _calculate_diversity(self) -> float:
        """Calculate population diversity based on AST hashes."""
        if len(self.genomes) <= 1:
            return 1.0
        unique_hashes = len(set(g.ast_hash for g in self.genomes if g.ast_hash))
        return unique_hashes / len(self.genomes)

    def _calculate_convergence(self) -> float:
        """Calculate how converged the population is."""
        # High convergence = low diversity
        return 1.0 - self._calculate_diversity()

    def add(self, genome: Genome) -> None:
        """Add a genome to the population."""
        self.genomes.append(genome)

    def remove(self, genome_id: str) -> bool:
        """Remove a genome by ID. Returns True if found and removed."""
        for i, g in enumerate(self.genomes):
            if g.id == genome_id:
                del self.genomes[i]
                return True
        return False

    def get_by_id(self, genome_id: str) -> Genome | None:
        """Get a genome by ID."""
        for g in self.genomes:
            if g.id == genome_id:
                return g
        return None


@dataclass
class EvolutionConfig:
    """Configuration for the evolution harness.

    Attributes:
        population_size: Target population size
        max_generations: Maximum generations to run
        mutation_rate: Probability of mutation (0 to 1)
        crossover_rate: Probability of crossover (0 to 1)
        elitism_count: Number of top genomes to preserve
        tournament_size: Size of tournament selection
        fitness_weights: Weights for multi-objective fitness
        convergence_threshold: Stop if fitness plateau detected
        diversity_threshold: Minimum diversity to maintain
    """

    population_size: int = 20
    max_generations: int = 50
    mutation_rate: float = 0.8
    crossover_rate: float = 0.6
    elitism_count: int = 2
    tournament_size: int = 3
    fitness_weights: dict[str, float] = field(
        default_factory=lambda: {
            "tests_passed": 0.4,
            "performance": 0.2,
            "quality": 0.2,
            "novelty": 0.2,
        }
    )
    convergence_threshold: float = 0.01
    diversity_threshold: float = 0.2
