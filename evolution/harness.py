"""Main evolution harness for evolutionary code synthesis.

Orchestrates the complete evolutionary process:
- Initialize population from spec/tests
- Run generations: select, crossover, mutate, evaluate
- Track lineage through genealogy
- Detect convergence
- Return best genome(s)
"""

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from .crossover import apply_crossover, select_crossover_parents
from .fitness import (
    FitnessFunction,
    NoveltyFitness,
    PerformanceFitness,
    QualityFitness,
    TestCaseFitness,
    evaluate_genome,
)
from .genome import create_genome, genome_hash
from .lethal import LethalGeneDB
from .mutation import apply_mutation, select_mutation_type
from .population import (
    check_convergence,
    diversity_maintenance,
    elitism,
    generational_replacement,
    get_selection_function,
    initialize_population,
    record_generation_stats,
    steady_state_replacement,
    tournament_selection,
    update_archive,
    update_gene_pool,
)
from .types import (
    EvolutionConfig,
    FitnessResult,
    Genome,
    MutationType,
    Population,
    PopulationStats,
)


class LLMProvider(Protocol):
    """Protocol for LLM providers used in evolution."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt."""
        ...


@dataclass
class EvolutionResult:
    """Result of an evolution run.

    Attributes:
        best_genome: The highest fitness genome
        pareto_front: Pareto-optimal solutions
        final_population: Final population state
        generations_run: Number of generations completed
        converged: Whether evolution converged
        history: Statistics from each generation
    """

    best_genome: Genome | None
    pareto_front: list[Genome]
    final_population: Population
    generations_run: int
    converged: bool
    history: list[PopulationStats]


@dataclass
class GenerationReport:
    """Report for a single generation.

    Attributes:
        generation: Generation number
        population_size: Size of population
        best_fitness: Best fitness in generation
        avg_fitness: Average fitness
        diversity: Population diversity
        mutations_applied: Number of mutations
        crossovers_applied: Number of crossovers
        lethal_filtered: Mutations filtered by lethal genes
    """

    generation: int
    population_size: int
    best_fitness: float
    avg_fitness: float
    diversity: float
    mutations_applied: int
    crossovers_applied: int
    lethal_filtered: int


class EvolutionHarness:
    """Main harness for running evolutionary code synthesis.

    Manages the complete evolutionary cycle from initial population
    through selection, variation, and evaluation.
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: EvolutionConfig | None = None,
        fitness_functions: list[FitnessFunction] | None = None,
        tests: list[dict[str, Any]] | None = None,
        benchmark_inputs: list[Any] | None = None,
    ):
        """Initialize the evolution harness.

        Args:
            llm: LLM provider for mutations and crossover
            config: Evolution configuration
            fitness_functions: Custom fitness functions
            tests: Test cases for TestCaseFitness
            benchmark_inputs: Inputs for PerformanceFitness
        """
        self.llm = llm
        self.config = config or EvolutionConfig()
        self.lethal_db = LethalGeneDB()
        self.tests = tests or []
        self.benchmark_inputs = benchmark_inputs or []

        # Set up fitness functions
        if fitness_functions:
            self.fitness_functions = fitness_functions
        else:
            self.fitness_functions = [
                TestCaseFitness(self.tests),
                PerformanceFitness(self.benchmark_inputs),
                QualityFitness(),
                NoveltyFitness(),
            ]

        self.population: Population | None = None
        self.task_prompt: str = ""
        self.generation_reports: list[GenerationReport] = []

    async def initialize_from_spec(
        self,
        task_prompt: str,
        initial_code: str | None = None,
        num_initial: int = 5,
    ) -> Population:
        """Initialize population from task specification.

        Uses LLM to generate initial diverse solutions.

        Args:
            task_prompt: Description of the coding task
            initial_code: Optional seed code
            num_initial: Number of initial genomes to generate

        Returns:
            Initialized population
        """
        self.task_prompt = task_prompt
        initial_genomes: list[Genome] = []

        # Add seed genome if provided
        if initial_code:
            seed = create_genome(initial_code, generation=0)
            initial_genomes.append(seed)

        # Generate diverse initial solutions
        generation_prompt = f"""Generate a Python solution for the following task:

{task_prompt}

Requirements:
- Write clean, functional Python code
- Include all necessary functions
- Handle edge cases appropriately

Return ONLY Python code, no explanations.
```python
"""

        # Generate remaining initial genomes
        tasks = []
        for i in range(num_initial - len(initial_genomes)):
            # Vary the prompt slightly for diversity
            varied_prompt = generation_prompt
            if i > 0:
                approaches = [
                    "Use a different algorithm than obvious approaches.",
                    "Prioritize code simplicity and readability.",
                    "Optimize for performance.",
                    "Use a recursive approach if applicable.",
                    "Use an iterative approach.",
                ]
                varied_prompt += f"\n\nHint: {random.choice(approaches)}"

            tasks.append(self._generate_initial_genome(varied_prompt, i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Genome):
                initial_genomes.append(result)

        # Initialize population
        self.population = initialize_population(
            initial_genomes,
            self.config.population_size,
        )

        # Evaluate initial population
        await self._evaluate_population()

        return self.population

    async def _generate_initial_genome(
        self, prompt: str, index: int
    ) -> Genome | Exception:
        """Generate a single initial genome."""
        try:
            response = await self.llm.generate(prompt)
            code = self._extract_code(response)

            if not code:
                return Exception("No code generated")

            genome = create_genome(code, generation=0)
            return genome

        except Exception as e:
            return e

    def _extract_code(self, response: str) -> str:
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

    async def run(
        self,
        task_prompt: str | None = None,
        initial_code: str | None = None,
        max_generations: int | None = None,
    ) -> EvolutionResult:
        """Run the complete evolution process.

        Args:
            task_prompt: Task description (if not already set)
            initial_code: Optional seed code
            max_generations: Override max generations from config

        Returns:
            EvolutionResult with best solutions
        """
        if task_prompt:
            self.task_prompt = task_prompt

        if self.population is None:
            await self.initialize_from_spec(
                self.task_prompt,
                initial_code,
                self.config.population_size,
            )

        max_gen = max_generations or self.config.max_generations
        converged = False

        for gen in range(max_gen):
            # Run one generation
            report = await self._run_generation()
            self.generation_reports.append(report)

            # Check for convergence
            if check_convergence(
                self.population,  # type: ignore
                self.config.convergence_threshold,
            ):
                converged = True
                break

            # Maintain diversity
            if self.population:
                diversity_maintenance(
                    self.population,
                    self.config.diversity_threshold,
                )

        # Collect results
        if self.population:
            update_archive(self.population)

        return EvolutionResult(
            best_genome=self.population.best_genome if self.population else None,
            pareto_front=self.population.archive if self.population else [],
            final_population=self.population or Population(),
            generations_run=len(self.generation_reports),
            converged=converged,
            history=self.population.stats_history if self.population else [],
        )

    async def _run_generation(self) -> GenerationReport:
        """Run a single generation of evolution.

        Returns:
            Report for this generation
        """
        if not self.population:
            raise ValueError("Population not initialized")

        mutations_applied = 0
        crossovers_applied = 0
        lethal_filtered = 0

        # Selection
        selection_func = get_selection_function("tournament")
        num_parents = self.config.population_size - self.config.elitism_count
        parents = selection_func(self.population, num_parents)

        offspring: list[Genome] = []

        # Crossover
        for i in range(0, len(parents) - 1, 2):
            if random.random() < self.config.crossover_rate:
                result = await apply_crossover(
                    parents[i],
                    parents[i + 1],
                    strategy="uniform",
                    prompt=self.task_prompt,
                    llm=self.llm,
                )
                if result.success:
                    for child in result.offspring:
                        # Check lethal genes
                        if not self.lethal_db.contains_lethal(child.code):
                            offspring.append(child)
                            crossovers_applied += 1
                        else:
                            lethal_filtered += 1

        # Mutation
        mutation_tasks = []
        for parent in parents:
            if random.random() < self.config.mutation_rate:
                mutation_tasks.append(
                    self._apply_mutation_safe(parent)
                )

        mutation_results = await asyncio.gather(*mutation_tasks, return_exceptions=True)

        for result in mutation_results:
            if isinstance(result, Genome):
                if not self.lethal_db.contains_lethal(result.code):
                    offspring.append(result)
                    mutations_applied += 1
                else:
                    lethal_filtered += 1

        # Evaluate offspring
        for genome in offspring:
            genome.fitness_scores = evaluate_genome(
                genome,
                self.fitness_functions,
                tests=self.tests,
                population=self.population,
            )

        # Track failures for lethal gene learning
        for genome in offspring:
            if genome.fitness_scores and genome.fitness_scores.tests_passed < 0.5:
                self.lethal_db.record_failure(genome)

        # Replacement
        self.population = generational_replacement(
            self.population,
            offspring,
            self.config.elitism_count,
        )

        # Update archives and stats
        update_archive(self.population)
        update_gene_pool(self.population)
        record_generation_stats(self.population)

        stats = self.population.get_stats()

        return GenerationReport(
            generation=self.population.generation,
            population_size=len(self.population.genomes),
            best_fitness=stats.best_fitness,
            avg_fitness=stats.avg_fitness,
            diversity=stats.diversity,
            mutations_applied=mutations_applied,
            crossovers_applied=crossovers_applied,
            lethal_filtered=lethal_filtered,
        )

    async def _apply_mutation_safe(self, genome: Genome) -> Genome | Exception:
        """Apply mutation with error handling."""
        try:
            result, _ = await apply_mutation(
                genome,
                self.task_prompt,
                self.llm,
            )
            return result if result else Exception("Mutation failed")
        except Exception as e:
            return e

    async def _evaluate_population(self) -> None:
        """Evaluate all genomes in the population."""
        if not self.population:
            return

        for genome in self.population.genomes:
            genome.fitness_scores = evaluate_genome(
                genome,
                self.fitness_functions,
                tests=self.tests,
                population=self.population,
            )

    def get_best_genomes(self, n: int = 5) -> list[Genome]:
        """Get the n best genomes by fitness.

        Args:
            n: Number of genomes to return

        Returns:
            List of best genomes
        """
        if not self.population:
            return []

        return elitism(self.population, n)

    def get_pareto_front(self) -> list[Genome]:
        """Get Pareto-optimal genomes.

        Returns:
            List of non-dominated genomes
        """
        if not self.population:
            return []

        return self.population.archive

    def add_test_case(self, test: dict[str, Any]) -> None:
        """Add a test case dynamically.

        Args:
            test: Test case specification
        """
        self.tests.append(test)

        # Update test fitness function
        for func in self.fitness_functions:
            if isinstance(func, TestCaseFitness):
                func.tests = self.tests
                break


async def evolve(
    task_prompt: str,
    llm: LLMProvider,
    tests: list[dict[str, Any]] | None = None,
    config: EvolutionConfig | None = None,
    initial_code: str | None = None,
) -> EvolutionResult:
    """Convenience function to run evolution.

    Args:
        task_prompt: Description of the coding task
        llm: LLM provider for generation
        tests: Test cases for fitness evaluation
        config: Evolution configuration
        initial_code: Optional seed code

    Returns:
        EvolutionResult with best solutions
    """
    harness = EvolutionHarness(
        llm=llm,
        config=config,
        tests=tests,
    )

    return await harness.run(task_prompt, initial_code)
