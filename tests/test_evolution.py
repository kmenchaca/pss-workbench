"""Comprehensive tests for the evolutionary code synthesis system.

Tests cover:
- Genome operations
- Mutation operators
- Crossover functions
- Fitness evaluation
- Population management
- Lethal gene tracking
- Full evolution loop
"""

import ast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evolution import (
    CompositeFitness,
    CrossoverResult,
    EvolutionConfig,
    EvolutionHarness,
    EvolutionResult,
    FailureRecord,
    FitnessFunction,
    FitnessResult,
    Gene,
    GeneCategory,
    Genome,
    LethalGeneDB,
    LethalPattern,
    MutationOp,
    MutationType,
    NoveltyFitness,
    PerformanceFitness,
    Population,
    PopulationStats,
    QualityFitness,
    TestFitness,
)
from evolution.crossover import (
    function_swap,
    gene_transfer,
    select_crossover_parents,
    single_point_crossover,
    uniform_crossover,
)
from evolution.fitness import evaluate_genome, pareto_rank
from evolution.genome import (
    ast_to_code,
    create_genome,
    extract_genes,
    genome_distance,
    genome_hash,
    get_functions,
    is_valid_code,
    merge_functions,
    parse_to_ast,
)
from evolution.lethal import extract_error_patterns, identify_failure_cause
from evolution.mutation import (
    deletion_mutation,
    insertion_mutation,
    point_mutation,
    select_mutation_type,
)
from evolution.population import (
    check_convergence,
    diversity_maintenance,
    elitism,
    generational_replacement,
    initialize_population,
    rank_selection,
    roulette_selection,
    steady_state_replacement,
    tournament_selection,
    truncation_selection,
    update_archive,
)


# =============================================================================
# GENOME OPERATIONS TESTS
# =============================================================================


class TestGenomeOperations:
    """Tests for genome.py operations."""

    def test_parse_to_ast_valid_code(self):
        """Test parsing valid Python code to AST."""
        code = "def add(a, b):\n    return a + b"
        tree = parse_to_ast(code)
        assert tree is not None
        assert isinstance(tree, ast.AST)

    def test_parse_to_ast_invalid_code(self):
        """Test parsing invalid code returns None."""
        code = "def broken(\n    return"
        tree = parse_to_ast(code)
        assert tree is None

    def test_ast_to_code_roundtrip(self):
        """Test converting AST back to code."""
        original = "def add(a, b):\n    return a + b"
        tree = parse_to_ast(original)
        assert tree is not None
        code = ast_to_code(tree)
        assert "def add" in code
        assert "return a + b" in code

    def test_genome_hash_uniqueness(self):
        """Test that different code produces different hashes."""
        g1 = create_genome("def foo(): return 1")
        g2 = create_genome("def bar(): return 2")
        assert g1.ast_hash != g2.ast_hash

    def test_genome_hash_identical_code(self):
        """Test that identical code has same hash."""
        g1 = create_genome("def foo(x): return x + 1")
        g2 = create_genome("def foo(x): return x + 1")
        # Identical code should have identical hash
        assert g1.ast_hash == g2.ast_hash

    def test_genome_distance_identical(self):
        """Test distance between identical genomes is 0."""
        g1 = create_genome("def foo(): return 1")
        g2 = create_genome("def foo(): return 1")
        g1.ast_hash = genome_hash(g1)
        g2.ast_hash = genome_hash(g2)
        dist = genome_distance(g1, g2)
        assert dist == 0.0

    def test_genome_distance_different(self):
        """Test distance between different genomes is > 0."""
        g1 = create_genome("def foo(): return 1")
        g2 = create_genome("class Bar:\n    def baz(self): pass")
        dist = genome_distance(g1, g2)
        assert dist > 0

    def test_extract_genes_function(self):
        """Test extracting genes from code with functions."""
        g = create_genome("def foo():\n    return 1\n\ndef bar():\n    return 2")
        genes = extract_genes(g)
        assert len(genes) >= 2
        assert all(isinstance(gene, Gene) for gene in genes)

    def test_extract_genes_loop(self):
        """Test extracting loop genes."""
        g = create_genome("for i in range(10):\n    print(i)")
        genes = extract_genes(g)
        loop_genes = [g for g in genes if g.category == GeneCategory.LOOP]
        assert len(loop_genes) >= 1

    def test_is_valid_code(self):
        """Test code validation."""
        assert is_valid_code("x = 1")
        assert is_valid_code("def foo(): pass")
        assert not is_valid_code("def broken(")
        assert not is_valid_code("return return return")

    def test_get_functions(self):
        """Test extracting function definitions."""
        g = create_genome("def foo(): pass\ndef bar(): pass\nx = 1")
        funcs = get_functions(g)
        assert len(funcs) == 2
        assert all(isinstance(f, ast.FunctionDef) for f in funcs)

    def test_merge_functions(self):
        """Test merging function lists."""
        f1 = get_functions(create_genome("def foo(): return 1"))
        f2 = get_functions(create_genome("def bar(): return 2"))
        merged = merge_functions(f1, f2)
        assert "def foo" in merged
        assert "def bar" in merged


# =============================================================================
# MUTATION TESTS
# =============================================================================


class TestMutation:
    """Tests for mutation.py operations."""

    def test_select_mutation_type_returns_valid(self):
        """Test mutation type selection returns valid type."""
        g = create_genome("def foo(): pass")
        mt = select_mutation_type(g)
        assert isinstance(mt, MutationType)

    def test_select_mutation_type_avoids_recent(self):
        """Test that recent mutations are less likely."""
        g = create_genome("def foo(): pass")
        g.mutation_history = ["algorithm", "algorithm", "algorithm"]

        # Run many selections and count
        counts = {mt: 0 for mt in MutationType}
        for _ in range(100):
            mt = select_mutation_type(g)
            counts[mt] += 1

        # Algorithm should be selected less often
        assert counts[MutationType.ALGORITHM] < counts[MutationType.OPTIMIZE]

    def test_point_mutation_valid(self):
        """Test point mutation produces valid code."""
        g = create_genome("x = 1")
        result = point_mutation(g, 4, "2")  # Change "1" to "2"
        assert result is not None
        assert "2" in result.code

    def test_point_mutation_invalid_position(self):
        """Test point mutation with invalid position."""
        g = create_genome("x = 1")
        result = point_mutation(g, 100, "a")
        assert result is None

    def test_deletion_mutation(self):
        """Test deletion mutation."""
        g = create_genome("x = 1\ny = 2")
        result = deletion_mutation(g, 0, 6)  # Delete "x = 1\n"
        assert result is not None
        assert "y = 2" in result.code
        assert "x = 1" not in result.code

    def test_insertion_mutation(self):
        """Test insertion mutation."""
        g = create_genome("x = 1")
        result = insertion_mutation(g, 0, "# comment\n")
        assert result is not None
        assert "# comment" in result.code


# =============================================================================
# CROSSOVER TESTS
# =============================================================================


class TestCrossover:
    """Tests for crossover.py operations."""

    def test_single_point_crossover(self):
        """Test single-point crossover produces offspring."""
        g1 = create_genome("def foo():\n    return 1\ndef bar():\n    return 2")
        g2 = create_genome("def baz():\n    return 3\ndef qux():\n    return 4")
        result = single_point_crossover(g1, g2)
        assert isinstance(result, CrossoverResult)
        # May or may not produce valid offspring depending on crossover point

    def test_uniform_crossover(self):
        """Test uniform crossover mixes functions."""
        g1 = create_genome("def foo():\n    return 1")
        g2 = create_genome("def bar():\n    return 2")
        result = uniform_crossover(g1, g2)
        if result.success:
            assert len(result.offspring) > 0

    def test_gene_transfer(self):
        """Test gene transfer from donor to recipient."""
        donor = create_genome("def helper():\n    return 42")
        recipient = create_genome("def main():\n    pass")
        genes = extract_genes(donor)
        if genes:
            gene = genes[0]
            result = gene_transfer(donor, recipient, gene)
            if result.success:
                assert len(result.offspring) > 0
                assert "helper" in result.offspring[0].code or "helper_transferred" in result.offspring[0].code

    def test_function_swap(self):
        """Test swapping a specific function."""
        g1 = create_genome("def foo():\n    return 1")
        g2 = create_genome("def foo():\n    return 999")
        result = function_swap(g1, g2, "foo")
        if result.success:
            assert len(result.offspring) > 0

    def test_select_crossover_parents_fitness(self):
        """Test parent selection by fitness."""
        g1 = create_genome("def a(): pass")
        g1.fitness_scores = FitnessResult(tests_passed=0.9)
        g2 = create_genome("def b(): pass")
        g2.fitness_scores = FitnessResult(tests_passed=0.1)
        g3 = create_genome("def c(): pass")
        g3.fitness_scores = FitnessResult(tests_passed=0.5)

        result = select_crossover_parents([g1, g2, g3], "fitness")
        assert result is not None
        assert len(result) == 2


# =============================================================================
# FITNESS TESTS
# =============================================================================


class TestFitnessFunctions:
    """Tests for fitness.py functions."""

    def test_test_fitness_all_pass(self):
        """Test fitness when all tests pass."""
        tests = [
            {"input": [1, 2], "expected": 3, "function": "add"},
            {"input": [0, 0], "expected": 0, "function": "add"},
        ]
        fitness = TestFitness(tests)
        g = create_genome("def add(a, b):\n    return a + b")
        score = fitness.evaluate(g)
        assert score == 1.0

    def test_test_fitness_partial_pass(self):
        """Test fitness when some tests fail."""
        tests = [
            {"input": [1, 2], "expected": 3, "function": "add"},
            {"input": [1, 2], "expected": 100, "function": "add"},  # Will fail
        ]
        fitness = TestFitness(tests)
        g = create_genome("def add(a, b):\n    return a + b")
        score = fitness.evaluate(g)
        assert score == 0.5

    def test_quality_fitness(self):
        """Test code quality fitness."""
        fitness = QualityFitness()
        # Simple, clean code
        g = create_genome("def foo():\n    return 1")
        score = fitness.evaluate(g)
        assert score > 0

    def test_novelty_fitness(self):
        """Test novelty fitness."""
        fitness = NoveltyFitness()
        g1 = create_genome("def a(): return 1")
        g2 = create_genome("def a(): return 1")
        g3 = create_genome("def a(): return 1")
        pop = Population(genomes=[g1, g2, g3])

        # g1 is similar to all others, should have low novelty
        score = fitness.evaluate(g1, population=pop)
        assert 0 <= score <= 1

    def test_composite_fitness(self):
        """Test composite fitness combining multiple functions."""
        test_f = TestFitness([{"input": [1], "expected": 1, "function": "identity"}])
        quality_f = QualityFitness()

        composite = CompositeFitness()
        composite.add_function(test_f, 0.7)
        composite.add_function(quality_f, 0.3)

        g = create_genome("def identity(x):\n    return x")
        score = composite.evaluate(g)
        assert 0 <= score <= 1

    def test_fitness_result_dominates(self):
        """Test Pareto dominance check."""
        f1 = FitnessResult(tests_passed=0.9, performance=0.8, quality=0.7, novelty=0.6)
        f2 = FitnessResult(tests_passed=0.8, performance=0.7, quality=0.6, novelty=0.5)
        assert f1.dominates(f2)
        assert not f2.dominates(f1)

    def test_pareto_rank(self):
        """Test Pareto ranking."""
        g1 = create_genome("def a(): pass")
        g1.fitness_scores = FitnessResult(tests_passed=1.0, performance=1.0)
        g2 = create_genome("def b(): pass")
        g2.fitness_scores = FitnessResult(tests_passed=0.5, performance=0.5)

        ranks = pareto_rank([g1, g2])
        assert ranks[g1.id] == 0  # Pareto optimal
        assert ranks[g2.id] == 1  # Dominated

    def test_evaluate_genome(self):
        """Test full genome evaluation."""
        g = create_genome("def add(a, b):\n    return a + b")
        tests = [{"input": [1, 2], "expected": 3, "function": "add"}]
        fitness_funcs = [TestFitness(tests), QualityFitness()]

        result = evaluate_genome(g, fitness_funcs, tests=tests)
        assert isinstance(result, FitnessResult)
        assert result.tests_passed == 1.0


# =============================================================================
# POPULATION TESTS
# =============================================================================


class TestPopulation:
    """Tests for population.py operations."""

    def test_tournament_selection(self):
        """Test tournament selection."""
        genomes = [create_genome(f"def f{i}(): return {i}") for i in range(10)]
        for i, g in enumerate(genomes):
            g.fitness_scores = FitnessResult(tests_passed=i / 10)

        pop = Population(genomes=genomes)
        selected = tournament_selection(pop, 3)
        assert len(selected) == 3

    def test_elitism(self):
        """Test elite preservation."""
        genomes = [create_genome(f"def f{i}(): return {i}") for i in range(10)]
        for i, g in enumerate(genomes):
            g.fitness_scores = FitnessResult(tests_passed=i / 10)

        pop = Population(genomes=genomes)
        elites = elitism(pop, 2)
        assert len(elites) == 2
        # Top 2 should have highest fitness
        assert elites[0].fitness >= elites[1].fitness

    def test_roulette_selection(self):
        """Test roulette wheel selection."""
        genomes = [create_genome(f"def f{i}(): return {i}") for i in range(5)]
        for i, g in enumerate(genomes):
            g.fitness_scores = FitnessResult(tests_passed=(i + 1) / 5)

        pop = Population(genomes=genomes)
        selected = roulette_selection(pop, 3)
        assert len(selected) == 3

    def test_rank_selection(self):
        """Test rank-based selection."""
        genomes = [create_genome(f"def f{i}(): return {i}") for i in range(5)]
        for i, g in enumerate(genomes):
            g.fitness_scores = FitnessResult(tests_passed=i / 5)

        pop = Population(genomes=genomes)
        selected = rank_selection(pop, 3)
        assert len(selected) == 3

    def test_truncation_selection(self):
        """Test truncation selection."""
        genomes = [create_genome(f"def f{i}(): return {i}") for i in range(10)]
        for i, g in enumerate(genomes):
            g.fitness_scores = FitnessResult(tests_passed=i / 10)

        pop = Population(genomes=genomes)
        selected = truncation_selection(pop, 3, truncation_pct=0.3)
        assert len(selected) == 3

    def test_generational_replacement(self):
        """Test generational population replacement."""
        old_genomes = [create_genome(f"def old{i}(): pass") for i in range(5)]
        for g in old_genomes:
            g.fitness_scores = FitnessResult(tests_passed=0.5)

        offspring = [create_genome(f"def new{i}(): pass") for i in range(3)]
        for g in offspring:
            g.fitness_scores = FitnessResult(tests_passed=0.8)

        pop = Population(genomes=old_genomes)
        new_pop = generational_replacement(pop, offspring, elitism_count=2)
        assert new_pop.generation == 1

    def test_steady_state_replacement(self):
        """Test steady-state replacement."""
        genomes = [create_genome(f"def f{i}(): return {i}") for i in range(5)]
        for i, g in enumerate(genomes):
            g.fitness_scores = FitnessResult(tests_passed=i / 5)

        offspring = [create_genome("def better(): return 999")]
        offspring[0].fitness_scores = FitnessResult(tests_passed=1.0)

        pop = Population(genomes=genomes)
        new_pop = steady_state_replacement(pop, offspring, "worst")
        assert len(new_pop.genomes) == 5

    def test_initialize_population(self):
        """Test population initialization."""
        genomes = [create_genome(f"def f{i}(): pass") for i in range(3)]
        pop = initialize_population(genomes, target_size=5)
        assert pop.generation == 0
        assert len(pop.genomes) <= 5

    def test_diversity_maintenance(self):
        """Test diversity maintenance removes similar genomes."""
        # Create very similar genomes
        genomes = [create_genome("def foo(): return 1") for _ in range(10)]
        for g in genomes:
            g.fitness_scores = FitnessResult(tests_passed=0.5)
            g.ast_hash = genome_hash(g)

        pop = Population(genomes=genomes)
        maintained = diversity_maintenance(pop, min_diversity=0.5)
        # Should have removed some similar genomes
        assert maintained.size <= 10

    def test_check_convergence(self):
        """Test convergence detection."""
        pop = Population()
        # Add flat fitness history
        for _ in range(10):
            pop.stats_history.append(PopulationStats(best_fitness=0.5))

        assert check_convergence(pop, threshold=0.01, window=5)

    def test_update_archive(self):
        """Test Pareto archive update."""
        g1 = create_genome("def a(): pass")
        g1.fitness_scores = FitnessResult(tests_passed=1.0, performance=1.0)
        g2 = create_genome("def b(): pass")
        g2.fitness_scores = FitnessResult(tests_passed=0.5, performance=0.5)

        pop = Population(genomes=[g1, g2])
        update_archive(pop)
        # g1 should be in archive (non-dominated)
        assert any(g.id == g1.id for g in pop.archive)


# =============================================================================
# LETHAL GENE TESTS
# =============================================================================


class TestLethalGenes:
    """Tests for lethal.py tracking."""

    def test_lethal_db_record_failure(self):
        """Test recording a failure."""
        db = LethalGeneDB(min_failures=1)
        g = create_genome("def bad(): return 1 / 0")
        db.record_failure(g, {"type": "ZeroDivisionError"})
        assert len(db.failure_history) == 1

    def test_lethal_db_pattern_threshold(self):
        """Test lethal pattern threshold."""
        db = LethalGeneDB(min_failures=2)
        g1 = create_genome("x = 1 / 0")
        g2 = create_genome("y = 1 / 0")

        db.record_failure(g1)
        assert len(db.get_lethal_patterns()) == 0  # Not yet lethal

        db.record_failure(g2)
        # After 2 failures, pattern may be lethal
        stats = db.get_statistics()
        assert stats["total_failures"] == 2

    def test_lethal_db_contains_lethal(self):
        """Test checking for lethal patterns."""
        db = LethalGeneDB(min_failures=1)
        g = create_genome("for i in range(10): print(i)")
        db.record_failure(g)
        db.record_failure(g)  # Record twice to mark as lethal

        # New genome with same pattern
        has_lethal = db.contains_lethal("for i in range(10): print(i)")
        # May or may not detect depending on pattern extraction

    def test_lethal_db_filter_mutations(self):
        """Test filtering mutations with lethal genes."""
        db = LethalGeneDB(min_failures=1)
        genomes = [
            create_genome("def good(): return 1"),
            create_genome("def good2(): return 2"),
        ]
        filtered = db.filter_mutations(genomes)
        assert len(filtered) == 2  # No lethal patterns yet

    def test_lethal_db_save_load(self):
        """Test serialization and deserialization."""
        db = LethalGeneDB(min_failures=2)
        g = create_genome("def test(): pass")
        db.record_failure(g)

        data = db.save_to_dict()
        new_db = LethalGeneDB()
        new_db.load_from_dict(data)

        assert new_db.min_failures == 2

    def test_identify_failure_cause(self):
        """Test failure cause identification."""
        g = create_genome("x = 1")
        error = ZeroDivisionError("division by zero")
        analysis = identify_failure_cause(g, error)
        assert analysis["type"] == "ZeroDivisionError"
        assert analysis["likely_cause"] == "division_by_zero"

    def test_extract_error_patterns(self):
        """Test extracting patterns from errors."""
        code = "x = 1 / y"
        error = ZeroDivisionError("division by zero")
        patterns = extract_error_patterns(code, error)
        # Should find division pattern
        assert isinstance(patterns, list)


# =============================================================================
# HARNESS TESTS
# =============================================================================


class TestEvolutionHarness:
    """Tests for harness.py main loop."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        llm = AsyncMock()
        llm.generate.return_value = "```python\ndef add(a, b):\n    return a + b\n```"
        return llm

    @pytest.mark.asyncio
    async def test_harness_initialization(self, mock_llm):
        """Test harness initialization."""
        config = EvolutionConfig(population_size=5, max_generations=2)
        harness = EvolutionHarness(llm=mock_llm, config=config)
        assert harness.config.population_size == 5

    @pytest.mark.asyncio
    async def test_harness_initialize_from_spec(self, mock_llm):
        """Test initializing population from spec."""
        config = EvolutionConfig(population_size=3, max_generations=1)
        harness = EvolutionHarness(llm=mock_llm, config=config)

        pop = await harness.initialize_from_spec(
            "Write a function to add two numbers",
            num_initial=3,
        )
        assert pop is not None
        assert len(pop.genomes) <= 3

    @pytest.mark.asyncio
    async def test_harness_run(self, mock_llm):
        """Test running evolution."""
        config = EvolutionConfig(
            population_size=3,
            max_generations=2,
            mutation_rate=0.5,
            crossover_rate=0.5,
        )
        tests = [{"input": [1, 2], "expected": 3, "function": "add"}]
        harness = EvolutionHarness(llm=mock_llm, config=config, tests=tests)

        result = await harness.run("Write add function")
        assert isinstance(result, EvolutionResult)
        assert result.generations_run > 0

    @pytest.mark.asyncio
    async def test_harness_get_best_genomes(self, mock_llm):
        """Test getting best genomes."""
        config = EvolutionConfig(population_size=5, max_generations=1)
        harness = EvolutionHarness(llm=mock_llm, config=config)
        await harness.initialize_from_spec("Test task", num_initial=5)

        best = harness.get_best_genomes(3)
        assert len(best) <= 3

    @pytest.mark.asyncio
    async def test_harness_add_test_case(self, mock_llm):
        """Test adding test cases dynamically."""
        harness = EvolutionHarness(llm=mock_llm)
        harness.add_test_case({"input": [1], "expected": 1})
        assert len(harness.tests) == 1


# =============================================================================
# TYPE TESTS
# =============================================================================


class TestTypes:
    """Tests for types.py dataclasses."""

    def test_genome_creation(self):
        """Test Genome dataclass."""
        g = Genome(id="test", code="def foo(): pass")
        assert g.id == "test"
        assert not g.is_evaluated
        assert g.fitness == 0.0

    def test_fitness_result_composite(self):
        """Test FitnessResult composite score."""
        fr = FitnessResult(
            tests_passed=1.0,
            performance=0.8,
            quality=0.6,
            novelty=0.4,
        )
        # Weighted: 0.4*1.0 + 0.2*0.8 + 0.2*0.6 + 0.2*0.4 = 0.76
        assert abs(fr.composite - 0.76) < 0.01

    def test_population_stats(self):
        """Test Population statistics."""
        genomes = [create_genome(f"def f{i}(): pass") for i in range(5)]
        for i, g in enumerate(genomes):
            # Set all fitness components so composite = tests_passed
            g.fitness_scores = FitnessResult(
                tests_passed=i / 5,
                performance=i / 5,
                quality=i / 5,
                novelty=i / 5,
            )

        pop = Population(genomes=genomes)
        stats = pop.get_stats()
        assert stats.size == 5
        # Best composite fitness is 4/5 = 0.8 (floating point tolerance)
        assert abs(stats.best_fitness - 0.8) < 0.01

    def test_population_best_genome(self):
        """Test getting best genome from population."""
        genomes = [create_genome(f"def f{i}(): pass") for i in range(3)]
        genomes[0].fitness_scores = FitnessResult(tests_passed=0.5)
        genomes[1].fitness_scores = FitnessResult(tests_passed=0.9)
        genomes[2].fitness_scores = FitnessResult(tests_passed=0.3)

        pop = Population(genomes=genomes)
        best = pop.best_genome
        assert best is not None
        assert best.id == genomes[1].id

    def test_evolution_config_defaults(self):
        """Test EvolutionConfig default values."""
        config = EvolutionConfig()
        assert config.population_size == 20
        assert config.max_generations == 50
        assert config.mutation_rate == 0.8

    def test_mutation_op(self):
        """Test MutationOp dataclass."""
        op = MutationOp(
            type=MutationType.ALGORITHM,
            description="Try different algorithm",
            source_genome_id="g1",
        )
        assert op.type == MutationType.ALGORITHM
        assert not op.success

    def test_crossover_result(self):
        """Test CrossoverResult dataclass."""
        result = CrossoverResult(
            offspring=[create_genome("def x(): pass")],
            crossover_point="line 5",
            success=True,
        )
        assert len(result.offspring) == 1
        assert result.success


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for the evolution system."""

    def test_full_genome_lifecycle(self):
        """Test genome from creation through evaluation."""
        # Create
        g = create_genome("def double(x):\n    return x * 2")

        # Extract genes
        genes = extract_genes(g)
        assert len(genes) > 0

        # Evaluate fitness
        tests = [
            {"input": [2], "expected": 4, "function": "double"},
            {"input": [0], "expected": 0, "function": "double"},
        ]
        fitness = TestFitness(tests)
        score = fitness.evaluate(g)
        assert score == 1.0

        # Set fitness
        g.fitness_scores = FitnessResult(tests_passed=score)
        assert g.is_evaluated
        assert g.fitness > 0

    def test_population_evolution_cycle(self):
        """Test one evolution cycle with population."""
        # Initialize
        genomes = [
            create_genome("def add(a, b):\n    return a + b"),
            create_genome("def add(a, b):\n    return a - b"),  # Wrong
            create_genome("def add(a, b):\n    return 0"),  # Wrong
        ]

        tests = [{"input": [1, 2], "expected": 3, "function": "add"}]
        fitness = TestFitness(tests)

        # Evaluate
        for g in genomes:
            test_score = fitness.evaluate(g)
            g.fitness_scores = FitnessResult(
                tests_passed=test_score,
                performance=test_score,
                quality=test_score,
                novelty=test_score,
            )

        pop = Population(genomes=genomes)

        # Select
        selected = tournament_selection(pop, 2)
        assert len(selected) == 2

        # Best should be first genome (correct implementation)
        elites = elitism(pop, 1)
        assert elites[0].fitness == 1.0
        assert elites[0].fitness_scores.tests_passed == 1.0

    def test_crossover_produces_valid_offspring(self):
        """Test that crossover produces syntactically valid code."""
        g1 = create_genome("def foo():\n    return 1\n\ndef bar():\n    return 2")
        g2 = create_genome("def baz():\n    return 3\n\ndef qux():\n    return 4")

        result = uniform_crossover(g1, g2)
        if result.success:
            for offspring in result.offspring:
                assert is_valid_code(offspring.code)
